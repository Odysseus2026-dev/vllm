# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the S6 weight-sync check extension and its fingerprint format."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.distributed.weight_transfer import s6_weight_check as wc
from vllm.distributed.weight_transfer.s6_fingerprint import (
    WFP_FORMAT,
    TensorUnavailable,
    chunk_fingerprints,
    compare_wfp,
    make_wfp,
    tensor_sha256,
    tensor_wfp,
)
from vllm.distributed.weight_transfer.s6_weight_check import S6WeightCheckExtension

H, I, E, V, V_PAD = 8, 4, 2, 10, 16  # noqa: E741
BF16 = torch.bfloat16


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False  # keep the module-scoped gloo groups alive


@pytest.fixture(scope="module")
def tp1():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, path = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{path}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()
    if os.path.exists(path):  # FileStore usually removes it
        os.unlink(path)


# ---------------------------------------------------------------- fingerprints


def _naive(tensor: torch.Tensor, chunk: int) -> list[list[float]]:
    flat = tensor.contiguous().reshape(-1).to(torch.float64)
    rows = []
    for start in range(0, flat.numel(), chunk):
        v = flat[start : start + chunk]
        index = torch.arange(start, start + v.numel(), dtype=torch.float64)
        rows.append(
            [
                v.sum().item(),
                v.abs().sum().item(),
                (v * v).sum().item(),
                (index * v).sum().item(),
            ]
        )
    return rows


@pytest.mark.parametrize("chunk", [1, 7, 64, 4096 + 5, 10**9])
def test_fingerprint_matches_definition_and_ignores_memory_layout(chunk):
    """Chunks are global logical ranges, so any layout of the same values gives
    bitwise-identical fingerprints (the reduced float64 vectors are identical)."""
    torch.manual_seed(0)
    base = torch.randn(70, 130).to(BF16)
    layouts = {
        "contiguous": base,
        "transposed": base.t().contiguous().t(),
        "strided": torch.randn(70, 260).to(BF16)[:, ::2].copy_(base),
        "permuted3d": base.reshape(70, 10, 13)
        .permute(2, 0, 1)
        .contiguous()
        .permute(1, 2, 0),
    }
    expected = chunk_fingerprints(base, chunk)
    for name, tensor in layouts.items():
        assert torch.equal(tensor, base.reshape(tensor.shape)), name
        assert chunk_fingerprints(tensor, chunk) == expected, name
    for got, want in zip(expected, _naive(base, chunk)):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-9)


def test_fingerprint_determinism_and_thread_count_noise():
    """Bitwise reproducible for a fixed device and thread count; CPU float64
    reductions regroup with the thread count, which stays far below the default
    ``rtol`` (1e-12) while one BF16 ulp in a full default chunk is ~1e-10."""
    torch.manual_seed(1)
    tensor = torch.randn(3_000_000).to(BF16)
    threads = torch.get_num_threads()
    docs = {}
    try:
        for n in sorted({1, 2, max(4, threads)}):
            torch.set_num_threads(n)
            first = make_wfp({"w": tensor}, 1 << 20)
            assert make_wfp({"w": tensor}, 1 << 20) == first
            docs[n] = first
    finally:
        torch.set_num_threads(threads)
    single = docs.pop(1)
    for doc in docs.values():
        result = compare_wfp(doc, single, rtol=1e-12)
        assert result["ok"], result
        loose = compare_wfp(doc, single, rtol=0.0)
        worst = max((m["max_rel_err"] for m in loose["mismatched"]), default=0.0)
        assert worst < 1e-14


def test_isum_detects_swapped_experts_within_one_chunk():
    torch.manual_seed(2)
    experts = torch.randn(4, 16, 8).to(BF16)
    swapped = experts[[1, 0, 2, 3]]
    a = chunk_fingerprints(experts, 10**6)[0]
    b = chunk_fingerprints(swapped, 10**6)[0]
    assert a[:3] == pytest.approx(b[:3], rel=1e-15)  # permutation-invariant moments
    assert a[3] != b[3]


def test_empty_meta_released_and_non_finite():
    empty = tensor_wfp(torch.empty(0, 5), 16, sha256=True)
    assert empty["chunks"] == [] and empty["numel"] == 0
    assert empty["sha256"] == hashlib.sha256(b"").hexdigest()
    with pytest.raises(TensorUnavailable, match="meta"):
        tensor_wfp(torch.empty(3, 3, device="meta"))
    released = torch.ones(4)
    released.untyped_storage().resize_(0)
    with pytest.raises(TensorUnavailable, match="released_storage"):
        tensor_wfp(released)
    bad = torch.tensor([1.0, float("nan"), float("inf")])
    doc = make_wfp({"bad": bad}, 16)
    json.dumps(doc, allow_nan=False)
    assert doc["tensors"]["bad"]["chunks"][0][0] == "nan"
    result = compare_wfp(doc, doc)
    assert not result["ok"] and "non_finite" in result["mismatched"][0]["reason"]


def test_sha256_is_logical_byte_order():
    torch.manual_seed(3)
    tensor = torch.randn(33, 17).to(BF16).t()
    raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    assert tensor_sha256(tensor, 50) == hashlib.sha256(raw).hexdigest()


def test_compare_detects_one_bf16_ulp():
    torch.manual_seed(4)
    tensor = torch.randn(1 << 16).to(BF16)
    ref = make_wfp({"w": tensor}, 1 << 16)
    nudged = tensor.clone()
    nudged.view(torch.int16)[12345] += 1  # one BF16 ulp
    assert not torch.equal(nudged, tensor)
    ok = compare_wfp(make_wfp({"w": tensor.clone()}, 1 << 16), ref)
    assert ok["ok"] and ok["checked"] == 1
    for rtol in (0.0, 1e-12):
        bad = compare_wfp(make_wfp({"w": nudged}, 1 << 16), ref, rtol=rtol)
        assert not bad["ok"] and bad["mismatched"][0]["bad_chunks"] == [0]


def test_default_rtol_detects_one_bf16_ulp_in_a_full_default_chunk():
    from vllm.distributed.weight_transfer.s6_fingerprint import DEFAULT_CHUNK_ELEMS

    torch.manual_seed(5)
    tensor = torch.randn(DEFAULT_CHUNK_ELEMS, dtype=torch.float32).to(BF16)
    ref = make_wfp({"w": tensor}, DEFAULT_CHUNK_ELEMS)
    index = 1_234_567
    tensor.view(torch.int16)[index] += 1
    result = compare_wfp(make_wfp({"w": tensor}, DEFAULT_CHUNK_ELEMS), ref)
    assert not result["ok"] and result["mismatched"][0]["bad_chunks"] == [0]


_PEAK_SCRIPT = textwrap.dedent(
    """
    import json, torch
    from vllm.distributed.weight_transfer.s6_fingerprint import (
        DEFAULT_CHUNK_ELEMS, chunk_fingerprints)

    def status(key):
        with open("/proc/self/status") as f:
            line = next(line for line in f if line.startswith(key + ":"))
        return int(line.split()[1]) * 1024

    def rss():
        return status("VmRSS")

    def peak():
        return status("VmHWM")

    base = torch.empty(4100, 8200, dtype=torch.float32)
    base.normal_()
    tensor = base.t()  # logical [8200, 4100]: rows straddle the chunk boundary
    assert not tensor.is_contiguous() and tensor.numel() > DEFAULT_CHUNK_ELEMS
    chunk_fingerprints(base.t()[:2, :2])  # warm up
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")  # reset the peak-RSS high-water mark
    before, peak_before = rss(), peak()
    chunk_fingerprints(tensor)
    print(json.dumps({"chunk_elems": DEFAULT_CHUNK_ELEMS, "rss_before": before,
                      "peak_before": peak_before, "peak_after": peak()}))
    """
)


def test_non_contiguous_peak_memory_is_one_float64_chunk():
    """A non-contiguous FP32 input larger than one default chunk raises the process
    peak by about the float64 chunk (8*C = 256 MiB), within the 512 MiB budget:
    strided views are copied straight into the float64 buffer."""
    out = subprocess.run(
        [sys.executable, "-c", _PEAK_SCRIPT],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "OMP_NUM_THREADS": "4"},
    ).stdout
    stats = json.loads(out.strip().splitlines()[-1])
    mib = 1024 * 1024
    assert stats["peak_before"] - stats["rss_before"] < 64 * mib  # valid baseline
    growth = stats["peak_after"] - stats["rss_before"]
    print(f"peak growth {growth / mib:.1f} MiB for chunk_elems={stats['chunk_elems']}")
    assert growth <= 512 * mib
    assert growth <= 8 * stats["chunk_elems"] + 64 * mib


def test_compare_reports_missing_extra_and_chunk_size_mismatch():
    a = make_wfp({"x": torch.ones(4), "y": torch.ones(2)}, 4)
    b = make_wfp({"x": torch.ones(4), "z": torch.ones(2)}, 4)
    result = compare_wfp(a, b)
    assert result["missing"] == ["z"] and result["extra"] == ["y"]
    assert not result["ok"]
    assert compare_wfp(a, b, strict_extra=False)["ok"] is False  # z still missing
    assert compare_wfp(make_wfp({"x": torch.ones(4)}, 2), b)["error"]


# ---------------------------------------------------------------- fake YOCO model


def _yoco_model(meta_param: bool = False):
    from vllm.model_executor.layers.linear import MergedColumnParallelLinear
    from vllm.model_executor.models.yoco import YOCOForCausalLM

    class VocabParallelEmbedding(nn.Module):  # name matched by the vocab-pad rule
        def __init__(self):
            super().__init__()
            self.tp_size = 1
            self.weight = nn.Parameter(
                torch.zeros(V_PAD, H, dtype=BF16), requires_grad=False
            )
            self.weight.weight_loader = self._load

        @staticmethod
        def _load(param, weight):
            param.data[: weight.shape[0]].copy_(weight)

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = nn.Parameter(
                torch.zeros(E, 2 * I, H, dtype=BF16), requires_grad=False
            )
            self.w2_weight = nn.Parameter(
                torch.zeros(E, H, I, dtype=BF16), requires_grad=False
            )
            self.w13_weight.weight_loader = self._load
            self.w2_weight.weight_loader = self._load

        @staticmethod
        def _load(param, weight, name, shard_id, expert_id):
            target = param.data[expert_id]
            if shard_id == "w1":
                target = target[:I]
            elif shard_id == "w3":
                target = target[I:]
            target.copy_(weight)

    def merged(sizes):
        return MergedColumnParallelLinear(
            H, sizes, bias=False, disable_tp=True, params_dtype=BF16
        )

    def plain(*shape, dtype=BF16):
        module = nn.Module()
        module.weight = nn.Parameter(
            torch.zeros(*shape, dtype=dtype), requires_grad=False
        )
        return module

    class FakeYoco(YOCOForCausalLM):
        def __init__(self):
            nn.Module.__init__(self)
            self.config = SimpleNamespace(
                num_hidden_layers=2,
                yoco_cross_layers=1,
                moe_intermediate_size=I,
                num_experts=E,
            )
            self._rotary_caches_initialized = True
            self._router_weight_caches_initialized = True
            self._shared_expert_weight_caches_initialized = True
            self.model = nn.Module()
            self.model.embed_tokens = VocabParallelEmbedding()
            self.model.yoco_kv_proj = merged([4, 4])
            layer0, layer1 = nn.Module(), nn.Module()
            layer0.self_attn = nn.Module()
            layer0.self_attn.qkv_lambda_proj = merged([8, 4, 4, 2])
            layer0.self_attn.o_proj = nn.Module()
            # Non-contiguous parameter (stored transposed).
            layer0.self_attn.o_proj.weight = nn.Parameter(
                torch.zeros(8, H, dtype=BF16).t(), requires_grad=False
            )
            layer0.mlp = nn.Module()
            layer0.mlp.gate = plain(E, H, dtype=torch.float32)
            layer0.mlp.experts = nn.Module()
            layer0.mlp.experts.routed_experts = Experts()
            layer0.mlp.transform = layer0.mlp.experts.routed_experts  # same object
            layer0.mlp.shared_experts = nn.Module()
            layer0.mlp.shared_experts.gate_up_proj = merged([I, I])
            layer0.mlp.empty_a = plain(0, H)
            layer0.mlp.empty_b = plain(0, H)
            layer1.self_attn = nn.Module()
            layer1.self_attn.q_lambda_proj = merged([8, 2])
            layer1.input_layernorm = plain(H)
            layer1.norm_view = nn.Module()
            # Distinct Parameter over the same memory: a storage-only alias.
            layer1.norm_view.weight = nn.Parameter(
                layer1.input_layernorm.weight.data, requires_grad=False
            )
            if meta_param:
                layer1.unused = plain(3, 3)
                layer1.unused.weight = nn.Parameter(
                    torch.empty(3, 3, device="meta"), requires_grad=False
                )
            self.model.layers = nn.ModuleList([layer0, layer1])
            self.lm_head = VocabParallelEmbedding()

    return FakeYoco()


def _hf_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)

    def r(*shape, dtype=BF16):
        return torch.randn(*shape, generator=gen).to(dtype)

    p0, p1 = "model.layers.0.", "model.layers.1."
    return {
        "model.embed_tokens.weight": r(V, H),
        p0 + "self_attn.q_proj.weight": r(8, H),
        p0 + "self_attn.k_proj.weight": r(4, H),
        p0 + "self_attn.v_proj.weight": r(4, H),
        p0 + "self_attn.lambda_proj.weight": r(2, H),
        p0 + "self_attn.o_proj.weight": r(H, 8),
        p0 + "mlp.gate.weight": r(E, H, dtype=torch.float32),
        p0 + "mlp.experts.w13_weight": r(E * 2 * I, H),
        p0 + "mlp.experts.w2_weight": r(E, H, I),
        p0 + "mlp.shared_experts.gate_up_proj.weight": r(2 * I, H),
        p0 + "mlp.empty_a.weight": r(0, H),
        p0 + "mlp.empty_b.weight": r(0, H),
        p1 + "self_attn.q_proj.weight": r(8, H),
        p1 + "self_attn.lambda_proj.weight": r(2, H),
        p1 + "input_layernorm.weight": r(H),
        "model.k_proj.weight": r(4, H),
        "model.v_proj.weight": r(4, H),
        "lm_head.weight": r(V, H),
    }


class _CudaGraphLikeWrapper:
    def __init__(self, runnable):
        self.runnable = runnable

    def __getattr__(self, key):
        return getattr(self.runnable, key)


class _Worker(S6WeightCheckExtension):
    def __init__(self, model):
        self.rank = self.local_rank = 0
        wrapper = _CudaGraphLikeWrapper(model)
        self.model_runner = SimpleNamespace(model=wrapper, get_model=lambda: model)


def _refit(worker, weights, bucket_size=5):
    """verl-style bucketed refit: tensors live in a reused buffer per bucket."""
    items = list(weights.items())
    for start in range(0, len(items), bucket_size):
        bucket = [(n, t.clone()) for n, t in items[start : start + bucket_size]]
        worker.model_runner.model.load_weights(bucket)
        for _, tensor in bucket:  # the transport reuses the buffer afterwards
            tensor.fill_(7)


def _roundtrip(obj):
    """Strict JSON and vLLM msgpack round trips (what collective_rpc carries)."""
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    assert json.loads(json.dumps(obj, allow_nan=False)) == obj
    decoded = MsgpackDecoder().decode(MsgpackEncoder().encode(obj))
    assert json.loads(json.dumps(decoded)) == obj
    return obj


CHUNK = 16
S2_M2_KEYS = {  # runs/S2/s2_vllm_probe.py vllm_load_report (review-S2 M2)
    "load_weights_calls", "received_unique", "received_total", "received_missing",
    "received_unexpected", "vllm_param_names_total", "vllm_params_canonical",
    "vllm_params_written", "vllm_param_aliases", "vllm_param_aliases_same_object",
    "vllm_param_aliases_same_storage_only", "vllm_param_aliases_all_point_to_written",
    "vllm_param_alias_examples", "vllm_params_not_loaded",
    "loaded_names_not_registered", "ignored", "pipeline_missing",
    "default_loader_fallbacks", "ok",
}  # fmt: skip


def test_refit_matches_trainer_ref_in_all_sources(tp1):
    worker = _Worker(_yoco_model())
    hf = _hf_weights()
    ref = make_wfp(hf, CHUNK, sha256=True)
    model = worker.model_runner.get_model()
    assert "load_weights" not in model.__dict__
    armed = _roundtrip(worker.s6_arm(chunk_elems=CHUNK, record="both", sha256=True))
    assert armed["armed"] and armed["model_class"] == "FakeYoco"
    _refit(worker, hf)

    for source in ("received", "written", "params"):
        result = _roundtrip(worker.s6_compare(ref, source=source))
        assert result["ok"], (source, result)
        assert result["checked"] == len(hf) and result["sha256_checked"] == len(hf)
    written = worker.s6_checksum(source="written")
    assert written["tensors"]["model.layers.0.mlp.gate.weight"]["dtype"] == "float32"
    assert written["tensors"]["model.layers.0.mlp.empty_a.weight"]["chunks"] == []

    report = _roundtrip(worker.s6_load_report(expected_names=sorted(hf)))
    assert report["ok"], report
    assert set(report) >= S2_M2_KEYS
    assert report["received_names"] == sorted(hf)
    assert report["load_weights_calls"] == 4
    assert report["vllm_params_not_loaded"] == []
    # transform.{w13,w2} are the same Parameter objects; norm_view shares storage.
    assert report["vllm_param_aliases_same_object"] == 2
    assert report["vllm_param_aliases_same_storage_only"] == 1
    assert report["vllm_param_aliases_all_point_to_written"]
    names = {pair[0] for pair in report["vllm_param_alias_examples"]}
    assert "model.layers.0.mlp.empty_b.weight" not in names  # empty is not an alias

    assert worker.s6_disarm()["restored"] is True
    assert "load_weights" not in model.__dict__
    assert model.load_weights.__func__ is type(model).load_weights


def test_fault_injection_is_reported(tp1):
    hf = _hf_weights()
    ref = make_wfp(hf, 10**6)

    def run(sent, **arm):
        worker = _Worker(_yoco_model())
        worker.s6_arm(chunk_elems=10**6, record="both", **arm)
        _refit(worker, sent)
        return worker

    swapped = dict(hf)
    w13 = hf["model.layers.0.mlp.experts.w13_weight"].view(E, 2 * I, H)
    swapped["model.layers.0.mlp.experts.w13_weight"] = w13[[1, 0]].reshape(-1, H)
    worker = run(swapped)
    for source in ("received", "written"):
        result = worker.s6_compare(ref, source=source)
        assert not result["ok"]
        assert [m["name"] for m in result["mismatched"]] == [
            "model.layers.0.mlp.experts.w13_weight"
        ]

    zeroed = dict(hf, **{"model.k_proj.weight": torch.zeros(4, H, dtype=BF16)})
    result = run(zeroed).s6_compare(ref)
    assert [m["name"] for m in result["mismatched"]] == ["model.k_proj.weight"]

    renamed = dict(hf)
    renamed["lm_head.weights"] = renamed.pop("lm_head.weight")
    worker = run(renamed)
    result = worker.s6_compare(ref)
    assert result["missing"] == ["lm_head.weight"]
    assert result["extra"] == ["lm_head.weights"] and not result["ok"]
    report = worker.s6_load_report(expected_names=sorted(hf))
    assert not report["ok"]
    assert report["ignored"] == ["lm_head.weights"]
    assert report["vllm_params_not_loaded"] == ["lm_head.weight"]
    assert report["received_missing"] == ["lm_head.weight"]

    # Corruption after loading: only the live read-back sees it.
    worker = run(hf)
    model = worker.model_runner.get_model()
    model.model.layers[0].self_attn.qkv_lambda_proj.weight.data[8:12].zero_()
    assert worker.s6_compare(ref, source="written")["ok"]
    result = worker.s6_compare(ref, source="params")
    assert [m["name"] for m in result["mismatched"]] == [
        "model.layers.0.self_attn.k_proj.weight"
    ]


def test_unarmed_reset_and_chunk_contract(tp1):
    worker = _Worker(_yoco_model())
    hf = _hf_weights()
    assert worker.s6_checksum()["error"] == "not_armed"
    assert worker.s6_compare(make_wfp(hf, CHUNK))["ok"] is False
    assert worker.s6_load_report()["ok"] is False
    assert worker.s6_disarm()["was_armed"] is False

    worker.s6_arm(chunk_elems=CHUNK, record="received")
    _refit(worker, hf)
    assert worker.s6_checksum(chunk_elems=CHUNK * 2)["error"]
    assert worker.s6_checksum(source="written")["error"]
    assert worker.s6_compare(make_wfp(hf, CHUNK * 2))["error"]
    names = ["lm_head.weight", "nope"]
    subset = worker.s6_checksum(names=names)
    assert list(subset["tensors"]) == ["lm_head.weight"] and subset["missing"] == [
        "nope"
    ]
    first = worker.s6_load_report(reset=True)
    assert first["ok"] and first["load_weights_calls"] == 4
    second = worker.s6_load_report()
    assert second["load_weights_calls"] == 0 and not second["ok"]
    assert worker.s6_checksum()["tensors"] == {}
    worker.s6_disarm()


def test_snapshot_and_meta_parameter(tp1):
    worker = _Worker(_yoco_model(meta_param=True))
    snap = _roundtrip(worker.s6_snapshot())
    params = snap["params"]
    assert params["model.layers.1.unused.weight"]["is_meta"]
    assert not params["model.layers.1.unused.weight"]["has_storage"]
    assert params["model.layers.0.mlp.experts.routed_experts.w13_weight"][
        "aliases"
    ] == ["model.layers.0.mlp.transform.w13_weight"]
    assert "model.layers.0.mlp.empty_a.weight" in params
    assert "model.layers.0.mlp.empty_b.weight" in params
    assert not params["model.layers.0.self_attn.o_proj.weight"]["contiguous"]
    partial = worker.s6_snapshot(names=["model.layers.0.mlp.transform.w2_weight", "x"])
    assert list(partial["params"]) == [
        "model.layers.0.mlp.experts.routed_experts.w2_weight"
    ]
    assert partial["unknown_names"] == ["x"]

    result = worker.s6_checksum(names=["model.layers.1.unused.weight"], source="params")
    assert result["unsupported"] == {
        "model.layers.1.unused.weight": "meta:model.layers.1.unused.weight"
    }

    worker.s6_arm(chunk_elems=CHUNK)
    _refit(worker, _hf_weights())
    report = worker.s6_load_report()
    assert report["vllm_params_not_loaded"] == ["model.layers.1.unused.weight"]
    assert not report["ok"]
    worker.s6_disarm()


def test_no_insecure_serialization_in_extension():
    for module in (
        wc,
        __import__("vllm.distributed.weight_transfer.s6_fingerprint", fromlist=["x"]),
    ):
        source = Path(module.__file__).read_text()
        for token in ("VLLM_ALLOW_INSECURE_SERIALIZATION", "pickle", "cloudpickle"):
            assert token not in source, (module.__name__, token)
    assert WFP_FORMAT == "s6.wfp.v1"
