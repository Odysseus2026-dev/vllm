# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.config.compilation import CompilationMode
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    RoutedExpertsManager,
    bind_routed_experts_capturer,
    get_routed_experts_attn_gid,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test

_REC_MODULE = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"


def _capturer_with_buffer(
    *,
    max_tokens: int = 8,
    num_layers: int = 4,
    num_experts_per_tok: int = 2,
    dp_rank: int = 0,
    tp_size: int = 1,
) -> RoutedExpertsCapturer:
    # Bypass __init__ so the test can use a CPU buffer and skip the
    # VllmConfig dependency. The CUDA device-tensor allocation in the
    # real constructor is not what we are exercising here.
    c = RoutedExpertsCapturer.__new__(RoutedExpertsCapturer)
    c.dp_rank = dp_rank
    c.tp_size = tp_size
    c.device_buffer = torch.full(
        (max_tokens, num_layers, num_experts_per_tok),
        -1,
        dtype=torch.int32,
    )
    return c


class DummyRouter(BaseRouter):
    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.FUSED_TOPK

    def _compute_routing(
        self, hidden_states, router_logits, indices_type, *, input_ids=None
    ):
        topk_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
        return topk_weights, topk_ids

    def _apply_eplb_mapping(self, topk_ids: torch.Tensor) -> torch.Tensor:
        # Make mapping observable without requiring CUDA EPLB path.
        return topk_ids + 10


def _make_router(eplb_state: EplbLayerState | None = None) -> DummyRouter:
    return DummyRouter(
        top_k=2,
        global_num_experts=16,
        eplb_state=eplb_state,
    )


def _make_modular_routed_experts():
    return types.SimpleNamespace(
        quant_method=types.SimpleNamespace(is_monolithic=False),
    )


def _make_model_config(hf_config):
    num_experts_per_token = ModelArchConfigConvertorBase(
        hf_config, hf_config
    ).get_num_experts_per_token()
    model_config = SimpleNamespace(
        hf_text_config=hf_config,
        model_arch_config=SimpleNamespace(
            num_experts_per_token=num_experts_per_token,
        ),
    )
    model_config.get_num_experts = lambda: hf_config.num_experts
    model_config.get_num_experts_per_tok = lambda: ModelConfig.get_num_experts_per_tok(
        model_config
    )
    model_config.get_total_num_hidden_layers = lambda: hf_config.num_hidden_layers
    return model_config


def test_routed_experts_manager_uses_gemma4_top_k_experts():
    hf_config = SimpleNamespace(
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_routed_experts_manager_uses_kimi_k3_experts_per_token():
    hf_config = SimpleNamespace(
        num_experts=8,
        num_experts_per_token=2,
        num_hidden_layers=3,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    kv_cache_spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], kv_cache_spec)],
    )

    manager = RoutedExpertsManager(vllm_config, kv_cache_config)

    assert manager.routed_experts_by_slot.shape == (8, 3, 2)


def test_base_router_capture_pre_eplb_mapping():
    router = _make_router()
    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    topk_weights, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert topk_weights.shape == topk_ids.shape
    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_base_router_capture_with_eplb_enabled():
    eplb_state = EplbLayerState()
    eplb_state.expert_load_view = torch.zeros(32, dtype=torch.int64)
    eplb_state.logical_to_physical_map = torch.arange(32).view(32, 1)
    eplb_state.logical_replica_count = torch.ones(32, dtype=torch.int64)
    eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    router = _make_router(eplb_state=eplb_state)

    captured = []

    def capture_fn(ids):
        captured.append(ids.clone())

    router.set_capture_fn(capture_fn)
    _, topk_ids = router.select_experts(
        hidden_states=torch.empty(1),
        router_logits=torch.empty(1),
    )

    assert len(captured) == 1
    # Capture should see logical ids pre-EPLB mapping.
    assert torch.equal(captured[0], torch.tensor([[1, 2], [3, 4]]))
    # Our DummyRouter mapping adds +10.
    assert torch.equal(topk_ids, torch.tensor([[11, 12], [13, 14]]))


def test_public_binding_only_visits_target_model(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method

    target_module = DummyFusedMoE(layer_id=7)
    draft_module = DummyFusedMoE(layer_id=0)

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    calls = []
    capturer = types.SimpleNamespace(capture=lambda *args: calls.append(args))

    bind_routed_experts_capturer(
        types.SimpleNamespace(modules=lambda: [target_module]), capturer
    )

    assert target_module.router.capture_fn is not None
    assert draft_module.router.capture_fn is None
    topk_ids = torch.tensor([[5, 6]])
    target_module.router.capture_fn(topk_ids)
    assert calls == [(7, topk_ids)]


def test_public_binding_rejects_monolithic_without_replay_support(monkeypatch):
    class DummyFusedMoE:
        def __init__(self):
            self.layer_id = 3
            self.router = _make_router()
            # Use a concrete monolithic expert and override its capability
            # instead of instantiating the abstract base class directly.
            from vllm.model_executor.layers.fused_moe.experts.cpu_moe import (
                CPUExpertsFp8,
            )

            fused_experts = CPUExpertsFp8.__new__(CPUExpertsFp8)
            self.routed_experts = types.SimpleNamespace(
                quant_method=types.SimpleNamespace(
                    is_monolithic=True,
                    moe_kernel=types.SimpleNamespace(
                        impl=types.SimpleNamespace(fused_experts=fused_experts)
                    ),
                )
            )
            self._quant_method = self.routed_experts.quant_method
            self._quant_method.moe_kernel.impl.fused_experts = fused_experts
            fused_experts.supports_routing_replay_capture = lambda: False

    class DummyCapturer:
        def capture(self, layer_id, topk_ids):
            pass

    dummy_module = DummyFusedMoE()
    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)

    with pytest.raises(ValueError, match="monolithic MoE kernel"):
        bind_routed_experts_capturer(
            types.SimpleNamespace(modules=lambda: [dummy_module]), DummyCapturer()
        )


def test_routed_experts_capturer_single_dp_no_metadata():
    """dp_metadata is None: capture writes the full topk_ids rows."""
    capturer = _capturer_with_buffer(dp_rank=0)
    topk = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)
    assert capturer.device_buffer[3, 0, 0].item() == -1


def test_routed_experts_capturer_dp_naive_concatenated_all_ranks():
    """n == sum(num_tokens_dp): slice this rank's segment from concatenated topk."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # Concatenated order: rank0 rows then rank1 rows.
    topk = torch.tensor(
        [[0, 1], [2, 3], [10, 11], [12, 13], [14, 15]], dtype=torch.int32
    )
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    want = topk[2:5]
    assert torch.equal(capturer.device_buffer[:3, 0, :], want)


def test_routed_experts_capturer_dp_modular_local_tokens():
    """n == token_num_per_dp: topk is already local to this DP rank."""
    capturer = _capturer_with_buffer(dp_rank=1)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    topk = torch.tensor([[10, 11], [12, 13], [14, 15]], dtype=torch.int32)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert torch.equal(capturer.device_buffer[:3, 0, :], topk)


def test_routed_experts_capturer_dp_unexpected_batch_raises():
    """Mismatch between topk batch dim and DP layout: fail fast."""
    capturer = _capturer_with_buffer(dp_rank=0)
    num_tokens_dp = torch.tensor([2, 3], dtype=torch.int32)
    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=num_tokens_dp)
    )
    # total=5, local=2: n=1 matches neither naive (5) nor modular (2).
    topk = torch.tensor([[1, 2]], dtype=torch.int32)
    with (
        patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx),
        pytest.raises(AssertionError, match="unexpected topk_ids batch dim"),
    ):
        capturer.capture(layer_id=0, topk_ids=topk)
    assert capturer.device_buffer[0, 0, 0].item() == -1


def test_routed_experts_attention_group_is_shared_and_fail_closed():
    """Both sides key routing data by this gid, so it must skip non-full-attention
    groups rather than defaulting to 0, and fail closed when none exists."""
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["swa_layer"],
                SlidingWindowMLASpec(block_size=4, sliding_window=8, **common),
            ),
            KVCacheGroupSpec(["full_layer"], FullAttentionSpec(block_size=4, **common)),
        ]
    )
    assert get_routed_experts_attn_gid(config) == 1

    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(SimpleNamespace(kv_cache_groups=[]))


def test_routed_experts_attention_group_unwraps_uniform_type_specs():
    """DeepSeek-V4-shaped groups wrap their specs in ``UniformTypeKVCacheSpecs``.

    The wrapper is not a ``FullAttentionSpec``, so a bare isinstance check finds
    no group and fails closed on every worker. Unwrap semantics themselves are
    covered by ``test_is_full_attention_spec_*`` in tests/v1/core.
    """
    common = dict(num_kv_heads=1, head_size=1, dtype=torch.float32)
    swa_spec = SlidingWindowMLASpec(block_size=4, sliding_window=8, **common)
    mla_spec = MLAAttentionSpec(block_size=4, **common)
    config = SimpleNamespace(
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["swa_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"swa_layer": swa_spec}
                ),
            ),
            KVCacheGroupSpec(
                ["mla_layer"],
                UniformTypeKVCacheSpecs(
                    block_size=4, kv_cache_specs={"mla_layer": mla_spec}
                ),
            ),
        ]
    )

    assert get_routed_experts_attn_gid(config) == 1

    swa_only = SimpleNamespace(kv_cache_groups=[config.kv_cache_groups[0]])
    with pytest.raises(ValueError, match="requires a full-attention KV cache group"):
        get_routed_experts_attn_gid(swa_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mrv2_async_output_returns_existing_routed_experts_field():
    from vllm.v1.outputs import ModelRunnerOutput, RoutedExpertsTensors
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    routed_experts = RoutedExpertsTensors(
        routing_data=torch.arange(6, dtype=torch.int32, device="cuda").reshape(3, 1, 2),
        slot_mapping=torch.tensor([11, 12, 13], device="cuda"),
    )
    num_sampled = torch.tensor([1], dtype=torch.int32, device="cuda")
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[1]], device="cuda"),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    output = AsyncOutput(
        model_runner_output=ModelRunnerOutput(req_ids=["req"], req_id_to_index={}),
        sampler_output=sampler_output,
        num_sampled_tokens=num_sampled,
        main_stream=torch.cuda.current_stream(),
        copy_stream=torch.cuda.Stream(),
        routed_experts=routed_experts,
    ).get_output()

    assert output.routed_experts is not None
    assert output.routed_experts.routing_data[:, 0, 0].tolist() == [0, 2, 4]
    assert output.routed_experts.slot_mapping.tolist() == [11, 12, 13]


@pytest.mark.parametrize("rank", [0, 1])
def test_all_tp_ranks_initialize_capture(monkeypatch, rank):
    pytest.importorskip("vllm.vllm_flash_attn", exc_type=ImportError)
    import vllm.v1.worker.gpu.model_runner as model_runner

    capturer = Mock()
    constructor = Mock(return_value=capturer)
    bind = Mock()
    monkeypatch.setattr(model_runner, "RoutedExpertsCapturer", constructor)
    monkeypatch.setattr(model_runner, "bind_routed_experts_capturer", bind)

    runner = model_runner.GPUModelRunner.__new__(model_runner.GPUModelRunner)
    runner.max_num_tokens = 32
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(rank=rank))
    runner.kv_cache_config = SimpleNamespace()
    runner.model = Mock()

    runner.init_routed_experts_capturer()

    constructor.assert_called_once_with(
        max_num_batched_tokens=32,
        vllm_config=runner.vllm_config,
        kv_cache_config=runner.kv_cache_config,
    )
    bind.assert_called_once_with(runner.model, capturer)
    assert runner.routed_experts_capturer is capturer


def test_v2_model_runner_accepts_routed_experts(monkeypatch):
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **_: ())
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_return_routed_experts=True,
            use_mla=False,
            logits_processors=None,
            enable_prompt_embeds=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            tensor_parallel_size=1,
            distributed_executor_backend=None,
            pipeline_parallel_size=1,
            enable_dbo=False,
            use_ubatching=False,
            enable_elastic_ep=False,
        ),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.NONE,
            pass_config=SimpleNamespace(enable_sp=False),
        ),
        cache_config=SimpleNamespace(
            kv_sharing_fast_prefill=False,
            mamba_cache_mode="none",
        ),
        ec_transfer_config=None,
    )

    unsupported = VllmConfig._get_v2_model_runner_unsupported_features(config)

    assert "routed experts capture" not in unsupported


# --------------------------------------------------------------------------- #
# YOCO logical-use capture (one routed-experts slot per universal-loop use).  #
# --------------------------------------------------------------------------- #

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (  # noqa: E402
    RoutedExpertsUseLayout,
    _check_use_layout_supported,
    get_routed_experts_use_layout,
)


def _agens_layer_id(use: int, num_self: int, loops: int) -> int:
    # Agens megatron/core/models/yoco/config.py YOCOPlacement.layer_id.
    self_uses = num_self * loops
    return use % num_self if use < self_uses else use - self_uses + num_self


def _rl_bridge_slot(layer: int, loop: int, num_self: int, loops: int) -> int:
    # rl-bridge models/agens_v3/patches/vllm-yoco-routing-replay.patch.
    if layer < num_self:
        return loop * num_self + layer
    return loops * num_self + layer - num_self


def _execution_order(layout: RoutedExpertsUseLayout) -> list[tuple[int, int]]:
    order = [
        (layer, loop)
        for loop in range(layout.universal_loop)
        for layer in range(layout.num_self_layers)
    ]
    order += [(layer, 0) for layer in range(layout.num_self_layers, layout.num_layers)]
    return [(layer, loop) for layer, loop in order if layer >= layout.dense_layers]


@pytest.mark.parametrize(
    "num_self,num_cross,loops", [(10, 10, 3), (4, 4, 3), (2, 3, 1)]
)
def test_yoco_use_layout_matches_agens_and_rl_bridge(num_self, num_cross, loops):
    layout = RoutedExpertsUseLayout(num_self + num_cross, num_self, loops)
    assert layout.num_uses == num_self * loops + num_cross
    uses = []
    for layer, loop in _execution_order(layout):
        use = layout.use_index(layer, loop)
        assert use == _rl_bridge_slot(layer, loop, num_self, loops)
        assert _agens_layer_id(use, num_self, loops) == layer
        assert layout.layer_of_use(use) == (layer, loop)
        uses.append(use)
    # Execution order is use order, every use exactly once.
    assert uses == list(range(layout.num_uses))


def test_yoco_use_layout_only_for_yoco():
    yoco = SimpleNamespace(
        model_type="yoco",
        num_hidden_layers=20,
        yoco_cross_layers=10,
        universal_loop=3,
        dense_layers=0,
    )
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=yoco))
    assert get_routed_experts_use_layout(vllm_config) == RoutedExpertsUseLayout(
        20, 10, 3, 0
    )
    other = SimpleNamespace(model_type="qwen3_moe", num_hidden_layers=20)
    vllm_config.model_config.hf_text_config = other
    assert get_routed_experts_use_layout(vllm_config) is None


def test_yoco_capture_refuses_fast_prefill_and_dbo():
    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=True),
        parallel_config=SimpleNamespace(enable_dbo=False),
    )
    with pytest.raises(ValueError, match="kv_sharing_fast_prefill"):
        _check_use_layout_supported(cfg)
    cfg.cache_config.kv_sharing_fast_prefill = False
    cfg.parallel_config.enable_dbo = True
    with pytest.raises(ValueError, match="dual-batch"):
        _check_use_layout_supported(cfg)


def _yoco_capturer(layout: RoutedExpertsUseLayout, *, tokens=5, top_k=2):
    c = _capturer_with_buffer(
        max_tokens=tokens, num_layers=layout.num_uses, num_experts_per_tok=top_k
    )
    c.use_layout = layout
    c._use_ctx = None
    c._use_counts = None
    return c


def _per_use_routes(layout, tokens, top_k, seed):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randint(0, 128, (tokens, top_k), generator=g, dtype=torch.int64)
        for _ in range(layout.num_uses)
    ]


def _fake_forward(capturer, routes, *, stop_after=None):
    """Call capture in YOCO launch order under one fresh ForwardContext."""
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        for i, (layer, loop) in enumerate(_execution_order(capturer.use_layout)):
            if stop_after is not None and i == stop_after:
                return
            capturer.capture(layer, routes[capturer.use_layout.use_index(layer, loop)])


def test_yoco_capture_writes_each_loop_to_its_own_use():
    layout = RoutedExpertsUseLayout(5, 3, 3)
    c = _yoco_capturer(layout)
    for seed in (0, 1):  # second forward = new context, counters restart
        routes = _per_use_routes(layout, 5, 2, seed)
        _fake_forward(c, routes)
        for use in range(layout.num_uses):
            assert torch.equal(c.device_buffer[:, use, :], routes[use].int())
        # The three loops of one self layer land in three different slots.
        assert len({layout.use_index(0, k) for k in range(3)}) == 3
        c.check_use_counts_complete()


def test_yoco_capture_rejects_extra_and_missing_uses():
    layout = RoutedExpertsUseLayout(5, 3, 3)
    routes = _per_use_routes(layout, 5, 2, 0)
    c = _yoco_capturer(layout)
    ctx = SimpleNamespace(dp_metadata=None)
    with patch(f"{_REC_MODULE}.get_forward_context", return_value=ctx):
        for _ in range(3):
            c.capture(0, routes[0])
        with pytest.raises(IndexError, match="routed uses per forward"):
            c.capture(0, routes[0])
        with pytest.raises(IndexError):
            c.capture(4, routes[0])
            c.capture(4, routes[0])

    c = _yoco_capturer(layout)
    _fake_forward(c, routes, stop_after=7)
    with pytest.raises(AssertionError, match="incomplete YOCO forward"):
        c.reset_after_snapshot()
    with pytest.raises(AssertionError, match="incomplete YOCO forward"):
        _fake_forward(c, routes)


def test_yoco_reset_after_snapshot_rearms_absent_sentinel():
    layout = RoutedExpertsUseLayout(5, 3, 3)
    c = _yoco_capturer(layout)
    _fake_forward(c, _per_use_routes(layout, 5, 2, 0))
    c.reset_after_snapshot()
    assert bool((c.device_buffer == -1).all())


def _yoco_manager(layout, num_experts=128, top_k=2, num_blocks=2, block_size=4):
    hf_config = SimpleNamespace(
        model_type="yoco",
        num_hidden_layers=layout.num_layers,
        yoco_cross_layers=layout.num_cross_layers,
        universal_loop=layout.universal_loop,
        dense_layers=layout.dense_layers,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
    )
    return RoutedExpertsManager(vllm_config, kv)


def test_yoco_manager_stores_present_mask_and_zeroes_absent_rows():
    layout = RoutedExpertsUseLayout(5, 3, 3)
    m = _yoco_manager(layout)
    assert m.routed_experts_by_slot.shape == (8, layout.num_uses, 2)
    data = (
        np.arange(3 * layout.num_uses * 2, dtype=np.int32).reshape(
            3, layout.num_uses, 2
        )
        % 128
    )
    data[1, 4] = -1  # use 4 not captured for the second token
    m.store_batch(data, np.array([5, 6, 7]))
    routes = m.get([0, 1], 8, token_start=5)
    present = m.get_present([0, 1], 8, token_start=5)
    assert present.shape == (3, layout.num_uses)
    assert not present[1, 4] and present.sum() == 3 * layout.num_uses - 1
    assert (routes[1, 4] == 0).all()
    expected = data.copy()
    expected[1, 4] = 0
    assert np.array_equal(routes, expected)
    with pytest.raises(AssertionError, match="partially"):
        bad = data.copy()
        bad[0, 0, 0] = -1
        m.present_rows(bad)


def test_non_yoco_manager_has_no_present_mask():
    hf_config = SimpleNamespace(
        num_experts=8, num_experts_per_tok=2, num_hidden_layers=3
    )
    vllm_config = SimpleNamespace(model_config=_make_model_config(hf_config))
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    kv = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["l"], spec)],
    )
    m = RoutedExpertsManager(vllm_config, kv)
    assert m.present_by_slot is None and m.get_present([0], 3) is None


def test_yoco_binding_requires_opaque_moe_op_when_compiled(monkeypatch):
    class DummyFusedMoE:
        def __init__(self, layer_id):
            self.layer_id = layer_id
            self.layer_name = f"model.layers.{layer_id}.mlp.experts"
            self.router = _make_router()
            self._quant_method = _make_modular_routed_experts().quant_method
            self._forward_entry = lambda *a: None  # traced, not opaque

    import vllm.model_executor.layers.fused_moe.layer as fused_moe_layer

    monkeypatch.setattr(fused_moe_layer, "MoERunner", DummyFusedMoE)
    layout = RoutedExpertsUseLayout(3, 2, 3)
    modules = [DummyFusedMoE(i) for i in range(3)]
    model = SimpleNamespace(modules=lambda: modules)
    capturer = SimpleNamespace(
        capture=lambda *a: None, use_layout=layout, compiled=True
    )
    with pytest.raises(ValueError, match="opaque"):
        bind_routed_experts_capturer(model, capturer)
    capturer.compiled = False
    bind_routed_experts_capturer(model, capturer)
    model = SimpleNamespace(modules=lambda: modules[:2])
    with pytest.raises(ValueError, match="expected"):
        bind_routed_experts_capturer(model, capturer)
