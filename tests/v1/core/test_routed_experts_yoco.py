# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-level contract of YOCO per-use routed-experts output (R3).

A fake model runner produces, for every computed token, routes that are a
deterministic function of (token prefix, logical use), i.e. what a
batch-invariant engine would capture. The real scheduler, KV-cache manager
and slot-indexed RoutedExpertsManager then assemble per-request outputs.
Checked: rows = T - 1 - start, every use present, multi-turn concatenation
(second turn sent with ``routed_experts_prompt_start``) equals a one-shot
request, chunked prefill and prefix-cache hits, and absent uses surfacing
in ``routed_experts_present``.
"""

import hashlib
import json

import numpy as np
import pytest

from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsManager,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput, RoutedExpertsLists
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from .utils import EOS_TOKEN_ID, create_scheduler

pytestmark = pytest.mark.cpu_test

NUM_SELF, NUM_CROSS, LOOPS, NUM_EXPERTS, TOP_K = 4, 2, 3, 16, 4
NUM_USES = NUM_SELF * LOOPS + NUM_CROSS
BLOCK = 16


@pytest.fixture(scope="module")
def yoco_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("yoco-cfg")
    config = {
        "architectures": ["YOCOForCausalLM"],
        "model_type": "yoco",
        "torch_dtype": "bfloat16",
        "hidden_size": 64,
        "d_model": 64,
        "intermediate_size": 128,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "cross_kv_head": 2,
        "head_dim": 16,
        "num_hidden_layers": NUM_SELF + NUM_CROSS,
        "n_layers": NUM_SELF + NUM_CROSS,
        "yoco_cross_layers": NUM_CROSS,
        "universal_loop": LOOPS,
        "dense_layers": 0,
        "num_experts": NUM_EXPERTS,
        "moe_expert_num": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_top_k": TOP_K,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "vocab_size": 60000,
        "max_position_embeddings": 4096,
        "sliding_window_size": 512,
        "rms_norm_eps": 1e-6,
        "router_weights_normalized": True,
        "tie_word_embeddings": False,
        "eos_token_id": EOS_TOKEN_ID,
    }
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _route(prefix: list[int], use: int) -> np.ndarray:
    digest = hashlib.sha256(np.asarray(prefix + [use], dtype=np.int64)).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return rng.choice(NUM_EXPERTS, TOP_K, replace=False).astype(np.int32)


def _expected_rows(tokens: list[int], start: int) -> np.ndarray:
    """Reference: routing of every forwarded position >= start."""
    return np.stack(
        [
            np.stack([_route(tokens[: pos + 1], u) for u in range(NUM_USES)])
            for pos in range(start, len(tokens) - 1)
        ]
    ).reshape(-1, NUM_USES, TOP_K)


_hasher_ready = False
CACHED: dict[str, int] = {}


def _request(req_id, prompt, max_tokens, start=0):
    global _hasher_ready
    if not _hasher_ready:
        init_none_hash(sha256)
        _hasher_ready = True
    params = SamplingParams(
        ignore_eos=True, max_tokens=max_tokens, routed_experts_prompt_start=start
    )
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _scheduler(yoco_dir, max_batched=8192, prefix_caching=True):
    scheduler = create_scheduler(
        model=yoco_dir,
        enable_prefix_caching=prefix_caching,
        num_blocks=400,
        block_size=BLOCK,
        max_num_batched_tokens=max_batched,
        max_model_len=2048,
        skip_tokenizer_init=True,
    )
    mgr = RoutedExpertsManager(scheduler.vllm_config, scheduler.kv_cache_config)
    assert mgr.routed_experts_by_slot.shape[1:] == (NUM_USES, TOP_K)
    scheduler.enable_return_routed_experts = True
    scheduler.routed_experts_mgr = mgr
    scheduler._re_block_ids = {}
    return scheduler


def _run(scheduler, requests, forced, absent=frozenset()):
    """Drive the scheduler with a fake runner until all requests finish.

    ``forced[req_id]`` lists the sampled tokens; ``absent`` holds
    (req_id, position, use) triples the fake runner leaves uncaptured (-1).
    Returns {req_id: (routes, present)} concatenated like OutputProcessor.
    """
    for req in requests:
        scheduler.add_request(req)
    chunks: dict[str, list] = {r.request_id: [] for r in requests}
    gid = scheduler.routed_experts_mgr.attn_gid
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(EngineCoreOutputs)
    while scheduler.has_unfinished_requests():
        so = scheduler.schedule()
        req_ids = list(so.num_scheduled_tokens)
        rows, slots, sampled = [], [], []
        for rid in req_ids:
            req = scheduler.requests[rid]
            n = so.num_scheduled_tokens[rid]
            begin = req.num_computed_tokens - n
            blocks = scheduler.kv_cache_manager.get_blocks(rid).get_block_ids()[gid]
            tokens = list(req.all_token_ids)
            for pos in range(begin, begin + n):
                row = np.stack([_route(tokens[: pos + 1], u) for u in range(NUM_USES)])
                for r, p, u in absent:
                    if r == rid and p == pos:
                        row[u] = -1
                rows.append(row)
                slots.append(blocks[pos // BLOCK] * BLOCK + pos % BLOCK)
            done_prefill = begin + n >= req.num_tokens
            sampled.append(
                [forced[rid][len(req.output_token_ids)]] if done_prefill else []
            )
        mro = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
            routed_experts=RoutedExpertsLists(
                routing_data=np.asarray(rows, dtype=np.int32),
                slot_mapping=np.asarray(slots, dtype=np.int64),
            ),
        )
        for eco in scheduler.update_from_output(so, mro).values():
            # Engine-core -> frontend transport is msgpack, no pickle.
            frames = encoder.encode(eco)
            eco = decoder.decode(frames)
            for out in eco.outputs:
                if out.prefill_stats is not None:
                    CACHED[out.request_id] = out.prefill_stats.num_cached_tokens
                if out.routed_experts is not None:
                    assert out.routed_experts_present is not None
                    chunks[out.request_id].append(
                        (out.routed_experts, out.routed_experts_present)
                    )
    result = {}
    for rid, parts in chunks.items():
        routes = np.concatenate([p[0] for p in parts])
        present = np.concatenate([p[1] for p in parts])
        assert present.shape == routes.shape[:2]
        result[rid] = (routes.astype(np.int32), present)
    return result


@pytest.mark.parametrize("max_batched", [8192, 24])
def test_single_request_rows_and_routes(yoco_dir, max_batched):
    prompt = list(range(100, 150))
    out = [7, 8, 9, 10, 11]
    s = _scheduler(yoco_dir, max_batched=max_batched)
    res = _run(s, [_request("a", prompt, len(out))], {"a": out})
    routes, present = res["a"]
    tokens = prompt + out
    assert routes.shape == (len(tokens) - 1, NUM_USES, TOP_K)  # T - 1
    assert present.all()
    assert np.array_equal(routes, _expected_rows(tokens, 0))


@pytest.mark.parametrize("prefix_caching", [True, False])
def test_two_turns_with_start_concat_equals_one_shot(yoco_dir, prefix_caching):
    p1 = list(range(200, 237))  # not block aligned
    o1 = [21, 22, 23, 24]
    tool = list(range(300, 311))
    o2 = [31, 32, 33]
    p2 = p1 + o1 + tool
    start2 = len(p1) + len(o1) - 1  # Miles routed_experts_start_len

    s = _scheduler(yoco_dir, prefix_caching=prefix_caching)
    r1 = _run(s, [_request("t1", p1, len(o1))], {"t1": o1})["t1"]
    turn2 = _request("t2", p2, len(o2), start=start2)
    r2 = _run(s, [turn2], {"t2": o2})["t2"]
    # Turn 2 re-reads turn 1's full blocks from the prefix cache.
    assert CACHED["t2"] == (32 if prefix_caching else 0)
    one = _run(
        _scheduler(yoco_dir, prefix_caching=prefix_caching),
        [_request("one", p2, len(o2))],
        {"one": o2},
    )["one"]

    tokens = p2 + o2
    assert r1[0].shape[0] == len(p1) + len(o1) - 1
    assert r2[0].shape[0] == len(tokens) - 1 - start2
    concat = np.concatenate([r1[0], r2[0]])
    assert concat.shape[0] == len(tokens) - 1
    assert np.array_equal(concat, one[0])
    assert np.concatenate([r1[1], r2[1]]).all() and one[1].all()


def test_concurrent_requests_share_prefix_cache(yoco_dir):
    shared = list(range(400, 432))  # two full blocks
    a, b = shared + [1, 2, 3], shared + [4, 5]
    s = _scheduler(yoco_dir)
    first = _run(s, [_request("a", a, 3)], {"a": [9, 9, 9]})
    req_b = _request("b", b, 2)
    second = _run(s, [req_b], {"b": [8, 8]})
    assert CACHED["b"] == 32
    for rid, prompt, out, res in (
        ("a", a, [9, 9, 9], first),
        ("b", b, [8, 8], second),
    ):
        tokens = prompt + out
        assert np.array_equal(res[rid][0], _expected_rows(tokens, 0))
        assert res[rid][1].all()


def test_absent_use_is_reported_and_zeroed(yoco_dir):
    prompt = list(range(500, 520))
    out = [1, 2, 3]
    # One prompt row (read back from the slot buffer) and one decode row.
    absent = {("x", 5, 13), ("x", len(prompt) + 1, 2)}
    s = _scheduler(yoco_dir)
    routes, present = _run(s, [_request("x", prompt, len(out))], {"x": out}, absent)[
        "x"
    ]
    assert sorted(zip(*np.nonzero(~present))) == [(5, 13), (len(prompt) + 1, 2)]
    assert (routes[5, 13] == 0).all() and (routes[len(prompt) + 1, 2] == 0).all()
    expected = _expected_rows(prompt + out, 0)
    keep = present[..., None].repeat(TOP_K, -1)
    assert np.array_equal(routes[keep], expected[keep])


def test_engine_core_output_present_defaults_to_none():
    out = EngineCoreOutput(request_id="r", new_token_ids=[1])
    assert out.routed_experts_present is None


def test_delta_coalescing_keeps_final_routing_snapshot():
    from vllm.outputs import CompletionOutput, RequestOutput

    def ro(tokens, routes=None, present=None, finished=False):
        c = CompletionOutput(
            index=0,
            text="",
            token_ids=tokens,
            cumulative_logprob=None,
            logprobs=None,
            routed_experts=routes,
            routed_experts_present=present,
            finish_reason="length" if finished else None,
        )
        return RequestOutput("r", None, [1, 2], None, [c], finished)

    routes = np.zeros((3, NUM_USES, TOP_K), np.uint8)
    present = np.ones((3, NUM_USES), bool)
    acc = ro([5])
    acc.add(ro([6], routes, present, finished=True), aggregate=True)
    assert acc.outputs[0].token_ids == [5, 6]
    assert acc.outputs[0].routed_experts is routes
    assert acc.outputs[0].routed_experts_present is present
