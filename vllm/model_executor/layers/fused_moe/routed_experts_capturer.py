# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from
# https://github.com/sgl-project/sglang/blob/bed301a5acaa9577c9aa706468bdf242f6a43051/python/sglang/srt/layers/moe/routed_experts_capturer.py

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig, is_full_attention_spec
from vllm.v1.outputs import RoutedExpertsTensors

logger = logging.getLogger(__name__)


@runtime_checkable
class RoutedExpertsCaptureSource(Protocol):
    layer_id: int
    capture_fn: Callable[[torch.Tensor], None] | None


@dataclass(frozen=True)
class RoutedExpertsUseLayout:
    """Logical-use layout of routed-experts rows for models that execute the
    same MoE layer more than once per forward (YOCO's ``universal_loop``).

    Second dimension of the routed-experts buffers is the *logical use*, the
    same ordering as Agens ``YOCOPlacement`` and rl-bridge's
    ``vllm-yoco-routing-replay.patch``:

    * self layer ``l`` (``l < num_self_layers``) on loop ``k``:
      ``use = k * num_self_layers + l``;
    * cross layer ``l`` (``l >= num_self_layers``), executed once:
      ``use = universal_loop * num_self_layers + (l - num_self_layers)``.

    Dense layers (``l < dense_layers``) keep their use slots but never route;
    their rows are reported absent in ``routed_experts_present``.
    """

    num_layers: int
    num_self_layers: int
    universal_loop: int
    dense_layers: int = 0

    def __post_init__(self) -> None:
        if not (
            0 < self.num_self_layers <= self.num_layers
            and self.universal_loop >= 1
            and 0 <= self.dense_layers <= self.num_layers
        ):
            raise ValueError(f"Invalid routed-experts use layout: {self}")

    @property
    def num_cross_layers(self) -> int:
        return self.num_layers - self.num_self_layers

    @property
    def num_uses(self) -> int:
        return self.num_self_layers * self.universal_loop + self.num_cross_layers

    def calls_per_forward(self, layer_id: int) -> int:
        """Routed MoE executions of ``layer_id`` in one full forward."""
        self._check_layer(layer_id)
        if layer_id < self.dense_layers:
            return 0
        return self.universal_loop if layer_id < self.num_self_layers else 1

    def use_index(self, layer_id: int, loop_idx: int) -> int:
        calls = self.calls_per_forward(layer_id)
        if not 0 <= loop_idx < calls:
            raise IndexError(
                f"routed-experts layer {layer_id} has {calls} routed uses per "
                f"forward; got use {loop_idx}"
            )
        if layer_id < self.num_self_layers:
            return loop_idx * self.num_self_layers + layer_id
        return (
            self.universal_loop * self.num_self_layers + layer_id - self.num_self_layers
        )

    def layer_of_use(self, use: int) -> tuple[int, int]:
        """Inverse map: ``use -> (layer_id, loop_idx)``."""
        if not 0 <= use < self.num_uses:
            raise IndexError(f"routed-experts use {use} outside [0, {self.num_uses})")
        self_uses = self.num_self_layers * self.universal_loop
        if use < self_uses:
            return use % self.num_self_layers, use // self.num_self_layers
        return use - self_uses + self.num_self_layers, 0

    def _check_layer(self, layer_id: int) -> None:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError(
                f"routed-experts layer {layer_id} outside [0, {self.num_layers})"
            )


def _cfg_first_int(config, *names: str, default: int | None = None) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    if default is None:
        raise ValueError(f"config has none of {names}")
    return default


def get_routed_experts_use_layout(
    vllm_config: VllmConfig,
) -> RoutedExpertsUseLayout | None:
    """Return the logical-use layout for YOCO, ``None`` for every other model.

    ``None`` keeps the historical per-layer layout bit-for-bit.
    """
    model_config = vllm_config.model_config
    hf_config = model_config.hf_text_config
    if getattr(hf_config, "model_type", None) != "yoco":
        return None
    num_layers = _cfg_first_int(hf_config, "num_hidden_layers", "n_layers")
    cross = _cfg_first_int(hf_config, "yoco_cross_layers", default=0)
    return RoutedExpertsUseLayout(
        num_layers=num_layers,
        num_self_layers=num_layers - cross,
        universal_loop=_cfg_first_int(hf_config, "universal_loop", default=1),
        dense_layers=_cfg_first_int(hf_config, "dense_layers", default=0),
    )


def _get_routed_experts_shape(vllm_config: VllmConfig) -> tuple[int, int, int]:
    model_config = vllm_config.model_config
    layout = get_routed_experts_use_layout(vllm_config)
    num_layers = (
        layout.num_uses
        if layout is not None
        else model_config.get_total_num_hidden_layers()
    )
    num_experts = model_config.get_num_experts()
    num_experts_per_tok = model_config.get_num_experts_per_tok()
    if num_layers <= 0 or num_experts <= 0 or num_experts_per_tok <= 0:
        raise ValueError(
            "Routed-experts capture requires positive layer, expert, and "
            "experts-per-token counts, got "
            f"{num_layers=}, {num_experts=}, {num_experts_per_tok=}."
        )
    return num_layers, num_experts, num_experts_per_tok


class RoutedExpertsCapturer:
    """Worker-side capturer for routed experts, lives on GPU.

    Layer-level hooks call :meth:`capture` from inside the forward pass
    with the per-layer ``topk_ids`` tensor. The tensor is sliced to the
    tokens owned by this DP rank and written into a preallocated device
    buffer. At the end of the step, :class:`GPUModelRunner` reads the
    device buffer, issues a D2H copy into a pinned CPU buffer, and hands
    the result to the scheduler via :class:`RoutedExpertsLists`.

    The device / pinned-CPU transit buffers use ``torch.int32`` (not a
    narrow ``uint8``/``uint16`` sized by ``num_experts``). This keeps the
    SP all-gather path free of dtype casts, matches the router's native
    ``topk_ids`` indices dtype more closely, and costs only a few MB per
    worker (``max_num_batched_tokens * num_layers * top_k * 4`` bytes).
    The scheduler-side slot buffer
    (``RoutedExpertsManager.routed_experts_by_slot``) still uses the
    narrow dtype -- numpy fancy-index assignment in ``store_batch``
    narrows the data on the way in.

    Invariants:
        - One instance per worker; shape is fixed at init and covers the
          worst-case step (``max_num_batched_tokens`` tokens).
        - Every routed layer overwrites the current step's token rows.
        - ``device_buffer.dtype`` is ``torch.int32``.
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        num_layers, _, num_experts_per_tok = _get_routed_experts_shape(vllm_config)
        logger.info(
            "RoutedExpertsCapturer: allocating buffer with "
            "max_tokens=%d, num_layers=%d, num_experts_per_tok=%d "
            "(hf_config.model_type=%s)",
            max_num_batched_tokens,
            num_layers,
            num_experts_per_tok,
            vllm_config.model_config.hf_text_config.model_type,
        )
        self.device_buffer = torch.zeros(
            (
                max_num_batched_tokens,
                num_layers,
                num_experts_per_tok,
            ),
            # Use int32 for the device / host transit buffers: it
            # matches the router's native topk_ids dtype, is universally
            # supported by NCCL (uint8/uint16 are version-dependent),
            # and the extra bytes are small (few MB per worker). The
            # big scheduler-side slot buffer stays narrow.
            dtype=torch.int32,
            device=current_platform.device_type,
        )
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        # KV cache group whose slot layout the routing data is keyed by.
        self.attn_gid = get_routed_experts_attn_gid(kv_cache_config)

        # Logical-use layout (YOCO universal loop); None = per-layer layout.
        self.use_layout = get_routed_experts_use_layout(vllm_config)
        # Per-forward use counters. ``_use_ctx`` is the ForwardContext the
        # counters belong to; holding a strong reference makes the identity
        # test immune to id() reuse.
        self._use_ctx: object | None = None
        self._use_counts: list[int] | None = None
        from vllm.config import CompilationMode

        self.compiled = vllm_config.compilation_config.mode != CompilationMode.NONE
        if self.use_layout is not None:
            _check_use_layout_supported(vllm_config)
            # -1 marks "not captured in this step" (present mask source).
            self.device_buffer.fill_(-1)

    def _next_use(self, layer_id: int) -> int:
        """Map this MoE execution of ``layer_id`` to its logical use.

        Runs as Python *inside* the opaque ``vllm.moe_forward`` custom op, so
        it executes in launch order: eagerly on every eager step, and once
        per call site while a CUDA graph is being captured (the resulting
        buffer address is then frozen in the graph and replayed). The k-th
        execution of a layer within one ForwardContext is loop ``k``.
        """
        layout = self.use_layout
        assert layout is not None
        ctx = get_forward_context()
        if ctx is not self._use_ctx:
            self.check_use_counts_complete()
            self._use_ctx = ctx
            self._use_counts = [0] * layout.num_layers
        counts = self._use_counts
        assert counts is not None
        loop_idx = counts[layer_id]
        use = layout.use_index(layer_id, loop_idx)  # raises past the last loop
        counts[layer_id] = loop_idx + 1
        return use

    def check_use_counts_complete(self) -> None:
        """Fail loudly if the last Python-executed forward skipped a use.

        Idempotent; a CUDA-graph replay executes no Python and leaves the
        (already validated) counters of the capturing forward untouched.
        """
        layout = getattr(self, "use_layout", None)
        counts = getattr(self, "_use_counts", None)
        if layout is None or counts is None:
            return
        missing = [
            (layer, counts[layer], layout.calls_per_forward(layer))
            for layer in range(layout.num_layers)
            if counts[layer] != layout.calls_per_forward(layer)
        ]
        if missing:
            raise AssertionError(
                "RoutedExpertsCapturer: incomplete YOCO forward, "
                "(layer, captured uses, expected uses) = " + repr(missing)
            )

    def reset_after_snapshot(self) -> None:
        """Re-arm the absent sentinel once this step's rows are snapshotted.

        Must be enqueued on the same stream after the D2H copy / clone of
        ``device_buffer``. No-op for the per-layer layout.
        """
        if getattr(self, "use_layout", None) is None:
            return
        self.check_use_counts_complete()
        self.device_buffer.fill_(-1)

    def capture(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """Capture expert routing decisions for a specific layer.

        Under data parallelism, ``topk_ids`` may have four different batch
        layouts depending on where the DP combine happens and whether
        Expert Parallelism (EP) or Sequence Parallelism (SP) is active for the
        MoE layer:
          - ``n == total`` (naive dispatch): all DP ranks' tokens are
            concatenated before routing; we slice out this rank's span
            using the cumulative per-rank counts.
          - ``n == token_num_per_dp`` (modular-kernel path): DP combine
            happens inside ``quant_method.apply``; ``select_experts`` only
            ever sees this rank's tokens, so we take the whole tensor.
          - ``n == sum(dp_metadata.local_sizes)`` (naive DP+EP dispatch):
            sequence-parallel shards from every DP rank are gathered through
            the flattened EP group. The shard sizes include CUDA-graph / SP
            padding, so we use them to locate this DP rank's unpadded rows.
          - ``n == ceil(token_num_per_dp / tp_size)`` (SP + modular-kernel
            path): tokens were split along dim=0 across the TP group by
            ``_sequence_parallel_context``
            (``moe_runner_base.py:_sequence_parallel_context``), so each
            TP rank only sees its shard. We all-gather along dim=0 to
            reconstruct this DP rank's full routing tensor. SP pads with
            ceil-div (see ``_compute_sp_num_tokens`` in
            ``forward_context.py``), so the gathered tensor may contain a
            few trailing padding rows which are trimmed by the downstream
            ``[:token_num_per_dp]`` slice.

        Args:
            layer_id: The layer index.
            topk_ids: Tensor of shape (batch_size, num_routed_experts).
        """

        ctx = get_forward_context()
        if ctx.dp_metadata is None:  # single dp
            start_loc = 0
            end_loc = topk_ids.shape[0]
            token_num_per_dp = topk_ids.shape[0]
        else:  # multi dp
            num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
            token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
            total = int(num_tokens_dp.sum().item())
            n = topk_ids.shape[0]
            shard_sizes = getattr(ctx.dp_metadata, "local_sizes", None)
            gathered_size = sum(shard_sizes) if shard_sizes is not None else -1

            if n == total:
                # Naive dispatch: all DP ranks' tokens concatenated
                # before routing. This rank owns tokens
                # [end_loc - token_num_per_dp, end_loc).
                cumsum = torch.cumsum(num_tokens_dp, dim=0)
                end_loc = int(cumsum[self.dp_rank].item())
                start_loc = end_loc - token_num_per_dp
            elif n == token_num_per_dp:
                # Modular-kernel path: DP combine happens inside
                # quant_method.apply; select_experts only sees this
                # rank's tokens, take the whole tensor.
                start_loc = 0
                end_loc = token_num_per_dp
            elif shard_sizes is not None and n == gathered_size:
                # Naive DP+EP dispatch gathers the sequence-parallel shards in
                # flattened DP-then-TP order. ``local_sizes`` is the exact
                # all-gatherv layout, including padding. Select this DP rank's
                # contiguous shard group, then trim its trailing padding.
                num_dp_ranks = len(num_tokens_dp)
                assert len(shard_sizes) % num_dp_ranks == 0
                shards_per_dp_rank = len(shard_sizes) // num_dp_ranks
                first_shard = self.dp_rank * shards_per_dp_rank
                start_loc = sum(shard_sizes[:first_shard])
                end_loc = start_loc + token_num_per_dp
            elif (
                self.tp_size > 1
                and n != token_num_per_dp
                and n == (token_num_per_dp + self.tp_size - 1) // self.tp_size
            ):
                # SP + modular-kernel path. All-gather across the TP
                # group along dim=0 to reconstruct the full per-DP-rank
                # tensor; keep only the first ``token_num_per_dp`` rows
                # (trailing rows are SP ceil-div padding). The TP group
                # is always initialized on real rollout workers, and
                # every rank in the group reaches this branch in
                # lockstep (bind is per-FusedMoEFactory layer, SP is a global
                # condition), so a bare all_gather here will not
                # deadlock -- let it raise if the precondition is
                # violated rather than skip silently.
                #
                # ``topk_ids`` is already whatever the router produced
                # (typically int32/int64, both supported by NCCL); the
                # downstream ``device_buffer[...] = topk_ids[...]``
                # setitem narrows into int32 automatically.
                topk_ids = get_tp_group().all_gather(topk_ids, dim=0)
                start_loc = 0
                end_loc = token_num_per_dp
            else:
                sp_expected = (
                    (token_num_per_dp + self.tp_size - 1) // self.tp_size
                    if self.tp_size > 0
                    else -1
                )
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected topk_ids batch "
                    f"dim {n} (expected {total}, {token_num_per_dp}, "
                    f"{gathered_size}, or {sp_expected} for "
                    f"dp_rank={self.dp_rank}, tp_size={self.tp_size})"
                )

        layout = getattr(self, "use_layout", None)
        slot = layer_id if layout is None else self._next_use(layer_id)
        if slot >= self.device_buffer.shape[1]:
            raise IndexError(
                f"routed-experts layer {layer_id} (slot {slot}) exceeds capture "
                f"buffer layer count {self.device_buffer.shape[1]}"
            )

        self.device_buffer[:token_num_per_dp, slot, :] = topk_ids[start_loc:end_loc, :]

    def get_device_buffer(self) -> torch.Tensor:
        """Return the underlying device buffer so the model runner can
        issue the D2H copy. The tensor is shared; callers must either
        clone or fully drain it before the next forward pass overwrites it.
        """
        return self.device_buffer

    def get_routed_experts(
        self, slot_mappings: torch.Tensor, num_tokens: int
    ) -> RoutedExpertsTensors:
        """Snapshot this step's routing data and its attention slot mapping.

        Both tensors are cloned since the capture buffer and the shared
        ``slot_mappings`` are overwritten by the next step, which may race
        with an in-flight D2H copy.

        Args:
            slot_mappings: Per-KV-cache-group slot mappings for this step,
                shape ``(num_kv_cache_groups, max_num_batched_tokens)``.
            num_tokens: Total number of tokens scheduled in this step.
        """
        snapshot = RoutedExpertsTensors(
            routing_data=self.device_buffer[:num_tokens].clone(),
            slot_mapping=slot_mappings[self.attn_gid, :num_tokens].clone(),
        )
        self.reset_after_snapshot()
        return snapshot


def _check_use_layout_supported(vllm_config: VllmConfig) -> None:
    """Configurations where per-use YOCO capture is known to be row-exact."""
    if vllm_config.cache_config.kv_sharing_fast_prefill:
        # Fast prefill runs the cross layers on compacted logits rows only, so
        # cross-use rows would be written at compact (wrong) offsets. Decision
        # S6-D4 / D149: R3 runs with fast prefill off.
        raise ValueError(
            "YOCO routed-experts capture (enable_return_routed_experts) "
            "requires kv_sharing_fast_prefill=False (S6-D4)."
        )
    if vllm_config.parallel_config.enable_dbo:
        raise ValueError(
            "YOCO routed-experts capture does not support dual-batch overlap."
        )
    if getattr(vllm_config, "speculative_config", None) is not None:
        # Drafters (e.g. YOCO Shared-KV / dspark) may run extra forwards or
        # share target modules; per-use counting is unverified there.
        raise ValueError(
            "YOCO routed-experts capture does not support speculative decoding."
        )


def _moe_entry_is_opaque(module) -> bool:
    entry = getattr(module, "_forward_entry", None)
    ops = getattr(torch.ops, "vllm", None)
    opaque = tuple(
        getattr(ops, name, None) for name in ("moe_forward", "moe_forward_shared")
    )
    return entry is not None and any(entry is op for op in opaque if op is not None)


def bind_routed_experts_capturer(
    model: torch.nn.Module,
    capturer: RoutedExpertsCapturer,
) -> None:
    """Attach capture callbacks to the target model's MoE routers."""
    from vllm.model_executor.layers.fused_moe.layer import MoERunner
    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEExpertsMonolithic,
    )
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    layout = getattr(capturer, "use_layout", None)
    compiled = getattr(capturer, "compiled", True)
    bound_layers: list[int] = []
    num_bound = 0
    for module in model.modules():
        if isinstance(module, RoutedExpertsCaptureSource):
            if layout is not None:
                raise ValueError(
                    "Per-use routed-experts capture supports MoERunner layers only."
                )
            module.capture_fn = partial(capturer.capture, module.layer_id)
            num_bound += 1
            continue
        if not isinstance(module, MoERunner):
            continue
        layer_id = module.layer_id
        if layout is not None:
            # The use counter must execute at MoE launch time, i.e. inside the
            # opaque custom op. A traced (non-opaque) entry would freeze the
            # counter at trace time instead.
            if compiled and not _moe_entry_is_opaque(module):
                raise ValueError(
                    "YOCO routed-experts capture under torch.compile needs the "
                    "opaque vllm.moe_forward custom op; "
                    f"{module.layer_name} uses {module._forward_entry!r}."
                )
            bound_layers.append(layer_id)

        def capture_fn(
            topk_ids: torch.Tensor,
            layer_id: int = layer_id,
            capturer: RoutedExpertsCapturer = capturer,
        ) -> None:
            capturer.capture(layer_id, topk_ids)

        quant_method = module._quant_method
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        impl = getattr(moe_kernel, "impl", None)
        fused_experts = getattr(impl, "fused_experts", None)
        if quant_method.is_monolithic:
            if not (
                isinstance(fused_experts, FusedMoEExpertsMonolithic)
                and fused_experts.supports_routing_replay_capture()
            ):
                raise ValueError(
                    "Routed-experts capture is not supported with monolithic "
                    f"MoE kernel {type(fused_experts).__name__}."
                )
            fused_experts.set_capture_fn(capture_fn)
            num_bound += 1
        elif isinstance(module.router, BaseRouter):
            module.router.set_capture_fn(capture_fn)
            num_bound += 1
        else:
            raise ValueError(
                "Routed-experts capture is not supported with router "
                f"{type(module.router).__name__}."
            )

    if num_bound == 0:
        raise ValueError("No supported MoE router found for routed-experts capture.")
    if layout is not None:
        expected = [
            layer
            for layer in range(layout.num_layers)
            if layout.calls_per_forward(layer) > 0
        ]
        if sorted(bound_layers) != expected:
            raise ValueError(
                "YOCO routed-experts capture bound MoE layers "
                f"{sorted(bound_layers)}, expected {expected}."
            )


def get_routed_experts_attn_gid(kv_cache_config: KVCacheConfig) -> int:
    """Return the full-attention KV cache group used for routed experts."""
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        if is_full_attention_spec(group.kv_cache_spec):
            return gid
    raise ValueError("Routed-experts capture requires a full-attention KV cache group.")


class RoutedExpertsManager:
    """Scheduler-side slot-indexed buffer for routed experts.

    Lives on CPU in the scheduler process. Each slot corresponds to
    ``block_id * block_size + offset_in_block`` where ``block_id`` is
    drawn from the physical KV-cache block pool, so routing data is
    tied to physical blocks and naturally survives preemption for
    prefix-cached blocks (prefix hits re-expose the same slots).

    Data flow per step:
      1. Worker D2Hs its device capture buffer into
         :class:`RoutedExpertsLists` and returns it via
         :class:`ModelRunnerOutput`.
      2. Scheduler calls :meth:`store_batch` with that step's
         ``(routing_data, slot_mapping)`` — a single CPU->CPU
         fancy-index assign, ~few MB per step.
      3. On request completion / abort / preemption, the scheduler
         calls :meth:`get` with the request's block IDs to recover
         the full per-token routing.

    Memory: ``routed_experts_by_slot`` is sized for the whole block
    pool (``num_blocks * block_size`` slots). For large block pools
    this can reach multiple GB; see the init log for the exact size.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        # Hybrid models also have KV groups whose slot layout differs.
        self.attn_gid = get_routed_experts_attn_gid(kv_cache_config)
        attn_group = kv_cache_config.kv_cache_groups[self.attn_gid]
        self.block_size = attn_group.kv_cache_spec.block_size

        # All kv_cache_groups share the same physical block pool, so
        # block IDs span [0, num_blocks) regardless of how many groups
        # exist. Sizing to the full pool avoids index-out-of-range
        # when different groups happen to land on the same block.
        num_layers, num_experts, num_experts_per_tok = _get_routed_experts_shape(
            vllm_config
        )
        max_num_slots = kv_cache_config.num_blocks * self.block_size
        # Expert IDs are 0..num_experts-1; uint8 fits 256 distinct
        # values so the boundary is ``<= 256`` (NOT ``< 256``). Keeping
        # this narrow matters because the slot buffer is sized for the
        # whole block pool and can reach multiple GB.
        expert_id_dtype = np.uint8 if num_experts <= 256 else np.uint16
        self.routed_experts_by_slot = np.zeros(
            (
                max_num_slots,
                num_layers,
                num_experts_per_tok,
            ),
            dtype=expert_id_dtype,
        )
        # Per-(slot, use) presence; only for the logical-use layout (YOCO).
        # A use is present iff the capturer wrote it in the step that
        # computed the slot. Absent rows are stored as expert 0 in
        # ``routed_experts_by_slot`` and must not be replayed.
        self.use_layout = get_routed_experts_use_layout(vllm_config)
        self.present_by_slot: np.ndarray | None = (
            np.zeros((max_num_slots, num_layers), dtype=np.bool_)
            if self.use_layout is not None
            else None
        )
        logger.info(
            "RoutedExpertsManager CPU buffer: %.2f GB "
            "(slots=%d, layers=%d, top_k=%d, dtype=%s)",
            self.routed_experts_by_slot.nbytes / 1e9,
            max_num_slots,
            num_layers,
            num_experts_per_tok,
            self.routed_experts_by_slot.dtype.name,
        )

    def store_batch(
        self,
        data: np.ndarray,
        slot_mapping: np.ndarray,
        present: np.ndarray | None = None,
    ) -> None:
        """Persist one step's routed experts into the slot buffer.

        Equivalent to ``slot_buffer[slot_mapping] = data``; numpy fancy
        indexing handles repeated / out-of-order indices. Called once
        per scheduler step in ``update_from_output``.
        """
        if self.present_by_slot is None:
            self.routed_experts_by_slot[slot_mapping] = data
            return
        if present is None:
            present = self.present_rows(data)
        if not present.all():
            data = np.where(present[..., None], data, 0)
        self.routed_experts_by_slot[slot_mapping] = data
        self.present_by_slot[slot_mapping] = present

    @staticmethod
    def present_rows(data: np.ndarray) -> np.ndarray:
        """``(tokens, uses)`` presence of raw (signed, pre-narrowing) capture
        data: a use is present iff all its top-k ids are >= 0 (the capturer
        writes all k at once; -1 is the not-captured sentinel)."""
        if not np.issubdtype(data.dtype, np.signedinteger):
            raise TypeError(
                f"presence needs the signed capture dtype, got {data.dtype}"
            )
        valid = data >= 0
        present = valid.all(axis=-1)
        if (valid.any(axis=-1) & ~present).any():
            raise AssertionError("routed-experts row is partially captured")
        return present

    def get_present(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray | None:
        """``(num_tokens - token_start, num_uses)`` presence; ``None`` for the
        per-layer layout. Same slot arithmetic as :meth:`get`."""
        if self.present_by_slot is None:
            return None
        return self.present_by_slot[
            self._slot_mapping(block_ids, num_tokens, token_start)
        ]

    def _slot_mapping(
        self, block_ids: list[int], num_tokens: int, token_start: int
    ) -> np.ndarray:
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        return slot_mapping[token_start:]

    def get(
        self,
        block_ids: list[int],
        num_tokens: int,
        token_start: int = 0,
    ) -> np.ndarray:
        """Read routed experts data for a completed / preempted request.

        Reconstructs a per-token slot_mapping from the request's block
        IDs and returns the routing slice. Because numpy fancy indexing
        returns a **copy** (not a view), the returned ndarray is safe
        to hold across subsequent :meth:`store_batch` calls — do not
        replace the fancy index with a slice without re-verifying.

        Args:
            block_ids: Block IDs from the attention KV-cache group.
            num_tokens: Number of tokens that have gone through a forward
                pass and therefore have routing data written to their
                slots (typically ``request.num_tokens - 1``; the last
                sampled token has not been forwarded yet). Slots beyond
                ``request.num_computed_tokens`` are zero-initialized.
            token_start: Skip the first ``token_start`` tokens from the
                result. The slot_mapping is sliced before the fancy-index
                read, so only the requested slots are fetched — no large
                intermediate array is allocated. Clamped to
                ``[0, num_tokens]`` automatically.

        Returns:
            Array of shape (num_tokens - token_start, num_layers,
            num_experts_per_tok).
        """
        bs = self.block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        block_offsets = np.arange(bs)
        # slot = block_id * block_size + offset_in_block; flatten the
        # (num_blocks, block_size) grid and trim to num_tokens, then
        # skip the first token_start entries so only the requested
        # range is fetched in a single fancy-index read.
        slot_mapping = (
            block_ids_array.reshape(-1, 1) * bs + block_offsets.reshape(1, -1)
        ).flatten()[:num_tokens]
        slot_mapping = slot_mapping[token_start:]
        return self.routed_experts_by_slot[slot_mapping]
