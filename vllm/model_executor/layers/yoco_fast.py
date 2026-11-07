# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision-aware building blocks shared by YOCO Fast implementations."""

import torch

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


def refresh_yoco_weight_cache(
    cached: torch.Tensor | None, value: torch.Tensor
) -> torch.Tensor:
    """Refresh derived weights/scales without invalidating captured addresses.

    A shape or precision change requires rebuilding graphs, not silently
    replacing storage that an existing graph may still reference.
    """
    if cached is None:
        return value
    if (
        cached.shape != value.shape
        or cached.dtype != value.dtype
        or cached.device != value.device
    ):
        raise ValueError("Incompatible YOCO weight cache on weight reload")
    # With dummy loading, the first forward (profile run, under inference_mode)
    # creates the cache, so it is an inference tensor; RL refit then reloads
    # outside inference_mode. Write through .data (as Worker.wake_up does for
    # saved buffers) to update the same storage in either mode.
    cached.data.copy_(value)
    return cached


def refresh_yoco_module_cache(
    module: torch.nn.Module, name: str, value: torch.Tensor
) -> torch.Tensor:
    """Recover captured storage when layerwise reload temporarily uses meta."""
    cached = getattr(module, name, None)
    if cached is None or cached.is_meta:
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

        saved = get_layerwise_info(module).kernel_tensors
        if saved is not None:
            cached = saved[1].get(name, cached)
            # Metadata can have been recorded before the derived buffer was
            # first created. In that case restore_layer_on_meta removed its
            # registration; restore it before assigning the refreshed value.
            if name in saved[1] and name not in module._buffers:
                if hasattr(module, name):
                    delattr(module, name)
                module.register_buffer(name, cached, persistent=False)
    return refresh_yoco_weight_cache(cached, value)


def yoco_fast_linear_fusion(
    execution_mode: str,
    tp_size: int,
    quant_config: QuantizationConfig | None,
    source_prefixes: tuple[str, ...],
    input_size: int,
    output_sizes: tuple[int, ...],
    *,
    allow_fp8: bool = False,
) -> tuple[bool, QuantizationConfig | None]:
    """Resolve a fused projection from its actual component precisions.

    Online block quantization can concatenate complete N blocks. Tensorwise
    quantization cannot: merging its scale domains changes quantization.
    Unrecognized checkpoint formats retain their original loading path.
    """
    if execution_mode != "fast" or tp_size != 1:
        return False, None
    if quant_config is None:
        return True, None

    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )
    from vllm.model_executor.layers.quantization.online.base import (
        OnlineQuantizationConfig,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8Static128BlockSym,
    )

    if not isinstance(quant_config, OnlineQuantizationConfig):
        return False, None
    spec = quant_config.args.linear
    if spec is None or spec.weight is None:
        return True, None
    ignored = [
        should_ignore_layer(
            prefix,
            ignore=quant_config.ignored_layers,
            fused_mapping=quant_config.packed_modules_mapping,
        )
        for prefix in source_prefixes
    ]
    if all(ignored):
        return True, None
    if (
        allow_fp8
        and not any(ignored)
        and spec.weight == kFp8Static128BlockSym
        and spec.activation is None
        and input_size % 128 == 0
        and all(size % 128 == 0 for size in output_sizes)
    ):
        return True, quant_config
    return False, None
