# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.yoco_fast import (
    refresh_yoco_weight_cache,
    yoco_fast_linear_fusion,
)
from vllm.model_executor.models.yoco_config import _maybe_build_yoco_quant_config


def quant_config(name, ignore=(), overrides=None):
    args = resolve_quantization_config(
        name, {} if name == "mxfp8" and overrides is None else overrides
    )
    if args is None:
        return None
    args = deepcopy(args)
    args.ignore.extend(ignore)
    return _maybe_build_yoco_quant_config(OnlineQuantizationConfig(args))


@pytest.mark.parametrize(
    "name,ignore,merge,quantized",
    [
        (None, (), True, False),
        ("fp8_per_block", (), True, True),
        ("fp8_per_tensor", (), False, False),
        ("mxfp8", (), False, False),
        ("fp8_per_block", ("model.yoco_k_proj",), False, False),
        ("fp8_per_block", ("model.yoco_v_proj",), False, False),
        ("fp8_per_block", ("re:model.yoco_[kv]_proj$",), True, False),
    ],
)
def test_shared_kv_fusion_preserves_component_precision(name, ignore, merge, quantized):
    config = quant_config(name, ignore)
    result, selected = yoco_fast_linear_fusion(
        "fast",
        1,
        config,
        ("model.yoco_k_proj", "model.yoco_v_proj"),
        3072,
        (512, 512),
        allow_fp8=True,
    )
    assert result == merge
    assert selected is (config if quantized else None)


@pytest.mark.parametrize(
    "mode,tp,width", [("align", 1, 512), ("fast", 2, 512), ("fast", 1, 192)]
)
def test_fp8_fusion_rejects_incompatible_boundaries(mode, tp, width):
    assert not yoco_fast_linear_fusion(
        mode,
        tp,
        quant_config("fp8_per_block"),
        ("model.yoco_k_proj", "model.yoco_v_proj"),
        3072,
        (width, width),
        allow_fp8=True,
    )[0]


@pytest.mark.parametrize("projection", ["q_proj", "qkv_proj"])
@pytest.mark.parametrize("moe_only", [False, True])
def test_bf16_projections_reuse_fusion_in_quantized_model(projection, moe_only):
    config = quant_config(
        "online" if moe_only else "fp8_per_block",
        overrides={"moe": {"weight": "fp8_per_block_static"}} if moe_only else None,
    )
    args = (
        "fast",
        1,
        config,
        (f"model.layers.0.self_attn.{projection}",),
        3072,
        (512, 64),
    )
    # Lambda stays BF16; quantized Q/QKV must not be merged into its GEMM.
    assert yoco_fast_linear_fusion(*args) == (moe_only, None)
    if not moe_only:
        config.ignored_layers.append("re:.*\\.self_attn\\.[qkv]_proj$")
        assert yoco_fast_linear_fusion(*args) == (True, None)


def test_cache_refresh_preserves_storage_and_rejects_graph_incompatible_change():
    cache = refresh_yoco_weight_cache(None, torch.arange(12).reshape(3, 4).float())
    address = cache.data_ptr()
    new = torch.ones_like(cache)
    assert refresh_yoco_weight_cache(cache, new) is cache
    assert cache.data_ptr() == address
    assert torch.equal(cache, new)
    for incompatible in (new.double(), new[:2], new.to("meta")):
        with pytest.raises(ValueError, match="Incompatible YOCO weight cache"):
            refresh_yoco_weight_cache(cache, incompatible)
    assert torch.equal(cache, new)


def _yoco_moe_router_stub(normalized: bool):
    from vllm.model_executor.layers.yoco_moe import YOCOMoE

    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.execution_mode = "fast"
    module.router_weights_normalized = normalized
    module.bf16_router = True
    module.register_buffer("_normalized_gate_weight", None, persistent=False)
    module.register_buffer("_bf16_gate_weight", None, persistent=False)
    module.gate = torch.nn.Linear(8, 4, bias=False)
    return module


@pytest.mark.parametrize("refresh_in_inference_mode", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_derived_caches_created_in_inference_mode_refresh_in_place(
    normalized, refresh_in_inference_mode
):
    """Dummy load + RL refit: the caches are first built by the profile run under
    inference_mode, then refreshed by load_weights outside it (S2 run4 error)."""
    from vllm.model_executor.layers.yoco_fast import refresh_yoco_module_cache

    router = _yoco_moe_router_stub(normalized)
    shared = torch.nn.Module()
    shared.register_buffer("_fast_down_weight_t", None, persistent=False)
    shared.down_proj = torch.nn.Linear(4, 6, bias=False)

    def refresh_all():
        router.initialize_router_weight_cache()
        # Same call as YOCOSharedExperts.initialize_fast_weight_cache (CUDA-only).
        shared._fast_down_weight_t = refresh_yoco_module_cache(
            shared, "_fast_down_weight_t", shared.down_proj.weight.t().contiguous()
        )

    with torch.inference_mode():
        refresh_all()
    caches = {
        "_bf16_gate_weight": router._bf16_gate_weight,
        "_fast_down_weight_t": shared._fast_down_weight_t,
    }
    if not normalized:
        caches["_normalized_gate_weight"] = router._normalized_gate_weight
    else:
        assert router._normalized_gate_weight is None
    assert all(cache.is_inference() for cache in caches.values())
    addresses = {name: cache.data_ptr() for name, cache in caches.items()}

    with torch.no_grad():
        router.gate.weight.copy_(torch.randn_like(router.gate.weight))
        shared.down_proj.weight.copy_(torch.randn_like(shared.down_proj.weight))
    if refresh_in_inference_mode:
        with torch.inference_mode():
            refresh_all()
    else:
        refresh_all()

    current = {
        "_bf16_gate_weight": router._bf16_gate_weight,
        "_fast_down_weight_t": shared._fast_down_weight_t,
        "_normalized_gate_weight": router._normalized_gate_weight,
    }
    for name, cache in caches.items():
        assert current[name] is cache
        assert cache.data_ptr() == addresses[name]
    gate = router.gate.weight.detach()
    if not normalized:
        gate = gate / gate.norm(dim=1, keepdim=True).clamp_min(1e-6)
        assert torch.equal(caches["_normalized_gate_weight"], gate)
    assert torch.equal(caches["_bf16_gate_weight"], gate.to(torch.bfloat16))
    assert torch.equal(
        caches["_fast_down_weight_t"], shared.down_proj.weight.detach().t()
    )


@pytest.mark.parametrize("reload_stage", ["none", "before_cache", "after_cache"])
def test_router_cache_refresh_uses_the_shared_storage_contract(reload_stage):
    from vllm.model_executor.layers.yoco_moe import YOCOMoE
    from vllm.model_executor.model_loader.reload.layerwise import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )

    module = YOCOMoE.__new__(YOCOMoE)
    torch.nn.Module.__init__(module)
    module.execution_mode = "fast"
    module.router_weights_normalized = False
    module.register_buffer("_normalized_gate_weight", None, persistent=False)
    module.gate = torch.nn.Linear(8, 4, bias=False)
    if reload_stage == "before_cache":
        record_metadata_for_reloading(module)
    module.initialize_router_weight_cache()
    cached = module._normalized_gate_weight
    replacement = torch.randn_like(module.gate.weight)
    if reload_stage != "none":
        if reload_stage == "after_cache":
            record_metadata_for_reloading(module)
        initialize_layerwise_reload(module)
        cached_meta = getattr(module, "_normalized_gate_weight", None)
        assert cached_meta is None or cached_meta.is_meta
        module.gate.weight.weight_loader(module.gate.weight, replacement)
    else:
        with torch.no_grad():
            module.gate.weight.copy_(replacement)
    module.initialize_router_weight_cache()
    if reload_stage != "none":
        finalize_layerwise_reload(module, SimpleNamespace(dtype=torch.float32))
    assert module._normalized_gate_weight is cached
    assert module._buffers["_normalized_gate_weight"] is cached
    expected = torch.nn.functional.normalize(module.gate.weight, dim=1)
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
