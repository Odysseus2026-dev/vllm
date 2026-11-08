# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only weight-sync checks for RL refits (S6 I2), as a worker extension mixin.

``S6WeightCheckExtension`` adds RPC methods that the driver calls by string name
through ``collective_rpc``; every argument and return value is plain JSON data
(no serialized callables, no insecure serialization). It defines no ``__new__`` /
``__init__`` and does not depend on any RL framework, so a framework combines it
with its own extension: ``class X(S6WeightCheckExtension, FrameworkExtension)``
(vLLM accepts one ``worker_extension_cls``).

Typical check step::

    s6_arm(chunk_elems=..., record="both")   # wraps model.load_weights (instance)
    <framework refit: model.load_weights(bucket) per bucket>
    s6_compare(ref, source="received")       # bytes delivered to load_weights
    s6_compare(ref, source="written")        # parameter regions right after loading
    s6_load_report()                         # coverage incl. MoE aliases (D51)
    s6_disarm()                              # removes the wrapper

Steps that are not checked never call ``s6_arm``, so nothing is installed. Nothing
here changes inference numerics: the tap only reads the tensors passed through
``load_weights`` and the parameter regions afterwards.

Sources of ``s6_checksum`` / ``s6_compare`` (all keyed by checkpoint (HF) names, in
the load layout; fingerprints from ``s6_fingerprint``):

* ``received``: each tensor as passed into ``load_weights``;
* ``written``: the parameter region the tensor was loaded into, read right after
  the ``load_weights`` call that loaded it (before ``process_weights_after_loading``
  may repack it). Fused projections are located through the layer's shard offsets;
  YOCO names go through YOCO's own loader mapping. Needs the checkpoint layout of an
  unsharded parameter (TP=1, no EP); anything else is reported ``unsupported``;
* ``params``: the same read-back from the live parameters at call time; valid only
  when the live layout is still the load layout.
"""

from __future__ import annotations

import json
import math
import os
import socket
from collections.abc import Iterable, Mapping
from typing import Any

import torch

from vllm.distributed.weight_transfer.s6_fingerprint import (
    DEFAULT_CHUNK_ELEMS,
    DEFAULT_RTOL,
    WFP_FORMAT,
    TensorUnavailable,
    check_readable,
    compare_wfp,
    dtype_name,
    tensor_wfp,
)

SNAPSHOT_FORMAT = "s6.snapshot.v1"
LOAD_REPORT_FORMAT = "s6.load_report.v1"
SOURCES = ("received", "written", "params")
RECORD_MODES = {
    "received": ("received",),
    "written": ("written",),
    "both": ("received", "written"),
}
MAX_ALIAS_EXAMPLES = 40
MAX_ERRORS = 50
_MISSING = object()
_YOCO_MODULE = "vllm.model_executor.models.yoco"


class Unsupported(Exception):
    """A checkpoint tensor has no readable load-layout region in this worker."""


def _checked_json(result: dict[str, Any]) -> dict[str, Any]:
    json.dumps(result, allow_nan=False)
    return result


# --------------------------------------------------------------------------------------
# Parameter identity and coverage (D51)
# --------------------------------------------------------------------------------------


def storage_key(tensor: torch.Tensor | None) -> tuple | None:
    """Identity of the memory a tensor views; ``None`` without real data.

    Empty, meta and released-storage (pointer 0) tensors return ``None`` so that
    distinct such tensors are not mistaken for aliases (review D51 M1).
    """
    if tensor is None or tensor.numel() == 0 or tensor.is_meta:
        return None
    ptr = tensor.untyped_storage().data_ptr()
    if ptr == 0:
        return None
    return (
        tensor.device.type,
        tensor.device.index,
        ptr,
        tensor.storage_offset(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
    )


def alias_groups(named_params: list[tuple[str, torch.Tensor]]) -> list[list[int]]:
    """Indices of ``named_params`` grouped by Parameter object or identical memory."""
    parent = list(range(len(named_params)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_by_key: dict[tuple, int] = {}
    for index, (_, tensor) in enumerate(named_params):
        if tensor is None:
            continue
        for key in (("id", id(tensor)), ("storage", storage_key(tensor))):
            if key[1] is None:
                continue
            if key in first_by_key:
                parent[find(index)] = find(first_by_key[key])
            else:
                first_by_key[key] = index
    groups: dict[int, list[int]] = {}
    for index in range(len(named_params)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def param_coverage(
    named_params: list[tuple[str, torch.Tensor]], loaded_names: Iterable[str]
) -> dict[str, Any]:
    """Coverage of vLLM parameters by tensor identity (D51; S2 M2 field names).

    A group (one Parameter registered under several names, e.g. YOCO MoE
    ``transform``/``shared_experts`` re-registering ``mlp.experts`` modules) is
    written when any of its names was loaded; its canonical name is the loaded one
    (else the first registered) and every other name is an alias.
    """
    loaded_names = set(loaded_names)
    groups = alias_groups(named_params)
    not_written, alias_pairs = [], []
    aliases = same_object = storage_only = to_unwritten = written = 0
    for members in groups:
        loaded = [i for i in members if named_params[i][0] in loaded_names]
        canonical = loaded[0] if loaded else members[0]
        if loaded:
            written += 1
        else:
            not_written.append(named_params[canonical][0])
        for i in members:
            if i == canonical:
                continue
            aliases += 1
            is_same = named_params[i][1] is named_params[canonical][1]
            same_object += is_same
            storage_only += not is_same
            to_unwritten += not loaded
            if len(alias_pairs) < MAX_ALIAS_EXAMPLES:
                alias_pairs.append(
                    [named_params[i][0], named_params[canonical][0], bool(is_same)]
                )
    all_names = {name for name, _ in named_params}
    return {
        "vllm_param_names_total": len(named_params),
        "vllm_params_canonical": len(groups),
        "vllm_params_written": written,
        "vllm_param_aliases": aliases,
        "vllm_param_aliases_same_object": same_object,
        "vllm_param_aliases_same_storage_only": storage_only,
        "vllm_param_aliases_all_point_to_written": to_unwritten == 0,
        "vllm_param_alias_examples": alias_pairs,
        "vllm_params_not_loaded": sorted(not_written),
        "loaded_names_not_registered": sorted(loaded_names - all_names),
    }


# --------------------------------------------------------------------------------------
# Checkpoint name -> load-layout parameter region
# --------------------------------------------------------------------------------------


def _is_yoco(model: torch.nn.Module) -> bool:
    return any(cls.__module__ == _YOCO_MODULE for cls in type(model).__mro__)


def _has_base(obj: Any, name: str) -> bool:
    return any(cls.__name__ == name for cls in type(obj).__mro__)


class HFParamMapper:
    """Maps a checkpoint tensor name to the parameter region it is loaded into."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.params = dict(model.named_parameters(remove_duplicate=False))
        self._loader = None
        self._kv_aliases: Mapping[str, str] = {}
        if _is_yoco(model):
            from vllm.model_executor.models import yoco_weights

            self._loader = yoco_weights._YocoWeightLoader(model)
            self.params = self._loader.params
            self._kv_aliases = yoco_weights._SHARED_KV_ALIASES

    def target(self, name: str) -> tuple[str, int | str | None]:
        """``(vLLM parameter name, shard id or None)`` used by ``load_weights``."""
        if self._loader is None:
            return name, None
        name = self._kv_aliases.get(name, name)
        projection = self._loader.projection_target(name)
        if projection is not None:
            return projection
        return self._loader.aliases.get(name, name), None

    def _layer(self, param_name: str) -> torch.nn.Module:
        return self.model.get_submodule(param_name.rpartition(".")[0])

    def _shard(
        self, param_name: str, param: torch.Tensor, shard: int | str
    ) -> torch.Tensor:
        layer = self._layer(param_name)
        dim = getattr(param, "output_dim", 0)
        if isinstance(shard, str):
            offset = layer._get_shard_offset_mapping(shard)
            size = layer._get_shard_size_mapping(shard)
        else:
            sizes = getattr(layer, "output_sizes", None)
            if sizes is None:
                raise Unsupported(f"no_output_sizes:{param_name}")
            tp_size = getattr(layer, "tp_size", 1)
            offset = sum(sizes[:shard]) // tp_size
            size = sizes[shard] // tp_size
        if offset is None or size is None:
            raise Unsupported(f"unknown_shard:{param_name}:{shard}")
        return param.detach().narrow(dim, offset, size)

    def region(self, name: str, numel: int | None = None) -> torch.Tensor:
        """The load-layout region of checkpoint tensor ``name`` (logical order).

        Raises:
            Unsupported: no such parameter, unknown shard layout, or a region whose
                size differs from ``numel`` (sharded or repacked parameter).
        """
        param_name, shard = self.target(name)
        param = self.params.get(param_name)
        if param is None:
            raise Unsupported(f"no_param:{param_name}")
        try:
            check_readable(param)
        except TensorUnavailable as exc:
            raise Unsupported(f"{exc}:{param_name}") from None
        region = (
            param.detach() if shard is None else self._shard(param_name, param, shard)
        )
        if numel is None or region.numel() == numel:
            return region
        layer = self._layer(param_name)
        row = region[0].numel() if region.dim() and region.shape[0] else 0
        if (
            _has_base(layer, "VocabParallelEmbedding")
            and getattr(layer, "tp_size", 1) == 1
            and row
            and numel < region.numel()
            and numel % row == 0
        ):
            return region.narrow(0, 0, numel // row)  # padded vocabulary rows
        raise Unsupported(
            f"layout:{param_name}{list(region.shape)} has {region.numel()} elements,"
            f" checkpoint tensor has {numel}"
        )


# --------------------------------------------------------------------------------------
# Load tap state
# --------------------------------------------------------------------------------------


class _S6State:
    """Per-worker check state between ``s6_arm`` and ``s6_disarm``."""

    def __init__(self, chunk_elems: int, record: str, sha256: bool) -> None:
        self.chunk_elems = chunk_elems
        self.record = record
        self.sha256 = sha256
        self.active = True
        self.model: torch.nn.Module | None = None
        self.previous_attr: Any = _MISSING
        self.tap: Any = None
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.received_counts: dict[str, int] = {}
        self.received_shapes: dict[str, list[int]] = {}
        self.received_total = 0
        self.loaded: set[str] = set()
        self.loaded_reported = True
        self.ignored: list[str] = []
        self.pipeline_missing: list[str] = []
        self.fallbacks: list[str] = []
        self.wfp: dict[str, dict[str, dict]] = {"received": {}, "written": {}}
        self.unsupported: dict[str, dict[str, str]] = {"received": {}, "written": {}}
        self.errors: list[str] = []

    def _error(self, message: str) -> None:
        if len(self.errors) < MAX_ERRORS:
            self.errors.append(message)

    def _digest(
        self, source: str, name: str, tensor: torch.Tensor, shape: list
    ) -> None:
        try:
            entry = tensor_wfp(tensor, self.chunk_elems, self.sha256)
        except (TensorUnavailable, Unsupported) as exc:
            self.unsupported[source][name] = str(exc)
            return
        except Exception as exc:  # never break the refit because of a check
            self.unsupported[source][name] = f"error:{type(exc).__name__}"
            self._error(f"{source} {name}: {type(exc).__name__}: {exc}")
            return
        entry["shape"] = list(shape)
        self.wfp[source][name] = entry
        self.unsupported[source].pop(name, None)

    def note_received(self, name: str, tensor: torch.Tensor) -> None:
        self.received_counts[name] = self.received_counts.get(name, 0) + 1
        self.received_total += 1
        shape = [int(s) for s in tensor.shape]
        self.received_shapes[name] = shape
        if "received" in RECORD_MODES[self.record]:
            self._digest("received", name, tensor, shape)

    def note_written(
        self, model: torch.nn.Module, names: list[tuple[str, int]]
    ) -> None:
        if "written" not in RECORD_MODES[self.record] or not names:
            return
        # Built per call: the framework may swap parameters (e.g. checkpoint-layout
        # staging of MoE weights) between arming and loading.
        try:
            mapper = HFParamMapper(model)
        except Exception as exc:
            self._error(f"mapper: {type(exc).__name__}: {exc}")
            for name, _ in names:
                self.unsupported["written"][name] = "error:mapper"
            return
        for name, numel in names:
            try:
                region = mapper.region(name, numel)
            except Unsupported as exc:
                self.unsupported["written"][name] = str(exc)
                continue
            self._digest("written", name, region, self.received_shapes[name])

    def note_call(
        self, model: torch.nn.Module, result: Any, previous_report: Any
    ) -> None:
        self.calls += 1
        if isinstance(result, (set, frozenset, list, tuple)):
            self.loaded.update(n for n in result if isinstance(n, str))
        else:
            self.loaded_reported = False
        report = getattr(model, "_yoco_weight_load_report", None)
        if report is not None and report is not previous_report:
            self.loaded.update(report.loaded)
            self.ignored.extend(report.ignored)
            self.pipeline_missing.extend(report.pipeline_missing)
            self.fallbacks.extend(report.default_loader_fallbacks)


def _install_tap(model: torch.nn.Module, state: _S6State) -> None:
    original = model.load_weights

    def load_weights(weights):
        if not state.active:
            return original(weights)
        loaded_now: list[tuple[str, int]] = []

        def tap():
            for name, tensor in weights:
                state.note_received(name, tensor)
                loaded_now.append((name, tensor.numel()))
                yield name, tensor

        previous_report = getattr(model, "_yoco_weight_load_report", None)
        result = original(tap())
        state.note_call(model, result, previous_report)
        with torch.no_grad():
            state.note_written(model, loaded_now)
        return result

    state.model = model
    state.previous_attr = model.__dict__.get("load_weights", _MISSING)
    state.tap = load_weights
    model.__dict__["load_weights"] = load_weights


def _remove_tap(state: _S6State) -> bool:
    """Deactivate the tap and restore ``load_weights``; False if re-wrapped since."""
    state.active = False
    model = state.model
    if model is None or model.__dict__.get("load_weights") is not state.tap:
        return False
    if state.previous_attr is _MISSING:
        del model.__dict__["load_weights"]
    else:
        model.__dict__["load_weights"] = state.previous_attr
    return True


# --------------------------------------------------------------------------------------
# RPC bodies (``worker`` is the vLLM worker the mixin is injected into)
# --------------------------------------------------------------------------------------


def _model(worker: Any) -> torch.nn.Module:
    return worker.model_runner.get_model()  # unwraps CUDA-graph wrappers


def _state(worker: Any) -> _S6State | None:
    return getattr(worker, "_s6_state", None)


def rank_info(worker: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "rank": getattr(worker, "rank", None),
        "local_rank": getattr(worker, "local_rank", None),
        "tp_rank": None,
        "tp_size": None,
        "pp_rank": None,
        "pp_size": None,
    }
    try:
        from vllm.distributed import parallel_state

        info["tp_rank"] = parallel_state.get_tensor_model_parallel_rank()
        info["tp_size"] = parallel_state.get_tensor_model_parallel_world_size()
        info["pp_rank"] = parallel_state.get_pp_group().rank_in_group
        info["pp_size"] = parallel_state.get_pp_group().world_size
    except (AssertionError, AttributeError, RuntimeError):
        pass  # parallel groups not initialized (unit tests)
    return info


def arm(
    worker: Any,
    chunk_elems: int = DEFAULT_CHUNK_ELEMS,
    record: str = "received",
    sha256: bool = False,
) -> dict[str, Any]:
    if record not in RECORD_MODES:
        raise ValueError(
            f"record must be one of {sorted(RECORD_MODES)}, got {record!r}"
        )
    chunk_elems = int(chunk_elems)
    if chunk_elems <= 0:
        raise ValueError(f"chunk_elems must be positive, got {chunk_elems}")
    disarm(worker)
    model = _model(worker)
    state = _S6State(chunk_elems, record, bool(sha256))
    _install_tap(model, state)
    worker._s6_state = state
    return _checked_json(
        {
            "armed": True,
            "chunk_elems": chunk_elems,
            "record": record,
            "sha256": bool(sha256),
            "model_class": type(model).__name__,
            "rank": rank_info(worker),
        }
    )


def disarm(worker: Any) -> dict[str, Any]:
    state = _state(worker)
    restored = None
    if state is not None:
        restored = _remove_tap(state)
        worker._s6_state = None
    return _checked_json(
        {"armed": False, "was_armed": state is not None, "restored": restored}
    )


def _empty_wfp(chunk_elems: int, source: str, worker: Any) -> dict[str, Any]:
    return {
        "format": WFP_FORMAT,
        "chunk_elems": chunk_elems,
        "source": source,
        "tensors": {},
        "missing": [],
        "unsupported": {},
        "error": None,
        "rank": rank_info(worker),
    }


def checksum(
    worker: Any,
    names: list[str] | None = None,
    chunk_elems: int | None = None,
    source: str = "received",
    shapes: Mapping[str, list[int]] | None = None,
    sha256: bool | None = None,
) -> dict[str, Any]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    state = _state(worker)
    if source != "params":
        doc = _empty_wfp(chunk_elems or 0, source, worker)
        if state is None:
            doc["error"] = "not_armed"
            return _checked_json(doc)
        doc["chunk_elems"] = state.chunk_elems
        if chunk_elems is not None and int(chunk_elems) != state.chunk_elems:
            doc["error"] = f"chunk_elems {chunk_elems} != armed {state.chunk_elems}"
            return _checked_json(doc)
        if source not in RECORD_MODES[state.record]:
            doc["error"] = (
                f"source {source!r} not recorded (armed with {state.record!r})"
            )
            return _checked_json(doc)
        recorded, unsupported = state.wfp[source], state.unsupported[source]
        wanted = sorted(set(recorded) | set(unsupported)) if names is None else names
        for name in wanted:
            if name in recorded:
                doc["tensors"][name] = recorded[name]
            elif name in unsupported:
                doc["unsupported"][name] = unsupported[name]
            else:
                doc["missing"].append(name)
        return _checked_json(doc)

    chunk_elems = int(chunk_elems or DEFAULT_CHUNK_ELEMS)
    doc = _empty_wfp(chunk_elems, source, worker)
    shapes = dict(shapes or {})
    if names is None:
        if state is None or not state.received_shapes:
            doc["error"] = "names required for source='params' (nothing received)"
            return _checked_json(doc)
        names = sorted(state.received_shapes)
    if state is not None:
        for name in names:
            shapes.setdefault(name, state.received_shapes.get(name))
    use_sha = bool(sha256) if sha256 is not None else bool(state and state.sha256)
    mapper = HFParamMapper(_model(worker))
    with torch.no_grad():
        for name in names:
            shape = shapes.get(name)
            numel = math.prod(shape) if shape is not None else None
            try:
                region = mapper.region(name, numel)
                entry = tensor_wfp(region, chunk_elems, use_sha)
            except (Unsupported, TensorUnavailable) as exc:
                doc["unsupported"][name] = str(exc)
                continue
            if shape is not None:
                entry["shape"] = list(shape)
            doc["tensors"][name] = entry
    return _checked_json(doc)


def compare(
    worker: Any,
    ref: Mapping[str, Any],
    source: str = "received",
    rtol: float = DEFAULT_RTOL,
    strict_extra: bool = True,
) -> dict[str, Any]:
    if not isinstance(ref, Mapping) or ref.get("format") != WFP_FORMAT:
        fmt = ref.get("format") if isinstance(ref, Mapping) else type(ref).__name__
        return _checked_json(
            {
                "ok": False,
                "checked": 0,
                "mismatched": [],
                "missing": [],
                "extra": [],
                "unsupported": {},
                "error": f"ref format {fmt!r} != {WFP_FORMAT!r}",
                "source": source,
                "rank": rank_info(worker),
            }
        )
    ref_tensors = ref.get("tensors") or {}
    chunk_elems = int(ref.get("chunk_elems") or 0)
    if source == "params":
        local = checksum(
            worker,
            sorted(ref_tensors),
            chunk_elems,
            source,
            shapes={n: e.get("shape") for n, e in ref_tensors.items()},
            sha256=any(e.get("sha256") for e in ref_tensors.values()),
        )
    else:
        local = checksum(worker, None, chunk_elems, source)
    result = compare_wfp(local, ref, float(rtol), bool(strict_extra))
    result.update(source=source, rtol=float(rtol), rank=local["rank"])
    return _checked_json(result)


def load_report(
    worker: Any, reset: bool = False, expected_names: list[str] | None = None
) -> dict[str, Any]:
    state = _state(worker)
    rank = rank_info(worker)
    if state is None:
        return _checked_json(
            {
                "format": LOAD_REPORT_FORMAT,
                "armed": False,
                "ok": False,
                "error": "not_armed",
                "rank": rank,
            }
        )
    model = _model(worker)
    named_params = list(model.named_parameters(remove_duplicate=False))
    received = set(state.received_counts)
    report: dict[str, Any] = {
        "format": LOAD_REPORT_FORMAT,
        "armed": True,
        "error": None,
        "rank": rank,
        "model_class": type(model).__name__,
        "load_weights_calls": state.calls,
        "received_names": sorted(received),
        "received_unique": len(received),
        "received_total": state.received_total,
        "received_duplicates": sorted(
            n for n, c in state.received_counts.items() if c > 1
        ),
        "loaded_names_reported": state.loaded_reported,
        **param_coverage(named_params, state.loaded),
        "ignored": sorted(set(state.ignored)),
        "pipeline_missing": sorted(set(state.pipeline_missing)),
        "default_loader_fallbacks": sorted(set(state.fallbacks)),
        "tap_errors": list(state.errors),
        "received_missing": [],
        "received_unexpected": [],
    }
    if expected_names is not None:
        expected = set(expected_names)
        report["received_missing"] = sorted(expected - received)
        report["received_unexpected"] = sorted(received - expected)
    pipeline_parallel = (rank.get("pp_size") or 1) > 1
    report["ok"] = bool(
        state.calls > 0
        and state.loaded_reported
        and not report["vllm_params_not_loaded"]
        and report["vllm_param_aliases_all_point_to_written"]
        and not report["loaded_names_not_registered"]
        and not report["ignored"]
        and not (report["pipeline_missing"] and not pipeline_parallel)
        and not report["received_missing"]
        and not report["received_unexpected"]
        and not report["tap_errors"]
    )
    if reset:
        state.reset()
    return _checked_json(report)


def snapshot(worker: Any, names: list[str] | None = None) -> dict[str, Any]:
    """Parameter metadata grouped by D51 identity; reads no parameter data."""
    model = _model(worker)
    named_params = list(model.named_parameters(remove_duplicate=False))
    params: dict[str, Any] = {}
    for members in alias_groups(named_params):
        canonical, tensor = named_params[members[0]]
        aliases = [named_params[i][0] for i in members[1:]]
        if names is not None and not ({canonical, *aliases} & set(names)):
            continue
        meta = tensor.is_meta
        params[canonical] = {
            "aliases": aliases,
            "shape": [int(s) for s in tensor.shape],
            "dtype": dtype_name(tensor.dtype),
            "numel": int(tensor.numel()),
            "device": str(tensor.device),
            "is_meta": bool(meta),
            "contiguous": bool(tensor.is_contiguous()),
            "has_storage": bool(storage_key(tensor) is not None),
        }
    registered = {name for name, _ in named_params}
    return _checked_json(
        {
            "format": SNAPSHOT_FORMAT,
            "rank": rank_info(worker),
            "model_class": type(model).__name__,
            "params": params,
            "unknown_names": sorted(set(names or ()) - registered),
        }
    )


class S6WeightCheckExtension:
    """Worker extension mixin with the S6 I2 weight-check RPCs (JSON in, JSON out)."""

    def s6_arm(
        self,
        chunk_elems: int = DEFAULT_CHUNK_ELEMS,
        record: str = "received",
        sha256: bool = False,
    ) -> dict:
        """Install the load tap for the next refit and reset its records."""
        return arm(self, chunk_elems, record, sha256)

    def s6_disarm(self) -> dict:
        """Remove the load tap and drop all records."""
        return disarm(self)

    def s6_checksum(
        self,
        names: list[str] | None = None,
        chunk_elems: int | None = None,
        source: str = "received",
    ) -> dict:
        """``s6.wfp.v1`` fingerprints by checkpoint name (see module docstring)."""
        return checksum(self, names, chunk_elems, source)

    def s6_compare(
        self,
        ref: dict,
        source: str = "received",
        rtol: float = DEFAULT_RTOL,
        strict_extra: bool = True,
    ) -> dict:
        """Compare this worker's fingerprints with an ``s6.wfp.v1`` reference."""
        return compare(self, ref, source, rtol, strict_extra)

    def s6_load_report(
        self, reset: bool = False, expected_names: list[str] | None = None
    ) -> dict:
        """Load coverage accumulated since ``s6_arm`` or the last reset."""
        return load_report(self, reset, expected_names)

    def s6_snapshot(self, names: list[str] | None = None) -> dict:
        """Parameter names, aliases, shapes and dtypes (no data reads)."""
        return snapshot(self, names)
