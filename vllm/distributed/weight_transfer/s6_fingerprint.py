# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked tensor fingerprints shared by the trainer and the vLLM workers (S6 I2).

Pure torch, no other vLLM import, so a trainer in the same environment imports this
exact module and both sides run one implementation (decision D113).

Format ``s6.wfp.v1``::

    {
        "format": "s6.wfp.v1",
        "chunk_elems": C,
        "tensors": {
            name: {
                "shape": [...],
                "dtype": "bfloat16",
                "numel": n,
                "chunks": [[sum, l1, l2sq, isum], ...],
                "sha256": "<hex>",
            }
        },
    }  # sha256 optional

Chunk ``i`` covers the global logical (row-major) elements ``[i*C, min((i+1)*C, n))``.
Its elements are promoted to one contiguous float64 vector ``v`` (the only large
temporary, ``8*C`` bytes, filled straight from strided views; 256 MiB at the default
C = 32M), and

* ``sum = sum(v)``, ``l1 = sum(|v|)``, ``l2sq = sum(v*v)``,
* ``isum = sum((i*C + k) * v[k])`` -- weighted by the global element index, so a
  permutation inside a chunk (e.g. two swapped experts) changes it, unlike the first
  three moments (S2 used only those).

``isum`` is computed without an index vector: ``v`` is viewed as ``[R, W]``
(``W = ISUM_WIDTH``) plus a tail, and ``sum(k*v_k) = W*sum_r r*rowsum_r +
sum_c c*colsum_c``.

Because the chunk boundaries are global element ranges, the float64 vector a chunk
reduces is the same whatever the memory layout of the source tensor (contiguous,
transposed, strided view): the result depends only on the logical values, the chunk
size and the device's reduction kernels. All reductions are fixed-shape torch
reductions without atomics: bitwise reproducible for a fixed device and thread
count. Different reduction groupings (e.g. CPU thread counts, GPU models) change the
float64 results only at the ~1e-16 relative level, hence ``DEFAULT_RTOL = 1e-12``:
one BF16 ulp in one typical element of a full 32M-element chunk still moves
``l1``/``l2sq`` by ~1e-10 relative, but a single FP32 ulp (or a change of a tiny
element) in a large chunk can stay below it -- use ``sha256`` for a bitwise verdict.
``sha256`` hashes the raw bytes of the tensor in logical row-major order (i.e. the
bytes of ``tensor.contiguous()``), streamed chunk by chunk.

Non-finite moments are encoded as the strings ``"nan"``, ``"inf"``, ``"-inf"`` so every
result is strict JSON; ``compare_wfp`` treats them as mismatches.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from typing import Any

import torch

WFP_FORMAT = "s6.wfp.v1"
DEFAULT_CHUNK_ELEMS = 32 * 1024 * 1024
DEFAULT_RTOL = 1e-12
ISUM_WIDTH = 4096
MAX_REPORTED_BAD_CHUNKS = 16


class TensorUnavailable(ValueError):
    """The tensor has no readable data (meta device or released storage)."""


def json_float(value: float) -> float | str:
    """A float, or ``"nan"``/``"inf"``/``"-inf"`` so the value is strict JSON."""
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def check_readable(tensor: torch.Tensor) -> None:
    """Raise ``TensorUnavailable`` for meta tensors and tensors without storage."""
    if tensor.is_meta or tensor.device.type == "meta":
        raise TensorUnavailable("meta")
    if tensor.numel() and tensor.untyped_storage().data_ptr() == 0:
        raise TensorUnavailable("released_storage")


def logical_blocks(
    tensor: torch.Tensor, start: int, stop: int
) -> Iterator[torch.Tensor]:
    """Views (no copies) whose row-major concatenation is elements ``[start, stop)``.

    One 1-D view when the tensor is viewable as 1-D; otherwise row blocks along
    dim 0, recursing into a partial first/last row.
    """
    if start >= stop:
        return
    if tensor.dim() == 0:
        yield tensor.reshape(1)[start:stop]
        return
    try:
        flat = tensor.view(-1)
    except RuntimeError:
        flat = None
    if flat is not None:
        yield flat[start:stop]
        return
    row = tensor[0].numel()
    first = start // row
    if start % row:
        end = min(stop, (first + 1) * row)
        yield from logical_blocks(tensor[first], start - first * row, end - first * row)
        start = end
    full_stop = stop // row
    if start < stop and full_stop > start // row:
        yield tensor[start // row : full_stop]
        start = full_stop * row
    if start < stop:
        yield from logical_blocks(tensor[full_stop], 0, stop - full_stop * row)


def iter_logical_chunks(
    tensor: torch.Tensor, chunk_elems: int = DEFAULT_CHUNK_ELEMS
) -> Iterator[tuple[int, int, list[torch.Tensor]]]:
    """``(start, length, views)`` for every chunk of ``chunk_elems`` elements."""
    if chunk_elems <= 0:
        raise ValueError(f"chunk_elems must be positive, got {chunk_elems}")
    numel = tensor.numel()
    for start in range(0, numel, chunk_elems):
        stop = min(start + chunk_elems, numel)
        yield start, stop - start, list(logical_blocks(tensor, start, stop))


def _gather_float64(blocks: list[torch.Tensor], length: int) -> torch.Tensor:
    """The chunk as one contiguous float64 vector: the only large temporary."""
    values = torch.empty(length, dtype=torch.float64, device=blocks[0].device)
    offset = 0
    for block in blocks:
        n = block.numel()
        values[offset : offset + n].view(block.shape).copy_(block)
        offset += n
    return values


def _chunk_moments(values: torch.Tensor, start: int) -> torch.Tensor:
    """float64 ``[sum, l1, l2sq, isum]`` of one chunk (4-element tensor)."""
    n = values.numel()
    device = values.device
    total = values.sum()
    l1 = torch.linalg.vector_norm(values, 1)
    l2sq = torch.dot(values, values)
    rows = n // ISUM_WIDTH
    body = rows * ISUM_WIDTH
    local = torch.zeros((), dtype=torch.float64, device=device)
    if rows:
        grid = values[:body].view(rows, ISUM_WIDTH)
        row_index = torch.arange(rows, dtype=torch.float64, device=device)
        col_index = torch.arange(ISUM_WIDTH, dtype=torch.float64, device=device)
        local = ISUM_WIDTH * torch.dot(row_index, grid.sum(1)) + torch.dot(
            col_index, grid.sum(0)
        )
    if body < n:
        tail = values[body:]
        tail_index = torch.arange(n - body, dtype=torch.float64, device=device)
        local = local + body * tail.sum() + torch.dot(tail_index, tail)
    isum = float(start) * total + local
    return torch.stack([total, l1, l2sq, isum])


@torch.no_grad()
def chunk_fingerprints(
    tensor: torch.Tensor, chunk_elems: int = DEFAULT_CHUNK_ELEMS
) -> list[list[float | str]]:
    """``[[sum, l1, l2sq, isum], ...]`` per chunk; ``[]`` for an empty tensor."""
    tensor = tensor.detach()
    check_readable(tensor)
    rows = []
    for start, length, blocks in iter_logical_chunks(tensor, chunk_elems):
        values = _gather_float64(blocks, length)
        rows.append(_chunk_moments(values, start))
        del values
    if not rows:
        return []
    values = torch.stack(rows).cpu().tolist()
    return [[json_float(x) for x in row] for row in values]


@torch.no_grad()
def tensor_sha256(tensor: torch.Tensor, chunk_elems: int = DEFAULT_CHUNK_ELEMS) -> str:
    """sha256 of the tensor's bytes in logical row-major order, streamed per chunk."""
    tensor = tensor.detach()
    check_readable(tensor)
    digest = hashlib.sha256()
    for _, _, blocks in iter_logical_chunks(tensor, chunk_elems):
        for block in blocks:
            raw = torch.empty(block.numel(), dtype=block.dtype)
            raw.view(block.shape).copy_(block)
            digest.update(raw.view(torch.uint8).numpy())
    return digest.hexdigest()


def tensor_wfp(
    tensor: torch.Tensor, chunk_elems: int = DEFAULT_CHUNK_ELEMS, sha256: bool = False
) -> dict[str, Any]:
    """One ``s6.wfp.v1`` tensor entry; ``TensorUnavailable`` for meta/no storage."""
    entry: dict[str, Any] = {
        "shape": [int(s) for s in tensor.shape],
        "dtype": dtype_name(tensor.dtype),
        "numel": int(tensor.numel()),
        "chunks": chunk_fingerprints(tensor, chunk_elems),
    }
    if sha256:
        entry["sha256"] = tensor_sha256(tensor, chunk_elems)
    return entry


def make_wfp(
    tensors: Mapping[str, torch.Tensor],
    chunk_elems: int = DEFAULT_CHUNK_ELEMS,
    sha256: bool = False,
) -> dict[str, Any]:
    """``s6.wfp.v1`` document for ``{name: tensor}`` (unreadable -> ``unsupported``)."""
    doc: dict[str, Any] = {
        "format": WFP_FORMAT,
        "chunk_elems": int(chunk_elems),
        "tensors": {},
        "missing": [],
        "unsupported": {},
    }
    for name, tensor in tensors.items():
        try:
            doc["tensors"][name] = tensor_wfp(tensor, chunk_elems, sha256)
        except TensorUnavailable as exc:
            doc["unsupported"][name] = str(exc)
    return doc


def _chunk_scale(
    component: int, ref_chunk: list, chunk_index: int, chunk_elems: int, length: int
) -> float:
    """Magnitude used for the relative tolerance of one moment."""
    l1 = abs(float(ref_chunk[1]))
    if component in (0, 1):
        return l1
    if component == 2:
        return abs(float(ref_chunk[2]))
    return (chunk_index * chunk_elems + length) * l1  # bound on sum |index * x|


def _compare_entry(
    local: Mapping[str, Any], ref: Mapping[str, Any], chunk_elems: int, rtol: float
) -> dict[str, Any] | None:
    """``None`` when equal, else a mismatch record."""
    reasons = []
    if list(local.get("shape") or []) != list(ref.get("shape") or []):
        reasons.append("shape")
    for key in ("dtype", "numel"):
        if local.get(key) != ref.get(key):
            reasons.append(key)
    local_chunks, ref_chunks = local.get("chunks") or [], ref.get("chunks") or []
    bad, non_finite, max_rel = [], False, 0.0
    if len(local_chunks) != len(ref_chunks):
        reasons.append("chunk_count")
    else:
        numel = int(ref.get("numel") or 0)
        for index, (a_row, b_row) in enumerate(zip(local_chunks, ref_chunks)):
            length = (
                min(chunk_elems, numel - index * chunk_elems) if numel else chunk_elems
            )
            row_bad = len(a_row) != 4 or len(b_row) != 4
            for component, (a, b) in enumerate(zip(a_row, b_row)):
                if isinstance(a, str) or isinstance(b, str):
                    non_finite = True
                    row_bad = True
                    continue
                diff = abs(float(a) - float(b))
                if diff == 0.0:
                    continue
                scale = _chunk_scale(component, b_row, index, chunk_elems, length)
                rel = diff / scale if scale else math.inf
                max_rel = max(max_rel, rel)
                if diff > rtol * scale:
                    row_bad = True
            if row_bad:
                bad.append(index)
        if bad:
            reasons.append("chunks")
        if non_finite:
            reasons.append("non_finite")
    sha_local, sha_ref = local.get("sha256"), ref.get("sha256")
    if sha_local is not None and sha_ref is not None and sha_local != sha_ref:
        reasons.append("sha256")
    if not reasons:
        return None
    return {
        "reason": reasons,
        "n_bad_chunks": len(bad),
        "bad_chunks": bad[:MAX_REPORTED_BAD_CHUNKS],
        "max_rel_err": json_float(max_rel),
        "local": {k: local.get(k) for k in ("shape", "dtype", "numel")},
        "ref": {k: ref.get(k) for k in ("shape", "dtype", "numel")},
    }


def compare_wfp(
    local: Mapping[str, Any],
    ref: Mapping[str, Any],
    rtol: float = DEFAULT_RTOL,
    strict_extra: bool = True,
) -> dict[str, Any]:
    """Compare two ``s6.wfp.v1`` documents.

    A moment may differ by ``rtol * scale`` (scale: L1 for sum and L1, L2^2 for L2^2,
    ``(end index) * L1`` for isum); ``rtol = 0`` demands bitwise-equal moments.
    ``sha256`` is compared when both sides carry it. Names only in ``local`` are
    ``extra``; they fail the check when ``strict_extra``. Nothing checked is a failure.

    Returns ``{"ok", "checked", "sha256_checked", "mismatched", "missing", "extra",
    "unsupported", "error"}`` (plain JSON).
    """
    result: dict[str, Any] = {
        "ok": False,
        "checked": 0,
        "sha256_checked": 0,
        "mismatched": [],
        "missing": [],
        "extra": [],
        "unsupported": {},
        "error": None,
    }
    for side, doc in (("local", local), ("ref", ref)):
        if doc.get("format") != WFP_FORMAT:
            result["error"] = f"{side} format {doc.get('format')!r} != {WFP_FORMAT!r}"
        elif doc.get("error"):
            result["error"] = f"{side}: {doc['error']}"
    if result["error"] is None and local.get("chunk_elems") != ref.get("chunk_elems"):
        result["error"] = (
            f"chunk_elems differ: local {local.get('chunk_elems')}"
            f" ref {ref.get('chunk_elems')}"
        )
    if result["error"] is not None:
        return result
    chunk_elems = int(ref["chunk_elems"])
    local_tensors = local.get("tensors") or {}
    local_unsupported = local.get("unsupported") or {}
    ref_tensors = ref.get("tensors") or {}
    for name in sorted(ref_tensors):
        if name not in local_tensors:
            if name in local_unsupported:
                result["unsupported"][name] = local_unsupported[name]
            else:
                result["missing"].append(name)
            continue
        result["checked"] += 1
        if (
            local_tensors[name].get("sha256") is not None
            and ref_tensors[name].get("sha256") is not None
        ):
            result["sha256_checked"] += 1
        mismatch = _compare_entry(
            local_tensors[name], ref_tensors[name], chunk_elems, rtol
        )
        if mismatch is not None:
            result["mismatched"].append({"name": name, **mismatch})
    result["extra"] = sorted(set(local_tensors) - set(ref_tensors))
    result["ok"] = bool(
        result["checked"]
        and not result["mismatched"]
        and not result["missing"]
        and not result["unsupported"]
        and not (strict_extra and result["extra"])
    )
    return result
