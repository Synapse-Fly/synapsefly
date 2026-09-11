"""Connectome data model: ``REGIONS``, ``CSR``, ``Connectome`` (SPEC section c.2).

Only numpy is imported at module level. Provenance tags: ``[V]`` verified (RESEARCH),
``[L]`` literature, ``[E]`` engineered. This module carries no biological constants except the
8-region taxonomy of RESEARCH section 6 ``[V]`` and the synthetic body-id base ``[E]``.
"""

from __future__ import annotations

import json
import re
import struct
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .csr import build_csr

__all__ = [
    "REGIONS",
    "REGION_ID",
    "SYNTHETIC_BODY_BASE",
    "NT_VALUES",
    "CSR",
    "Connectome",
    "count_regions",
    "count_groups",
    "utc_now_iso",
]

#: 8-region taxonomy, ids 0..7 in this order everywhere (arrays, JSON, raster lanes) [V] RESEARCH section 6.
REGIONS: tuple[str, ...] = (
    "optic_lobe",
    "antennal_lobe",
    "mushroom_body",
    "central_complex",
    "sez",
    "central_other",
    "descending_motor",
    "vnc",
)
#: name -> 0..7
REGION_ID: dict[str, int] = {name: i for i, name in enumerate(REGIONS)}
#: synthetic body ids are ``SYNTHETIC_BODY_BASE + i`` [E] (SPEC section 0.1).
SYNTHETIC_BODY_BASE: int = 1_000_000
#: normalised neurotransmitter strings accepted in ``Connectome.nt`` (SPEC section c.2 / c.7).
NT_VALUES: tuple[str, ...] = (
    "acetylcholine",
    "gaba",
    "glutamate",
    "histamine",
    "dopamine",
    "octopamine",
    "serotonin",
    "unknown",
)

#: meta keys every source must provide (SPEC section c.2).
META_KEYS: tuple[str, ...] = (
    "e",
    "synapses",
    "license",
    "citation",
    "seed",
    "build_args",
    "patches_applied",
    "gain_default",
    "weights_mode",
    "engineered_edges",
    "region_counts",
    "group_counts",
    "created",
)

_GROUP_NPZ_PREFIX = "group__"
_NODE_ARRAYS: tuple[str, ...] = ("sign", "region", "side", "type_idx", "body_id", "nt")


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with second precision (``meta['created']``)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def count_regions(region: np.ndarray) -> dict[str, int]:
    """``{region_name: count}`` over all 8 regions (zeros included) from a uint8 region array."""
    counts = np.bincount(np.asarray(region, dtype=np.int64), minlength=len(REGIONS))
    return {name: int(counts[i]) for i, name in enumerate(REGIONS)}


def count_groups(groups: dict[str, np.ndarray]) -> dict[str, int]:
    """``{group_name: size}`` for every key of a groups dict (sided keys included)."""
    return {name: int(np.asarray(idx).size) for name, idx in groups.items()}


# --------------------------------------------------------------------------- CSR


@dataclass(frozen=True)
class CSR:
    """Outgoing sparse view: one CSR from 20k to 166.7k neurons (SPEC section 0).

    ``indptr`` int64[n+1]; ``indices`` int32[E] postsynaptic index, sorted within each row;
    ``data`` float32[E] SIGNED synapse count = ``weight * sign[pre]`` (no w_syn, no gain);
    ``indptr32`` int32 shadow copy when E < 2**31 (used for fast gathers), else None.
    """

    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    indptr32: np.ndarray | None

    @property
    def n(self) -> int:
        """Number of rows (neurons)."""
        return int(self.indptr.shape[0]) - 1

    @property
    def e(self) -> int:
        """Number of stored edges."""
        return int(self.indices.shape[0])

    def row(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(indices, data) slice of row ``i`` (views, no copy)."""
        a, b = int(self.indptr[i]), int(self.indptr[i + 1])
        return self.indices[a:b], self.data[a:b]


# --------------------------------------------------------------------------- Connectome


@dataclass(slots=True)
class Connectome:
    """Edge list + per-neuron annotations + named groups (SPEC section c.2).

    Field dtypes (binding): ``pre``/``post`` int32[E] (``pre`` non-decreasing, row-major CSR
    order, no self-loops, no duplicate (pre, post)); ``weight`` float32[E] synapse count > 0
    (UNSIGNED); ``sign`` float32[n] +1/-1 (presynaptic NT sign); ``region`` uint8[n] index into
    ``REGIONS``; ``side`` int8[n] -1 L / +1 R / 0 unknown or midline; ``types`` unique type
    labels ("" = untyped); ``type_idx`` int32[n]; ``body_id`` int64[n]; ``nt`` object[n]
    normalised NT string; ``groups`` name -> sorted int32 indices (may overlap; sided keys
    included); ``meta`` dict with the keys listed in ``META_KEYS``.
    """

    name: str
    source: str
    n: int
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    sign: np.ndarray
    region: np.ndarray
    side: np.ndarray
    types: list[str]
    type_idx: np.ndarray
    body_id: np.ndarray
    nt: np.ndarray
    groups: dict[str, np.ndarray]
    meta: dict
    _csr_cache: CSR | None = field(default=None, init=False, repr=False, compare=False)

    # ---- simple accessors -------------------------------------------------

    @property
    def e(self) -> int:
        """Number of edges (``len(pre)``)."""
        return int(self.pre.shape[0])

    def type_of(self, i: int) -> str:
        """Type label of neuron ``i`` ("" when untyped)."""
        return self.types[int(self.type_idx[i])]

    def where(self, type_regex: str, side: int | None = None) -> np.ndarray:
        """Sorted int32 indices whose type fullmatches ``type_regex`` (``re.fullmatch``),
        optionally filtered by ``side`` (-1 / 0 / +1)."""
        pattern = re.compile(type_regex)
        type_mask = np.fromiter(
            (pattern.fullmatch(t) is not None for t in self.types), dtype=bool, count=len(self.types)
        )
        mask = type_mask[self.type_idx]
        if side is not None:
            mask &= self.side == np.int8(side)
        return np.flatnonzero(mask).astype(np.int32)

    def csr(self) -> CSR:
        """Build once (cached on the instance) from pre/post/weight/sign; ``data = weight * sign[pre]``."""
        if self._csr_cache is None:
            signed = (self.weight.astype(np.float32, copy=False) * self.sign[self.pre]).astype(np.float32)
            indptr, indices, data = build_csr(self.pre, self.post, signed, self.n, sum_duplicates=True)
            indptr32 = indptr.astype(np.int32) if indices.shape[0] < 2**31 else None
            self._csr_cache = CSR(indptr=indptr, indices=indices, data=data, indptr32=indptr32)
        return self._csr_cache

    # ---- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Assert the field invariants; raise ``ValueError`` with a precise message.

        Checks: ``len(pre)==len(post)==len(weight)``; pre sorted (non-decreasing); ``0<=pre,post<n``;
        no self loops; no duplicate (pre, post); ``weight>0``; ``sign in {+1,-1}``; ``region<8``;
        ``side in {-1,0,1}``; every ``groups[]`` array sorted, int32, within range;
        ``sum(region_counts)==n``; per-neuron arrays have length n; ``type_idx`` within ``types``.
        The order of ``post`` within a row is NOT enforced (``csr()`` sorts rows itself).
        """
        n = int(self.n)
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        e = int(self.pre.shape[0])
        if not (self.post.shape[0] == e and self.weight.shape[0] == e):
            raise ValueError(
                f"len(pre)={e}, len(post)={self.post.shape[0]}, len(weight)={self.weight.shape[0]} must be equal"
            )
        for name in _NODE_ARRAYS:
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if self.pre.dtype != np.int32 or self.post.dtype != np.int32:
            raise ValueError(f"pre/post must be int32, got {self.pre.dtype}/{self.post.dtype}")
        if self.weight.dtype != np.float32:
            raise ValueError(f"weight must be float32, got {self.weight.dtype}")
        if self.sign.dtype != np.float32:
            raise ValueError(f"sign must be float32, got {self.sign.dtype}")
        if self.region.dtype != np.uint8:
            raise ValueError(f"region must be uint8, got {self.region.dtype}")
        if self.side.dtype != np.int8:
            raise ValueError(f"side must be int8, got {self.side.dtype}")
        if self.type_idx.dtype != np.int32:
            raise ValueError(f"type_idx must be int32, got {self.type_idx.dtype}")
        if self.body_id.dtype != np.int64:
            raise ValueError(f"body_id must be int64, got {self.body_id.dtype}")
        if e:
            if self.pre.min() < 0 or self.pre.max() >= n:
                raise ValueError(f"pre out of range [0, {n})")
            if self.post.min() < 0 or self.post.max() >= n:
                raise ValueError(f"post out of range [0, {n})")
            if np.any(self.pre[1:] < self.pre[:-1]):
                bad = int(np.flatnonzero(self.pre[1:] < self.pre[:-1])[0]) + 1
                raise ValueError(f"pre must be non-decreasing (violation at edge {bad})")
            loops = np.flatnonzero(self.pre == self.post)
            if loops.size:
                raise ValueError(f"self-loops are not allowed ({loops.size} found, first at edge {int(loops[0])})")
            key = self.pre.astype(np.int64) * n + self.post.astype(np.int64)
            dup = np.flatnonzero(key[1:] == key[:-1])
            if dup.size:
                i = int(dup[0]) + 1
                raise ValueError(
                    f"duplicate (pre, post) edges ({dup.size} adjacent duplicates, first at edge {i}: "
                    f"({int(self.pre[i])}, {int(self.post[i])}))"
                )
            if not np.all(self.weight > 0):
                bad = int(np.flatnonzero(~(self.weight > 0))[0])
                raise ValueError(f"weight must be > 0 (edge {bad} has {float(self.weight[bad])})")
        if n:
            if not np.all((self.sign == 1.0) | (self.sign == -1.0)):
                raise ValueError("sign must be +1.0 or -1.0 for every neuron")
            if self.region.max() >= len(REGIONS):
                raise ValueError(f"region must be < {len(REGIONS)}, got max {int(self.region.max())}")
            if not np.all((self.side >= -1) & (self.side <= 1)):
                raise ValueError("side must be in {-1, 0, 1}")
            if self.type_idx.min() < 0 or self.type_idx.max() >= len(self.types):
                raise ValueError(f"type_idx out of range [0, {len(self.types)})")
        for gname, idx in self.groups.items():
            idx = np.asarray(idx)
            if idx.dtype != np.int32:
                raise ValueError(f"groups[{gname!r}] must be int32, got {idx.dtype}")
            if idx.ndim != 1:
                raise ValueError(f"groups[{gname!r}] must be 1-D")
            if idx.size:
                if idx.min() < 0 or idx.max() >= n:
                    raise ValueError(f"groups[{gname!r}] has indices out of range [0, {n})")
                if np.any(idx[1:] <= idx[:-1]):
                    raise ValueError(f"groups[{gname!r}] must be strictly increasing (sorted, unique)")
        rc = self.meta.get("region_counts")
        if not isinstance(rc, dict):
            raise ValueError("meta['region_counts'] missing or not a dict")
        total = int(sum(int(v) for v in rc.values()))
        if total != n:
            raise ValueError(f"sum(meta['region_counts']) = {total} != n = {n}")

    # ---- persistence -----------------------------------------------------------

    def save(self, stem: Path) -> None:
        """Write ``<stem>.npz`` (``np.savez``, UNCOMPRESSED: pre, post, weight, sign, region, side,
        type_idx, body_id, nt) and ``<stem>.json`` ({name, source, n, types, groups, meta}).

        Group index arrays are stored inside the npz as ``group__<name>``; the json ``groups``
        entry maps each name to its size only (keeps the json small). ``nt`` is stored as a
        fixed-width unicode array (no pickle needed on load).
        """
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "pre": np.ascontiguousarray(self.pre, dtype=np.int32),
            "post": np.ascontiguousarray(self.post, dtype=np.int32),
            "weight": np.ascontiguousarray(self.weight, dtype=np.float32),
            "sign": np.ascontiguousarray(self.sign, dtype=np.float32),
            "region": np.ascontiguousarray(self.region, dtype=np.uint8),
            "side": np.ascontiguousarray(self.side, dtype=np.int8),
            "type_idx": np.ascontiguousarray(self.type_idx, dtype=np.int32),
            "body_id": np.ascontiguousarray(self.body_id, dtype=np.int64),
            "nt": np.asarray([str(x) for x in self.nt], dtype=str),
        }
        for gname, idx in self.groups.items():
            arrays[_GROUP_NPZ_PREFIX + gname] = np.ascontiguousarray(idx, dtype=np.int32)
        np.savez(str(stem.with_suffix(".npz")), **arrays)
        doc = {
            "name": self.name,
            "source": self.source,
            "n": int(self.n),
            "types": list(self.types),
            "groups": {gname: int(np.asarray(idx).size) for gname, idx in self.groups.items()},
            "meta": self.meta,
        }
        with open(stem.with_suffix(".json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, default=_json_default)

    @classmethod
    def load(cls, stem: Path, mmap: bool = False) -> "Connectome":
        """Inverse of ``save``; ``mmap=True`` memory-maps the numeric arrays (full-graph mode).

        ``np.load(..., mmap_mode='r')`` silently ignores ``mmap_mode`` for ``.npz`` archives, so
        for ``mmap=True`` the uncompressed members are mapped directly with ``np.memmap`` at their
        offsets inside the zip; any member that cannot be mapped is read normally.
        """
        stem = Path(stem)
        with open(stem.with_suffix(".json"), "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        arrays = _read_npz(stem.with_suffix(".npz"), mmap=mmap)
        groups: dict[str, np.ndarray] = {}
        for key, arr in arrays.items():
            if key.startswith(_GROUP_NPZ_PREFIX):
                groups[key[len(_GROUP_NPZ_PREFIX):]] = np.asarray(arr, dtype=np.int32)
        json_groups = doc.get("groups") or {}
        for gname, val in json_groups.items():
            if gname not in groups and isinstance(val, list):  # legacy layout: indices in json
                groups[gname] = np.asarray(val, dtype=np.int32)
            elif gname not in groups:
                groups[gname] = np.zeros(0, dtype=np.int32)
        nt = np.asarray(arrays["nt"]).astype(object)
        conn = cls(
            name=str(doc["name"]),
            source=str(doc["source"]),
            n=int(doc["n"]),
            pre=_typed(arrays["pre"], np.int32),
            post=_typed(arrays["post"], np.int32),
            weight=_typed(arrays["weight"], np.float32),
            sign=_typed(arrays["sign"], np.float32),
            region=_typed(arrays["region"], np.uint8),
            side=_typed(arrays["side"], np.int8),
            types=[str(t) for t in doc["types"]],
            type_idx=_typed(arrays["type_idx"], np.int32),
            body_id=_typed(arrays["body_id"], np.int64),
            nt=nt,
            groups=groups,
            meta=dict(doc.get("meta") or {}),
        )
        return conn

    # ---- derived connectomes --------------------------------------------------

    def subset(self, keep: np.ndarray, min_weight: float = 1.0) -> "Connectome":
        """Restrict to ``keep x keep`` (keep sorted int32), drop edges with ``weight < min_weight``,
        remap indices, rebuild groups by intersection, update ``meta['e']``,
        ``meta['region_counts']`` (and ``synapses`` / ``group_counts``). Returns a new instance."""
        keep = np.unique(np.asarray(keep, dtype=np.int64))
        if keep.size and (keep[0] < 0 or keep[-1] >= self.n):
            raise ValueError(f"keep indices out of range [0, {self.n})")
        n2 = int(keep.size)
        new_id = np.full(self.n, -1, dtype=np.int64)
        new_id[keep] = np.arange(n2, dtype=np.int64)
        emask = (new_id[self.pre] >= 0) & (new_id[self.post] >= 0) & (self.weight >= np.float32(min_weight))
        pre2 = new_id[self.pre[emask]].astype(np.int32)
        post2 = new_id[self.post[emask]].astype(np.int32)
        w2 = np.ascontiguousarray(self.weight[emask], dtype=np.float32)
        groups2: dict[str, np.ndarray] = {}
        for gname, idx in self.groups.items():
            mapped = new_id[np.asarray(idx, dtype=np.int64)]
            groups2[gname] = mapped[mapped >= 0].astype(np.int32)
        region2 = np.ascontiguousarray(self.region[keep], dtype=np.uint8)
        meta2 = dict(self.meta)
        meta2["e"] = int(pre2.shape[0])
        meta2["synapses"] = float(w2.sum(dtype=np.float64)) if w2.size else 0.0
        meta2["region_counts"] = count_regions(region2)
        meta2["group_counts"] = count_groups(groups2)
        meta2["subset"] = {"parent_n": int(self.n), "parent_e": int(self.e), "min_weight": float(min_weight)}
        return Connectome(
            name=self.name,
            source=self.source,
            n=n2,
            pre=pre2,
            post=post2,
            weight=w2,
            sign=np.ascontiguousarray(self.sign[keep], dtype=np.float32),
            region=region2,
            side=np.ascontiguousarray(self.side[keep], dtype=np.int8),
            types=list(self.types),
            type_idx=np.ascontiguousarray(self.type_idx[keep], dtype=np.int32),
            body_id=np.ascontiguousarray(self.body_id[keep], dtype=np.int64),
            nt=np.asarray(self.nt, dtype=object)[keep],
            groups=groups2,
            meta=meta2,
        )


# --------------------------------------------------------------------------- npz helpers


def _typed(arr: np.ndarray, dtype: type) -> np.ndarray:
    """Return ``arr`` viewed with ``dtype`` without copying when it already matches (memmaps stay memmaps)."""
    arr = np.asarray(arr) if not isinstance(arr, np.memmap) else arr
    return arr if arr.dtype == dtype else arr.astype(dtype)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


_LOCAL_HEADER = struct.Struct("<4s2B4HL2L2H")  # zipfile.structFileHeader, 30 bytes


def _read_npz(path: Path, mmap: bool) -> dict[str, np.ndarray]:
    """Read every member of an ``.npz``; with ``mmap=True`` map stored (uncompressed) numeric members."""
    if mmap:
        try:
            return _memmap_npz(path)
        except Exception:  # pragma: no cover - defensive fallback to a plain read
            pass
    out: dict[str, np.ndarray] = {}
    with np.load(str(path), allow_pickle=False) as z:
        for key in z.files:
            out[key] = z[key]
    return out


def _memmap_npz(path: Path) -> dict[str, np.ndarray]:
    """Map each ZIP_STORED ``.npy`` member of ``path`` read-only at its byte offset in the archive."""
    out: dict[str, np.ndarray] = {}
    path = Path(path)
    with zipfile.ZipFile(path) as zf, open(path, "rb") as raw:
        for info in zf.infolist():
            key = info.filename[:-4] if info.filename.endswith(".npy") else info.filename
            arr: np.ndarray | None = None
            if info.compress_type == zipfile.ZIP_STORED and info.flag_bits & 0x1 == 0:
                with zf.open(info) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        shape, fortran, dtype = None, False, np.dtype(object)
                    header_len = member.tell()
                if shape is not None and not dtype.hasobject:
                    raw.seek(info.header_offset)
                    fields = _LOCAL_HEADER.unpack(raw.read(_LOCAL_HEADER.size))
                    name_len, extra_len = fields[10], fields[11]
                    offset = info.header_offset + _LOCAL_HEADER.size + name_len + extra_len + header_len
                    count = int(np.prod(shape, dtype=np.int64)) if len(shape) else 1
                    if count == 0:
                        arr = np.zeros(shape, dtype=dtype)
                    else:
                        arr = np.memmap(
                            path, dtype=dtype, mode="r", offset=offset, shape=tuple(shape),
                            order="F" if fortran else "C",
                        )
            if arr is None:
                with zf.open(info) as member:
                    arr = np.lib.format.read_array(member, allow_pickle=False)
            out[key] = arr
    return out
