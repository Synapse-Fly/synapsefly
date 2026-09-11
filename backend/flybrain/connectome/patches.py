"""Electrical-coupling patches: ``GapJunction``, ``GAP_JUNCTIONS``, ``apply_patches`` (SPEC section c.5).

The MaleCNS synapse table holds chemical synapses only; the giant-fibre escape circuit relies on
gap junctions that the EM weights cannot see (DNp01 -> TTMn is 45 chemical synapses per side but
mostly electrical in reality, RESEARCH section 5.1 [V]). ``GAP_JUNCTIONS`` adds engineered ``[E]``
proxy weights on top of the chemical counts so the escape reflex works on every source
(synthetic, csv, neuprint). Every application is recorded in ``meta['patches_applied']`` and the
function is idempotent, so a cached (already patched) graph is never patched twice.

Only numpy is imported at module level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .csr import sum_duplicates
from .schema import Connectome

__all__ = ["GapJunction", "GAP_JUNCTIONS", "LATERALITIES", "apply_patches", "is_applied", "patch_pairs"]

log = logging.getLogger("flybrain.connectome.patches")


@dataclass(frozen=True)
class GapJunction:
    """One type-to-type electrical-coupling proxy.

    ``pre`` / ``post`` are exact MaleCNS ``type`` strings, ``weight`` is the UNSIGNED synapse-count
    equivalent added to every (pre cell, post cell) pair selected by ``laterality``
    (``ipsi`` = same side, ``contra`` = opposite side, ``both`` = every pair). Side 0 (midline
    ``'M'`` or an unknown ``somaSide``) is a WILDCARD for laterality: a pair in which either cell
    has side 0 matches ``ipsi`` and ``contra`` alike, so a real export with a missing soma side
    never silently loses its escape-reflex proxy (a warning is logged instead). Self-loops are
    never created. The edge sign follows ``sign[pre]`` like every other edge (all listed pres are
    excitatory [V]).
    """

    pre: str
    post: str
    weight: float
    laterality: str  # ipsi|contra|both


#: allowed ``GapJunction.laterality`` values.
LATERALITIES: tuple[str, ...] = ("ipsi", "contra", "both")

#: Patch table of SPEC section c.5 (verbatim). Chemical counts quoted from RESEARCH section 5.1 [V];
#: the proxy weights themselves are engineered [E].
GAP_JUNCTIONS: tuple[GapJunction, ...] = (
    GapJunction("DNp01", "TTMn", 300.0, "ipsi"),     # [E] electrical GF->TTMn proxy (chemical count is 45/side)
    GapJunction("DNp01", "PSI",  200.0, "ipsi"),     # [E] electrical GF->PSI proxy (chemical 4/side)
    GapJunction("DNp01", "DNp01", 80.0, "contra"),   # [E] GF<->GF coupling
    GapJunction("LC4",   "DNp01",  6.0, "ipsi"),     # [E] dendro-dendritic proxy, added on top of chemical counts
    GapJunction("LPLC2", "DNp01",  6.0, "ipsi"),
)


# --------------------------------------------------------------------------- helpers


def _record_of(gj: GapJunction) -> dict:
    return {"pre": gj.pre, "post": gj.post, "weight": float(gj.weight), "laterality": gj.laterality}


def _same_patch(rec: dict, gj: GapJunction) -> bool:
    try:
        return (
            str(rec.get("pre")) == gj.pre
            and str(rec.get("post")) == gj.post
            and float(rec.get("weight")) == float(gj.weight)
            and str(rec.get("laterality")) == gj.laterality
        )
    except (TypeError, ValueError):
        return False


def is_applied(conn: Connectome, gj: GapJunction) -> bool:
    """True when ``conn.meta['patches_applied']`` already holds a record equal to ``gj``."""
    applied = conn.meta.get("patches_applied") or []
    return any(isinstance(rec, dict) and _same_patch(rec, gj) for rec in applied)


def _members(conn: Connectome, type_name: str) -> np.ndarray:
    """int64 indices of every neuron whose type string equals ``type_name`` exactly."""
    try:
        t = conn.types.index(type_name)
    except ValueError:
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.asarray(conn.type_idx) == t).astype(np.int64)


def patch_pairs(conn: Connectome, gj: GapJunction) -> tuple[np.ndarray, np.ndarray]:
    """(pre, post) int64 index pairs selected by ``gj`` in ``conn`` (laterality applied, no self-loops).

    Laterality uses ``conn.side``: ``ipsi`` keeps pairs with equal side, ``contra`` pairs with
    opposite side, ``both`` every pair. Side 0 (midline / unknown) is a wildcard: a pair in which
    EITHER cell has side 0 matches ``ipsi`` and ``contra`` alike (two side-0 cells therefore behave
    like ``both``). When wildcard pairs are selected under ``ipsi`` / ``contra`` a warning names
    the patch and the number of side-0 cells involved (ASCII only).
    """
    if gj.laterality not in LATERALITIES:
        raise ValueError(f"GapJunction {gj.pre}->{gj.post}: laterality must be one of {LATERALITIES}, got {gj.laterality!r}")
    src = _members(conn, gj.pre)
    dst = _members(conn, gj.post)
    if src.size == 0 or dst.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pp, qq = np.meshgrid(src, dst, indexing="ij")
    pp, qq = pp.ravel(), qq.ravel()
    side = np.asarray(conn.side, dtype=np.int8)
    wildcard = (side[pp] == 0) | (side[qq] == 0)
    if gj.laterality == "ipsi":
        mask = (side[pp] == side[qq]) | wildcard
    elif gj.laterality == "contra":
        mask = (side[pp] == -side[qq]) | wildcard
    else:
        mask = np.ones(pp.shape, dtype=bool)
    mask &= pp != qq
    n_wild = int((wildcard & mask).sum())
    if gj.laterality != "both" and n_wild:
        log.warning(
            "gap-junction patch %s->%s (%s): %d %s and %d %s cells have side 0 (midline/unknown); "
            "matched as wildcards (%d pairs)",
            gj.pre, gj.post, gj.laterality, int((side[src] == 0).sum()), gj.pre, int((side[dst] == 0).sum()),
            gj.post, n_wild,
        )
    return pp[mask], qq[mask]


# --------------------------------------------------------------------------- public API


def apply_patches(conn: Connectome, table: Sequence[GapJunction] = GAP_JUNCTIONS) -> Connectome:
    """Add (or increase) the UNSIGNED weight of the listed type-to-type edges with laterality.

    Returns a NEW ``Connectome`` (per-neuron arrays, ``types`` and ``groups`` are shared with the
    input; ``pre``/``post``/``weight`` are rebuilt sorted by (pre, post) with no duplicates and no
    self-loops; ``meta`` is a shallow copy). Existing (pre, post) pairs get ``weight += gj.weight``,
    missing pairs are created with ``gj.weight``; the edge sign follows ``sign[pre]`` because the
    data model stores signs per presynaptic neuron. For every patch applied a record
    ``{"pre","post","weight","laterality","edges_added","edges_increased"}`` is appended to
    ``meta['patches_applied']`` (``edges_added`` = new pairs, ``edges_increased`` = pairs whose
    weight was raised; both 0 when a type is absent from the graph, which is still recorded so
    the table is not retried on every start). Idempotent: patches already present in
    ``meta['patches_applied']`` are skipped, and when nothing is pending the input instance is
    returned unchanged. ``meta['e']`` / ``meta['synapses']`` are refreshed.
    """
    pending = [gj for gj in table if not is_applied(conn, gj)]
    for gj in pending:
        if gj.laterality not in LATERALITIES:
            raise ValueError(
                f"GapJunction {gj.pre}->{gj.post}: laterality must be one of {LATERALITIES}, got {gj.laterality!r}"
            )
        if not (float(gj.weight) > 0.0):
            raise ValueError(f"GapJunction {gj.pre}->{gj.post}: weight must be > 0, got {gj.weight!r}")
    if not pending:
        return conn

    n = int(conn.n)
    old_pre = np.asarray(conn.pre, dtype=np.int64)
    old_post = np.asarray(conn.post, dtype=np.int64)
    old_w = np.asarray(conn.weight, dtype=np.float32)
    existing_keys = old_pre * n + old_post  # unique because a valid Connectome has no duplicates
    existing_sorted = np.sort(existing_keys)

    add_pre: list[np.ndarray] = [old_pre]
    add_post: list[np.ndarray] = [old_post]
    add_w: list[np.ndarray] = [old_w]
    records: list[dict] = []
    for gj in pending:
        pp, qq = patch_pairs(conn, gj)
        rec = _record_of(gj)
        if pp.size:
            keys = pp * n + qq
            pos = np.searchsorted(existing_sorted, keys)
            pos = np.minimum(pos, max(existing_sorted.size - 1, 0))
            present = existing_sorted.size > 0
            hit = (existing_sorted[pos] == keys) if present else np.zeros(keys.shape, dtype=bool)
            rec["edges_added"] = int((~hit).sum())
            rec["edges_increased"] = int(hit.sum())
            add_pre.append(pp)
            add_post.append(qq)
            add_w.append(np.full(pp.shape[0], float(gj.weight), dtype=np.float32))
            # later patches in the same table must see these pairs as existing
            existing_sorted = np.union1d(existing_sorted, keys)
        else:
            rec["edges_added"] = 0
            rec["edges_increased"] = 0
        records.append(rec)

    pre_all = np.concatenate(add_pre)
    post_all = np.concatenate(add_post)
    w_all = np.concatenate(add_w)
    pre, post, weight = sum_duplicates(pre_all, post_all, w_all, n)  # sorted, merged, no self-loops

    meta = dict(conn.meta)
    meta["patches_applied"] = list(conn.meta.get("patches_applied") or []) + records
    meta["e"] = int(pre.shape[0])
    meta["synapses"] = float(weight.sum(dtype=np.float64)) if weight.size else 0.0
    return Connectome(
        name=conn.name,
        source=conn.source,
        n=n,
        pre=pre.astype(np.int32, copy=False),
        post=post.astype(np.int32, copy=False),
        weight=weight.astype(np.float32, copy=False),
        sign=conn.sign,
        region=conn.region,
        side=conn.side,
        types=conn.types,
        type_idx=conn.type_idx,
        body_id=conn.body_id,
        nt=conn.nt,
        groups=conn.groups,
        meta=meta,
    )
