"""Sparse (COO -> CSR) helpers, pure numpy (SPEC section c.3).

Every function here is deterministic and uses only ``np.argsort(kind='quicksort')``,
``np.bincount`` and ``cumsum`` (never ``kind='stable'``, per the contract). All
returned index arrays are ``int32``, row pointers are ``int64``, data is ``float32``.

No biological numbers live in this module.
"""

from __future__ import annotations

import numpy as np

__all__ = ["build_csr", "sum_duplicates", "remap_ids", "transpose_csr", "in_degree"]


# --------------------------------------------------------------------------- internals


def _as_edge_arrays(
    pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce inputs to (int64 pre, int64 post, float32 w) and bounds-check them."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    pre64 = np.asarray(pre).astype(np.int64, copy=False).ravel()
    post64 = np.asarray(post).astype(np.int64, copy=False).ravel()
    w32 = np.asarray(w).astype(np.float32, copy=False).ravel()
    if not (pre64.shape == post64.shape == w32.shape):
        raise ValueError(
            f"pre/post/w must have equal length, got {pre64.size}/{post64.size}/{w32.size}"
        )
    if pre64.size:
        if pre64.min() < 0 or pre64.max() >= n:
            raise ValueError(f"pre indices out of range [0, {n})")
        if post64.min() < 0 or post64.max() >= n:
            raise ValueError(f"post indices out of range [0, {n})")
    return pre64, post64, w32


def _sort_coo(
    pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int, merge: bool, drop_self_loops: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort COO triplets by (pre, post); optionally drop self-loops and merge duplicates.

    Returns int64 pre, int64 post, float32 w. Sort key = pre*n + post (int64, unique per pair)
    with ``np.argsort(kind='quicksort')``; because keys are unique after merging, the result is
    deterministic. When ``merge`` is False duplicates stay adjacent in unspecified relative order.
    """
    if drop_self_loops and pre.size:
        keep = pre != post
        if not keep.all():
            pre, post, w = pre[keep], post[keep], w[keep]
    if pre.size == 0:
        return pre, post, w
    key = pre * np.int64(max(n, 1)) + post
    order = np.argsort(key, kind="quicksort")
    key, pre, post, w = key[order], pre[order], post[order], w[order]
    if merge and key.size > 1:
        first = np.empty(key.size, dtype=bool)
        first[0] = True
        np.not_equal(key[1:], key[:-1], out=first[1:])
        if not first.all():
            starts = np.flatnonzero(first)
            # accumulate in float64, then back to float32 (synapse counts stay exact well below 2**24)
            w = np.add.reduceat(w.astype(np.float64), starts).astype(np.float32)
            pre, post = pre[starts], post[starts]
    return pre, post, w


# --------------------------------------------------------------------------- public API


def build_csr(
    pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int, sum_duplicates: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """COO -> (indptr int64[n+1], indices int32[E], data float32[E]).

    Sort key = ``pre.astype(int64)*n + post`` using ``np.argsort(kind='quicksort')`` (never
    'stable'); duplicate (pre, post) pairs are summed when ``sum_duplicates``; ``indptr`` comes
    from ``np.bincount(pre, minlength=n).cumsum()``. Self-loops are dropped. The returned arrays
    are sorted by (pre, post), i.e. ``indices`` is sorted within every row.

    ``w`` may be signed (``weight * sign[pre]``) or unsigned; this function does not care.
    Raises ``ValueError`` on shape mismatch or out-of-range indices.
    """
    pre64, post64, w32 = _as_edge_arrays(pre, post, w, n)
    pre64, post64, w32 = _sort_coo(pre64, post64, w32, n, merge=sum_duplicates, drop_self_loops=True)
    indptr = np.zeros(n + 1, dtype=np.int64)
    if pre64.size:
        indptr[1:] = np.bincount(pre64, minlength=n).cumsum()
    return indptr, post64.astype(np.int32), w32


def sum_duplicates(
    pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int, drop_self_loops: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge duplicate (pre, post) pairs of a COO edge list by summing their weights.

    Returns ``(pre int32[E'], post int32[E'], w float32[E'])`` sorted by (pre, post), i.e. the
    canonical ``Connectome.pre/post/weight`` order. Self-loops are dropped by default because a
    ``Connectome`` never contains them (SPEC section c.2); pass ``drop_self_loops=False`` to keep
    them. Pure numpy: argsort + reduceat.
    """
    pre64, post64, w32 = _as_edge_arrays(pre, post, w, n)
    pre64, post64, w32 = _sort_coo(pre64, post64, w32, n, merge=True, drop_self_loops=drop_self_loops)
    return pre64.astype(np.int32), post64.astype(np.int32), w32


def remap_ids(
    pre_ids: np.ndarray, post_ids: np.ndarray, body_ids_sorted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """int64 external ids -> int32 indices via ``np.searchsorted`` on the sorted ``body_id`` array.

    Returns ``(pre int32[E], post int32[E], keep_mask bool[E])`` where ``keep_mask`` marks edges
    whose both endpoints exist in ``body_ids_sorted``. Endpoints that do not exist are reported
    as ``-1`` in ``pre``/``post`` (callers filter with ``keep_mask``). ``body_ids_sorted`` must be
    strictly increasing (``ValueError`` otherwise).
    """
    body = np.asarray(body_ids_sorted).astype(np.int64, copy=False).ravel()
    if body.size > 1 and not np.all(body[1:] > body[:-1]):
        raise ValueError("body_ids_sorted must be strictly increasing (sorted, unique)")
    pre64 = np.asarray(pre_ids).astype(np.int64, copy=False).ravel()
    post64 = np.asarray(post_ids).astype(np.int64, copy=False).ravel()
    if pre64.shape != post64.shape:
        raise ValueError(f"pre_ids/post_ids must have equal length, got {pre64.size}/{post64.size}")

    def _lookup(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if body.size == 0:
            return np.full(ids.shape, -1, dtype=np.int32), np.zeros(ids.shape, dtype=bool)
        pos = np.searchsorted(body, ids)
        pos_c = np.minimum(pos, body.size - 1)
        found = body[pos_c] == ids
        out = pos_c.astype(np.int32)
        out[~found] = -1
        return out, found

    pre_idx, pre_ok = _lookup(pre64)
    post_idx, post_ok = _lookup(post64)
    return pre_idx, post_idx, pre_ok & post_ok


def transpose_csr(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Incoming view of a CSR matrix (used by calibrate/validate and tests); pure numpy.

    Given the outgoing CSR ``(indptr, indices, data)`` over ``n`` rows/columns, return the CSR of
    the transposed matrix: row ``j`` of the result lists the presynaptic partners of neuron ``j``
    (sorted), with the same ``data`` values. Duplicates are neither created nor merged and
    self-loops are preserved, so ``transpose_csr(transpose_csr(A)) == A`` exactly for any CSR
    whose rows are sorted and duplicate-free.
    """
    indptr = np.asarray(indptr).astype(np.int64, copy=False)
    if indptr.size != n + 1:
        raise ValueError(f"indptr must have length n+1 = {n + 1}, got {indptr.size}")
    counts = np.diff(indptr)
    if counts.size and counts.min() < 0:
        raise ValueError("indptr must be non-decreasing")
    src = np.repeat(np.arange(n, dtype=np.int64), counts)
    dst = np.asarray(indices).astype(np.int64, copy=False)
    if dst.size != src.size:
        raise ValueError(f"indices length {dst.size} does not match indptr[-1] = {src.size}")
    w = np.asarray(data).astype(np.float32, copy=False)
    if w.size != dst.size:
        raise ValueError(f"data length {w.size} does not match indices length {dst.size}")
    if dst.size and (dst.min() < 0 or dst.max() >= n):
        raise ValueError(f"indices out of range [0, {n})")
    # transposed edges: new pre = old post, new post = old pre
    t_pre, t_post, t_w = _sort_coo(dst, src, w, n, merge=False, drop_self_loops=False)
    t_indptr = np.zeros(n + 1, dtype=np.int64)
    if t_pre.size:
        t_indptr[1:] = np.bincount(t_pre, minlength=n).cumsum()
    return t_indptr, t_post.astype(np.int32), t_w


def in_degree(indptr: np.ndarray, indices: np.ndarray, n: int) -> np.ndarray:
    """int64[n] number of incoming edges (``np.bincount(indices, minlength=n)``)."""
    idx = np.asarray(indices)
    if idx.size == 0:
        return np.zeros(n, dtype=np.int64)
    return np.bincount(idx, minlength=n).astype(np.int64, copy=False)
