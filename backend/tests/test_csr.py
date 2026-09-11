"""Tests for flybrain.connectome.csr and the Connectome / CSR data model (SPEC c.2, c.3)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from flybrain.connectome.csr import build_csr, in_degree, remap_ids, sum_duplicates, transpose_csr
from flybrain.connectome.schema import CSR, REGION_ID, REGIONS, Connectome, count_regions

# --------------------------------------------------------------------------- helpers


def _rows(indptr: np.ndarray, indices: np.ndarray) -> list[list[int]]:
    return [indices[indptr[i] : indptr[i + 1]].tolist() for i in range(len(indptr) - 1)]


def _copy_with(conn: Connectome, **changes) -> Connectome:
    """Shallow-copied Connectome with some fields replaced (never mutates the fixture)."""
    fields = {f.name: getattr(conn, f.name) for f in dataclasses.fields(conn) if f.init}
    fields.update(changes)
    return Connectome(**fields)


# --------------------------------------------------------------------------- build_csr


def test_build_csr_small_merges_and_drops_self_loops():
    pre = [2, 0, 1, 0, 2, 2, 1]
    post = [0, 1, 2, 1, 2, 1, 0]
    w = [1, 2, 3, 4, 5, 6, 7]  # (0,1) appears twice -> 6 ; (2,2) is a self-loop -> dropped
    indptr, indices, data = build_csr(pre, post, w, 3)
    assert indptr.dtype == np.int64 and indices.dtype == np.int32 and data.dtype == np.float32
    assert indptr.tolist() == [0, 1, 3, 5]
    assert indices.tolist() == [1, 0, 2, 0, 1]
    assert data.tolist() == [6.0, 7.0, 3.0, 1.0, 6.0]


def test_build_csr_keeps_duplicates_when_asked():
    indptr, indices, data = build_csr([0, 0, 1], [1, 1, 0], [2.0, 4.0, 1.0], 2, sum_duplicates=False)
    assert indptr.tolist() == [0, 2, 3]
    assert indices.tolist() == [1, 1, 0]
    assert sorted(data[:2].tolist()) == [2.0, 4.0]


def test_build_csr_empty_and_errors():
    indptr, indices, data = build_csr(np.zeros(0), np.zeros(0), np.zeros(0), 5)
    assert indptr.tolist() == [0] * 6 and indices.size == 0 and data.size == 0
    with pytest.raises(ValueError):
        build_csr([0, 1], [1], [1.0, 1.0], 3)
    with pytest.raises(ValueError):
        build_csr([0, 3], [1, 0], [1.0, 1.0], 3)
    with pytest.raises(ValueError):
        build_csr([0, -1], [1, 0], [1.0, 1.0], 3)


def test_build_csr_invariants_on_fixture(tiny_connectome: Connectome):
    conn = tiny_connectome
    signed = conn.weight * conn.sign[conn.pre]
    indptr, indices, data = build_csr(conn.pre, conn.post, signed, conn.n)
    assert indptr.shape == (conn.n + 1,)
    assert np.all(np.diff(indptr) >= 0) and indptr[0] == 0 and indptr[-1] == conn.e
    for row in _rows(indptr, indices):
        assert row == sorted(row) and len(set(row)) == len(row)
    # already-sorted, duplicate-free input is returned in identical order
    assert np.array_equal(indices, conn.post)
    assert np.array_equal(data, signed.astype(np.float32))
    assert np.array_equal(np.diff(indptr), np.bincount(conn.pre, minlength=conn.n))


# --------------------------------------------------------------------------- sum_duplicates


def test_sum_duplicates_merges_pairs():
    pre = np.array([3, 1, 3, 1, 0, 2])
    post = np.array([0, 2, 0, 2, 0, 1])
    w = np.array([1.5, 2.0, 2.5, 3.0, 9.0, 1.0])
    p, q, ww = sum_duplicates(pre, post, w, 4)
    assert p.dtype == np.int32 and q.dtype == np.int32 and ww.dtype == np.float32
    assert p.tolist() == [1, 2, 3] and q.tolist() == [2, 1, 0]
    assert ww.tolist() == [5.0, 1.0, 4.0]
    p2, q2, w2 = sum_duplicates(pre, post, w, 4, drop_self_loops=False)
    assert p2.tolist() == [0, 1, 2, 3] and w2.tolist() == [9.0, 5.0, 1.0, 4.0]


# --------------------------------------------------------------------------- transpose / in_degree


def test_transpose_roundtrip_and_in_degree(tiny_connectome: Connectome):
    csr = tiny_connectome.csr()
    t_indptr, t_indices, t_data = transpose_csr(csr.indptr, csr.indices, csr.data, csr.n)
    assert t_indptr.dtype == np.int64 and t_indices.dtype == np.int32 and t_data.dtype == np.float32
    assert t_indptr[-1] == csr.e
    for row in _rows(t_indptr, t_indices):
        assert row == sorted(row)
    assert np.array_equal(np.diff(t_indptr), in_degree(csr.indptr, csr.indices, csr.n))
    assert in_degree(csr.indptr, csr.indices, csr.n).dtype == np.int64
    # total signed weight is preserved and the transpose of the transpose is the original
    assert np.isclose(t_data.sum(dtype=np.float64), csr.data.sum(dtype=np.float64))
    b_indptr, b_indices, b_data = transpose_csr(t_indptr, t_indices, t_data, csr.n)
    assert np.array_equal(b_indptr, csr.indptr)
    assert np.array_equal(b_indices, csr.indices)
    assert np.array_equal(b_data, csr.data)


def test_transpose_small_explicit():
    indptr, indices, data = build_csr([0, 0, 1], [1, 2, 2], [1.0, 2.0, 3.0], 3)
    t_indptr, t_indices, t_data = transpose_csr(indptr, indices, data, 3)
    assert t_indptr.tolist() == [0, 0, 1, 3]
    assert t_indices.tolist() == [0, 0, 1]
    assert t_data.tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        transpose_csr(indptr, indices, data, 2)


# --------------------------------------------------------------------------- remap_ids


def test_remap_ids_bijection():
    rng = np.random.default_rng(7)
    body = np.sort(rng.choice(10**12, size=60, replace=False).astype(np.int64))
    pre_idx = rng.integers(0, 60, size=500)
    post_idx = rng.integers(0, 60, size=500)
    pre, post, keep = remap_ids(body[pre_idx], body[post_idx], body)
    assert pre.dtype == np.int32 and post.dtype == np.int32 and keep.dtype == bool
    assert keep.all()
    assert np.array_equal(pre, pre_idx) and np.array_equal(post, post_idx)
    assert np.array_equal(body[pre], body[pre_idx])


def test_remap_ids_marks_missing_endpoints():
    body = np.array([10, 20, 30, 40], dtype=np.int64)
    pre, post, keep = remap_ids(np.array([10, 20, 99, 40, 5]), np.array([20, 77, 30, 40, 10]), body)
    assert keep.tolist() == [True, False, False, True, False]
    assert pre.tolist() == [0, 1, -1, 3, -1]
    assert post.tolist() == [1, -1, 2, 3, 0]
    with pytest.raises(ValueError):
        remap_ids(np.array([1]), np.array([2]), np.array([3, 2, 1], dtype=np.int64))
    p, q, k = remap_ids(np.array([1]), np.array([2]), np.zeros(0, dtype=np.int64))
    assert not k.any() and p.tolist() == [-1]


# --------------------------------------------------------------------------- Connectome / CSR


def test_regions_table():
    assert REGIONS == ("optic_lobe", "antennal_lobe", "mushroom_body", "central_complex",
                       "sez", "central_other", "descending_motor", "vnc")
    assert [REGION_ID[r] for r in REGIONS] == list(range(8))


def test_fixture_shape_and_dtypes(tiny_connectome: Connectome):
    c = tiny_connectome
    assert c.n == 400 and 2500 <= c.e <= 4000
    assert c.pre.dtype == np.int32 and c.post.dtype == np.int32 and c.weight.dtype == np.float32
    assert c.sign.dtype == np.float32 and c.region.dtype == np.uint8 and c.side.dtype == np.int8
    assert c.type_idx.dtype == np.int32 and c.body_id.dtype == np.int64 and c.nt.dtype == object
    assert set(np.unique(c.region).tolist()) == set(range(8))
    assert set(np.unique(c.side).tolist()) == {-1, 0, 1}
    assert 0.2 <= float((c.sign < 0).mean()) <= 0.4
    assert np.all(c.weight >= 1.0)
    assert sum(c.meta["region_counts"].values()) == c.n
    assert c.meta["e"] == c.e
    assert c.type_of(int(c.groups["gf"][0])) == "DNp01"
    c.validate()  # must not raise


def test_csr_is_signed_and_cached(tiny_connectome: Connectome):
    c = tiny_connectome
    csr = c.csr()
    assert isinstance(csr, CSR) and c.csr() is csr
    assert csr.n == c.n and csr.e == c.e
    assert csr.indptr.dtype == np.int64 and csr.indptr32 is not None and csr.indptr32.dtype == np.int32
    assert np.array_equal(csr.indptr32, csr.indptr)
    assert np.array_equal(csr.data, (c.weight * c.sign[c.pre]).astype(np.float32))
    inhib = np.flatnonzero(c.sign < 0)
    for i in inhib[:20]:
        _, d = csr.row(int(i))
        assert np.all(d < 0)
    exc = np.flatnonzero(c.sign > 0)
    for i in exc[:20]:
        _, d = csr.row(int(i))
        assert np.all(d > 0)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda c: {"weight": np.where(np.arange(c.e) == 5, 0.0, c.weight).astype(np.float32)}, "weight"),
        (lambda c: {"post": np.where(np.arange(c.e) == 0, c.pre[0], c.post).astype(np.int32)}, "self-loop"),
        (lambda c: {"pre": c.pre[::-1].copy()}, "non-decreasing"),
        (lambda c: {"post": np.where(np.arange(c.e) == 1, c.post[0], c.post).astype(np.int32)}, "duplicate"),
        (lambda c: {"region": np.where(np.arange(c.n) == 3, 8, c.region).astype(np.uint8)}, "region"),
        (lambda c: {"side": np.where(np.arange(c.n) == 3, 2, c.side).astype(np.int8)}, "side"),
        (lambda c: {"sign": np.where(np.arange(c.n) == 3, 0.0, c.sign).astype(np.float32)}, "sign"),
        (lambda c: {"weight": c.weight[:-1]}, r"len\(pre\)"),
        (lambda c: {"groups": {**c.groups, "gf": c.groups["gf"][::-1].copy()}}, "gf"),
        (lambda c: {"groups": {**c.groups, "gf": c.groups["gf"].astype(np.int64)}}, "int32"),
        (lambda c: {"meta": {**c.meta, "region_counts": {"optic_lobe": 1}}}, "region_counts"),
    ],
)
def test_validate_rejects_corruption(tiny_connectome: Connectome, mutate, needle):
    bad = _copy_with(tiny_connectome, **mutate(tiny_connectome))
    with pytest.raises(ValueError, match=needle):
        bad.validate()


def test_where_matches_groups(tiny_connectome: Connectome):
    c = tiny_connectome
    assert np.array_equal(c.where("DNp01"), c.groups["gf"])
    assert np.array_equal(c.where("DNp01", side=-1), c.groups["gf_L"])
    assert np.array_equal(c.where("DNp01", side=1), c.groups["gf_R"])
    assert c.where("DNp01").dtype == np.int32
    assert np.array_equal(c.where(r"(LC4|LPLC2)"), c.groups["lc_loom"])
    assert c.where("DNp0").size == 0  # fullmatch, not search


@pytest.mark.parametrize("mmap", [False, True])
def test_save_load_roundtrip(tiny_connectome: Connectome, tmp_path, mmap):
    c = tiny_connectome
    stem = tmp_path / "cache" / "tiny"
    c.save(stem)
    assert stem.with_suffix(".npz").exists() and stem.with_suffix(".json").exists()
    d = Connectome.load(stem, mmap=mmap)
    assert d.name == c.name and d.source == c.source and d.n == c.n and d.types == c.types
    for name in ("pre", "post", "weight", "sign", "region", "side", "type_idx", "body_id"):
        a, b = getattr(c, name), getattr(d, name)
        assert b.dtype == a.dtype, name
        assert np.array_equal(a, b), name
        if mmap:
            assert isinstance(b, np.memmap), name
    assert d.nt.dtype == object and [str(x) for x in d.nt] == [str(x) for x in c.nt]
    assert set(d.groups) == set(c.groups)
    for g in c.groups:
        assert d.groups[g].dtype == np.int32 and np.array_equal(d.groups[g], c.groups[g]), g
    assert d.meta["e"] == c.meta["e"] and d.meta["region_counts"] == c.meta["region_counts"]
    assert d.meta["citation"] is None
    d.validate()
    dcsr, ccsr = d.csr(), c.csr()
    assert np.array_equal(dcsr.indptr, ccsr.indptr) and np.array_equal(dcsr.data, ccsr.data)
    del d, dcsr  # release memmaps before tmp_path cleanup on Windows


def test_subset_restricts_and_remaps(tiny_connectome: Connectome):
    c = tiny_connectome
    keep = np.unique(np.concatenate([c.groups["gf"], c.groups["ttmn"], c.groups["lc4"], c.groups["psi"],
                                     np.arange(0, c.n, 7, dtype=np.int32)])).astype(np.int32)
    s = c.subset(keep, min_weight=2.0)
    s.validate()
    assert s.n == keep.size
    assert np.array_equal(s.body_id, c.body_id[keep])
    assert np.array_equal(s.region, c.region[keep]) and np.array_equal(s.side, c.side[keep])
    assert s.types == c.types and np.array_equal(s.type_idx, c.type_idx[keep])
    assert np.all(s.weight >= 2.0)
    # every retained edge maps back onto an original edge with the same weight
    orig = {(int(p), int(q)): float(w) for p, q, w in zip(c.pre, c.post, c.weight)}
    for p, q, w in zip(s.pre, s.post, s.weight):
        assert orig[(int(keep[p]), int(keep[q]))] == float(w)
    expected_e = int(np.sum(np.isin(c.pre, keep) & np.isin(c.post, keep) & (c.weight >= 2.0)))
    assert s.e == expected_e == s.meta["e"]
    # groups are intersections, remapped
    assert np.array_equal(keep[s.groups["gf"]], c.groups["gf"])
    assert np.array_equal(keep[s.groups["ttmn_L"]], c.groups["ttmn_L"])
    assert s.groups["kc"].size == int(np.isin(c.groups["kc"], keep).sum())
    assert sum(s.meta["region_counts"].values()) == s.n
    assert s.meta["region_counts"] == count_regions(s.region)
    # GF -> TTMn pathway survives the subset
    gf, ttmn = s.groups["gf_L"][0], s.groups["ttmn_L"][0]
    idx, _ = s.csr().row(int(gf))
    assert ttmn in idx.tolist()
    with pytest.raises(ValueError):
        c.subset(np.array([0, c.n], dtype=np.int32))
