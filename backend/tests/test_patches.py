"""Tests for flybrain.connectome.patches (SPEC section c.5)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from flybrain.connectome.patches import (
    GAP_JUNCTIONS,
    LATERALITIES,
    GapJunction,
    apply_patches,
    is_applied,
    patch_pairs,
)
from flybrain.connectome.schema import Connectome


def _w(conn: Connectome, i: int, j: int) -> float | None:
    mask = (conn.pre == i) & (conn.post == j)
    return float(conn.weight[mask][0]) if mask.any() else None


def _only(conn: Connectome, key: str) -> int:
    idx = conn.groups[key]
    assert idx.size == 1, key
    return int(idx[0])


# --------------------------------------------------------------------------- table


def test_table_is_verbatim():
    assert GAP_JUNCTIONS == (
        GapJunction("DNp01", "TTMn", 300.0, "ipsi"),
        GapJunction("DNp01", "PSI", 200.0, "ipsi"),
        GapJunction("DNp01", "DNp01", 80.0, "contra"),
        GapJunction("LC4", "DNp01", 6.0, "ipsi"),
        GapJunction("LPLC2", "DNp01", 6.0, "ipsi"),
    )
    assert LATERALITIES == ("ipsi", "contra", "both")
    for gj in GAP_JUNCTIONS:
        assert gj.weight > 0 and gj.laterality in LATERALITIES


def test_gapjunction_is_frozen():
    with pytest.raises(Exception):
        GAP_JUNCTIONS[0].weight = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- apply


def test_apply_adds_and_increases_with_laterality(tiny_connectome):
    conn = tiny_connectome
    gf_l, gf_r = _only(conn, "gf_L"), _only(conn, "gf_R")
    ttmn_l, ttmn_r = _only(conn, "ttmn_L"), _only(conn, "ttmn_R")
    # the chemical GF -> TTMn count is the fixture's business (conftest, another workstream): assert the
    # DELTA the patch adds, never an absolute weight
    ipsi_before = {(gf_l, ttmn_l): _w(conn, gf_l, ttmn_l), (gf_r, ttmn_r): _w(conn, gf_r, ttmn_r)}
    assert all(v is not None and v > 0 for v in ipsi_before.values())  # the fixture has both ipsi pairs
    e0, synapses0 = conn.e, float(conn.weight.sum(dtype=np.float64))
    gf_gf_before = sum(1 for a, b in ((gf_l, gf_r), (gf_r, gf_l)) if _w(conn, a, b) is not None)

    patched = apply_patches(conn)
    patched.validate()
    assert patched is not conn
    # GF -> TTMn ipsi increased by 300, never contralateral
    for (a, b), w0 in ipsi_before.items():
        assert _w(patched, a, b) == w0 + 300.0
    assert _w(patched, gf_l, ttmn_r) == _w(conn, gf_l, ttmn_r) and _w(patched, gf_r, ttmn_l) == _w(conn, gf_r, ttmn_l)
    # GF <-> GF contralateral coupling created (80 on top of whatever the fixture had)
    for a, b in ((gf_l, gf_r), (gf_r, gf_l)):
        assert _w(patched, a, b) == (_w(conn, a, b) or 0.0) + 80.0
    # LC4 / LPLC2 -> GF ipsi +6, contra untouched
    for key in ("lc4", "lplc2"):
        for i in conn.groups[key].tolist():
            target = gf_l if conn.side[i] == -1 else gf_r
            other = gf_r if conn.side[i] == -1 else gf_l
            assert _w(patched, i, target) == (_w(conn, i, target) or 0.0) + 6.0
            assert _w(patched, i, other) == _w(conn, i, other)
    # bookkeeping
    recs = patched.meta["patches_applied"]
    assert len(recs) == 5
    assert [r["pre"] for r in recs] == ["DNp01", "DNp01", "DNp01", "LC4", "LPLC2"]
    assert all({"pre", "post", "weight", "laterality", "edges_added", "edges_increased"} <= set(r) for r in recs)
    assert recs[0]["edges_increased"] == len(ipsi_before) and recs[0]["edges_added"] == 0
    assert recs[2]["edges_added"] == 2 - gf_gf_before
    assert patched.e == e0 + sum(r["edges_added"] for r in recs)
    assert patched.meta["e"] == patched.e
    expected_syn = synapses0 + 2 * 300 + 2 * 200 + 2 * 80 + 6 * (conn.groups["lc4"].size + conn.groups["lplc2"].size)
    assert patched.meta["synapses"] == pytest.approx(expected_syn)
    assert float(patched.weight.sum(dtype=np.float64)) == pytest.approx(expected_syn)


def test_result_is_sorted_unique_no_self_loops(tiny_connectome):
    patched = apply_patches(tiny_connectome)
    assert patched.pre.dtype == np.int32 and patched.post.dtype == np.int32 and patched.weight.dtype == np.float32
    assert np.all(patched.pre[1:] >= patched.pre[:-1])
    key = patched.pre.astype(np.int64) * patched.n + patched.post
    assert np.unique(key).size == key.size
    assert not np.any(patched.pre == patched.post)
    assert np.all(patched.weight > 0)


def test_idempotent(tiny_connectome):
    p1 = apply_patches(tiny_connectome)
    p2 = apply_patches(p1)
    assert p2 is p1
    assert len(p2.meta["patches_applied"]) == 5
    p3 = apply_patches(p1, table=GAP_JUNCTIONS[:2])  # subset already recorded -> no-op too
    assert p3 is p1
    assert all(is_applied(p1, gj) for gj in GAP_JUNCTIONS)
    assert not any(is_applied(tiny_connectome, gj) for gj in GAP_JUNCTIONS)


def test_input_is_not_mutated(tiny_connectome):
    conn = tiny_connectome
    pre, post, weight = conn.pre.copy(), conn.post.copy(), conn.weight.copy()
    meta_e, patches = conn.meta["e"], list(conn.meta["patches_applied"])
    apply_patches(conn)
    assert np.array_equal(conn.pre, pre) and np.array_equal(conn.post, post) and np.array_equal(conn.weight, weight)
    assert conn.meta["e"] == meta_e and conn.meta["patches_applied"] == patches


def test_shares_neuron_arrays_and_groups(tiny_connectome):
    patched = apply_patches(tiny_connectome)
    for name in ("sign", "region", "side", "type_idx", "body_id", "nt"):
        assert getattr(patched, name) is getattr(tiny_connectome, name)
    assert patched.groups is tiny_connectome.groups and patched.types is tiny_connectome.types
    assert patched.name == tiny_connectome.name and patched.source == tiny_connectome.source and patched.n == tiny_connectome.n


def test_signed_csr_follows_presynaptic_sign(tiny_connectome):
    conn = tiny_connectome
    gf_l, ttmn_l = _only(conn, "gf_L"), _only(conn, "ttmn_L")
    inh = int(conn.groups["feed_pre_inh"][0])  # GNG015, GABA -> sign -1
    mn9 = int(conn.groups["feed_mn"][conn.side[conn.groups["feed_mn"]] == conn.side[inh]][0])
    assert conn.sign[inh] == -1.0 and conn.sign[gf_l] == 1.0
    table = GAP_JUNCTIONS + (GapJunction(conn.type_of(inh), "MN9", 5.0, "ipsi"),)
    gf_ttmn_before = _w(conn, gf_l, ttmn_l) or 0.0
    patched = apply_patches(conn, table)
    csr = patched.csr()
    idx, data = csr.row(gf_l)
    assert float(data[idx == ttmn_l][0]) == gf_ttmn_before + 300.0  # excitatory pre -> positive CSR data
    idx, data = csr.row(inh)
    assert float(data[idx == mn9][0]) == -((_w(conn, inh, mn9) or 0.0) + 5.0)
    assert float(patched.weight[(patched.pre == inh) & (patched.post == mn9)][0]) > 0  # stored weight stays unsigned


def test_missing_type_is_recorded_with_zero_edges(tiny_connectome):
    gj = GapJunction("NoSuchType", "DNp01", 5.0, "ipsi")
    patched = apply_patches(tiny_connectome, (gj,))
    assert patched.e == tiny_connectome.e
    assert np.array_equal(patched.weight, tiny_connectome.weight)
    rec = patched.meta["patches_applied"][-1]
    assert rec["edges_added"] == 0 and rec["edges_increased"] == 0 and rec["pre"] == "NoSuchType"
    assert is_applied(patched, gj)
    assert apply_patches(patched, (gj,)) is patched


def test_laterality_both_and_contra_pair_counts(tiny_connectome):
    conn = tiny_connectome
    lc4, dnp09 = conn.groups["lc4"], conn.groups["dn_freeze"]
    n_l = int((conn.side[lc4] == -1).sum())
    n_r = int((conn.side[lc4] == 1).sum())
    pp, qq = patch_pairs(conn, GapJunction("LC4", "DNp09", 3.0, "both"))
    assert pp.size == lc4.size * dnp09.size
    pp, qq = patch_pairs(conn, GapJunction("LC4", "DNp09", 3.0, "contra"))
    assert pp.size == n_l * int((conn.side[dnp09] == 1).sum()) + n_r * int((conn.side[dnp09] == -1).sum())
    assert np.all(conn.side[pp] == -conn.side[qq])
    pp, qq = patch_pairs(conn, GapJunction("LC4", "DNp09", 3.0, "ipsi"))
    assert np.all(conn.side[pp] == conn.side[qq])
    patched = apply_patches(conn, (GapJunction("LC4", "DNp09", 3.0, "both"),))
    rec = patched.meta["patches_applied"][-1]
    assert rec["edges_added"] + rec["edges_increased"] == lc4.size * dnp09.size


def test_unknown_side_matches_both_lateralities(tiny_connectome):
    conn = tiny_connectome
    conn.side = conn.side.copy()
    conn.side[conn.groups["gf"]] = 0
    pp_c, _ = patch_pairs(conn, GapJunction("DNp01", "DNp01", 80.0, "contra"))
    pp_i, _ = patch_pairs(conn, GapJunction("DNp01", "DNp01", 80.0, "ipsi"))
    assert pp_c.size == 2 and pp_i.size == 2  # both GF cells, no self-loop


def test_side_zero_is_wildcard_when_only_one_endpoint_is_unknown(tiny_connectome, caplog):
    # regression: a sided DNp01 paired with a TTMn whose side is 0 (somaSide 'M' or empty) got no patch
    conn = tiny_connectome
    conn.side = conn.side.copy()
    ttmn = conn.groups["ttmn"]
    gf_l, gf_r = _only(conn, "gf_L"), _only(conn, "gf_R")
    conn.side[ttmn] = 0
    conn.groups = dict(conn.groups)
    conn.groups["ttmn_L"] = conn.groups["ttmn_R"] = np.zeros(0, dtype=np.int32)
    with caplog.at_level(logging.WARNING, logger="flybrain.connectome.patches"):
        pp, qq = patch_pairs(conn, GAP_JUNCTIONS[0])  # DNp01 -> TTMn ipsi
    assert pp.size == 2 * ttmn.size and set(pp.tolist()) == {gf_l, gf_r} and set(qq.tolist()) == set(ttmn.tolist())
    pp_c, _ = patch_pairs(conn, GapJunction("DNp01", "TTMn", 300.0, "contra"))
    assert pp_c.size == 2 * ttmn.size
    msgs = [rec.getMessage() for rec in caplog.records if "wildcard" in rec.getMessage()]
    assert len(msgs) == 2  # one warning per lateral patch evaluated above (ipsi, contra)
    assert "DNp01->TTMn (ipsi)" in msgs[0] and f"{ttmn.size} TTMn cells have side 0" in msgs[0]
    assert all(ord(ch) < 128 for m in msgs for ch in m)
    patched = apply_patches(conn)
    rec = patched.meta["patches_applied"][0]
    assert rec["pre"] == "DNp01" and rec["edges_added"] + rec["edges_increased"] == 2 * ttmn.size
    for t in ttmn.tolist():
        assert _w(patched, gf_l, t) == (_w(conn, gf_l, t) or 0.0) + 300.0
        assert _w(patched, gf_r, t) == (_w(conn, gf_r, t) or 0.0) + 300.0
    # sided pairs are still strictly lateral: nothing under 'both' or 'ipsi' changes for the sided PSI cells
    psi = conn.groups["psi"]
    assert np.all(conn.side[psi] != 0)
    pp, qq = patch_pairs(conn, GAP_JUNCTIONS[1])
    assert np.all(conn.side[pp] == conn.side[qq])


def test_side_zero_wildcard_from_csv_export(tmp_path):
    """The reviewer's repro: a neuPrint export whose TTMn is 'M' / PSI has no somaSide keeps its escape proxy."""
    from flybrain.connectome.loaders import load_csv_dir

    (tmp_path / "neurons.csv").write_text(
        "bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt,predictedNt\n"
        "1,DNp01,DNp01(GF)_L,descending_neuron,,,,Traced,acetylcholine,\n"
        "2,DNp01,DNp01(GF)_R,descending_neuron,,,,Traced,acetylcholine,\n"
        "3,TTMn,TTMn_R,vnc_motor,,,M,Traced,glutamate,\n"
        "4,TTMn,TTMn_L,vnc_motor,,,l,Traced,glutamate,\n"
        "5,PSI,PSI,vnc_efferent,,,,Traced,,\n", encoding="utf-8")
    (tmp_path / "connections.csv").write_text("bodyId_pre,bodyId_post,weight\n1,4,45\n2,3,45\n", encoding="utf-8")
    c = load_csv_dir(tmp_path, min_weight=1, subset="all")
    assert dict(zip(c.body_id.tolist(), c.side.tolist())) == {1: -1, 2: 1, 3: 0, 4: -1, 5: 0}
    pp, qq = patch_pairs(c, GAP_JUNCTIONS[0])
    assert sorted(zip(c.body_id[pp].tolist(), c.body_id[qq].tolist())) == [(1, 3), (1, 4), (2, 3)]
    pp, qq = patch_pairs(c, GAP_JUNCTIONS[1])
    assert sorted(zip(c.body_id[pp].tolist(), c.body_id[qq].tolist())) == [(1, 5), (2, 5)]
    patched = apply_patches(c)
    recs = {(r["pre"], r["post"]): (r["edges_added"], r["edges_increased"]) for r in patched.meta["patches_applied"]}
    assert recs[("DNp01", "TTMn")] == (1, 2) and recs[("DNp01", "PSI")] == (2, 0) and recs[("DNp01", "DNp01")] == (2, 0)
    i2, i3 = int(np.searchsorted(c.body_id, 2)), int(np.searchsorted(c.body_id, 3))
    assert _w(patched, i2, i3) == 345.0


def test_self_pairs_never_created():
    # a single-cell type patched onto itself yields no edge at all
    from tests.conftest import build_tiny_connectome

    conn = build_tiny_connectome()
    apl = conn.groups["apl"]
    pp, qq = patch_pairs(conn, GapJunction("APL", "APL", 10.0, "both"))
    assert pp.size == apl.size * (apl.size - 1)
    assert not np.any(pp == qq)


def test_bad_table_entries_raise(tiny_connectome):
    with pytest.raises(ValueError, match="laterality"):
        apply_patches(tiny_connectome, (GapJunction("DNp01", "TTMn", 1.0, "sideways"),))
    with pytest.raises(ValueError, match="weight"):
        apply_patches(tiny_connectome, (GapJunction("DNp01", "TTMn", 0.0, "ipsi"),))
    with pytest.raises(ValueError, match="laterality"):
        patch_pairs(tiny_connectome, GapJunction("DNp01", "TTMn", 1.0, "x"))


def test_empty_table_returns_input(tiny_connectome):
    assert apply_patches(tiny_connectome, ()) is tiny_connectome


def test_later_patch_sees_earlier_patch_edges(tiny_connectome):
    conn = tiny_connectome
    gf_l, gf_r = _only(conn, "gf_L"), _only(conn, "gf_R")
    had = sum(1 for a, b in ((gf_l, gf_r), (gf_r, gf_l)) if _w(conn, a, b) is not None)
    table = (GapJunction("DNp01", "DNp01", 80.0, "contra"), GapJunction("DNp01", "DNp01", 20.0, "contra"))
    patched = apply_patches(conn, table)
    r0, r1 = patched.meta["patches_applied"][-2:]
    assert r0["edges_added"] == 2 - had and r1["edges_added"] == 0 and r1["edges_increased"] == 2
    assert _w(patched, gf_l, gf_r) == (_w(conn, gf_l, gf_r) or 0.0) + 100.0
