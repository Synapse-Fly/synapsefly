"""Tests for flybrain.connectome.groups (SPEC c.4): regex table, cascades, resolution, partition."""

from __future__ import annotations

import dataclasses
import re

import numpy as np
import pytest

from flybrain.connectome.groups import (
    GROUP_REGEX,
    READOUTS,
    SIDED,
    STAR_TYPES,
    fold_sided_counts,
    readout_partition,
    readout_sizes,
    region_of,
    resolve_groups,
    side_of,
    validate_groups,
)
from flybrain.connectome.schema import REGION_ID, Connectome

# --------------------------------------------------------------------------- static tables


def test_tables_are_consistent():
    for name, rx in GROUP_REGEX.items():
        re.compile(rx)  # every regex compiles
        assert name == name.lower() or name in ("aipg",)  # snake_case keys
    assert len(READOUTS) == len(set(READOUTS))
    assert set(READOUTS) <= set(GROUP_REGEX)
    assert SIDED <= set(GROUP_REGEX)
    assert "grn_sugar_labellar" not in READOUTS and "lc_loom" not in READOUTS
    assert READOUTS[0] == "gf" and READOUTS[-1] == "leg_premotor"
    assert len(STAR_TYPES) == 31 and "DNp01" in STAR_TYPES


@pytest.mark.parametrize(
    "group, label",
    [
        ("grn_sugar", "LgLG3"), ("grn_sugar_labellar", "PhG1c"), ("grn_bitter", "LB1e"), ("jo_aud", "JO-A-unclear"),
        ("jo_groom", "JO-EV2"), ("photoreceptor", "R1-R6"), ("photoreceptor", "R7y"), ("lamina", "L5"),
        ("alpn", "VP2+Z_lvPN"), ("alpn", "DM1_lPN"), ("alln", "il3LN6"), ("pen", "PEN_b(PEN2)"), ("ring", "ER4d"),
        ("dfb_sleep", "FB6A"), ("dn_halt", "DNg74_b"), ("flight_dn", "DNg02_g"),
        ("wing_power", "DLMn a, b"), ("wing_steer", "iii3 MN"), ("wing_steer", "MNwm35"), ("song_mn", "hg4 MN"),
        ("leg_mn", "Acc. ti flexor MN"), ("leg_mn", "Tergotr. MN"), ("leg_mn", "Pleural remotor/abductor MN"),
        ("song_vnc", "vPR9_c"), ("p1", "pC1_14a"), ("mal", "mAL_m8"), ("orn", "ORN_DM1"), ("kc", "KCg-m"),
    ],
)
def test_regex_positive(group, label):
    assert re.fullmatch(GROUP_REGEX[group], label) is not None


@pytest.mark.parametrize(
    "group, label",
    [
        ("gf", "DNp010"), ("gf", "DNp0"), ("alpn", "ORN_DM1"), ("lamina", "L6"), ("lamina", "LC4"),
        ("photoreceptor", "R7"), ("pen", "PEN_a"), ("wing_steer", "hi1 MN"), ("wing_steer", "tpN MN"),
        ("feed_mn", "MN9x"), ("song_vnc", "vPR9"), ("kc", "kc"), ("dng100", "DNg1000"),
    ],
)
def test_regex_negative(group, label):
    assert re.fullmatch(GROUP_REGEX[group], label) is None


# --------------------------------------------------------------------------- region_of / side_of


@pytest.mark.parametrize(
    "superclass, cls, type_, expected",
    [
        ("ol_intrinsic", "", "Mi1", "optic_lobe"),
        ("visual_projection", "", "LC4", "optic_lobe"),
        ("cb_intrinsic", "visual", "X", "optic_lobe"),
        ("cb_sensory", "olfactory", "ORN_DM1", "antennal_lobe"),
        ("cb_intrinsic", "ALPN", "DM1_lPN", "antennal_lobe"),
        ("cb_intrinsic", "", "ORN_VA1", "antennal_lobe"),
        ("cb_intrinsic", "Kenyon_Cell", "KCg-m", "mushroom_body"),
        ("cb_intrinsic", "MBON", "MBON01", "mushroom_body"),
        ("cb_intrinsic", "", "APL", "mushroom_body"),
        ("cb_intrinsic", "CX", "EPG", "central_complex"),
        ("cb_motor", "", "MN9", "sez"),
        ("cb_sensory", "gustatory", "LB3b", "sez"),
        ("cb_intrinsic", "SEZPN", "X", "sez"),
        ("cb_intrinsic", "", "GNG108", "sez"),
        ("cb_intrinsic", "", "PRW046", "sez"),
        ("cb_intrinsic", "", "MN11D", "sez"),
        ("descending_neuron", "", "DNp01", "descending_motor"),
        ("vnc_motor", "", "TTMn", "descending_motor"),
        ("vnc_efferent", "", "PSI", "descending_motor"),
        ("vnc_intrinsic", "", "GFC2", "vnc"),
        ("vnc_sensory", "", "LgLG3", "vnc"),
        ("ascending_neuron", "", "AN13B002", "vnc"),
        ("cb_sensory", "", "JO-A1", "central_other"),
        ("cb_intrinsic", "", "pC1_14a", "central_other"),
        (None, None, None, "central_other"),
        ("", "", "", "central_other"),
    ],
)
def test_region_of_cascade(superclass, cls, type_, expected):
    assert region_of(superclass, cls, type_) == REGION_ID[expected]


def test_region_of_rule_order():
    # rule 1 beats everything; rule 5 (SEZ type) beats rule 6 (descending superclass); rule 6 beats 7
    assert region_of("ol_intrinsic", "CX", "GNG001") == REGION_ID["optic_lobe"]
    assert region_of("descending_neuron", "", "GNG001") == REGION_ID["sez"]
    assert region_of("vnc_motor", "", "X") == REGION_ID["descending_motor"]


@pytest.mark.parametrize(
    "soma, instance, expected",
    [
        ("L", None, -1), ("R", None, 1), ("M", None, 0), ("", None, 0), (None, None, 0),
        (" l ", None, -1), ("r", None, 1), ("", "DNp01_L", -1), (None, "DNp01_R", 1), ("M", "DNp01_R", 0),
        ("X", "foo", 0), ("", "DNp01", 0),
    ],
)
def test_side_of(soma, instance, expected):
    assert side_of(soma, instance) == expected


# --------------------------------------------------------------------------- resolve_groups


def test_resolve_groups_keys_and_dtypes(tiny_connectome: Connectome):
    c = tiny_connectome
    g = resolve_groups(c.types, c.type_idx, c.side)
    expected_keys = set(GROUP_REGEX) | {s + "_L" for s in SIDED} | {s + "_R" for s in SIDED}
    assert set(g) == expected_keys
    for name, idx in g.items():
        assert idx.dtype == np.int32 and idx.ndim == 1, name
        assert np.all(np.diff(idx) > 0), name  # sorted, unique
        if idx.size:
            assert idx.min() >= 0 and idx.max() < c.n
    assert set(c.groups) == expected_keys
    for name in g:
        assert np.array_equal(g[name], c.groups[name]), name


def test_resolve_groups_missing_groups_are_empty():
    types = ["", "DNp01", "KCg-m"]
    type_idx = np.array([0, 1, 1, 2, 0], dtype=np.int32)
    side = np.array([0, -1, 1, 1, 0], dtype=np.int8)
    g = resolve_groups(types, type_idx, side)
    assert g["gf"].tolist() == [1, 2] and g["gf_L"].tolist() == [1] and g["gf_R"].tolist() == [2]
    assert g["kc"].tolist() == [3]
    assert g["epg"].size == 0 and g["epg"].dtype == np.int32
    assert g["epg_L"].size == 0 and g["epg_R"].size == 0
    assert g["leg_mn"].size == 0


def test_resolve_groups_side_zero_is_in_union_only():
    g = resolve_groups(["DNp01"], np.zeros(3, dtype=np.int32), np.array([-1, 1, 0], dtype=np.int8))
    assert g["gf"].tolist() == [0, 1, 2]
    assert g["gf_L"].tolist() == [0] and g["gf_R"].tolist() == [1]


def test_every_readout_non_empty(tiny_connectome: Connectome):
    empty = [name for name in READOUTS if tiny_connectome.groups[name].size == 0]
    assert empty == []
    for name in ("lc_loom", "escape_vnc", "grn_sugar_labellar", "song_mn", "pip10", "dms2", "b1", "i1", "hg1"):
        assert tiny_connectome.groups[name].size > 0, name


def test_sided_union_equals_unsuffixed(tiny_connectome: Connectome):
    g = tiny_connectome.groups
    for name in SIDED:
        left, right = g[name + "_L"], g[name + "_R"]
        assert np.intersect1d(left, right).size == 0, name
        assert np.array_equal(np.union1d(left, right), g[name]), name
        assert np.all(tiny_connectome.side[left] == -1) and np.all(tiny_connectome.side[right] == 1), name
        if g[name].size:
            assert left.size > 0 and right.size > 0, name  # fixture puts every sided type on both sides


# --------------------------------------------------------------------------- readout_partition


def test_readout_partition_is_a_partition(tiny_connectome: Connectome):
    c = tiny_connectome
    pop_id, names = readout_partition(c.groups, c.n)
    assert pop_id.dtype == np.int16 and pop_id.shape == (c.n,)
    assert len(names) == len(set(names))
    # names in READOUTS order, sided ones expanded to name, name_L, name_R
    expected: list[str] = []
    for name in READOUTS:
        expected.append(name)
        if name in SIDED:
            expected.extend((name + "_L", name + "_R"))
    assert names == expected
    assert pop_id.max() < len(names) and pop_id.min() >= -1
    # independent oracle: naive first-match-wins loop over READOUTS, sided members split by side
    expected_id = np.full(c.n, -1, dtype=np.int64)
    for name in READOUTS:
        for i in c.groups[name].tolist():
            if expected_id[i] >= 0:
                continue  # claimed by an earlier readout -> at most one readout per neuron
            if name in SIDED and c.side[i] == -1:
                expected_id[i] = names.index(name + "_L")
            elif name in SIDED and c.side[i] == 1:
                expected_id[i] = names.index(name + "_R")
            else:
                expected_id[i] = names.index(name)
    assert np.array_equal(pop_id.astype(np.int64), expected_id)
    # every neuron in any readout group is assigned; every other neuron is -1
    in_any = np.zeros(c.n, dtype=bool)
    for name in READOUTS:
        in_any[c.groups[name]] = True
    assert np.array_equal(in_any, pop_id >= 0)
    # sided readouts: the unsuffixed id is unused because the fixture has no midline member there
    for name in READOUTS:
        if name in SIDED:
            assert not np.any(pop_id == names.index(name)), name
            l_id, r_id = names.index(name + "_L"), names.index(name + "_R")
            assert np.all(c.side[pop_id == l_id] == -1) and np.all(c.side[pop_id == r_id] == 1), name


def test_readout_partition_first_match_wins(tiny_connectome: Connectome):
    c = tiny_connectome
    pop_id, names = readout_partition(c.groups, c.n)
    dng100 = c.where("DNg100")
    assert np.all(np.isin(pop_id[dng100], [names.index("dng100_L"), names.index("dng100_R")]))
    others = c.where(r"(DNge053|DNg97)")
    assert np.all(np.isin(pop_id[others], [names.index("dn_fwd_L"), names.index("dn_fwd_R")]))
    assert np.all(np.isin(c.groups["dn_fwd"], np.concatenate([dng100, others])))
    # all 107 ids resolve to a non-empty population after folding L/R into the unsuffixed name
    counts = np.bincount(pop_id[pop_id >= 0], minlength=len(names))
    folded = fold_sided_counts(counts, names)
    unsuffixed = [i for i, nm in enumerate(names) if not nm.endswith(("_L", "_R"))]
    assert np.all(folded[unsuffixed] > 0)
    assert folded[names.index("gf")] == counts[names.index("gf_L")] + counts[names.index("gf_R")] == c.groups["gf"].size
    sizes = readout_sizes(pop_id, names)
    assert sizes.dtype == np.int64 and np.array_equal(sizes, folded)
    assert sizes[names.index("dn_fwd")] == 4 and c.groups["dn_fwd"].size == 6  # DNg100 pair went to dng100
    assert sizes[names.index("dng100")] == 2


def test_readout_partition_midline_member_gets_unsuffixed_id():
    g = resolve_groups(["DNp01", "TTMn"], np.array([0, 0, 0, 1], dtype=np.int32),
                       np.array([-1, 1, 0, -1], dtype=np.int8))
    pop_id, names = readout_partition(g, 4)
    assert pop_id.tolist() == [names.index("gf_L"), names.index("gf_R"), names.index("gf"), names.index("ttmn_L")]
    counts = fold_sided_counts(np.bincount(pop_id, minlength=len(names)), names)
    assert counts[names.index("gf")] == 3 and counts[names.index("ttmn")] == 1
    assert readout_sizes(pop_id, names)[names.index("gf")] == 3


# --------------------------------------------------------------------------- validate_groups


def test_validate_groups_ok_and_raises(tiny_connectome: Connectome):
    c = tiny_connectome
    sizes = validate_groups(c)
    assert set(sizes) == set(c.groups)
    assert sizes["gf"] == 2 and sizes["gf_L"] == 1 and sizes["lc_loom"] == 12
    with pytest.raises(ValueError, match="no_such_group"):
        validate_groups(c, required=("gf", "no_such_group"))
    fields = {f.name: getattr(c, f.name) for f in dataclasses.fields(c) if f.init}
    fields["groups"] = {**c.groups, "gf": np.zeros(0, dtype=np.int32), "feed_mn": np.zeros(0, dtype=np.int32)}
    broken = Connectome(**fields)
    with pytest.raises(ValueError) as ei:
        validate_groups(broken)
    assert "gf" in str(ei.value) and "feed_mn" in str(ei.value) and "epg" not in str(ei.value)
