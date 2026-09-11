"""Tests for flybrain.connectome.loaders (SPEC section c.7) plus the two data scripts (SPEC h.4).

The SPEC h.2 names (``test_nt_normalise_aliases``, ``test_detect_schema``, ``test_load_neuprint_small``,
``test_load_codex_small``, ``test_subset_core_keeps_functional``, ``test_load_connectome_dispatch``,
``test_cache_roundtrip``) are normative and kept verbatim; the remaining tests are additional.

Fixture contract (SPEC h.1): ``neuprint_small`` = 40 neurons (consensusNt spellings ACH / glut / GABA /
unc / '') and 120 connection rows = 115 unique (pre, post) pairs + 3 duplicate-pair rows + 2 rows to
unknown bodies; ``codex_small`` = the same 40 neurons in the FlyWire-Codex schema with 130 rows
(the 115 pairs, 15 of them split into two per-neuropil rows). The third schema (Codex MaleCNS
mirror, ``'Root ID'`` headers) is generated at test time from the same 40 neurons (``mcns_dir``).
"""

from __future__ import annotations

import csv
import gc
import gzip
import importlib.util
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from flybrain.connectome import cache as cache_mod
from flybrain.connectome import loaders as loaders_mod
from flybrain.connectome.cache import cache_key, load_cached, store_cached
from flybrain.connectome.groups import GROUP_REGEX, READOUTS, validate_groups
from flybrain.connectome.loaders import (
    NT_ALIASES,
    NT_SIGN,
    Schema,
    detect_schema,
    load_connectome,
    load_csv_dir,
    nt_normalise,
    nt_sign,
)
from flybrain.connectome.schema import NT_VALUES, REGION_ID, REGIONS
from tests.conftest import build_tiny_connectome

FIXTURES = Path(__file__).resolve().parent / "fixtures"
NP_DIR = FIXTURES / "neuprint_small"
CX_DIR = FIXTURES / "codex_small"
REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
B = 720575940610000000  # codex_small root_id = B + neuPrint bodyId
FOREIGN_IDS = (987654321, 123456789)  # the two unknown bodies referenced by neuprint_small/connections.csv
DUP_PAIRS = {(20082, 20081): 5.0, (20072, 20036): 10.0, (20064, 20034): 10.0}  # summed duplicate rows
CODEX_NT = {"acetylcholine": "ACH", "glutamate": "GLUT", "gaba": "GABA", "unknown": ""}


# --------------------------------------------------------------------------- helpers


def _rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _index_of(conn, body_id: int) -> int:
    pos = int(np.searchsorted(conn.body_id, body_id))
    assert conn.body_id[pos] == body_id
    return pos


def _w(conn, pre_body: int, post_body: int) -> float | None:
    i, j = _index_of(conn, pre_body), _index_of(conn, post_body)
    mask = (conn.pre == i) & (conn.post == j)
    return float(conn.weight[mask][0]) if mask.any() else None


def _type_index(conn, type_name: str, side: int | None = None) -> int:
    idx = conn.where(re.escape(type_name), side=side)
    assert idx.size >= 1, type_name
    return int(idx[0])


def _summed_pairs() -> dict[tuple[int, int], int]:
    """(pre, post) -> summed weight of the neuprint_small rows among known bodies (the loader's contract)."""
    ids = {int(r["bodyId"]) for r in _rows(NP_DIR / "neurons.csv")}
    out: dict[tuple[int, int], int] = {}
    for r in _rows(NP_DIR / "connections.csv"):
        a, b = int(r["bodyId_pre"]), int(r["bodyId_post"])
        if a in ids and b in ids and a != b:
            out[(a, b)] = out.get((a, b), 0) + int(r["weight"])
    return out


def _assert_same_graph(a, b, *, same_body_ids: bool = True) -> None:
    assert a.n == b.n and a.e == b.e
    for name in ("pre", "post", "weight", "sign", "region", "side", "type_idx"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.types == b.types and [str(x) for x in a.nt] == [str(x) for x in b.nt]
    if same_body_ids:
        assert np.array_equal(a.body_id, b.body_id)
    for k in a.groups:
        assert np.array_equal(a.groups[k], b.groups[k]), k


def make_settings(tmp_path: Path, **over) -> SimpleNamespace:
    base = dict(
        connectome_source="csv",
        connectome_dir=NP_DIR,
        connectome_name="",
        n_neurons=20_000,
        mean_outdeg=25,
        synth_weights="calibrated",
        subset="core",
        min_weight=3,
        seed=0,
        data_dir=tmp_path / "data",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_gz(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())


def _append_gz(path: Path, rows: list[list[Any]]) -> None:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        text = fh.read()
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write(text + buf.getvalue())


def _neuron_nt_label(r: dict[str, str]) -> str:
    cons = r["consensusNt"].strip()
    return nt_normalise(cons if cons and cons.lower() != "none" else r["predictedNt"])


def build_mcns_dir(target: Path) -> Path:
    """The Codex MaleCNS-mirror layout ('Root ID' headers + connections_princeton.csv.gz) of the same 40 neurons."""
    target.mkdir(parents=True, exist_ok=True)
    neurons = _rows(NP_DIR / "neurons.csv")
    _write_gz(
        target / "neurons.csv.gz",
        ["Root ID", "Super Class", "Class", "Soma side", "Primary Cell Type", "Predicted NT type"],
        [[r["bodyId"], r["superclass"], r["class"], r["somaSide"], r["type"], CODEX_NT[_neuron_nt_label(r)]] for r in neurons],
    )
    edges = _rows(CX_DIR / "connections.csv.gz")  # 130 per-neuropil rows, ids shifted by B
    _write_gz(
        target / "connections_princeton.csv.gz",
        ["Pre Root ID", "Post Root ID", "Neuropil", "Syn Count", "NT Type"],
        [[int(r["pre_root_id"]) - B, int(r["post_root_id"]) - B, r["neuropil"], r["syn_count"], r["nt_type"]] for r in edges],
    )
    return target


@pytest.fixture(scope="module")
def mcns_dir(tmp_path_factory) -> Path:
    return build_mcns_dir(tmp_path_factory.mktemp("codex_mcns_small"))


@pytest.fixture(scope="module")
def np_all():
    return load_csv_dir(NP_DIR, min_weight=1, subset="all")


# --------------------------------------------------------------------------- NT tables (SPEC h.2)


def test_nt_normalise_aliases():
    # every NT_ALIASES key, any case / surrounding whitespace, incl. '', 'unc', 'GLUT', 'Glu', 'His' -> a valid sign
    for key, canonical in NT_ALIASES.items():
        for variant in (key, key.upper(), key.capitalize(), f" {key} "):
            assert nt_normalise(variant) == canonical, variant
            assert nt_sign(variant) == NT_SIGN[canonical] and isinstance(nt_sign(variant), float)
    assert nt_normalise("") == "unknown" and nt_sign("") == 1.0
    assert nt_normalise("unc") == "unknown" and nt_sign("unc") == 1.0
    assert nt_normalise("GLUT") == "glutamate" and nt_sign("GLUT") == -1.0
    assert nt_normalise("Glu") == "glutamate" and nt_sign("Glu") == -1.0
    assert nt_normalise("His") == "histamine" and nt_sign("His") == -1.0
    assert nt_normalise(None) == "unknown" and nt_normalise("something-new") == "unknown" and nt_normalise("NaN") == "unknown"
    assert set(NT_SIGN) == set(NT_VALUES) and set(NT_ALIASES.values()) == set(NT_VALUES)
    assert all(NT_SIGN[v] in (1, -1) for v in NT_VALUES)


# --------------------------------------------------------------------------- detect_schema (SPEC h.2)


def test_detect_schema(mcns_dir, tmp_path):
    assert detect_schema(NP_DIR) is Schema.NEUPRINT
    assert detect_schema(CX_DIR) is Schema.CODEX_FAFB
    assert detect_schema(mcns_dir) is Schema.CODEX_MCNS
    assert Schema.NEUPRINT == "neuprint" and Schema.CODEX_FAFB.value == "codex_fafb" and Schema.CODEX_MCNS.value == "codex_mcns"
    with pytest.raises(ValueError):
        detect_schema(tmp_path)  # empty dir
    with pytest.raises(ValueError):
        detect_schema(tmp_path / "missing")
    (tmp_path / "neurons.csv").write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        detect_schema(tmp_path)


def test_detect_schema_gz_neuprint(tmp_path):
    with gzip.open(tmp_path / "neurons.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write((NP_DIR / "neurons.csv").read_text(encoding="utf-8"))
    assert detect_schema(tmp_path) is Schema.NEUPRINT


def test_detect_schema_mixed_directory_prefers_neuprint(tmp_path):
    """A directory holding BOTH schemas resolves to neuPrint (SPEC c.7 probes neurons before classification)."""
    d = _copy_fixture(tmp_path)
    shutil.copy(CX_DIR / "classification.csv.gz", d / "classification.csv.gz")
    assert detect_schema(d) is Schema.NEUPRINT
    conn = load_csv_dir(d, min_weight=1, subset="all")
    assert conn.n == 40 and conn.meta["build_args"]["files"] == ["neurons.csv", "connections.csv"]
    assert conn.meta["license"] == "CC-BY 4.0 (MaleCNS)"
    # a Codex download (root_id neurons.csv.gz, no bodyId anywhere) is still detected as Codex
    cx = tmp_path / "cx"
    shutil.copytree(CX_DIR, cx)
    assert detect_schema(cx) is Schema.CODEX_FAFB


# --------------------------------------------------------------------------- neuPrint fixture


def test_fixture_files_have_contract_shape():
    neurons, edges = _rows(NP_DIR / "neurons.csv"), _rows(NP_DIR / "connections.csv")
    assert len(neurons) == 40 and len(edges) == 120
    assert list(neurons[0]) == ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "status",
                                "consensusNt", "predictedNt"]
    assert list(edges[0]) == ["bodyId_pre", "bodyId_post", "weight"]
    assert all(r["status"] == "Traced" for r in neurons)
    assert all(r["type"] for r in neurons)  # no untyped rows
    assert any(r["type"] == "DLMn a, b" for r in neurons)  # quoted comma survives csv round trip
    # SPEC h.1: abbreviated consensusNt spellings, 3 duplicate pairs, 2 edges to unknown bodies
    assert {"ACH", "glut", "GABA", "unc", ""} <= {r["consensusNt"] for r in neurons}
    ids = {int(r["bodyId"]) for r in neurons}
    pairs = [(int(r["bodyId_pre"]), int(r["bodyId_post"])) for r in edges]
    foreign = [p for p in pairs if p[0] not in ids or p[1] not in ids]
    assert len(foreign) == 2 and {x for p in foreign for x in p if x not in ids} == set(FOREIGN_IDS)
    inside = [p for p in pairs if p not in foreign]
    assert len(inside) - len(set(inside)) == 3 and len(set(inside)) == 115
    assert set(DUP_PAIRS) == {p for p in set(inside) if inside.count(p) == 2}
    # codex_small: the same 40 neurons, 130 rows with per-neuropil duplicates
    cls = _rows(CX_DIR / "classification.csv.gz")
    assert len(cls) == 40 and {int(r["root_id"]) - B for r in cls} == ids
    cx_edges = _rows(CX_DIR / "connections.csv.gz")
    assert len(cx_edges) == 130
    cx_pairs = [(int(r["pre_root_id"]) - B, int(r["post_root_id"]) - B) for r in cx_edges]
    assert set(cx_pairs) == set(inside) and len(cx_pairs) - len(set(cx_pairs)) == 15
    for f in ("classification", "consolidated_cell_types", "neurons", "connections"):
        raw = gzip.decompress((CX_DIR / f"{f}.csv.gz").read_bytes())
        assert all(b < 128 for b in raw), f  # ASCII fixtures


def test_load_neuprint_small(np_all):
    conn = np_all
    conn.validate()
    assert conn.source == "csv" and conn.n == 40 and conn.e == 115
    assert conn.meta["build_args"]["e_file"] == 120 and conn.meta["build_args"]["e_inside"] == 118
    # duplicate (pre, post) rows are summed
    for (a, b), wsum in DUP_PAIRS.items():
        assert _w(conn, a, b) == wsum
    # edges to unknown bodies are dropped (and never create neurons)
    assert not set(FOREIGN_IDS) & set(conn.body_id.tolist())
    assert conn.meta["synapses"] == pytest.approx(sum(_summed_pairs().values()))
    # min_weight=3 pruning
    pruned = load_csv_dir(NP_DIR, min_weight=3, subset="all")
    expected = sum(1 for w in _summed_pairs().values() if w >= 3)
    assert pruned.e == expected and 0 < expected < 115 and np.all(pruned.weight >= 3) and pruned.meta["e"] == expected
    assert _w(pruned, 20082, 20081) == 5.0  # 3 + 2 survives only because the duplicate rows were summed first
    # groups resolve
    assert set(conn.body_id[conn.groups["gf"]].tolist()) == {10001, 10010}
    assert set(conn.body_id[conn.groups["feed_mn"]].tolist()) == {10331, 16949}
    assert conn.meta["license"] == "CC-BY 4.0 (MaleCNS)" and "MaleCNS" in conn.meta["citation"]


def test_neuprint_load_basic(np_all):
    conn = np_all
    assert conn.meta["weights_mode"] == "real" and conn.meta["patches_applied"] == [] and conn.meta["gain_default"] == 0.65
    assert conn.meta["build_args"]["schema"] == "neuprint" and conn.meta["build_args"]["nt_from_edges"] == 0
    assert np.all(conn.body_id[1:] > conn.body_id[:-1])  # ordered by external id
    assert conn.pre.dtype == np.int32 and conn.weight.dtype == np.float32 and conn.body_id.dtype == np.int64
    assert conn.nt.dtype == object and all(v in NT_VALUES for v in conn.nt)
    assert "DLMn a, b" in conn.types and "" not in conn.types
    assert conn.name == "neuprint-neuprint_small-40-all-w1"
    assert sum(conn.meta["region_counts"].values()) == 40
    assert all(conn.meta["region_counts"][r] > 0 for r in REGIONS)  # every region represented


def test_neuprint_groups_and_sides(np_all):
    conn = np_all
    assert conn.body_id[conn.groups["gf_L"]].tolist() == [10010] and conn.body_id[conn.groups["gf_R"]].tolist() == [10001]
    assert set(conn.body_id[conn.groups["steer_a02"]].tolist()) == {10360, 523769}
    assert conn.body_id[conn.groups["dng100_L"]].tolist() == [10045]
    assert set(conn.body_id[conn.groups["ttmn"]].tolist()) == {804642, 800146}
    sizes = validate_groups(conn)
    assert sizes["lc_loom"] == 4 and sizes["wing_power"] == 2 and sizes["p1"] == 2 and sizes["escape_vnc"] == 3
    assert sizes["orn"] == 1 and sizes["kc"] == 1 and sizes["dms2"] == 1 and sizes["sugar2_exc"] == 2
    assert set(conn.groups) >= set(GROUP_REGEX)
    assert conn.side[_index_of(conn, 10001)] == 1 and conn.side[_index_of(conn, 10010)] == -1
    assert int(np.sum(conn.side == 0)) == 0


@pytest.mark.parametrize(
    "type_name,region",
    [("DNp01", "descending_motor"), ("DNg02_a", "descending_motor"), ("TTMn", "descending_motor"),
     ("PSI", "descending_motor"), ("DLMn a, b", "descending_motor"), ("MN9", "sez"), ("LB3b", "sez"),
     ("GNG215", "sez"), ("LC4", "optic_lobe"), ("LPLC2", "optic_lobe"), ("EPG", "central_complex"),
     ("PFL3", "central_complex"), ("pC1_14a", "central_other"), ("KCg-m", "mushroom_body"),
     ("ORN_DM1", "antennal_lobe"), ("dMS2", "vnc")],
)
def test_neuprint_region_cascade(np_all, type_name, region):
    i = _type_index(np_all, type_name)
    assert int(np_all.region[i]) == REGION_ID[region]


@pytest.mark.parametrize(
    "body,nt,sign",
    [(10001, "acetylcholine", 1.0), (10010, "acetylcholine", 1.0),   # 'ACH' and 'acetylcholine' spellings
     (20035, "gaba", -1.0), (20036, "gaba", -1.0),                   # 'GABA' / 'gaba'
     (10331, "glutamate", -1.0), (800146, "glutamate", -1.0),        # 'glut' / 'glutamate'
     (20001, "unknown", 1.0),                                        # 'unc' with empty predictedNt
     (20041, "acetylcholine", 1.0)],                                 # consensus '' -> predictedNt
)
def test_neuprint_nt_priority_and_sign(np_all, body, nt, sign):
    i = _index_of(np_all, body)
    assert np_all.nt[i] == nt and np_all.sign[i] == sign


def test_neuprint_edge_values_and_laterality(np_all):
    conn = np_all
    assert _w(conn, 20011, 10010) == 50.0 and _w(conn, 20012, 10001) == 50.0        # LC4 -> GF ipsi [V] mean
    assert _w(conn, 20013, 10010) == 26.0                                             # LPLC2 -> GF
    assert _w(conn, 10010, 804642) == 45.0 and _w(conn, 10001, 800146) == 45.0      # GF -> TTMn
    assert _w(conn, 20063, 10360) == 31.0 and _w(conn, 20064, 523769) == 31.0       # PFL3 -> DNa02 contralateral
    assert _w(conn, 20063, 523769) is None                                            # never ipsilateral in the fixture
    assert _w(conn, 20033, 10331) == 190.0 and _w(conn, 20035, 10331) == 239.0      # GNG108 / GNG015 -> MN9
    csr = conn.csr()
    idx, data = csr.row(_index_of(conn, 20035))
    assert float(data[idx == _index_of(conn, 10331)][0]) == -239.0                   # GABA pre -> negative signed data


def test_min_weight_validation():
    with pytest.raises(ValueError):
        load_csv_dir(NP_DIR, min_weight=0, subset="all")
    with pytest.raises(ValueError):
        load_csv_dir(NP_DIR, subset="some")


def test_gz_variant_matches_plain(tmp_path, np_all):
    for name in ("neurons", "connections"):
        with open(NP_DIR / f"{name}.csv", "rb") as src, gzip.open(tmp_path / f"{name}.csv.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
    conn = load_csv_dir(tmp_path, min_weight=1, subset="all")
    _assert_same_graph(conn, np_all)
    assert conn.meta["build_args"]["files"] == ["neurons.csv.gz", "connections.csv.gz"]


def _copy_fixture(tmp_path: Path) -> Path:
    d = tmp_path / "np"
    shutil.copytree(NP_DIR, d)
    return d


def test_status_filter_and_dangling_edges(tmp_path):
    d = _copy_fixture(tmp_path)
    with open(d / "neurons.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([99999, "DNp01", "DNp01_L", "descending_neuron", "", "", "L", "Orphan", "acetylcholine", ""])
        w.writerow([99998, "", "", "cb_intrinsic", "", "", "", "", "", ""])   # empty status is kept, untyped, no side
        w.writerow([10001, "DNp01", "DNp01(GF)_R", "descending_neuron", "", "", "R", "Traced", "acetylcholine", ""])  # duplicate id
    with open(d / "connections.csv", "a", encoding="utf-8", newline="") as fh:
        fh.write("99999,10001,50\n10001,99999,50\n10001,12345678,9\n10001,10001,7\n")  # orphan, unknown id, self-loop
    conn = load_csv_dir(d, min_weight=1, subset="all")
    conn.validate()
    assert conn.n == 41 and conn.e == 115
    assert 99999 not in conn.body_id.tolist() and 99998 in conn.body_id.tolist()
    i = _index_of(conn, 99998)
    assert conn.type_of(i) == "" and conn.side[i] == 0 and conn.nt[i] == "unknown"
    assert conn.meta["build_args"]["n_untraced"] == 1 and conn.meta["build_args"]["n_file"] == 43
    assert conn.groups["gf"].size == 2  # the duplicate id row did not create a third GF


def test_duplicate_edge_rows_are_summed(tmp_path):
    d = _copy_fixture(tmp_path)
    with open(d / "connections.csv", "a", encoding="utf-8", newline="") as fh:
        fh.write("20011,10010,25\n")
    conn = load_csv_dir(d, min_weight=1, subset="all")
    assert conn.e == 115 and _w(conn, 20011, 10010) == 75.0


def test_float_weights_and_crlf_are_accepted(tmp_path):
    d = _copy_fixture(tmp_path)
    text = (d / "connections.csv").read_text(encoding="utf-8").replace("\n", "\r\n").replace("20011,10010,50", "20011,10010,50.0")
    (d / "connections.csv").write_text(text, encoding="utf-8", newline="")
    conn = load_csv_dir(d, min_weight=1, subset="all")
    assert conn.e == 115 and _w(conn, 20011, 10010) == 50.0


def test_header_only_connections(tmp_path):
    d = _copy_fixture(tmp_path)
    (d / "connections.csv").write_text("bodyId_pre,bodyId_post,weight\n", encoding="utf-8")
    conn = load_csv_dir(d, min_weight=1, subset="all")
    assert conn.n == 40 and conn.e == 0 and conn.meta["synapses"] == 0.0
    conn.validate()


def test_missing_columns_raise(tmp_path):
    d = _copy_fixture(tmp_path)
    (d / "connections.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="column"):
        load_csv_dir(d, min_weight=1, subset="all")
    (d / "connections.csv").unlink()
    with pytest.raises(FileNotFoundError):
        load_csv_dir(d, min_weight=1, subset="all")


@pytest.mark.parametrize(
    "bad_table",
    [
        "bodyId_pre,bodyId_post,weight\n10001,804642,\n",        # empty weight cell
        "bodyId_pre,bodyId_post,weight\n10001,804642\n",         # ragged row
        "bodyId_pre,bodyId_post,weight\n10001,804642,many\n",    # non-numeric weight
        "bodyId_pre,bodyId_post,weight\n10001,x,45\n",           # non-numeric id
    ],
)
def test_malformed_numeric_columns_name_the_file(tmp_path, bad_table):
    # numpy names the row/column but not the file; the loader must say which table failed
    d = _copy_fixture(tmp_path)
    (d / "connections.csv").write_text(bad_table, encoding="utf-8")
    with pytest.raises(ValueError, match=r"connections\.csv: cannot parse the numeric columns"):
        load_csv_dir(d, min_weight=1, subset="all")


@pytest.mark.parametrize("bad_rows", ["20011,10010,inf\n", "20011,10010,nan\n", "20011,10010,3e38\n20011,10010,3e38\n"])
def test_non_finite_or_huge_weights_are_rejected(tmp_path, bad_rows):
    # regression: two 3e38 rows for one pair used to sum to inf and pass validate()
    d = _copy_fixture(tmp_path)
    with open(d / "connections.csv", "a", encoding="utf-8", newline="") as fh:
        fh.write(bad_rows)
    with pytest.raises(ValueError, match="finite"):
        load_csv_dir(d, min_weight=1, subset="all")


def _with_fillers(tmp_path: Path, count: int = 10) -> Path:
    d = _copy_fixture(tmp_path)
    with open(d / "neurons.csv", "a", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        for k in range(count):
            w.writerow([900000 + k, "", "", "cb_intrinsic", "", "", "L" if k % 2 else "R", "Traced", "gaba", ""])
    return d


def test_subset_core_keeps_functional(tmp_path):
    d = _with_fillers(tmp_path, 10)
    full = load_csv_dir(d, min_weight=1, subset="all")
    assert full.n == 50
    # n_max below the functional count: every functional neuron is kept, nothing is filled
    capped30 = load_csv_dir(d, min_weight=1, subset="core", n_max=30)
    assert capped30.n == 40 and not any(b >= 900000 for b in capped30.body_id.tolist())
    assert capped30.meta["build_args"]["n_core"] == 40 and capped30.meta["build_args"]["n_fill"] == 0
    assert set(capped30.body_id[capped30.groups["gf"]].tolist()) == {10001, 10010}
    assert validate_groups(capped30)
    # n_max above it: random fill is seeded / deterministic
    capped = load_csv_dir(d, min_weight=1, subset="core", n_max=45, seed=3)
    assert capped.n == 45 and sum(1 for b in capped.body_id.tolist() if b >= 900000) == 5
    again = load_csv_dir(d, min_weight=1, subset="core", n_max=45, seed=3)
    assert np.array_equal(again.body_id, capped.body_id) and np.array_equal(again.pre, capped.pre)
    other = load_csv_dir(d, min_weight=1, subset="core", n_max=45, seed=4)
    assert other.n == 45 and set(other.body_id[other.groups["gf"]].tolist()) == {10001, 10010}
    core_only = load_csv_dir(d, min_weight=1, subset="core", n_max=None)
    assert core_only.n == 40
    assert capped.e == full.subset(np.flatnonzero(np.isin(full.body_id, capped.body_id)).astype(np.int32)).e


def test_untyped_groups_are_empty_not_missing(tmp_path):
    d = _with_fillers(tmp_path, 4)
    conn = load_csv_dir(d, min_weight=1, subset="all")
    assert "" in conn.types
    assert all(name in conn.groups for name in READOUTS)
    assert conn.groups["mbon_avoid"].size == 0 and conn.groups["mbon_avoid"].dtype == np.int32


def test_parquet_input_via_pyarrow(tmp_path, np_all):
    pa_csv = pytest.importorskip("pyarrow.csv")
    pq = pytest.importorskip("pyarrow.parquet")
    for name in ("neurons", "connections"):
        table = pa_csv.read_csv(str(NP_DIR / f"{name}.csv"))
        pq.write_table(table, str(tmp_path / f"{name}.parquet"))
    assert detect_schema(tmp_path) is Schema.NEUPRINT
    conn = load_csv_dir(tmp_path, min_weight=1, subset="all")
    assert conn.n == 40 and conn.e == 115
    _assert_same_graph(conn, np_all)


# --------------------------------------------------------------------------- Codex (FlyWire) fixture


def test_load_codex_small(np_all):
    conn = load_csv_dir(CX_DIR, min_weight=1, subset="all")
    conn.validate()
    assert conn.n == 40 and conn.e == 115
    assert conn.meta["license"] == "CC-BY-NC 4.0 (FlyWire)" and "FlyWire" in conn.meta["citation"]
    assert conn.meta["build_args"]["schema"] == "codex_fafb" and conn.meta["build_args"]["e_file"] == 130
    assert set(conn.meta["build_args"]["files"]) == {"classification.csv.gz", "consolidated_cell_types.csv.gz",
                                                     "neurons.csv.gz", "connections.csv.gz"}
    # the same 40 neurons: identical graph up to the root_id offset
    _assert_same_graph(conn, np_all, same_body_ids=False)
    assert np.array_equal(conn.body_id - B, np_all.body_id)
    # per-neuropil rows summed (LC4_L -> GF_L is 25 + 25 in the fixture)
    assert _w(conn, B + 20011, B + 10010) == 50.0 and _w(conn, B + 20033, B + 10331) == 190.0
    # FlyWire vocabulary: 'left'/'right' sides, super_class mapping, uppercase nt_type labels
    assert conn.side[_index_of(conn, B + 10010)] == -1 and conn.side[_index_of(conn, B + 10001)] == 1
    assert conn.region[_index_of(conn, B + 10001)] == REGION_ID["descending_motor"]
    assert conn.region[_index_of(conn, B + 10331)] == REGION_ID["sez"] and conn.region[_index_of(conn, B + 20082)] == REGION_ID["antennal_lobe"]
    assert conn.nt[_index_of(conn, B + 10331)] == "glutamate" and conn.sign[_index_of(conn, B + 10331)] == -1.0
    assert conn.nt[_index_of(conn, B + 20035)] == "gaba" and conn.nt[_index_of(conn, B + 20001)] == "unknown"
    assert conn.meta["build_args"]["nt_from_edges"] == 0
    pruned = load_csv_dir(CX_DIR, min_weight=3, subset="all")
    assert pruned.e == load_csv_dir(NP_DIR, min_weight=3, subset="all").e
    assert validate_groups(conn)["pfl3"] == 2


def test_codex_untyped_cell_is_not_core(tmp_path):
    d = tmp_path / "cx"
    shutil.copytree(CX_DIR, d)
    _append_gz(d / "classification.csv.gz", [[B + 1, "intrinsic", "central", "", "", "", "center", ""]])
    _append_gz(d / "neurons.csv.gz", [[B + 1, "", "untyped_C", "GABA", "0.9"]])
    full = load_csv_dir(d, min_weight=1, subset="all")
    i = _index_of(full, B + 1)
    assert full.n == 41 and full.type_of(i) == "" and full.side[i] == 0 and full.nt[i] == "gaba"
    assert full.region[i] == REGION_ID["central_other"]
    core = load_csv_dir(d, min_weight=1, subset="core", n_max=None)
    assert core.n == 40 and (B + 1) not in core.body_id.tolist()


def test_codex_partial_download_warns_about_empty_types(tmp_path, caplog):
    """classification + connections only: the load is tolerant but says why no group can resolve."""
    d = tmp_path / "cx"
    d.mkdir()
    for f in ("classification.csv.gz", "connections.csv.gz"):
        shutil.copy(CX_DIR / f, d / f)
    with caplog.at_level(logging.WARNING, logger="flybrain.connectome.loaders"):
        conn = load_csv_dir(d, min_weight=1, subset="all")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("consolidated_cell_types" in m for m in msgs) and any("nt_type" in m for m in msgs)
    assert all(m.isascii() for m in msgs)
    assert conn.n == 40 and conn.types == [""]
    assert conn.meta["build_args"]["nt_from_edges"] == 37  # NT rescued from the edge table's nt_type column
    assert all(int(v.size) == 0 for v in conn.groups.values())
    with pytest.raises(ValueError):
        validate_groups(conn)  # no functional group resolves without the type table


def test_codex_nt_from_edges_histamine(tmp_path):
    """A cell absent from neurons.csv.gz takes the synapse-weighted majority NT of its outgoing edges.

    R1-R6 photoreceptors are histaminergic (RESEARCH section 4 [V], inhibitory sign) and L1 is
    glutamatergic [V]; R1-R6 -> L1 is 26.7 synapses per pair [V] RESEARCH section 5.
    """
    d = tmp_path / "cx"
    shutil.copytree(CX_DIR, d)
    r16, l1 = B + 30001, B + 30002
    _append_gz(d / "classification.csv.gz", [[r16, "afferent", "sensory", "visual", "", "", "left", ""],
                                              [l1, "intrinsic", "optic", "", "", "", "left", ""]])
    _append_gz(d / "consolidated_cell_types.csv.gz", [[r16, "R1-R6", ""], [l1, "L1", ""]])
    _append_gz(d / "neurons.csv.gz", [[l1, "", "L1_L", "GLUT", "0.9"]])  # R1-R6 deliberately absent
    _append_gz(d / "connections.csv.gz", [[r16, l1, "LA_L", 20, "HIS"], [r16, l1, "LA_L", 7, "HIS"]])
    conn = load_csv_dir(d, min_weight=1, subset="all")
    conn.validate()
    i, j = _index_of(conn, r16), _index_of(conn, l1)
    assert conn.n == 42 and conn.e == 116 and _w(conn, r16, l1) == 27.0
    assert conn.nt[i] == "histamine" and conn.sign[i] == -1.0 and conn.meta["build_args"]["nt_from_edges"] == 1
    assert conn.nt[j] == "glutamate" and conn.sign[j] == -1.0
    assert conn.region[i] == REGION_ID["optic_lobe"] and conn.region[j] == REGION_ID["optic_lobe"]
    assert conn.groups["photoreceptor"].tolist() == [i] and conn.groups["lamina"].tolist() == [j]
    csr = conn.csr()
    idx, data = csr.row(i)
    assert float(data[idx == j][0]) == -27.0  # inhibitory photoreceptor drive


# --------------------------------------------------------------------------- Codex MaleCNS mirror


def test_load_codex_mcns_mirror(mcns_dir, np_all):
    """End-to-end CODEX_MCNS load: 'Predicted NT type' is read for every neuron (regression for the alias typo)."""
    conn = load_csv_dir(mcns_dir, min_weight=1, subset="all")
    conn.validate()
    assert conn.meta["build_args"]["schema"] == "codex_mcns" and conn.meta["license"] == "CC-BY 4.0 (MaleCNS)"
    assert set(conn.meta["build_args"]["files"]) == {"neurons.csv.gz", "connections_princeton.csv.gz"}
    _assert_same_graph(conn, np_all)
    assert conn.meta["build_args"]["nt_from_edges"] == 0  # every NT came from the neuron table
    # MN9_L (10331) has no outgoing edges in the fixture: its NT can only come from the neuron column
    i = _index_of(conn, 10331)
    assert conn.nt[i] == "glutamate" and conn.sign[i] == -1.0
    assert conn.nt[_index_of(conn, 20001)] == "unknown" and conn.sign[_index_of(conn, 20001)] == 1.0


def test_codex_mcns_nt_column_is_read(tmp_path):
    (tmp_path / "neurons.csv.gz").write_bytes(gzip.compress(
        b"Root ID,Super Class,Class,Soma side,Primary Cell Type,Predicted NT type\n"
        b"10001,descending_neuron,,R,DNp01,ACH\n10010,descending_neuron,,L,DNp01,ACH\n804642,vnc_motor,,L,TTMn,GLUT\n"))
    (tmp_path / "connections_princeton.csv.gz").write_bytes(gzip.compress(
        b"Pre Root ID,Post Root ID,Neuropil,Syn Count,NT Type\n10010,804642,VNC,45,ACH\n"))
    assert detect_schema(tmp_path) is Schema.CODEX_MCNS
    conn = load_csv_dir(tmp_path, min_weight=1, subset="all")
    assert [str(x) for x in conn.nt] == ["acetylcholine", "acetylcholine", "glutamate"]
    assert conn.sign.tolist() == [1.0, 1.0, -1.0] and conn.meta["build_args"]["nt_from_edges"] == 0
    assert conn.e == 1 and _w(conn, 10010, 804642) == 45.0
    assert conn.side.tolist() == [1, -1, -1]


# --------------------------------------------------------------------------- load_connectome


def test_load_connectome_dispatch(tmp_path, monkeypatch):
    # synthetic -> build_synthetic (a fake module: the real generator is not needed here)
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return build_tiny_connectome()

    fake = types.ModuleType("flybrain.connectome.synthetic")
    fake.build_synthetic = fake_build
    fake.GENERATOR_VERSION = "fake"
    monkeypatch.setitem(sys.modules, "flybrain.connectome.synthetic", fake)
    syn = load_connectome(make_settings(tmp_path, connectome_source="synthetic", n_neurons=400, seed=5))
    assert calls == [{"n_neurons": 400, "seed": 5, "mean_outdeg": 25, "weights": "calibrated"}]
    assert syn.n == 400 and syn.source == "synthetic"
    # csv -> load_csv_dir on connectome_dir
    seen: list[Path] = []
    real = loaders_mod.load_csv_dir

    def spy(path, **kw):
        seen.append(Path(path))
        return real(path, **kw)

    monkeypatch.setattr(loaders_mod, "load_csv_dir", spy)
    conn = load_connectome(make_settings(tmp_path))
    assert seen == [NP_DIR] and conn.source == "csv" and conn.n == 40
    # neuprint with a missing directory -> FileNotFoundError naming the fetch script
    with pytest.raises(FileNotFoundError) as exc:
        load_connectome(make_settings(tmp_path, connectome_source="neuprint"))
    msg = str(exc.value)
    assert "fetch_neuprint.py" in msg and str(tmp_path / "data" / "connectome" / "neuprint") in msg
    assert all(ord(ch) < 128 for ch in msg)


def test_cache_roundtrip(tmp_path, np_all):
    cache_dir = tmp_path / "cache"
    key = "abc123def456"
    store_cached(key, np_all, cache_dir)
    loaded = load_cached(key, cache_dir)
    assert loaded is not None
    _assert_same_graph(loaded, np_all)
    assert loaded.meta["cache_key"] == key and loaded.meta["license"] == np_all.meta["license"]
    # cache_key changes with n_neurons, seed and the file mtime
    d = _copy_fixture(tmp_path)
    s = make_settings(tmp_path, connectome_dir=d)
    k0 = cache_key(s)
    assert cache_key(make_settings(tmp_path, connectome_dir=d, n_neurons=20_001)) != k0
    assert cache_key(make_settings(tmp_path, connectome_dir=d, seed=1)) != k0
    st = (d / "connections.csv").stat()
    os.utime(d / "connections.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))  # content unchanged, mtime +5 s
    assert cache_key(s) != k0


def test_load_connectome_csv_caches_and_patches(tmp_path, monkeypatch, caplog):
    s = make_settings(tmp_path)
    conn = load_connectome(s)
    conn.validate()
    key = cache_key(s)
    assert conn.meta["cache_key"] == key
    assert conn.source == "csv" and conn.n == 40
    assert [r["pre"] for r in conn.meta["patches_applied"]] == ["DNp01", "DNp01", "DNp01", "LC4", "LPLC2"]
    assert _w(conn, 10010, 804642) == 345.0  # 45 chemical + 300 patch
    cache_dir = tmp_path / "data" / "cache"
    assert (cache_dir / f"{key}.npz").is_file() and (cache_dir / f"{key}.json").is_file()
    # second call must be served from the cache: the CSV loader is not allowed to run
    monkeypatch.setattr(loaders_mod, "load_csv_dir", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache miss")))
    again = load_connectome(s)
    assert again.e == conn.e and np.array_equal(again.weight, conn.weight) and np.array_equal(again.body_id, conn.body_id)
    assert len(again.meta["patches_applied"]) == 5  # patches not re-applied to the cached graph
    assert all(ord(ch) < 128 for rec in caplog.records for ch in rec.getMessage())


def test_load_connectome_name_override_and_all_subset_mmap(tmp_path):
    s = make_settings(tmp_path, subset="all", min_weight=1, connectome_name="my-graph")
    first = load_connectome(s)
    assert first.name == "my-graph" and first.e == 115 + 2  # +2 GF<->GF patch edges
    second = load_connectome(s)
    assert second.name == "my-graph"
    assert isinstance(second.pre, np.memmap) and isinstance(second.weight, np.memmap)
    assert np.array_equal(second.weight, first.weight)
    del first, second
    gc.collect()


def test_load_connectome_neuprint_source_reads_data_dir(tmp_path):
    target = tmp_path / "data" / "connectome" / "neuprint"
    shutil.copytree(NP_DIR, target)
    s = make_settings(tmp_path, connectome_source="neuprint", connectome_dir=Path("somewhere/else"))
    conn = load_connectome(s)
    assert conn.n == 40 and conn.meta["build_args"]["path"] == str(target)


def test_load_connectome_csv_dir_missing(tmp_path):
    s = make_settings(tmp_path, connectome_dir=tmp_path / "nothing-here")
    with pytest.raises(FileNotFoundError, match="prepare_malecns"):
        load_connectome(s)


def test_load_connectome_bad_source(tmp_path):
    with pytest.raises(ValueError, match="FLY_CONNECTOME_SOURCE"):
        load_connectome(make_settings(tmp_path, connectome_source="ftp"))


def test_load_connectome_requires_functional_groups(tmp_path):
    d = tmp_path / "few"
    d.mkdir()
    rows = [r for r in _rows(NP_DIR / "neurons.csv") if r["type"] in ("DNp01", "TTMn", "MN9")]
    with open(d / "neurons.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    shutil.copy(NP_DIR / "connections.csv", d / "connections.csv")
    with pytest.raises(ValueError, match="pfl3"):
        load_connectome(make_settings(tmp_path, connectome_dir=d, min_weight=1))


def test_load_connectome_synthetic_is_lazy(tmp_path, monkeypatch):
    real_present = "flybrain.connectome.synthetic" in sys.modules
    calls: list[dict] = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return build_tiny_connectome()

    fake = types.ModuleType("flybrain.connectome.synthetic")
    fake.build_synthetic = fake_build
    fake.GENERATOR_VERSION = "fake"
    monkeypatch.setitem(sys.modules, "flybrain.connectome.synthetic", fake)
    s = make_settings(tmp_path, connectome_source="synthetic", n_neurons=400, seed=5, mean_outdeg=7, synth_weights="literature")
    conn = load_connectome(s)
    assert calls == [{"n_neurons": 400, "seed": 5, "mean_outdeg": 7, "weights": "literature"}]
    assert conn.n == 400 and len(conn.meta["patches_applied"]) == 5
    key = cache_key(s)
    assert (tmp_path / "data" / "cache" / f"{key}.npz").is_file()
    cached = load_connectome(s)
    assert len(calls) == 1 and cached.e == conn.e
    if not real_present:
        assert not hasattr(loaders_mod, "build_synthetic")


def test_load_connectome_synthetic_missing_module_message(tmp_path, monkeypatch):
    if "flybrain.connectome.synthetic" in sys.modules or importlib.util.find_spec("flybrain.connectome.synthetic"):
        pytest.skip("real synthetic module present")
    monkeypatch.setitem(sys.modules, "flybrain.connectome.synthetic", None)  # forces ImportError
    with pytest.raises(ImportError, match="synthetic"):
        load_connectome(make_settings(tmp_path, connectome_source="synthetic"))


def test_load_connectome_cache_write_failure_is_not_fatal(tmp_path, caplog):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cache").write_text("i am a file, not a directory", encoding="utf-8")
    s = make_settings(tmp_path)
    with caplog.at_level(logging.WARNING, logger="flybrain.connectome"):
        conn = load_connectome(s)
    assert conn.n == 40
    msgs = [rec.getMessage() for rec in caplog.records if "could not store connectome cache" in rec.getMessage()]
    assert msgs
    assert all(ord(ch) < 128 for m in msgs for ch in m)  # localized WinError text must be escaped


def test_cache_write_failure_log_is_ascii_for_localized_oserror(tmp_path, monkeypatch, caplog):
    # regression: the localized '[WinError 5] Eri<s-cedilla>im engellendi' text used to reach the log verbatim
    def boom(key, conn, cache_dir):
        raise PermissionError("[WinError 5] Eri\u015fim engellendi: 'x\u00e7'")

    monkeypatch.setattr(cache_mod, "store_cached", boom)
    with caplog.at_level(logging.WARNING, logger="flybrain.connectome"):
        conn = load_connectome(make_settings(tmp_path))
    assert conn.n == 40
    msgs = [rec.getMessage() for rec in caplog.records if "could not store connectome cache" in rec.getMessage()]
    assert len(msgs) == 1 and "PermissionError" in msgs[0] and "\\u015f" in msgs[0]
    assert all(ord(ch) < 128 for ch in msgs[0])


def test_e1_modules_import_no_optional_dependencies():
    # a fresh interpreter: importing patches / loaders / cache must not pull in any optional dependency
    code = (
        "import sys; import flybrain.connectome.patches, flybrain.connectome.loaders, flybrain.connectome.cache; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('torch','pandas','pyarrow','neuprint')); "
        "print('BAD=' + ','.join(bad))"
    )
    proc = subprocess.run([sys.executable, "-W", "error", "-c", code], capture_output=True, text=True,
                          cwd=str(REPO / "backend"), timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "BAD=\n" in proc.stdout or proc.stdout.strip() == "BAD="


# --------------------------------------------------------------------------- scripts/fetch_neuprint.py


class FakeNeuprint:
    """Offline stand-in for the neuPrint custom-Cypher endpoint, backed by the neuprint_small fixture + fillers.

    Answers the five query shapes the script issues (distinct types, neurons by type list, neurons by
    class/superclass, random fill, ConnectsTo edges among two body lists) and counts them per kind.
    """

    NEURON_COLS = ["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "status",
                   "consensusNt", "predictedNt"]

    def __init__(self, n_fillers: int = 10, fail_on_edge_query: int | None = None) -> None:
        self.neurons: dict[int, list[Any]] = {}
        for r in _rows(NP_DIR / "neurons.csv"):
            self.neurons[int(r["bodyId"])] = [int(r["bodyId"])] + [r[c] or None for c in self.NEURON_COLS[1:]]
        for k in range(n_fillers):
            bid = 900000 + k
            self.neurons[bid] = [bid, None, None, "cb_intrinsic", None, None, "L" if k % 2 else "R", "Traced", "gaba", "gaba"]
        self.edges: dict[tuple[int, int], int] = {}
        for r in _rows(NP_DIR / "connections.csv"):  # the server holds one ConnectsTo per pair: pre-summed
            key = (int(r["bodyId_pre"]), int(r["bodyId_post"]))
            self.edges[key] = self.edges.get(key, 0) + int(r["weight"])
        self.counts: dict[str, int] = {"types": 0, "by_type": 0, "by_class": 0, "random": 0, "edges": 0}
        self.fail_on_edge_query = fail_on_edge_query

    @staticmethod
    def _list(cypher: str, prefix: str) -> list[Any]:
        m = re.search(re.escape(prefix) + r" IN (\[[^\]]*\])", cypher)
        assert m, cypher[:120]
        return json.loads(m.group(1))

    def query(self, cypher: str, retries: int = 3) -> tuple[list[str], list[list[Any]]]:
        if "RETURN DISTINCT n.type" in cypher:
            self.counts["types"] += 1
            return ["type"], [[t] for t in sorted({v[1] for v in self.neurons.values() if v[1]})]
        if "ConnectsTo" in cypher:
            self.counts["edges"] += 1
            if self.fail_on_edge_query is not None and self.counts["edges"] == self.fail_on_edge_query:
                raise RuntimeError("simulated HTTP 504 on edge chunk")
            pre = set(self._list(cypher, "a.bodyId"))
            post = set(self._list(cypher, "b.bodyId"))
            minw = int(re.search(r"c\.weight >= (\d+)", cypher).group(1))
            rows = [[a, b, w] for (a, b), w in self.edges.items() if a in pre and b in post and w >= minw]
            return ["bodyId_pre", "bodyId_post", "weight"], rows
        if "n.type IN" in cypher:
            self.counts["by_type"] += 1
            wanted = set(self._list(cypher, "n.type"))
            return list(self.NEURON_COLS), [v for v in self.neurons.values() if v[1] in wanted]
        if "n.class IN" in cypher:
            self.counts["by_class"] += 1
            classes = set(self._list(cypher, "n.class"))
            supers = set(self._list(cypher, "n.superclass"))
            return list(self.NEURON_COLS), [v for v in self.neurons.values() if v[4] in classes or v[3] in supers]
        if "rand() <" in cypher:
            self.counts["random"] += 1
            return list(self.NEURON_COLS), list(self.neurons.values())
        raise AssertionError(f"unexpected cypher: {cypher[:120]}")


def test_fetch_neuprint_spec_command_line_and_defaults():
    mod = _load_script("fetch_neuprint")
    # the SPEC i.5 / h.4 command line must parse (regression: --n-random was rejected by argparse)
    args = mod.build_parser().parse_args(["--out", "data\\connectome\\neuprint", "--n-random", "25000", "--min-weight", "3"])
    assert args.n_random == 25000 and args.min_weight == 3 and args.chunk == 2000 and args.token is None
    defaults = mod.build_parser().parse_args([])
    assert defaults.n_random == 25000 and defaults.min_weight == 3 and defaults.chunk == 2000 and defaults.concurrency == 3
    aliases = mod.build_parser().parse_args(["--n-max", "10", "--chunk-ids", "5", "--token", "abc", "--anonymous"])
    assert aliases.n_random == 10 and aliases.chunk == 5 and aliases.token == "abc" and aliases.anonymous
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0


def test_fetch_neuprint_anonymous_is_default_without_token(monkeypatch, capsys):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    mod = _load_script("fetch_neuprint")
    t = mod.make_transport(mod.DEFAULT_SERVER, mod.DEFAULT_DATASET, None)
    assert isinstance(t, mod.HttpTransport) and t.token is None
    out = capsys.readouterr().out
    assert "anonymous" in out and all(ord(ch) < 128 for ch in out)
    # the anonymous urllib route sends NO Authorization header; a token adds a bearer header
    seen: list[dict[str, str]] = []

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"columns": ["x"], "data": [[1]]}).encode()

    def fake_urlopen(req, timeout=0):
        seen.append({k.lower(): v for k, v in req.header_items()})
        assert req.get_method() == "POST" and req.full_url.endswith("/api/custom/custom")
        assert json.loads(req.data.decode())["dataset"] == mod.DEFAULT_DATASET
        return FakeResp()

    monkeypatch.setattr(mod.urllib.request, "urlopen", fake_urlopen)
    assert t.query("RETURN 1") == (["x"], [[1]])
    assert "authorization" not in seen[-1]
    with_token = mod.HttpTransport(mod.DEFAULT_SERVER, mod.DEFAULT_DATASET, "tok")
    assert with_token.query("RETURN 1") == (["x"], [[1]])
    assert seen[-1]["authorization"] == "Bearer tok"


def test_fetch_neuprint_end_to_end_with_fake_server(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    mod = _load_script("fetch_neuprint")
    fake = FakeNeuprint(n_fillers=10)
    out = tmp_path / "np"
    rc = mod.main(["--out", str(out), "--n-random", "45", "--min-weight", "3", "--chunk", "8"],
                  transport_factory=lambda server, dataset, token: fake)
    text = capsys.readouterr().out
    assert rc == 0, text
    assert all(ord(ch) < 128 for ch in text)
    assert fake.counts == {"types": 1, "by_type": 1, "by_class": 1, "random": 1, "edges": 6}  # ceil(45 / 8) chunks
    neurons = _rows(out / "neurons.csv")
    assert len(neurons) == 45 and list(neurons[0]) == list(mod.NEURON_COLUMNS)
    assert sum(1 for r in neurons if int(r["bodyId"]) >= 900000) == 5  # 40 core + 5 random fill
    edges = _rows(out / "connections.csv")
    assert list(edges[0]) == ["bodyId_pre", "bodyId_post", "weight"] and all(int(r["weight"]) >= 3 for r in edges)
    assert len(edges) == sum(1 for w in _summed_pairs().values() if w >= 3)
    assert not (out / mod.CHUNK_DIRNAME).exists()  # merged and cleaned up
    conn = load_csv_dir(out, min_weight=3, subset="all")
    assert conn.n == 45 and conn.e == len(edges) and set(conn.body_id[conn.groups["gf"]].tolist()) == {10001, 10010}
    assert _w(conn, 20011, 10010) == 50.0 and validate_groups(conn)


def test_fetch_neuprint_resumes_per_chunk(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NEUPRINT_APPLICATION_CREDENTIALS", raising=False)
    mod = _load_script("fetch_neuprint")
    out = tmp_path / "np"
    argv = ["--out", str(out), "--n-random", "45", "--min-weight", "3", "--chunk", "8", "--concurrency", "1"]
    failing = FakeNeuprint(n_fillers=10, fail_on_edge_query=3)
    rc = mod.main(argv, transport_factory=lambda *a: failing)
    text = capsys.readouterr().out
    assert rc == 1 and "ERROR: RuntimeError" in text and "resume" in text and all(ord(ch) < 128 for ch in text)
    chunk_dir = out / mod.CHUNK_DIRNAME
    assert (out / "neurons.csv").is_file() and not (out / "connections.csv").exists()
    assert sorted(p.name for p in chunk_dir.iterdir()) == ["edges_00000.csv", "edges_00001.csv", mod.STATE_FILENAME]
    # second run: neurons.csv and the two stored chunks are reused, only the 4 missing chunks are fetched
    working = FakeNeuprint(n_fillers=10)
    rc = mod.main(argv, transport_factory=lambda *a: working)
    text = capsys.readouterr().out
    assert rc == 0 and "resuming" in text
    assert working.counts == {"types": 0, "by_type": 0, "by_class": 0, "random": 0, "edges": 4}
    assert not chunk_dir.exists()
    edges = _rows(out / "connections.csv")
    assert len(edges) == sum(1 for w in _summed_pairs().values() if w >= 3)
    # different arguments (or --fresh) start over
    fresh = FakeNeuprint(n_fillers=10)
    rc = mod.main(argv + ["--fresh"], transport_factory=lambda *a: fresh)
    assert rc == 0 and fresh.counts["types"] == 1 and fresh.counts["edges"] == 6
    assert _rows(out / "connections.csv") == edges


def test_fetch_neuprint_subprocess_help_is_ascii():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "ascii"
    proc = subprocess.run([sys.executable, str(SCRIPTS / "fetch_neuprint.py"), "--help"],
                          capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "--n-random" in proc.stdout and "--min-weight" in proc.stdout and "--chunk" in proc.stdout and "--token" in proc.stdout


def test_fetch_neuprint_helpers():
    mod = _load_script("fetch_neuprint")
    pats = mod.group_patterns()
    assert mod.matched_types(["DNp01", "LC4", "DLMn a, b", "nonsense", "", "KCg-m"], pats) == ["DLMn a, b", "DNp01", "KCg-m", "LC4"]
    assert mod.cypher_literal([1, 2]) == "[1, 2]" and mod.cypher_literal(["DLMn a, b"]) == '["DLMn a, b"]'
    assert mod.chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    into: dict[int, dict[str, str]] = {}
    cols = ["bodyId", "type", "somaSide", "consensusNt"]
    added = mod.rows_to_neurons(cols, [[10001, "DNp01", "R", None], [10001, "DNp01", "R", "ach"], [None, "x", "L", ""]], into)
    assert added == 1 and into[10001]["consensusNt"] == "" and into[10001]["instance"] == ""
    assert mod.ascii_safe("Eri\u015fim") == "Eri\\u015fim"


# --------------------------------------------------------------------------- scripts/prepare_malecns.py


class _FakeHttp:
    """urllib.request.urlopen stand-in serving ``payload`` with Range support and an optional short read."""

    def __init__(self, payload: bytes, deliver: int | None = None, honour_range: bool = True) -> None:
        self.payload = payload
        self.deliver = deliver
        self.honour_range = honour_range
        self.requests: list[dict[str, str]] = []

    def __call__(self, req, timeout=0):
        headers = {k.lower(): v for k, v in req.header_items()}
        self.requests.append(headers)
        start = 0
        status = 200
        if "range" in headers and self.honour_range:
            start = int(headers["range"].split("=")[1].rstrip("-"))
            status = 206
        body = self.payload[start:]
        served = body if self.deliver is None else body[: self.deliver]
        fake = self

        class Resp:
            def __init__(self):
                self.status = status
                self.headers = {"Content-Length": str(len(body))}
                self._buf = io.BytesIO(served)

            def read(self, n=-1):
                return self._buf.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        del fake
        return Resp()


def test_prepare_malecns_download_size_check_and_resume(tmp_path, monkeypatch, capsys):
    mod = _load_script("prepare_malecns")
    payload = bytes(range(256)) * 40  # 10240 bytes
    dest = tmp_path / "raw" / "file.feather"
    # a short read (Content-Length 10240, 4000 delivered) is detected; the .part file survives for the resume
    short = _FakeHttp(payload, deliver=4000)
    monkeypatch.setattr(mod.urllib.request, "urlopen", short)
    with pytest.raises(RuntimeError, match="incomplete"):
        mod.download("https://example.invalid/file.feather", dest, chunk=1000, expected_mb=0.01)
    part = dest.with_name(dest.name + ".part")
    assert part.is_file() and part.stat().st_size == 4000 and not dest.exists()
    # the resume sends a Range header, gets 206 and completes; the byte count matches Content-Length
    full = _FakeHttp(payload)
    monkeypatch.setattr(mod.urllib.request, "urlopen", full)
    assert mod.download("https://example.invalid/file.feather", dest, chunk=1000, expected_mb=0.01) == dest
    assert full.requests[0]["range"] == "bytes=4000-"
    assert dest.read_bytes() == payload and not part.exists()
    out = capsys.readouterr().out
    assert "downloaded file.feather" in out and "WARNING" not in out and all(ord(ch) < 128 for ch in out)
    # RESEARCH 2.1 size expectation: a mismatch warns (upstream may re-upload) but does not fail
    assert mod.check_size(dest, 0.01024) is True
    assert mod.check_size(dest, 14.5) is False
    assert "RESEARCH 2.1 lists 14.5 MB" in capsys.readouterr().out
    assert mod.EXPECTED_MB == {"annotations": 14.5, "neurotransmitters": 43.3, "weights": 508.0}
    assert mod.RAW_DIRNAME == "raw" and mod.ascii_safe("\u00e7") == "\\xe7"


def test_prepare_malecns_skip_download_reports_raw_dir(tmp_path, capsys):
    pytest.importorskip("pyarrow")
    mod = _load_script("prepare_malecns")
    rc = mod.main(["--out", str(tmp_path / "malecns"), "--skip-download"])
    out = capsys.readouterr().out
    assert rc == 1 and "ERROR: missing" in out and str(tmp_path / "malecns" / "raw") in out
    assert all(ord(ch) < 128 for ch in out)


def test_prepare_malecns_convert_and_help(tmp_path):
    pa = pytest.importorskip("pyarrow")
    mod = _load_script("prepare_malecns")
    with pytest.raises(SystemExit) as exc:
        mod.main(["--help"])
    assert exc.value.code == 0
    ann = pa.table({
        "bodyId": pa.array([10010, 10001, 20011, 30000, 40000], pa.int64()),
        "type": ["DNp01", "DNp01", "LC4", None, "MN9"],
        "instance": ["DNp01(GF)_L", "DNp01(GF)_R", "LC4_L", None, "MN9_L"],
        "somaSide": ["L", "R", "L", None, "L"],
        "superclass": ["descending_neuron", "descending_neuron", "visual_projection", "cb_intrinsic", "cb_motor"],
        "class": [None, None, None, None, None],
        "subclass": [None, None, None, None, None],
        "status": ["Traced", "Traced", "Traced", "Orphan", "Traced"],
    })
    nt = pa.table({
        "body": pa.array([10001, 40000, 20011], pa.int64()),
        "consensus_nt": ["acetylcholine", "glutamate", None],
        "predicted_nt": ["acetylcholine", "glutamate", "acetylcholine"],
    })
    wt = pa.table({
        "body_pre": pa.array([20011, 10010, 10001, 30000], pa.int64()),
        "body_post": pa.array([10010, 10001, 40000, 10001], pa.int64()),
        "weight": pa.array([50, 1, 4, 9], pa.int64()),
        "type_pre": ["LC4", "DNp01", "DNp01", None],
        "type_post": ["DNp01", "DNp01", "MN9", "DNp01"],
    })
    out = tmp_path / "out"
    stats = mod.convert(ann, nt, wt, out, min_weight=2)
    assert stats["neurons_out"] == 4 and stats["edges_out"] == 3 and stats["nt_matched"] == 3
    rows = _rows(out / "neurons.csv")
    assert [r["bodyId"] for r in rows] == ["10001", "10010", "20011", "40000"]
    assert rows[2]["consensusNt"] == "" and rows[2]["predictedNt"] == "acetylcholine"
    conn = load_csv_dir(out, min_weight=1, subset="all")
    assert conn.n == 4 and conn.e == 2  # the edge from the orphan body 30000 is dropped by the loader
    assert _w(conn, 20011, 10010) == 50.0 and conn.nt[_index_of(conn, 20011)] == "acetylcholine"
    stats_gz = mod.convert(ann, nt, wt, tmp_path / "gz", min_weight=1, traced_only=False, gzip_out=True)
    assert stats_gz["neurons_out"] == 5 and stats_gz["edges_out"] == 4
    assert detect_schema(tmp_path / "gz") is Schema.NEUPRINT
    assert load_csv_dir(tmp_path / "gz", min_weight=1, subset="all").n == 4  # Orphan filtered by the loader
