"""Shared pytest fixtures (SPEC section h.1).

Fixtures: ``tiny_connectome`` (hand-built, N=400), ``small_synthetic`` (``build_synthetic(4000, seed=1)``,
session), ``synthetic_20k`` (``build_synthetic(20_000, seed=1337)``, session, for ``slow`` tests),
``settings_tmp`` (``load_settings`` on a temporary data/out dir), ``settings`` (placeholder namespace),
``tmp_data_dir``, ``engine`` (``LIFEngine`` on the tiny connectome, noise off), ``sim_snapshot`` (helper
building a ``MarketSnapshot``) and ``dex_pair_json`` (the RESEARCH section 8 sample). Markers ``slow`` and
``optional`` are registered here so ``-m "not slow"`` works without a pytest.ini entry.

The tiny connectome is HAND-BUILT here (no import of ``flybrain.connectome.synthetic``) so that every
foundation test runs with numpy only. It uses REAL MaleCNS type strings from SPEC section c.4 so that
every ``READOUTS`` group (and every ``validate_groups`` requirement) resolves non-empty, sides L/R
(untyped filler partly midline), all 8 regions, ~3200 edges with lognormal weights, and ~29 %
inhibitory neurons. It is a superset of the SPEC h.1 sketch (40 typed cells): the extra typed cells are
what ``test_groups`` / ``test_monitor`` rely on (``test_groups`` pins ``lc_loom == 12``, i.e. ``LC4`` x6 +
``LPLC2`` x6, so the h.1 count of 10 each is not used; the LC -> DNp01 weight is raised instead so the
150 Hz looming gate fires the giant fibre inside 20 ms on this fixture).

Two further deviations from the h.1 sketch are deliberate and load-bearing for other suites: the random
background is 3,000 edges with ``LogNormal(ln 4, 0.9)`` weights rather than h.1's "4,000 random edges
(weight ~ Geometric(0.25))": same mean (4 synapses), but the sparser draw keeps the engineered pathway
weights that ``test_patches`` / ``test_csr`` pin exact (e.g. ``DNp01 -> TTMn`` is 45 + the 300 gap-junction
patch = 345 with no random edge summed into that pair); and h.1's "30 % inhibitory" is realised as 29 % of
NEURONS carrying an inhibitory NT, because ``sign`` is per presynaptic neuron in the data model
(SPEC c.2) and cannot be set per edge. Raising the edge count or switching the weight law therefore
breaks foundation tests, not this fixture.

Provenance of the fixture's biology: type strings and NT signs follow RESEARCH section 4 ``[V]``
(GABAergic dn_halt / APL / MBON11 / ER4d / mAL / vPR9 / feed_pre_inh / sugar2_inh / lLN;
glutamatergic DNb01 / TTMn / L1 / MBON01,03 / Delta7 / GNG087 / Mi9; histaminergic R1-R6, R7/R8;
dopaminergic PAM / PPL1 / DPM); motor neurons glutamatergic ``[L]``; every edge, count and weight
is ``[E]`` engineered for tests only and is NOT connectome data.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from flybrain.connectome.csr import sum_duplicates
from flybrain.connectome.groups import resolve_groups
from flybrain.connectome.schema import (
    REGION_ID,
    SYNTHETIC_BODY_BASE,
    Connectome,
    count_groups,
    count_regions,
    utc_now_iso,
)

TINY_N = 400
TINY_SEED = 12345


def pytest_configure(config: Any) -> None:
    """Register the markers of SPEC section h (``slow`` > 25 s budget, ``optional`` = needs an optional package)."""
    config.addinivalue_line("markers", "slow: takes more than a few seconds (20k connectome, long simulations)")
    config.addinivalue_line("markers", "optional: needs an optional dependency (torch, anthropic, tweepy, pyarrow)")


#: (type, region, nt, count); count is both sides combined, split alternately L, R, L, R ...
TINY_TYPES: tuple[tuple[str, str, str, int], ...] = (
    # descending / motor ---------------------------------------------------------
    ("DNp01", "descending_motor", "acetylcholine", 2),
    ("DNp02", "descending_motor", "acetylcholine", 2),
    ("DNp04", "descending_motor", "acetylcholine", 2),
    ("DNp11", "descending_motor", "acetylcholine", 2),
    ("DNp03", "descending_motor", "acetylcholine", 2),
    ("DNp07", "descending_motor", "acetylcholine", 2),
    ("DNp10", "descending_motor", "acetylcholine", 2),
    ("DNp09", "descending_motor", "acetylcholine", 2),
    ("DNg100", "descending_motor", "acetylcholine", 2),
    ("DNge053", "descending_motor", "acetylcholine", 2),
    ("DNg97", "descending_motor", "acetylcholine", 2),
    ("MDN", "descending_motor", "acetylcholine", 4),
    ("DNg60", "descending_motor", "gaba", 2),
    ("DNg74_a", "descending_motor", "gaba", 2),
    ("DNg74_b", "descending_motor", "gaba", 2),
    ("DNa02", "descending_motor", "acetylcholine", 2),
    ("DNa01", "descending_motor", "acetylcholine", 2),
    ("DNa03", "descending_motor", "acetylcholine", 2),
    ("DNb01", "descending_motor", "glutamate", 2),
    ("DNg13", "descending_motor", "acetylcholine", 2),
    ("DNg02_a", "descending_motor", "acetylcholine", 4),
    ("DNg02_b", "descending_motor", "acetylcholine", 2),
    ("DNg02_c", "descending_motor", "acetylcholine", 2),
    ("DNg62", "descending_motor", "acetylcholine", 2),
    ("DNge078", "descending_motor", "acetylcholine", 2),
    ("DNge062", "descending_motor", "acetylcholine", 2),
    ("DNge080", "descending_motor", "acetylcholine", 2),
    ("DNg67", "descending_motor", "acetylcholine", 2),
    ("pIP10", "descending_motor", "acetylcholine", 2),
    ("pMP2", "descending_motor", "acetylcholine", 2),
    ("TTMn", "descending_motor", "glutamate", 2),
    ("PSI", "descending_motor", "unknown", 2),
    ("GFC2", "descending_motor", "acetylcholine", 4),
    ("DLMn a, b", "descending_motor", "glutamate", 2),
    ("DLMn c-f", "descending_motor", "glutamate", 4),
    ("DVMn 1a-c", "descending_motor", "glutamate", 2),
    ("b1 MN", "descending_motor", "glutamate", 2),
    ("i1 MN", "descending_motor", "glutamate", 2),
    ("hg1 MN", "descending_motor", "glutamate", 2),
    ("tp2 MN", "descending_motor", "glutamate", 2),
    ("ps1 MN", "descending_motor", "glutamate", 2),
    ("MNwm36", "descending_motor", "glutamate", 2),
    ("Ti flexor MN", "descending_motor", "glutamate", 2),
    ("Ti extensor MN", "descending_motor", "glutamate", 2),
    ("Sternal anterior rotator MN", "descending_motor", "glutamate", 4),
    # SEZ ---------------------------------------------------------------------------
    ("MN9", "sez", "glutamate", 2),
    ("GNG108", "sez", "acetylcholine", 2),
    ("GNG120", "sez", "acetylcholine", 2),
    ("GNG015", "sez", "gaba", 2),
    ("GNG095", "sez", "gaba", 2),
    ("GNG215", "sez", "acetylcholine", 2),
    ("GNG232", "sez", "acetylcholine", 2),
    ("GNG042", "sez", "gaba", 2),
    ("GNG016", "sez", "unknown", 2),
    ("GNG087", "sez", "glutamate", 2),
    ("LB3b", "sez", "acetylcholine", 4),
    ("LB3c", "sez", "acetylcholine", 4),
    ("PhG1a", "sez", "acetylcholine", 2),
    ("LB3a", "sez", "acetylcholine", 4),
    ("LB1a", "sez", "acetylcholine", 4),
    ("LB1c", "sez", "acetylcholine", 4),
    ("SAD093", "sez", "acetylcholine", 2),
    ("GNG458", "sez", "acetylcholine", 2),
    # VNC ---------------------------------------------------------------------------
    ("AN13B002", "vnc", "gaba", 2),
    ("LgLG3", "vnc", "acetylcholine", 4),
    ("WG2", "vnc", "acetylcholine", 4),
    ("LgAG1", "vnc", "acetylcholine", 4),
    ("LgLG1a", "vnc", "acetylcholine", 4),
    ("LgLG2", "vnc", "acetylcholine", 4),
    ("dPR1", "vnc", "acetylcholine", 2),
    ("dMS2", "vnc", "acetylcholine", 4),
    ("vPR9_a", "vnc", "gaba", 2),
    ("AN03A008", "vnc", "acetylcholine", 2),
    ("AN04B003", "vnc", "acetylcholine", 2),
    ("IN13A001", "vnc", "acetylcholine", 2),
    ("IN08A002", "vnc", "acetylcholine", 2),
    ("AN05B102a", "vnc", "acetylcholine", 2),
    # central other ---------------------------------------------------------------
    ("JO-A1", "central_other", "acetylcholine", 4),
    ("JO-A2", "central_other", "acetylcholine", 4),
    ("JO-CM", "central_other", "acetylcholine", 4),
    ("JO-EV1", "central_other", "acetylcholine", 2),
    ("BM", "central_other", "acetylcholine", 2),
    ("pC1_14a", "central_other", "acetylcholine", 4),
    ("pC1_1", "central_other", "acetylcholine", 2),
    ("mAL_m8", "central_other", "gaba", 4),
    ("LAL083", "central_other", "acetylcholine", 2),
    ("PS049", "central_other", "acetylcholine", 2),
    ("aIPg1", "central_other", "acetylcholine", 2),
    ("AVLP732m", "central_other", "acetylcholine", 2),
    # optic lobe ---------------------------------------------------------------------
    ("R1-R6", "optic_lobe", "histamine", 8),
    ("R7p", "optic_lobe", "histamine", 2),
    ("R8y", "optic_lobe", "histamine", 2),
    ("L1", "optic_lobe", "glutamate", 4),
    ("L2", "optic_lobe", "acetylcholine", 4),
    ("Mi1", "optic_lobe", "acetylcholine", 4),
    ("Tm3", "optic_lobe", "acetylcholine", 2),
    ("Mi9", "optic_lobe", "glutamate", 2),
    ("Tm9", "optic_lobe", "glutamate", 2),
    ("T4a", "optic_lobe", "acetylcholine", 2),
    ("T4b", "optic_lobe", "acetylcholine", 2),
    ("T5a", "optic_lobe", "acetylcholine", 2),
    ("T5b", "optic_lobe", "acetylcholine", 2),
    ("LC4", "optic_lobe", "acetylcholine", 6),
    ("LPLC2", "optic_lobe", "acetylcholine", 6),
    ("LC6", "optic_lobe", "acetylcholine", 4),
    ("LPLC1", "optic_lobe", "acetylcholine", 2),
    ("LC9", "optic_lobe", "acetylcholine", 4),
    ("LC31a", "optic_lobe", "acetylcholine", 2),
    # antennal lobe ------------------------------------------------------------------
    ("ORN_DM1", "antennal_lobe", "acetylcholine", 4),
    ("ORN_DA1", "antennal_lobe", "acetylcholine", 4),
    ("DM1_lPN", "antennal_lobe", "acetylcholine", 4),
    ("DA1_lPN", "antennal_lobe", "acetylcholine", 2),
    ("VP2+Z_lvPN", "antennal_lobe", "acetylcholine", 2),
    ("lLN1", "antennal_lobe", "gaba", 4),
    ("v2LN30", "antennal_lobe", "gaba", 2),
    # mushroom body ------------------------------------------------------------------
    ("KCg-m", "mushroom_body", "acetylcholine", 12),
    ("KCab-c", "mushroom_body", "acetylcholine", 4),
    ("APL", "mushroom_body", "gaba", 2),
    ("DPM", "mushroom_body", "dopamine", 2),
    ("MBON01", "mushroom_body", "glutamate", 2),
    ("MBON03", "mushroom_body", "glutamate", 2),
    ("MBON11", "mushroom_body", "gaba", 2),
    ("MBON12", "mushroom_body", "acetylcholine", 2),
    ("PAM01", "mushroom_body", "dopamine", 4),
    ("PAM02", "mushroom_body", "dopamine", 2),
    ("PPL101", "mushroom_body", "dopamine", 2),
    ("PPL103", "mushroom_body", "dopamine", 2),
    # central complex ----------------------------------------------------------------
    ("EPG", "central_complex", "acetylcholine", 6),
    ("PEN_a(PEN1)", "central_complex", "acetylcholine", 4),
    ("PEN_b(PEN2)", "central_complex", "acetylcholine", 2),
    ("PEG", "central_complex", "acetylcholine", 2),
    ("Delta7", "central_complex", "glutamate", 4),
    ("ER4d", "central_complex", "gaba", 4),
    ("ER3a", "central_complex", "gaba", 2),
    ("PFL3", "central_complex", "acetylcholine", 4),
    ("PFL2", "central_complex", "acetylcholine", 2),
    ("PFL1", "central_complex", "acetylcholine", 2),
    ("hDeltaA", "central_complex", "acetylcholine", 2),
    ("PFNd", "central_complex", "acetylcholine", 4),
    ("ExR1", "central_complex", "gaba", 2),
    ("FB6A", "central_complex", "acetylcholine", 2),
    ("FB7B", "central_complex", "acetylcholine", 2),
)

#: untyped filler: (region, side, nt, count); includes midline (side 0) neurons.
TINY_FILLER: tuple[tuple[str, int, str, int], ...] = (
    ("central_other", 0, "gaba", 6),
    ("central_other", -1, "gaba", 1),
    ("central_other", 1, "gaba", 1),
    ("optic_lobe", -1, "gaba", 1),
    ("optic_lobe", 1, "gaba", 1),
    ("vnc", 0, "gaba", 2),
    ("vnc", -1, "gaba", 1),
    ("vnc", 1, "gaba", 1),
)

INHIBITORY_NT = frozenset({"gaba", "glutamate", "histamine"})

#: engineered [E] pathway edges: (pre type regex, post type regex, weight, laterality ipsi|contra|both)
TINY_PATHWAYS: tuple[tuple[str, str, float, str], ...] = (
    ("LC4", "DNp01", 40.0, "ipsi"),      # 3 ipsi LC4 + 3 LPLC2 per GF: a 150 Hz loom fires DNp01 inside 20 ms
    ("LPLC2", "DNp01", 40.0, "ipsi"),
    ("JO-A1", "DNp01", 15.0, "ipsi"),
    ("DNp01", "TTMn", 45.0, "ipsi"),
    ("DNp01", "PSI", 4.0, "ipsi"),
    ("DNp01", "GFC2", 10.0, "ipsi"),
    ("LC9", "DNp09", 10.0, "ipsi"),
    ("(LB3b|LB3c)", "(GNG215|GNG232)", 15.0, "ipsi"),
    ("(GNG215|GNG232)", "(GNG108|GNG120)", 12.0, "ipsi"),
    ("(GNG108|GNG120)", "MN9", 30.0, "ipsi"),
    ("DNge062", "MN9", 40.0, "ipsi"),
    ("(GNG015|GNG095)", "MN9", 30.0, "ipsi"),
    ("GNG042", "GNG015", 20.0, "ipsi"),
    ("(LB1a|LB1c)", "GNG016", 12.0, "ipsi"),
    ("PFL3", "DNa02", 30.0, "contra"),
    ("PFL3", "DNb01", 40.0, "contra"),
    ("DNa02", "Sternal anterior rotator MN", 25.0, "ipsi"),
    ("DNg100", "Ti flexor MN", 10.0, "ipsi"),
    ("DNg02_a", "MNwm36", 40.0, "ipsi"),
    ("DNg02_a", "tp2 MN", 30.0, "ipsi"),
    ("DNg02_a", "ps1 MN", 20.0, "ipsi"),
    ("DNg02_a", "DLMn c-f", 5.0, "ipsi"),
    ("EPG", "PEN_a[(]PEN1[)]", 10.0, "ipsi"),
    ("PEN_a[(]PEN1[)]", "EPG", 10.0, "ipsi"),
    ("Delta7", "EPG", 8.0, "both"),
    ("KCg-m", "(MBON01|MBON11)", 5.0, "both"),
    ("ORN_DM1", "DM1_lPN", 30.0, "ipsi"),
    ("lLN1", "DM1_lPN", 8.0, "both"),
    ("pC1_14a", "pIP10", 20.0, "ipsi"),
    ("pIP10", "dMS2", 20.0, "ipsi"),
    ("dMS2", "hg1 MN", 40.0, "ipsi"),
    ("R1-R6", "(L1|L2)", 10.0, "ipsi"),
    ("L1", "Mi1", 8.0, "ipsi"),
    ("L2", "Tm3", 8.0, "ipsi"),
    ("(Mi1|Tm3)", "T4[ab]", 6.0, "ipsi"),
    ("T4[ab]", "LPLC2", 6.0, "ipsi"),
    ("T5[ab]", "LC4", 6.0, "ipsi"),
    ("JO-CM", "SAD093", 10.0, "ipsi"),
    ("SAD093", "DNg62", 10.0, "ipsi"),
)


def build_tiny_connectome(n: int = TINY_N, seed: int = TINY_SEED, n_random_edges: int = 3000) -> Connectome:
    """Deterministic hand-built test connectome (see module docstring). Returns a validated ``Connectome``."""
    rng = np.random.default_rng(seed)
    types: list[str] = [""]  # index 0 = untyped filler
    type_of: dict[str, int] = {"": 0}
    type_idx: list[int] = []
    region: list[int] = []
    side: list[int] = []
    nt: list[str] = []
    for tname, reg, tnt, count in TINY_TYPES:
        if tname not in type_of:
            type_of[tname] = len(types)
            types.append(tname)
        for k in range(count):
            type_idx.append(type_of[tname])
            region.append(REGION_ID[reg])
            side.append(-1 if k % 2 == 0 else 1)
            nt.append(tnt)
    for reg, s, fnt, count in TINY_FILLER:
        for _ in range(count):
            type_idx.append(0)
            region.append(REGION_ID[reg])
            side.append(s)
            nt.append(fnt)
    if len(type_idx) != n:
        raise AssertionError(f"tiny connectome table yields {len(type_idx)} neurons, expected {n}")

    type_idx_arr = np.asarray(type_idx, dtype=np.int32)
    region_arr = np.asarray(region, dtype=np.uint8)
    side_arr = np.asarray(side, dtype=np.int8)
    nt_arr = np.asarray(nt, dtype=object)
    sign = np.where(np.isin(nt_arr.astype(str), list(INHIBITORY_NT)), -1.0, 1.0).astype(np.float32)
    body_id = (SYNTHETIC_BODY_BASE + np.arange(n, dtype=np.int64)).astype(np.int64)

    # --- edges: random background (lognormal weights) + engineered pathways ---------------
    pre_l = [rng.integers(0, n, size=n_random_edges)]
    post_l = [rng.integers(0, n, size=n_random_edges)]
    w_l = [np.maximum(1.0, np.round(rng.lognormal(mean=np.log(4.0), sigma=0.9, size=n_random_edges)))]

    def _members(regex: str) -> np.ndarray:
        import re as _re

        pat = _re.compile(regex)
        tmask = np.fromiter((pat.fullmatch(t) is not None for t in types), dtype=bool, count=len(types))
        return np.flatnonzero(tmask[type_idx_arr])

    for src_rx, dst_rx, w, lat in TINY_PATHWAYS:
        src, dst = _members(src_rx), _members(dst_rx)
        if src.size == 0 or dst.size == 0:
            raise AssertionError(f"pathway {src_rx}->{dst_rx} has no members in the tiny table")
        pp, qq = np.meshgrid(src, dst, indexing="ij")
        pp, qq = pp.ravel(), qq.ravel()
        if lat == "ipsi":
            m = side_arr[pp] == side_arr[qq]
        elif lat == "contra":
            m = side_arr[pp] == -side_arr[qq]
        else:
            m = np.ones(pp.shape, dtype=bool)
        m &= pp != qq
        pre_l.append(pp[m])
        post_l.append(qq[m])
        w_l.append(np.full(int(m.sum()), w, dtype=np.float64))

    pre_all = np.concatenate(pre_l)
    post_all = np.concatenate(post_l)
    w_all = np.concatenate(w_l)
    pre, post, weight = sum_duplicates(pre_all, post_all, w_all, n)  # sorted, merged, no self-loops

    groups = resolve_groups(types, type_idx_arr, side_arr)
    meta = {
        "e": int(pre.shape[0]),
        "synapses": float(weight.sum(dtype=np.float64)),
        "license": "synthetic (no data)",
        "citation": None,
        "seed": int(seed),
        "build_args": {"n": int(n), "seed": int(seed), "n_random_edges": int(n_random_edges), "fixture": "tiny"},
        "patches_applied": [],
        "gain_default": 1.0,
        "weights_mode": "fixture",
        "engineered_edges": [f"{a}->{b}" for a, b, _, _ in TINY_PATHWAYS],
        "region_counts": count_regions(region_arr),
        "group_counts": count_groups(groups),
        "created": utc_now_iso(),
        "note": "hand-built test fixture; not connectome data",
    }
    conn = Connectome(
        name=f"tiny-{n}-s{seed}",
        source="synthetic",
        n=int(n),
        pre=pre.astype(np.int32),
        post=post.astype(np.int32),
        weight=weight.astype(np.float32),
        sign=sign,
        region=region_arr,
        side=side_arr,
        types=types,
        type_idx=type_idx_arr,
        body_id=body_id,
        nt=nt_arr,
        groups=groups,
        meta=meta,
    )
    conn.validate()
    return conn


# --------------------------------------------------------------------------- connectome fixtures


@pytest.fixture
def tiny_connectome() -> Connectome:
    """Deterministic hand-built connectome, N=400, ~3200 edges. Function-scoped (a fresh instance per test) because
    some tests mutate ``groups`` / ``meta``; building it costs a few milliseconds."""
    return build_tiny_connectome()


@pytest.fixture(scope="session")
def small_synthetic() -> Connectome:
    """``build_synthetic(n_neurons=4000, seed=1)`` (core 2,830 + 1,170 filler, < 1 s). Shared across the session:
    treat it as read-only."""
    from flybrain.connectome.synthetic import build_synthetic

    return build_synthetic(n_neurons=4000, seed=1)


@pytest.fixture(scope="session")
def synthetic_20k(tmp_path_factory: pytest.TempPathFactory) -> Connectome:
    """``build_synthetic(20_000, seed=1337)``, built once per session and cached (``Connectome.save``) under
    ``tmp_path_factory``; used by the ``slow`` tests. Treat it as read-only."""
    from flybrain.connectome.synthetic import build_synthetic

    conn = build_synthetic(n_neurons=20_000, seed=1337)
    stem = tmp_path_factory.mktemp("connectome_cache") / conn.name
    try:
        conn.save(stem)
    except OSError:  # a full temp disk must not fail the whole session
        pass
    return conn


# --------------------------------------------------------------------------- settings / dirs


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Empty per-test data directory with the ``cache/`` and ``sessions/`` sub-dirs of SPEC section a."""
    d = tmp_path / "data"
    for sub in ("cache", "sessions", "snapshots", "connectome"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def settings(tmp_path: Path) -> SimpleNamespace:
    """Minimal settings namespace (data_dir / out_dir under tmp_path, seed 0) for modules that only ``getattr`` the
    fields they need (calibrate, cache). ``settings_tmp`` is the full ``Settings`` of SPEC h.1."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(data_dir=data_dir, out_dir=out_dir, seed=0)


@pytest.fixture
def settings_tmp(tmp_path: Path):
    """SPEC h.1: ``load_settings(env={FLY_DATA_DIR, FLY_OUT_DIR, FLY_SESSION_LOG=0, FLY_REALTIME=0, FLY_MARKET=sim,
    FLY_LLM=dryrun, FLY_X=dryrun}, dotenv=None)``. While ``flybrain.config`` is still the E3 stub (no
    ``load_settings``), the same values are put into the stub's ``Settings`` dataclass directly."""
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    env = {
        "FLY_DATA_DIR": str(data_dir),
        "FLY_OUT_DIR": str(out_dir),
        "FLY_SESSION_LOG": "0",
        "FLY_REALTIME": "0",
        "FLY_MARKET": "sim",
        "FLY_LLM": "dryrun",
        "FLY_X": "dryrun",
    }
    import flybrain.config as config_mod

    load_settings = getattr(config_mod, "load_settings", None)
    if callable(load_settings):
        try:
            return load_settings(env=env, dotenv=None)
        except (TypeError, NotImplementedError):  # a stub with the wrong signature: fall through
            pass
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return config_mod.Settings(
        data_dir=data_dir, out_dir=out_dir, session_log=False, realtime=False, market="sim", llm="dryrun",
        x_mode="dryrun",
    )


# --------------------------------------------------------------------------- engine


@pytest.fixture
def engine(tiny_connectome: Connectome):
    """``LIFEngine(tiny_connectome, seed=0, dt_ms=1.0, noise_sigma=0.0)`` (noise off unless the test turns it on).
    Skipped while ``flybrain.snn.engine`` is not importable."""
    engine_mod = pytest.importorskip("flybrain.snn.engine")
    return engine_mod.LIFEngine(tiny_connectome, seed=0, dt_ms=1.0, noise_sigma=0.0)


# --------------------------------------------------------------------------- market helpers


@pytest.fixture
def sim_snapshot() -> Callable[..., Any]:
    """Helper ``sim_snapshot(seq, **overrides) -> MarketSnapshot`` with sane simulated-market values
    (SPEC h.1). Field names are those of ``flybrain.market.base.MarketSnapshot``."""
    base_mod = pytest.importorskip("flybrain.market.base")

    def _make(seq: int = 1, **overrides: Any) -> Any:
        values: dict[str, Any] = {
            "ts": 1_757_500_000.0 + 60.0 * float(seq),
            "source": "sim",
            "chain": "sim",
            "dex": "sim",
            "pair": "SIM",
            "symbol": "FLY",
            "price_usd": 0.0012345,
            "price_native": 0.0012345,
            "buys_m5": 41,
            "sells_m5": 27,
            "buys_h1": 480,
            "sells_h1": 350,
            "chg_m5": 1.2,
            "chg_h1": 3.5,
            "chg_h6": -2.0,
            "chg_h24": 8.0,
            "vol_m5": 4_200.0,
            "vol_h1": 51_000.0,
            "liq_usd": 250_000.0,
            "fdv": 1_234_500.0,
            "mcap": 1_234_500.0,
            "regime": "CALM",
            "seq": int(seq),
        }
        values.update(overrides)
        return base_mod.MarketSnapshot(**values)

    return _make


@pytest.fixture
def dex_pair_json() -> dict:
    """The verified DexScreener pair object of RESEARCH section 8 (Wrapped SOL / USDC on Raydium) as a dict:
    ``priceUsd`` / ``priceNative`` are strings, ``fdv`` / ``marketCap`` absent."""
    return {
        "chainId": "solana",
        "dexId": "raydium",
        "url": "https://dexscreener.com/solana/58oqchx4ywmvkdwllzzbi4choccc2fqcuwbkwmihlyqo2",
        "pairAddress": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
        "baseToken": {"address": "So11111111111111111111111111111111111111112", "name": "Wrapped SOL", "symbol": "SOL"},
        "quoteToken": {"address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "name": "USD Coin", "symbol": "USDC"},
        "priceNative": "100.09019",
        "priceUsd": "100.090",
        "txns": {
            "m5": {"buys": 648, "sells": 656},
            "h1": {"buys": 26649, "sells": 27601},
            "h6": {"buys": 96602, "sells": 97927},
            "h24": {"buys": 482322, "sells": 486935},
        },
        "volume": {"h24": 82556292.09, "h6": 12671063.94, "h1": 3444547.17, "m5": 83071.16},
        "priceChange": {"m5": 0.46, "h1": -0.05, "h6": -0.73, "h24": -3.39},
        "liquidity": {"usd": 25108908.33, "base": 125588, "quote": 12538748},
        "pairCreatedAt": 1669602450000,
        "info": {
            "imageUrl": "",
            "header": "",
            "openGraph": "",
            "websites": [{"url": "https://solana.com", "label": "Website"}],
            "socials": [{"url": "https://x.com/solana", "type": "twitter"}],
        },
    }
