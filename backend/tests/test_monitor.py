"""Tests for flybrain.snn.monitor (SPEC section c.12): raster sample, counts, EMA rates, wire order (d.1/d.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flybrain.connectome.groups import READOUTS, SIDED, STAR_TYPES, readout_partition, readout_sizes
from flybrain.connectome.schema import REGION_ID, REGIONS
from flybrain.snn.engine import LIFEngine
from flybrain.snn.monitor import (
    DERIVED_KEYS,
    DT_SAMPLE_MAX,
    RasterRow,
    RateEstimator,
    SpikeMonitor,
    StepStats,
)


# --------------------------------------------------------------------------- raster


def test_raster_rows_layout(tiny_connectome):
    per = 12
    mon = SpikeMonitor(tiny_connectome, per_region=per, cap=100, seed=1, star_per_side=None)
    assert len(mon.rows) == 8 * per
    for i, row in enumerate(mon.rows):
        assert isinstance(row, RasterRow)
        assert row.slot == i and row.region == i // per
        assert row.side in ("L", "R", "M")
        if row.neuron >= 0:
            assert mon.slot_of_neuron[row.neuron] == i
            assert tiny_connectome.region[row.neuron] == row.region
            assert row.label == tiny_connectome.type_of(row.neuron)
            assert row.star == (row.label in STAR_TYPES)
            assert row.side == {-1: "L", 1: "R", 0: "M"}[int(tiny_connectome.side[row.neuron])]
        else:
            assert row.label == "" and row.star is False
    sampled = np.flatnonzero(mon.slot_of_neuron >= 0)
    assert sampled.shape[0] == sum(1 for r in mon.rows if r.neuron >= 0)
    assert np.unique(mon.slot_of_neuron[sampled]).shape[0] == sampled.shape[0]
    # every region fills up to per_region (the fixture has >= 12 neurons per region), stars first
    for r in range(8):
        lane = mon.rows[r * per:(r + 1) * per]
        assert all(row.neuron >= 0 for row in lane)
        stars = [row.star for row in lane]
        assert stars == sorted(stars, reverse=True)
    # DNp01 (STAR) rows in descending_motor: L before R, first lane rows; star order follows STAR_TYPES
    dm = REGION_ID["descending_motor"]
    lane = mon.rows[dm * per:(dm + 1) * per]
    assert lane[0].label == "DNp01" and lane[0].side == "L" and lane[0].star
    assert lane[1].label == "DNp01" and lane[1].side == "R"
    labels = [row.label for row in lane if row.star]
    ranks = [STAR_TYPES.index(lb) for lb in labels]
    assert ranks == sorted(ranks)
    # without a cap every star neuron of the region comes first (DNg02_a has 2 per side in the fixture;
    # with 12 rows the lane is truncated after the 6th star type, so check on a 48-row lane)
    wide = SpikeMonitor(tiny_connectome, per_region=48, cap=100, seed=1, star_per_side=None)
    lane48 = wide.rows[dm * 48:(dm + 1) * 48]
    assert sum(1 for row in lane48 if row.label == "DNg02_a") == 4
    assert sum(1 for row in lane48 if row.star) == sum(
        1 for i in np.flatnonzero(tiny_connectome.region == dm) if tiny_connectome.type_of(int(i)) in STAR_TYPES)


def test_raster_star_cap(tiny_connectome):
    """``star_per_side`` limits the star rows per (type, side) so large star populations cannot fill a lane."""
    per = 48
    dm = REGION_ID["descending_motor"]
    capped = SpikeMonitor(tiny_connectome, per_region=per, seed=1, star_per_side=1)
    lane = capped.rows[dm * per:(dm + 1) * per]
    assert sum(1 for row in lane if row.label == "DNg02_a") == 2
    assert [row.side for row in lane if row.label == "DNg02_a"] == ["L", "R"]
    assert lane[0].label == "DNp01" and lane[1].label == "DNp01"
    # the DEFAULT is the literal SPEC c.12 rule: no cap, every star neuron of the region first
    default = SpikeMonitor(tiny_connectome, per_region=per, seed=1)
    assert default.star_per_side is None
    lane = default.rows[dm * per:(dm + 1) * per]
    assert sum(1 for row in lane if row.label == "DNg02_a") == 4
    stars_in_region = sum(1 for i in np.flatnonzero(tiny_connectome.region == dm)
                          if tiny_connectome.type_of(int(i)) in STAR_TYPES)
    assert sum(1 for row in lane if row.star) == min(per, stars_in_region)
    two = SpikeMonitor(tiny_connectome, per_region=per, seed=1, star_per_side=2)
    assert two.star_per_side == 2
    assert sum(1 for row in two.rows[dm * per:(dm + 1) * per] if row.label == "DNg02_a") == 4
    # the remaining rows of the lane are a random sample of non-star cells of that region
    rest = [row for row in lane if not row.star and row.neuron >= 0]
    assert rest and all(tiny_connectome.region[row.neuron] == dm for row in rest)


def test_raster_seed_determinism_and_small_regions(tiny_connectome):
    a = SpikeMonitor(tiny_connectome, per_region=48, seed=7)
    b = SpikeMonitor(tiny_connectome, per_region=48, seed=7)
    c = SpikeMonitor(tiny_connectome, per_region=48, seed=8)
    assert a.rows == b.rows
    assert np.array_equal(a.slot_of_neuron, b.slot_of_neuron)
    assert a.rows != c.rows
    # a region with fewer neurons than per_region leaves empty slots (neuron == -1)
    big = SpikeMonitor(tiny_connectome, per_region=128, seed=1)
    counts = np.bincount(tiny_connectome.region.astype(np.int64), minlength=8)
    for r in range(8):
        lane = big.rows[r * 128:(r + 1) * 128]
        filled = sum(1 for row in lane if row.neuron >= 0)
        assert filled == min(128, int(counts[r]))
        assert all(row.neuron == -1 for row in lane[filled:])
    wire = big.rows_wire()
    assert len(wire) == 8 * 128 and all(w["slot"] == i for i, w in enumerate(wire))
    assert wire[0] == {"region": 0, "slot": 0, "neuron": big.rows[0].neuron, "label": big.rows[0].label,
                       "side": big.rows[0].side, "star": big.rows[0].star}


def test_raster_cap_uniform_subsample(tiny_connectome):
    eng = LIFEngine(tiny_connectome, seed=0, monitor=SpikeMonitor(tiny_connectome, per_region=48, cap=25, seed=0))
    eng.inject(np.arange(tiny_connectome.n, dtype=np.int32), rate_hz=400.0, duration_ms=100.0, tag="all")
    s = eng.step(50)
    assert s.capped is True
    assert s.spike_indices_sample.shape == (25,) and s.spike_dt_sample.shape == (25,)
    assert np.all(np.diff(s.spike_dt_sample) >= 0)
    assert s.spike_dt_sample[0] <= 5 and s.spike_dt_sample[-1] >= 40   # spread over the window
    assert s.total_spikes > 25
    assert np.all(s.spike_indices_sample >= 0) and np.all(s.spike_indices_sample < 8 * 48)
    # below the cap nothing is dropped
    quiet = LIFEngine(tiny_connectome, seed=0, noise_mu=0.0, noise_sigma=0.0,
                      monitor=SpikeMonitor(tiny_connectome, per_region=48, cap=25, seed=0))
    quiet.kick("gf")
    s = quiet.step(20)
    assert s.capped is False and s.total_spikes == tiny_connectome.groups["gf"].size
    assert s.spike_indices_sample.shape[0] == s.total_spikes   # both DNp01 are star rows
    # cap 0 -> no raster events at all, still not "capped" when there was nothing to drop
    none = SpikeMonitor(tiny_connectome, per_region=48, cap=0, seed=0)
    none.begin(0)
    none.record(tiny_connectome.groups["gf"], 0)
    s = none.flush()
    assert s.spike_indices_sample.shape == (0,) and s.capped is True and s.total_spikes == 2


def test_spike_dt_sample_survives_long_windows(tiny_connectome):
    """Regression: ``spike_dt_sample`` is int16 (SPEC c.10) but the offsets are gathered in int32, so a
    window longer than 32768 steps can no longer wrap to negative dt values on the wire (d.2 says
    ``dt[k] = 0..steps-1``); offsets beyond the int16 range are clipped, never wrapped."""
    mon = SpikeMonitor(tiny_connectome, per_region=48, cap=10**7, seed=0, dt_ms=1.0)
    eng = LIFEngine(tiny_connectome, seed=1, dt_ms=1.0, monitor=mon)
    s = eng.step(40_000)
    d = s.spike_dt_sample
    assert d.dtype == np.int16
    assert d.size > 0 and int(d.min()) >= 0 and int(d.max()) <= DT_SAMPLE_MAX
    assert np.all(np.diff(d) >= 0)
    assert s.spike_indices_sample.shape == d.shape
    # a window that still fits stays exact (no clipping at all)
    eng2 = LIFEngine(tiny_connectome, seed=1, dt_ms=1.0,
                     monitor=SpikeMonitor(tiny_connectome, per_region=48, cap=10**7, seed=0, dt_ms=1.0))
    s2 = eng2.step(1000)
    assert int(s2.spike_dt_sample.max()) < 1000 and int(s2.spike_dt_sample.min()) >= 0


# --------------------------------------------------------------------------- counts


def test_record_flush_counts_match_manual(tiny_connectome):
    mon = SpikeMonitor(tiny_connectome, per_region=48, cap=2000, seed=0, dt_ms=1.0)
    g = tiny_connectome.groups
    spk0 = np.sort(np.concatenate([g["gf_L"], g["feed_mn"], g["kc"][:3]])).astype(np.int32)
    spk1 = np.sort(np.concatenate([g["gf_R"], g["gf_L"], g["pip10_L"], g["b1"], g["dng100"]])).astype(np.int32)
    spk2 = np.zeros(0, dtype=np.int32)
    mon.begin(100)
    mon.record(spk0, 0)
    mon.record(spk1, 1)
    mon.record(spk2, 2)
    s = mon.flush()
    assert isinstance(s, StepStats)
    assert (s.n_steps, s.t0_ms, s.t1_ms) == (3, 100, 103)
    assert s.total_spikes == spk0.size + spk1.size
    assert s.active_frac_max == pytest.approx(max(spk0.size, spk1.size) / tiny_connectome.n)
    c = s.spike_counts_by_group
    assert list(c) == mon.names
    assert c["gf_L"] == 2 * g["gf_L"].size and c["gf_R"] == g["gf_R"].size
    assert c["gf"] == c["gf_L"] + c["gf_R"]                       # sided folding
    assert s.gf_spikes == {"L": c["gf_L"], "R": c["gf_R"]}
    assert c["feed_mn"] == g["feed_mn"].size and c["kc"] == 3
    assert c["pip10"] == g["pip10_L"].size == c["pip10_L"] and c["pip10_R"] == 0
    assert c["song_dn"] == 0                                      # partition: pIP10 belongs to pip10, not song_dn
    assert c["b1"] == g["b1"].size and c["b1_L"] == g["b1_L"].size and c["b1_R"] == g["b1_R"].size
    assert c["wing_steer"] == 0                                   # b1 MN cells belong to the b1 readout
    assert c["dng100"] == g["dng100"].size and c["dn_fwd"] == 0   # DNg100 belongs to dng100, not dn_fwd
    assert c["escape_dn"] == 0 and c["ttmn"] == 0
    region_manual = np.bincount(tiny_connectome.region[np.concatenate([spk0, spk1])].astype(np.int64), minlength=8)
    assert np.array_equal(s.region_counts, region_manual)
    assert s.region_counts.dtype == np.int64
    # raster events: only sampled neurons, dt = step index
    for slot, dt in zip(s.spike_indices_sample, s.spike_dt_sample):
        row = mon.rows[slot]
        assert row.neuron in (spk0 if dt == 0 else spk1)
    assert s.capped is False
    # counters reset after flush
    mon.begin(103)
    s2 = mon.flush()
    assert s2.total_spikes == 0 and s2.n_steps == 0 and s2.active_frac_max == 0.0
    assert all(v == 0 for v in s2.spike_counts_by_group.values())


def test_partition_names_and_sizes(tiny_connectome):
    mon = SpikeMonitor(tiny_connectome, seed=0)
    pop_id, names = readout_partition(tiny_connectome.groups, tiny_connectome.n)
    assert mon.names == names and np.array_equal(mon.pop_id, pop_id)
    assert np.array_equal(mon.sizes, readout_sizes(pop_id, names))
    g = tiny_connectome.groups
    assert mon.size_of("gf") == g["gf"].size == mon.size_of("gf_L") + mon.size_of("gf_R")
    assert mon.size_of("hg1") == g["hg1"].size and mon.size_of("hg1_L") == g["hg1_L"].size
    assert mon.size_of("dn_fwd") == g["dn_fwd"].size - g["dng100"].size   # first match wins
    assert mon.size_of("song_dn") == g["song_dn"].size - g["pip10"].size
    assert mon.size_of("nope") == 0
    assert mon.rates.sizes.tolist() == mon.sizes.tolist()
    # every readout of the SPEC table is a non-empty population of the fixture
    assert all(mon.size_of(r) > 0 for r in READOUTS)


def test_wire_order_matches_spec_d2(tiny_connectome):
    mon = SpikeMonitor(tiny_connectome, seed=0)
    names = mon.names
    _, part = readout_partition(tiny_connectome.groups, tiny_connectome.n)
    assert names == part
    assert len(READOUTS) == 70 and sum(1 for r in READOUTS if r in SIDED) == 26
    assert len(names) == 70 + 2 * 26 == 122
    pos = {n: i for i, n in enumerate(names)}
    for r in READOUTS:
        assert r in pos
        if r in SIDED:
            assert pos[r + "_L"] == pos[r] + 1 and pos[r + "_R"] == pos[r] + 2
    # SPEC d.1 hello.pops prefix
    assert names[:24] == ["gf", "gf_L", "gf_R", "escape_dn", "escape_dn_L", "escape_dn_R", "dn_saccade", "dn_saccade_L",
                          "dn_saccade_R", "dn_land", "dn_freeze", "dn_freeze_L", "dn_freeze_R", "dng100", "dng100_L",
                          "dng100_R", "dn_fwd", "dn_fwd_L", "dn_fwd_R", "dn_back", "dn_halt", "steer_a02", "steer_a02_L",
                          "steer_a02_R"]
    keys = mon.rates.keys()
    assert keys == names + list(DERIVED_KEYS) and len(keys) == 131
    assert DERIVED_KEYS == ("lc_loom", "escape_vnc", "steer_a02_diff", "steer_a01_diff", "steer_g13_diff",
                            "b1_diff", "i1_diff", "hg1_diff", "dn_mean")
    assert list(mon.rates.snapshot().keys()) == keys
    assert names[-3:] == ["song_vnc", "an_steer", "leg_premotor"]


def test_gf_spikes_per_side(tiny_connectome):
    mon = SpikeMonitor(tiny_connectome, seed=0)
    g = tiny_connectome.groups
    mon.begin(0)
    mon.record(g["gf_L"], 0)
    mon.record(g["gf_L"], 1)
    mon.record(g["gf_R"], 2)
    s = mon.flush()
    assert s.gf_spikes == {"L": 2 * g["gf_L"].size, "R": g["gf_R"].size}
    assert s.spike_counts_by_group["gf"] == 3 * g["gf_L"].size


# --------------------------------------------------------------------------- rate estimator


def test_rate_estimator_ema_analytic():
    names = ["gf", "gf_L", "gf_R", "feed_mn", "kc", "steer_a02_L", "empty"]
    sizes = np.array([4, 2, 2, 10, 100, 3, 0])
    est = RateEstimator(names, sizes, tick_s=0.05)
    assert est.tau.tolist() == [0.05, 0.05, 0.05, 0.30, 0.20, 0.15, 0.20]
    counts = np.array([8, 4, 4, 5, 10, 3, 7])
    regs = np.zeros(8)
    rsz = np.full(8, 50)
    est.update(counts, regs, rsz, 0.05)
    for i, (n, sz) in enumerate(zip(names, sizes)):
        inst = counts[i] / (sz * 0.05) if sz else 0.0
        d = math.exp(-0.05 / est.tau[i])
        assert est.get(n) == pytest.approx(inst * (1 - d))
    assert est.get("empty") == 0.0
    # constant input converges to the instantaneous rate
    for _ in range(400):
        est.update(counts, regs, rsz, 0.05)
    assert est.get("gf") == pytest.approx(8 / (4 * 0.05), rel=1e-6)        # 40 Hz
    assert est.get("kc") == pytest.approx(10 / (100 * 0.05), rel=1e-6)     # 2 Hz
    # window scaling: 2 x 25 ms windows with half the counts == 1 x 50 ms window (stationary)
    e1 = RateEstimator(names, sizes, 0.05)
    e2 = RateEstimator(names, sizes, 0.05)
    for _ in range(100):
        e1.update(counts, regs, rsz, 0.05)
        e2.update(counts / 2, regs, rsz, 0.025)
        e2.update(counts / 2, regs, rsz, 0.025)
    assert e1.get("feed_mn") == pytest.approx(e2.get("feed_mn"), rel=1e-3)
    # default window = tick_s; zero window is ignored
    e3 = RateEstimator(names, sizes, 0.05)
    e3.update(counts, regs, rsz)
    first = 8 / (4 * 0.05) * (1 - math.exp(-1.0))
    assert e3.get("gf") == pytest.approx(first)
    e3.update(counts, regs, rsz, 0.0)
    assert e3.get("gf") == pytest.approx(first)
    e3.reset()
    assert e3.get("gf") == 0.0 and np.all(e3.regions() == 0)
    assert e3.get("unknown") == 0.0 and e3.size_of("unknown") == 0
    with pytest.raises(ValueError):
        RateEstimator(names, np.zeros(2), 0.05)


def test_rate_estimator_regions_and_derived():
    names = ["lc4", "lplc2", "ttmn", "psi", "gfc2", "steer_a02_L", "steer_a02_R", "b1_L", "b1_R"]
    sizes = np.array([10, 30, 1, 1, 1, 1, 1, 1, 1])
    est = RateEstimator(names, sizes, 0.05)
    region_sizes = np.array([100, 10, 10, 10, 10, 10, 20, 10])
    region_counts = np.array([50, 0, 0, 0, 0, 0, 10, 0])
    counts = np.array([10, 30, 1, 2, 3, 1, 3, 2, 5])
    for _ in range(500):
        est.update(counts, region_counts, region_sizes, 0.05)
    snap = est.snapshot()
    assert snap["lc4"] == pytest.approx(20.0) and snap["lplc2"] == pytest.approx(20.0)
    assert snap["lc_loom"] == pytest.approx(20.0)                                     # size-weighted mean
    assert snap["escape_vnc"] == pytest.approx((20 + 40 + 60) / 3.0)
    assert snap["steer_a02_diff"] == pytest.approx(60.0 - 20.0)
    assert snap["b1_diff"] == pytest.approx(100.0 - 40.0)
    assert snap["steer_a01_diff"] == 0.0 and snap["hg1_diff"] == 0.0 and snap["i1_diff"] == 0.0
    regs = est.regions()
    assert regs.dtype == np.float32 and regs.shape == (8,)
    assert regs[0] == pytest.approx(50 / (100 * 0.05)) and regs[6] == pytest.approx(10 / (20 * 0.05))
    assert snap["dn_mean"] == pytest.approx(float(regs[6]))
    assert list(snap.keys()) == names + list(DERIVED_KEYS)
    # SPEC c.4 calls the derived block ``derived()``, c.12 folds it into ``snapshot()``: same numbers
    der = est.derived()
    assert list(der.keys()) == list(DERIVED_KEYS)
    assert all(der[k] == snap[k] for k in DERIVED_KEYS)
    # size-weighted lc_loom with unequal rates
    e2 = RateEstimator(names, sizes, 0.05)
    for _ in range(500):
        e2.update(np.array([10, 0, 0, 0, 0, 0, 0, 0, 0]), region_counts, region_sizes, 0.05)
    assert e2.snapshot()["lc_loom"] == pytest.approx(20.0 * 10 / 40)
    # region EMA time constant 0.25 s: after one 50 ms window the fraction is 1 - exp(-0.2)
    e = RateEstimator(names, sizes, 0.05)
    e.update(counts, region_counts, region_sizes, 0.05)
    assert e.regions()[0] == pytest.approx(10.0 * (1 - math.exp(-0.05 / 0.25)), rel=1e-5)
    # derived keys are zero (not KeyError) when the base readouts are absent
    e3 = RateEstimator(["kc"], np.array([5]), 0.05)
    s3 = e3.snapshot()
    assert s3["lc_loom"] == 0.0 and s3["escape_vnc"] == 0.0 and s3["steer_a02_diff"] == 0.0


def test_tau_table_prefix_rules():
    names = ["steer_g13", "steer_g13_R", "wing_steer", "feed_pre_inh", "sugar2_exc", "mbon_avoid", "song_vnc",
             "pip10_L", "dms2", "orn", "photoreceptor", "dn_land", "escape_dn_L", "gfc2"]
    est = RateEstimator(names, np.ones(len(names)), 0.05)
    assert est.tau.tolist() == [0.15, 0.15, 0.15, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.20, 0.20, 0.20, 0.05, 0.05]
    assert RateEstimator.REGION_TAU_S == 0.25 and RateEstimator.DEFAULT_TAU_S == 0.20


def test_monitor_rates_update_through_engine(tiny_connectome):
    eng = LIFEngine(tiny_connectome, seed=0, noise_sigma=0.0, noise_mu=0.0)
    gf = tiny_connectome.groups["gf"]
    eng.inject("gf", rate_hz=100.0, duration_ms=2000.0, tag="g")
    for _ in range(40):
        s = eng.step(50)
    # DNp01 driven at 100 Hz kicks but merged inside latency + refractory -> ~60-100 Hz per neuron
    assert 50.0 <= s.rates["gf"] <= 100.0
    assert s.rates["gf"] == pytest.approx((s.rates["gf_L"] * tiny_connectome.groups["gf_L"].size
                                           + s.rates["gf_R"] * tiny_connectome.groups["gf_R"].size) / gf.size, rel=1e-6)
    assert s.rates["kc"] == 0.0
    assert s.region_rates[REGION_ID["descending_motor"]] > 0 and s.rates["dn_mean"] == pytest.approx(
        float(s.region_rates[REGION_ID["descending_motor"]]))
    assert s.region_rates[REGION_ID["optic_lobe"]] == 0.0
    assert len(REGIONS) == s.region_rates.shape[0]
    eng.reset()
    assert all(v == 0.0 for v in eng.rates().values())
