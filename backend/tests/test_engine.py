"""Tests for flybrain.snn.params / engine (SPEC sections c.9, c.10, h.2; RESEARCH section 7).

Every biological expectation asserted here is tagged: ``[V]`` verified in RESEARCH section 7,
``[L]`` literature (Shiu 2024 defaults), ``[E]`` engineered test scaffolding. Test names of SPEC h.2
are kept verbatim; the remaining tests are additions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from flybrain.connectome.csr import sum_duplicates
from flybrain.connectome.groups import resolve_groups
from flybrain.connectome.schema import SYNTHETIC_BODY_BASE, Connectome, count_groups, count_regions, utc_now_iso
from flybrain.snn.engine import (
    HOMEOSTASIS_FLOOR,
    NOISE_POOL,
    TONIC_TABLE_MV,
    Drive,
    Injection,
    LIFEngine,
    StepStats,
    apply_tonic_table,
)
from flybrain.snn.monitor import SpikeMonitor
from flybrain.snn.params import LIFParams, StepConstants, current_from_rate, rate_from_current

# --------------------------------------------------------------------------- helpers


def small_connectome(n: int, edges: list[tuple[int, int, float]], types: list[str] | None = None,
                     signs: np.ndarray | None = None, seed: int = 0) -> Connectome:
    """Hand-built [E] connectome with explicit edges (pre, post, synapse count); all neurons excitatory
    unless ``signs`` is given; untyped unless ``types`` (one label per neuron) is given."""
    if edges:
        pre = np.asarray([e[0] for e in edges], dtype=np.int64)
        post = np.asarray([e[1] for e in edges], dtype=np.int64)
        w = np.asarray([e[2] for e in edges], dtype=np.float64)
        pre, post, weight = sum_duplicates(pre, post, w, n)
    else:
        pre = np.zeros(0, dtype=np.int32)
        post = np.zeros(0, dtype=np.int32)
        weight = np.zeros(0, dtype=np.float32)
    sign = np.ones(n, dtype=np.float32) if signs is None else np.asarray(signs, dtype=np.float32)
    region = np.zeros(n, dtype=np.uint8)
    side = np.where(np.arange(n) % 2 == 0, -1, 1).astype(np.int8)
    if types is None:
        type_list = [""]
        type_idx = np.zeros(n, dtype=np.int32)
    else:
        type_list = sorted(set(types))
        type_idx = np.asarray([type_list.index(t) for t in types], dtype=np.int32)
    groups = resolve_groups(type_list, type_idx, side)
    meta = {
        "e": int(pre.shape[0]), "synapses": float(np.sum(weight)), "license": "synthetic (no data)", "citation": None,
        "seed": seed, "build_args": {"n": n}, "patches_applied": [], "gain_default": 1.0, "weights_mode": "test",
        "engineered_edges": [], "region_counts": count_regions(region), "group_counts": count_groups(groups),
        "created": utc_now_iso(),
    }
    conn = Connectome(
        name=f"small-{n}", source="synthetic", n=n, pre=pre.astype(np.int32), post=post.astype(np.int32),
        weight=weight.astype(np.float32), sign=sign, region=region, side=side, types=type_list, type_idx=type_idx,
        body_id=SYNTHETIC_BODY_BASE + np.arange(n, dtype=np.int64), nt=np.asarray(["acetylcholine"] * n, dtype=object),
        groups=groups, meta=meta,
    )
    conn.validate()
    return conn


def random_connectome(n: int, e: int, seed: int) -> Connectome:
    rng = np.random.default_rng(seed)
    edges = [(int(a), int(b), float(w)) for a, b, w in
             zip(rng.integers(0, n, e), rng.integers(0, n, e), np.round(rng.lognormal(np.log(4), 0.9, e)) + 1)]
    signs = np.where(rng.random(n) < 0.3, -1.0, 1.0)
    return small_connectome(n, edges, signs=signs, seed=seed)


def quiet_engine(conn: Connectome, dt: float = 1.0, gain: float = 1.0, seed: int = 0, **kw) -> LIFEngine:
    """Engine without background noise (isolated-neuron experiments)."""
    return LIFEngine(conn, seed=seed, dt_ms=dt, gain=gain, noise_mu=0.0, noise_sigma=0.0, **kw)


def spike_steps_of(eng: LIFEngine, stats: StepStats, neuron: int) -> np.ndarray:
    slot = int(eng.monitor.slot_of_neuron[neuron])
    assert slot >= 0, "neuron must be in the raster sample"
    return stats.spike_dt_sample[stats.spike_indices_sample == slot]


# --------------------------------------------------------------------------- params


@pytest.mark.parametrize(
    "dt,a_m,a_s,b,ref,delay",
    [
        (1.0, 0.951229, 0.818731, 0.044166, 2, 2),
        (0.5, 0.975310, 0.904837, 0.023491, 4, 4),
        (0.2, 0.990050, 0.960789, 0.009753, 11, 9),
        (0.1, 0.995012, 0.980199, 0.004938, 22, 18),
    ],
)
def test_step_constants_table(dt, a_m, a_s, b, ref, delay):
    """``LIFParams().constants(dt)`` reproduces the c.9 / RESEARCH section 7 table to 6 dp [V]."""
    c = LIFParams().constants(dt)
    assert isinstance(c, StepConstants)
    assert c.dt_ms == dt
    assert c.a_m == pytest.approx(a_m, abs=5e-7)
    assert c.a_s == pytest.approx(a_s, abs=5e-7)
    assert c.b == pytest.approx(b, abs=5e-7)
    assert c.ref_steps == ref
    assert c.delay_steps == delay
    assert c.kick_mv == pytest.approx(68.75)
    # closed forms
    assert c.a_m == pytest.approx(math.exp(-dt / 20.0))
    assert c.a_s == pytest.approx(math.exp(-dt / 5.0))
    assert c.b == pytest.approx(5.0 / 15.0 * (c.a_m - c.a_s))


def test_lif_defaults_are_shiu_2024():
    """[L] Shiu et al. 2024 defaults."""
    p = LIFParams()
    assert (p.v_rest, p.v_reset, p.v_th) == (-52.0, -52.0, -45.0)
    assert (p.tau_m, p.tau_s, p.t_ref, p.delay) == (20.0, 5.0, 2.2, 1.8)
    assert p.w_syn == 0.275 and p.f_poi == 250.0
    with pytest.raises(ValueError):
        p.constants(0.0)
    with pytest.raises(ValueError):
        LIFParams(tau_s=20.0).constants(1.0)


@pytest.mark.parametrize("rate,i_mv", [(10, 7.05), (20, 7.71), (50, 11.88), (100, 21.68), (150, 34.97)])
def test_current_from_rate_table(rate, i_mv):
    """RESEARCH section 7 constant-current inversion [V]."""
    assert current_from_rate(rate) == pytest.approx(i_mv, abs=0.006)
    assert rate_from_current(current_from_rate(rate)) == pytest.approx(rate, rel=1e-9)


def test_current_rate_edge_cases():
    assert current_from_rate(0.0) == 0.0
    assert current_from_rate(-5.0) == 0.0
    assert rate_from_current(7.0) == 0.0          # exactly the threshold gap: never fires
    assert rate_from_current(0.0) == 0.0
    assert rate_from_current(7.0001) > 0.0
    assert current_from_rate(10_000.0) > 1e6     # ISI clamped just above t_ref -> huge current, finite
    assert math.isfinite(current_from_rate(10_000.0))


# --------------------------------------------------------------------------- latency contract


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.2, 0.1])
def test_forced_kick_latency(dt):
    """SPEC c.10 latency contract [V]: one 68.75 mV kick (Poisson forcing with p = 1 for a single step,
    via ``inject(indices, rate_hz=...)``) into an isolated neuron at step k -> first spike at
    3.0 <= t - t_k <= 3.0 + dt."""
    conn = small_connectome(1, [])
    eng = quiet_engine(conn, dt=dt)
    k = 5
    eng.step(k)                                   # the kick lands at step k, not at step 0
    inj = eng.inject(np.array([0], dtype=np.int32), rate_hz=1000.0 / dt, duration_ms=dt, tag="kick")
    assert isinstance(inj, Injection) and inj.tag == "kick"
    stats = eng.step(int(round(12.0 / dt)))
    assert stats.forced_events == 1, "p = rate*dt/1000 = 1 -> exactly one kick"
    assert stats.total_spikes == 1, "one kick produces exactly one spike"
    steps = spike_steps_of(eng, stats, 0)
    assert steps.shape[0] == 1
    t_spike = float(steps[0]) * dt               # relative to the kick step (window starts at step k)
    assert 3.0 - 1e-9 <= t_spike <= 3.0 + dt + 1e-9
    assert eng.injections == []                  # expired


@pytest.mark.parametrize("dt", [1.0, 0.5, 0.1])
def test_forced_kick_latency_direct_kick(dt):
    """Same contract through ``kick()`` (queued forced kick, applied in substep 4 of the next step)."""
    conn = small_connectome(1, [])
    eng = quiet_engine(conn, dt=dt)
    assert eng.kick(np.array([0])) == 1
    stats = eng.step(int(round(10.0 / dt)))
    steps = spike_steps_of(eng, stats, 0)
    assert steps.shape[0] == 1 and stats.forced_events == 1 and stats.total_spikes == 1
    assert 3.0 - 1e-9 <= float(steps[0]) * dt <= 3.0 + dt + 1e-9


def test_kick_default_and_custom_mv():
    conn = small_connectome(2, [])
    eng = quiet_engine(conn)
    eng.kick(np.array([0]))            # 68.75 mV -> spike
    eng.kick(np.array([1]), mv=10.0)   # too small -> no spike
    eng.step(1)
    assert eng.g[0] == pytest.approx(68.75)
    assert eng.g[1] == pytest.approx(10.0)
    stats = eng.step(20)
    assert stats.total_spikes == 1
    assert spike_steps_of(eng, stats, 1).shape[0] == 0


def test_unitary_epsp():
    """0.275 mV kick (one unitary synapse) -> peak v - v_rest 0.0433 +- 0.002 mV at 9-10 ms [V]."""
    conn = small_connectome(1, [])
    eng = quiet_engine(conn, dt=1.0)
    eng.kick(np.array([0]), mv=0.275)
    eng.step(1)
    trace = []
    for _ in range(30):
        eng.step(1)
        trace.append(float(eng.v[0]) + 52.0)
    peak_step = int(np.argmax(trace)) + 1        # ms after the kick
    assert 9 <= peak_step <= 10
    assert max(trace) == pytest.approx(0.0433, abs=0.002)
    assert trace[-1] < max(trace) * 0.5          # decays back towards rest


# --------------------------------------------------------------------------- update / refractory


def test_exact_update_matches_closed_form():
    """v <- v_eq + (v - v_eq)*a_m + b*g_old ; g <- g_old*a_s [V]."""
    conn = small_connectome(1, [])
    eng = quiet_engine(conn)
    c = eng.constants
    eng.kick(np.array([0]), mv=20.0)
    eng.step(1)
    v, g = -52.0, 20.0
    for _ in range(2):
        eng.step(1)
        v = -52.0 + (v + 52.0) * c.a_m + c.b * g
        g = g * c.a_s
        assert eng.v[0] == pytest.approx(v, abs=1e-4)
        assert eng.g[0] == pytest.approx(g, abs=1e-4)


def test_refractory_freezes_v_and_g():
    """Brian2 'unless refractory' [L]: during ref_steps steps after a spike v == v_reset and g does not
    decay; incoming input still accumulates and is visible on the first non-refractory step."""
    conn = small_connectome(1, [])
    eng = quiet_engine(conn)
    c = eng.constants
    assert c.ref_steps == 2
    eng.kick(np.array([0]))
    eng.step(3)  # kick at step 0, fires during step 3 (t = 3.0 ms)
    eng.step(1)
    assert eng.ref[0] == c.ref_steps
    assert eng.v[0] == pytest.approx(-52.0) and eng.g[0] == pytest.approx(0.0)   # reset v=v_reset, g=0
    eng.kick(np.array([0]), mv=30.0)
    eng.step(1)  # refractory step 1: frozen, input accumulates
    assert eng.ref[0] == c.ref_steps - 1
    assert eng.v[0] == pytest.approx(-52.0)
    assert eng.g[0] == pytest.approx(30.0)
    eng.step(1)  # refractory step 2: still frozen, no decay of g
    assert eng.ref[0] == 0
    assert eng.v[0] == pytest.approx(-52.0)
    assert eng.g[0] == pytest.approx(30.0)
    eng.step(1)  # active again: v integrates the frozen g, g decays
    assert eng.v[0] == pytest.approx(-52.0 + c.b * 30.0, abs=1e-4)
    assert eng.g[0] == pytest.approx(30.0 * c.a_s, abs=1e-4)


def test_refractory_neuron_cannot_fire_even_with_huge_input():
    conn = small_connectome(1, [])
    eng = quiet_engine(conn)
    eng.kick(np.array([0]))
    eng.step(4)   # fired at step 3, now refractory
    eng.kick(np.array([0]), mv=500.0)
    s = eng.step(2)  # both refractory steps
    assert s.total_spikes == 0
    s = eng.step(1)  # first active step: b*500 > 7 mV -> fires
    assert s.total_spikes == 1


@pytest.mark.parametrize("dt", [1.0, 0.1])
def test_refractory_length_matches_ref_steps(dt):
    """A neuron held above threshold by a large constant current fires with ISI == ref_steps + 1 steps
    (one integration step after the refractory period) at every dt."""
    conn = small_connectome(1, [])
    eng = quiet_engine(conn, dt=dt)
    eng.set_tonic(np.array([0], dtype=np.int32), 5000.0)
    s = eng.step(int(round(40.0 / dt)))
    steps = spike_steps_of(eng, s, 0)
    assert steps.shape[0] >= 3
    isi = np.diff(steps.astype(np.int64))
    assert np.all(isi == eng.constants.ref_steps + 1)


# --------------------------------------------------------------------------- delay ring


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_delay_ring(dt):
    """A spike at step k arrives in the target's g at step k + delay_steps exactly."""
    w = 12.0
    conn = small_connectome(2, [(0, 1, w)])
    eng = quiet_engine(conn, dt=dt)
    D = eng.constants.delay_steps
    assert D == int(round(1.8 / dt))
    eng.kick(np.array([0]))
    k_fire = None
    for step in range(60):
        s = eng.step(1)
        if k_fire is None and s.total_spikes:
            k_fire = step
            assert spike_steps_of(eng, s, 0).shape[0] == 1
            continue
        if k_fire is not None:
            arrived = eng.g[1] > 0
            if step < k_fire + D:
                assert not arrived, f"input arrived early at step {step} (fire {k_fire}, D {D})"
            elif step == k_fire + D:
                assert eng.g[1] == pytest.approx(w * 0.275, abs=1e-5)
                break
    else:
        pytest.fail("no arrival observed")


def test_sign_convention():
    """GABA presynaptic neuron lowers the post ``g``, ACh raises it; gain scales both [L]."""
    conn = small_connectome(3, [(0, 2, 10.0), (1, 2, 10.0)], signs=np.array([-1.0, 1.0, 1.0]))
    for gain in (1.0, 0.5):
        eng = quiet_engine(conn, gain=gain)
        D = eng.constants.delay_steps
        eng.kick(np.array([0]))
        eng.step(3 + D + 1)
        assert eng.g[2] == pytest.approx(-10.0 * 0.275 * gain, abs=1e-5)
        eng2 = quiet_engine(conn, gain=gain)
        eng2.kick(np.array([1]))
        eng2.step(3 + D + 1)
        assert eng2.g[2] == pytest.approx(+10.0 * 0.275 * gain, abs=1e-5)
    csr = conn.csr()
    assert csr.data[0] == -10.0 and csr.data[1] == 10.0
    assert eng.data_mv[0] == pytest.approx(-10.0 * 0.275 * 0.5) and eng.data_mv[1] == pytest.approx(10.0 * 0.275 * 0.5)


def test_ring_slots_do_not_leak_between_windows():
    """Inputs written into the ring must be consumed exactly once (zeroed after being read)."""
    conn = small_connectome(2, [(0, 1, 10.0)])
    eng = quiet_engine(conn)
    eng.kick(np.array([0]))
    eng.step(3 + eng.constants.delay_steps + 1)
    g1 = eng.g[1]
    assert g1 > 0
    assert np.all(eng.ring == 0.0)  # everything consumed
    eng.step(1)
    assert eng.g[1] == pytest.approx(g1 * eng.constants.a_s, abs=1e-5)


# --------------------------------------------------------------------------- noise / rest activity


@pytest.mark.parametrize("dt", [1.0, 0.5])
def test_noise_rest_rate(dt):
    """400 isolated neurons, mu 0.5 mV/ms, sigma 3.5 mV/sqrt(ms), 2 s -> mean rate in [1.5, 3.5] Hz
    (RESEARCH section 7: ~2.3 Hz at dt 1.0 and 0.5) [V]; sigma 0 -> 0 spikes."""
    n = 400
    conn = small_connectome(n, [])
    eng = LIFEngine(conn, seed=9, dt_ms=dt, noise_mu=0.5, noise_sigma=3.5)
    total = 0
    steps = int(round(2000.0 / dt))
    per = int(round(50.0 / dt))
    for _ in range(steps // per):
        s = eng.step(per)
        total += s.total_spikes
        assert s.noise_on
    rate = total / n / 2.0
    assert 1.5 <= rate <= 3.5, rate
    silent = LIFEngine(conn, seed=9, dt_ms=dt, noise_mu=0.0, noise_sigma=0.0)
    s = silent.step(per * 4)
    assert s.total_spikes == 0 and not s.noise_on


def test_noise_pool_samples_are_gaussian_and_fresh():
    """Per-step noise increments are N(mu*dt, sigma*sqrt(dt)) across neurons (pooled float32 normals),
    differ from step to step, and the pool is continuously refreshed."""
    n = 20_000
    conn = small_connectome(n, [])
    eng = LIFEngine(conn, LIFParams(v_th=1e9), seed=4, dt_ms=0.5, noise_mu=0.5, noise_sigma=3.5)
    assert eng._pool.shape[0] >= max(NOISE_POOL, 4 * n)
    assert abs(float(eng._pool.mean())) < 0.01 and abs(float(eng._pool.std()) - 1.0) < 0.01
    g0 = eng.g.copy()
    eng.step(1)
    d1 = eng.g - g0
    assert d1.mean() == pytest.approx(0.5 * 0.5, abs=0.05)
    assert d1.std() == pytest.approx(3.5 * math.sqrt(0.5), abs=0.05)
    g1 = eng.g.copy()
    eng.step(1)
    d2 = eng.g - g1
    assert not np.array_equal(d1, d2)
    assert eng._pool_pos == 2 * eng._pool_refresh


def test_noise_sigma_zero_disables_mean_term():
    """Regression: ``sigma <= 0`` switches the WHOLE noise term off, mean included (SPEC c.10 step 4
    draws only ``if sigma > 0``), so a quiet engine rests exactly at ``v_rest`` and ``noise_on`` is
    False - the mu term used to keep depolarising ``g`` by ``mu*dt`` every step."""
    conn = small_connectome(80, [])
    eng = LIFEngine(conn, seed=0, dt_ms=1.0, noise_mu=0.5, noise_sigma=0.0)   # h.1 `engine` fixture style
    s = eng.step(500)
    assert not s.noise_on and s.total_spikes == 0
    assert float(eng.g.max()) == 0.0
    np.testing.assert_array_equal(eng.v, np.full(conn.n, LIFParams().v_rest, dtype=np.float32))
    # turning sigma on makes both terms live again (mean of the increment = mu*dt)
    eng.set_noise(0.5, 3.5)
    g0 = eng.g.copy()
    eng.step(1)
    assert float((eng.g - g0).mean()) == pytest.approx(0.5, abs=0.4)
    # and the h.1 fixture itself rests at v_rest
    eng2 = LIFEngine(conn, seed=0, dt_ms=1.0, noise_sigma=0.0)
    eng2.step(200)
    assert float(eng2.g.max()) == 0.0 and float(eng2.v.mean()) == pytest.approx(LIFParams().v_rest, abs=1e-6)


def test_set_noise_and_negative_sigma():
    conn = small_connectome(50, [])
    eng = LIFEngine(conn, seed=1, noise_sigma=0.0, noise_mu=0.0)
    s = eng.step(200)
    assert s.total_spikes == 0 and not s.noise_on
    eng.set_noise(0.5, 3.5)
    s = eng.step(500)
    assert s.total_spikes > 0 and s.noise_on
    eng.set_noise(0.0, -1.0)   # negative sigma clamps to 0
    assert eng.noise_sigma == 0.0


# --------------------------------------------------------------------------- determinism


def test_determinism(tiny_connectome):
    """Two engines, same seed, 500 steps with drives -> identical spike counts per step and bit-identical v."""
    def make(seed: int) -> LIFEngine:
        eng = LIFEngine(tiny_connectome, seed=seed)
        eng.inject("grn_sugar_labellar", rate_hz=100.0, duration_ms=500.0, tag="s")
        eng.inject("lc_loom", rate_hz=60.0, duration_ms=500.0, recruit=0.6, episode=2, tag="l")
        return eng

    a, b = make(1337), make(1337)
    for _ in range(500):
        sa, sb = a.step(1), b.step(1)
        assert sa.total_spikes == sb.total_spikes
        assert np.array_equal(sa.spike_indices_sample, sb.spike_indices_sample)
    assert np.array_equal(a.v, b.v) and np.array_equal(a.g, b.g) and np.array_equal(a.ring, b.ring)


def test_reseed_changes_noise(tiny_connectome):
    a = LIFEngine(tiny_connectome, seed=1337)
    b = LIFEngine(tiny_connectome, seed=1338)
    ta = [a.step(50).total_spikes for _ in range(4)]
    tb = [b.step(50).total_spikes for _ in range(4)]
    assert ta != tb or not np.array_equal(a.v, b.v)
    # reseed + reset reproduces the original run exactly; reseeding differently does not
    c = LIFEngine(tiny_connectome, seed=5)
    s1 = c.step(100)
    c.reset()
    assert c.t_ms == 0 and np.all(c.v == -52.0) and np.all(c.ref == 0) and np.all(c.ring == 0)
    s2 = c.step(100)
    assert s1.total_spikes == s2.total_spikes
    assert np.array_equal(s1.spike_indices_sample, s2.spike_indices_sample)
    c.reseed(6)
    c.reset(reseed=False)
    s3 = c.step(100)
    assert not (s3.total_spikes == s1.total_spikes and np.array_equal(s3.spike_indices_sample, s1.spike_indices_sample))


# --------------------------------------------------------------------------- injections


def test_inject_tag_replaces(tiny_connectome):
    """Re-injecting the same tag replaces, ``clear_injections(tag)`` removes, expiry at ``until_ms``."""
    eng = LIFEngine(tiny_connectome, seed=0)
    inj = eng.inject("gf", rate_hz=50.0, duration_ms=10.0, tag="a")
    assert inj.until_ms == 10 and inj.drive.group == "gf" and inj.drive.mode == "poisson"
    assert isinstance(inj.drive, Drive)
    eng.inject("gf", rate_hz=80.0, duration_ms=10.0, tag="a")     # replaces
    assert len(eng.injections) == 1 and eng.injections[0].drive.rate_hz == 80.0
    eng.inject("gf", rate_hz=80.0, duration_ms=10.0)              # empty tag -> unique
    eng.inject("gf", rate_hz=80.0, duration_ms=10.0)
    assert len(eng.injections) == 3
    eng.step(9)
    assert len(eng.injections) == 3
    eng.step(1)                                                    # t_ms == 10 -> expired at substep 1 of the next step
    assert len(eng.injections) == 3
    eng.step(1)
    assert eng.injections == []
    eng.inject("gf", rate_hz=1.0, duration_ms=1000.0, tag="x")
    eng.inject("gf", current_mv=5.0, duration_ms=1000.0, tag="y")
    assert eng.i_ext[tiny_connectome.groups["gf"]].max() == pytest.approx(5.0)
    eng.clear_injections("y")
    assert np.all(eng.i_ext == 0.0)
    assert [i.tag for i in eng.injections] == ["x"]
    eng.clear_injections()
    assert eng.injections == []
    with pytest.raises(ValueError):
        eng.inject("gf")
    with pytest.raises(ValueError):
        eng.inject("gf", rate_hz=1.0, current_mv=1.0)
    with pytest.raises(ValueError):
        eng.inject("no_such_group", rate_hz=1.0)
    with pytest.raises(ValueError):
        eng.inject(np.array([-1, 5]), rate_hz=1.0)


def test_poisson_forcing_statistics():
    """p = rate*dt/1000 per recruited neuron per step; every kick produces one spike (isolated neurons) [V]."""
    n = 200
    conn = small_connectome(n, [])
    eng = quiet_engine(conn, seed=2)
    eng.inject(np.arange(n, dtype=np.int32), rate_hz=100.0, duration_ms=1000.0, tag="p")
    forced = spikes = 0
    for _ in range(20):
        s = eng.step(50)
        forced += s.forced_events
        spikes += s.total_spikes
    expect = n * 0.1 * 1000
    assert abs(forced - expect) < 0.05 * expect
    # kicks landing inside the 3 ms latency + 2.2 ms refractory window merge, so spikes < kicks but same order
    assert 0.5 * forced < spikes <= forced


def test_current_from_rate_inversion():
    """``set_tonic`` with ``current_from_rate(r)`` for r in {20, 50, 100} -> measured rate within 5 % over 2 s
    at dt 1.0 (RESEARCH section 7: 20.0 / 50.0 / 100.0 Hz simulated) [V]."""
    rates = (20.0, 50.0, 100.0)
    n = 3 * 5
    conn = small_connectome(n, [])
    eng = quiet_engine(conn)
    for j, r in enumerate(rates):
        idx = np.arange(5 * j, 5 * (j + 1), dtype=np.int32)
        eng.set_tonic(idx, current_from_rate(r))
        assert rate_from_current(current_from_rate(r)) == pytest.approx(r, rel=1e-9)
    counts = np.zeros(n, dtype=int)
    for _ in range(40):
        s = eng.step(50)
        for i in range(n):
            counts[i] += spike_steps_of(eng, s, i).shape[0]
    measured = counts / 2.0
    for j, r in enumerate(rates):
        np.testing.assert_allclose(measured[5 * j:5 * (j + 1)], r, rtol=0.05)


def test_current_mode_equivalent():
    """``drive_mode='current'`` with rate 100 Hz on a group -> ``i_ext = current_from_rate(100)`` and 100 +- 10 Hz."""
    types = ["DNp01"] * 10 + [""] * 10
    conn = small_connectome(20, [], types=types)
    eng = quiet_engine(conn, drive_mode="current")
    inj = eng.inject("gf", rate_hz=100.0, duration_ms=3000.0, tag="c")
    assert inj.drive.mode == "current" and inj.drive.rate_hz == 100.0
    gf = conn.groups["gf"]
    np.testing.assert_allclose(eng.i_ext[gf], current_from_rate(100.0), rtol=1e-5)
    assert np.all(eng.i_ext[10:] == 0.0)
    counts = np.zeros(20, dtype=int)
    for _ in range(40):
        s = eng.step(50)
        for i in range(20):
            counts[i] += spike_steps_of(eng, s, i).shape[0]
    assert np.all(np.abs(counts[gf] / 2.0 - 100.0) <= 10.0)
    assert np.all(counts[10:] == 0)
    # inject_drive honours Drive.mode = 'current' in a poisson engine too
    eng2 = quiet_engine(conn)
    inj2 = eng2.inject_drive(Drive(group="gf", rate_hz=100.0, mode="current"), duration_ms=50.0, tag="d")
    assert inj2.drive.mode == "current"
    np.testing.assert_allclose(eng2.i_ext[gf], current_from_rate(100.0), rtol=1e-5)


def test_recruit_and_side(tiny_connectome):
    """``recruit=0.5, side=-1`` drives exactly ceil(0.5 * n_L) left cells; same episode -> same subset,
    different episode -> different subset; weights follow the recruited cells."""
    eng = LIFEngine(tiny_connectome, seed=3)
    g = tiny_connectome.groups
    full = g["lc_loom"]
    n_l = g["lc_loom_L"].shape[0]
    eng.inject("lc_loom", rate_hz=10.0, duration_ms=100.0, recruit=0.5, side=-1, episode=0, tag="a")
    idx_a = eng._inj["a"].idx
    assert idx_a.shape[0] == math.ceil(0.5 * n_l)
    assert np.all(np.isin(idx_a, g["lc_loom_L"])) and np.all(np.diff(idx_a) > 0)
    assert np.all(tiny_connectome.side[idx_a] == -1)
    eng.inject("lc_loom", rate_hz=10.0, duration_ms=100.0, recruit=0.5, side=-1, episode=0, tag="b")
    assert np.array_equal(eng._inj["b"].idx, idx_a)                 # same episode -> same recruitment
    eng.inject("lc_loom", rate_hz=10.0, duration_ms=100.0, recruit=0.5, side=-1, episode=1, tag="c")
    assert not np.array_equal(eng._inj["c"].idx, idx_a)             # different episode -> different set
    eng.inject("lc_loom", rate_hz=10.0, duration_ms=100.0, recruit=0.7, side=-1, tag="c7")
    assert eng._inj["c7"].idx.shape[0] == math.ceil(0.7 * n_l - 1e-9)
    eng.inject("lc_loom", rate_hz=10.0, duration_ms=100.0, side=-1, tag="d")
    assert np.array_equal(eng._inj["d"].idx, g["lc_loom_L"])
    w = np.linspace(0.0, 2.0, full.shape[0])
    eng.inject("lc_loom", rate_hz=100.0, duration_ms=100.0, side=1, weights=w, tag="e")
    rec = eng._inj["e"]
    assert rec.idx.shape[0] == g["lc_loom_R"].shape[0]
    np.testing.assert_allclose(rec.p, np.clip(0.1 * w[tiny_connectome.side[full] == 1], 0, 1).astype(np.float32))
    with pytest.raises(ValueError):
        eng.inject("lc_loom", rate_hz=1.0, weights=np.ones(3))
    with pytest.raises(ValueError):
        eng.inject("lc_loom", rate_hz=1.0, recruit=1.5)
    # Drive objects go through inject_drive
    inj = eng.inject_drive(Drive(group="gf", rate_hz=20.0, side=1), duration_ms=30.0, tag="f")
    assert inj.tag == "f" and np.array_equal(eng._inj["f"].idx, g["gf_R"])


# --------------------------------------------------------------------------- homeostasis


def test_homeostasis_gain_rule(tiny_connectome):
    """active fraction > 5 % for 10 consecutive steps -> gain *= 0.9 (floor 0.3) [E], SPEC section 0."""
    eng = LIFEngine(tiny_connectome, seed=0, noise_sigma=0.0, noise_mu=0.0)
    s = eng.step(50)
    assert eng.apply_homeostasis(s) is None and eng.gain == 1.0
    # p = 0.3 per neuron per step: desynchronised forcing keeps > 5 % of the neurons spiking on EVERY step
    # (p = 1 would synchronise the population into bursts every 3 steps and never form a 10-step hot run)
    eng.inject(np.arange(tiny_connectome.n, dtype=np.int32), rate_hz=300.0, duration_ms=1000.0, tag="all")
    s = eng.step(20)
    assert s.active_frac_max > 0.05 and s.hot_steps_max >= 10
    ev = eng.apply_homeostasis(s)
    assert ev is not None
    assert ev["gain_before"] == pytest.approx(1.0) and ev["gain_after"] == pytest.approx(0.9)
    assert ev["active_frac"] == pytest.approx(s.active_frac_max)
    assert eng.gain == pytest.approx(0.9) and eng.hot_steps == 0
    np.testing.assert_allclose(eng.data_mv, tiny_connectome.csr().data * np.float32(0.275 * 0.9), rtol=1e-6)
    for _ in range(30):
        s = eng.step(12)
        eng.apply_homeostasis(s)
    assert eng.gain == pytest.approx(HOMEOSTASIS_FLOOR)
    # a short hot burst (< 10 steps) does not trigger
    eng2 = LIFEngine(tiny_connectome, seed=0, noise_sigma=0.0, noise_mu=0.0)
    eng2.inject(np.arange(tiny_connectome.n, dtype=np.int32), rate_hz=1000.0, duration_ms=1.0, tag="one")
    s = eng2.step(20)
    assert s.hot_steps_max < 10 and eng2.apply_homeostasis(s) is None and eng2.gain == 1.0


# --------------------------------------------------------------------------- state / misc


def test_state_dict_roundtrip():
    conn = random_connectome(60, 700, seed=4)
    rng = np.random.default_rng(0)
    kicks = [np.sort(rng.choice(60, size=5, replace=False)) for _ in range(40)]
    e1 = quiet_engine(conn)
    for k in kicks[:20]:
        e1.kick(k)
        e1.step(1)
    snap = e1.state_dict()
    assert set(snap) >= {"v", "g", "ref", "i_ext", "tonic", "ring", "t_step", "gain"}
    assert snap["t_step"] == 20 and snap["v"] is not e1.v
    assert np.any(snap["ref"] > 0), "some neurons must be refractory at the snapshot"
    e2 = quiet_engine(conn)
    e2.load_state_dict(snap)
    assert e2.t_ms == 20
    for k in kicks[20:]:
        e1.kick(k)
        e2.kick(k)
        s1 = e1.step(1)
        s2 = e2.step(1)
        assert s1.total_spikes == s2.total_spikes
        assert np.array_equal(s1.spike_indices_sample, s2.spike_indices_sample)
    assert np.array_equal(e1.v, e2.v) and np.array_equal(e1.ring, e2.ring) and np.array_equal(e1.ref, e2.ref)
    with pytest.raises(ValueError):
        e2.load_state_dict({"v": np.zeros(3, np.float32)})


def test_state_dict_restores_random_stream():
    """Regression: ``state_dict`` carries the RNG bit-generator state and the Gaussian noise pool, so a
    restored engine reproduces the original run step for step (replay from a snapshot, not only t=0)."""
    conn = random_connectome(200, 2000, seed=3)
    a = LIFEngine(conn, seed=11, dt_ms=1.0)
    a.step(37)
    snap = a.state_dict()
    assert {"rng_state", "noise_pool", "noise_pool_pos"} <= set(snap)
    b = LIFEngine(conn, seed=11, dt_ms=1.0)
    b.load_state_dict(snap)
    assert b.t_step == a.t_step
    ra = [a.step(10).total_spikes for _ in range(5)]
    rb = [b.step(10).total_spikes for _ in range(5)]
    assert ra == rb and sum(ra) > 0
    assert np.array_equal(a.v, b.v) and np.array_equal(a.g, b.g) and np.array_equal(a.ring, b.ring)
    # a document without the random keys still loads (arrays only) and is accepted
    c = LIFEngine(conn, seed=11, dt_ms=1.0)
    c.load_state_dict({k: v for k, v in snap.items() if k not in ("rng_state", "noise_pool", "noise_pool_pos")})
    assert c.t_step == int(snap["t_step"])
    with pytest.raises(ValueError):
        c.load_state_dict({"noise_pool": np.zeros(3, np.float32)})


def test_set_gain_rescales(tiny_connectome):
    eng = LIFEngine(tiny_connectome, seed=0, gain=0.65)
    csr = tiny_connectome.csr()
    np.testing.assert_allclose(eng.data_mv, csr.data * np.float32(0.275 * 0.65), rtol=1e-6)
    eng.set_gain(1.0)
    np.testing.assert_allclose(eng.data_mv, csr.data * np.float32(0.275), rtol=1e-6)
    assert eng.prop.data_mv is eng.data_mv
    assert eng.data_mv.dtype == np.float32
    # the rescaled weights are what the ring receives
    conn = small_connectome(2, [(0, 1, 10.0)])
    for gain in (1.0, 0.5, 2.0):
        eng = quiet_engine(conn, gain=gain)
        eng.kick(np.array([0]))
        eng.step(3 + eng.constants.delay_steps + 1)
        assert eng.g[1] == pytest.approx(10.0 * 0.275 * gain, abs=1e-5)


def test_tonic_table_application(tiny_connectome):
    eng = LIFEngine(tiny_connectome, seed=0)
    applied = apply_tonic_table(eng)
    assert set(applied) == set(TONIC_TABLE_MV) == {"lamina", "motion_in", "feed_pre_inh", "mal"}
    for group, mv in TONIC_TABLE_MV.items():
        idx = tiny_connectome.groups[group]
        assert applied[group] == idx.shape[0] > 0
        np.testing.assert_allclose(eng.tonic[idx], mv)
    assert np.all(eng.tonic[tiny_connectome.groups["gf"]] == 0.0)
    assert TONIC_TABLE_MV["feed_pre_inh"] == pytest.approx(current_from_rate(10.0), abs=0.01)


def test_step_stats_shape(tiny_connectome):
    """All c.10 fields present, region_counts length 8, raster arrays parallel and sorted, 131 rate keys."""
    eng = LIFEngine(tiny_connectome, seed=0)
    eng.inject("grn_sugar_labellar", rate_hz=100.0, duration_ms=100.0, tag="s")
    s = eng.step(50)
    assert isinstance(s, StepStats)
    assert (s.n_steps, s.t0_ms, s.t1_ms) == (50, 0, 50)
    assert eng.t_ms == 50
    assert s.region_counts.dtype == np.int64 and s.region_counts.shape == (8,)
    assert int(s.region_counts.sum()) == s.total_spikes > 0
    assert 0.0 < s.active_frac_max <= 1.0
    assert s.spike_indices_sample.dtype == np.int32 and s.spike_dt_sample.dtype == np.int16
    assert s.spike_indices_sample.shape == s.spike_dt_sample.shape
    assert np.all(np.diff(s.spike_dt_sample) >= 0) and (s.spike_dt_sample.max() < 50 if s.spike_dt_sample.size else True)
    assert s.capped is False
    assert list(s.rates) == eng.monitor.rates.keys() and len(s.rates) == 131
    assert s.region_rates.dtype == np.float32 and s.region_rates.shape == (8,)
    assert s.forced_events > 0 and s.edge_visits > 0 and s.step_ms_mean > 0
    assert set(s.gf_spikes) == {"L": 0, "R": 0}.keys()
    assert s.noise_on is True and s.gain == 1.0
    assert s.spike_counts_by_group["grn_sugar"] > 0
    assert list(s.spike_counts_by_group) == eng.monitor.names
    assert s.spike_counts_by_group["gf"] == s.spike_counts_by_group["gf_L"] + s.spike_counts_by_group["gf_R"]
    assert s.spike_counts_by_group["gf"] == s.gf_spikes["L"] + s.gf_spikes["R"]
    assert eng.rates() == s.rates
    assert eng.backend == "numpy" and eng.n == tiny_connectome.n
    assert eng.step(0).n_steps == 0


def test_custom_monitor_and_bad_drive_mode(tiny_connectome):
    mon = SpikeMonitor(tiny_connectome, per_region=8, cap=10, seed=3)
    eng = LIFEngine(tiny_connectome, seed=0, monitor=mon)
    assert eng.monitor is mon and mon.dt_ms == 1.0
    assert len(mon.rows) == 64
    with pytest.raises(ValueError):
        LIFEngine(tiny_connectome, drive_mode="voltage")
    with pytest.raises(ValueError):
        LIFEngine(tiny_connectome, backend="cuda")


def test_t_ms_with_fractional_dt():
    conn = small_connectome(1, [])
    eng = quiet_engine(conn, dt=0.5)
    eng.step(3)
    assert eng.t_ms == 1 and eng.t_ms_f == pytest.approx(1.5)
    inj = eng.inject(np.array([0]), rate_hz=1.0, duration_ms=10.0)
    assert inj.until_ms == 12   # ceil(1.5 + 10)
