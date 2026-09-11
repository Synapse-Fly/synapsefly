"""Regression tests for the end-to-end defects found by the integrator (fixer pass).

Each test pins one concrete bug that a passing unit suite did not catch, because both bugs only show up when
the modules are wired together for minutes at a time:

1. ``flybrain/server/app.py`` handed ``Connectome.meta`` to ``TweetAgent`` as ``conn_meta``. ``meta`` (SPEC c.2)
   has no ``name``/``n``/``source`` - those are attributes of the ``Connectome`` - so every LLM summary reported
   ``connectome {"name": "", "n": 0}`` instead of the provenance line of SPEC d.6.
2. ``flybrain/decoder.py`` froze the fly for as long as DNp09 stayed above the f.3 exit rate. A market drawdown
   pins ``looming`` near 0.5 (f.1 ``mom = tanh(chg_m5/2)`` saturates below -4 %), which keeps f.2 row 8 driving
   ``lc_freeze`` forever: the fly stopped walking for the whole drawdown and the Paint canvas stayed empty.
   ``FREEZE_MAX_MS`` / ``FREEZE_REFRACTORY_MS`` cap a bout so the fly alternates freeze and walk.

These tests are deterministic and offline (no server, no market thread).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from flybrain import decoder as D
from flybrain.agent.summary import build_brain_summary
from flybrain.decoder import FlyBody, MotorDecoder, Readouts
from flybrain.server.app import _agent_conn_meta

DT = 0.05


# --------------------------------------------------------------------------- 1. agent conn_meta
def test_agent_conn_meta_carries_name_and_size(tiny_connectome):
    """SPEC d.6: the summary's ``connectome`` object names the graph and its size (never ``""`` / 0)."""
    conn = tiny_connectome
    meta = _agent_conn_meta(conn)
    assert meta["name"] == conn.name and meta["name"] != ""
    assert meta["n"] == conn.n > 0
    assert meta["e"] == conn.e > 0
    assert meta["source"] == conn.source
    # every key of Connectome.meta survives (license / note / citation feed the same summary)
    for k, v in (conn.meta or {}).items():
        if k not in ("name", "n", "source", "e"):
            assert meta[k] == v

    summary = build_brain_summary({}, None, "manual", meta, {}, {}, "en")
    assert summary["connectome"]["name"] == conn.name
    assert summary["connectome"]["n"] == conn.n
    assert summary["connectome"]["e"] == conn.e
    assert summary["connectome"]["source"] == conn.source


def test_agent_conn_meta_survives_a_bare_connectome():
    """Errors never stop the sim (SPEC 0.1): a stub without ``meta`` still yields usable identity fields."""
    meta = _agent_conn_meta(SimpleNamespace(name="x-1", source="synthetic", n=7, e=9))
    assert meta == {"name": "x-1", "n": 7, "source": "synthetic", "e": 9}
    meta2 = _agent_conn_meta(SimpleNamespace())
    assert meta2["name"] == "" and meta2["n"] == 0 and meta2["source"] == "synthetic"


# --------------------------------------------------------------------------- 2. freeze bout cap
def _feat(**kw) -> SimpleNamespace:
    base = dict(looming=0.0, sustained_loom=False, candle="flat", mom=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def _settings() -> SimpleNamespace:
    return SimpleNamespace(wander_sigma=0.0, explore_baseline=0.25, canvas_w=800, canvas_h=500, walls="bounce")


def _drawdown_run(seconds: float) -> list[str]:
    """Run the decoder + body under a *permanent* sell-wall drawdown and return the mode of every tick.

    ``freeze = 55 Hz`` / ``looming = 0.5`` / ``sustained_loom`` is what f.2 row 8 produces while ``chg_m5``
    stays around -10 %: the pre-fix decoder answered ``freeze`` on every single tick.
    """
    d = MotorDecoder(_settings(), seed=0)
    body = FlyBody(800, 500, "bounce", seed=0)
    r = Readouts.zeros(freeze=55.0, fwd=9.5)     # explore baseline keeps fwd alive (f.2 row 17)
    f = _feat(looming=0.5, sustained_loom=True, candle="down", mom=-1.0)
    modes: list[str] = []
    for k in range(int(seconds / DT)):
        t_ms = int((k + 1) * DT * 1000)
        cmd = d.update(r, f, "PANIC", t_ms, DT)
        kin, _ = body.integrate(cmd, DT, t_ms)
        d.observe(kin)
        modes.append(cmd.mode)
    return modes


def test_freeze_bout_is_capped_under_a_sustained_drawdown():
    modes = _drawdown_run(60.0)
    assert set(modes) <= {"freeze", "walk"}
    first_walk = modes.index("walk")
    # the bout ends one tick after the FREEZE_MAX_MS budget is spent (the 800 ms min dwell is the earlier bound)
    assert D.FREEZE_MAX_MS / (DT * 1000.0) <= first_walk <= D.FREEZE_MAX_MS / (DT * 1000.0) + 3
    walk_frac = modes.count("walk") / len(modes)
    assert walk_frac > 0.35, f"fly still latched: walk fraction {walk_frac:.2f}"
    # it alternates rather than releasing once: several bouts in a minute
    bouts = sum(1 for a, b in zip(modes, modes[1:]) if a == "walk" and b == "freeze")
    assert bouts >= 3


def test_freeze_refractory_blocks_an_immediate_refreeze():
    d = MotorDecoder(_settings(), seed=0)
    r = Readouts.zeros(freeze=55.0)
    f = _feat(looming=0.5, sustained_loom=True)
    t = 0
    for _ in range(int((D.FREEZE_MAX_MS + 50) / 50)):
        t += 50
        cmd = d.update(r, f, "PANIC", t, DT)
    assert cmd.mode == "walk" and d.n_freeze_capped == 1
    assert d.freeze_block_until == t + D.FREEZE_REFRACTORY_MS
    # still under threat, but no new bout until the refractory window closes
    while t < d.freeze_block_until - 50:
        t += 50
        assert d.update(r, f, "PANIC", t, DT).mode == "walk"
    t += 50
    assert d.update(r, f, "PANIC", t, DT).mode == "freeze"


def test_freeze_cap_survives_a_gf_jump_storm():
    """A sell wall fires DNp01 every jump refractory; a jump clears ``frozen`` (f.3), so a *per-bout* cap would be
    restarted forever. The live 20k session showed exactly that: freeze 0.81 / jump 0.19 / walk 0.00 of all ticks.
    """
    d = MotorDecoder(_settings(), seed=0)
    body = FlyBody(800, 500, "bounce", seed=0)
    f = _feat(looming=0.95, sustained_loom=True, candle="down", mom=-1.0)
    modes: list[str] = []
    travel = 0.0
    prev = (body.kin.x, body.kin.y)
    for k in range(int(90.0 / DT)):
        t_ms = int((k + 1) * DT * 1000)
        gf = 1 if t_ms % 1500 == 0 else 0        # one GF spike per jump refractory window
        r = Readouts.zeros(freeze=70.0, fwd=9.5, gf_spikes_l=gf)
        cmd = d.update(r, f, "PANIC", t_ms, DT)
        kin, _ = body.integrate(cmd, DT, t_ms)
        d.observe(kin)
        modes.append(cmd.mode)
        if cmd.mode != "jump":
            step = float(np.hypot(kin.x - prev[0], kin.y - prev[1]))
            travel += step if step <= 40.0 else 0.0
        prev = (kin.x, kin.y)
    n = len(modes)
    assert d.n_freeze_capped >= 3, f"cap never fired under a jump storm ({d.n_freeze_capped})"
    assert modes.count("freeze") / n <= 0.62, f"still latched: freeze {modes.count('freeze') / n:.2f}"
    # the escape storm keeps it airborne rather than walking (f.3: jump end + looming >= 0.5 -> fly), which still
    # lays down a trail - with the cap off this run paints half as much (2.1 kpx vs 4.0 kpx of segments).
    assert modes.count("fly") / n >= 0.1 and travel > 3000.0


def test_freeze_cap_does_not_shorten_a_normal_bout():
    """A real loom (threat gone after 2 s) still follows the plain f.3 exit rule, not the cap."""
    d = MotorDecoder(_settings(), seed=0)
    f_on = _feat(looming=0.5, sustained_loom=True)
    f_off = _feat(looming=0.0, sustained_loom=False)
    t = 0
    for _ in range(40):                                  # 2 s of threat
        t += 50
        assert d.update(Readouts.zeros(freeze=55.0), f_on, "PANIC", t, DT).mode == "freeze"
    for _ in range(9):                                   # 450 ms of release (< FREEZE_EXIT_MS)
        t += 50
        assert d.update(Readouts.zeros(freeze=0.0), f_off, "CRUISING", t, DT).mode == "freeze"
    t += 50
    assert d.update(Readouts.zeros(freeze=0.0), f_off, "CRUISING", t, DT).mode == "walk"
    assert d.n_freeze_capped == 0


def test_frozen_fly_paints_again_after_the_cap():
    """The user-visible symptom: a trail is only laid down while the fly moves (SPEC f.5/e.4)."""
    d = MotorDecoder(_settings(), seed=0)
    body = FlyBody(800, 500, "bounce", seed=0)
    r = Readouts.zeros(freeze=55.0, fwd=9.5)
    f = _feat(looming=0.5, sustained_loom=True, candle="down", mom=-1.0)
    travel = 0.0
    prev = (body.kin.x, body.kin.y)
    for k in range(int(40.0 / DT)):
        t_ms = int((k + 1) * DT * 1000)
        cmd = d.update(r, f, "PANIC", t_ms, DT)
        kin, _ = body.integrate(cmd, DT, t_ms)
        d.observe(kin)
        if cmd.mode != "freeze":                      # a frozen fly only tremors (wire pose, f.4)
            step = float(np.hypot(kin.x - prev[0], kin.y - prev[1]))
            if step <= 40.0:                          # the client lifts the pen above 40 px (SPEC f.4)
                travel += step
        prev = (kin.x, kin.y)
    assert travel > 500.0, f"trail would still be empty: {travel:.0f} px in 40 s"
