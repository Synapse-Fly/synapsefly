"""Tests for ``flybrain.decoder`` (SPEC c.18, f.3-f.5, h.2 ``test_decoder.py``).

``Features`` and ``Settings`` are duck-typed (``SimpleNamespace``) so this file depends on nothing but
numpy and the decoder itself. Wander noise is switched off (``wander_sigma = 0``) wherever an exact
``omega`` is asserted.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain import decoder as D
from flybrain.decoder import (
    FlyBody,
    InkStyle,
    Kinematics,
    MotorCommand,
    MotorDecoder,
    Readouts,
    Wander,
    wrap_angle,
)

DT = 0.05
PI = math.pi

#: the ``tick.fly`` keys of SPEC d.2
FLY_WIRE_KEYS = {"x", "y", "vx", "vy", "heading", "speed", "omega", "wing_hz", "wing_amp", "wing_ext", "mode",
                 "leg_phase", "proboscis", "jump_t_ms"}


# --------------------------------------------------------------------------- helpers
def feat(**kw) -> SimpleNamespace:
    base = dict(looming=0.0, sustained_loom=False, candle="flat", mom=0.0)
    base.update(kw)
    return SimpleNamespace(**base)


def settings(sigma: float = 0.0, explore: float = 0.25) -> SimpleNamespace:
    return SimpleNamespace(wander_sigma=sigma, explore_baseline=explore, canvas_w=800, canvas_h=500, walls="bounce")


def dec(seed: int = 0, sigma: float = 0.0, explore: float = 0.25) -> MotorDecoder:
    return MotorDecoder(settings(sigma, explore), seed)


def R(**kw) -> Readouts:
    return Readouts.zeros(**kw)


def walk_cmd(v: float, omega: float = 0.0) -> MotorCommand:
    return MotorCommand(mode="walk", v_target=v, omega=omega, wing_hz=0.0, wing_amp=0.0, wing_ext=0,
                        proboscis=0.0, jump=None, halt_reason=None, events=[])


def still_cmd(mode: str) -> MotorCommand:
    return MotorCommand(mode=mode, v_target=0.0, omega=0.0, wing_hz=0.0, wing_amp=0.0, wing_ext=0,
                        proboscis=0.0, jump=None, halt_reason=mode, events=[])


def run(d: MotorDecoder, r: Readouts, f, n: int, t0: int = 0, mood: str = "CRUISING", dt: float = DT):
    """n ticks; returns (last t_ms, last cmd, all events)."""
    events: list[dict] = []
    t = t0
    cmd = None
    for _ in range(n):
        t += int(round(dt * 1000))
        cmd = d.update(r, f, mood, t, dt)
        events.extend(cmd.events)
    return t, cmd, events


def kinds(events: list[dict]) -> list[str]:
    return [e["kind"] for e in events]


def kinds_no_gf(events: list[dict]) -> list[str]:
    """Event kinds without the per-tick ``gf_spike`` bookkeeping event (SPEC d.8)."""
    return [e["kind"] for e in events if e["kind"] != "gf_spike"]


# --------------------------------------------------------------------------- Readouts
def test_readouts_from_stats():
    rates = {
        "dng100": 9.5, "dn_fwd": 12.0, "dn_halt": 1.5, "dn_back": 0.4, "dn_freeze": 2.1,
        "steer_a02_L": 4.0, "steer_a02_R": 12.0, "steer_a01_L": 2.5, "steer_a01_R": 3.5,
        "steer_g13_L": 1.8, "steer_g13_R": 2.6, "flight_dn": 0.3, "wing_power": 0.7,
        "b1": 5.0, "b1_L": 4.0, "b1_R": 6.0, "i1": 1.0, "i1_L": 1.0, "i1_R": 1.0,
        "hg1": 7.0, "hg1_L": 6.0, "hg1_R": 8.0, "dn_saccade_L": 0.2, "dn_saccade_R": 0.4, "dn_land": 0.1,
        "feed_mn": 18.0, "groom_dn": 0.5, "p1": 0.1, "pip10": 0.0, "dms2_L": 0.3, "dms2_R": 0.6,
        "lc_loom": 0.46, "dn_mean": 4.12, "song_mn": 99.0,   # song_mn is recomputed per SPEC f.3
    }
    s = SimpleNamespace(rates=rates, gf_spikes={"L": 1, "R": 2},
                        spike_counts_by_group={"escape_dn": 3, "dnp11": 1, "gf": 3})
    r = Readouts.from_stats(s)
    assert r.fwd == 12.0 and r.halt == 1.5 and r.back == 0.4 and r.freeze == 2.1
    assert (r.a02_l, r.a02_r, r.a01_l, r.a01_r, r.g13_l, r.g13_r) == (4.0, 12.0, 2.5, 3.5, 1.8, 2.6)
    assert r.flight == 0.3 and r.wing_power == 0.7
    assert (r.b1_l, r.b1_r, r.i1_l, r.i1_r, r.hg1_l, r.hg1_r) == (4.0, 6.0, 1.0, 1.0, 6.0, 8.0)
    assert (r.saccade_l, r.saccade_r, r.land) == (0.2, 0.4, 0.1)
    assert r.mn9 == 18.0 and r.groom == 0.5 and r.p1 == 0.1 and r.pip10 == 0.0
    assert r.song_mn == pytest.approx(0.5 * (7.0 + 5.0))
    assert (r.dms2_l, r.dms2_r) == (0.3, 0.6)
    assert (r.gf_spikes_l, r.gf_spikes_r) == (1, 2) and r.dnp11_spikes == 3
    assert r.lc_loom == 0.46 and r.dn_mean == 4.12
    # fwd = max(dng100, dn_fwd) the other way round; missing keys -> 0; NaN -> 0; dnp11 fallback key
    s2 = SimpleNamespace(rates={"dng100": 20.0, "dn_fwd": 3.0, "feed_mn": float("nan")},
                         gf_spikes={}, spike_counts_by_group={"dnp11": 2})
    r2 = Readouts.from_stats(s2)
    assert r2.fwd == 20.0 and r2.mn9 == 0.0 and r2.a02_r == 0.0 and r2.gf_spikes_l == 0 and r2.dnp11_spikes == 2
    # frozen
    with pytest.raises(Exception):
        r2.fwd = 1.0  # type: ignore[misc]


# --------------------------------------------------------------------------- f.3 derived signals
def test_turn_sign():
    d = dec()
    f = feat()
    cmd = d.update(R(fwd=30.0, a02_r=40.0, a02_l=0.0), f, "CRUISING", 50, DT)
    assert cmd.omega == pytest.approx(4.0) and cmd.omega > 0          # clockwise
    cmd = d.update(R(fwd=30.0, a02_l=40.0, a02_r=0.0), f, "CRUISING", 100, DT)
    assert cmd.omega == pytest.approx(-4.0)
    cmd = d.update(R(fwd=30.0, a02_r=4000.0), f, "CRUISING", 150, DT)  # clipped to +-1
    assert cmd.omega == pytest.approx(4.0)
    cmd = d.update(R(fwd=30.0, a02_l=4000.0, g13_r=6000.0), f, "CRUISING", 200, DT)
    assert cmd.omega == pytest.approx(-4.0)
    cmd = d.update(R(fwd=30.0, g13_r=60.0), f, "CRUISING", 250, DT)   # 0.5 * 60/60
    assert cmd.omega == pytest.approx(2.0)
    cmd = d.update(R(fwd=30.0, a01_r=60.0), f, "CRUISING", 300, DT)   # 0.3 * 60/60
    assert cmd.omega == pytest.approx(1.2)
    cmd = d.update(R(fwd=30.0, a02_r=20.0, g13_l=30.0, a01_r=30.0), f, "CRUISING", 350, DT)
    assert cmd.omega == pytest.approx(4.0 * (0.5 - 0.25 + 0.15))


def test_walk_speed_from_fwd():
    d = dec()
    f = feat()
    cmd = d.update(R(fwd=30.0), f, "CRUISING", 50, DT)
    assert cmd.mode == "walk" and cmd.v_target == pytest.approx(120.0) and cmd.halt_reason is None
    cmd = d.update(R(fwd=15.0), f, "CRUISING", 100, DT)
    assert cmd.v_target == pytest.approx(60.0)
    cmd = d.update(R(fwd=300.0), f, "CRUISING", 150, DT)              # sat
    assert cmd.v_target == pytest.approx(120.0)
    cmd = d.update(R(fwd=30.0, halt=30.0), f, "CRUISING", 200, DT)    # halt 1 -> 0, halt_dn
    assert cmd.v_target == pytest.approx(0.0) and cmd.halt_reason == "halt_dn"
    cmd = d.update(R(fwd=30.0, halt=15.0), f, "CRUISING", 250, DT)    # halt 0.5 -> 60, no halt_dn (< 0.8)
    assert cmd.v_target == pytest.approx(60.0) and cmd.halt_reason is None
    cmd = d.update(R(fwd=30.0, back=16.0), f, "CRUISING", 300, DT)    # back 0.8 -> -32
    assert cmd.v_target == pytest.approx(-32.0)
    cmd = d.update(R(fwd=30.0, back=8.0), f, "CRUISING", 350, DT)     # back 0.4 <= 0.5 -> forward
    assert cmd.v_target == pytest.approx(120.0)
    assert cmd.wing_hz == 0.0 and cmd.wing_amp == 0.0 and cmd.wing_ext == 0 and cmd.jump is None


# --------------------------------------------------------------------------- feeding / freeze / groom
def test_feed_hysteresis():
    d = dec()
    f = feat()
    t, cmd, ev = run(d, R(mn9=30.0), f, 3)                             # 150 ms
    assert cmd.mode == "walk" and not d.feeding and ev == []
    t, cmd, ev = run(d, R(mn9=30.0), f, 1, t0=t)                       # 200 ms
    assert cmd.mode == "feed" and d.feeding and cmd.halt_reason == "feed" and cmd.v_target == 0.0
    assert kinds(ev) == ["feed_start"] and ev[0]["data"]["mn9_hz"] == 30.0 and ev[0]["t_ms"] == t
    assert cmd.proboscis == pytest.approx((30.0 - 15.0) / 45.0)
    t_start = t
    t, cmd, ev = run(d, R(mn9=60.0), f, 2, t0=t)
    assert cmd.proboscis == pytest.approx(1.0)
    t, cmd, ev = run(d, R(mn9=20.0), f, 40, t0=t)                     # 15 <= mn9 < 30: keeps feeding
    assert cmd.mode == "feed" and ev == []
    t, cmd, ev = run(d, R(mn9=10.0), f, 19, t0=t)                     # 950 ms below 15
    assert cmd.mode == "feed" and ev == []
    t, cmd, ev = run(d, R(mn9=10.0), f, 1, t0=t)                      # 1000 ms
    assert cmd.mode == "walk" and not d.feeding and cmd.proboscis == 0.0
    assert kinds(ev) == ["feed_stop"] and ev[0]["data"]["meal_ms"] == t - t_start
    # a 100 ms blip of mn9 >= 30 does not start a meal
    t, cmd, ev = run(d, R(mn9=30.0), f, 2, t0=t)
    t, cmd, ev = run(d, R(mn9=0.0), f, 2, t0=t)
    assert not d.feeding and ev == []


def test_freeze():
    d = dec()
    t, cmd, ev = run(d, R(freeze=45.0), feat(looming=0.3), 1)
    assert cmd.mode == "freeze" and cmd.halt_reason == "freeze" and cmd.v_target == 0.0 and cmd.omega == 0.0
    assert kinds(ev) == ["freeze"] and ev[0]["data"] == {"dnp09_hz": 45.0, "looming": 0.3}
    t_in = t
    # release condition met at once, but the 800 ms minimum holds
    t, cmd, ev = run(d, R(freeze=0.0), feat(looming=0.0), 15, t0=t)  # 750 ms
    assert cmd.mode == "freeze" and ev == []
    t, cmd, ev = run(d, R(freeze=0.0), feat(looming=0.0), 1, t0=t)   # 800 ms
    assert cmd.mode == "walk" and kinds(ev) == ["unfreeze"] and t - t_in == 800
    # no freeze without looming (or sustained_loom)
    d2 = dec()
    t, cmd, ev = run(d2, R(freeze=45.0), feat(looming=0.1), 5)
    assert cmd.mode == "walk" and ev == []
    t, cmd, ev = run(d2, R(freeze=45.0), feat(looming=0.0, sustained_loom=True), 1, t0=t)
    assert cmd.mode == "freeze"
    # exit needs the release condition held 500 ms once the minimum is over
    t, cmd, ev = run(d2, R(freeze=45.0), feat(looming=0.5), 30, t0=t)  # 1.5 s frozen, threat persists
    assert cmd.mode == "freeze"
    t, cmd, ev = run(d2, R(freeze=10.0), feat(looming=0.5), 9, t0=t)   # 450 ms of DNp09 < 20
    assert cmd.mode == "freeze"
    t, cmd, ev = run(d2, R(freeze=10.0), feat(looming=0.5), 1, t0=t)
    assert cmd.mode == "walk" and kinds(ev) == ["unfreeze"]


def test_groom_min_duration():
    d = dec()
    t, cmd, ev = run(d, R(groom=45.0), feat(), 1)
    assert cmd.mode == "groom" and cmd.halt_reason == "groom" and kinds(ev) == ["groom"]
    assert ev[0]["data"]["side"] in ("L", "R", "both")
    t_in = t
    t, cmd, ev = run(d, R(groom=0.0), feat(), 59, t0=t)               # 2950 ms
    assert cmd.mode == "groom"
    t, cmd, ev = run(d, R(groom=0.0), feat(), 1, t0=t)                # 3000 ms
    assert cmd.mode == "walk" and t - t_in == 3000
    # after the minimum, grooming continues while groom_dn stays >= 15 Hz
    d2 = dec()
    t, cmd, ev = run(d2, R(groom=45.0), feat(), 61)
    t, cmd, ev = run(d2, R(groom=20.0), feat(), 20, t0=t)
    assert cmd.mode == "groom"
    t, cmd, ev = run(d2, R(groom=14.0), feat(), 1, t0=t)
    assert cmd.mode == "walk"


# --------------------------------------------------------------------------- jump / fly / saccade
def test_jump_on_gf_spike():
    d = dec(seed=3)
    f = feat()
    heading = 0.0
    cmd = d.update(R(fwd=30.0, gf_spikes_r=1), f, "CRUISING", 50, DT)
    assert cmd.mode == "jump" and cmd.jump is not None and cmd.jump["side"] == "R" and cmd.jump["forward"] is False
    assert cmd.v_target == 600.0 and cmd.wing_hz == 250.0 and cmd.wing_amp == 1.0 and cmd.omega == 0.0
    assert cmd.jump["impulse"] == 600.0
    assert set(cmd.jump) == {"heading", "impulse", "side", "forward"}   # exactly the c.18 shape
    centre = heading + PI + PI / 6
    diff = wrap_angle(cmd.jump["heading"] - centre)
    assert abs(diff) <= PI / 4 + 1e-9
    assert wrap_angle(d.jump_d_heading - (PI + PI / 6)) == pytest.approx(diff)
    assert kinds(cmd.events) == ["gf_spike", "jump"]                   # d.8: gf_spike {side, count} precedes the jump
    assert cmd.events[0]["data"] == {"side": "R", "count": 1}
    assert set(cmd.events[1]["data"]) == {"heading_out", "impulse", "side", "forward"}
    assert cmd.events[1]["data"]["heading_out"] == pytest.approx(cmd.jump["heading"])
    # a second GF spike within 1.5 s is ignored (no new jump event, same jump); jump mode covers [t0, t0 + 400)
    jump_dict = cmd.jump
    t, cmd, ev = run(d, R(fwd=30.0, gf_spikes_r=1), f, 7, t0=50)     # 100..400 ms
    assert cmd.mode == "jump" and kinds_no_gf(ev) == [] and cmd.jump is jump_dict
    assert kinds(ev) == ["gf_spike"] * 7                               # ... but every DNp01 spike is still reported
    t, cmd, ev = run(d, R(fwd=30.0, gf_spikes_r=1), f, 1, t0=t)       # 450 ms -> jump over, no looming -> walk
    assert cmd.mode == "walk" and cmd.jump is None and kinds_no_gf(ev) == [] and t == 450
    t, cmd, ev = run(d, R(fwd=30.0, gf_spikes_r=1), f, 21, t0=t)      # 500..1500 ms: within the refractory window
    assert cmd.mode == "walk" and kinds_no_gf(ev) == [] and t == 1500
    t, cmd, ev = run(d, R(fwd=30.0, gf_spikes_l=1), f, 1, t0=t)       # 1550 ms: refractory over
    assert cmd.mode == "jump" and cmd.jump["side"] == "L" and kinds(ev) == ["gf_spike", "jump"]
    d.observe(Kinematics(x=0, y=0, vx=0, vy=0, heading=1.0, speed=0, omega=0, wing_hz=0, wing_amp=0, wing_ext=0,
                         mode="walk", leg_phase=0, proboscis=0, jump_t_ms=None))
    # forward takeoff when DNp02/04/11 fired, both sides -> "both"
    d2 = dec(seed=5)
    cmd = d2.update(R(gf_spikes_l=1, gf_spikes_r=1, dnp11_spikes=2), f, "CRUISING", 50, DT)
    assert cmd.jump["side"] == "both" and cmd.jump["forward"] is True
    assert abs(wrap_angle(cmd.jump["heading"] - 0.0)) <= PI / 6 + 1e-9
    # observe() gives the decoder the true heading for the absolute jump heading
    d3 = dec(seed=5)
    d3.observe(Kinematics(x=0, y=0, vx=0, vy=0, heading=2.0, speed=0, omega=0, wing_hz=0, wing_amp=0,
                          wing_ext=0, mode="walk", leg_phase=0, proboscis=0, jump_t_ms=None))
    cmd = d3.update(R(gf_spikes_l=1, gf_spikes_r=1, dnp11_spikes=2), f, "CRUISING", 50, DT)
    assert abs(wrap_angle(cmd.jump["heading"] - 2.0)) <= PI / 6 + 1e-9
    # a jump at the end of which the threat still looms turns into flight (event takeoff)
    d4 = dec()
    t, cmd, ev = run(d4, R(gf_spikes_r=1), feat(looming=0.8), 8)
    assert cmd.mode == "jump"
    t, cmd, ev = run(d4, R(), feat(looming=0.8), 1, t0=t)
    assert cmd.mode == "fly" and kinds(ev) == ["takeoff"] and "wing_hz" in ev[0]["data"]
    # a jump interrupts a meal (feed_stop) and a freeze (unfreeze)
    d5 = dec()
    t, cmd, ev = run(d5, R(mn9=50.0), f, 5)
    assert d5.feeding
    t, cmd, ev = run(d5, R(mn9=50.0, gf_spikes_l=1), f, 1, t0=t)
    assert cmd.mode == "jump" and kinds(ev) == ["gf_spike", "jump", "feed_stop"] and not d5.feeding


def test_jump_forward_only_from_this_tick_dnp11():
    """Regression (SPEC f.3 ``forward = dnp11_spikes > 0``): a DNp02/04/11 spike on an *earlier* tick used to
    bias the takeoff forward for another 100 ms, so the wire reported ``forward: true`` with
    ``dnp11_spikes == 0``. ``dnp11_recent_ms`` is still recorded (c.18 held timer) but never read."""
    f = feat()
    d = dec(seed=3)
    d.update(R(dnp11_spikes=1), f, "CRUISING", 50, DT)                 # DNp11 fires, no GF spike
    assert d.dnp11_recent_ms == 50                                     # ... timer recorded
    cmd = d.update(R(gf_spikes_r=1), f, "CRUISING", 100, DT)            # GF one tick later: no forward bias
    assert cmd.jump["forward"] is False
    assert abs(wrap_angle(cmd.jump["heading"] - (PI + PI / 6))) <= PI / 4 + 1e-9
    # the same tick does bias it forward
    d2 = dec(seed=3)
    cmd2 = d2.update(R(gf_spikes_r=1, dnp11_spikes=1), f, "CRUISING", 50, DT)
    assert cmd2.jump["forward"] is True and abs(wrap_angle(cmd2.jump["heading"])) <= PI / 6 + 1e-9


def test_jump_dict_and_events_not_mutated_by_the_body():
    """Regression (SPEC c.18): ``FlyBody.integrate`` used to rewrite ``cmd.jump["heading"]`` and the ``jump``
    event's ``heading_out`` in place, so anything that snapshotted the events before step 6 of c.27 saw a
    different heading than the wire. The decoder now publishes the absolute heading itself."""
    d = dec(seed=3)
    b = FlyBody(800, 500, "wrap", 0)
    f = feat()
    # turn the fly first so the body heading is not 0 (wander off, pure steering)
    t = 0
    for _ in range(10):
        t += 50
        cmd = d.update(R(fwd=30.0, a02_r=40.0), f, "CRUISING", t, DT)
        kin, _ = b.integrate(cmd, DT, t)
        d.observe(kin)
    assert abs(b.heading) > 0.5
    t += 50
    cmd = d.update(R(fwd=30.0, gf_spikes_r=1), f, "CRUISING", t, DT)
    before = (dict(cmd.jump), [dict(e["data"]) for e in cmd.events])
    kin, _ = b.integrate(cmd, DT, t)
    after = (dict(cmd.jump), [dict(e["data"]) for e in cmd.events])
    assert before == after                                             # nothing in the command was rewritten
    ev_jump = [e for e in cmd.events if e["kind"] == "jump"][0]
    assert ev_jump["data"]["heading_out"] == pytest.approx(cmd.jump["heading"])
    assert b.heading == pytest.approx(cmd.jump["heading"])             # the body adopted it
    assert kin.heading == pytest.approx(cmd.jump["heading"])


def test_fly_enter_exit():
    d = dec()
    f = feat()
    t, cmd, ev = run(d, R(flight=50.0), f, 3)                          # wing_power 1 for 150 ms
    assert cmd.mode == "walk" and ev == []
    t, cmd, ev = run(d, R(flight=50.0), f, 1, t0=t)                    # 200 ms -> takeoff
    assert cmd.mode == "fly" and kinds(ev) == ["takeoff"] and ev[0]["data"]["wing_hz"] == pytest.approx(250.0)
    assert cmd.v_target == pytest.approx(180.0 + 220.0) and cmd.wing_hz == pytest.approx(250.0)
    assert cmd.wing_amp == pytest.approx(0.6) and cmd.halt_reason is None and cmd.wing_ext == 0
    cmd = d.update(R(flight=25.0, wing_power=15.0), f, "CRUISING", t + 50, DT)
    t += 50
    wp = min(1.0, 0.5 + 0.3 * 0.5)                                     # 0.65
    assert cmd.v_target == pytest.approx(180.0 + 220.0 * wp) and cmd.wing_hz == pytest.approx(150.0 + 100.0 * wp)
    assert cmd.wing_amp == pytest.approx(0.6 + 0.4 * 1.0)
    # wing steering asymmetry in flight: 0.02 * ((b1_r + i1_r) - (b1_l + i1_l))
    cmd = d.update(R(flight=50.0, b1_r=10.0, i1_r=10.0), f, "CRUISING", t + 50, DT)
    t += 50
    assert cmd.omega == pytest.approx(0.02 * 20.0)
    # landing by dn_land
    cmd = d.update(R(flight=50.0, land=20.0), f, "CRUISING", t + 50, DT)
    t += 50
    assert cmd.mode == "walk" and kinds(cmd.events) == ["landing"] and cmd.events[0]["data"]["reason"] == "dn_land"
    # re-takeoff, then landing by low wing power held 2 s
    t, cmd, ev = run(d, R(flight=50.0), f, 4, t0=t)
    assert cmd.mode == "fly"
    t, cmd, ev = run(d, R(flight=5.0), f, 39, t0=t)                    # wing_power 0.1 for 1.95 s
    assert cmd.mode == "fly"
    t, cmd, ev = run(d, R(flight=5.0), f, 1, t0=t)
    assert cmd.mode == "walk" and kinds(ev) == ["landing"] and ev[0]["data"]["reason"] == "wing_power"


def test_saccade():
    d = dec()
    f = feat()
    t, cmd, ev = run(d, R(flight=50.0), f, 4)
    assert cmd.mode == "fly"
    cmd = d.update(R(flight=50.0, saccade_r=40.0), f, "CRUISING", t + 50, DT)
    t += 50
    assert kinds(cmd.events) == ["saccade"] and cmd.events[0]["data"]["side"] == "R"
    ramp = (PI / 2) / 0.1
    assert cmd.omega == pytest.approx(ramp)
    cmd2 = d.update(R(flight=50.0, saccade_r=40.0), f, "CRUISING", t + 50, DT)   # still ramping, no new event
    t += 50
    assert cmd2.omega == pytest.approx(ramp) and cmd2.events == []
    cmd3 = d.update(R(flight=50.0, saccade_r=40.0), f, "CRUISING", t + 50, DT)   # 100 ms over; refractory 500 ms
    t += 50
    assert cmd3.omega == pytest.approx(0.0) and cmd3.events == []
    # integrate the two ramp ticks through the body: heading advances by pi/2
    body = FlyBody(800, 500, "wrap", 0)
    body.integrate(cmd, DT, 1000)
    body.integrate(cmd2, DT, 1050)
    assert body.heading == pytest.approx(PI / 2, abs=1e-9)
    # a left saccade after the refractory window
    t, cmd, ev = run(d, R(flight=50.0), f, 7, t0=t)
    cmd = d.update(R(flight=50.0, saccade_l=40.0), f, "CRUISING", t + 50, DT)
    assert kinds(cmd.events) == ["saccade"] and cmd.events[0]["data"]["side"] == "L" and cmd.omega == pytest.approx(-ramp)
    # never on the ground
    d2 = dec()
    cmd = d2.update(R(fwd=30.0, saccade_r=40.0), f, "CRUISING", 50, DT)
    assert cmd.events == [] and cmd.omega == pytest.approx(0.0)


@pytest.mark.parametrize("speed", [1.0, 0.85, 0.7225, 0.5, 0.26, 0.25, 3.0])
def test_saccade_total_angle_is_pi_over_2_for_any_dt(speed):
    """Regression (SPEC f.3 ``heading += +-pi/2 ... over 100 ms``): a deadline plus a constant ramp rate
    integrated to more than pi/2 whenever the brain ms per tick did not divide 100 ms (114.8 deg at
    ``sim.speed`` 0.85, 135 deg at 0.25 ... ). The remaining angle is now tracked, so the total turn is
    exactly pi/2 at every speed the c.27 pacing can produce."""
    brain_ms = 50.0 * speed
    dt = brain_ms / 1000.0
    n_ramp = int(math.ceil(100.0 / brain_ms)) + 2                      # ticks that cover the 100 ms ramp
    n_gap = int(math.ceil(600.0 / brain_ms)) + 1                       # ticks that clear the 500 ms refractory
    ramp = (PI / 2) / 0.1
    d = dec()
    b = FlyBody(800, 500, "wrap", 0)
    f = feat()
    t = 0.0

    def tick(r: Readouts) -> MotorCommand:
        nonlocal t
        t += brain_ms
        cmd = d.update(r, f, "CRUISING", int(round(t)), dt)
        b.integrate(cmd, dt, int(round(t)))
        return cmd

    for _ in range(60):                                                # takeoff (wing_power 1, >= 200 ms)
        if tick(R(flight=50.0)).mode == "fly":
            break
    assert d.mode == "fly"
    for sign, sacc in ((+1, dict(saccade_r=40.0)), (-1, dict(saccade_l=40.0))):
        h0 = b.heading
        for i in range(n_ramp):
            cmd = tick(R(flight=50.0, **sacc) if i == 0 else R(flight=50.0))
            assert abs(cmd.omega) <= ramp + 1e-9                       # never faster than the f.3 ramp
            assert cmd.omega * sign >= -1e-9                           # ... and never the wrong way
        assert wrap_angle(b.heading - h0) == pytest.approx(sign * PI / 2, abs=1e-9)
        assert d.saccade_left_rad == 0.0
        for _ in range(n_gap):                                         # clear the 500 ms refractory
            tick(R(flight=50.0))


# --------------------------------------------------------------------------- court / song
def test_court_and_song():
    d = dec()
    f = feat()
    cmd = d.update(R(p1=15.0), f, "CRUISING", 50, DT)
    assert cmd.mode == "court" and cmd.v_target == 60.0 and cmd.omega == pytest.approx(3.0) and cmd.omega > 0
    assert cmd.wing_hz == 0.0 and cmd.wing_ext == 0 and cmd.halt_reason is None
    # mood COURTSHIP alone is enough
    d2 = dec()
    cmd = d2.update(R(), f, "COURTSHIP", 50, DT)
    assert cmd.mode == "court"
    # song: pip10 >= 20, dms2_r > dms2_l -> right wing extended, event song R (once per bout)
    cmd = d.update(R(p1=15.0, pip10=20.0, dms2_r=5.0, dms2_l=1.0), f, "CRUISING", 100, DT)
    assert cmd.wing_ext == 1 and kinds(cmd.events) == ["song"] and cmd.events[0]["data"]["side"] == "R"
    cmd = d.update(R(p1=15.0, pip10=20.0, dms2_r=5.0, dms2_l=1.0), f, "CRUISING", 150, DT)
    assert cmd.wing_ext == 1 and cmd.events == []
    cmd = d.update(R(p1=15.0, song_mn=20.0, dms2_r=0.0, dms2_l=2.0), f, "CRUISING", 200, DT)
    assert cmd.wing_ext == -1 and kinds(cmd.events) == ["song"] and cmd.events[0]["data"]["side"] == "L"
    cmd = d.update(R(p1=15.0), f, "CRUISING", 250, DT)
    assert cmd.wing_ext == 0
    # SPEC f.3: court "while mood == COURTSHIP or p1 >= 15 Hz" - no hangover, walking resumes the next tick
    cmd = d.update(R(fwd=30.0), f, "CRUISING", 300, DT)
    assert cmd.mode == "walk" and cmd.v_target == pytest.approx(120.0)
    assert d.court_until == 250
    # song without courtship still extends a wing while walking (wing_ext)
    d3 = dec()
    cmd = d3.update(R(fwd=30.0, pip10=25.0, dms2_r=1.0), f, "CRUISING", 50, DT)
    assert cmd.mode == "walk" and cmd.wing_ext == 1


def test_court_no_hangover():
    """Regression: court mode used to persist 300 ms after P1 dropped (SPEC f.3 has no hangover)."""
    d = dec()
    f = feat()
    d.update(R(p1=20.0), f, "CRUISING", 50, DT)
    modes = [d.update(R(fwd=30.0, p1=0.0), f, "CRUISING", t, DT).mode for t in (100, 150, 200, 250, 300, 350)]
    assert modes == ["walk"] * 6
    # mood COURTSHIP alone holds the mode, and it drops with the mood
    assert d.update(R(fwd=30.0), f, "COURTSHIP", 400, DT).mode == "court"
    assert d.update(R(fwd=30.0), f, "CRUISING", 450, DT).mode == "walk"


def test_song_rule_in_every_mode():
    """Regression: SPEC f.3's song rule carries no mode condition - pIP10 >= 20 Hz extends a wing in flight too."""
    d = dec()
    f = feat()
    t, cmd, ev = run(d, R(flight=50.0, pip10=25.0, dms2_r=2.0), f, 5)
    assert cmd.mode == "fly" and cmd.wing_ext == 1
    assert "song" in kinds(ev) and [e["data"]["side"] for e in ev if e["kind"] == "song"] == ["R"]
    # the jump too (400 ms window), and the side follows dMS2
    d2 = dec()
    cmd = d2.update(R(gf_spikes_l=1, song_mn=25.0, dms2_l=3.0, dms2_r=0.0), f, "CRUISING", 50, DT)
    assert cmd.mode == "jump" and cmd.wing_ext == -1 and "song" in kinds(cmd.events)


def test_mode_priority():
    f = feat()
    d = dec()
    # arm every sub-behaviour at once
    d.feeding = True
    d.feed_start_ms = 0
    d.frozen = True
    d.freeze_since_ms = 0
    d.grooming = True
    d.groom_until = 10 ** 9
    d.flying = True
    r_all = R(mn9=40.0, freeze=45.0, groom=45.0, flight=50.0, p1=20.0)
    assert d.update(r_all, feat(looming=0.5), "SLEEP", 50, DT).mode == "sleep"
    cmd = d.update(R(mn9=40.0, freeze=45.0, groom=45.0, flight=50.0, p1=20.0, gf_spikes_l=1), feat(looming=0.5),
                   "SLEEP", 100, DT)
    assert cmd.mode == "jump"                                          # jump beats sleep
    d.jump_start_ms = None
    d.jump = None
    d.last_jump_ms = -10 ** 9
    d.feeding = True
    d.feed_start_ms = 0
    d.frozen = True
    d.freeze_since_ms = 0
    d.grooming = True
    d.groom_until = 10 ** 9
    d.flying = True
    assert d.update(r_all, feat(looming=0.5), "CRUISING", 150, DT).mode == "court"
    r_nocourt = R(mn9=40.0, freeze=45.0, groom=45.0, flight=50.0)
    d.court_until = -1
    assert d.update(r_nocourt, feat(looming=0.5), "CRUISING", 200, DT).mode == "feed"
    d.feeding = False
    assert d.update(R(freeze=45.0, groom=45.0, flight=50.0), feat(looming=0.5), "CRUISING", 250, DT).mode == "freeze"
    d.frozen = False
    assert d.update(R(groom=45.0, flight=50.0), f, "CRUISING", 300, DT).mode == "groom"
    d.grooming = False
    assert d.update(R(flight=50.0), f, "CRUISING", 350, DT).mode == "fly"
    d.flying = False
    assert d.update(R(fwd=30.0), f, "CRUISING", 400, DT).mode == "walk"
    # sleep / wake events follow the mood
    d2 = dec()
    cmd = d2.update(R(fwd=30.0), f, "SLEEP", 50, DT)
    assert cmd.mode == "sleep" and cmd.halt_reason == "sleep" and cmd.v_target == 0.0 and kinds(cmd.events) == ["sleep"]
    cmd = d2.update(R(fwd=30.0), f, "SLEEP", 100, DT)
    assert cmd.events == []
    cmd = d2.update(R(fwd=30.0), f, "CRUISING", 150, DT)
    assert cmd.mode == "walk" and kinds(cmd.events) == ["wake"] and "reason" in cmd.events[0]["data"]
    # mood may be passed as an Enum or a MoodState-like object
    from flybrain.mood import Mood, MoodMachine
    mm = MoodMachine(0.05)
    assert d2.update(R(fwd=30.0), f, Mood.SLEEP, 200, DT).mode == "sleep"
    mm.state.state = Mood.COURTSHIP
    assert d2.update(R(fwd=30.0), f, mm.state, 250, DT).mode == "court"
    assert d2.update(R(fwd=30.0), f, "CRUISING", 600, DT).mode == "walk"


# --------------------------------------------------------------------------- wander floor / OU
def test_wander_floor():
    d = dec(explore=0.25)
    f = feat()
    t, cmd, ev = run(d, R(fwd=0.0), f, 60)                             # 3.0 s stalled: not yet (> 3000 required)
    assert cmd.v_target == 0.0 and ev == []
    t, cmd, ev = run(d, R(fwd=0.0), f, 2, t0=t)                        # 3.1 s
    assert cmd.mode == "walk" and cmd.v_target == pytest.approx(40.0)
    assert kinds(ev) == ["wander_floor"] and ev[0]["data"]["stalled_ms"] > 3000
    t, cmd, ev = run(d, R(fwd=3.0), f, 100, t0=t)                      # fwd <= 5: floor stays, once per engagement
    assert cmd.v_target == pytest.approx(40.0) and ev == [] and d.n_wander_floor == 1
    t, cmd, ev = run(d, R(fwd=30.0), f, 1, t0=t)                       # fwd > 5 releases it
    assert cmd.v_target == pytest.approx(120.0) and not d.wander_floor_on
    # a second stall engages it again (new event)
    t, cmd, ev = run(d, R(fwd=0.0), f, 70, t0=t)
    assert kinds(ev) == ["wander_floor"] and d.n_wander_floor == 2
    # disabled when explore_baseline == 0
    d0 = dec(explore=0.0)
    t, cmd, ev = run(d0, R(fwd=0.0), f, 200)
    assert cmd.v_target == 0.0 and ev == [] and d0.n_wander_floor == 0
    # never engages while the body is observed moving; never outside walk mode
    d1 = dec()
    moving = Kinematics(x=0, y=0, vx=50, vy=0, heading=0, speed=50, omega=0, wing_hz=0, wing_amp=0, wing_ext=0,
                        mode="walk", leg_phase=0, proboscis=0, jump_t_ms=None)
    t = 0
    for _ in range(100):
        t += 50
        d1.observe(moving)
        cmd = d1.update(R(fwd=0.0), f, "CRUISING", t, DT)
        assert cmd.events == []
    d2 = dec()
    t, cmd, ev = run(d2, R(mn9=50.0), f, 100)
    assert cmd.mode == "feed" and "wander_floor" not in kinds(ev)


def test_wander_floor_reports_halt_dn():
    """Regression (SPEC f.3 walk row: ``halt_reason = halt_dn if halt >= 0.8``): the wander floor used to
    suppress ``halt_reason`` while it was engaged, so a fully halted fly reported ``halt_reason None`` next
    to ``v_target 40``. The floor overrides ``v_target`` only; ``halt_reason`` is diagnostic (not on the d.2
    wire) and keeps telling the truth. The floor also engages on the *stall* alone (f.3 gives only ``fwd > 5
    Hz`` as the release), and a halt DN stall therefore shows both at once."""
    d = dec(explore=0.25)
    b = FlyBody(800, 500, "bounce", 0)
    f = feat()
    t = 0
    engaged: list[tuple[int, float, str | None]] = []
    for _ in range(400):                                               # 20 s of a DN-commanded halt, fwd 30 Hz
        t += 50
        cmd = d.update(R(fwd=30.0, halt=30.0), f, "CRUISING", t, DT)
        kin, _ = b.integrate(cmd, DT, t)
        d.observe(kin)
        assert cmd.halt_reason == "halt_dn"                            # always, floor or no floor
        assert cmd.v_target in (0.0, 40.0)
        if cmd.v_target == 40.0:
            engaged.append((t, cmd.v_target, cmd.halt_reason))
    assert engaged and d.n_wander_floor == len(engaged)                # one engagement per event, fwd > 5 releases
    assert [e[2] for e in engaged] == ["halt_dn"] * len(engaged)
    assert not d.wander_floor_on                                       # released again by fwd 30 Hz
    # a genuine stall (fwd 0) engages it and keeps it engaged, with no halt_reason to report
    d2 = dec(explore=0.25)
    t, cmd, ev = run(d2, R(fwd=0.0), f, 62)
    assert cmd.v_target == pytest.approx(40.0) and cmd.halt_reason is None and kinds(ev) == ["wander_floor"]
    t, cmd, ev = run(d2, R(fwd=0.0, halt=30.0), f, 10, t0=t)
    assert cmd.v_target == pytest.approx(40.0) and cmd.halt_reason == "halt_dn" and ev == []


def test_wander_ou_stats():
    w = Wander(seed=1, sigma=0.6, tau_s=1.0)
    xs = np.array([w.sample(DT) for _ in range(10_200)])[200:]
    assert abs(float(xs.mean())) < 0.08
    expect = 0.6 * math.sqrt(1.0 / 2.0)
    assert abs(float(xs.std()) - expect) / expect < 0.15
    # deterministic per seed; sigma 0 -> exactly 0
    w1, w2 = Wander(7), Wander(7)
    assert [w1.sample(DT) for _ in range(50)] == [w2.sample(DT) for _ in range(50)]
    w0 = Wander(3, sigma=0.0)
    assert all(w0.sample(DT) == 0.0 for _ in range(10))
    # decoder noise on: omega varies tick to tick, mean turn is unbiased
    d = dec(seed=11, sigma=0.6)
    oms = []
    t = 0
    for _ in range(2000):
        t += 50
        oms.append(d.update(R(fwd=30.0), feat(), "CRUISING", t, DT).omega)
    assert np.std(oms) > 0.2 and abs(np.mean(oms)) < 0.1


# --------------------------------------------------------------------------- body
def test_body_bounce():
    cases = [
        # (start x, y, heading, expected side, expected heading after reflection, clamped axis/value)
        (790.0, 250.0, 0.3, "L", wrap_angle(PI - 0.3), ("x", 792.0)),
        (10.0, 250.0, PI - 0.3, "R", wrap_angle(0.3), ("x", 8.0)),
        (400.0, 10.0, -PI / 2 + 0.3, "L", wrap_angle(PI / 2 - 0.3), ("y", 8.0)),
        (400.0, 490.0, PI / 2 - 0.3, "R", wrap_angle(-(PI / 2 - 0.3)), ("y", 492.0)),
        (400.0, 3.0, 0.0, "L", 0.0, ("y", 8.0)),          # heading +x, wall straight above: on the fly's LEFT
        (400.0, 497.0, 0.0, "R", 0.0, ("y", 492.0)),      # wall straight below: on the RIGHT
    ]
    for x0, y0, h0, side, h_after, (axis, val) in cases:
        body = FlyBody(800, 500, "bounce", 0)
        body.teleport(x0, y0)
        body.heading = h0
        kin, ev = body.integrate(walk_cmd(400.0), 0.1, 100)
        assert kinds(ev) == ["wall_bump"], (x0, y0, h0, ev)
        assert ev[0]["data"]["side"] == side, (x0, y0, h0, ev)
        assert body.heading == pytest.approx(h_after, abs=1e-9), (x0, y0, h0)
        assert getattr(kin, axis) == pytest.approx(val)
        assert 8.0 <= kin.x <= 792.0 and 8.0 <= kin.y <= 492.0
        assert body.ink(kin, feat(), "CRUISING", 100).stamp == "bump"
    # a long random walk never leaves the margin
    body = FlyBody(800, 500, "bounce", 1)
    rng = np.random.default_rng(0)
    bumps = 0
    for i in range(2000):
        kin, ev = body.integrate(walk_cmd(300.0, float(rng.normal(0, 2.0))), DT, 50 * (i + 1))
        bumps += len(ev)
        assert 8.0 <= kin.x <= 792.0 and 8.0 <= kin.y <= 492.0
        assert -PI < kin.heading <= PI
    assert bumps > 0 and body.n_bumps == bumps


def test_body_corner_bounce():
    """Regression: a corner hit emitted two wall_bump events but counted one bump, and judged the second side
    against the already-reflected heading. SPEC f.4: side is judged before reflection; counter == events."""
    body = FlyBody(800, 500, "bounce", 0)
    body.teleport(12.0, 12.0)
    h0 = wrap_angle(PI + PI / 4)                                       # up-left, straight into the corner
    body.heading = h0
    kin, ev = body.integrate(walk_cmd(400.0), 0.2, 100)
    assert kinds(ev) == ["wall_bump", "wall_bump"]
    # left wall (normal -x) lies to the fly's left, top wall (normal -y) to its right - both judged from h0
    assert [e["data"]["side"] for e in ev] == ["L", "R"]
    assert body.n_bumps == len(ev) == 2
    assert kin.x == pytest.approx(8.0) and kin.y == pytest.approx(8.0)
    assert body.heading == pytest.approx(wrap_angle(-(PI - h0)), abs=1e-9)   # both reflections applied: down-right
    assert body.heading == pytest.approx(PI / 4, abs=1e-9)
    assert body.ink(kin, feat(), "CRUISING", 100).stamp == "bump"
    # the mirrored corner
    body.teleport(788.0, 488.0)
    body.heading = PI / 4                                              # down-right
    kin, ev = body.integrate(walk_cmd(400.0), 0.2, 200)
    assert [e["data"]["side"] for e in ev] == ["L", "R"] and body.n_bumps == 4
    assert body.heading == pytest.approx(wrap_angle(PI + PI / 4), abs=1e-9)


def test_body_wrap():
    body = FlyBody(800, 500, "wrap", 0)
    body.teleport(795.0, 250.0)
    body.heading = 0.0
    kin, ev = body.integrate(walk_cmd(400.0), 0.1, 100)
    v = 400.0 * (1.0 - math.exp(-0.1 / 0.15))
    assert kinds(ev) == ["wrap"] and ev[0]["data"] == {"edge": "right"}
    assert kin.x == pytest.approx(795.0 + v * 0.1 - 800.0) and kin.heading == 0.0 and kin.speed == pytest.approx(v)
    body.teleport(5.0, 250.0)
    body.heading = PI
    kin, ev = body.integrate(walk_cmd(400.0), 0.1, 200)
    assert ev[0]["data"]["edge"] == "left" and kin.x > 700.0
    body.teleport(400.0, 5.0)
    body.heading = -PI / 2
    kin, ev = body.integrate(walk_cmd(400.0), 0.1, 300)
    assert ev[0]["data"]["edge"] == "top" and kin.y > 400.0
    body.teleport(400.0, 495.0)
    body.heading = PI / 2
    kin, ev = body.integrate(walk_cmd(400.0), 0.1, 400)
    assert ev[0]["data"]["edge"] == "bottom" and kin.y < 100.0
    assert body.n_wraps == 4
    # never a wall_bump in wrap mode; positions stay inside [0, w) x [0, h)
    rng = np.random.default_rng(2)
    for i in range(1000):
        kin, ev = body.integrate(walk_cmd(300.0, float(rng.normal(0, 2.0))), DT, 500 + 50 * i)
        assert all(e["kind"] == "wrap" for e in ev)
        assert 0.0 <= kin.x < 800.0 and 0.0 <= kin.y < 500.0


def test_body_jump_decay():
    body = FlyBody(800, 500, "wrap", 0)
    body.heading = 0.5
    jump = {"heading": 1.5, "impulse": 600.0, "side": "R", "forward": False}
    ev_jump = {"kind": "jump", "t_ms": 1000, "data": {"heading_out": 1.5, "impulse": 600.0, "side": "R", "forward": False}}
    frozen = (dict(jump), dict(ev_jump["data"]))
    cmd = MotorCommand(mode="jump", v_target=600.0, omega=0.0, wing_hz=250.0, wing_amp=1.0, wing_ext=0,
                       proboscis=0.0, jump=jump, halt_reason=None, events=[ev_jump])
    kin, ev = body.integrate(cmd, DT, 1000)
    assert kin.mode == "jump" and kin.jump_t_ms == 0 and kin.speed == pytest.approx(600.0)
    assert kin.wing_hz == 250.0 and kin.wing_amp == 1.0
    # the body adopts the commanded absolute heading and never writes into the command or the event
    assert body.heading == pytest.approx(1.5)
    assert (dict(jump), dict(ev_jump["data"])) == frozen
    x0, y0 = body.x, body.y
    for k in range(1, 8):
        t = 1000 + 50 * k
        kin, ev = body.integrate(cmd, DT, t)
        assert kin.jump_t_ms == 50 * k
        assert kin.speed == pytest.approx(600.0 * math.exp(-(0.05 * k) / 0.15))
        assert kin.heading == pytest.approx(1.5)
    assert body.ink(kin, feat(), "CRUISING", 1350).stamp == "dash"
    assert body.ink(kin, feat(), "CRUISING", 1350).alpha == 0.0
    v_last = kin.speed
    kin, ev = body.integrate(walk_cmd(0.0), DT, 1400)
    assert kin.jump_t_ms is None and kin.mode == "walk"
    assert kin.speed == pytest.approx(v_last * math.exp(-DT / 0.15))
    # displacement along the jump heading, ~impulse * tau
    dist = math.hypot(body.x - x0, body.y - y0)
    assert 60.0 < dist < 100.0
    # re-entering jump mode starts a new jump (speed back to the impulse)
    jump2 = {"heading": 0.0, "impulse": 600.0, "side": "L", "forward": True}
    cmd2 = MotorCommand(mode="jump", v_target=600.0, omega=0.0, wing_hz=250.0, wing_amp=1.0, wing_ext=0,
                        proboscis=0.0, jump=jump2, halt_reason=None, events=[])
    kin, ev = body.integrate(cmd2, DT, 2000)
    assert kin.speed == pytest.approx(600.0) and kin.jump_t_ms == 0 and kin.heading == pytest.approx(0.0)


def test_body_jump_spec_shape_dict():
    """Regression: a jump dict restarted the jump on every tick (speed stuck at 600, jump_t_ms stuck at 0) unless
    it carried a private ``t0_ms`` key. The body tracks the jump start itself - mode ``jump`` entered from another
    mode - so the c.18 four-key dict (and a fresh copy of it per tick, as a replay produces) decays correctly."""
    body = FlyBody(800, 500, "wrap", 0)
    body.heading = 0.0
    j = {"heading": 1.0, "impulse": 600.0, "side": "R", "forward": False}
    cmd = MotorCommand("jump", 600.0, 0.0, 250.0, 1.0, 0, 0.0, j, None, [])
    out = []
    for k in range(4):
        kin, _ = body.integrate(cmd, DT, 1000 + 50 * k)
        out.append((kin.jump_t_ms, kin.speed))
    assert [o[0] for o in out] == [0, 50, 100, 150]
    assert [o[1] for o in out] == pytest.approx([600.0 * math.exp(-(0.05 * k) / 0.15) for k in range(4)])
    assert body.heading == pytest.approx(1.0)                         # the commanded absolute heading is honoured
    assert j == {"heading": 1.0, "impulse": 600.0, "side": "R", "forward": False}   # the spec dict is not mutated
    # leaving jump mode and re-entering it (a later GF spike) starts a fresh jump even with the same dict object
    body.integrate(walk_cmd(0.0), DT, 1200)
    kin, _ = body.integrate(cmd, DT, 3000)
    assert kin.jump_t_ms == 0 and kin.speed == pytest.approx(600.0)
    # a fresh dict per tick (replay from JSON) also decays instead of restarting
    body.integrate(walk_cmd(0.0), DT, 3050)
    speeds = []
    for k in range(3):
        c = MotorCommand("jump", 600.0, 0.0, 250.0, 1.0, 0, 0.0, dict(j), None, [])
        speeds.append(body.integrate(c, DT, 4000 + 50 * k)[0].speed)
    assert speeds[0] > speeds[1] > speeds[2]


def test_leg_phase_advances_one_cycle_per_20px():
    body = FlyBody(800, 500, "wrap", 0)
    expected = 0.0
    dist = 0.0
    for i in range(300):
        kin, _ = body.integrate(walk_cmd(100.0, 0.3), DT, 50 * (i + 1))
        dist += abs(kin.speed) * DT
        expected = (expected + abs(kin.speed) / 20.0 * DT) % 1.0
        assert 0.0 <= kin.leg_phase < 1.0
        assert kin.leg_phase == pytest.approx(expected, abs=1e-9)
    assert dist > 1000.0
    # backing (negative v) still advances the gait; standing still does not
    body2 = FlyBody(800, 500, "wrap", 0)
    kin, _ = body2.integrate(walk_cmd(-40.0), 1.0, 1000)
    assert kin.speed < 0 and kin.leg_phase > 0.0
    lp = kin.leg_phase
    kin, _ = body2.integrate(still_cmd("feed"), 5.0, 6000)
    assert kin.speed == pytest.approx(0.0, abs=1e-6) and kin.leg_phase == pytest.approx(lp, abs=1e-3)


def test_ink_style_table():
    body = FlyBody(800, 500, "bounce", 0)
    kin, _ = body.integrate(walk_cmd(50.0), DT, 50)
    # candle colours
    assert body.ink(kin, feat(candle="up"), "CRUISING", 50).color == "#00a800"
    assert body.ink(kin, feat(candle="down"), "CRUISING", 50).color == "#a80000"
    assert body.ink(kin, feat(candle="flat"), "CRUISING", 50).color == "#000000"
    assert body.ink(kin, feat(candle="up"), "ANXIOUS", 50).color == "#00a800"   # no override for ANXIOUS
    # mood colour + style overrides
    table = {
        "FEEDING": ("#ff8c00", "dotted"), "COURTSHIP": ("#ff69b4", "hearts"), "PANIC": ("#ff0000", "zigzag"),
        "ESCAPE": ("#ff3300", "zigzag"), "SLEEP": ("#6b6b6b", "solid"), "EUPHORIA": ("#ff00ff", "rainbow"),
        "CRUISING": ("#00a800", "solid"), "ANXIOUS": ("#00a800", "solid"),
    }
    for mood, (color, style) in table.items():
        ink = body.ink(kin, feat(candle="up"), mood, 50)
        assert isinstance(ink, InkStyle) and ink.color == color and ink.style == style, mood
        assert ink.style in D.INK_STYLES and (ink.stamp is None or ink.stamp in D.STAMPS)
    # width 1..4 from |chg_m5| (attribute if present, else inverted from mom = tanh(chg/2))
    for chg, w in ((0.0, 1.0), (0.4, 1.0), (0.5, 2.0), (1.0, 2.0), (1.5, 3.0), (2.0, 3.0), (2.5, 4.0), (3.0, 4.0),
                   (-10.0, 4.0), (100.0, 4.0)):
        assert body.ink(kin, feat(chg_m5=chg), "CRUISING", 50).width == w, chg
        assert body.ink(kin, feat(mom=math.tanh(chg / 2.0)), "CRUISING", 50).width == w, chg
    assert body.ink(kin, feat(chg_m5=float("nan")), "CRUISING", 50).width == 1.0
    # alpha per mode
    for mode, alpha in (("walk", 1.0), ("feed", 1.0), ("freeze", 1.0), ("groom", 1.0), ("sleep", 1.0),
                        ("fly", 0.4), ("court", 0.7), ("jump", 0.0)):
        k = Kinematics(x=1, y=1, vx=0, vy=0, heading=0, speed=0, omega=0, wing_hz=0, wing_amp=0, wing_ext=0,
                       mode=mode, leg_phase=0, proboscis=0, jump_t_ms=0 if mode == "jump" else None)
        assert body.ink(k, feat(), "CRUISING", 50).alpha == alpha, mode
    # stamp cadence
    b2 = FlyBody(800, 500, "bounce", 0)
    kf, _ = b2.integrate(still_cmd("feed"), DT, 1000)
    assert b2.ink(kf, feat(), "FEEDING", 1000).stamp == "blob"
    assert b2.ink(kf, feat(), "FEEDING", 1100).stamp is None
    assert b2.ink(kf, feat(), "FEEDING", 1250).stamp == "blob"
    kc, _ = b2.integrate(still_cmd("court"), DT, 2000)
    assert b2.ink(kc, feat(), "COURTSHIP", 2000).stamp == "heart"
    assert b2.ink(kc, feat(), "COURTSHIP", 2400).stamp is None
    assert b2.ink(kc, feat(), "COURTSHIP", 2500).stamp == "heart"
    ks, _ = b2.integrate(still_cmd("sleep"), DT, 3000)
    assert b2.ink(ks, feat(), "SLEEP", 3000).stamp == "zzz"
    assert b2.ink(ks, feat(), "SLEEP", 3450).stamp is None
    assert b2.ink(ks, feat(), "SLEEP", 3500).stamp == "zzz"
    kw, _ = b2.integrate(walk_cmd(50.0), DT, 4000)
    assert b2.ink(kw, feat(), "CRUISING", 4000).stamp is None
    # the freeze tremor only touches the wire pose
    b3 = FlyBody(800, 500, "bounce", 4)
    b3.teleport(300.0, 200.0)
    xs = []
    for i in range(20):
        k, _ = b3.integrate(still_cmd("freeze"), DT, 50 * (i + 1))
        xs.append((k.x, k.y))
        assert abs(k.x - 300.0) <= 1.5 and abs(k.y - 200.0) <= 1.5
    assert len(set(xs)) > 1                                            # the tremor jitters the wire pose
    assert len({x for x, _ in xs}) > 1 and len({y for _, y in xs}) > 1  # ... on both axes
    assert (b3.x, b3.y) == (300.0, 200.0)                              # ... and never the true position


def test_kinematics_to_wire_keys():
    body = FlyBody(800, 500, "bounce", 0)
    kin, _ = body.integrate(walk_cmd(50.0, 0.1), DT, 50)
    w = kin.to_wire()
    assert set(w) == FLY_WIRE_KEYS
    assert w["mode"] == "walk" and w["jump_t_ms"] is None and isinstance(w["wing_ext"], int)
    assert w["x"] == round(kin.x, 1) and w["heading"] == round(kin.heading, 3)
    assert set(InkStyle("#000000", 1.0, 1.0, "solid", None).to_wire()) == {"color", "width", "alpha", "style", "stamp"}
    # initial pose: centre, heading 0, mode walk
    b0 = FlyBody(800, 500, "bounce", 0)
    assert (b0.kin.x, b0.kin.y, b0.kin.heading, b0.kin.mode) == (400.0, 250.0, 0.0, "walk")
    b0.teleport(10.0, 20.0)
    assert (b0.kin.x, b0.kin.y) == (10.0, 20.0) and b0.pen_lifted


def test_wing_hz_serialised_at_two_dp():
    """SPEC d preamble rounding: 'positions/speeds 1 dp, angles 3 dp, rates 2 dp, ...'. ``wing_hz`` is a Hz
    quantity (``150 + 100*wing_power``, so fractional in flight), and the frequency bucket is the only one that
    fits it - it is not a px/s speed (1 dp) nor an angle (3 dp). Regression: ``to_wire()`` grouped it with the
    1-dp speeds, dropping the second decimal of the wingbeat."""
    def wire_wing_hz(v: float) -> float:
        k = Kinematics(x=0, y=0, vx=0, vy=0, heading=0, speed=0, omega=0, wing_hz=v, wing_amp=0.5, wing_ext=0,
                       mode="fly", leg_phase=0, proboscis=0, jump_t_ms=None)
        return k.to_wire()["wing_hz"]

    assert wire_wing_hz(215.756) == 215.76                            # 2 dp, not 215.8
    assert wire_wing_hz(150.0 + 100.0 * 0.3337) == round(150.0 + 100.0 * 0.3337, 2)
    assert wire_wing_hz(0.0) == 0.0
    # the other fly.* fields keep their own buckets: x/speed 1 dp, heading/omega 3 dp, 0-1 scores 3 dp
    body = FlyBody(800, 500, "wrap", 0)
    fly = MotorCommand(mode="fly", v_target=180.0, omega=0.123456, wing_hz=222.2222, wing_amp=0.654321,
                       wing_ext=0, proboscis=0.0, jump=None, halt_reason=None, events=[])
    kin, _ = body.integrate(fly, DT, 50)
    w = kin.to_wire()
    assert w["wing_hz"] == round(kin.wing_hz, 2) and w["heading"] == round(kin.heading, 3)
    assert w["speed"] == round(kin.speed, 1) and w["wing_amp"] == round(kin.wing_amp, 3)


# --------------------------------------------------------------------------- event semantics (regressions)
def test_jump_from_flight_lands():
    """Regression: a GF jump taken mid-flight ended with the fly silently on the ground (flying False, no
    'landing') and a fresh 'takeoff' 150 ms later. Now the jump end emits landing {reason: 'jump'} when the
    threat is gone, and no second takeoff when the fly never landed (looming >= 0.5 -> fly continues)."""
    d = dec()
    f = feat()
    t, cmd, ev = run(d, R(flight=50.0), f, 5)
    assert cmd.mode == "fly" and kinds(ev) == ["takeoff"]
    out = []
    for i in range(12):
        t += 50
        cmd = d.update(R(flight=50.0, gf_spikes_l=1 if i == 0 else 0), f, "CRUISING", t, DT)
        out += [(t, cmd.mode, e["kind"], e["data"].get("reason")) for e in cmd.events]
    assert out == [(300, "jump", "gf_spike", None), (300, "jump", "jump", None),
                   (700, "walk", "landing", "jump"), (850, "fly", "takeoff", None)]
    # fly -> jump -> fly (looming still >= 0.5 at the jump end): no landing, no second takeoff
    d2 = dec()
    t, cmd, ev = run(d2, R(flight=50.0), f, 5)
    t, cmd, ev = run(d2, R(flight=50.0, gf_spikes_r=1), feat(looming=0.9), 1, t0=t)
    assert cmd.mode == "jump"
    t, cmd, ev = run(d2, R(flight=50.0), feat(looming=0.9), 9, t0=t)
    assert cmd.mode == "fly" and kinds(ev) == [] and d2.flying
    # walk -> jump -> fly: the takeoff is emitted at the jump end (SPEC f.3 'then fly ... (event takeoff)')
    d3 = dec()
    t, cmd, ev = run(d3, R(gf_spikes_r=1), feat(looming=0.9), 1)
    t, cmd, ev = run(d3, R(), feat(looming=0.9), 8, t0=t)
    assert cmd.mode == "fly" and kinds(ev) == ["takeoff"]


def test_masked_subbehaviour_events():
    """Regression: takeoff/freeze/groom events fired while a higher-priority mode masked them (a 'takeoff'
    published while the fly visibly fed) and the later real mode switch was silent. Events now mark the
    visible mode entering / leaving fly / freeze / groom; feed_* follow the public feeding flag."""
    d = dec()
    f = feat()
    out = []
    t = 0
    for _ in range(10):
        t += 50
        cmd = d.update(R(mn9=50.0, flight=50.0), f, "CRUISING", t, DT)
        out += [(t, cmd.mode, e["kind"], e["data"].get("reason")) for e in cmd.events]
    for _ in range(25):
        t += 50
        cmd = d.update(R(mn9=0.0, flight=50.0), f, "CRUISING", t, DT)
        out += [(t, cmd.mode, e["kind"], e["data"].get("reason")) for e in cmd.events]
    assert out == [(200, "feed", "feed_start", None), (1500, "fly", "feed_stop", None), (1500, "fly", "takeoff", None)]
    assert d.flying and d._fly_visible
    # a meal that starts in flight lands the fly (reason = the masking mode) and it takes off again afterwards
    d2 = dec()
    t, cmd, ev = run(d2, R(flight=50.0), f, 5)
    assert cmd.mode == "fly"
    t, cmd, ev = run(d2, R(flight=50.0, mn9=50.0), f, 4, t0=t)
    assert cmd.mode == "feed" and kinds(ev) == ["feed_start", "landing"] and ev[1]["data"]["reason"] == "feed"
    t, cmd, ev = run(d2, R(flight=50.0, mn9=0.0), f, 20, t0=t)
    assert cmd.mode == "fly" and kinds(ev) == ["feed_stop", "takeoff"]
    # a freeze underneath a meal never emits freeze/unfreeze; it becomes visible only when the meal ends
    d3 = dec()
    t, cmd, ev = run(d3, R(mn9=50.0), f, 4)
    assert cmd.mode == "feed"
    t, cmd, ev = run(d3, R(mn9=50.0, freeze=45.0), feat(looming=0.5), 5, t0=t)
    assert cmd.mode == "feed" and d3.frozen and ev == []
    t, cmd, ev = run(d3, R(mn9=0.0, freeze=45.0), feat(looming=0.5), 20, t0=t)
    assert cmd.mode == "freeze" and kinds(ev) == ["feed_stop", "freeze"]
    t, cmd, ev = run(d3, R(mn9=0.0, freeze=0.0), feat(looming=0.0), 20, t0=t)
    assert cmd.mode == "walk" and kinds(ev) == ["unfreeze"]
    # a jump breaks a visible freeze: exactly one unfreeze
    d4 = dec()
    t, cmd, ev = run(d4, R(freeze=45.0), feat(looming=0.5), 2)
    assert cmd.mode == "freeze" and kinds(ev) == ["freeze"]
    t, cmd, ev = run(d4, R(gf_spikes_r=1), feat(looming=0.5), 1, t0=t)
    assert cmd.mode == "jump" and kinds(ev) == ["gf_spike", "jump", "unfreeze"] and not d4.frozen
    # groom masked by court: the groom event waits for the visible switch
    d5 = dec()
    t, cmd, ev = run(d5, R(groom=45.0, p1=20.0), f, 3)
    assert cmd.mode == "court" and d5.grooming and ev == []
    t, cmd, ev = run(d5, R(groom=45.0), f, 1, t0=t)
    assert cmd.mode == "groom" and kinds(ev) == ["groom"]


def test_gf_spike_event():
    """SPEC d.8: gf_spike {side, count} - one per tick with DNp01 spikes, also inside the refractory window."""
    d = dec()
    f = feat()
    cmd = d.update(R(gf_spikes_l=2, gf_spikes_r=1), f, "CRUISING", 50, DT)
    assert cmd.events[0] == {"kind": "gf_spike", "t_ms": 50, "data": {"side": "both", "count": 3}}
    cmd = d.update(R(gf_spikes_l=1), f, "CRUISING", 100, DT)          # refractory: no jump, still reported
    assert kinds(cmd.events) == ["gf_spike"] and cmd.events[0]["data"] == {"side": "L", "count": 1}
    cmd = d.update(R(), f, "CRUISING", 150, DT)
    assert cmd.events == []


def test_mood_name_upper():
    assert D.mood_name("sleep") == "SLEEP" and D.mood_name(None) == "CRUISING"
    from flybrain.mood import Mood, MoodMachine
    assert D.mood_name(Mood.PANIC) == "PANIC" and D.mood_name(MoodMachine(0.05).state) == "CRUISING"
    d = dec()
    assert d.update(R(fwd=30.0), feat(), "sleep", 50, DT).mode == "sleep"


def test_dataclass_slots():
    """SPEC c preamble: dataclasses are slots=True unless frozen."""
    from flybrain.mood import MoodState
    for cls in (MotorCommand, Kinematics, MoodState):
        assert hasattr(cls, "__slots__"), cls
    cmd = walk_cmd(1.0)
    with pytest.raises(AttributeError):
        cmd.extra = 1  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- determinism / robustness
def test_determinism():
    def trace(seed: int) -> list[tuple]:
        d = MotorDecoder(settings(sigma=0.6), seed)
        body = FlyBody(800, 500, "bounce", seed)
        rng = np.random.default_rng(99)
        out = []
        for i in range(300):
            t = 50 * (i + 1)
            r = R(fwd=float(rng.uniform(0, 40)), a02_r=float(rng.uniform(0, 30)), a02_l=float(rng.uniform(0, 30)),
                  mn9=float(rng.uniform(0, 50)), freeze=float(rng.uniform(0, 50)),
                  gf_spikes_r=int(rng.random() < 0.02), flight=float(rng.uniform(0, 60)))
            f = feat(looming=float(rng.uniform(0, 1)), candle="up")
            cmd = d.update(r, f, "CRUISING", t, DT)
            kin, ev = body.integrate(cmd, DT, t)
            d.observe(kin)
            ink = body.ink(kin, f, "CRUISING", t)
            out.append((cmd.mode, cmd.v_target, cmd.omega, kin.x, kin.y, kin.heading, ink.width, ink.stamp,
                        tuple(e["kind"] for e in cmd.events + ev)))
        return out

    a, b, c = trace(1337), trace(1337), trace(42)
    assert a == b
    assert a != c
    assert any(m == "jump" for (m, *_rest) in a)


def test_no_nan_for_zero_or_extreme_rates():
    big = 1e12
    extremes = [
        R(),
        R(**{k: big for k in ("fwd", "halt", "back", "freeze", "a02_l", "a02_r", "a01_l", "a01_r", "g13_l", "g13_r",
                              "flight", "wing_power", "b1_l", "b1_r", "i1_l", "i1_r", "hg1_l", "hg1_r", "saccade_l",
                              "saccade_r", "land", "mn9", "groom", "p1", "pip10", "song_mn", "dms2_l", "dms2_r",
                              "lc_loom", "dn_mean")}, gf_spikes_l=10 ** 6, gf_spikes_r=10 ** 6, dnp11_spikes=10 ** 6),
        R(fwd=float("nan"), a02_r=float("inf"), a02_l=float("-inf"), mn9=float("nan"), freeze=float("inf"),
          flight=float("nan"), back=float("inf")),
        R(fwd=-big, a02_r=-big, halt=-big, back=-big, flight=-big, mn9=-big),
    ]
    feats = [feat(), feat(looming=float("nan"), mom=float("inf"), candle="up"), feat(looming=1e9, sustained_loom=True)]
    for walls in ("bounce", "wrap"):
        d = MotorDecoder(settings(sigma=0.6), 5)
        body = FlyBody(800, 500, walls, 5)
        t = 0
        for i in range(400):
            t += 50
            r = extremes[i % len(extremes)]
            f = feats[i % len(feats)]
            mood = ["CRUISING", "SLEEP", "COURTSHIP", "PANIC"][i % 4]
            for dt in (DT, 0.0, 0.0125):
                cmd = d.update(r, f, mood, t, dt)
                for v in (cmd.v_target, cmd.omega, cmd.wing_hz, cmd.wing_amp, cmd.proboscis):
                    assert math.isfinite(v), (i, dt, cmd)
                assert cmd.mode in D.MODES and cmd.wing_ext in (-1, 0, 1)
                assert 0.0 <= cmd.wing_amp <= 1.0 and 0.0 <= cmd.proboscis <= 1.0
                kin, ev = body.integrate(cmd, dt, t)
                w = kin.to_wire()
                for k, v in w.items():
                    if isinstance(v, float):
                        assert math.isfinite(v), (i, k, v)
                assert 0.0 <= w["x"] <= 800.0 and 0.0 <= w["y"] <= 500.0
                ink = body.ink(kin, f, mood, t)
                assert 1.0 <= ink.width <= 4.0 and 0.0 <= ink.alpha <= 1.0 and ink.color.startswith("#")
                for e in cmd.events + ev:
                    assert set(e) == {"kind", "t_ms", "data"} and isinstance(e["data"], dict)
                    for v in e["data"].values():
                        if isinstance(v, float):
                            assert math.isfinite(v)
    # a NaN Features looming / a None mood never break the decoder
    d = dec()
    cmd = d.update(R(fwd=30.0), SimpleNamespace(), None, 50, DT)
    assert cmd.mode == "walk" and math.isfinite(cmd.omega)


def test_ink_colour_width_vs_candle_and_wing_hz():
    """SPEC f.5 derives the colour from the candle sign (mood overriding it) and the width from |chg_m5| only -
    the wingbeat never enters the ink. The only wing-correlated effect is the 0.4 alpha of ``fly`` mode (and the
    0.0 of ``jump``), so the same market state paints the same colour and width on the ground and in the air."""
    body = FlyBody(800, 500, "wrap", 0)
    fly = MotorCommand(mode="fly", v_target=400.0, omega=0.0, wing_hz=250.0, wing_amp=1.0, wing_ext=1,
                       proboscis=0.0, jump=None, halt_reason=None, events=[])
    out = {}
    for t, cmd in ((1000, walk_cmd(100.0)), (1050, fly)):
        kin, _ = body.integrate(cmd, DT, t)
        assert kin.wing_hz == (0.0 if cmd.mode == "walk" else 250.0)
        out[cmd.mode] = body.ink(kin, feat(candle="down", chg_m5=2.0), "CRUISING", t)
    assert out["walk"].color == out["fly"].color == D.INK_DOWN
    assert out["walk"].width == out["fly"].width == 3.0
    assert out["walk"].style == out["fly"].style == "solid"
    assert (out["walk"].alpha, out["fly"].alpha) == (1.0, 0.4)
    # candle sign flips the colour both ways at the same width; a mood override ignores the candle entirely
    kin, _ = body.integrate(walk_cmd(100.0), DT, 1100)
    for candle, colour in (("up", D.INK_UP), ("down", D.INK_DOWN), ("flat", D.INK_FLAT)):
        ink = body.ink(kin, feat(candle=candle, chg_m5=-2.0), "CRUISING", 1100)
        assert (ink.color, ink.width) == (colour, 3.0), candle      # |-2.0| / 3 -> 1 + round(2.0) = 3 px
        assert body.ink(kin, feat(candle=candle, chg_m5=-2.0), "PANIC", 1100).color == "#ff0000"
    # width is monotone non-decreasing in |chg_m5| and stays in 1..4
    widths = [body.ink(kin, feat(candle="up", chg_m5=c / 10.0), "CRUISING", 1100).width for c in range(0, 80)]
    assert widths[0] == 1.0 and widths[-1] == 4.0
    assert all(1.0 <= w <= 4.0 for w in widths)
    assert all(b >= a for a, b in zip(widths, widths[1:]))


def test_ink_width_rounding_and_mom_fallback():
    """Pins the two unspecified corners of SPEC f.5 ``width = 1 + round(3 * sat(|chg_m5| / 3))``:

    * the rounding convention is **half-up** (``floor(x + 0.5)``, the JS ``Math.round`` the client would use),
      so 0.5 % -> 2 px, 1.5 % -> 3 px, 2.5 % -> 4 px where Python's banker's ``round`` would give 1/3/3 px;
    * ``Features`` (SPEC c.17) carries ``mom = tanh(chg_m5 / 2)`` and no ``chg_m5`` field, so the width is
      derived by inverting ``mom`` when the attribute is missing. Both paths must agree.
    """
    body = FlyBody(800, 500, "wrap", 0)
    kin, _ = body.integrate(walk_cmd(10.0), DT, 50)

    def w(**kw) -> float:
        return body.ink(kin, feat(**kw), "CRUISING", 50).width

    for chg, expect in ((0.0, 1.0), (0.4999, 1.0), (0.5, 2.0), (1.4999, 2.0), (1.5, 3.0), (2.5, 4.0)):
        assert w(chg_m5=chg) == expect, chg
        assert w(chg_m5=-chg) == expect, -chg
    # the mom fallback reproduces the chg_m5 path to the pixel over the whole sensitive range
    for i in range(-80, 81):
        chg = i / 10.0
        assert w(mom=math.tanh(chg / 2.0)) == w(chg_m5=chg), chg
    # saturation: |mom| -> 1 is 4 px, and a mom of exactly +-1 (chg beyond float atanh) stays 4 px
    assert w(mom=1.0) == 4.0 and w(mom=-1.0) == 4.0
    assert w(chg_m5=1e9) == 4.0 and w(mom=float("nan")) == 1.0
