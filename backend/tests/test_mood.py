"""Tests for ``flybrain.mood`` (SPEC c.19, f.6, h.2 ``test_mood.py``).

Scores are EMAs with tau = 2 s by default; the transition tests use ``tau_score_s=1e-6`` so that the
scores track ``E_raw`` / ``A_raw`` instantly and every "held T" rule can be driven by plain input
sequences. ``test_scores`` checks the EMA itself with the default tau.
"""

from __future__ import annotations

import math

import pytest

from flybrain.mood import (
    CONFIRM_HOLD_MS,
    COURTSHIP_DWELL_MS,
    MIN_DWELL_MS,
    MOOD_COLORS,
    MOOD_PRIORITY,
    TIMER_KEYS,
    Mood,
    MoodInputs,
    MoodMachine,
    MoodState,
    Transition,
)

TICK_S = 0.05
TICK_MS = 50

#: the ``tick.mood`` keys of SPEC d.2
WIRE_KEYS = {"state", "prev", "since_ms", "euphoria", "anxiety", "arousal", "valence", "fear", "hunger", "sleep",
             "dwell_left_ms"}


def inp(t_ms: int, **kw) -> MoodInputs:
    base = dict(
        t_ms=t_ms, sugar=0.0, looming=0.0, bitter=0.0, activity=0.5, sleep_pressure=0.0,
        mn9=0.0, dnp09=0.0, lc_loom=0.0, pam=0.0, ppl1=0.0, mbon_approach=0.0, mbon_avoid=0.0,
        p1=0.0, pip10=0.0, dn_mean=3.0, feeding=False, jumped=False, gf_spikes_10s=0,
        any_drive_max=0.3, poked=False, forced=None, hunger=0.5,
    )
    base.update(kw)
    return MoodInputs(**base)


def mk(tau: float = 1e-6) -> MoodMachine:
    return MoodMachine(tick_s=TICK_S, tau_score_s=tau)


def run(m: MoodMachine, t0: int, n: int, **kw) -> tuple[int, list[Transition]]:
    """Run ``n`` ticks of 50 ms starting at ``t0 + 50``; returns (last t_ms, transitions)."""
    trs: list[Transition] = []
    t = t0
    for _ in range(n):
        t += TICK_MS
        _, tr = m.update(inp(t, **kw))
        if tr is not None:
            trs.append(tr)
    return t, trs


# looming 1.0 + dnp09 40 Hz -> A_raw 0.80 (PANIC); looming 1.0 -> 0.50; looming 0.5 -> 0.25; looming 0.8 -> 0.40
PANIC_KW = dict(looming=1.0, dnp09=40.0)
# sugar 1 + mn9 60 -> E_raw 0.80
EUPH_KW = dict(sugar=1.0, mn9=60.0)


# --------------------------------------------------------------------------- scores
def test_scores():
    m = mk(tau=2.0)
    x = inp(50, sugar=0.5, mn9=30.0, pam=20.0, looming=0.4, dnp09=20.0, lc_loom=30.0, dn_mean=2.5,
            any_drive_max=0.6, mbon_approach=10.0, mbon_avoid=5.0, hunger=0.3, sleep_pressure=0.7)
    e_raw = 0.45 * 0.5 + 0.35 * 0.5 + 0.20 * 0.5          # 0.5
    a_raw = 0.50 * 0.4 + 0.30 * 0.5 + 0.20 * 0.5          # 0.45
    ar_raw = 0.5 * 0.5 + 0.5 * 0.6                          # 0.55
    alpha = 1.0 - math.exp(-0.05 / 2.0)
    s, _ = m.update(x)
    assert s.euphoria == pytest.approx(e_raw * alpha, abs=1e-9)
    assert s.anxiety == pytest.approx(a_raw * alpha, abs=1e-9)
    assert s.fear == s.anxiety
    assert s.arousal == pytest.approx(ar_raw * alpha, abs=1e-9)
    mbon = (10.0 - 5.0) / (10.0 + 5.0 + 5.0)
    assert s.valence == pytest.approx(s.euphoria - s.anxiety + 0.2 * mbon, abs=1e-9)
    assert s.hunger == pytest.approx(0.3)
    assert s.sleep == pytest.approx(0.7)
    # converge: after 20 s the EMA sits at the raw value; valence clipped to [-1, 1]
    for k in range(2, 402):
        s, _ = m.update(inp(50 * k, **{kk: getattr(x, kk) for kk in (
            "sugar", "mn9", "pam", "looming", "dnp09", "lc_loom", "dn_mean", "any_drive_max", "mbon_approach",
            "mbon_avoid", "hunger", "sleep_pressure")}))
    assert s.euphoria == pytest.approx(e_raw, abs=1e-4)
    assert s.anxiety == pytest.approx(a_raw, abs=1e-4)
    assert s.arousal == pytest.approx(ar_raw, abs=1e-4)
    assert -1.0 <= s.valence <= 1.0
    # NaN inputs never poison the scores
    s, _ = m.update(inp(50 * 402, sugar=float("nan"), looming=float("inf"), mn9=float("-inf")))
    assert all(math.isfinite(v) for v in (s.euphoria, s.anxiety, s.arousal, s.valence))


# --------------------------------------------------------------------------- one test per table row
def test_escape_on_jump():
    # immediate from any state, ignoring dwell
    for start_kw, start_state in ((PANIC_KW, Mood.PANIC), (EUPH_KW, Mood.EUPHORIA), ({}, Mood.CRUISING)):
        m = mk()
        t, _ = run(m, 0, 80, **start_kw)
        assert m.state.state is start_state
        s, tr = m.update(inp(t + TICK_MS, jumped=True, **start_kw))
        assert tr is not None and tr.dst is Mood.ESCAPE and tr.src is start_state
        assert s.state is Mood.ESCAPE and s.since_ms == 0 and s.dwell_left_ms == 1500
        assert "jump" in tr.reason
    # exits after exactly 1.5 s by anxiety: PANIC (>= 0.45) / ANXIOUS (>= 0.20) / CRUISING
    for kw, dst in ((dict(looming=1.0), Mood.PANIC), (dict(looming=0.5), Mood.ANXIOUS), ({}, Mood.CRUISING)):
        m = mk()
        s, tr = m.update(inp(50, jumped=True, **kw))
        assert tr.dst is Mood.ESCAPE
        t, trs = run(m, 50, 29, **kw)                 # 1450 ms in ESCAPE: nothing yet
        assert trs == [] and m.state.state is Mood.ESCAPE
        s, tr = m.update(inp(t + TICK_MS, **kw))      # 1500 ms
        assert tr is not None and tr.src is Mood.ESCAPE and tr.dst is dst, (kw, tr)
        assert "escape over" in tr.reason
    # a second jump while escaping extends the escape, no re-entry
    m = mk()
    m.update(inp(50, jumped=True))
    t, trs = run(m, 50, 10)
    s, tr = m.update(inp(t + TICK_MS, jumped=True))
    assert tr is None and s.state is Mood.ESCAPE and s.dwell_left_ms == 1500


def anxious_machine() -> tuple[MoodMachine, int]:
    """A machine sitting in ANXIOUS (anxiety 0.40) with its dwell over - the natural state before PANIC."""
    m = mk()
    t, trs = run(m, 0, 20, looming=0.8)               # anxiety 0.40 >= 0.35 held 1.0 s -> ANXIOUS
    assert [tr.dst for tr in trs] == [Mood.ANXIOUS]
    t, trs = run(m, t, 30, looming=0.8)               # dwell over
    assert trs == [] and m.state.dwell_left_ms == 0
    return m, t


def test_panic_requires_hold():
    # from CRUISING a high anxiety first passes through ANXIOUS (held 1 s), then PANIC once held 2 s + dwell
    m0 = mk()
    t, trs = run(m0, 0, 60, **PANIC_KW)
    assert [(tr.dst, tr.t_ms) for tr in trs] == [(Mood.ANXIOUS, 1000), (Mood.PANIC, 2500)]
    # the PANIC hold itself: 1.9 s -> nothing, 2.0 s -> PANIC
    m, t = anxious_machine()
    t, trs = run(m, t, 38, **PANIC_KW)                # 1.9 s at anxiety 0.80
    assert trs == [] and m.state.state is Mood.ANXIOUS
    assert m.state.timers["panic_enter"] == 1900
    t, trs = run(m, t, 2, **PANIC_KW)                 # 2.0 s
    assert len(trs) == 1 and trs[0].dst is Mood.PANIC and trs[0].src is Mood.ANXIOUS
    assert "anxiety 0.80 >= 0.70 for 2.0 s" == trs[0].reason
    # exit: anxiety < 0.45 held 3000 -> ANXIOUS
    t, trs = run(m, t, 59, looming=0.5)               # anxiety 0.25 for 2.95 s
    assert trs == [] and m.state.state is Mood.PANIC
    t, trs = run(m, t, 1, looming=0.5)
    assert len(trs) == 1 and trs[0].dst is Mood.ANXIOUS and "for 3.0 s" in trs[0].reason


def test_panic_on_gf_burst():
    m = mk()
    s, tr = m.update(inp(50, gf_spikes_10s=3))
    assert tr is not None and tr.dst is Mood.PANIC and "gf_spikes_10s 3 >= 3" == tr.reason
    m2 = mk()
    s, tr = m2.update(inp(50, gf_spikes_10s=2))
    assert tr is None and s.state is Mood.CRUISING


def test_courtship_dwell_8s():
    m = mk()
    t, trs = run(m, 0, 9, p1=20.0)                    # 450 ms held
    assert trs == []
    t, trs = run(m, t, 1, p1=20.0)                    # 500 ms
    assert len(trs) == 1 and trs[0].dst is Mood.COURTSHIP
    t_enter = t
    # p1 and pip10 silent immediately: exit needs 8 s dwell THEN 2 s held -> 10 s after entry
    t, trs = run(m, t, 199)                           # 9.95 s
    assert trs == [] and m.state.state is Mood.COURTSHIP
    t, trs = run(m, t, 1)
    assert len(trs) == 1 and trs[0].dst is Mood.CRUISING and t - t_enter == COURTSHIP_DWELL_MS + 2000
    # forced == "COURTSHIP" enters at once (after the general dwell)
    m2 = mk()
    s, tr = m2.update(inp(50, forced="COURTSHIP"))
    assert tr is not None and tr.dst is Mood.COURTSHIP and tr.reason == "forced COURTSHIP"
    # pip10 path
    m3 = mk()
    t, trs = run(m3, 0, 10, pip10=25.0)
    assert len(trs) == 1 and trs[0].dst is Mood.COURTSHIP
    # a threat still pre-empts courtship after the general 1.5 s dwell (PANIC > COURTSHIP)
    m4 = mk()
    t, trs = run(m4, 0, 10, p1=20.0)
    assert m4.state.state is Mood.COURTSHIP
    t, trs = run(m4, t, 40, p1=20.0, **PANIC_KW)
    assert [tr.dst for tr in trs] == [Mood.PANIC]


def test_euphoria_needs_low_anxiety():
    m = mk()
    t, trs = run(m, 0, 60, **EUPH_KW)                 # euphoria 0.80 held 3.0 s, anxiety 0 -> EUPHORIA
    assert len(trs) == 1 and trs[0].dst is Mood.EUPHORIA
    assert trs[0].reason.startswith("euphoria 0.80 >= 0.70 for 3.0 s")
    # with anxiety 0.40 (looming 0.8) the entry is blocked even though euphoria is held
    m2 = mk()
    t, trs = run(m2, 0, 100, looming=0.8, **EUPH_KW)
    assert all(tr.dst is not Mood.EUPHORIA for tr in trs)
    assert m2.state.state is Mood.ANXIOUS              # anxiety 0.40 >= 0.35 held 1 s wins instead
    # exit: euphoria < 0.40 held 2 s -> FEEDING if feeding else CRUISING
    t, trs = run(m, t, 40, sugar=0.4, feeding=True)   # E_raw 0.18
    assert len(trs) == 1 and trs[0].dst is Mood.FEEDING and "euphoria 0.18 < 0.40 for 2.0 s" == trs[0].reason
    m3 = mk()
    t, trs = run(m3, 0, 60, **EUPH_KW)
    t, trs = run(m3, t, 40, sugar=0.4)
    assert len(trs) == 1 and trs[0].dst is Mood.CRUISING


def test_feeding_follows_flag():
    m = mk()
    s, tr = m.update(inp(50, feeding=True, mn9=45.0))
    assert tr is not None and tr.dst is Mood.FEEDING and "feeding" in tr.reason
    # exit only after the 1.5 s dwell, immediately when the flag drops
    t, trs = run(m, 50, 29, feeding=False, mn9=0.0)
    assert trs == [] and m.state.state is Mood.FEEDING and m.state.dwell_left_ms == 50
    t, trs = run(m, t, 1, feeding=False)
    assert len(trs) == 1 and trs[0].dst is Mood.CRUISING and trs[0].reason == "mn9 < 15 Hz for 1.0 s"
    # while the flag holds it stays
    m2 = mk()
    t, trs = run(m2, 0, 200, feeding=True, mn9=40.0)
    assert [tr.dst for tr in trs] == [Mood.FEEDING] and m2.state.state is Mood.FEEDING


def test_anxious_bitter_path():
    m = mk()
    t, trs = run(m, 0, 39, bitter=0.5)                # 1.95 s
    assert trs == []
    t, trs = run(m, t, 1, bitter=0.5)
    assert len(trs) == 1 and trs[0].dst is Mood.ANXIOUS and trs[0].reason == "bitter 0.50 >= 0.40 for 2.0 s"
    # exit: anxiety < 0.20 and bitter < 0.2 held 5 s -> CRUISING
    t, trs = run(m, t, 99, bitter=0.1)                # 4.95 s
    assert trs == [] and m.state.state is Mood.ANXIOUS
    t, trs = run(m, t, 1, bitter=0.1)
    assert len(trs) == 1 and trs[0].dst is Mood.CRUISING and "for 5.0 s" in trs[0].reason
    # anxiety path: 0.40 held 1 s
    m2 = mk()
    t, trs = run(m2, 0, 20, looming=0.8)
    assert len(trs) == 1 and trs[0].dst is Mood.ANXIOUS and trs[0].reason == "anxiety 0.40 >= 0.35 for 1.0 s"


def test_sleep_20s_and_wake():
    quiet = dict(sleep_pressure=0.96, activity=0.2, any_drive_max=0.05, dn_mean=1.0)
    m = mk()
    t, trs = run(m, 0, 399, **quiet)                  # 19.95 s
    assert trs == [] and m.state.state is Mood.CRUISING
    t, trs = run(m, t, 1, **quiet)
    assert len(trs) == 1 and trs[0].dst is Mood.SLEEP and "quiet for 20.0 s" in trs[0].reason
    # each wake condition -> CRUISING (after the 1.5 s dwell)
    for wake_kw, tag in ((dict(any_drive_max=0.5), "any_drive_max"), (dict(poked=True), "poked"),
                         (dict(activity=0.5), "activity")):
        m2 = mk()
        t, _ = run(m2, 0, 400, **quiet)
        assert m2.state.state is Mood.SLEEP
        t, trs = run(m2, t, 29, **quiet)              # dwell not over
        assert trs == []
        kw = dict(quiet)
        kw.update(wake_kw)
        t, trs = run(m2, t, 1, **kw)
        assert len(trs) == 1 and trs[0].dst is Mood.CRUISING and trs[0].reason.startswith("wake: " + tag), trs
    # a jump while asleep goes to ESCAPE (which beats the wake rule)
    m3 = mk()
    t, _ = run(m3, 0, 440, **quiet)
    kw = dict(quiet)
    kw["jumped"] = True
    s, tr = m3.update(inp(t + TICK_MS, **kw))
    assert tr.dst is Mood.ESCAPE
    # any drive breaks the 20 s timer
    m4 = mk()
    t, _ = run(m4, 0, 390, **quiet)
    kw = dict(quiet)
    kw["any_drive_max"] = 0.2
    t, _ = run(m4, t, 1, **kw)
    assert m4.state.timers["sleep_enter"] == 0
    t, trs = run(m4, t, 20, **quiet)
    assert trs == []


def test_min_dwell_1500ms():
    m = mk()
    s, tr = m.update(inp(50, feeding=True))
    assert tr.dst is Mood.FEEDING and s.dwell_left_ms == MIN_DWELL_MS
    # PANIC condition holds (gf burst) but the dwell blocks it for 1.5 s
    t, trs = run(m, 50, 29, feeding=True, gf_spikes_10s=5)
    assert trs == [] and m.state.dwell_left_ms == 50
    t, trs = run(m, t, 1, feeding=True, gf_spikes_10s=5)
    assert len(trs) == 1 and trs[0].dst is Mood.PANIC and m.state.dwell_left_ms == MIN_DWELL_MS
    # the wire dwell counts down to 0 and stays there
    t, trs = run(m, t, 40, feeding=True)
    assert m.state.dwell_left_ms == 0


def test_priority_order():
    assert MOOD_PRIORITY == (Mood.ESCAPE, Mood.PANIC, Mood.COURTSHIP, Mood.EUPHORIA, Mood.FEEDING, Mood.ANXIOUS,
                             Mood.SLEEP, Mood.CRUISING)
    # all enter conditions satisfied at once: the machine climbs the priority list one transition per tick
    m = mk()
    every = dict(jumped=True, gf_spikes_10s=3, forced="COURTSHIP", feeding=True, **EUPH_KW)
    s, tr = m.update(inp(50, **every))
    assert tr.dst is Mood.ESCAPE
    m2 = mk()
    every.pop("jumped")
    s, tr = m2.update(inp(50, **every))
    assert tr.dst is Mood.PANIC
    m3 = mk()
    every.pop("gf_spikes_10s")
    s, tr = m3.update(inp(50, **every))
    assert tr.dst is Mood.COURTSHIP
    m4 = mk()
    every.pop("forced")
    t, trs = run(m4, 0, 60, **every)                  # euphoria needs 3 s hold; FEEDING wins first
    assert [tr.dst for tr in trs] == [Mood.FEEDING, Mood.EUPHORIA]
    # at most one transition per tick, a state never re-enters itself
    m5 = mk()
    t, trs = run(m5, 0, 200, feeding=True)
    assert len(trs) == 1


def test_timers_reset_when_condition_breaks():
    m, t = anxious_machine()
    t, _ = run(m, t, 30, **PANIC_KW)                  # 1.5 s of anxiety 0.80
    assert m.state.timers["panic_enter"] == 1500
    t, _ = run(m, t, 1, looming=0.8)                  # one tick back at 0.40
    assert m.state.timers["panic_enter"] == 0
    t, trs = run(m, t, 39, **PANIC_KW)                # another 1.95 s: still not 2 s continuous
    assert trs == [] and m.state.state is Mood.ANXIOUS
    t, trs = run(m, t, 1, **PANIC_KW)
    assert len(trs) == 1 and trs[0].dst is Mood.PANIC
    assert set(m.state.timers) == set(TIMER_KEYS)
    # the same for an entry timer that never fired: euphoria 2.9 s, a dip, 2.9 s -> no EUPHORIA
    m2 = mk()
    t, trs = run(m2, 0, 58, **EUPH_KW)
    t, trs2 = run(m2, t, 1)
    t, trs3 = run(m2, t, 58, **EUPH_KW)
    assert all(tr.dst is not Mood.EUPHORIA for tr in trs + trs2 + trs3)
    assert m2.state.timers["euphoria_enter"] == 2900


def test_confirmed_once_after_3s():
    m = mk()
    m.update(inp(50, feeding=True))
    assert m.confirmed() is None
    t, _ = run(m, 50, 59, feeding=True)               # 2.95 s held
    assert m.confirmed() is None
    t, _ = run(m, t, 1, feeding=True)                 # 3.0 s
    c = m.confirmed()
    assert isinstance(c, Transition) and c.dst is Mood.FEEDING
    assert m.confirmed() is None                      # exactly once
    t, _ = run(m, t, 100, feeding=True)
    assert m.confirmed() is None
    # ESCAPE never confirms
    m2 = mk()
    m2.update(inp(50, jumped=True))
    t, _ = run(m2, 50, 29)
    assert m2.confirmed() is None
    # a transition that does not hold 3 s is dropped (flicker never tweets)
    m3 = mk()
    m3.update(inp(50, feeding=True))
    t, _ = run(m3, 50, 30, feeding=False)             # exits to CRUISING at 1.5 s
    assert m3.state.state is Mood.CRUISING
    t, _ = run(m3, t, 70)
    c = m3.confirmed()
    assert c is not None and c.dst is Mood.CRUISING   # the CRUISING entry itself confirms after 3 s
    assert m3.confirmed() is None
    # custom hold
    m4 = mk()
    m4.update(inp(50, gf_spikes_10s=3))
    t, _ = run(m4, 50, 10, gf_spikes_10s=3)
    assert m4.confirmed(hold_ms=500).dst is Mood.PANIC
    assert CONFIRM_HOLD_MS == 3000


def test_reason_strings():
    m = mk()
    t, trs = run(m, 0, 60, **EUPH_KW)
    assert trs[0].reason == "euphoria 0.80 >= 0.70 for 3.0 s (anxiety 0.00 < 0.35)"
    m2 = mk()
    m2.update(inp(50, feeding=True))
    t, trs = run(m2, 50, 30)
    assert trs[0].reason == "mn9 < 15 Hz for 1.0 s"
    m3 = mk()
    s, tr = m3.update(inp(50, jumped=True))
    assert tr.reason == "jumped (GF spike)"
    for tr in trs:
        assert isinstance(tr.t_ms, int) and isinstance(tr.src, Mood) and isinstance(tr.dst, Mood)


# --------------------------------------------------------------------------- regressions
def test_clock_uses_wall_tick():
    """SPEC f preamble: "``dt`` is the wall tick in seconds (``tick_ms/1000``, 0.05) for market features **and
    mood timers**". Regression: the timers, dwell, the confirm hold and the score EMA used to advance by the
    *brain* ms between updates (``x.t_ms`` deltas), so every f.6 duration stretched with ``sim.speed`` (the
    1000 ms ANXIOUS hold took 4 wall seconds at speed 0.25, the 20 s SLEEP hold 80 s) and the ``tick_s``
    constructor argument was dead. One ``update()`` is now worth exactly one tick for every timer, whatever the
    brain clock does. (``since_ms`` is the exception - it is the brain ms in state, see
    ``test_since_ms_is_brain_ms_confirm_stays_wall``.)"""
    for speed in (1.0, 0.85, 0.5, 0.25):
        m = mk()
        t = 0.0
        n = None
        for k in range(1, 400):
            t += TICK_MS * speed                                     # brain time at this sim.speed
            _, tr = m.update(inp(int(round(t)), looming=0.8))        # anxiety 0.40 >= 0.35 held 1000 ms
            if tr is not None and n is None:
                n, dst = k, tr.dst
        assert (n, dst) == (20, Mood.ANXIOUS), speed                 # always 20 ticks = 1000 wall ms
    # the transition is stamped with BRAIN time (x.t_ms), which is what the wire and the session log need
    m2 = mk()
    t, trs = run(m2, 0, 20, looming=0.8)
    assert [tr.t_ms for tr in trs] == [1000] and t == 1000
    m3 = mk()
    tr_last = None
    for k in range(1, 21):
        _, tr = m3.update(inp(10_000 + 13 * k, looming=0.8))
        tr_last = tr or tr_last
    assert tr_last.t_ms == 10_260 and m3.state.since_ms == 0          # 20 ticks regardless of the brain clock
    # tick_s is live: a 25 ms tick needs twice the updates, a 100 ms tick half of them
    for tick_s, n_expect in ((0.025, 40), (0.05, 20), (0.1, 10)):
        m4 = MoodMachine(tick_s=tick_s, tau_score_s=1e-6)
        n = None
        for k in range(1, 100):
            _, tr = m4.update(inp(k * 50, looming=0.8))
            if tr is not None and n is None:
                n = k
        assert n == n_expect, tick_s
    # a stalled, jumping or backwards brain clock credits exactly one tick to every TIMER, never more (no
    # instant SLEEP); since_ms, by contrast, is the brain ms in state (SPEC d.2), so it follows x.t_ms.
    m5 = mk()
    s, tr = m5.update(inp(600_000, sleep_pressure=0.96, activity=0.2, any_drive_max=0.05, dn_mean=1.0))
    assert tr is None and s.timers["sleep_enter"] == TICK_MS       # timer: one wall tick, not the brain jump
    assert s.since_ms == 600_000                                    # since_ms: brain ms in state (t_ms - entry 0)
    m5.update(inp(0, sleep_pressure=0.96, activity=0.2, any_drive_max=0.05, dn_mean=1.0))
    assert m5.state.timers["sleep_enter"] == 2 * TICK_MS           # ... another wall tick, whatever x.t_ms does
    m6 = mk()
    for _ in range(3):
        m6.update(inp(500, looming=0.8))
    assert m6.state.timers["anxious_enter"] == 150                  # 3 wall ticks credited to the timer
    assert m6.state.since_ms == 500                                 # brain frozen at 500 -> 500 brain ms in state


def test_since_ms_is_brain_ms_confirm_stays_wall():
    """SPEC d.2: ``tick.mood.since_ms`` is "brain ms in the current state"; SPEC f preamble: the timers, dwell
    and the 3 s confirm hold advance by the *wall* tick. Regression: ``since_ms`` used to be credited one wall
    tick per ``update()``, so at ``sim.speed < 1`` the wire (and the c.20 ``since_s``) over-reported the brain
    time in state (500 wall ms where only 250 brain ms had passed at speed 0.5). The fix stamps ``since_ms`` from
    ``x.t_ms - entry_t_ms`` while keeping the confirm / COURTSHIP gate wall-paced, reconciling both clauses.

    Here ``sim.speed`` 0.5 -> 25 brain ms per 50 ms wall tick; brain time and wall time diverge by 2x."""
    speed = 0.5
    # (a) a plain run stays in CRUISING: since_ms tracks the BRAIN clock, not the wall counter
    m = mk()
    t = 0.0
    s = None
    for _ in range(10):
        t += TICK_MS * speed
        s, _ = m.update(inp(int(round(t)), activity=0.5))
    assert s.state is Mood.CRUISING
    assert s.since_ms == 250                                        # 10 ticks * 25 brain ms (would be 500 wall ms)
    # (b) the 3 s confirm still fires after 3000 WALL ms, and since_ms then reads the BRAIN ms in state
    m2 = mk()
    t = 0.0
    confirmed_at = since_at_confirm = None
    s2 = None
    for k in range(1, 120):
        t += TICK_MS * speed
        s2, _ = m2.update(inp(int(round(t)), feeding=True))        # FEEDING enters on tick 1, then holds
        c = m2.confirmed()
        if c is not None and confirmed_at is None:
            confirmed_at, since_at_confirm = k, s2.since_ms
            assert c.dst is Mood.FEEDING
    assert s2.state is Mood.FEEDING
    assert confirmed_at == 61                                       # 60 wall ticks after entry = 3000 WALL ms
    assert since_at_confirm == 1500                                 # ... while since_ms reports 1500 BRAIN ms
    # (c) the 8 s COURTSHIP exit gate is wall-paced too: entry on tick 10 (500 wall ms of p1), exit 10 s of wall
    m3 = mk()
    t = 0.0
    entered = exited = None
    for k in range(1, 260):
        t += TICK_MS * speed
        p1 = 20.0 if k <= 10 else 0.0                              # court held 500 ms, then silent
        _, tr = m3.update(inp(int(round(t)), p1=p1))
        if tr is not None and tr.dst is Mood.COURTSHIP:
            entered = k
        if tr is not None and tr.src is Mood.COURTSHIP and tr.dst is Mood.CRUISING:
            exited = k
    assert entered == 10                                           # 500 wall ms of p1 (unchanged by speed)
    assert exited is not None and (exited - entered) * TICK_MS == COURTSHIP_DWELL_MS + 2000   # 10 s WALL ms


def test_sleep_timer_resets_only_when_quiet_breaks():
    """SPEC f.6: "timers accumulate only while the condition holds and reset otherwise" - and the SLEEP row's
    condition is the quiet itself, not "SLEEP was left". An earlier version cleared ``sleep_enter`` on every exit
    from SLEEP (an undeclared [E] extension worth another 20 s of forced wakefulness); now a poke that leaves the
    fly in genuinely quiet conditions lets it doze off again after the 1500 ms dwell, while a poke that actually
    raises a drive breaks the quiet by the normal rule."""
    quiet = dict(sleep_pressure=0.96, activity=0.2, any_drive_max=0.05, dn_mean=1.0)
    m = mk()
    t, trs = run(m, 0, 400, **quiet)
    assert [tr.dst for tr in trs] == [Mood.SLEEP] and t == 20_000
    t, trs = run(m, t, 30, **quiet)
    t, trs = run(m, t, 1, poked=True, **quiet)
    assert [tr.dst for tr in trs] == [Mood.CRUISING]
    assert m.state.timers["sleep_enter"] == 21_550                     # kept: the quiet never broke
    t_wake = t
    t, trs = run(m, t, 29, **quiet)                                    # the 1500 ms dwell blocks the re-entry
    assert trs == [] and m.state.state is Mood.CRUISING
    t, trs = run(m, t, 1, **quiet)
    assert [tr.dst for tr in trs] == [Mood.SLEEP] and t - t_wake == MIN_DWELL_MS
    # a wake that really raises a drive resets the timer by the f.6 rule, and 20 s of quiet are needed again
    t, trs = run(m, t, 30, **quiet)
    t, trs = run(m, t, 1, sleep_pressure=0.96, activity=0.2, any_drive_max=0.9, dn_mean=1.0)
    assert [tr.dst for tr in trs] == [Mood.CRUISING] and m.state.timers["sleep_enter"] == 0
    t, trs = run(m, t, 399, **quiet)
    assert trs == [] and m.state.state is Mood.CRUISING
    t, trs = run(m, t, 1, **quiet)
    assert [tr.dst for tr in trs] == [Mood.SLEEP]


def test_moodstate_slots():
    m = mk()
    assert hasattr(MoodState, "__slots__")
    with pytest.raises(AttributeError):
        m.state.extra = 1  # type: ignore[attr-defined]
    m.state.state = Mood.PANIC                                        # declared fields stay assignable
    assert m.state.to_wire()["state"] == "PANIC"


def test_to_wire_keys():
    m = mk()
    s, _ = m.update(inp(50, sugar=0.3, hunger=0.42))
    w = s.to_wire()
    assert set(w) == WIRE_KEYS
    assert w["state"] == "CRUISING" and w["prev"] == "CRUISING"
    assert isinstance(w["since_ms"], int) and isinstance(w["dwell_left_ms"], int)
    for k in ("euphoria", "anxiety", "arousal", "fear", "hunger", "sleep"):
        assert 0.0 <= w[k] <= 1.0
    assert -1.0 <= w["valence"] <= 1.0
    assert w["hunger"] == 0.42
    assert set(MOOD_COLORS) == {mm.value for mm in Mood}
    assert MOOD_COLORS["EUPHORIA"] == "rainbow"
    assert isinstance(s, MoodState)


def test_escape_burst_never_tweets_but_panic_does():
    """SPEC section 0 / f.6: ESCAPE never confirms (so a jump never tweets), while the ESCAPE_BURST trigger of
    the agent (>= 3 jumps / 60 s, SPEC c.24) is counted outside the mood machine from the decoder's jump events.
    What the machine must guarantee: every jump enters ESCAPE (so the burst is countable), no ESCAPE is ever
    handed to ``confirmed()``, and the PANIC a burst of GF spikes produces DOES confirm (the tweetable path)."""
    m = mk()
    jump_ts = (1_000, 21_000, 41_000)             # 3 jumps inside 60 s
    escapes, confirmed_dsts = [], []
    t = 0
    for _ in range(1200):                          # 60 s at 50 ms
        t += TICK_MS
        _, tr = m.update(inp(t, jumped=t in jump_ts))
        if tr is not None and tr.dst is Mood.ESCAPE:
            escapes.append(tr.t_ms)
        c = m.confirmed()
        if c is not None:
            confirmed_dsts.append(c.dst)
    assert escapes == list(jump_ts)                           # every jump is an ESCAPE entry -> countable burst
    assert Mood.ESCAPE not in confirmed_dsts                  # ... and never a tweet
    assert confirmed_dsts and all(d is Mood.CRUISING for d in confirmed_dsts)
    # the GF burst path (gf_spikes_10s >= 3) lands in PANIC, and PANIC confirms after 3 s of held anxiety
    m2 = mk()
    _, tr = m2.update(inp(50, gf_spikes_10s=3, **PANIC_KW))
    assert tr.dst is Mood.PANIC
    t, _ = run(m2, 50, 59, **PANIC_KW)                        # 2.95 s held
    assert m2.confirmed() is None
    t, _ = run(m2, t, 1, **PANIC_KW)
    c = m2.confirmed()
    assert c is not None and c.dst is Mood.PANIC and m2.confirmed() is None
    # a PANIC that ends exactly when its 3 s exit hold completes never confirms (flicker never tweets)
    m3 = mk()
    m3.update(inp(50, gf_spikes_10s=3))
    t, trs = run(m3, 50, 60)                                  # anxiety 0 -> exit at 3.0 s, same tick as the hold
    assert [tr.dst for tr in trs] == [Mood.ANXIOUS] and m3.confirmed() is None


def test_update_returns_the_live_state_object():
    """Documented in the c.19 docstring (SPEC c.27 serialises the state immediately, which is safe): ``update()``
    hands back the machine's live ``MoodState``, so a consumer that retains it sees later ticks. Pinned so that
    switching to per-tick snapshots is a deliberate contract change, not an accident."""
    m = mk()
    s1, _ = m.update(inp(50, looming=0.8))
    s2, _ = m.update(inp(100, looming=0.8))
    assert s1 is s2 is m.state
    wire = s1.to_wire()                                                # ... a serialised copy does not move
    m.update(inp(150, looming=0.8))
    assert wire["since_ms"] == 100 and s1.since_ms == 150
