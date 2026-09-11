"""Mood machine (SPEC sections c.19 and f.6).

Eight moods with a fixed priority ``ESCAPE > PANIC > COURTSHIP > EUPHORIA > FEEDING > ANXIOUS > SLEEP >
CRUISING``. Scores are exponential moving averages (``tau_score_s``, default 2 s) of sensory / neural
inputs; every "held T" rule of the section f.6 table is a timer that accumulates only while its
condition holds and resets to zero otherwise. Every state is held for a minimum dwell of 1500 ms
before a non-ESCAPE transition (COURTSHIP additionally gates its *exit* rule behind an 8000 ms dwell,
so a threat can still pre-empt courtship after 1.5 s); ESCAPE ignores the dwell. ``confirmed()`` is
the agent trigger: a transition is handed out exactly once after its destination has been held for
3 s, and never for ESCAPE, so flicker never tweets.

Provenance: every threshold, weight and timer in this module is ``[E]`` (engineered by the SPEC
section f.6 mood table for the demo; they are puppeteering constants, not measured biology). The
inputs they read (sugar GRN drive, MN9 proboscis rate, PAM dopamine, LC4/LPLC2 looming, DNp09
freeze, P1/pIP10 courtship) are the biologically motivated pathways of RESEARCH sections 4-5
``[V]``; their mapping to a "mood" word is engineered.

Timer keys (``MoodState.timers``, ms): ``panic_enter`` (anxiety >= 0.70), ``panic_exit``
(anxiety < 0.45), ``court_enter`` (p1 >= 15 or pip10 >= 20), ``court_exit`` (p1 < 5 and pip10 < 5),
``euphoria_enter`` (euphoria >= 0.70), ``euphoria_exit`` (euphoria < 0.40), ``anxious_enter``
(anxiety >= 0.35), ``anxious_bitter`` (bitter >= 0.4), ``anxious_exit`` (anxiety < 0.20 and
bitter < 0.2), ``sleep_enter`` (quiet + sleep pressure). Entry timers accumulate regardless of the
current state (so priorities compare real durations); the exit timers of a state are reset to zero
when it is entered, so an exit "held T" always means "held T *while in the state*" (``court_exit``
only starts counting once the 8000 ms COURTSHIP dwell is over: "dwell 8000 ms, then ... held 2000").
Every other timer follows the section f.6 table literally: it resets only when its own condition
breaks. In particular ``sleep_enter`` is **not** cleared when SLEEP is left - a poke that leaves the
fly in genuinely quiet conditions lets it doze off again after the 1500 ms dwell, and a poke that
actually raises a drive (``any_drive_max``, ``dn_mean``) breaks the quiet condition and resets the
timer by the normal rule.

Clock: the score EMA, every "held T" timer, the 1500 ms dwell and the 3 s ``confirmed()`` hold advance
by exactly one **wall tick** (``tick_s``) per ``update()`` call, as required by the section f preamble
("``dt`` is the wall tick in seconds (``tick_ms/1000``, 0.05) for market features **and mood timers**").
A skipped, stalled, replayed or restarted brain clock can never credit an arbitrary amount of held time
to a timer, because one ``update()`` is always worth one tick. ``since_ms`` is the exception: SPEC d.2
defines it as "brain ms in the current state", so it is stamped as ``max(0, x.t_ms - entry_t_ms)`` -
the brain time elapsed since the state was entered (``entry_t_ms`` = the brain ``t_ms`` of the last
transition, 0 for the initial CRUISING). At ``sim.speed == 1`` the brain and wall clocks coincide;
below it ``since_ms`` reports the brain time (which the d.2 wire and the c.20 summary's ``since_s`` need)
while the confirm hold, the 8 s COURTSHIP gate and every timer keep wall-clock pace, so behaviour timing
is unaffected by ``sim.speed`` (SPEC c.27). ``x.t_ms`` is also stamped onto ``Transition.t_ms``.

Only the standard library is imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "Mood", "MOOD_PRIORITY", "MOOD_COLORS", "MoodInputs", "MoodState", "Transition", "MoodMachine",
    "TIMER_KEYS", "MIN_DWELL_MS", "ESCAPE_DWELL_MS", "COURTSHIP_DWELL_MS", "CONFIRM_HOLD_MS",
]


class Mood(str, Enum):
    """The eight mood states (SPEC c.19)."""

    SLEEP = "SLEEP"
    CRUISING = "CRUISING"
    FEEDING = "FEEDING"
    EUPHORIA = "EUPHORIA"
    ANXIOUS = "ANXIOUS"
    PANIC = "PANIC"
    ESCAPE = "ESCAPE"
    COURTSHIP = "COURTSHIP"


#: priority order, highest first (SPEC section 0 "Mood" decision + c.19).
MOOD_PRIORITY: tuple[Mood, ...] = (
    Mood.ESCAPE, Mood.PANIC, Mood.COURTSHIP, Mood.EUPHORIA, Mood.FEEDING, Mood.ANXIOUS, Mood.SLEEP, Mood.CRUISING,
)

#: UI colours (SPEC c.19); ``"rainbow"`` is a style name the frontend animates.
MOOD_COLORS: dict[str, str] = {
    "SLEEP": "#6b6b6b", "CRUISING": "#000000", "FEEDING": "#ff8c00", "EUPHORIA": "rainbow",
    "ANXIOUS": "#ffd700", "PANIC": "#ff0000", "ESCAPE": "#ff3300", "COURTSHIP": "#ff69b4",
}

# --------------------------------------------------------------------------- constants (all [E], SPEC f.6)
TAU_SCORE_S: float = 2.0            #: [E] score EMA time constant (s)
MIN_DWELL_MS: int = 1500            #: [E] minimum dwell in every state before a non-ESCAPE transition
ESCAPE_DWELL_MS: int = 1500         #: [E] ESCAPE lasts exactly this long, then exits by anxiety
COURTSHIP_DWELL_MS: int = 8000      #: [E] COURTSHIP dwell before its exit rule may fire (higher-priority entries still use MIN_DWELL_MS)
CONFIRM_HOLD_MS: int = 3000         #: [E] hold before ``confirmed()`` hands a transition to the agent

E_W_SUGAR: float = 0.45             #: [E] euphoria weight of the sugar drive
E_W_MN9: float = 0.35               #: [E] euphoria weight of sat(mn9 / 60 Hz)
E_MN9_HZ: float = 60.0              #: [E] MN9 rate that saturates the euphoria term
E_W_PAM: float = 0.20               #: [E] euphoria weight of sat(pam / 40 Hz)
E_PAM_HZ: float = 40.0              #: [E] PAM rate that saturates the euphoria term
A_W_LOOM: float = 0.50              #: [E] anxiety weight of the looming drive
A_W_DNP09: float = 0.30             #: [E] anxiety weight of sat(dnp09 / 40 Hz)
A_DNP09_HZ: float = 40.0            #: [E] DNp09 rate that saturates the anxiety term
A_W_LCLOOM: float = 0.20            #: [E] anxiety weight of sat(lc_loom / 60 Hz)
A_LCLOOM_HZ: float = 60.0           #: [E] LC4/LPLC2 rate that saturates the anxiety term
AROUSAL_W_DN: float = 0.5           #: [E] arousal weight of sat(dn_mean / 5 Hz)
AROUSAL_DN_HZ: float = 5.0          #: [E] descending-neuron mean rate that saturates arousal
AROUSAL_W_DRIVE: float = 0.5        #: [E] arousal weight of any_drive_max
VALENCE_MBON_W: float = 0.2         #: [E] valence weight of the MBON approach/avoid balance
VALENCE_MBON_FLOOR_HZ: float = 5.0  #: [E] softening constant in the MBON balance denominator

PANIC_ANXIETY: float = 0.70         #: [E] anxiety >= this held PANIC_HOLD_MS enters PANIC
PANIC_HOLD_MS: int = 2000           #: [E]
PANIC_GF_BURST: int = 3             #: [E] gf_spikes_10s >= this enters PANIC at once
PANIC_EXIT_ANXIETY: float = 0.45    #: [E] anxiety < this held PANIC_EXIT_HOLD_MS -> ANXIOUS
PANIC_EXIT_HOLD_MS: int = 3000      #: [E]
ESCAPE_TO_PANIC_ANXIETY: float = 0.45   #: [E] ESCAPE exit -> PANIC when anxiety >= this
ESCAPE_TO_ANXIOUS_ANXIETY: float = 0.20 #: [E] ESCAPE exit -> ANXIOUS when anxiety >= this (else CRUISING)
COURT_P1_HZ: float = 15.0           #: [E] P1 rate that (held COURT_HOLD_MS) enters COURTSHIP
COURT_PIP10_HZ: float = 20.0        #: [E] pIP10 rate that (held COURT_HOLD_MS) enters COURTSHIP
COURT_HOLD_MS: int = 500            #: [E]
COURT_EXIT_P1_HZ: float = 5.0       #: [E] exit needs p1 < this and pip10 < COURT_EXIT_PIP10_HZ held COURT_EXIT_HOLD_MS
COURT_EXIT_PIP10_HZ: float = 5.0    #: [E]
COURT_EXIT_HOLD_MS: int = 2000      #: [E]
EUPHORIA_ENTER: float = 0.70        #: [E] euphoria >= this held EUPHORIA_HOLD_MS (with anxiety < EUPHORIA_MAX_ANXIETY)
EUPHORIA_HOLD_MS: int = 3000        #: [E]
EUPHORIA_MAX_ANXIETY: float = 0.35  #: [E]
EUPHORIA_EXIT: float = 0.40         #: [E] euphoria < this held EUPHORIA_EXIT_HOLD_MS -> FEEDING/CRUISING
EUPHORIA_EXIT_HOLD_MS: int = 2000   #: [E]
ANXIOUS_ENTER: float = 0.35         #: [E] anxiety >= this held ANXIOUS_HOLD_MS
ANXIOUS_HOLD_MS: int = 1000         #: [E]
ANXIOUS_BITTER: float = 0.4         #: [E] bitter >= this held ANXIOUS_BITTER_HOLD_MS
ANXIOUS_BITTER_HOLD_MS: int = 2000  #: [E]
ANXIOUS_EXIT_ANXIETY: float = 0.20  #: [E] exit needs anxiety < this and bitter < ANXIOUS_EXIT_BITTER held ANXIOUS_EXIT_HOLD_MS
ANXIOUS_EXIT_BITTER: float = 0.2    #: [E]
ANXIOUS_EXIT_HOLD_MS: int = 5000    #: [E]
SLEEP_PRESSURE: float = 0.95        #: [E] sleep_pressure >= this ...
SLEEP_MAX_ACTIVITY: float = 0.35    #: [E] ... and activity < this ...
SLEEP_MAX_DRIVE: float = 0.10       #: [E] ... and any_drive_max < this ...
SLEEP_MAX_DN_MEAN: float = 2.0      #: [E] ... and dn_mean < this Hz, all held SLEEP_HOLD_MS
SLEEP_HOLD_MS: int = 20000          #: [E]
WAKE_DRIVE: float = 0.10            #: [E] any_drive_max > this wakes
WAKE_ACTIVITY: float = 0.35         #: [E] activity >= this wakes

#: every timer key of ``MoodState.timers`` (ms).
TIMER_KEYS: tuple[str, ...] = (
    "panic_enter", "panic_exit", "court_enter", "court_exit", "euphoria_enter", "euphoria_exit",
    "anxious_enter", "anxious_bitter", "anxious_exit", "sleep_enter",
)
#: exit timers reset to zero when their state is entered (see module docstring).
_EXIT_TIMERS_OF: dict[Mood, tuple[str, ...]] = {
    Mood.PANIC: ("panic_exit",),
    Mood.COURTSHIP: ("court_exit",),
    Mood.EUPHORIA: ("euphoria_exit",),
    Mood.ANXIOUS: ("anxious_exit",),
}


def _sat(x: float) -> float:
    """clip to [0, 1]; NaN -> 0."""
    if x != x:
        return 0.0
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))


def _clip(x: float, lo: float, hi: float) -> float:
    if x != x:
        return lo if lo > 0 else 0.0
    return lo if x < lo else (hi if x > hi else float(x))


def _f(x: object, default: float = 0.0) -> float:
    """float() with NaN/inf/None -> default."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


# --------------------------------------------------------------------------- dataclasses
@dataclass(frozen=True)
class MoodInputs:
    """One tick of inputs (SPEC c.19). Plain floats/ints/bools so no cross-workstream import is needed.

    ``hunger`` is not in the SPEC c.19 field list but section f.6 reads ``x.hunger`` (passthrough
    from ``Features`` via the loop); it is therefore an optional trailing field (default 0.0).
    """

    t_ms: int
    sugar: float
    looming: float
    bitter: float
    activity: float
    sleep_pressure: float
    mn9: float
    dnp09: float
    lc_loom: float
    pam: float
    ppl1: float
    mbon_approach: float
    mbon_avoid: float
    p1: float
    pip10: float
    dn_mean: float
    feeding: bool
    jumped: bool
    gf_spikes_10s: int
    any_drive_max: float
    poked: bool
    forced: str | None
    hunger: float = 0.0


@dataclass(slots=True)
class MoodState:
    """Live state of the machine; ``to_wire()`` is ``tick.mood`` of SPEC d.2. ``dwell_left_ms`` is the general
    1500 ms dwell of every state (COURTSHIP's 8000 ms exit gate is a separate rule, not this countdown)."""

    state: Mood
    prev: Mood
    since_ms: int
    euphoria: float
    anxiety: float
    arousal: float
    valence: float
    fear: float
    hunger: float
    sleep: float
    dwell_left_ms: int
    timers: dict[str, int] = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "state": self.state.value,
            "prev": self.prev.value,
            "since_ms": int(self.since_ms),
            "euphoria": round(float(self.euphoria), 3),
            "anxiety": round(float(self.anxiety), 3),
            "arousal": round(float(self.arousal), 3),
            "valence": round(float(self.valence), 3),
            "fear": round(float(self.fear), 3),
            "hunger": round(float(self.hunger), 3),
            "sleep": round(float(self.sleep), 3),
            "dwell_left_ms": int(self.dwell_left_ms),
        }


@dataclass(frozen=True)
class Transition:
    t_ms: int
    src: Mood
    dst: Mood
    reason: str


# --------------------------------------------------------------------------- machine
class MoodMachine:
    """Section f.6 state machine. ``update()`` performs at most one transition per tick."""

    def __init__(self, tick_s: float, tau_score_s: float = TAU_SCORE_S) -> None:
        self.tick_s = max(1e-3, _f(tick_s, 0.05))
        #: ms credited to every timer / EMA / dwell / confirm hold per ``update()`` (SPEC f preamble: the wall tick).
        self.tick_ms = max(1, int(round(self.tick_s * 1000.0)))
        self.tau_score_s = max(1e-9, _f(tau_score_s, TAU_SCORE_S))
        self.state = MoodState(
            state=Mood.CRUISING, prev=Mood.CRUISING, since_ms=0,
            euphoria=0.0, anxiety=0.0, arousal=0.0, valence=0.0, fear=0.0, hunger=0.0, sleep=0.0,
            dwell_left_ms=0, timers={k: 0 for k in TIMER_KEYS},
        )
        #: brain ``t_ms`` at which the current state was entered (0 = the initial CRUISING at sim start);
        #: ``state.since_ms`` (the d.2 wire field) is ``max(0, x.t_ms - self._entry_t_ms)``.
        self._entry_t_ms: int = 0
        #: wall ms in the current state (paces the 3 s confirm and the 8 s COURTSHIP gate, SPEC f preamble).
        self._since_wall_ms: int = 0
        self.last_transition: Transition | None = None
        self._pending: Transition | None = None
        self.n_transitions: int = 0

    # ----------------------------------------------------------------- helpers
    def _ema(self, prev: float, raw: float, dt_ms: int) -> float:
        a = 1.0 - math.exp(-(dt_ms / 1000.0) / self.tau_score_s)
        return prev + (raw - prev) * a

    def _held(self, key: str, cond: bool, dt_ms: int) -> None:
        t = self.state.timers
        t[key] = t.get(key, 0) + dt_ms if cond else 0

    def _enter(self, t_ms: int, dst: Mood, reason: str) -> Transition:
        s = self.state
        tr = Transition(t_ms=int(t_ms), src=s.state, dst=dst, reason=reason)
        s.prev = s.state
        s.state = dst
        s.since_ms = 0
        self._entry_t_ms = int(t_ms)   # brain time of this entry: since_ms = x.t_ms - this (SPEC d.2)
        self._since_wall_ms = 0        # wall clock in the new state restarts (confirm / COURTSHIP gate)
        s.dwell_left_ms = ESCAPE_DWELL_MS if dst is Mood.ESCAPE else MIN_DWELL_MS
        for key in _EXIT_TIMERS_OF.get(dst, ()):
            s.timers[key] = 0
        self.last_transition = tr
        self._pending = None if dst is Mood.ESCAPE else tr
        self.n_transitions += 1
        return tr

    # ----------------------------------------------------------------- scores
    def _update_scores(self, x: MoodInputs, dt_ms: int) -> None:
        s = self.state
        sugar = _sat(_f(x.sugar))
        looming = _sat(_f(x.looming))
        e_raw = E_W_SUGAR * sugar + E_W_MN9 * _sat(_f(x.mn9) / E_MN9_HZ) + E_W_PAM * _sat(_f(x.pam) / E_PAM_HZ)
        a_raw = (A_W_LOOM * looming + A_W_DNP09 * _sat(_f(x.dnp09) / A_DNP09_HZ)
                 + A_W_LCLOOM * _sat(_f(x.lc_loom) / A_LCLOOM_HZ))
        ar_raw = AROUSAL_W_DN * _sat(_f(x.dn_mean) / AROUSAL_DN_HZ) + AROUSAL_W_DRIVE * _sat(_f(x.any_drive_max))
        s.euphoria = self._ema(s.euphoria, e_raw, dt_ms)
        s.anxiety = self._ema(s.anxiety, a_raw, dt_ms)
        s.fear = s.anxiety
        s.arousal = self._ema(s.arousal, ar_raw, dt_ms)
        appr, avoid = max(0.0, _f(x.mbon_approach)), max(0.0, _f(x.mbon_avoid))
        mbon = (appr - avoid) / (appr + avoid + VALENCE_MBON_FLOOR_HZ)
        s.valence = _clip(s.euphoria - s.anxiety + VALENCE_MBON_W * mbon, -1.0, 1.0)
        s.hunger = _sat(_f(getattr(x, "hunger", 0.0)))
        s.sleep = _sat(_f(x.sleep_pressure))

    def _update_timers(self, x: MoodInputs, dt_ms: int) -> None:
        s = self.state
        p1, pip10, bitter = _f(x.p1), _f(x.pip10), _sat(_f(x.bitter))
        self._held("panic_enter", s.anxiety >= PANIC_ANXIETY, dt_ms)
        self._held("panic_exit", s.anxiety < PANIC_EXIT_ANXIETY, dt_ms)
        self._held("court_enter", p1 >= COURT_P1_HZ or pip10 >= COURT_PIP10_HZ, dt_ms)
        # "dwell 8000 ms, THEN held 2000": the hold only counts ticks that start after the dwell is complete.
        # The 8 s gate is a mood timer, so it uses the wall clock (SPEC f preamble), not the brain since_ms.
        court_dwell_over = s.state is Mood.COURTSHIP and self._since_wall_ms > COURTSHIP_DWELL_MS
        self._held("court_exit", court_dwell_over and p1 < COURT_EXIT_P1_HZ and pip10 < COURT_EXIT_PIP10_HZ, dt_ms)
        self._held("euphoria_enter", s.euphoria >= EUPHORIA_ENTER, dt_ms)
        self._held("euphoria_exit", s.euphoria < EUPHORIA_EXIT, dt_ms)
        self._held("anxious_enter", s.anxiety >= ANXIOUS_ENTER, dt_ms)
        self._held("anxious_bitter", bitter >= ANXIOUS_BITTER, dt_ms)
        self._held("anxious_exit", s.anxiety < ANXIOUS_EXIT_ANXIETY and bitter < ANXIOUS_EXIT_BITTER, dt_ms)
        quiet = (_f(x.sleep_pressure) >= SLEEP_PRESSURE and _f(x.activity) < SLEEP_MAX_ACTIVITY
                 and _f(x.any_drive_max) < SLEEP_MAX_DRIVE and _f(x.dn_mean) < SLEEP_MAX_DN_MEAN)
        self._held("sleep_enter", quiet, dt_ms)

    # ----------------------------------------------------------------- rules
    def _enter_rule(self, target: Mood, x: MoodInputs) -> str | None:
        """Reason string when ``target``'s enter condition holds now, else None."""
        s, t = self.state, self.state.timers
        if target is Mood.ESCAPE:
            return "jumped (GF spike)" if x.jumped else None
        if target is Mood.PANIC:
            if t["panic_enter"] >= PANIC_HOLD_MS:
                return f"anxiety {s.anxiety:.2f} >= {PANIC_ANXIETY:.2f} for {PANIC_HOLD_MS / 1000:.1f} s"
            if int(_f(x.gf_spikes_10s)) >= PANIC_GF_BURST:
                return f"gf_spikes_10s {int(_f(x.gf_spikes_10s))} >= {PANIC_GF_BURST}"
            return None
        if target is Mood.COURTSHIP:
            if x.forced == "COURTSHIP":
                return "forced COURTSHIP"
            if t["court_enter"] >= COURT_HOLD_MS:
                return (f"p1 {_f(x.p1):.1f} Hz >= {COURT_P1_HZ:.0f} or pip10 {_f(x.pip10):.1f} Hz >= "
                        f"{COURT_PIP10_HZ:.0f} for {COURT_HOLD_MS / 1000:.1f} s")
            return None
        if target is Mood.EUPHORIA:
            if t["euphoria_enter"] >= EUPHORIA_HOLD_MS and s.anxiety < EUPHORIA_MAX_ANXIETY:
                return (f"euphoria {s.euphoria:.2f} >= {EUPHORIA_ENTER:.2f} for {EUPHORIA_HOLD_MS / 1000:.1f} s "
                        f"(anxiety {s.anxiety:.2f} < {EUPHORIA_MAX_ANXIETY:.2f})")
            return None
        if target is Mood.FEEDING:
            return f"feeding (mn9 {_f(x.mn9):.0f} Hz, proboscis extended)" if x.feeding else None
        if target is Mood.ANXIOUS:
            if t["anxious_enter"] >= ANXIOUS_HOLD_MS:
                return f"anxiety {s.anxiety:.2f} >= {ANXIOUS_ENTER:.2f} for {ANXIOUS_HOLD_MS / 1000:.1f} s"
            if t["anxious_bitter"] >= ANXIOUS_BITTER_HOLD_MS:
                return f"bitter {_sat(_f(x.bitter)):.2f} >= {ANXIOUS_BITTER:.2f} for {ANXIOUS_BITTER_HOLD_MS / 1000:.1f} s"
            return None
        if target is Mood.SLEEP:
            if t["sleep_enter"] >= SLEEP_HOLD_MS:
                return (f"sleep_pressure {_f(x.sleep_pressure):.2f} >= {SLEEP_PRESSURE:.2f}, quiet for "
                        f"{SLEEP_HOLD_MS / 1000:.1f} s")
            return None
        return None  # CRUISING is the default state, never "entered" by rule

    def _exit_rule(self, x: MoodInputs) -> tuple[Mood, str] | None:
        """(target, reason) when the current state's exit condition holds now, else None."""
        s, t = self.state, self.state.timers
        cur = s.state
        if cur is Mood.ESCAPE:
            if s.anxiety >= ESCAPE_TO_PANIC_ANXIETY:
                return Mood.PANIC, (f"escape over after {ESCAPE_DWELL_MS / 1000:.1f} s; anxiety {s.anxiety:.2f} >= "
                                    f"{ESCAPE_TO_PANIC_ANXIETY:.2f}")
            if s.anxiety >= ESCAPE_TO_ANXIOUS_ANXIETY:
                return Mood.ANXIOUS, (f"escape over after {ESCAPE_DWELL_MS / 1000:.1f} s; anxiety {s.anxiety:.2f} >= "
                                      f"{ESCAPE_TO_ANXIOUS_ANXIETY:.2f}")
            return Mood.CRUISING, (f"escape over after {ESCAPE_DWELL_MS / 1000:.1f} s; anxiety {s.anxiety:.2f} < "
                                   f"{ESCAPE_TO_ANXIOUS_ANXIETY:.2f}")
        if cur is Mood.PANIC:
            if t["panic_exit"] >= PANIC_EXIT_HOLD_MS:
                return Mood.ANXIOUS, f"anxiety {s.anxiety:.2f} < {PANIC_EXIT_ANXIETY:.2f} for {PANIC_EXIT_HOLD_MS / 1000:.1f} s"
            return None
        if cur is Mood.COURTSHIP:
            if t["court_exit"] >= COURT_EXIT_HOLD_MS:
                return Mood.CRUISING, (f"p1 {_f(x.p1):.1f} < {COURT_EXIT_P1_HZ:.0f} and pip10 {_f(x.pip10):.1f} < "
                                       f"{COURT_EXIT_PIP10_HZ:.0f} Hz for {COURT_EXIT_HOLD_MS / 1000:.1f} s after "
                                       f"{COURTSHIP_DWELL_MS / 1000:.1f} s dwell")
            return None
        if cur is Mood.EUPHORIA:
            if t["euphoria_exit"] >= EUPHORIA_EXIT_HOLD_MS:
                dst = Mood.FEEDING if x.feeding else Mood.CRUISING
                return dst, f"euphoria {s.euphoria:.2f} < {EUPHORIA_EXIT:.2f} for {EUPHORIA_EXIT_HOLD_MS / 1000:.1f} s"
            return None
        if cur is Mood.FEEDING:
            if not x.feeding:
                # the decoder's feed_stop rule (SPEC f.3): mn9 < 15 Hz held 1000 ms
                return Mood.CRUISING, "mn9 < 15 Hz for 1.0 s"
            return None
        if cur is Mood.ANXIOUS:
            if t["anxious_exit"] >= ANXIOUS_EXIT_HOLD_MS:
                return Mood.CRUISING, (f"anxiety {s.anxiety:.2f} < {ANXIOUS_EXIT_ANXIETY:.2f} and bitter "
                                       f"{_sat(_f(x.bitter)):.2f} < {ANXIOUS_EXIT_BITTER:.2f} for "
                                       f"{ANXIOUS_EXIT_HOLD_MS / 1000:.1f} s")
            return None
        if cur is Mood.SLEEP:
            drive = _f(x.any_drive_max)
            if drive > WAKE_DRIVE:
                return Mood.CRUISING, f"wake: any_drive_max {drive:.2f} > {WAKE_DRIVE:.2f}"
            if x.poked:
                return Mood.CRUISING, "wake: poked"
            if x.jumped:
                # kept for fidelity with the f.6 SLEEP row; step (2) of update() normally routes a jump to
                # ESCAPE before any exit rule runs, so this branch is only reachable if that order changes.
                return Mood.CRUISING, "wake: jumped"
            if _f(x.activity) >= WAKE_ACTIVITY:
                return Mood.CRUISING, f"wake: activity {_f(x.activity):.2f} >= {WAKE_ACTIVITY:.2f}"
            return None
        return None  # CRUISING has no exit rule

    # ----------------------------------------------------------------- public
    def update(self, x: MoodInputs) -> tuple[MoodState, Transition | None]:
        """Section f.6. Exactly one transition per tick at most.

        Every duration advances by exactly one wall tick (``self.tick_ms``) per call; ``x.t_ms`` is only
        stamped onto ``Transition.t_ms`` (see the module docstring).

        The returned ``MoodState`` is the machine's **live** object, not a snapshot: it keeps changing on
        later ticks. SPEC c.27 serialises it immediately (``to_wire()``), which is safe; a consumer that
        wants to retain it (history, agent summary, a test) must copy what it needs.
        """
        s = self.state
        t_ms = int(_f(x.t_ms))     # brain time: stamped on Transition.t_ms and the d.2 wire since_ms
        dt_ms = self.tick_ms
        # (1) clocks, scores, timers. since_ms is the *brain* ms in state (SPEC d.2); the wall counter, the
        # dwell and every timer advance by one wall tick (SPEC f preamble), so timing is unaffected by sim.speed.
        s.since_ms = max(0, t_ms - self._entry_t_ms)
        self._since_wall_ms += dt_ms
        s.dwell_left_ms = max(0, s.dwell_left_ms - dt_ms)
        self._update_scores(x, dt_ms)
        self._update_timers(x, dt_ms)
        # (2) ESCAPE preempts everything, ignores dwell
        if x.jumped:
            if s.state is Mood.ESCAPE:
                s.dwell_left_ms = ESCAPE_DWELL_MS   # a second jump extends the escape, no re-entry
                return s, None
            return s, self._enter(t_ms, Mood.ESCAPE, "jumped (GF spike)")
        # (3) dwell
        if s.dwell_left_ms > 0:
            return s, None
        # (4) higher-priority entries, first hit wins
        cur_rank = MOOD_PRIORITY.index(s.state)
        for target in MOOD_PRIORITY[:cur_rank]:
            reason = self._enter_rule(target, x)
            if reason is not None:
                return s, self._enter(t_ms, target, reason)
        # (5) exit of the current state
        ex = self._exit_rule(x)
        if ex is not None:
            dst, reason = ex
            return s, self._enter(t_ms, dst, reason)
        return s, None

    def confirmed(self, hold_ms: int = CONFIRM_HOLD_MS) -> Transition | None:
        """The last transition once its destination has been held for ``hold_ms`` (returned exactly once;
        never for ESCAPE) - the agent trigger of SPEC c.24."""
        p = self._pending
        if p is None:
            return None
        if self.state.state is not p.dst:
            self._pending = None
            return None
        # the confirm hold is a mood timer -> wall clock (SPEC f preamble), not the brain since_ms, so a
        # transition confirms after 3 s of wall time regardless of sim.speed.
        if self._since_wall_ms >= hold_ms:
            self._pending = None
            return p
        return None

    @property
    def mood(self) -> str:
        return self.state.state.value
