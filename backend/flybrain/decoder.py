"""Motor decoder and fly body (SPEC sections c.18, f.3, f.4, f.5).

Pipeline per wall tick (SPEC c.27 step 6-7)::

    r   = Readouts.from_stats(stats)                       # population rates -> named readouts
    cmd = decoder.update(r, features, mood, t_ms, brain_s) # rates -> MotorCommand (mode, v_target, omega, wings, events)
    kin, wall_events = body.integrate(cmd, brain_s, t_ms)  # MotorCommand -> pose on the 800 x 500 canvas
    ink = body.ink(kin, features, mood, t_ms)              # pose + market + mood -> trail style

Conventions (SPEC 0.1): canvas px, y down, heading in radians with ``0 = +x`` and **positive = clockwise on
screen**; speed px/s (negative while backing); angular velocity rad/s; wingHz Hz; brain time ``t_ms`` (int, ms).
``dt_s`` is the *brain* seconds of the tick (``steps * dt_ms / 1000``), so the fly slows down with the brain.

Provenance tags: ``[V]`` verified pathway/count (docs/RESEARCH.md), ``[L]`` literature-anchored mapping,
``[E]`` engineered (puppeteering, stand-in numbers). Everything that turns Hz into px/s is a stand-in by
construction; the pathways it reads (DNa02/DNa01/DNg13 steering [V] Yang 2024, DNp01 giant fiber jump [V]
von Reyn 2014, DNp09 freezing [L] Zacarias 2018, MDN backward walking [L] Bidaye 2014, DNg02 wingbeat power
[V] Namiki 2022, DNp03 saccades / DNp07,10 landing [V], MN9 proboscis [V], pIP10 / dMS2 song [V]) are real.

Event semantics (``MotorCommand.events``, SPEC c.18 / d.8): the sub-behaviour state machines of SPEC f.3 (feeding,
freezing, grooming, flight) may be armed underneath a higher-priority mode; ``takeoff`` / ``landing``, ``freeze`` /
``unfreeze`` and ``groom`` are therefore emitted when the **visible mode** enters / leaves ``fly`` / ``freeze`` /
``groom`` (a ``landing`` names the cause in ``reason``: ``dn_land``, ``wing_power``, ``jump`` or the masking mode),
so a consumer never sees a takeoff while the fly is visibly feeding. ``feed_start`` / ``feed_stop`` follow the public
``feeding`` flag (the mood machine reads that flag), ``sleep`` / ``wake`` follow the mood, and ``gf_spike`` (d.8:
``{side, count}``, one per tick with DNp01 spikes), ``jump``, ``song``, ``saccade``, ``wander_floor`` fire on their
triggers.

Pose feedback: the decoder does not own the body, so ``MotorDecoder.observe(kin)`` may be called by the loop
after ``FlyBody.integrate`` to hand back the true heading and speed. Without it the decoder mirrors SPEC f.4
internally (it integrates the heading it commanded and relaxes its own speed estimate, used by the wander
floor), so ``cmd.jump`` always carries the **absolute** ``heading`` of SPEC c.18 and the ``jump`` event's
``heading_out`` is the same number: ``FlyBody.integrate`` only reads it and never writes into the command or
into events the loop has already collected. Without ``observe()`` the estimate drifts from the body after a
wall bounce (the body reflects the heading), which shifts the *choice* of jump direction but never makes the
wire inconsistent - the body adopts ``jump["heading"]`` as its heading at the jump start either way.

Only numpy is imported at module import time; ``Settings``, ``Features`` and ``StepStats`` are duck-typed
(imported under ``TYPE_CHECKING`` only) so import order across workstreams never matters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flybrain.config import Settings
    from flybrain.encoder import Features
    from flybrain.snn.monitor import StepStats

__all__ = [
    "Readouts", "MotorCommand", "Kinematics", "InkStyle", "Wander", "MotorDecoder", "FlyBody",
    "MODES", "INK_STYLES", "STAMPS", "INK_MOOD_COLOR", "INK_STYLE_OF_MOOD",
]

PI = math.pi
TWO_PI = 2.0 * math.pi

#: valid ``MotorCommand.mode`` / ``Kinematics.mode`` values (SPEC c.18, d.2).
MODES: tuple[str, ...] = ("walk", "fly", "jump", "feed", "freeze", "groom", "court", "sleep")
#: valid ``InkStyle.style`` values (SPEC c.18).
INK_STYLES: tuple[str, ...] = ("solid", "rainbow", "zigzag", "dotted", "hearts")
#: valid ``InkStyle.stamp`` values (SPEC c.18); ``None`` = no stamp.
STAMPS: tuple[str, ...] = ("blob", "heart", "zzz", "dash", "bump")

# ============================================================================ constants, SPEC f.3 (decoder)
# --- turn (Yang 2024: DNa02/DNa01/DNg13 are ipsiversive, + = clockwise = toward the fly's right) ---
TURN_A02_HZ: float = 40.0        #: [L] DNa02 L/R rate difference that saturates the turn command
TURN_G13_W: float = 0.5          #: [L] weight of the DNg13 difference
TURN_G13_HZ: float = 60.0        #: [L] DNg13 difference normaliser
TURN_A01_W: float = 0.3          #: [L] weight of the DNa01 difference
TURN_A01_HZ: float = 60.0        #: [L] DNa01 difference normaliser
# --- halt / back / flight power ---
HALT_HZ: float = 30.0            #: [L] dn_halt (DNg60/DNg74, GABAergic halt DNs) rate that fully halts walking
BACK_HZ: float = 20.0            #: [L] MDN rate that saturates backward walking (Bidaye 2014)
FLIGHT_DN_HZ: float = 50.0       #: [L] DNg02 rate that saturates wingbeat power (Namiki 2022)
WING_POWER_W: float = 0.3        #: [E] weight of the DLMn/DVMn power-muscle readout in wing_power
WING_POWER_HZ: float = 30.0      #: [E] DLMn/DVMn rate normaliser
# --- feeding hysteresis (MN9 proboscis motor neuron [V]) ---
FEED_ENTER_HZ: float = 30.0      #: [E] MN9 rate that starts a meal ...
FEED_ENTER_MS: int = 200         #: [E] ... when held this long
FEED_EXIT_HZ: float = 15.0       #: [E] MN9 rate below which a meal ends ...
FEED_EXIT_MS: int = 1000         #: [E] ... when held this long
PROBOSCIS_LO_HZ: float = 15.0    #: [E] proboscis = sat((mn9 - LO) / SPAN) while feeding
PROBOSCIS_SPAN_HZ: float = 45.0  #: [E]
# --- freezing (DNp09, Zacarias 2018 [L]) ---
FREEZE_ENTER_HZ: float = 40.0    #: [E] DNp09 rate that freezes ...
FREEZE_LOOM_MIN: float = 0.2     #: [E] ... when looming >= this (or sustained_loom)
FREEZE_MIN_MS: int = 800         #: [E] minimum freeze duration
FREEZE_EXIT_HZ: float = 20.0     #: [E] exit condition: DNp09 < this ...
FREEZE_EXIT_LOOM: float = 0.1    #: [E] ... or looming < this ...
FREEZE_EXIT_MS: int = 500        #: [E] ... held this long
#: [E] **Freeze bout cap / refractory** (guaranteed motion, SPEC section 0 "Guaranteed motion"; reported as a
#: spec issue). f.3 releases a freeze only when DNp09 drops below 20 Hz or ``looming`` below 0.1 for 500 ms. A
#: market drawdown holds ``looming = sat(1.2*relu(-val) + imp_loom)`` near 0.4-0.6 for as long as ``chg_m5`` stays
#: below about -4 % (``mom = tanh(chg_m5/2)`` saturates there), so f.1's ``sustained_loom`` latches on, f.2 row 8
#: keeps ``lc_freeze`` at ``80*hill(looming; 0.3, 2)`` = 45-60 Hz and DNp09 never falls back: the fly then freezes
#: for the whole drawdown, stops walking, and the Paint canvas it exists to fill stays empty (the end-to-end run
#: measured 0 walk ticks and a 6-dot trail over five minutes). Freezing in Drosophila is a transient bout
#: (Zacarias 2018 ``[L]``), so a bout is capped here and a refractory window keeps the still-high drive from
#: re-freezing on the very next tick - the fly alternates freeze and walk instead of latching, and the cap is
#: dialable to "off" (``FREEZE_MAX_MS = 0``) like every other puppeteering knob. The budget is *accumulated*
#: freezing (``freeze_load_ms``, decaying 1:1 while the fly is free), because a GF jump clears ``frozen`` and a
#: per-bout timer would be restarted by every jump of an escape storm.
FREEZE_MAX_MS: int = 6000        #: [E] freezing budget under a continuous threat (0 disables the cap)
FREEZE_REFRACTORY_MS: int = 6000  #: [E] no re-freeze for this long after a capped bout
# --- grooming (DNg62/DNge078 [V] identity, [E] numbers) ---
GROOM_ENTER_HZ: float = 40.0     #: [E]
GROOM_MIN_MS: int = 3000         #: [E] minimum grooming bout
GROOM_EXIT_HZ: float = 15.0      #: [E]
# --- courtship / song (P1 [V], pIP10 [V], dMS2 wing-extension side [V] identity; numbers [E]) ---
COURT_P1_HZ: float = 15.0        #: [E] P1 rate that puts the body in court mode (or mood == COURTSHIP); no hangover (SPEC f.3)
SONG_PIP10_HZ: float = 20.0      #: [E] pIP10 rate that extends a wing (song)
SONG_MN_HZ: float = 20.0         #: [E] song motor neuron rate (0.5 * (hg1 + b1)) that extends a wing
# --- giant-fiber jump (DNp01 [V] von Reyn 2014; DNp02/04/11 forward-takeoff bias [E]) ---
JUMP_REFRACTORY_MS: int = 1500   #: [E] GF refractory: a second GF spike within this window is ignored
JUMP_DURATION_MS: int = 400      #: [E] jump mode lasts this long
JUMP_IMPULSE: float = 600.0      #: [E] px/s initial velocity
JUMP_WING_HZ: float = 250.0      #: [E] wingbeat during the jump
JUMP_FWD_SPREAD: float = PI / 6  #: [E] forward jump: heading + U(-spread, +spread)
JUMP_BACK_SPREAD: float = PI / 4 #: [E] backward jump: heading + pi + U(-spread, +spread) + side bias
JUMP_SIDE_BIAS: float = PI / 6   #: [E] flee away from the threatened side (L -> -bias, R -> +bias)
JUMP_TAKEOFF_LOOM: float = 0.5   #: [E] looming at the end of a jump that turns it into flight
DNP11_RECENT_MS: int = 100       #: [E] window of the ``dnp11_recent_ms`` timer (c.18 state, introspection only:
                                 #: SPEC f.3 decides the takeoff direction from *this tick's* ``dnp11_spikes > 0``)
# --- flight (DNg02 [V]; DNp07/DNp10 landing [V] identity; numbers [E]) ---
FLY_ENTER_WP: float = 0.3        #: [E] wing_power >= this ...
FLY_ENTER_MS: int = 200          #: [E] ... held this long -> takeoff
FLY_EXIT_WP: float = 0.15        #: [E] wing_power < this ...
FLY_EXIT_MS: int = 2000          #: [E] ... held this long -> landing
LAND_HZ: float = 20.0            #: [E] dn_land (DNp07/10) rate that lands immediately
# --- saccade (DNp03 [V] identity; numbers [E]) ---
SACCADE_HZ: float = 40.0         #: [E] dn_saccade_L/R rate that triggers a saccade in flight
SACCADE_ANGLE: float = PI / 2    #: [E] rad per saccade (+ = clockwise for the right DNp03), total turn
SACCADE_MS: int = 100            #: [E] ramp duration -> peak rate SACCADE_ANGLE / (SACCADE_MS / 1000)
SACCADE_REFRACTORY_MS: int = 500 #: [E] at most one saccade per this window
# --- per-mode command table ---
WALK_VMAX: float = 120.0         #: [E] px/s at fwd >= WALK_FWD_HZ
WALK_FWD_HZ: float = 30.0        #: [E] DNg100/dn_fwd rate that saturates walking speed
BACK_THRESH: float = 0.5         #: [E] back > this -> backward walking
BACK_VMAX: float = 40.0          #: [E] px/s backward at back == 1
WALK_OMEGA_GAIN: float = 4.0     #: [E] rad/s per unit turn while walking
HALT_DN_THRESH: float = 0.8      #: [E] halt >= this -> halt_reason "halt_dn"
FLY_V0: float = 180.0            #: [E] px/s flight speed at wing_power 0
FLY_V_GAIN: float = 220.0        #: [E] px/s per unit wing_power
FLY_OMEGA_GAIN: float = 7.0      #: [E] rad/s per unit turn in flight
FLY_WING_STEER_GAIN: float = 0.02  #: [E] rad/s per Hz of (b1+i1) L/R asymmetry (steering muscles [V] identity)
FLY_WING_HZ0: float = 150.0      #: [L] wingbeat Hz at wing_power 0 (Drosophila ~150-250 Hz)
FLY_WING_HZ_GAIN: float = 100.0  #: [L] extra Hz at wing_power 1
FLY_WING_AMP0: float = 0.6       #: [E]
FLY_WING_AMP_GAIN: float = 0.4   #: [E]
FLY_WING_AMP_HZ: float = 15.0    #: [E] DLMn/DVMn rate that saturates the wing amplitude term
COURT_V: float = 60.0            #: [E] px/s while courting
COURT_OMEGA: float = 3.0         #: [E] rad/s circling (clockwise) while courting
# --- wander (OU heading noise) and wander floor (guaranteed motion) ---
WANDER_SIGMA_DEFAULT: float = 0.6  #: [E] rad/s/sqrt(s) (settings.wander_sigma)
WANDER_TAU_S: float = 1.0        #: [E] OU time constant
STALL_SPEED: float = 5.0         #: [E] |speed| < this counts as stalled ...
STALL_MS: int = 3000             #: [E] ... for more than this -> wander floor engages
WANDER_FLOOR_V: float = 40.0     #: [E] px/s forced while the floor is engaged
WANDER_FLOOR_RELEASE_HZ: float = 5.0  #: [E] fwd > this releases the floor (and blocks its engagement)

# ============================================================================ constants, SPEC f.4 (body)
TAU_V_WALK: float = 0.15         #: [E] s, speed relaxation while walking
TAU_V_FLY: float = 0.30          #: [E] s, in flight
TAU_V_STILL: float = 0.05        #: [E] s, court/feed/freeze/groom/sleep
JUMP_TAU_S: float = 0.15         #: [E] s, jump velocity decay: v = impulse * exp(-t / tau)
WALL_MARGIN_PX: float = 8.0      #: [E] px, bounce margin
BODY_LENGTH_PX: float = 20.0     #: [E] px, one gait cycle per body length
TREMOR_PX: float = 1.5           #: [E] px, wire-pose tremor amplitude while frozen

# ============================================================================ constants, SPEC f.5 (ink)
INK_UP: str = "#00a800"          #: [E] Win95 green: candle up
INK_DOWN: str = "#a80000"        #: [E] Win95 red: candle down
INK_FLAT: str = "#000000"        #: [E] black: candle flat
#: mood colour overrides of the candle colour (SPEC f.5; EUPHORIA is magenta, its rainbow *style* overrides).
INK_MOOD_COLOR: dict[str, str] = {
    "FEEDING": "#ff8c00", "COURTSHIP": "#ff69b4", "PANIC": "#ff0000", "ESCAPE": "#ff3300",
    "SLEEP": "#6b6b6b", "EUPHORIA": "#ff00ff",
}
INK_WIDTH_CHG_PCT: float = 3.0   #: [E] |chg_m5| (%) that saturates the width: width = 1 + round(3 * sat(|chg|/3))
INK_WIDTH_MAX_EXTRA: int = 3     #: [E] width 1..4 px
INK_ALPHA_OF_MODE: dict[str, float] = {"fly": 0.4, "court": 0.7, "jump": 0.0}   #: [E] else 1.0
INK_STYLE_OF_MOOD: dict[str, str] = {
    "EUPHORIA": "rainbow", "PANIC": "zigzag", "ESCAPE": "zigzag", "FEEDING": "dotted", "COURTSHIP": "hearts",
}                                #: [E] else "solid"
STAMP_BLOB_MS: int = 250         #: [E] "blob" cadence while feeding
STAMP_HEART_MS: int = 500        #: [E] "heart" cadence while courting
STAMP_ZZZ_MS: int = 500          #: [E] "zzz" cadence while sleeping

_NEG_INF_MS: int = -(10 ** 9)


# ============================================================================ helpers
def _f(x: Any, default: float = 0.0) -> float:
    """float() with None/NaN/inf/garbage -> ``default`` (never lets a NaN into the body)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _sat(x: float) -> float:
    """clip(x, 0, 1) with NaN -> 0."""
    if x != x:
        return 0.0
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))


def _clip(x: float, lo: float, hi: float) -> float:
    if x != x:
        return 0.0
    return lo if x < lo else (hi if x > hi else float(x))


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = _f(a)
    a = (a + PI) % TWO_PI - PI
    if a <= -PI:
        a += TWO_PI
    return a


def mood_name(mood: Any) -> str:
    """Accepts a mood string, a ``Mood`` enum, or a ``MoodState`` (uses ``.state``); returns the upper-case name
    (``"sleep"`` -> ``"SLEEP"``); ``None`` -> ``"CRUISING"``."""
    if mood is None:
        return "CRUISING"
    if not isinstance(mood, str):
        inner = getattr(mood, "state", None)
        if inner is not None:
            mood = inner
    value = getattr(mood, "value", None)
    if isinstance(value, str):
        return value.upper()
    return str(mood).upper()


def _ev(kind: str, t_ms: int, **data: Any) -> dict:
    return {"kind": kind, "t_ms": int(t_ms), "data": dict(data)}


# ============================================================================ Readouts
_RATE_FIELDS: tuple[tuple[str, str], ...] = (
    ("halt", "dn_halt"), ("back", "dn_back"), ("freeze", "dn_freeze"),
    ("a02_l", "steer_a02_L"), ("a02_r", "steer_a02_R"), ("a01_l", "steer_a01_L"), ("a01_r", "steer_a01_R"),
    ("g13_l", "steer_g13_L"), ("g13_r", "steer_g13_R"),
    ("flight", "flight_dn"), ("wing_power", "wing_power"),
    ("b1_l", "b1_L"), ("b1_r", "b1_R"), ("i1_l", "i1_L"), ("i1_r", "i1_R"), ("hg1_l", "hg1_L"), ("hg1_r", "hg1_R"),
    ("saccade_l", "dn_saccade_L"), ("saccade_r", "dn_saccade_R"), ("land", "dn_land"),
    ("mn9", "feed_mn"), ("groom", "groom_dn"), ("p1", "p1"), ("pip10", "pip10"),
    ("dms2_l", "dms2_L"), ("dms2_r", "dms2_R"),
    ("lc_loom", "lc_loom"), ("dn_mean", "dn_mean"),
)


@dataclass(frozen=True)
class Readouts:
    """Named population rates (Hz per neuron, EMA) taken from ``StepStats.rates`` (SPEC c.18, f.3).

    ``fwd = max(dng100, dn_fwd)``, ``song_mn = 0.5 * (hg1 + b1)``, ``gf_spikes_l/r`` = DNp01 spikes per side in
    the window, ``dnp11_spikes`` = pooled DNp02/04/11 spikes (``spike_counts_by_group["escape_dn"]``, the
    forward-takeoff bias [E]). Missing keys read as 0; NaN/inf are sanitised to 0.
    """

    fwd: float; halt: float; back: float; freeze: float
    a02_l: float; a02_r: float; a01_l: float; a01_r: float; g13_l: float; g13_r: float
    flight: float; wing_power: float; b1_l: float; b1_r: float; i1_l: float; i1_r: float; hg1_l: float; hg1_r: float
    saccade_l: float; saccade_r: float; land: float
    mn9: float; groom: float; p1: float; pip10: float; song_mn: float; dms2_l: float; dms2_r: float
    gf_spikes_l: int; gf_spikes_r: int; dnp11_spikes: int
    lc_loom: float; dn_mean: float

    @classmethod
    def from_stats(cls, s: "StepStats") -> "Readouts":
        rates = getattr(s, "rates", None) or {}

        def g(key: str) -> float:
            return _f(rates.get(key, 0.0))

        kw: dict[str, Any] = {name: g(key) for name, key in _RATE_FIELDS}
        kw["fwd"] = max(g("dng100"), g("dn_fwd"))
        kw["song_mn"] = 0.5 * (g("hg1") + g("b1"))
        gf = getattr(s, "gf_spikes", None) or {}
        kw["gf_spikes_l"] = max(0, int(_f(gf.get("L", 0))))
        kw["gf_spikes_r"] = max(0, int(_f(gf.get("R", 0))))
        by_group = getattr(s, "spike_counts_by_group", None) or {}
        pooled = by_group.get("escape_dn", by_group.get("dnp11", 0))
        kw["dnp11_spikes"] = max(0, int(_f(pooled)))
        return cls(**kw)

    @classmethod
    def zeros(cls, **overrides: Any) -> "Readouts":
        """All-zero readouts with keyword overrides (test / replay helper)."""
        kw: dict[str, Any] = {name: 0.0 for name, _ in _RATE_FIELDS}
        kw.update(fwd=0.0, song_mn=0.0, gf_spikes_l=0, gf_spikes_r=0, dnp11_spikes=0)
        kw.update(overrides)
        return cls(**kw)


# ============================================================================ command / kinematics / ink
@dataclass(slots=True)
class MotorCommand:
    """One tick of motor intent (SPEC c.18). ``jump`` is non-None for the whole 400 ms jump and carries exactly
    the four c.18 keys ``heading`` (absolute rad), ``impulse`` px/s, ``side`` L|R|both, ``forward`` bool; the dict
    and the ``jump`` event are read-only for every consumer (``FlyBody`` never writes into them)."""

    mode: str
    v_target: float
    omega: float
    wing_hz: float
    wing_amp: float
    wing_ext: int
    proboscis: float
    jump: dict | None
    halt_reason: str | None
    events: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class Kinematics:
    """Pose on the canvas (SPEC c.18); ``to_wire()`` is ``tick.fly`` of SPEC d.2."""

    x: float; y: float; vx: float; vy: float; heading: float; speed: float; omega: float
    wing_hz: float; wing_amp: float; wing_ext: int; mode: str; leg_phase: float; proboscis: float
    jump_t_ms: int | None

    def to_wire(self) -> dict:
        return {
            "x": round(_f(self.x), 1), "y": round(_f(self.y), 1),
            "vx": round(_f(self.vx), 1), "vy": round(_f(self.vy), 1),
            "heading": round(_f(self.heading), 3), "speed": round(_f(self.speed), 1),
            "omega": round(_f(self.omega), 3),
            # wing_hz is a Hz quantity (150 + 100*wing_power); the d preamble rounds "rates 2 dp" and the
            # frequency bucket is the only one that fits it (not the 1 dp of the px/s speeds).
            "wing_hz": round(_f(self.wing_hz), 2), "wing_amp": round(_sat(_f(self.wing_amp)), 3),
            "wing_ext": int(self.wing_ext), "mode": str(self.mode),
            "leg_phase": round(_sat(_f(self.leg_phase)), 3), "proboscis": round(_sat(_f(self.proboscis)), 3),
            "jump_t_ms": None if self.jump_t_ms is None else int(self.jump_t_ms),
        }


@dataclass(frozen=True)
class InkStyle:
    """Trail style of one tick (SPEC c.18, f.5); serialised as ``tick.ink``."""

    color: str
    width: float
    alpha: float
    style: str
    stamp: str | None

    def to_wire(self) -> dict:
        return {"color": self.color, "width": float(self.width), "alpha": float(self.alpha),
                "style": self.style, "stamp": self.stamp}


# ============================================================================ Wander
class Wander:
    """Ornstein-Uhlenbeck angular-velocity noise (rad/s) [E]: ``eta <- eta*exp(-dt/tau) + sigma*sqrt(dt)*N(0,1)``.

    Stationary std is ``sigma * sqrt(tau / 2)`` (0.42 rad/s for the defaults). ``seed`` is the int (or
    ``SeedSequence``) derived from ``SeedSequence(FLY_SEED)`` child ``[3]`` (SPEC 0.1).
    """

    def __init__(self, seed: int | np.random.SeedSequence, sigma: float = WANDER_SIGMA_DEFAULT,
                 tau_s: float = WANDER_TAU_S) -> None:
        self.sigma = max(0.0, _f(sigma, WANDER_SIGMA_DEFAULT))
        self.tau_s = max(1e-6, _f(tau_s, WANDER_TAU_S))
        self.rng = np.random.default_rng(seed)
        self.eta = 0.0

    def sample(self, dt_s: float) -> float:
        dt = max(0.0, _f(dt_s))
        decay = math.exp(-dt / self.tau_s)
        self.eta = self.eta * decay + self.sigma * math.sqrt(dt) * float(self.rng.standard_normal())
        if not math.isfinite(self.eta):
            self.eta = 0.0
        return self.eta

    def reset(self) -> None:
        self.eta = 0.0


# ============================================================================ MotorDecoder
class MotorDecoder:
    """Population rates -> ``MotorCommand`` (SPEC f.3).

    Held timers (all brain ms), the c.18 list plus the ones the f.3 rules need: ``last_jump_ms``,
    ``feed_enter_timer`` / ``feed_exit_timer``, ``groom_until``, ``stall_timer`` (wander floor), plus the freeze
    (``freeze_since_ms``, ``freeze_exit_timer``) and flight (``fly_enter_timer`` / ``fly_exit_timer``) timers.
    Three of the c.18 timers are **recorded but never read** by any f.3 rule, and are kept only because c.18
    mandates them (they are reported by ``timers()`` for the selftest / debug UI): ``court_until`` (the last tick
    at which the court condition held - f.3's court rule has no hangover), ``saccade_until`` (the saccade is
    driven by its remaining angle, see ``update``) and ``dnp11_recent_ms`` (f.3 decides the takeoff direction
    from this tick's ``dnp11_spikes`` alone). ``feeding`` is the public
    flag the loop hands to ``FeatureExtractor.update`` and ``MoodInputs``. ``mode`` is the visible mode of the last
    tick; the ``_*_visible`` flags track which of fly / freeze / groom is currently on screen (event bookkeeping,
    see the module docstring).
    """

    def __init__(self, settings: "Settings", seed: int | np.random.SeedSequence) -> None:
        self.wander_sigma = max(0.0, _f(getattr(settings, "wander_sigma", WANDER_SIGMA_DEFAULT), WANDER_SIGMA_DEFAULT))
        self.explore_baseline = max(0.0, _f(getattr(settings, "explore_baseline", 0.25), 0.25))
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(int(seed) & 0xFFFFFFFFFFFFFFFF)
        c_wander, c_jump = ss.spawn(2)
        self.wander = Wander(c_wander, sigma=self.wander_sigma, tau_s=WANDER_TAU_S)
        self._rng = np.random.default_rng(c_jump)
        # pose knowledge (see module docstring)
        self.heading: float = 0.0
        self._speed_obs: float | None = None
        self._v_est: float = 0.0
        # mode + flags
        self.mode: str = "walk"
        self.feeding: bool = False
        self.feed_enter_timer: float = 0.0
        self.feed_exit_timer: float = 0.0
        self.feed_start_ms: int | None = None
        self.frozen: bool = False
        self.freeze_since_ms: int = _NEG_INF_MS
        self.freeze_exit_timer: float = 0.0
        self.freeze_load_ms: float = 0.0             # accumulated freezing under threat (FREEZE_MAX_MS cap)
        self.freeze_block_until: int = _NEG_INF_MS   # set only by the FREEZE_MAX_MS cap
        self.n_freeze_capped: int = 0
        self.grooming: bool = False
        self.groom_until: int = _NEG_INF_MS
        self.court_until: int = _NEG_INF_MS
        self.flying: bool = False
        self.fly_enter_timer: float = 0.0
        self.fly_exit_timer: float = 0.0
        self.last_jump_ms: int = _NEG_INF_MS
        self.jump: dict | None = None
        self.jump_start_ms: int | None = None
        self.jump_d_heading: float = 0.0   # heading offset of the current jump (introspection; not on the wire)
        self.saccade_until: int = _NEG_INF_MS
        self.last_saccade_ms: int = _NEG_INF_MS
        self.saccade_sign: int = 0
        self.saccade_left_rad: float = 0.0
        self.stall_timer: float = 0.0
        self.wander_floor_on: bool = False
        self.n_wander_floor: int = 0
        self.dnp11_recent_ms: int = _NEG_INF_MS
        self.sleeping: bool = False
        self.song_side: int = 0
        # visible sub-behaviours (event bookkeeping)
        self._fly_visible: bool = False
        self._freeze_visible: bool = False
        self._groom_visible: bool = False
        self.last_cmd: MotorCommand | None = None

    # ------------------------------------------------------------------ pose feedback
    def observe(self, kin: "Kinematics") -> None:
        """Optional: hand the true pose back (call after ``FlyBody.integrate``). Used for the wander floor speed
        and the absolute jump heading; consumed once per ``update``."""
        self.heading = wrap_angle(_f(getattr(kin, "heading", self.heading), self.heading))
        self._speed_obs = _f(getattr(kin, "speed", 0.0))

    # ------------------------------------------------------------------ main
    def update(self, r: Readouts, f: "Features", mood: Any, t_ms: int, dt_s: float) -> MotorCommand:
        """SPEC f.3: derived signals, hysteretic sub-behaviours, mode priority, per-mode command."""
        t_ms = int(t_ms)
        dt_s = max(0.0, _f(dt_s))
        dt_ms = dt_s * 1000.0
        mood_s = mood_name(mood)
        events: list[dict] = []

        # ---- inputs (sanitised) ----
        looming = _sat(_f(getattr(f, "looming", 0.0)))
        sustained = bool(getattr(f, "sustained_loom", False))
        fwd, halt_hz, back_hz, freeze_hz = _f(r.fwd), _f(r.halt), _f(r.back), _f(r.freeze)
        a02_l, a02_r, a01_l, a01_r = _f(r.a02_l), _f(r.a02_r), _f(r.a01_l), _f(r.a01_r)
        g13_l, g13_r = _f(r.g13_l), _f(r.g13_r)
        flight, wp_hz = _f(r.flight), _f(r.wing_power)
        b1_l, b1_r, i1_l, i1_r = _f(r.b1_l), _f(r.b1_r), _f(r.i1_l), _f(r.i1_r)
        sacc_l, sacc_r, land = _f(r.saccade_l), _f(r.saccade_r), _f(r.land)
        mn9, groom_hz, p1, pip10, song_mn = _f(r.mn9), _f(r.groom), _f(r.p1), _f(r.pip10), _f(r.song_mn)
        dms2_l, dms2_r = _f(r.dms2_l), _f(r.dms2_r)
        gf_l, gf_r = max(0, int(_f(r.gf_spikes_l))), max(0, int(_f(r.gf_spikes_r)))
        dnp11 = max(0, int(_f(r.dnp11_spikes)))

        # ---- derived signals (SPEC f.3) ----
        turn = _clip((a02_r - a02_l) / TURN_A02_HZ + TURN_G13_W * (g13_r - g13_l) / TURN_G13_HZ
                     + TURN_A01_W * (a01_r - a01_l) / TURN_A01_HZ, -1.0, 1.0)
        halt = _sat(halt_hz / HALT_HZ)
        back = _sat(back_hz / BACK_HZ)
        wing_power = _sat(_sat(flight / FLIGHT_DN_HZ) + WING_POWER_W * _sat(wp_hz / WING_POWER_HZ))
        wing_hz_now = FLY_WING_HZ0 + FLY_WING_HZ_GAIN * wing_power

        if dnp11 > 0:
            self.dnp11_recent_ms = t_ms

        # ---- GF spikes (SPEC d.8 gf_spike {side, count}: one per tick with DNp01 spikes) ----
        gf_side = "both" if (gf_l > 0 and gf_r > 0) else ("L" if gf_l > 0 else "R")
        if (gf_l + gf_r) > 0:
            events.append(_ev("gf_spike", t_ms, side=gf_side, count=gf_l + gf_r))

        # ---- jump trigger (GF spike, refractory 1.5 s) ----
        land_reason: str | None = None
        jumping = self.jump_start_ms is not None and (t_ms - self.jump_start_ms) < JUMP_DURATION_MS
        if (gf_l + gf_r) > 0 and not jumping and (t_ms - self.last_jump_ms) >= JUMP_REFRACTORY_MS:
            side = gf_side
            forward = dnp11 > 0            # SPEC f.3 literally: this tick's pooled DNp02/04/11 count
            if forward:
                d_heading = float(self._rng.uniform(-JUMP_FWD_SPREAD, JUMP_FWD_SPREAD))
            else:
                bias = -JUMP_SIDE_BIAS if side == "L" else (JUMP_SIDE_BIAS if side == "R" else 0.0)
                d_heading = PI + float(self._rng.uniform(-JUMP_BACK_SPREAD, JUMP_BACK_SPREAD)) + bias
            heading_out = wrap_angle(self.heading + d_heading)
            # SPEC c.18 shape, exactly four keys; the body reads it read-only (module docstring).
            self.jump = {"heading": heading_out, "impulse": JUMP_IMPULSE, "side": side, "forward": bool(forward)}
            self.jump_d_heading = d_heading
            self.jump_start_ms = t_ms
            self.heading = heading_out     # the body adopts this heading at the jump start (SPEC f.4)
            self.last_jump_ms = t_ms
            jumping = True
            events.append(_ev("jump", t_ms, heading_out=heading_out, impulse=JUMP_IMPULSE, side=side,
                              forward=bool(forward)))
            # a jump breaks whatever the body was doing [E] (unfreeze / groom end are visible-mode events below)
            if self.feeding:
                self.feeding = False
                events.append(_ev("feed_stop", t_ms, meal_ms=int(t_ms - (self.feed_start_ms or t_ms))))
                self.feed_start_ms = None
            self.frozen = False
            self.grooming = False
            self.feed_enter_timer = self.feed_exit_timer = self.freeze_exit_timer = 0.0
            self.fly_enter_timer = self.fly_exit_timer = 0.0
            self.wander_floor_on = False
            self.stall_timer = 0.0
        elif not jumping and self.jump is not None:
            # the 400 ms jump just ended: fly if the threat is still looming, else walk (SPEC f.3)
            self.jump = None
            self.jump_start_ms = None
            if looming >= JUMP_TAKEOFF_LOOM:
                self.flying = True
                self.fly_exit_timer = 0.0
            elif self.flying:
                # a GF jump taken mid-flight ends on the ground: an honest landing, not a silent reset
                self.flying = False
                self.fly_enter_timer = 0.0
                land_reason = "jump"

        # ---- feeding hysteresis (MN9) ----
        if not self.feeding:
            self.feed_enter_timer = self.feed_enter_timer + dt_ms if mn9 >= FEED_ENTER_HZ else 0.0
            if mn9 >= FEED_ENTER_HZ and self.feed_enter_timer >= FEED_ENTER_MS and not jumping:
                self.feeding = True
                self.feed_start_ms = t_ms
                self.feed_exit_timer = 0.0
                events.append(_ev("feed_start", t_ms, mn9_hz=mn9))
        else:
            self.feed_exit_timer = self.feed_exit_timer + dt_ms if mn9 < FEED_EXIT_HZ else 0.0
            if self.feed_exit_timer >= FEED_EXIT_MS:
                self.feeding = False
                events.append(_ev("feed_stop", t_ms, meal_ms=int(t_ms - (self.feed_start_ms or t_ms))))
                self.feed_start_ms = None
                self.feed_enter_timer = 0.0
        proboscis = _sat((mn9 - PROBOSCIS_LO_HZ) / PROBOSCIS_SPAN_HZ) if self.feeding else 0.0

        # ---- freezing (DNp09) ----
        # ``freeze_load_ms`` is the *accumulated* time spent standing still under threat, not the length of the
        # current bout: a GF jump clears ``frozen`` (above), so an escape storm (a sell wall fires DNp01 every
        # 1.5 s) would otherwise restart a per-bout timer forever and the cap would never fire - exactly what the
        # live 20k session showed (freeze 0.81 / jump 0.19 / walk 0.00 of all ticks). It decays 1:1 while the fly
        # is free, so a permanent threat yields at most half the ticks frozen.
        self.freeze_load_ms = (min(float(FREEZE_MAX_MS), self.freeze_load_ms + dt_ms) if self.frozen
                               else max(0.0, self.freeze_load_ms - dt_ms))
        if not self.frozen:
            if (freeze_hz >= FREEZE_ENTER_HZ and (looming >= FREEZE_LOOM_MIN or sustained) and not jumping
                    and t_ms >= self.freeze_block_until):
                self.frozen = True
                self.freeze_since_ms = t_ms
                self.freeze_exit_timer = 0.0
        else:
            release = freeze_hz < FREEZE_EXIT_HZ or looming < FREEZE_EXIT_LOOM
            self.freeze_exit_timer = self.freeze_exit_timer + dt_ms if release else 0.0
            if (t_ms - self.freeze_since_ms) >= FREEZE_MIN_MS and self.freeze_exit_timer >= FREEZE_EXIT_MS:
                self.frozen = False
            elif FREEZE_MAX_MS > 0 and self.freeze_load_ms >= FREEZE_MAX_MS:
                # the threat outlasted a freezing bout: move again and do not re-freeze immediately
                self.frozen = False
                self.freeze_exit_timer = 0.0
                self.freeze_block_until = t_ms + FREEZE_REFRACTORY_MS
                self.n_freeze_capped += 1

        # ---- grooming ----
        if not self.grooming:
            if groom_hz >= GROOM_ENTER_HZ and not jumping:
                self.grooming = True
                self.groom_until = t_ms + GROOM_MIN_MS
        elif t_ms >= self.groom_until and groom_hz < GROOM_EXIT_HZ:
            self.grooming = False

        # ---- courtship: while mood == COURTSHIP or p1 >= 15 Hz (no hangover, SPEC f.3) ----
        court = mood_s == "COURTSHIP" or p1 >= COURT_P1_HZ
        if court:
            self.court_until = t_ms

        # ---- flight enter / exit ----
        if not jumping:
            if not self.flying:
                self.fly_enter_timer = self.fly_enter_timer + dt_ms if wing_power >= FLY_ENTER_WP else 0.0
                if self.fly_enter_timer >= FLY_ENTER_MS:
                    self.flying = True
                    self.fly_exit_timer = 0.0
            else:
                if land >= LAND_HZ:
                    self.flying = False
                    self.fly_enter_timer = 0.0
                    land_reason = "dn_land"
                else:
                    self.fly_exit_timer = self.fly_exit_timer + dt_ms if wing_power < FLY_EXIT_WP else 0.0
                    if self.fly_exit_timer >= FLY_EXIT_MS:
                        self.flying = False
                        self.fly_enter_timer = 0.0
                        land_reason = "wing_power"

        # ---- sleep / wake (mood-driven) ----
        sleeping = mood_s == "SLEEP"
        if sleeping and not self.sleeping:
            events.append(_ev("sleep", t_ms))
        elif self.sleeping and not sleeping:
            events.append(_ev("wake", t_ms, reason="mood left SLEEP"))
        self.sleeping = sleeping

        # ---- mode priority: jump > sleep > court > feed > freeze > groom > fly > walk ----
        if jumping:
            mode = "jump"
        elif sleeping:
            mode = "sleep"
        elif court:
            mode = "court"
        elif self.feeding:
            mode = "feed"
        elif self.frozen:
            mode = "freeze"
        elif self.grooming:
            mode = "groom"
        elif self.flying:
            mode = "fly"
        else:
            mode = "walk"

        # ---- visible-mode events (see module docstring): takeoff/landing, freeze/unfreeze, groom ----
        if self._fly_visible and mode not in ("fly", "jump"):
            self._fly_visible = False
            events.append(_ev("landing", t_ms, reason=land_reason or mode))
        if mode == "fly" and not self._fly_visible:
            self._fly_visible = True
            events.append(_ev("takeoff", t_ms, wing_hz=wing_hz_now))
        if self._freeze_visible and mode != "freeze":
            self._freeze_visible = False
            events.append(_ev("unfreeze", t_ms, dnp09_hz=freeze_hz, looming=looming))
        if mode == "freeze" and not self._freeze_visible:
            self._freeze_visible = True
            events.append(_ev("freeze", t_ms, dnp09_hz=freeze_hz, looming=looming))
        if self._groom_visible and mode != "groom":
            self._groom_visible = False
        if mode == "groom" and not self._groom_visible:
            self._groom_visible = True
            events.append(_ev("groom", t_ms, side="both"))

        # ---- song (wing extension; every mode, SPEC f.3) ----
        wing_ext = 0
        if pip10 >= SONG_PIP10_HZ or song_mn >= SONG_MN_HZ:
            wing_ext = 1 if dms2_r >= dms2_l else -1
        if wing_ext != 0 and wing_ext != self.song_side:
            events.append(_ev("song", t_ms, side="R" if wing_ext > 0 else "L"))
        self.song_side = wing_ext

        # ---- saccade (flight only) ----
        # SPEC f.3: "heading += (+pi/2 if R else -pi/2) over 100 ms". The *total* turn is the contract, so the
        # remaining angle is tracked and the rate is capped at pi/2 per 100 ms: the integral is exactly pi/2 for
        # any dt (a deadline plus a constant rate overshoots whenever the brain ms per tick does not divide 100).
        omega_saccade = 0.0
        saccade_ramp = SACCADE_ANGLE / (SACCADE_MS / 1000.0)
        if mode == "fly":
            if (sacc_l >= SACCADE_HZ or sacc_r >= SACCADE_HZ) and (t_ms - self.last_saccade_ms) >= SACCADE_REFRACTORY_MS:
                self.saccade_sign = 1 if sacc_r >= sacc_l else -1
                self.saccade_left_rad = self.saccade_sign * SACCADE_ANGLE
                self.saccade_until = t_ms + SACCADE_MS   # c.18 held timer (introspection; the angle is authoritative)
                self.last_saccade_ms = t_ms
                events.append(_ev("saccade", t_ms, side="R" if self.saccade_sign > 0 else "L"))
            if self.saccade_left_rad != 0.0 and dt_s > 0.0:
                omega_saccade = _clip(self.saccade_left_rad / dt_s, -saccade_ramp, saccade_ramp)
                self.saccade_left_rad -= omega_saccade * dt_s
                if abs(self.saccade_left_rad) < 1e-12:
                    self.saccade_left_rad = 0.0
        else:
            self.saccade_left_rad = 0.0   # a saccade belongs to flight; landing abandons the rest of the turn

        # ---- wander floor (guaranteed motion, walk only, only when explore_baseline > 0) ----
        # SPEC f.3 literally: engages after > 3000 ms with |kin.speed| < 5 px/s (whatever fwd is doing) and is
        # released by fwd > 5 Hz; one event per engagement. The explore baseline normally prevents it from ever
        # engaging because the fly keeps moving at >= 30 px/s, not because fwd is high.
        speed_now = self._speed_obs if self._speed_obs is not None else self._v_est
        self._speed_obs = None
        if mode == "walk" and self.explore_baseline > 0.0:
            self.stall_timer = self.stall_timer + dt_ms if abs(speed_now) < STALL_SPEED else 0.0
            if self.wander_floor_on and fwd > WANDER_FLOOR_RELEASE_HZ:
                self.wander_floor_on = False
                self.stall_timer = 0.0
            if not self.wander_floor_on and self.stall_timer > STALL_MS:
                self.wander_floor_on = True
                self.n_wander_floor += 1
                events.append(_ev("wander_floor", t_ms, stalled_ms=int(self.stall_timer)))
        else:
            self.stall_timer = 0.0
            self.wander_floor_on = False

        # ---- per-mode command (SPEC f.3 table) ----
        eta = self.wander.sample(dt_s)   # sampled every tick so the RNG stream is input-independent
        v_target = 0.0
        omega = 0.0
        wing_hz = 0.0
        wing_amp = 0.0
        halt_reason: str | None = None
        if mode == "walk":
            v_target = WALK_VMAX * _sat(fwd / WALK_FWD_HZ) * (1.0 - halt)
            if back > BACK_THRESH:
                v_target = -BACK_VMAX * back
            omega = WALK_OMEGA_GAIN * turn + eta
            halt_reason = "halt_dn" if halt >= HALT_DN_THRESH else None
            if self.wander_floor_on:
                # SPEC f.3 overrides only v_target; halt_reason stays exactly what the walk row says (it is a
                # diagnostic field, absent from the d.2 wire), so "halted but pushed to 40 px/s" is visible.
                v_target = WANDER_FLOOR_V
        elif mode == "fly":
            v_target = FLY_V0 + FLY_V_GAIN * wing_power
            omega = (FLY_OMEGA_GAIN * turn + FLY_WING_STEER_GAIN * ((b1_r + i1_r) - (b1_l + i1_l)) + eta
                     + omega_saccade)
            wing_hz = wing_hz_now
            wing_amp = FLY_WING_AMP0 + FLY_WING_AMP_GAIN * _sat(wp_hz / FLY_WING_AMP_HZ)
        elif mode == "jump":
            v_target = JUMP_IMPULSE
            wing_hz = JUMP_WING_HZ
            wing_amp = 1.0
        elif mode == "court":
            v_target = COURT_V
            omega = COURT_OMEGA
        else:  # feed | freeze | groom | sleep
            halt_reason = mode

        # ---- internal pose estimate mirroring SPEC f.4 (observe() overrides it when the loop calls it) ----
        if mode == "jump" and self.jump_start_ms is not None:
            self._v_est = JUMP_IMPULSE * math.exp(-((t_ms - self.jump_start_ms) / 1000.0) / JUMP_TAU_S)
        else:
            tau = TAU_V_WALK if mode == "walk" else (TAU_V_FLY if mode == "fly" else TAU_V_STILL)
            self._v_est += (v_target - self._v_est) * (1.0 - math.exp(-dt_s / tau))
        self.heading = wrap_angle(self.heading + _f(omega) * dt_s)

        self.mode = mode
        cmd = MotorCommand(
            mode=mode, v_target=_f(v_target), omega=_f(omega), wing_hz=_f(wing_hz), wing_amp=_sat(_f(wing_amp)),
            wing_ext=int(wing_ext), proboscis=_sat(_f(proboscis)),
            jump=self.jump if mode == "jump" else None, halt_reason=halt_reason, events=events,
        )
        self.last_cmd = cmd
        return cmd

    # ------------------------------------------------------------------ introspection
    def timers(self) -> dict[str, float]:
        """Snapshot of the held timers (debug / selftest)."""
        return {
            "last_jump_ms": self.last_jump_ms, "feed_enter_timer": self.feed_enter_timer,
            "feed_exit_timer": self.feed_exit_timer, "groom_until": self.groom_until, "court_until": self.court_until,
            "stall_timer": self.stall_timer, "saccade_until": self.saccade_until,
            "saccade_left_rad": self.saccade_left_rad,
            "dnp11_recent_ms": self.dnp11_recent_ms, "freeze_exit_timer": self.freeze_exit_timer,
            "fly_enter_timer": self.fly_enter_timer, "fly_exit_timer": self.fly_exit_timer,
            "freeze_load_ms": self.freeze_load_ms, "freeze_block_until": self.freeze_block_until,
            "n_freeze_capped": self.n_freeze_capped,
        }


# ============================================================================ FlyBody
class FlyBody:
    """Kinematics integration (SPEC f.4) and ink (SPEC f.5) on a ``w x h`` canvas.

    Starts at ``(w/2, h/2)``, heading 0, mode walk. ``walls`` is ``"bounce"`` (reflect at an 8 px margin, event
    ``wall_bump {side}`` per wall hit - a corner hit yields two events and counts two bumps) or ``"wrap"`` (torus,
    event ``wrap {edge}``). ``seed`` (SeedSequence child [3] derived) only feeds the freeze tremor of the wire
    pose; the true position is never randomised.

    A jump starts on the tick mode ``jump`` is entered from another mode (the GF refractory of 1500 ms is longer
    than the 400 ms jump, so two jumps are always separated by at least one non-jump tick) and the body then
    decays ``v = impulse * exp(-t/0.15)`` from that tick. ``cmd.jump`` and ``cmd.events`` are only ever read
    here: the c.18 ``jump`` dict the decoder published (four keys, absolute ``heading``) is the single source of
    truth, so a consumer that snapshots the events before ``integrate`` sees exactly what the wire shows.
    """

    def __init__(self, w: int, h: int, walls: str, seed: int | np.random.SeedSequence) -> None:
        self.w = float(w)
        self.h = float(h)
        self.walls = walls if walls in ("bounce", "wrap") else "bounce"
        self._rng = np.random.default_rng(seed if isinstance(seed, np.random.SeedSequence)
                                          else int(seed) & 0xFFFFFFFFFFFFFFFF)
        self.x = self.w / 2.0
        self.y = self.h / 2.0
        self.heading = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.mode = "walk"
        self.leg_phase = 0.0
        self._jump_t0: int | None = None
        self._jump_impulse: float = JUMP_IMPULSE
        self._bumped_t_ms: int | None = None
        self._stamp_last: dict[str, int] = {"blob": _NEG_INF_MS, "heart": _NEG_INF_MS, "zzz": _NEG_INF_MS}
        self.pen_lifted = True
        self.n_bumps = 0
        self.n_wraps = 0
        self._kin = Kinematics(x=self.x, y=self.y, vx=0.0, vy=0.0, heading=0.0, speed=0.0, omega=0.0,
                               wing_hz=0.0, wing_amp=0.0, wing_ext=0, mode="walk", leg_phase=0.0, proboscis=0.0,
                               jump_t_ms=None)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _bump_side(heading: float, nx: float, ny: float) -> str:
        """"L" when the wall (outward normal ``n``) lies to the fly's left before reflection:
        ``cross(heading_vec, n) < 0`` (y down, clockwise-positive heading), else "R"."""
        cross = math.cos(heading) * ny - math.sin(heading) * nx
        return "L" if cross < 0.0 else "R"

    # ------------------------------------------------------------------ integrate
    def integrate(self, cmd: MotorCommand, dt_s: float, t_ms: int) -> tuple[Kinematics, list[dict]]:
        """SPEC f.4. Returns the (wire) kinematics and the wall events of this tick."""
        t_ms = int(t_ms)
        dt = max(0.0, _f(dt_s))
        mode = cmd.mode if cmd.mode in MODES else "walk"
        events: list[dict] = []
        jump_t_ms: int | None = None

        if mode == "jump":
            j = cmd.jump if isinstance(cmd.jump, dict) else None
            if self._jump_t0 is None:
                # jump start (mode 'jump' entered): adopt the commanded heading, read-only (SPEC f.4)
                self._jump_t0 = t_ms
                if j is not None:
                    self.heading = wrap_angle(_f(j.get("heading", self.heading), self.heading))
                    self._jump_impulse = max(0.0, _f(j.get("impulse", JUMP_IMPULSE), JUMP_IMPULSE))
                else:
                    self._jump_impulse = JUMP_IMPULSE
                self.pen_lifted = True
            jump_t_ms = max(0, t_ms - self._jump_t0)
            self.v = self._jump_impulse * math.exp(-(jump_t_ms / 1000.0) / JUMP_TAU_S)
        else:
            self._jump_t0 = None
            tau = TAU_V_WALK if mode == "walk" else (TAU_V_FLY if mode == "fly" else TAU_V_STILL)
            self.v += (_f(cmd.v_target) - self.v) * (1.0 - math.exp(-dt / tau))

        omega = _f(cmd.omega)
        self.omega = omega
        self.heading = wrap_angle(self.heading + omega * dt)
        vx = self.v * math.cos(self.heading)
        vy = self.v * math.sin(self.heading)
        self.x += vx * dt
        self.y += vy * dt

        # ---- walls ----
        if self.walls == "bounce":
            m = WALL_MARGIN_PX
            h0 = self.heading   # SPEC f.4: side is judged against the heading BEFORE reflection (both walls)
            hit_x = hit_y = False
            if self.x < m:
                events.append(_ev("wall_bump", t_ms, side=self._bump_side(h0, -1.0, 0.0)))
                self.x = m
                hit_x = True
            elif self.x > self.w - m:
                events.append(_ev("wall_bump", t_ms, side=self._bump_side(h0, 1.0, 0.0)))
                self.x = self.w - m
                hit_x = True
            if self.y < m:
                events.append(_ev("wall_bump", t_ms, side=self._bump_side(h0, 0.0, -1.0)))
                self.y = m
                hit_y = True
            elif self.y > self.h - m:
                events.append(_ev("wall_bump", t_ms, side=self._bump_side(h0, 0.0, 1.0)))
                self.y = self.h - m
                hit_y = True
            if hit_x:
                self.heading = wrap_angle(PI - self.heading)
            if hit_y:
                self.heading = wrap_angle(-self.heading)
            if hit_x or hit_y:
                self.n_bumps += int(hit_x) + int(hit_y)   # one count per wall_bump event
                self._bumped_t_ms = t_ms
                vx = self.v * math.cos(self.heading)
                vy = self.v * math.sin(self.heading)
        else:  # wrap
            wrapped = False
            if self.x < 0.0:
                self.x += self.w
                events.append(_ev("wrap", t_ms, edge="left"))
                wrapped = True
            elif self.x >= self.w:
                self.x -= self.w
                events.append(_ev("wrap", t_ms, edge="right"))
                wrapped = True
            if self.y < 0.0:
                self.y += self.h
                events.append(_ev("wrap", t_ms, edge="top"))
                wrapped = True
            elif self.y >= self.h:
                self.y -= self.h
                events.append(_ev("wrap", t_ms, edge="bottom"))
                wrapped = True
            if wrapped:
                self.n_wraps += 1
                self.pen_lifted = True
            # the arena is a torus; guard against a huge single-tick displacement
            self.x = self.x % self.w if self.w > 0 else self.x
            self.y = self.y % self.h if self.h > 0 else self.y

        # ---- gait ----
        self.leg_phase = (self.leg_phase + abs(self.v) / BODY_LENGTH_PX * dt) % 1.0
        self.mode = mode

        # ---- wire pose (freeze tremor never touches the true position) ----
        wx, wy = self.x, self.y
        if mode == "freeze":
            wx += float(self._rng.uniform(-TREMOR_PX, TREMOR_PX))
            wy += float(self._rng.uniform(-TREMOR_PX, TREMOR_PX))
        self._kin = Kinematics(
            x=wx, y=wy, vx=vx, vy=vy, heading=self.heading, speed=self.v, omega=omega,
            wing_hz=_f(cmd.wing_hz), wing_amp=_sat(_f(cmd.wing_amp)), wing_ext=int(cmd.wing_ext), mode=mode,
            leg_phase=self.leg_phase, proboscis=_sat(_f(cmd.proboscis)), jump_t_ms=jump_t_ms,
        )
        if mode != "jump":
            self.pen_lifted = False
        return self._kin, events

    # ------------------------------------------------------------------ ink
    def ink(self, kin: Kinematics, f: "Features", mood: Any, t_ms: int) -> InkStyle:
        """SPEC f.5: colour from the candle (overridden by mood), width from |chg_m5|, alpha/style/stamp.

        Width rounding is half-up (``floor(x + 0.5)``, the JS ``Math.round`` convention) rather than Python's
        banker's ``round``; the two differ only at exactly 0.5 / 1.5 / 2.5 % - SPEC f.5 writes ``round`` without
        naming a convention.
        """
        t_ms = int(t_ms)
        mood_s = mood_name(mood)
        candle = getattr(f, "candle", "flat")
        color = INK_UP if candle == "up" else (INK_DOWN if candle == "down" else INK_FLAT)
        color = INK_MOOD_COLOR.get(mood_s, color)

        chg = getattr(f, "chg_m5", None)
        if chg is None:
            # Features carries mom = tanh(chg_m5 / 2) (SPEC f.1), not chg_m5 itself: invert it.
            mom = _clip(_f(getattr(f, "mom", 0.0)), -0.999999, 0.999999)
            chg = 2.0 * math.atanh(mom)
        chg = _f(chg)
        # round half-up (not banker's) so 0.5 % -> 2 px, 1.5 % -> 3 px, 2.5 % -> 4 px
        width = float(1 + int(math.floor(INK_WIDTH_MAX_EXTRA * _sat(abs(chg) / INK_WIDTH_CHG_PCT) + 0.5)))

        mode = kin.mode if kin.mode in MODES else "walk"
        alpha = INK_ALPHA_OF_MODE.get(mode, 1.0)
        style = INK_STYLE_OF_MOOD.get(mood_s, "solid")

        stamp: str | None = None
        if self._bumped_t_ms == t_ms:
            stamp = "bump"
        elif mode == "jump":
            stamp = "dash"
        elif mode == "feed":
            stamp = self._cadence("blob", STAMP_BLOB_MS, t_ms)
        elif mode == "court":
            stamp = self._cadence("heart", STAMP_HEART_MS, t_ms)
        elif mode == "sleep":
            stamp = self._cadence("zzz", STAMP_ZZZ_MS, t_ms)
        return InkStyle(color=color, width=width, alpha=float(alpha), style=style, stamp=stamp)

    def _cadence(self, name: str, period_ms: int, t_ms: int) -> str | None:
        if t_ms - self._stamp_last[name] >= period_ms:
            self._stamp_last[name] = t_ms
            return name
        return None

    # ------------------------------------------------------------------ state
    @property
    def kin(self) -> Kinematics:
        return self._kin

    def teleport(self, x: float, y: float) -> None:
        """Move the true position (clamped inside the canvas), zero the velocity and lift the pen."""
        self.x = _clip(_f(x, self.w / 2.0), 0.0, self.w)
        self.y = _clip(_f(y, self.h / 2.0), 0.0, self.h)
        self.v = 0.0
        self.pen_lifted = True
        k = self._kin
        self._kin = Kinematics(x=self.x, y=self.y, vx=0.0, vy=0.0, heading=k.heading, speed=0.0, omega=0.0,
                               wing_hz=k.wing_hz, wing_amp=k.wing_amp, wing_ext=k.wing_ext, mode=k.mode,
                               leg_phase=k.leg_phase, proboscis=k.proboscis, jump_t_ms=k.jump_t_ms)
