"""Market -> ``Features`` -> ``list[Drive]`` (SPEC sections c.17, f.1, f.2).

Two stages run once per wall tick on the sim thread:

* ``FeatureExtractor.update`` turns the latest ``MarketSnapshot`` plus the trades of the tick into the
  ``Features`` record of section f.1 (buy pressure, momentum, EMAs, trade impulses, looming episodes with a
  300 ms expanding-disc ramp, whale flashes, sleep pressure, hunger, courtship, easter eggs).
* ``SensoryEncoder.encode`` maps ``Features`` (+ heading, mood, pokes) onto the 20-row drive table of
  section f.2: one ``Drive`` per active channel with the exact group keys of section c.4.

Every numeric constant of sections f.1 / f.2 is a named module constant carrying its provenance tag:
``[V]`` verified in RESEARCH, ``[L]`` literature, ``[E]`` engineered (puppeteering / stand-in),
``[?]``/``[D]`` unverified identity / documentation-only. Almost everything here is ``[E]`` by design:
the market has no biology, the mapping is the art.

Randomness: ``SeedSequence(seed)`` child ``[6]`` ("encoder episodes", SPEC section 0.1) is split into a
feature stream (looming side of simulated sells) and an encoder stream (photoreceptor flicker phases).
Only numpy is imported at module level; the engine's ``Drive`` is a plain frozen dataclass.

Robustness (SPEC 0.1 "errors never stop the sim"): every scalar that enters the extractor (snapshot fields,
trade sizes / stamps, ``mn9_hz``, ``dt``) and the encoder (``heading``) is finite-checked; ``sat`` / ``relu`` /
``hill`` map NaN to 0 so one bad input can never poison the state integrators or ``tick_to_json``.
Source switches (``sim(fallback)`` <-> ``dexscreener``) reset the per-source snapshot references (previous
price, session ATH, liquidity start, volume EMA) so a change of price scale is not read as a tick / rug / ATH.
Wiring: f.2 says a ``loom`` poke "starts a ramp episode in the FeatureExtractor", but the c.27 loop hands pokes
to the encoder only (step 4) and never to ``FeatureExtractor.update`` (step 3), so the encoder needs a reference
to the extractor. Preferred: ``SensoryEncoder(conn, settings, seed, features=fx)`` or ``encoder.bind(fx)``
(equivalently the loop may pass ``features.update(..., pokes=pokes)`` itself). As a fallback for a
spec-literal three-argument construction, an unbound encoder binds to the **one** live ``FeatureExtractor``
built with the same ``Settings`` *object* (weak registry); with two candidates it stays unbound and warns
instead of cross-wiring. ``encoder.features`` / ``encoder.bound_via`` make the result inspectable.
"""

from __future__ import annotations

import logging
import math
import re
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np

from .market.base import DEX_SOURCE, MarketSnapshot, Trade
from .snn.engine import Drive
from .snn.params import LIFParams, current_from_rate

if TYPE_CHECKING:  # pragma: no cover
    from .config import Settings
    from .connectome.schema import Connectome
    from .snn.engine import LIFEngine

__all__ = [
    "hill",
    "sat",
    "relu",
    "finite",
    "Features",
    "FeatureExtractor",
    "Poke",
    "ChannelDrive",
    "SensoryEncoder",
    "POKE_TABLE",
    "POKE_CHANNELS",
    "DRIVE_CHANNELS",
    "FOOD_GLOMERULI",
    "MOOD_STATES_FEEDBACK",
]

log = logging.getLogger("flybrain.encoder")

# =====================================================================================================
# f.1 constants (market -> Features)
# =====================================================================================================

#: SPEC 0.1 spawn order: ``[6] encoder episodes``.
SEED_CHILD_ENCODER: int = 6

#: ``bp = (buys_m5 - sells_m5) / (n + 1)`` ``[E]``.
BP_DENOM_OFFSET: float = 1.0
#: ``mom = tanh(chg_m5 / 2.0)`` (percent) ``[E]``.
MOM_SCALE_PCT: float = 2.0
#: ``tick = tanh(100 * (P/P_prev - 1))`` ``[E]``.
TICK_SCALE: float = 100.0
#: ``val_snap = clip(0.5*bp + 0.3*mom + 0.2*tick, -1, 1)`` ``[E]``.
VAL_W_BP: float = 0.5
VAL_W_MOM: float = 0.3
VAL_W_TICK: float = 0.2
#: ``act_snap = sat(log1p(n) / log1p(500))`` ``[E]`` (500 trades / 5 min saturates activity).
ACT_LOG_REF: float = 500.0
#: ``rug_snap = 1`` when ``chg_m5 <= -15 %`` or ``liq < 0.7 * liq_start`` ``[E]``.
RUG_CHG_M5_PCT: float = -15.0
RUG_LIQ_FRAC: float = 0.7
#: ``ema_vol_1h <- EMA_3600(vol_m5)``; ``vol_snap = sat(vol_m5 / (3 * ema_vol_1h + 1))`` ``[E]``.
EMA_VOL_TAU_S: float = 3600.0
VOL_NORM_MULT: float = 3.0
VOL_NORM_OFFSET: float = 1.0
#: ``water_imp += sat((liq/liq_prev - 1) / 0.05)`` on a liquidity add ``[E]``.
WATER_LIQ_STEP: float = 0.05
#: ``candle = up|down|flat`` with a +-0.05 % dead band ``[E]``.
CANDLE_EPS: float = 0.0005
#: ``consecutive_bull_snaps`` counts snapshots with ``buys >= 3 * sells and chg_m5 > 5 %`` ``[E]``.
BULL_BUY_RATIO: float = 3.0
BULL_CHG_M5_PCT: float = 5.0

#: EMA time constants (wall seconds) ``[E]``: val 3, activity 10, vol_norm 5; rug decay 20; water 10; impulses 1.5.
TAU_VAL_S: float = 3.0
TAU_ACT_S: float = 10.0
TAU_VOL_S: float = 5.0
TAU_RUG_S: float = 20.0
TAU_WATER_S: float = 10.0
TAU_IMP_S: float = 1.5

#: ``usd_ref = max(20, median(last 200 trade sizes))``; ``s = min(1.5, usd / usd_ref)`` ``[E]``.
USD_REF_FLOOR: float = 20.0
TRADE_SIZES_MAXLEN: int = 200
TRADE_S_CAP: float = 1.5
#: buy -> ``imp_sugar += 0.15 * s``; sell -> ``imp_loom += 0.20 * s`` ``[E]``.
IMP_SUGAR_PER_TRADE: float = 0.15
IMP_LOOM_PER_TRADE: float = 0.20
#: A looming episode stays active for 2000 ms after the last sell; each sell restarts a 300 ms ramp ``[E]``
#: (the 300 ms expanding disc mimics the loom stimulus duration of LC4/LPLC2 experiments ``[L]``).
#: ``loom_t_ms`` is defined on the **closed** window ``[0, LOOM_RAMP_MS]``: f.1 writes the guard as
#: ``t_ms - loom_ramp_start_ms < 300``, but read half-open the ramp's last sample at a 50 ms tick is 250 ms =
#: ``(250/300)^2 = 69 %`` of the peak, so the ``220 * looming = 154 Hz`` at 300 ms of the f.2 sanity paragraph
#: (and the ``r(t)`` of section 0) is never reached and the g.6 gate-3 expectation "DNp01 fires 100-150 ms into
#: the ramp" is weakened. Including the endpoint costs one extra 50 ms sample and makes the peak reachable
#: (reported as a spec issue; the prose + gate win over the strict inequality).
LOOM_EPISODE_GAP_MS: int = 2000
LOOM_RAMP_MS: int = 300
#: whale = trade ``>= 10 * usd_ref``; a whale SELL flashes the full field for 300 ms ``[E]``.
WHALE_RATIO: float = 10.0
FLASH_MS: int = 300
#: ``looming = sat(1.2 * relu(-val) + imp_loom)``; ``bitter = sat(0.5 * relu(-val) + 0.5 * rug)`` ``[E]``.
LOOM_VAL_GAIN: float = 1.2
BITTER_W_VAL: float = 0.5
BITTER_W_RUG: float = 0.5
#: ``sustained_loom`` after ``looming >= 0.20`` for 3 s ``[E]``.
SUSTAINED_LOOM_THR: float = 0.20
SUSTAINED_LOOM_S: float = 3.0
#: ``odor = sat(0.4*activity + 0.6*vol_norm) * (0.4 + 0.6*exp(-(t - odor_rise_t)/2))``: olfactory adaptation
#: tau 2 s ``[L]`` (ORN adaptation), weights / floor ``[E]``; ``odor_rise_t`` resets on a > 0.1 rise per tick.
ODOR_W_ACT: float = 0.4
ODOR_W_VOL: float = 0.6
ODOR_ADAPT_FLOOR: float = 0.4
ODOR_ADAPT_SPAN: float = 0.6
ODOR_ADAPT_TAU_S: float = 2.0
ODOR_RISE_STEP: float = 0.1
#: ``chop = sat(activity * (1 - min(1, |chg_m5| / 2)))`` ``[E]``.
CHOP_CHG_SCALE_PCT: float = 2.0
#: courtship: new session high by 2 %, or 3 bull snapshots, or an easter egg, or a pheromone poke; decays tau 8 s ``[E]``.
COURT_ATH_MULT: float = 1.02
COURT_BULL_SNAPS: int = 3
COURT_TAU_S: float = 8.0
#: sleep pressure ``z += dt/90`` while ``activity < 0.35`` else ``-= dt/10`` ``[E]``.
SLEEP_ACT_THR: float = 0.35
SLEEP_RISE_S: float = 90.0
SLEEP_FALL_S: float = 10.0
#: hunger ``+= dt/600``; feeding ``-= 0.06 * sat(mn9/60) * dt``; initial 0.5 ``[E]``.
HUNGER_RISE_S: float = 600.0
HUNGER_FEED_RATE: float = 0.06
HUNGER_MN9_REF_HZ: float = 60.0
HUNGER_INIT: float = 0.5
#: ``sweet_gain = 0.6 + 0.4 * hunger``: hunger raises sweet sensitivity ``[L]`` (Inagaki 2012 dopamine/sNPF).
SWEET_GAIN_BASE: float = 0.6
SWEET_GAIN_HUNGER: float = 0.4
#: ``explore = explore_baseline + 0.25 * activity`` (0 in SLEEP) ``[E]`` guaranteed-motion floor.
EXPLORE_ACT_GAIN: float = 0.25
#: ``since_last_trade_s`` when no trade was ever seen.
NO_TRADE_S: float = 1e9
#: Price ring kept for consumers (sparkline / summary), seconds ``[E]``.
PRICE_RING_S: float = 90.0
#: Easter eggs ``[E]``: ``buys_m5 in (69, 420)``, price digits ``420`` / ``69``, local clock ``04:20`` / ``16:20``.
EASTER_BUYS: tuple[int, ...] = (69, 420)
EASTER_PRICE_DIGITS: tuple[str, ...] = ("420", "69")
EASTER_CLOCKS: tuple[str, ...] = ("04:20", "16:20")
#: Minimum spacing of repeated ``easter_egg`` events with the same reason ``[E]``.
EASTER_EVENT_REPEAT_S: float = 60.0

# =====================================================================================================
# f.2 constants (Features -> Drives)
# =====================================================================================================

#: Drives below this rate are skipped (SPEC f.2).
MIN_DRIVE_HZ: float = 1.0
#: Row 1 sugar: ``150 * hill(sweet_gain * sugar; 0.25, 1.5)``, recruit ``0.4 + 0.6 * sugar`` ``[L]`` Shiu: sugar GRNs 10-200 Hz.
#: The f.2 calibration anchors in that row (sugar 1 -> 133 Hz, 0.5 -> 111, 0.2 -> 62, 0.05 -> 12) are quoted at
#: ``sweet_gain = 1``; f.1 defines ``sweet_gain = 0.6 + 0.4*hunger``, so a running sim reads ~97 Hz at sugar 0.5
#: and the full 111 Hz only at ``hunger = 1`` (reported as a spec issue; the formula is implemented as written).
SUGAR_RATE_MAX_HZ: float = 150.0
SUGAR_HILL_K: float = 0.25
SUGAR_HILL_N: float = 1.5
GRN_RECRUIT_BASE: float = 0.4
GRN_RECRUIT_GAIN: float = 0.6
#: Row 2 water: ``60 * water`` ``[E]``.
WATER_RATE_MAX_HZ: float = 60.0
#: Row 3 bitter: ``120 * hill(bitter; 0.25, 1.5)`` ``[L]``.
BITTER_RATE_MAX_HZ: float = 120.0
BITTER_HILL_K: float = 0.25
BITTER_HILL_N: float = 1.5
#: Row 4 loom_ramp: ``220 * looming * (loom_t_ms/300)^2`` on 60 % of LC4/LPLC2 ``[L]`` loom-tuned LC response, ``[E]`` numbers.
LOOM_RAMP_RATE_MAX_HZ: float = 220.0
LOOM_RECRUIT: float = 0.6
#: Row 5 loom_tonic: ``12 * relu((looming - 0.3)/0.7)^2`` ``[E]`` primes the GF (5 mV g_ss at 12 Hz).
LOOM_TONIC_RATE_MAX_HZ: float = 12.0
LOOM_TONIC_THR: float = 0.3
LOOM_TONIC_SPAN: float = 0.7
#: Row 6 loom_aux: ``lc_loom2`` at ``0.5 x (row 4 + row 5)`` ``[E]``.
LOOM_AUX_FACTOR: float = 0.5
#: Row 7 flash: 220 Hz full-field while ``flash > 0`` ``[E]``.
FLASH_RATE_HZ: float = 220.0
#: Row 8 freeze: ``80 * hill(looming; 0.3, 2)`` while ``sustained_loom`` -> LC9/LC31a -> DNp09 ``[V]`` pathway, ``[E]`` numbers.
FREEZE_RATE_MAX_HZ: float = 80.0
FREEZE_HILL_K: float = 0.3
FREEZE_HILL_N: float = 2.0
#: Row 9 light: ``15 + 45 * activity`` Hz with flicker weights ``1 + 0.5 sin(2 pi f t + phi_i)``, ``f = 2 + 6 * activity`` ``[E]``.
LIGHT_RATE_BASE_HZ: float = 15.0
LIGHT_RATE_GAIN_HZ: float = 45.0
FLICKER_F_BASE_HZ: float = 2.0
FLICKER_F_GAIN_HZ: float = 6.0
FLICKER_DEPTH: float = 0.5
#: Row 10 odor: ``80 * hill(odor; 0.3, 1.5)``; food glomeruli weight 1.0, others 0.25 ``[E]`` volume = smell of food.
ODOR_RATE_MAX_HZ: float = 80.0
ODOR_HILL_K: float = 0.3
ODOR_HILL_N: float = 1.5
ORN_FOOD_W: float = 1.0
ORN_OTHER_W: float = 0.25
FOOD_GLOMERULI: tuple[str, ...] = ("DM1", "DM4", "DM2", "VM2", "VA2", "DL1")
_FOOD_ORN_RE = re.compile(r"ORN_(DM1|DM4|DM2|VM2|VA2|DL1)")
#: Rows 11/12: ``pam 40 * up``, ``ppl1 40 * down`` ``[E]``.
REWARD_RATE_MAX_HZ: float = 40.0
PUNISH_RATE_MAX_HZ: float = 40.0
#: Row 13 dust: ``jo_groom 60 * hill(chop; 0.3, 1.5)`` ``[E]`` chop = dust on the antenna.
DUST_RATE_MAX_HZ: float = 60.0
DUST_HILL_K: float = 0.3
DUST_HILL_N: float = 1.5
#: Rows 14/15: ``grn_pher`` ``[?]`` identity / ``jo_aud`` ``[D]`` at ``100 * hill(courtship; 0.3, 1.5)``.
PHER_RATE_MAX_HZ: float = 100.0
SONG_IN_RATE_MAX_HZ: float = 100.0
COURT_HILL_K: float = 0.3
COURT_HILL_N: float = 1.5
#: Row 16 sleep: ``dfb_sleep 10 * sleep_pressure`` ``[L]`` dFB sleep-promoting neurons.
SLEEP_RATE_MAX_HZ: float = 10.0
#: Row 17 explore: ``dng100 30 * explore + 20 * sugar * (1 - feeding)`` ``[E]`` guaranteed motion (>= 7.5 Hz at baseline 0.25).
EXPLORE_RATE_GAIN_HZ: float = 30.0
EXPLORE_SUGAR_GAIN_HZ: float = 20.0
#: Row 18 compass: ``epg`` 40 Hz on the heading wedge, 0.375 (-> 15 Hz) on the neighbours, 16 wedges ``[E]``.
COMPASS_RATE_HZ: float = 40.0
COMPASS_WEDGES: int = 16
COMPASS_NEIGHBOUR_W: float = 0.375
#: Row 19 mood feedback ``[E]`` (circular by design): EUPHORIA pam 60 / flight_dn 40*euphoria; PANIC ppl1 60 / dn_freeze 20;
#: ANXIOUS ppl1 20; courtship crossing 0.5 (or pheromone poke) -> p1 80 Hz for 6 s; SLEEP x0.3; grogginess x0.5 for 2 s.
MOOD_EUPHORIA_PAM_HZ: float = 60.0
MOOD_EUPHORIA_FLIGHT_HZ: float = 40.0
MOOD_PANIC_PPL1_HZ: float = 60.0
MOOD_PANIC_FREEZE_HZ: float = 20.0
MOOD_ANXIOUS_PPL1_HZ: float = 20.0
COURT_P1_RATE_HZ: float = 80.0
COURT_P1_MS: int = 6000
COURT_CROSS_THR: float = 0.5
SLEEP_RATE_SCALE: float = 0.3
GROGGY_RATE_SCALE: float = 0.5
GROGGY_MS: int = 2000
#: Euphoria score assumed for ``flight_dn 40 * euphoria`` when the loop does not pass the mood scores ``[E]``.
EUPHORIA_DEFAULT: float = 0.7
#: Mood states that produce feedback drives.
MOOD_STATES_FEEDBACK: tuple[str, ...] = ("EUPHORIA", "PANIC", "ANXIOUS")

#: Row 20 pokes (SPEC c.17): stim -> (group, rate_hz at strength 1). Key order = ``hello.channels``.
POKE_TABLE: dict[str, tuple[str, float]] = {
    "sugar": ("grn_sugar", 150.0),
    "bitter": ("grn_bitter", 120.0),
    "loom": ("lc_loom", 150.0),
    "water": ("grn_water", 60.0),
    "dust": ("jo_groom", 80.0),
    "pheromone": ("grn_pher", 100.0),
    "sleep": ("dfb_sleep", 20.0),
    "reward": ("pam", 80.0),
    "punish": ("ppl1", 80.0),
    "explore": ("dng100", 40.0),
}

#: Row 20 channel per ``Poke.side``: the loop's tag is ``group:channel`` and the engine keeps one injection per
#: tag, so sided pokes of the same stim need distinct channels (``poke`` both, ``poke_L`` -1, ``poke_R`` +1).
POKE_CHANNELS: dict[int, str] = {0: "poke", -1: "poke_L", 1: "poke_R"}

#: Channel names of the f.2 table in row order (row 19 is ``mood`` / ``court``, row 20 ``poke`` / ``poke_L`` / ``poke_R``).
DRIVE_CHANNELS: tuple[str, ...] = (
    "sugar", "water", "bitter", "loom_ramp", "loom_tonic", "loom_aux", "flash", "freeze", "light", "odor",
    "reward", "punish", "dust", "pheromone", "song_in", "sleep", "explore", "compass", "mood", "court", "poke",
    "poke_L", "poke_R",
)


# =====================================================================================================
# helpers
# =====================================================================================================


def hill(x: float, k: float, n: float = 1.5) -> float:
    """``x^n / (k^n + x^n)`` for ``x >= 0``, else 0 (SPEC c.17). NaN -> 0, +inf -> 1 (the limit)."""
    x = float(x)
    if not (x > 0.0):  # 0, negative or NaN
        return 0.0
    if math.isinf(x):
        return 1.0
    try:
        xn = x ** n
    except OverflowError:
        return 1.0
    return xn / (k ** n + xn)


def sat(x: float) -> float:
    """``clip(x, 0, 1)``; NaN -> 0 (a non-finite input never leaves the unit range)."""
    x = float(x)
    if not (x > 0.0):
        return 0.0
    return 1.0 if x >= 1.0 else x


def relu(x: float) -> float:
    """``max(0, x)``; NaN and +-inf -> 0."""
    x = float(x)
    if not (x > 0.0) or math.isinf(x):
        return 0.0
    return x


def finite(x: object, default: float = 0.0) -> float:
    """``float(x)`` when finite, else ``default`` (None / NaN / inf / unparsable)."""
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return v if math.isfinite(v) else float(default)


def _ema_alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-dt / tau) if dt > 0.0 else 0.0


def _price_prefix(price: float, ndigits: int) -> str:
    """Decimal text of ``price`` truncated to its first ``ndigits`` significant digits (``0.000420``)."""
    mant, exp = f"{price:.15e}".split("e")
    digits = mant.replace(".", "").replace("-", "")[:ndigits]
    e = int(exp)
    if e < 0:
        return "0." + "0" * (-e - 1) + digits
    if e >= ndigits - 1:
        return digits + "0" * (e - ndigits + 1)
    return digits[: e + 1] + "." + digits[e + 1:]


def _sig_digits(price: float, ndigits: int) -> str:
    mant, _ = f"{price:.15e}".split("e")
    return mant.replace(".", "").replace("-", "")[:ndigits]


# =====================================================================================================
# Features
# =====================================================================================================


@dataclass(slots=True)
class Features:
    """The section f.1 feature record, computed every tick (all in ``[0, 1]`` unless stated).

    ``bp`` buy pressure -1..1; ``mom = tanh(chg_m5/2)``; ``tick = tanh(100 (P/P_prev - 1))``; ``val`` /
    ``activity`` / ``rug`` EMAs; ``up = sat(val)``, ``down = sat(-val)``; ``loom_side`` -1|0|+1; ``loom_t_ms``
    ms since the current ramp start (None outside the 300 ms ramp); ``candle`` up|down|flat; ``whale`` the whale
    trade of this tick (None otherwise); ``usd_ref`` the running median trade size. ``feeding`` is the decoder
    flag passed to ``update`` (needed by drive row 17; not in the SPEC field list, trailing extra).
    """

    t_ms: int = 0
    bp: float = 0.0
    mom: float = 0.0
    tick: float = 0.0
    val: float = 0.0
    activity: float = 0.0
    rug: float = 0.0
    up: float = 0.0
    down: float = 0.0
    sugar: float = 0.0
    bitter: float = 0.0
    water: float = 0.0
    looming: float = 0.0
    loom_side: int = 0
    flash: float = 0.0
    loom_episode: int = 0
    loom_t_ms: int | None = None
    sustained_loom: bool = False
    vol_norm: float = 0.0
    odor: float = 0.0
    chop: float = 0.0
    courtship: float = 0.0
    sleep_pressure: float = 0.0
    hunger: float = HUNGER_INIT
    sweet_gain: float = SWEET_GAIN_BASE + SWEET_GAIN_HUNGER * HUNGER_INIT
    explore: float = 0.0
    candle: str = "flat"
    since_last_trade_s: float = NO_TRADE_S
    whale: Trade | None = None
    usd_ref: float = USD_REF_FLOOR
    feeding: bool = False

    @property
    def any_max(self) -> float:
        """``max`` of sugar/bitter/water/looming/flash/odor/chop/courtship/sleep_pressure (SPEC d.2 ``drives.any_max``)."""
        return max(self.sugar, self.bitter, self.water, self.looming, self.flash, self.odor, self.chop,
                   self.courtship, self.sleep_pressure)

    def to_wire(self) -> dict:
        """The ``tick.drives`` object of SPEC d.2 (floats rounded to 3 dp)."""
        r = lambda x: round(float(x), 3)  # noqa: E731
        return {
            "sugar": r(self.sugar), "bitter": r(self.bitter), "water": r(self.water), "looming": r(self.looming),
            "loom_side": int(self.loom_side), "flash": r(self.flash), "odor": r(self.odor), "chop": r(self.chop),
            "courtship": r(self.courtship), "sleep_pressure": r(self.sleep_pressure), "explore": r(self.explore),
            "up": r(self.up), "down": r(self.down), "activity": r(self.activity), "hunger": r(self.hunger),
            "candle": self.candle, "any_max": r(self.any_max),
        }


# =====================================================================================================
# Poke
# =====================================================================================================


@dataclass(frozen=True)
class Poke:
    """A user poke (SPEC c.17): ``stim`` in ``POKE_TABLE``; ``strength`` 0..1; ``side`` -1|0|+1; ``until_ms`` brain ms."""

    stim: str
    strength: float
    side: int
    until_ms: int


# =====================================================================================================
# FeatureExtractor
# =====================================================================================================


class FeatureExtractor:
    """Market -> ``Features`` (SPEC c.17 / f.1).

    State kept between ticks: EMAs (``val, activity, vol_norm``), ``rug``, ``water_imp``, trade impulses
    (``imp_sugar, imp_loom``), ``flash_until``, hunger, courtship, sleep pressure ``z``, ``odor_rise_t``, session
    ATH, ``liq_start / liq_prev / price_prev``, ``ema_vol_1h``, the last 200 trade sizes (``usd_ref`` median,
    floor 20), ``last_trade_wall``, the looming episode (``loom_episode, loom_ramp_start_ms, loom_last_sell_ms,
    loom_side``), ``sustained_timer``, ``consecutive_bull_snaps`` and a 90 s price ring.

    The feature clock ``t_ms`` is the accumulated wall ``dt_s`` in ms (the loop may pass its brain ``t_ms``
    through the optional keyword instead). Pokes live on the **brain** clock (``Poke.until_ms``): a bound
    ``SensoryEncoder`` reports it every tick through ``sync_brain_clock`` so pheromone pins and poke
    idempotency expire at the right time even when ``speed < 1``. ``note_poke`` only *queues* a poke; the
    effect is applied at the start of the next ``update`` so a poke's loom ramp starts at ``loom_t_ms == 0``
    exactly like a market sell wall. Events (``easter_egg``, ``whale``) are
    queued for ``drain_events``. A source switch of the snapshots (``snap.source`` changes) resets the
    per-source references (``price_prev, ath, liq_start, liq_prev, ema_vol_1h, consecutive_bull_snaps,
    candle, price ring``) and keeps the EMAs / impulses / hunger / sleep pressure.
    """

    #: Weak registry of live extractors (most recent last) for ``SensoryEncoder`` auto-binding (same ``settings``).
    _registry: list["weakref.ReferenceType[FeatureExtractor]"] = []
    _registry_version: int = 0
    _REGISTRY_MAX: int = 8

    @classmethod
    def _register(cls, fx: "FeatureExtractor") -> None:
        alive = [r for r in cls._registry if r() is not None]
        cls._registry = alive[-(cls._REGISTRY_MAX - 1):] + [weakref.ref(fx)]
        cls._registry_version += 1

    @classmethod
    def live_for(cls, settings: object) -> list["FeatureExtractor"]:
        """Every live extractor built with exactly this ``settings`` object (oldest first)."""
        out: list["FeatureExtractor"] = []
        for r in cls._registry:
            fx = r()
            if fx is not None and fx.settings is settings:
                out.append(fx)
        return out

    @classmethod
    def bindable_for(cls, settings: object) -> "FeatureExtractor | None":
        """The **sole** live extractor built with exactly this ``settings`` object, else ``None``.

        Auto-binding is only safe when the pairing is unambiguous: two loops in one process sharing one
        ``Settings`` instance must not cross-wire, so with more than one candidate nothing is bound and the
        caller is expected to pass ``features=`` / call ``bind()`` explicitly.
        """
        live = cls.live_for(settings)
        return live[0] if len(live) == 1 else None

    def __init__(self, settings: "Settings", seed: int) -> None:
        self.settings = settings
        self.explore_baseline: float = float(getattr(settings, "explore_baseline", 0.25))
        self.easter_eggs: bool = bool(getattr(settings, "easter_eggs", True))
        self.seed = int(seed)
        child = np.random.SeedSequence(self.seed).spawn(SEED_CHILD_ENCODER + 1)[SEED_CHILD_ENCODER]
        self.rng = np.random.default_rng(child.spawn(2)[0])
        # clocks ----------------------------------------------------------------------------
        self._t_s: float = 0.0
        self._t_ms: int = 0
        self._brain_t_ms: int | None = None
        self._last_now: float = 0.0
        self._last_snap_wall: float | None = None
        self._last_seq: int | None = None
        self._source: str | None = None
        # snapshot-level state -----------------------------------------------------------------
        self.val_snap = 0.0
        self.act_snap = 0.0
        self.vol_snap = 0.0
        self.rug_snap = 0.0
        self.bp = 0.0
        self.mom = 0.0
        self.tick = 0.0
        self.chg_m5 = 0.0
        self.candle = "flat"
        self.price_prev: float | None = None
        self.liq_start: float | None = None
        self.liq_prev: float | None = None
        self.ema_vol_1h: float | None = None
        self.ath: float | None = None
        self.consecutive_bull_snaps = 0
        self.price_ring: deque[tuple[float, float]] = deque()
        # tick-level state -----------------------------------------------------------------------
        self.val = 0.0
        self.activity = 0.0
        self.vol_norm = 0.0
        self.rug = 0.0
        self.water_imp = 0.0
        self.imp_sugar = 0.0
        self.imp_loom = 0.0
        self.flash_until: int = -1
        self.hunger = HUNGER_INIT
        self.courtship = 0.0
        self.z = 0.0
        self.odor_rise_t = -math.inf
        self._odor_raw_prev = 0.0
        self.trade_sizes: deque[float] = deque(maxlen=TRADE_SIZES_MAXLEN)
        self.usd_ref = USD_REF_FLOOR
        self.last_trade_wall: float | None = None
        self.loom_episode = 0
        self.loom_ramp_start_ms: int | None = None
        self.loom_last_sell_ms: int | None = None
        self.loom_side = 0
        self.sustained_timer = 0.0
        self._pher_until_ms: int = -1
        self._seen_pokes: dict[tuple, int] = {}
        self._pending_pokes: list[Poke] = []
        # easter eggs / events ---------------------------------------------------------------------
        self._last_clock_key: tuple | None = None
        self._last_egg: tuple[str, float] | None = None
        self._events: list[dict] = []
        self.last: Features = Features()
        FeatureExtractor._register(self)

    # ------------------------------------------------------------------ public helpers

    @property
    def t_ms(self) -> int:
        """Feature clock (ms)."""
        return self._t_ms

    @property
    def brain_t_ms(self) -> int:
        """The clock pokes are compared against: the last synced brain clock, else the feature clock."""
        return self._t_ms if self._brain_t_ms is None else self._brain_t_ms

    @property
    def source(self) -> str | None:
        """``source`` of the last processed snapshot (None before the first)."""
        return self._source

    def sync_brain_clock(self, t_ms: int) -> None:
        """Report the engine's brain clock (called by the bound ``SensoryEncoder`` every tick)."""
        self._brain_t_ms = int(t_ms)

    def drain_events(self) -> list[dict]:
        """Queued ``{"kind": "easter_egg"|"whale", "data": {...}}`` events (SPEC d.8) since the last call."""
        ev, self._events = self._events, []
        return ev

    def note_poke(self, poke: Poke) -> None:
        """Queue a poke so it shapes the features (SPEC f.2): a ``loom`` poke starts a ramp episode on
        ``poke.side`` (exactly like a sell wall); a ``pheromone`` poke pins courtship to 1.0 while it is active.
        Idempotent per poke.

        The effect is applied at the **start of the next** ``update`` (or immediately when the poke is handed to
        ``update(pokes=...)`` itself), never against the clock of an ``update`` that has already run: the loop
        order of c.27 is update (step 3) then encode (step 4), so a poke forwarded by the encoder would
        otherwise stamp ``loom_ramp_start_ms`` one tick in the past and the ramp would never be sampled at
        ``loom_t_ms == 0`` - a poke must look exactly like a sell wall (f.2 ``POKE_TABLE`` note).
        """
        key = (poke.stim, float(poke.strength), int(poke.side), int(poke.until_ms))
        if key in self._seen_pokes:
            return
        self._seen_pokes[key] = int(poke.until_ms)
        self._pending_pokes.append(poke)

    def _apply_pending_pokes(self, t_ms: int) -> None:
        """Apply every queued poke against ``t_ms`` (the clock of the tick being computed)."""
        if not self._pending_pokes:
            return
        pending, self._pending_pokes = self._pending_pokes, []
        for poke in pending:
            self._apply_poke(poke, t_ms)

    def _apply_poke(self, poke: Poke, t_ms: int) -> None:
        if poke.stim == "loom":
            strength = sat(poke.strength)
            if not self._episode_active(t_ms):
                self.loom_episode += 1
            self.loom_side = int(poke.side) if poke.side in (-1, 1) else (self.loom_side or self._pick_side(None, None))
            self.loom_ramp_start_ms = t_ms
            self.loom_last_sell_ms = t_ms
            self.imp_loom += IMP_LOOM_PER_TRADE * TRADE_S_CAP * strength
        elif poke.stim == "pheromone":
            self.courtship = 1.0
            self._pher_until_ms = max(self._pher_until_ms, int(poke.until_ms))

    def easter_egg(self, snap: MarketSnapshot | None, now: float) -> str | None:
        """``FLY_EASTER_EGGS``: ``buys_m5 in {69, 420}``, leading significant digits of the price ``420`` / ``69``,
        local time ``04:20`` / ``16:20`` (once per minute) -> reason string of SPEC d.8, else None."""
        if not self.easter_eggs:
            return None
        if snap is not None:
            buys = int(snap.buys_m5)
            if buys in EASTER_BUYS:
                return f"buys_m5 == {buys}"
            price = snap.price
            if price is not None and price > 0.0 and math.isfinite(price):
                if _sig_digits(price, 3) == "420":
                    return f"price {_price_prefix(price, 3)}"
                if _sig_digits(price, 2) == "69":
                    return f"price {_price_prefix(price, 2)}"
        lt = time.localtime(now)
        hhmm = f"{lt.tm_hour:02d}:{lt.tm_min:02d}"
        if hhmm in EASTER_CLOCKS:
            key = (lt.tm_year, lt.tm_yday, lt.tm_hour, lt.tm_min)
            if key != self._last_clock_key:
                self._last_clock_key = key
                return f"clock {hhmm}"
        return None

    # ------------------------------------------------------------------ internals

    def _episode_active(self, t_ms: int) -> bool:
        return self.loom_last_sell_ms is not None and t_ms - self.loom_last_sell_ms <= LOOM_EPISODE_GAP_MS

    def _pick_side(self, snap: MarketSnapshot | None, tr: Trade | None) -> int:
        """Looming side: seeded coin for the sim, ``ts`` parity for DexScreener trades (SPEC f.1)."""
        if snap is not None and snap.source == DEX_SOURCE and tr is not None:
            return +1 if int(tr.ts * 1000) % 2 else -1
        return int(self.rng.choice([-1, 1]))

    def _reset_source_state(self, new_source: str) -> None:
        """A different market source (``sim(fallback)`` <-> ``dexscreener`` <-> ``sim``) has its own price and
        liquidity scale: forget the references that compare against the previous snapshot / session (previous
        price, session ATH, liquidity start, volume EMA, bull streak, candle, price ring). EMAs, impulses, hunger,
        sleep pressure and the looming episode carry over."""
        log.info("features: market source %s -> %s: resetting per-source references", self._source, new_source)
        self.price_prev = None
        self.ath = None
        self.liq_start = None
        self.liq_prev = None
        self.ema_vol_1h = None
        self.consecutive_bull_snaps = 0
        self.candle = "flat"
        self.tick = 0.0
        self.price_ring.clear()
        self._last_snap_wall = None

    def _on_snapshot(self, snap: MarketSnapshot, now: float) -> None:
        source = str(snap.source)
        if self._source is not None and source != self._source:
            self._reset_source_state(source)
        self._source = source
        price = finite(snap.price, 0.0)
        price = price if price > 0.0 else None
        p_prev = self.price_prev
        buys = max(0, int(finite(snap.buys_m5, 0.0)))
        sells = max(0, int(finite(snap.sells_m5, 0.0)))
        chg_m5 = finite(snap.chg_m5, 0.0)
        n = buys + sells
        self.bp = (buys - sells) / (n + BP_DENOM_OFFSET)
        self.mom = math.tanh(chg_m5 / MOM_SCALE_PCT)
        self.tick = math.tanh(TICK_SCALE * (price / p_prev - 1.0)) if (price and p_prev) else 0.0
        self.val_snap = max(-1.0, min(1.0, VAL_W_BP * self.bp + VAL_W_MOM * self.mom + VAL_W_TICK * self.tick))
        self.act_snap = sat(math.log1p(n) / math.log1p(ACT_LOG_REF))
        liq = finite(snap.liq_usd, 0.0)
        liq = liq if liq > 0.0 else None
        if liq is not None and self.liq_start is None:
            self.liq_start = liq
        self.rug_snap = 1.0 if (chg_m5 <= RUG_CHG_M5_PCT or
                                (liq is not None and self.liq_start and liq < RUG_LIQ_FRAC * self.liq_start)) else 0.0
        vol_m5 = max(0.0, finite(snap.vol_m5, 0.0))
        if self.ema_vol_1h is None:
            self.ema_vol_1h = vol_m5
        else:
            dts = max(0.0, now - (self._last_snap_wall if self._last_snap_wall is not None else now))
            self.ema_vol_1h += (vol_m5 - self.ema_vol_1h) * _ema_alpha(dts, EMA_VOL_TAU_S)
        self.vol_snap = sat(vol_m5 / (VOL_NORM_MULT * self.ema_vol_1h + VOL_NORM_OFFSET))
        if liq is not None and self.liq_prev and liq > self.liq_prev:
            self.water_imp += sat((liq / self.liq_prev - 1.0) / WATER_LIQ_STEP)
        if price and p_prev:
            self.candle = "up" if price > (1.0 + CANDLE_EPS) * p_prev else "down" if price < (1.0 - CANDLE_EPS) * p_prev else "flat"
        else:
            self.candle = "flat"
        if price:
            if self.ath is not None and price >= COURT_ATH_MULT * self.ath:
                self.courtship = 1.0
            self.ath = price if self.ath is None else max(self.ath, price)
            self.price_prev = price
            self.price_ring.append((now, price))
            while self.price_ring and now - self.price_ring[0][0] > PRICE_RING_S:
                self.price_ring.popleft()
        if buys >= BULL_BUY_RATIO * sells and chg_m5 > BULL_CHG_M5_PCT:
            self.consecutive_bull_snaps += 1
        else:
            self.consecutive_bull_snaps = 0
        if self.consecutive_bull_snaps >= COURT_BULL_SNAPS:
            self.courtship = 1.0
        if liq is not None:
            self.liq_prev = liq
        self.chg_m5 = chg_m5
        self._last_seq = int(snap.seq)
        self._last_snap_wall = now

    def _on_trade(self, tr: Trade, snap: MarketSnapshot | None, t_ms: int, now: float) -> Trade | None:
        usd = max(0.0, finite(tr.usd, 0.0))
        self.trade_sizes.append(usd)
        self.usd_ref = max(USD_REF_FLOOR, float(np.median(np.fromiter(self.trade_sizes, dtype=np.float64))))
        self.last_trade_wall = finite(tr.ts, now)
        s = min(TRADE_S_CAP, usd / self.usd_ref)
        is_sell = tr.kind == "sell"
        if is_sell:
            self.imp_loom += IMP_LOOM_PER_TRADE * s
            if not self._episode_active(t_ms):
                self.loom_episode += 1
                self.loom_side = self._pick_side(snap, tr)
            self.loom_last_sell_ms = t_ms
            self.loom_ramp_start_ms = t_ms
        else:
            self.imp_sugar += IMP_SUGAR_PER_TRADE * s
        if usd >= WHALE_RATIO * self.usd_ref:
            if is_sell:
                self.flash_until = t_ms + FLASH_MS
            self._events.append({"kind": "whale", "data": {"kind": tr.kind, "usd": round(usd, 2),
                                                           "ratio": round(usd / self.usd_ref, 2)}})
            return tr
        return None

    # ------------------------------------------------------------------ update

    def update(self, snap: MarketSnapshot | None, trades: list[Trade], now: float, dt_s: float, mood: str,
               feeding: bool, mn9_hz: float, *, pokes: Sequence[Poke] | None = None,
               t_ms: int | None = None) -> Features:
        """Formulas of SPEC section f.1. ``dt_s`` is the wall tick in seconds; ``now`` wall seconds; ``mood`` the
        current mood name; ``feeding`` / ``mn9_hz`` from the decoder (hunger). ``pokes`` (new pokes of this
        tick) and ``t_ms`` (brain clock override) are optional extras for the loop."""
        dt = max(0.0, finite(dt_s, 0.0))
        self._t_s += dt
        if t_ms is None:
            t_ms = int(round(self._t_s * 1000.0))
        else:
            t_ms = int(t_ms)
        self._t_ms = t_ms
        t = self._t_s
        now = finite(now, self._last_now)
        self._last_now = now
        mn9 = max(0.0, finite(mn9_hz, 0.0))

        # pokes forwarded by the encoder after the previous update() apply with *this* tick's clock, so their
        # loom ramp is sampled from loom_t_ms == 0 exactly like a market sell wall; pokes handed to update()
        # itself apply in the same tick.
        self._apply_pending_pokes(t_ms)
        if pokes:
            for p in pokes:
                self.note_poke(p)
            self._apply_pending_pokes(t_ms)
        brain_t = self.brain_t_ms
        if self._seen_pokes:
            self._seen_pokes = {k: u for k, u in self._seen_pokes.items() if u > brain_t}

        # --- snapshot -------------------------------------------------------------------------
        if snap is not None and (self._last_seq is None or int(snap.seq) != self._last_seq):
            self._on_snapshot(snap, now)

        # --- EMAs / decays ------------------------------------------------------------------------
        self.val += (self.val_snap - self.val) * _ema_alpha(dt, TAU_VAL_S)
        self.activity += (self.act_snap - self.activity) * _ema_alpha(dt, TAU_ACT_S)
        self.vol_norm += (self.vol_snap - self.vol_norm) * _ema_alpha(dt, TAU_VOL_S)
        self.rug = max(self.rug_snap, self.rug * math.exp(-dt / TAU_RUG_S))
        self.water_imp *= math.exp(-dt / TAU_WATER_S)

        # --- trades -----------------------------------------------------------------------------
        whale: Trade | None = None
        if trades:
            for tr in sorted(trades, key=lambda x: finite(x.ts, now)):
                w = self._on_trade(tr, snap, t_ms, now)
                if w is not None:
                    whale = w
        decay_imp = math.exp(-dt / TAU_IMP_S)
        self.imp_sugar *= decay_imp
        self.imp_loom *= decay_imp

        # --- channels ---------------------------------------------------------------------------
        val = self.val
        up = sat(val)
        down = sat(-val)
        sugar = sat(relu(val) + self.imp_sugar)
        bitter = sat(BITTER_W_VAL * relu(-val) + BITTER_W_RUG * self.rug)
        looming = sat(LOOM_VAL_GAIN * relu(-val) + self.imp_loom)
        flash = 1.0 if t_ms < self.flash_until else 0.0
        if self.loom_ramp_start_ms is not None and 0 <= t_ms - self.loom_ramp_start_ms <= LOOM_RAMP_MS:
            loom_t_ms: int | None = t_ms - self.loom_ramp_start_ms
        else:
            loom_t_ms = None
        if not self._episode_active(t_ms):
            self.loom_side = 0
        if looming >= SUSTAINED_LOOM_THR:
            self.sustained_timer += dt
        else:
            self.sustained_timer = 0.0
        sustained = self.sustained_timer >= SUSTAINED_LOOM_S
        water = sat(self.water_imp)
        odor_raw = sat(ODOR_W_ACT * self.activity + ODOR_W_VOL * self.vol_norm)
        if odor_raw - self._odor_raw_prev > ODOR_RISE_STEP:
            self.odor_rise_t = t
        self._odor_raw_prev = odor_raw
        odor = odor_raw * (ODOR_ADAPT_FLOOR + ODOR_ADAPT_SPAN * math.exp(-(t - self.odor_rise_t) / ODOR_ADAPT_TAU_S))
        chop = sat(self.activity * (1.0 - min(1.0, abs(self.chg_m5) / CHOP_CHG_SCALE_PCT)))

        # --- courtship (easter eggs, pheromone poke) ----------------------------------------------
        reason = self.easter_egg(snap, now)
        if reason is not None:
            self.courtship = 1.0
            last = self._last_egg
            if last is None or last[0] != reason or t - last[1] >= EASTER_EVENT_REPEAT_S:
                self._events.append({"kind": "easter_egg", "data": {"reason": reason}})
                self._last_egg = (reason, t)
        if self._pher_until_ms > brain_t:
            self.courtship = 1.0
        courtship = self.courtship
        self.courtship *= math.exp(-dt / COURT_TAU_S)

        # --- sleep pressure, hunger, explore ----------------------------------------------------
        self.z += dt / SLEEP_RISE_S if self.activity < SLEEP_ACT_THR else -dt / SLEEP_FALL_S
        self.z = sat(self.z)
        self.hunger += dt / HUNGER_RISE_S
        if feeding:
            self.hunger -= HUNGER_FEED_RATE * sat(mn9 / HUNGER_MN9_REF_HZ) * dt
        self.hunger = sat(self.hunger)
        sweet_gain = SWEET_GAIN_BASE + SWEET_GAIN_HUNGER * self.hunger
        explore = 0.0 if mood == "SLEEP" else self.explore_baseline + EXPLORE_ACT_GAIN * self.activity
        # clamped at 0: a trade stamp from the future (surrogate stamps, a clock step) must not make the
        # "seconds since the last trade" negative (f.1 reads it as an age)
        since = max(0.0, now - self.last_trade_wall) if self.last_trade_wall is not None else NO_TRADE_S

        f = Features(
            t_ms=t_ms, bp=self.bp, mom=self.mom, tick=self.tick, val=val, activity=self.activity, rug=self.rug,
            up=up, down=down, sugar=sugar, bitter=bitter, water=water, looming=looming, loom_side=int(self.loom_side),
            flash=flash, loom_episode=int(self.loom_episode), loom_t_ms=loom_t_ms, sustained_loom=bool(sustained),
            vol_norm=self.vol_norm, odor=odor, chop=chop, courtship=courtship, sleep_pressure=self.z,
            hunger=self.hunger, sweet_gain=sweet_gain, explore=explore, candle=self.candle,
            since_last_trade_s=float(since), whale=whale, usd_ref=self.usd_ref, feeding=bool(feeding),
        )
        self.last = f
        return f


# =====================================================================================================
# Drives
# =====================================================================================================


@dataclass(frozen=True)
class ChannelDrive(Drive):
    """A ``Drive`` (SPEC c.10) that also carries its f.2 ``channel`` (the loop's tag is ``group + ':' + channel``)
    and, in ``FLY_DRIVE_MODE=current``, ``current_mv = current_from_rate(rate_hz)``."""

    channel: str = ""
    current_mv: float | None = None

    @property
    def tag(self) -> str:
        return f"{self.group}:{self.channel}"


class SensoryEncoder:
    """``Features`` -> ``list[Drive]`` (SPEC c.17 / f.2).

    Precomputes the EPG wedge of every EPG neuron (``floor(16 * rank_within_side / n_side)``), one fixed
    flicker phase ``phi_i ~ U(0, 2 pi)`` per photoreceptor (seeded) and the ORN weights of the six food
    glomeruli (``ORN_(DM1|DM4|DM2|VM2|VA2|DL1)`` -> 1.0, others 0.25). ``encode`` returns the complete drive
    list for the next tick; the loop injects each with ``duration = tick brain-ms`` and ``tag = group:channel``
    (``apply`` does exactly that). Mood feedback only when ``settings.mood_feedback``; the explore baseline
    only when ``settings.explore_baseline > 0``.

    Poke rows use one channel per side (``poke`` / ``poke_L`` / ``poke_R``) so that a left and a right poke of
    the same stim keep distinct tags (the engine replaces an injection with the same tag). Pass ``features=``
    (or call ``bind()``) to wire the extractor that loom / pheromone pokes and the brain clock are forwarded
    to; without it the encoder binds to the single live ``FeatureExtractor`` built with the same ``settings``
    object and refuses to guess when there are several (module docstring). A non-finite ``heading`` reuses the
    last good compass wedge.
    """

    POKE_TABLE: dict[str, tuple[str, float]] = POKE_TABLE
    #: Every group key the encoder can emit (all are keys of ``Connectome.groups``, SPEC c.4).
    DRIVE_GROUPS: tuple[str, ...] = (
        "grn_sugar", "grn_water", "grn_bitter", "lc_loom", "lc_loom2", "lc_freeze", "photoreceptor", "orn",
        "pam", "ppl1", "jo_groom", "grn_pher", "jo_aud", "dfb_sleep", "dng100", "epg", "flight_dn", "dn_freeze", "p1",
    )

    def __init__(self, conn: "Connectome", settings: "Settings", seed: int,
                 features: FeatureExtractor | None = None) -> None:
        self.conn = conn
        self.settings = settings
        self.seed = int(seed)
        self.mood_feedback: bool = bool(getattr(settings, "mood_feedback", True))
        self.explore_baseline: float = float(getattr(settings, "explore_baseline", 0.25))
        self.drive_mode: str = str(getattr(settings, "drive_mode", "poisson"))
        if self.drive_mode not in ("poisson", "current"):
            raise ValueError(f"drive_mode must be poisson|current, got {self.drive_mode!r}")
        self.params = LIFParams()
        self.features: FeatureExtractor | None = features
        #: How ``self.features`` was obtained: ``"explicit"`` (``features=`` / ``bind()``), ``"auto"`` or ``None``.
        self.bound_via: str | None = "explicit" if features is not None else None
        self._autobind = features is None
        self._registry_seen = -1
        self._last_heading = 0.0
        self._maybe_autobind()
        child = np.random.SeedSequence(self.seed).spawn(SEED_CHILD_ENCODER + 1)[SEED_CHILD_ENCODER]
        rng = np.random.default_rng(child.spawn(2)[1])
        groups = conn.groups
        empty = np.zeros(0, dtype=np.int32)
        # EPG wedges ------------------------------------------------------------------------------
        epg = np.asarray(groups.get("epg", empty), dtype=np.int64)
        self.epg_wedge = np.zeros(epg.shape[0], dtype=np.int16)
        if epg.shape[0]:
            sides = np.asarray(conn.side[epg], dtype=np.int64)
            for s in (-1, 1, 0):
                pos = np.flatnonzero(sides == s)
                n_side = int(pos.shape[0])
                for rank, j in enumerate(pos):
                    self.epg_wedge[j] = int(math.floor(COMPASS_WEDGES * rank / n_side))
        # photoreceptor flicker phases --------------------------------------------------------------
        pr = np.asarray(groups.get("photoreceptor", empty), dtype=np.int64)
        self.pr_phase = rng.uniform(0.0, 2.0 * math.pi, pr.shape[0]).astype(np.float64)
        # ORN weights ----------------------------------------------------------------------------------
        orn = np.asarray(groups.get("orn", empty), dtype=np.int64)
        self.orn_w = np.full(orn.shape[0], ORN_OTHER_W, dtype=np.float32)
        types = conn.types
        for j, i in enumerate(orn):
            if _FOOD_ORN_RE.fullmatch(types[int(conn.type_idx[i])]) is not None:
                self.orn_w[j] = ORN_FOOD_W
        self.n_food_orn = int((self.orn_w == ORN_FOOD_W).sum())
        # per-tick state ------------------------------------------------------------------------------
        self._active_pokes: list[Poke] = []
        self._prev_mood: str | None = None
        self._groggy_until_ms: int = -1
        self._court_until_ms: int = -1
        self._prev_courtship: float = 0.0
        self.last_drives: list[ChannelDrive] = []

    # ------------------------------------------------------------------ helpers

    def bind(self, features: FeatureExtractor | None) -> None:
        """Attach the ``FeatureExtractor`` so that loom / pheromone pokes and the brain clock are forwarded
        (``note_poke`` / ``sync_brain_clock``). ``None`` detaches and disables auto-binding."""
        self.features = features
        self.bound_via = "explicit" if features is not None else None
        self._autobind = False

    def _maybe_autobind(self) -> None:
        """Bind to the one live ``FeatureExtractor`` constructed with the same ``settings`` object (cheap: only
        re-scans when a new extractor was registered). Ambiguous (two candidates) -> stay unbound and warn."""
        if self.features is not None or not self._autobind:
            return
        if FeatureExtractor._registry_version == self._registry_seen:
            return
        self._registry_seen = FeatureExtractor._registry_version
        fx = FeatureExtractor.bindable_for(self.settings)
        if fx is not None:
            self.features = fx
            self.bound_via = "auto"
            log.info("encoder: bound to the FeatureExtractor built with the same settings object "
                     "(loom / pheromone pokes and the brain clock are forwarded to it)")
        elif len(FeatureExtractor.live_for(self.settings)) > 1:
            log.warning("encoder: %d live FeatureExtractors share this Settings object - not auto-binding; pass "
                        "features= or call encoder.bind(features) so pokes reach the right extractor",
                        len(FeatureExtractor.live_for(self.settings)))

    @staticmethod
    def tag_of(drive: Drive, channel: str | None = None) -> str:
        """``group + ':' + channel`` (SPEC c.17)."""
        ch = channel if channel is not None else getattr(drive, "channel", "")
        return f"{drive.group}:{ch}"

    @property
    def active_pokes(self) -> list[Poke]:
        return list(self._active_pokes)

    def wedge_weights(self, heading: float) -> np.ndarray:
        """float64[16]: 1.0 on wedge ``k = floor(((heading mod 2 pi) / 2 pi) * 16)``, 0.375 on ``k +- 1``, else 0.
        A non-finite ``heading`` (NaN / inf) reuses the last finite heading (0 before any)."""
        hf = finite(heading, math.nan)
        if math.isnan(hf):
            hf = self._last_heading
        else:
            self._last_heading = hf
        h = hf % (2.0 * math.pi)
        k = int(math.floor(h / (2.0 * math.pi) * COMPASS_WEDGES)) % COMPASS_WEDGES
        w = np.zeros(COMPASS_WEDGES, dtype=np.float64)
        w[k] = 1.0
        w[(k - 1) % COMPASS_WEDGES] = COMPASS_NEIGHBOUR_W
        w[(k + 1) % COMPASS_WEDGES] = COMPASS_NEIGHBOUR_W
        return w

    def compass_weights(self, heading: float) -> np.ndarray:
        """Per-EPG-cell weights (``len == len(groups['epg'])``) for the compass row."""
        return self.wedge_weights(heading)[self.epg_wedge].astype(np.float32)

    def flicker_weights(self, activity: float, t_s: float) -> np.ndarray:
        """Per-photoreceptor ``1 + 0.5 sin(2 pi f t + phi_i)`` with ``f = 2 + 6 * activity`` Hz."""
        f_flick = FLICKER_F_BASE_HZ + FLICKER_F_GAIN_HZ * sat(activity)
        return (1.0 + FLICKER_DEPTH * np.sin(2.0 * math.pi * f_flick * t_s + self.pr_phase)).astype(np.float32)

    def _mk(self, channel: str, group: str, rate_hz: float, recruit: float = 1.0, side: int = 0, episode: int = 0,
            weights: np.ndarray | None = None) -> ChannelDrive:
        rate = float(rate_hz)
        rec = max(0.0, min(1.0, float(recruit)))
        if self.drive_mode == "current":
            return ChannelDrive(group=group, rate_hz=rate, recruit=rec, side=int(side), episode=int(episode),
                                mode="current", weights=weights, channel=channel,
                                current_mv=current_from_rate(rate, self.params))
        return ChannelDrive(group=group, rate_hz=rate, recruit=rec, side=int(side), episode=int(episode),
                            mode="poisson", weights=weights, channel=channel, current_mv=None)

    # ------------------------------------------------------------------ encode

    def encode(self, f: Features, heading: float, mood: str, pokes: Iterable[Poke], t_ms: int,
               scores: Mapping[str, float] | None = None) -> list[Drive]:
        """SPEC section f.2: the complete list of Drives for the next tick (rows with ``rate_hz < 1`` skipped).
        ``scores`` (optional) carries the mood scores (``euphoria``) for the ``flight_dn`` feedback row."""
        t_ms = int(t_ms)
        t = t_ms / 1000.0
        self._maybe_autobind()
        # --- pokes: remember new ones, drop expired -----------------------------------------------
        new: list[Poke] = []
        for p in pokes or ():
            if p.until_ms > t_ms and p not in self._active_pokes and p not in new:
                new.append(p)
        self._active_pokes = [p for p in self._active_pokes if p.until_ms > t_ms] + new
        if self.features is not None:
            self.features.sync_brain_clock(t_ms)
            for p in new:
                self.features.note_poke(p)
        pher_poke = any(p.stim == "pheromone" for p in new)
        # --- mood bookkeeping ---------------------------------------------------------------------
        if self._prev_mood == "SLEEP" and mood != "SLEEP":
            self._groggy_until_ms = t_ms + GROGGY_MS
        self._prev_mood = mood
        # SLEEP: every row except the mood feedback x0.3; grogginess (2 s after leaving SLEEP): ALL rates x0.5
        groggy = GROGGY_RATE_SCALE if (mood != "SLEEP" and t_ms < self._groggy_until_ms) else 1.0
        scale = SLEEP_RATE_SCALE if mood == "SLEEP" else groggy
        # courtship upward crossing of 0.5 (or a pheromone poke) -> p1 80 Hz for 6 s
        if (self._prev_courtship < COURT_CROSS_THR <= f.courtship) or pher_poke:
            self._court_until_ms = t_ms + COURT_P1_MS
        self._prev_courtship = float(f.courtship)

        rows: list[tuple[str, str, float, float, int, int, np.ndarray | None]] = []
        add = rows.append
        sugar = sat(f.sugar)
        looming = sat(f.looming)
        # 1 sugar
        add(("sugar", "grn_sugar", SUGAR_RATE_MAX_HZ * hill(sat(f.sweet_gain * sugar), SUGAR_HILL_K, SUGAR_HILL_N),
             GRN_RECRUIT_BASE + GRN_RECRUIT_GAIN * sugar, 0, 0, None))
        # 2 water
        add(("water", "grn_water", WATER_RATE_MAX_HZ * sat(f.water), 1.0, 0, 0, None))
        # 3 bitter
        bitter = sat(f.bitter)
        add(("bitter", "grn_bitter", BITTER_RATE_MAX_HZ * hill(bitter, BITTER_HILL_K, BITTER_HILL_N),
             GRN_RECRUIT_BASE + GRN_RECRUIT_GAIN * bitter, 0, 0, None))
        # 4 loom_ramp
        r_ramp = 0.0
        if f.loom_t_ms is not None:
            r_ramp = LOOM_RAMP_RATE_MAX_HZ * looming * (min(float(f.loom_t_ms), LOOM_RAMP_MS) / LOOM_RAMP_MS) ** 2
            add(("loom_ramp", "lc_loom", r_ramp, LOOM_RECRUIT, f.loom_side, f.loom_episode, None))
        # 5 loom_tonic
        r_tonic = LOOM_TONIC_RATE_MAX_HZ * relu((looming - LOOM_TONIC_THR) / LOOM_TONIC_SPAN) ** 2
        add(("loom_tonic", "lc_loom", r_tonic, 1.0, f.loom_side, 0, None))
        # 6 loom_aux
        add(("loom_aux", "lc_loom2", LOOM_AUX_FACTOR * (r_ramp + r_tonic), LOOM_RECRUIT, f.loom_side, f.loom_episode, None))
        # 7 flash
        if f.flash > 0.0:
            add(("flash", "lc_loom", FLASH_RATE_HZ, 1.0, 0, f.loom_episode, None))
        # 8 freeze
        if f.sustained_loom:
            add(("freeze", "lc_freeze", FREEZE_RATE_MAX_HZ * hill(looming, FREEZE_HILL_K, FREEZE_HILL_N), 1.0,
                 f.loom_side, 0, None))
        # 9 light
        activity = sat(f.activity)
        add(("light", "photoreceptor", LIGHT_RATE_BASE_HZ + LIGHT_RATE_GAIN_HZ * activity, 1.0, 0, 0,
             self.flicker_weights(activity, t)))
        # 10 odor
        add(("odor", "orn", ODOR_RATE_MAX_HZ * hill(sat(f.odor), ODOR_HILL_K, ODOR_HILL_N), 1.0, 0, 0, self.orn_w))
        # 11 reward / 12 punish
        add(("reward", "pam", REWARD_RATE_MAX_HZ * sat(f.up), 1.0, 0, 0, None))
        add(("punish", "ppl1", PUNISH_RATE_MAX_HZ * sat(f.down), 1.0, 0, 0, None))
        # 13 dust
        add(("dust", "jo_groom", DUST_RATE_MAX_HZ * hill(sat(f.chop), DUST_HILL_K, DUST_HILL_N), 1.0, 0, 0, None))
        # 14 pheromone / 15 song_in
        court = hill(sat(f.courtship), COURT_HILL_K, COURT_HILL_N)
        add(("pheromone", "grn_pher", PHER_RATE_MAX_HZ * court, 1.0, 0, 0, None))
        add(("song_in", "jo_aud", SONG_IN_RATE_MAX_HZ * court, 1.0, 0, 0, None))
        # 16 sleep
        add(("sleep", "dfb_sleep", SLEEP_RATE_MAX_HZ * sat(f.sleep_pressure), 1.0, 0, 0, None))
        # 17 explore (guaranteed motion). c.17 ("explore baseline only when settings.explore_baseline > 0") and
        # the f.2 table ("0 when explore_baseline = 0") drop the whole row, the sugar term included - even
        # though f.1 still computes explore = explore_baseline + 0.25*activity (reported as a spec issue; the
        # explicit statements and the normative test name test_explore_baseline_zero_disables win).
        if self.explore_baseline > 0.0:
            add(("explore", "dng100", EXPLORE_RATE_GAIN_HZ * max(0.0, f.explore)
                 + EXPLORE_SUGAR_GAIN_HZ * sugar * (0.0 if f.feeding else 1.0), 1.0, 0, 0, None))
        # 18 compass
        add(("compass", "epg", COMPASS_RATE_HZ, 1.0, 0, 0, self.compass_weights(heading)))

        out: list[ChannelDrive] = []
        for ch, group, rate, recruit, side, episode, w in rows:
            rate *= scale
            if rate < MIN_DRIVE_HZ:
                continue
            out.append(self._mk(ch, group, rate, recruit, side, episode, w))

        # 19 mood feedback ([E] circular by design): the EUPHORIA / PANIC / ANXIOUS rows belong to moods that
        # cannot coexist with SLEEP, so ``scale`` is just the grogginess factor for them; the p1 court row *can*
        # be alive in SLEEP and is scaled x0.3 like every other row ("SLEEP -> every other row's rate x 0.3",
        # f.2 row 19) - at the full 80 Hz it would re-enter COURTSHIP after 500 ms (f.6) and defeat the raised
        # arousal threshold the x0.3 models. (Reported as a spec issue: read strictly, row 19 is not an "other
        # row", so scaling p1:court by 0.3 is a literal deviation; kept for the COURTSHIP re-entry reason above.)
        if self.mood_feedback:
            if mood == "EUPHORIA":
                euph = EUPHORIA_DEFAULT if scores is None or "euphoria" not in scores else sat(finite(scores["euphoria"]))
                out.append(self._mk("mood", "pam", MOOD_EUPHORIA_PAM_HZ * scale))
                r = MOOD_EUPHORIA_FLIGHT_HZ * euph * scale
                if r >= MIN_DRIVE_HZ:
                    out.append(self._mk("mood", "flight_dn", r))
            elif mood == "PANIC":
                out.append(self._mk("mood", "ppl1", MOOD_PANIC_PPL1_HZ * scale))
                out.append(self._mk("mood", "dn_freeze", MOOD_PANIC_FREEZE_HZ * scale))
            elif mood == "ANXIOUS":
                out.append(self._mk("mood", "ppl1", MOOD_ANXIOUS_PPL1_HZ * scale))
            if t_ms < self._court_until_ms:
                r = COURT_P1_RATE_HZ * scale
                if r >= MIN_DRIVE_HZ:
                    out.append(self._mk("court", "p1", r))

        # 20 pokes (strongest per stim/side; one channel per side so the tags never collide)
        best: dict[tuple[str, int], Poke] = {}
        for p in self._active_pokes:
            if p.stim not in POKE_TABLE:
                continue
            side = int(p.side) if p.side in (-1, 0, 1) else 0
            key = (p.stim, side)
            if key not in best or p.strength > best[key].strength:
                best[key] = p
        for (stim, side), p in best.items():
            group, base = POKE_TABLE[stim]
            rate = base * sat(p.strength) * scale
            if rate < MIN_DRIVE_HZ:
                continue
            out.append(self._mk(POKE_CHANNELS[side], group, rate, 1.0, side, 0, None))

        self.last_drives = out
        return list(out)

    # ------------------------------------------------------------------ apply

    def apply(self, engine: "LIFEngine", drives: Sequence[Drive], duration_ms: float) -> int:
        """Inject every drive into ``engine`` for ``duration_ms`` (SPEC c.27 step 4): tag ``group:channel``,
        ``rate_hz`` (the engine converts to a current itself in ``drive_mode='current'``). Returns the count."""
        n = 0
        for d in drives:
            try:
                engine.inject(d.group, rate_hz=float(d.rate_hz), duration_ms=float(duration_ms),
                              recruit=float(d.recruit), side=int(d.side), episode=int(d.episode),
                              tag=self.tag_of(d), weights=d.weights)
                n += 1
            except Exception as exc:  # noqa: BLE001 - errors never stop the sim (SPEC 0.1)
                log.warning("encoder: inject %s failed: %s", self.tag_of(d), exc)
        return n
