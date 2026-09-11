#!/usr/bin/env python3
"""Mock FlyBrain WebSocket server - frontend development and demo stand-in for `backend/flybrain/server`.

Speaks the SPEC section d protocol (version 1) on ws://127.0.0.1:4000/ws with zero backend dependencies:

  * one `hello` frame per connection (section d.1) with 384 raster rows and the 131 `pops` keys in wire order,
  * `tick` frames at 20 Hz (section d.2) with a plausible moving fly (ellipse + OU noise), per-region and
    per-population rates, sampled raster spikes and in-tick events,
  * `mood_change` about every 10 s (d.3), a `market` frame every 1 s (d.5), a `tweet` frame every 30 s (d.4)
    preceded by a `snapshot_request` (d.9) to the most recently active client,
  * out-of-band `event` frames (d.8) for poke / clear / market_source / whale / easter_egg / homeostasis,
  * every in-tick event kind of the d.8 table: jump, takeoff, landing, freeze, unfreeze, feed_start, feed_stop,
    groom, song, saccade, wander_floor, sleep, wake, wall_bump (or wrap with `--walls wrap`), gf_spike, mood,
  * the client frames of d.7: `ping` -> `pong`, `poke` (2 per second per client, else `error rate_limited`),
    `set_market_mode`, `clear`, `tweet_test`, `snapshot`; anything else -> `error bad_message`.

It also answers the HTTP endpoints of section c.28 that `frontend/lib/api.ts` (section e.4) uses, so the REST
fallbacks the UI takes while the socket is down can be exercised too:

    GET  /api/health  /api/state  /api/tweets  /api/groups
    POST /api/poke  /api/market/mode  /api/tweet/test  /api/snapshot  /api/clear      (+ OPTIONS preflight)

The HTTP side is served by a small front door (class `HttpFront`) that owns the TCP port and pipes WebSocket
upgrade requests to an internal `websockets` listener: websockets 16 parses the handshake request before
`process_request` runs and rejects every method but GET, so POST cannot be served from that hook at all.

NOTHING here simulates neurons: every number is a hand-written stand-in shaped like the real wire format, so the
Win95 Paint UI can be built, reviewed and demoed before the Python brain is wired up. Run it with

    py -3 scripts/mock_ws.py [--host 127.0.0.1] [--port 4000] [--tick-hz 20] [--seed 1337] [--quiet]
                             [--run-id mock0042] [--walls bounce|wrap] [--cap 2000]

`run_id` is random per process (a restart therefore looks like a new session to the browser, which is what
exercises the d.1 trail-clear message box); pass `--run-id` to pin it.

Output is ASCII only (the Windows console is cp1254).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024

# --------------------------------------------------------------------------------------------- wire vocabularies

REGIONS: tuple[str, ...] = (
    "optic_lobe", "antennal_lobe", "mushroom_body", "central_complex",
    "sez", "central_other", "descending_motor", "vnc",
)

# SPEC section c.4 READOUTS, verbatim order (70 names); the wire order of tick.rates.pops.
READOUTS: tuple[str, ...] = (
    "gf", "escape_dn", "dn_saccade", "dn_land", "dn_freeze", "dng100", "dn_fwd", "dn_back", "dn_halt",
    "steer_a02", "steer_a01", "steer_a03", "steer_b01", "steer_g13", "flight_dn", "groom_dn", "feed_dn",
    "pip10", "song_dn",
    "ttmn", "psi", "gfc2", "wing_power", "b1", "i1", "hg1", "wing_steer", "leg_mn", "feed_mn",
    "feed_pre_exc", "feed_pre_inh", "sugar2_exc", "sugar2_inh",
    "bitter2", "grn_sugar", "grn_water", "grn_bitter", "grn_pher", "jo_aud", "jo_groom",
    "photoreceptor", "lamina", "motion_in", "t4t5",
    "lc4", "lplc2", "lc_loom2", "lc_freeze", "orn", "alpn", "alln", "kc", "apl",
    "mbon_avoid", "mbon_approach", "pam", "ppl1",
    "epg", "pen", "delta7", "ring", "pfl3", "dfb_sleep", "p1", "mal", "lal_ps", "dms2", "song_vnc",
    "an_steer", "leg_premotor",
)

# SPEC section c.4 SIDED (only the names that are also readouts contribute _L / _R keys: 26 of them).
SIDED: frozenset[str] = frozenset({
    "gf", "steer_a02", "steer_a01", "steer_a03", "steer_b01", "steer_g13", "dn_freeze", "dng100",
    "dn_fwd", "escape_dn", "dn_saccade", "lc4", "lplc2", "lc_loom", "lc_freeze", "epg", "pfl3", "ttmn",
    "b1", "i1", "hg1", "dms2", "feed_mn", "p1", "pip10", "flight_dn", "grn_sugar_labellar", "grn_bitter",
})

# The 9 derived keys appended by RateEstimator.derived() (SPEC section d.2).
DERIVED: tuple[str, ...] = (
    "lc_loom", "escape_vnc", "steer_a02_diff", "steer_a01_diff", "steer_g13_diff",
    "b1_diff", "i1_diff", "hg1_diff", "dn_mean",
)

MOOD_STATES: tuple[str, ...] = (
    "SLEEP", "CRUISING", "FEEDING", "EUPHORIA", "ANXIOUS", "PANIC", "ESCAPE", "COURTSHIP",
)
CHANNELS: tuple[str, ...] = (
    "sugar", "bitter", "loom", "water", "dust", "pheromone", "sleep", "reward", "punish", "explore",
)
MARKET_MODES: tuple[str, ...] = (
    "sim", "dexscreener", "CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD",
)
REGIMES: tuple[str, ...] = ("CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD")

REGION_COUNTS: dict[str, int] = {
    "optic_lobe": 11126, "antennal_lobe": 705, "mushroom_body": 1523, "central_complex": 623,
    "sez": 520, "central_other": 2505, "descending_motor": 404, "vnc": 2594,
}

# SPEC section c.4 STAR_TYPES (raster rows drawn 2 px tall with a gutter label).
STAR_TYPES: frozenset[str] = frozenset({
    "DNp01", "DNa02", "DNa01", "DNg13", "DNp09", "DNg100", "DNg02_a", "MN9", "TTMn", "PSI", "PFL3", "EPG",
    "MBON01", "MBON11", "PAM01", "PPL101", "LB3b", "LC4", "LPLC2", "LC9", "pC1_14a", "pIP10", "dMS2",
    "hg1 MN", "DLMn c-f", "R1-R6", "L1", "Mi1", "T4a", "ORN_DM1", "KCg-m",
})

# (label, population hint) per region: the raster rows cycle through this list, so spikes follow the same
# drives as tick.rates.pops.
REGION_ROW_TYPES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "optic_lobe": (("R1-R6", "photoreceptor"), ("L1", "lamina"), ("Mi1", "motion_in"), ("T4a", "t4t5"),
                   ("T5b", "t4t5"), ("Tm9", "motion_in"), ("LC4", "lc4"), ("LPLC2", "lplc2"),
                   ("LC9", "lc_freeze"), ("LPLC1", "lc_loom2")),
    "antennal_lobe": (("ORN_DM1", "orn"), ("ORN_VA2", "orn"), ("DM1_lPN", "alpn"), ("VA2_lPN", "alpn"),
                      ("lLN2", "alln"), ("il3LN1", "alln"), ("JO-A1", "jo_aud"), ("JO-C1", "jo_groom")),
    "mushroom_body": (("KCg-m", "kc"), ("KCab-c", "kc"), ("KCa'b'-ap1", "kc"), ("MBON01", "mbon_avoid"),
                      ("MBON11", "mbon_approach"), ("PAM01", "pam"), ("PPL101", "ppl1"), ("APL", "apl")),
    "central_complex": (("EPG", "epg"), ("PEN_a(PEN1)", "pen"), ("Delta7", "delta7"), ("ER1a", "ring"),
                        ("PFL3", "pfl3"), ("PFL2", None), ("hDeltaB", None), ("FB6A", "dfb_sleep")),
    "sez": (("LB3b", "grn_sugar"), ("LB3c", "grn_sugar"), ("LB3a", "grn_water"), ("LB1a", "grn_bitter"),
            ("GNG215", "sugar2_exc"), ("GNG108", "feed_pre_exc"), ("MN9", "feed_mn"),
            ("GNG042", "sugar2_inh"), ("PRW046", "sugar2_exc")),
    "central_other": (("pC1_14a", "p1"), ("mAL_b1", "mal"), ("aIPg1", None), ("LAL083", "lal_ps"),
                      ("PS049", "lal_ps"), ("AVLP732m", None), ("VES051", "lal_ps")),
    "descending_motor": (("DNp01", "gf"), ("DNa02", "steer_a02"), ("DNa01", "steer_a01"), ("DNg13", "steer_g13"),
                         ("DNp09", "dn_freeze"), ("DNg100", "dng100"), ("DNg02_a", "flight_dn"),
                         ("pIP10", "pip10"), ("MDN", "dn_back"), ("DNp03", "dn_saccade")),
    "vnc": (("TTMn", "ttmn"), ("PSI", "psi"), ("hg1 MN", "hg1"), ("b1 MN", "b1"), ("DLMn c-f", "wing_power"),
            ("dMS2", "dms2"), ("Ti flexor MN", "leg_mn"), ("IN13A001", "leg_premotor"),
            ("AN03A008", "an_steer")),
}

VOCAB = ("sugar GRNs LB3b", "MN9 proboscis", "PAM dopamine", "mushroom body", "DNa02 steering",
         "giant fiber DNp01", "LC4/LPLC2 looming", "central complex EPG/PFL3")


def pop_keys() -> list[str]:
    """The exact key order of `tick.rates.pops` and `hello.pops` (131 keys)."""
    keys: list[str] = []
    for name in READOUTS:
        keys.append(name)
        if name in SIDED:
            keys.append(name + "_L")
            keys.append(name + "_R")
    keys.extend(DERIVED)
    return keys


POP_KEYS: list[str] = pop_keys()


# --------------------------------------------------------------------------------------------- helpers

def sat(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def wrap_pi(a: float) -> float:
    a = (a + math.pi) % (2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


def r1(x: float) -> float:
    return round(float(x), 1)


def r2(x: float) -> float:
    return round(float(x), 2)


def r3(x: float) -> float:
    return round(float(x), 3)


def sig_digits(x: float) -> str:
    """Significant digits of ``x`` as a string, leading/trailing zeros stripped (0.0004201 -> '4201')."""
    mantissa = ("%.12e" % abs(x)).split("e", 1)[0].replace(".", "")
    return mantissa.rstrip("0") or "0"


def poisson(rng: random.Random, lam: float, cap: int = 12) -> int:
    """Knuth's small-lambda Poisson draw, hard-capped (the mock never needs large counts)."""
    if lam <= 0.0:
        return 0
    p = math.exp(-min(20.0, lam))
    acc = rng.random()
    n = 0
    while acc > p and n < cap:
        acc *= rng.random()
        n += 1
    return n


def log(msg: str) -> None:
    sys.stdout.write("[mock_ws] " + msg + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------------------------- market

@dataclass(slots=True)
class Market:
    rng: random.Random
    symbol: str = "FLY"
    price: float = 0.0012345
    regime: str = "CALM"
    regime_left_s: float = 90.0
    forced_left_s: float = 0.0
    seq: int = 0
    mode: str = "sim"
    buys_m5: int = 40
    sells_m5: int = 28
    buys_h1: int = 400
    sells_h1: int = 370
    vol_m5: float = 5000.0
    vol_h1: float = 60000.0
    liq_usd: float = 50000.0
    chg_m5: float = 0.0
    chg_h1: float = 0.0
    chg_h6: float = 0.0
    chg_h24: float = 0.0
    supply: float = 1_000_000_000.0
    ts: float = 0.0
    trades: list[dict[str, Any]] = field(default_factory=list)
    last_trade: dict[str, Any] | None = None
    price_5m_ago: float = 0.0012345
    regime_s: float = 90.0

    # mu (per second drift), sigma, trade rate (per second), sell share
    PARAMS = {
        "CALM": (0.00002, 0.0020, 1.2, 0.48),
        "PUMP": (0.00220, 0.0060, 4.5, 0.32),
        "DUMP": (-0.00190, 0.0060, 4.0, 0.70),
        "CHOP": (0.00000, 0.0110, 3.0, 0.50),
        "RUG": (-0.01200, 0.0160, 6.0, 0.88),
        "DEAD": (0.00000, 0.0004, 0.1, 0.50),
    }

    def set_mode(self, mode: str) -> str:
        """`sim` / `dexscreener` switch the source; a regime name forces the sim regime for 60 s."""
        if mode in ("sim", "dexscreener"):
            self.mode = mode
            return "source " + mode
        self.regime = mode
        self.forced_left_s = 60.0
        self.regime_left_s = 60.0
        return "regime " + mode

    def step(self, dt: float) -> list[dict[str, Any]]:
        """Advance `dt` seconds; returns the trades generated in this step."""
        self.regime_left_s -= dt
        if self.forced_left_s > 0.0:
            self.forced_left_s -= dt
        if self.regime_left_s <= 0.0:
            self.regime = self.rng.choice(REGIMES)
            self.regime_left_s = max(8.0, self.rng.expovariate(1.0 / self.regime_s))
        mu, sigma, rate, sell_share = self.PARAMS[self.regime]
        drift = (mu - 0.5 * sigma * sigma) * dt
        shock = sigma * math.sqrt(dt) * self.rng.gauss(0.0, 1.0)
        self.price = max(1e-9, self.price * math.exp(drift + shock))

        out: list[dict[str, Any]] = []
        n = poisson(self.rng, rate * dt, 20)
        for _ in range(n):
            kind = "sell" if self.rng.random() < sell_share else "buy"
            usd = math.exp(self.rng.gauss(3.4, 1.3))
            if self.rng.random() < 0.02:
                usd *= 30.0                                  # whale
            tr = {"kind": kind, "usd": r1(usd), "ts": time.time(), "surrogate": self.mode == "dexscreener"}
            out.append(tr)
            self.last_trade = tr
            if kind == "buy":
                self.buys_m5 += 1
                self.buys_h1 += 1
            else:
                self.sells_m5 += 1
                self.sells_h1 += 1
            self.vol_m5 += usd
            self.vol_h1 += usd
        # 5-minute windows decay towards their rate
        self.buys_m5 = max(0, int(self.buys_m5 * (1.0 - dt / 300.0)))
        self.sells_m5 = max(0, int(self.sells_m5 * (1.0 - dt / 300.0)))
        self.buys_h1 = max(0, int(self.buys_h1 * (1.0 - dt / 3600.0)))
        self.sells_h1 = max(0, int(self.sells_h1 * (1.0 - dt / 3600.0)))
        self.vol_m5 *= (1.0 - dt / 300.0)
        self.vol_h1 *= (1.0 - dt / 3600.0)
        self.liq_usd = max(500.0, self.liq_usd * (1.0 + 0.2 * (self.price / max(1e-12, self.price_5m_ago) - 1.0) * dt))
        self.price_5m_ago += (self.price - self.price_5m_ago) * (dt / 300.0)
        self.chg_m5 = r1(100.0 * (self.price / max(1e-12, self.price_5m_ago) - 1.0))
        self.chg_h1 += (self.chg_m5 - self.chg_h1) * (dt / 60.0)
        self.chg_h6 += (self.chg_h1 - self.chg_h6) * (dt / 600.0)
        self.chg_h24 += (self.chg_h6 - self.chg_h24) * (dt / 2400.0)
        self.trades.extend(out)
        if len(self.trades) > 200:
            del self.trades[:len(self.trades) - 200]
        return out

    def bump_seq(self) -> None:
        self.seq += 1
        self.ts = time.time()

    def snapshot(self) -> dict[str, Any]:
        source = "dexscreener" if self.mode == "dexscreener" else "sim"
        mcap = self.price * self.supply
        return {
            "source": source, "ts": self.ts or time.time(), "seq": self.seq,
            "chain": "solana" if source == "dexscreener" else "sim",
            "dex": "raydium" if source == "dexscreener" else "sim",
            "pair": "MOCK" if source == "dexscreener" else "SIM", "symbol": self.symbol,
            "price_usd": self.price, "price_native": self.price,
            "buys_m5": self.buys_m5, "sells_m5": self.sells_m5,
            "buys_h1": self.buys_h1, "sells_h1": self.sells_h1,
            "chg_m5": r1(self.chg_m5), "chg_h1": r1(self.chg_h1),
            "chg_h6": r1(self.chg_h6), "chg_h24": r1(self.chg_h24),
            "vol_m5": r1(self.vol_m5), "vol_h1": r1(self.vol_h1),
            "liq_usd": r1(self.liq_usd), "fdv": r1(mcap), "mcap": r1(mcap),
            "regime": self.regime if source == "sim" else None,
        }


# --------------------------------------------------------------------------------------------- the mock brain

@dataclass(slots=True)
class Poke:
    stim: str
    strength: float
    side: int
    until_wall: float


class MockBrain:
    """The whole stand-in world: drives, fly kinematics, mood, rates and raster sampling."""

    def __init__(self, seed: int, canvas_w: int, canvas_h: int, tick_hz: int, dt_ms: float,
                 per_region: int, cap: int, regime_s: float, walls: str = "bounce",
                 run_id: str | None = None) -> None:
        self.rng = random.Random(seed)
        # One run_id per process (SPEC d.1: a different run_id after a reconnect means a new session). It must not
        # come from --seed, or restarting the mock would keep the session identity and the browser's trail-clear
        # confirmation could never be seen.
        self.run_id = run_id or ("mock%04d" % random.Random().randrange(10000))
        self.walls = "wrap" if walls == "wrap" else "bounce"
        self.w = canvas_w
        self.h = canvas_h
        self.tick_hz = tick_hz
        self.tick_ms = 1000.0 / tick_hz
        self.dt_ms = dt_ms
        self.steps_per_tick = max(1, int(round(self.tick_ms / dt_ms)))
        self.per_region = per_region
        self.cap = cap
        self.market = Market(rng=random.Random(seed + 2), regime_s=regime_s)
        self.rows = self._build_rows()
        self.row_groups: tuple[list[dict[str, Any]], ...] = tuple(
            [r for r in self.rows if r["region"] == ri] for ri in range(len(REGIONS)))

        self.seq = 0
        self.t_ms = 0
        self.t0_wall = time.time()
        # fly state: heading-integrated wander toward a roaming target (the fly overshoots it, so it does reach the
        # walls and the d.8 wall_bump / wrap events are real).
        self.x = canvas_w / 2.0 + 0.33 * min(canvas_w, canvas_h)
        self.y = canvas_h / 2.0
        self.heading = math.pi / 2.0
        self.speed = 40.0
        self.omega = 0.0
        self.leg_phase = 0.0
        self.jump_start_ms: int | None = None
        self.pending_jump = False
        self.noise_h = 0.0
        self.tx = canvas_w / 2.0
        self.ty = canvas_h / 2.0
        self.target_ms = 0
        self.last_edge_ms = -99999
        # short modes layered on the mood, and the transition bookkeeping behind the d.8 event kinds
        self.prev_fly_mode = "walk"
        self.freeze_until_ms = -1
        self.freeze_cool_ms = 0
        self.groom_until_ms = -1
        self.groom_cool_ms = 4000
        self.stalled_ms = 0
        self.wander_cool_ms = 0
        self.last_wing_ext = 0
        self.pending_events: list[dict[str, Any]] = []
        # mood
        self.mood = "CRUISING"
        self.prev_mood = "CRUISING"
        self.mood_since_ms = 0
        self.mood_next_ms = 9000
        self.scores = {"euphoria": 0.1, "anxiety": 0.05, "arousal": 0.3, "valence": 0.05,
                       "hunger": 0.4, "sleep": 0.0}
        self.feeding = False
        self.meal_start_ms = 0
        # pokes / drives
        self.pokes: list[Poke] = []
        self.drives = {k: 0.0 for k in ("sugar", "bitter", "water", "looming", "flash", "odor", "chop",
                                        "courtship", "sleep_pressure", "explore", "up", "down", "activity")}
        self.loom_side = 0
        self.gain = 1.0
        self.tweets: list[dict[str, Any]] = []
        self.tweet_n = 0
        self.easter_egg_ms = -999999

    # ------------------------------------------------------------------ hello

    def _build_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        neuron = 0
        slot = 0
        for ri, region in enumerate(REGIONS):
            pool = REGION_ROW_TYPES[region]
            for k in range(self.per_region):
                label, pop = pool[k % len(pool)]
                side = "M" if label in ("APL", "Delta7", "MDN", "mAL_b1") else ("L" if k % 2 == 0 else "R")
                rows.append({
                    "region": ri, "slot": slot, "neuron": neuron, "label": label,
                    "side": side, "star": label in STAR_TYPES, "pop": pop,
                })
                slot += 1
                neuron += 1 + (k % 3)
        return rows

    def hello(self) -> dict[str, Any]:
        return {
            "type": "hello", "v": PROTOCOL_VERSION, "run_id": self.run_id,
            "server_wall": time.time(),
            "dt_ms": self.dt_ms, "tick_ms": self.tick_ms, "steps_per_tick": self.steps_per_tick,
            "tick_hz": self.tick_hz, "realtime": True, "backend": "mock",
            "canvas": {"w": self.w, "h": self.h, "walls": self.walls},
            "connectome": {
                "name": "mock-20000-s1337-cal", "source": "synthetic", "n": 20000, "e": 548120,
                "synapses": 2691003.0, "gain": self.gain, "weights_mode": "calibrated",
                "license": "synthetic (no data)", "citation": None,
                "note": "MOCK SERVER: no neurons are simulated; synthetic structured stand-in shaped like "
                        "MaleCNS v1.0, not real connectome data",
                "region_counts": dict(REGION_COUNTS),
            },
            "regions": list(REGIONS),
            "raster": {
                "per_region": self.per_region, "cap": self.cap,
                "rows": [{k: r[k] for k in ("region", "slot", "neuron", "label", "side", "star")}
                         for r in self.rows],
            },
            "pops": list(POP_KEYS),
            "mood_states": list(MOOD_STATES),
            "channels": list(CHANNELS),
            "market_modes": list(MARKET_MODES),
            "market": {"mode": self.market.mode, "chain": "sim", "symbol": self.market.symbol,
                       "token": "", "poll_s": 1},
            "agent": {"llm": "dryrun", "x": "dryrun", "cooldown_s": 900, "reason_cooldown_s": 2700,
                      "tweets_per_day": 12, "lang": "en"},
            "features": {"explore_baseline": 0.25, "wander_sigma": 0.6, "mood_feedback": True,
                         "easter_eggs": True, "drive_mode": "poisson", "noise_mu": 0.5, "noise_sigma": 3.5},
        }

    # ------------------------------------------------------------------ pokes

    def push_poke(self, stim: str, strength: float, side: int, duration_ms: float) -> None:
        self.pokes.append(Poke(stim, strength, side, time.time() + duration_ms / 1000.0))

    # ------------------------------------------------------------------ one tick

    def tick(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Advance one wall tick; returns (tick frame, out-of-band event frames)."""
        self.seq += 1
        dt = self.tick_ms / 1000.0
        self.t_ms += int(round(self.tick_ms))
        wall = time.time()
        oob: list[dict[str, Any]] = []
        # events queued between ticks (the d.8 `mood` duplicate of the mood_change frame) ride this tick
        events: list[dict[str, Any]] = self.pending_events
        self.pending_events = []

        trades = self.market.step(dt)
        self._update_drives(dt, trades)
        for tr in trades:
            if tr["usd"] >= 2000.0:
                oob.append(self.event("whale", {"kind": tr["kind"], "usd": tr["usd"],
                                                 "ratio": r1(tr["usd"] / 60.0)}))
        egg = self._easter_egg(wall)
        if egg is not None:
            oob.append(self.event("easter_egg", {"reason": egg}))

        pops = self._pops()
        regions = self._region_rates(pops)
        self._update_mood(dt, pops, events)
        self._update_fly(dt, pops, events)
        mode = self._fly_mode()
        self._mode_events(mode, pops, events)
        spikes, total, capped = self._sample_spikes(pops, regions)

        if pops["gf"] > 0.5:
            events.append({"kind": "gf_spike", "t_ms": self.t_ms,
                           "data": {"side": "L" if self.loom_side < 0 else "R", "count": 1}})
        if self.rng.random() < 0.0008:
            gb = self.gain
            self.gain = max(0.3, round(gb * 0.9, 3))
            oob.append(self.event("homeostasis", {"gain_before": gb, "gain_after": self.gain,
                                                   "active_frac": r3(min(0.08, 0.004 + total / 20000.0))}))

        frame = {
            "type": "tick", "seq": self.seq, "t_ms": self.t_ms, "wall": wall,
            "sim": {
                "rtf": r2(1.8 + 0.6 * self.rng.random()), "speed": 1.0,
                "steps": self.steps_per_tick, "step_ms": r2(0.18 + 0.05 * self.rng.random()),
                "spikes": total, "active_frac": r3(min(0.05, total / (20000.0 * 1.2))),
                "gain": self.gain, "edge_visits": int(total * 38.5), "forced": int(60 * self.drives["any_max"] * 4),
                "noise": True, "backend": "mock",
            },
            "fly": self._fly_wire(),
            "ink": self._ink(events),
            "mood": self._mood_wire(),
            "market": self._tick_market(),
            "drives": self._drives_wire(),
            "rates": {"regions": [r2(v) for v in regions],
                      "pops": {k: r2(v) for k, v in pops.items()}},
            "spikes": spikes,
            "events": events,
        }
        return frame, oob

    def event(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        return {"type": "event", "seq": self.seq, "t_ms": self.t_ms, "wall": time.time(),
                "kind": kind, "data": data}

    def _easter_egg(self, wall: float) -> str | None:
        """SPEC c.19/d.8 easter_egg reason, or None. All three d.8 reasons are reachable; 30 s cooldown.

        ``buys_m5 in (69, 420)`` -> ``"buys_m5 == 420"``; first three significant price digits ``420`` (or first
        two ``69``) -> ``"price 0.000420"``; local clock ``04:20``/``16:20`` -> ``"clock 04:20"``.
        """
        if self.t_ms - self.easter_egg_ms <= 30000:
            return None
        digits = sig_digits(self.market.price)
        clock = time.strftime("%H:%M", time.localtime(wall))
        if self.market.buys_m5 in (69, 420):
            reason: str | None = "buys_m5 == 420"
        elif digits.startswith("420") or digits.startswith("69"):
            reason = "price 0.000420"
        elif clock in ("04:20", "16:20"):
            reason = "clock 04:20"
        else:
            reason = None
        if reason is not None:
            self.easter_egg_ms = self.t_ms
        return reason

    # ------------------------------------------------------------------ drives

    def _update_drives(self, dt: float, trades: list[dict[str, Any]]) -> None:
        now = time.time()
        self.pokes = [p for p in self.pokes if p.until_wall > now]
        active: dict[str, float] = {}
        side = 0
        for p in self.pokes:
            active[p.stim] = max(active.get(p.stim, 0.0), p.strength)
            if p.stim == "loom" and p.side != 0:
                side = p.side

        m = self.market
        up = sat(m.chg_m5 / 4.0)
        down = sat(-m.chg_m5 / 4.0)
        sells = sum(1 for t in trades if t["kind"] == "sell")
        chop = sat(abs(m.chg_m5) / 8.0) * (1.0 if m.regime == "CHOP" else 0.4)
        target = {
            "sugar": max(active.get("sugar", 0.0), active.get("reward", 0.0) * 0.6, 0.85 * up),
            "bitter": max(active.get("bitter", 0.0), active.get("punish", 0.0) * 0.7, 0.6 * down),
            "water": active.get("water", 0.0),
            "looming": max(active.get("loom", 0.0), sat(sells / 3.0) * (0.9 if m.regime in ("RUG", "DUMP") else 0.35)),
            "flash": 1.0 if any(t["usd"] > 2000.0 and t["kind"] == "sell" for t in trades) else 0.0,
            "odor": max(active.get("dust", 0.0) * 0.4, 0.25 + 0.3 * up),
            "chop": chop,
            "courtship": active.get("pheromone", 0.0),
            "sleep_pressure": max(active.get("sleep", 0.0), 1.0 if m.regime == "DEAD" else 0.0),
            "explore": max(0.25, active.get("explore", 0.0), 0.5 * (1.0 - chop)),
            "up": up,
            "down": down,
            "activity": sat(abs(self.speed) / 60.0),
        }
        for k, v in target.items():
            tau = 0.15 if v > self.drives[k] else 0.8
            self.drives[k] += (v - self.drives[k]) * (1.0 - math.exp(-dt / tau))
        if side:
            self.loom_side = side
        elif self.drives["looming"] < 0.05:
            self.loom_side = 0
        elif self.loom_side == 0:
            self.loom_side = self.rng.choice((-1, 1))
        self.scores["hunger"] = clamp(self.scores["hunger"] + (0.02 if not self.feeding else -0.25) * dt, 0.0, 1.0)
        self.drives["any_max"] = max(self.drives[k] for k in ("sugar", "bitter", "water", "looming", "flash",
                                                             "odor", "chop", "courtship", "sleep_pressure"))

    def _drives_wire(self) -> dict[str, Any]:
        d = self.drives
        candle = "up" if self.market.chg_m5 > 0.3 else "down" if self.market.chg_m5 < -0.3 else "flat"
        out = {k: r3(d[k]) for k in ("sugar", "bitter", "water", "looming")}
        out["loom_side"] = self.loom_side
        for k in ("flash", "odor", "chop", "courtship", "sleep_pressure", "explore", "up", "down", "activity"):
            out[k] = r3(d[k])
        out["hunger"] = r3(self.scores["hunger"])
        out["candle"] = candle
        out["any_max"] = r3(d["any_max"])
        return out

    # ------------------------------------------------------------------ rates

    def _pops(self) -> dict[str, float]:
        d = self.drives
        rng = self.rng
        j = lambda: 1.0 + 0.12 * (rng.random() - 0.5)                                     # noqa: E731

        sugar, loom, bitter = d["sugar"], d["looming"], d["bitter"]
        feeding = sugar > 0.35
        flying = self.mood == "EUPHORIA"
        court = self.mood == "COURTSHIP"
        asleep = self.mood == "SLEEP"

        steer_bias = clamp(self.omega * 3.0, -1.0, 1.0)
        a02 = (4.0 + 8.0 * d["explore"]) * j()
        a02_l = a02 * (1.0 - 0.6 * steer_bias)
        a02_r = a02 * (1.0 + 0.6 * steer_bias)

        gf = 45.0 if (loom > 0.85 and rng.random() < 0.25) else 0.0
        v: dict[str, float] = {
            "gf": gf,
            "escape_dn": 12.0 * sat((loom - 0.6) / 0.4) * j(),
            "dn_saccade": 6.0 * abs(steer_bias) * j(),
            "dn_land": 3.0 * (1.0 if flying else 0.0) * j(),
            "dn_freeze": (1.2 + 26.0 * sat((loom - 0.25) / 0.75)) * j(),
            "dng100": (7.5 + 14.0 * d["explore"]) * j(),
            "dn_fwd": (3.0 + 6.0 * d["explore"]) * j(),
            "dn_back": 0.4 * j(),
            "dn_halt": (1.2 + 6.0 * sat(loom)) * j(),
            "steer_a02": a02, "steer_a01": 3.0 * j(), "steer_a03": 2.0 * j(), "steer_b01": 1.0 * j(),
            "steer_g13": 2.2 * j(),
            "flight_dn": (22.0 if flying else 0.3) * j(),
            "groom_dn": (9.0 if d["odor"] > 0.6 else 0.5) * j(),
            "feed_dn": (14.0 if feeding else 0.8) * j(),
            "pip10": (24.0 if court else 0.0) * j(),
            "song_dn": (9.0 if court else 0.0) * j(),
            "ttmn": 30.0 if gf > 0 else 0.0,
            "psi": 22.0 if gf > 0 else 0.0,
            "gfc2": 8.0 if gf > 0 else 0.0,
            "wing_power": (120.0 if flying else 0.0) * j(),
            "b1": (35.0 if flying or court else 0.0) * j(),
            "i1": (20.0 if flying else 0.0) * j(),
            "hg1": (26.0 if court else (12.0 if flying else 0.0)) * j(),
            "wing_steer": (9.0 if flying else 0.2) * j(),
            "leg_mn": (0.5 if asleep else 6.4 + 10.0 * sat(abs(self.speed) / 60.0)) * j(),
            "feed_mn": (52.0 * sat(sugar / 0.6) if feeding else 0.6) * j(),
            "feed_pre_exc": (22.0 * sat(sugar / 0.6) + 0.8) * j(),
            "feed_pre_inh": (9.5 + 6.0 * bitter) * j(),
            "sugar2_exc": (48.0 * sat(sugar / 0.7) + 0.5) * j(),
            "sugar2_inh": (35.0 * sat(sugar / 0.7) + 0.5) * j(),
            "bitter2": (1.1 + 28.0 * bitter) * j(),
            "grn_sugar": (131.0 * sat(sugar / 0.8) + 0.4) * j(),
            "grn_water": (60.0 * d["water"]) * j(),
            "grn_bitter": (2.0 + 90.0 * bitter) * j(),
            "grn_pher": (40.0 * d["courtship"]) * j(),
            "jo_aud": (2.1 + 10.0 * d["chop"]) * j(),
            "jo_groom": (3.9 + 18.0 * d["odor"]) * j(),
            "photoreceptor": (21.0 + 40.0 * d["flash"]) * j(),
            "lamina": (14.0 + 20.0 * d["flash"]) * j(),
            "motion_in": (15.5 + 10.0 * sat(abs(self.speed) / 80.0)) * j(),
            "t4t5": (4.0 + 8.0 * sat(abs(self.speed) / 80.0)) * j(),
            "lc4": (0.4 + 180.0 * sat(loom)) * j(),
            "lplc2": (0.5 + 160.0 * sat(loom)) * j(),
            "lc_loom2": (0.6 + 30.0 * sat(loom)) * j(),
            "lc_freeze": (0.3 + 40.0 * sat((loom - 0.3) / 0.7)) * j(),
            "orn": (18.0 + 30.0 * d["odor"]) * j(),
            "alpn": (11.0 + 18.0 * d["odor"]) * j(),
            "alln": (7.0 + 8.0 * d["odor"]) * j(),
            "kc": (1.4 + 1.2 * d["odor"]) * j(),
            "apl": (9.0 + 4.0 * d["odor"]) * j(),
            "mbon_avoid": (1.0 + 14.0 * bitter) * j(),
            "mbon_approach": (6.0 + 16.0 * sugar) * j(),
            "pam": (4.0 + 30.0 * sugar) * j(),
            "ppl1": (0.5 + 18.0 * bitter) * j(),
            "epg": (9.1 if not asleep else 2.0) * j(),
            "pen": 6.0 * j(), "delta7": 4.0 * j(), "ring": 2.0 * j(),
            "pfl3": (7.5 + 5.0 * abs(steer_bias)) * j(),
            "dfb_sleep": (0.2 + 24.0 * d["sleep_pressure"]) * j(),
            "p1": (22.0 if court else 0.1) * j(),
            "mal": (10.0 + 6.0 * bitter) * j(),
            "lal_ps": (3.0 + 5.0 * abs(steer_bias)) * j(),
            "dms2": (12.0 if court else 0.0) * j(),
            "song_vnc": (8.0 if court else 0.0) * j(),
            "an_steer": (2.0 + 4.0 * abs(steer_bias)) * j(),
            "leg_premotor": (0.4 if asleep else 5.2 + 6.0 * sat(abs(self.speed) / 60.0)) * j(),
        }
        out: dict[str, float] = {}
        for name in READOUTS:
            tot = v[name]
            out[name] = tot
            if name in SIDED:
                if name == "steer_a02":
                    out["steer_a02_L"], out["steer_a02_R"] = a02_l, a02_r
                elif name == "gf":
                    left = tot if self.loom_side <= 0 else 0.0
                    out["gf_L"], out["gf_R"] = left, tot - left
                else:
                    frac = 0.5 + 0.08 * (self.rng.random() - 0.5)
                    out[name + "_L"] = tot * frac
                    out[name + "_R"] = tot - out[name + "_L"]
        out["lc_loom"] = 0.5 * (out["lc4"] + out["lplc2"])
        out["escape_vnc"] = out["ttmn"] + out["psi"] + out["gfc2"]
        for nm in ("steer_a02", "steer_a01", "steer_g13", "b1", "i1", "hg1"):
            out[nm + "_diff"] = out[nm + "_R"] - out[nm + "_L"]
        dn_keys = ("gf", "escape_dn", "dn_saccade", "dn_land", "dn_freeze", "dng100", "dn_fwd", "dn_back",
                   "dn_halt", "steer_a02", "steer_a01", "steer_a03", "steer_b01", "steer_g13", "flight_dn",
                   "groom_dn", "feed_dn", "pip10", "song_dn")
        out["dn_mean"] = sum(out[k] for k in dn_keys) / len(dn_keys)
        return {k: out[k] for k in POP_KEYS}

    def _region_rates(self, pops: dict[str, float]) -> list[float]:
        d = self.drives
        asleep = 0.35 if self.mood == "SLEEP" else 1.0
        return [
            (2.6 + 2.5 * d["flash"] + 3.0 * sat(d["looming"])) * asleep,
            (1.1 + 3.0 * d["odor"]) * asleep,
            (0.4 + 1.0 * d["odor"] + 0.6 * d["sugar"]) * asleep,
            (2.0 + 1.2 * d["explore"]) * asleep,
            (1.4 + 7.0 * d["sugar"] + 3.0 * d["bitter"]) * asleep,
            (1.7 + 1.5 * d["courtship"]) * asleep,
            (1.2 + pops["dn_mean"] * 0.35) * asleep,
            (1.0 + 0.25 * pops["leg_mn"] + 0.05 * pops["wing_power"]) * asleep,
        ]

    # ------------------------------------------------------------------ raster sampling

    def _sample_spikes(self, pops: dict[str, float], regions: list[float]) -> tuple[dict[str, Any], int, bool]:
        steps = self.steps_per_tick
        win_s = self.tick_ms / 1000.0
        pairs: list[tuple[int, int]] = []                                  # (dt step, slot)
        rng = self.rng
        # SPEC d.10 budget: the sampled rows must carry the same activity the frame advertises, so per region the
        # row rates are rescaled to mean == rates.regions[i]. The population hint only shapes the contrast inside a
        # lane (sqrt-compressed: a 131 Hz sugar GRN row stays ~8x brighter than its 2 Hz neighbours), it no longer
        # sets the absolute rate - that made the raster emit ~4x the events the published rates imply.
        for ri, rows in enumerate(self.row_groups):
            target = regions[ri]
            if target <= 0.0 or not rows:
                continue
            shape = [math.sqrt(max(0.05, pops.get(r["pop"], 0.0) if r["pop"] else target)) for r in rows]
            total_shape = sum(shape)
            if total_shape <= 0.0:
                continue
            k = target * len(rows) / total_shape
            for row, sh in zip(rows, shape):
                # Poisson spike count for this row in the 50 ms window
                for _ in range(poisson(rng, sh * k * win_s)):
                    pairs.append((rng.randrange(steps), row["slot"]))
        total = len(pairs)
        capped = total > self.cap
        if capped:
            stride = total / float(self.cap)
            pairs = [pairs[int(i * stride)] for i in range(self.cap)]
        pairs.sort()
        return ({"t0_ms": self.t_ms - int(round(self.tick_ms)), "win_ms": int(round(self.tick_ms)),
                 "total": total, "capped": capped,
                 "slots": [s for _, s in pairs], "dt": [d for d, _ in pairs]}, total, capped)

    # ------------------------------------------------------------------ mood

    def _update_mood(self, dt: float, pops: dict[str, float], events: list[dict[str, Any]]) -> None:
        d = self.drives
        s = self.scores
        e_raw = 0.45 * d["sugar"] + 0.35 * sat(pops["feed_mn"] / 60.0) + 0.20 * sat(pops["pam"] / 40.0)
        a_raw = 0.50 * d["looming"] + 0.30 * sat(pops["dn_freeze"] / 40.0) + 0.20 * sat(pops["lc_loom"] / 60.0)
        k = 1.0 - math.exp(-dt / 2.0)
        s["euphoria"] += (e_raw - s["euphoria"]) * k
        s["anxiety"] += (a_raw - s["anxiety"]) * k
        s["arousal"] += (0.5 * sat(pops["dn_mean"] / 5.0) + 0.5 * d["any_max"] - s["arousal"]) * k
        ma, mv = pops["mbon_approach"], pops["mbon_avoid"]
        s["valence"] = clamp(s["euphoria"] - s["anxiety"] + 0.2 * (ma - mv) / (ma + mv + 5.0), -1.0, 1.0)
        s["sleep"] = d["sleep_pressure"]
        self.mood_since_ms += int(round(self.tick_ms))
        # MN9 above 20 Hz is a meal; falling asleep always ends it (so feed_stop cannot be starved, d.8)
        feeding_now = pops["feed_mn"] >= 20.0 and self.mood != "SLEEP"
        if feeding_now and not self.feeding:
            self.feeding = True
            self.meal_start_ms = self.t_ms
            events.append({"kind": "feed_start", "t_ms": self.t_ms, "data": {"mn9_hz": r2(pops["feed_mn"])}})
        elif not feeding_now and self.feeding:
            self.feeding = False
            events.append({"kind": "feed_stop", "t_ms": self.t_ms,
                           "data": {"meal_ms": self.t_ms - self.meal_start_ms}})

    def next_mood(self) -> tuple[str, str]:
        """Pick the next mood (never the current one) and a human reason; fakes the section f.6 table."""
        s, d, cur = self.scores, self.drives, self.mood
        rules: list[tuple[str, str]] = []
        if d["looming"] > 0.75:
            rules.append(("ESCAPE", "jumped (DNp01 spike)") if self.rng.random() < 0.4
                         else ("PANIC", "anxiety %.2f >= 0.70 for 2.0 s" % s["anxiety"]))
        if d["sleep_pressure"] > 0.9:
            rules.append(("SLEEP", "sleep_pressure 0.97 and activity < 0.35 for 20.0 s"))
        if d["courtship"] > 0.4:
            rules.append(("COURTSHIP", "p1 22 >= 15 Hz for 0.5 s"))
        if s["euphoria"] > 0.6:
            rules.append(("EUPHORIA", "euphoria %.2f >= 0.70 for 3.0 s" % s["euphoria"]))
        if self.feeding:
            rules.append(("FEEDING", "feeding (MN9 >= 20 Hz)"))
        if s["anxiety"] > 0.3 or d["bitter"] > 0.35:
            rules.append(("ANXIOUS", "anxiety %.2f >= 0.35 for 1.0 s" % s["anxiety"]))
        for dst, reason in rules:
            if dst != cur:
                return dst, reason
        if cur in ("PANIC", "ESCAPE"):
            return "ANXIOUS", "anxiety %.2f < 0.45 for 3.0 s" % s["anxiety"]
        if cur != "CRUISING" and self.rng.random() < 0.5:
            return "CRUISING", "%s exit condition held 2.0 s" % cur.lower()
        pool = [m for m in ("CRUISING", "FEEDING", "EUPHORIA", "ANXIOUS", "COURTSHIP", "SLEEP") if m != cur]
        m = self.rng.choice(pool)
        return m, "mock scheduler (%s scores settled)" % m.lower()

    def force_mood(self, dst: str, reason: str) -> dict[str, Any]:
        self.prev_mood = self.mood
        self.mood = dst
        self.mood_since_ms = 0
        if dst == "ESCAPE":
            self.jump_start_ms = self.t_ms
            self.pending_jump = True
        # d.8: the same transition also rides the next tick as an in-tick `mood` event, for rAF consumers
        self.pending_events.append({"kind": "mood", "t_ms": self.t_ms,
                                    "data": {"from": self.prev_mood, "to": dst, "reason": reason}})
        return {"type": "mood_change", "seq": self.seq, "t_ms": self.t_ms, "wall": time.time(),
                "from": self.prev_mood, "to": dst, "reason": reason, "mood": self._mood_wire()}

    def _mood_wire(self) -> dict[str, Any]:
        s = self.scores
        return {"state": self.mood, "prev": self.prev_mood, "since_ms": self.mood_since_ms,
                "euphoria": r3(s["euphoria"]), "anxiety": r3(s["anxiety"]), "arousal": r3(s["arousal"]),
                "valence": r3(s["valence"]), "fear": r3(s["anxiety"]), "hunger": r3(s["hunger"]),
                "sleep": r3(s["sleep"]), "dwell_left_ms": max(0, 1500 - self.mood_since_ms)}

    # ------------------------------------------------------------------ fly kinematics

    def _update_fly(self, dt: float, pops: dict[str, float], events: list[dict[str, Any]]) -> None:
        mood = self.mood
        if mood == "SLEEP":
            v_target, tau = 0.0, 0.05
        elif mood == "FEEDING":
            v_target, tau = 6.0, 0.05
        elif mood == "EUPHORIA":
            v_target, tau = 150.0, 0.30
        elif mood == "PANIC":
            v_target, tau = 110.0, 0.15
        elif mood == "ESCAPE":
            v_target, tau = 200.0, 0.15
        elif mood == "COURTSHIP":
            v_target, tau = 25.0, 0.05
        else:
            v_target, tau = 40.0 + 30.0 * self.drives["explore"], 0.15
        self.speed += (v_target - self.speed) * (1.0 - math.exp(-dt / tau))

        # freeze (looming + DNp09) and grooming (dust odour / idle) are short modes layered on the mood
        if (self.t_ms >= self.freeze_until_ms and self.t_ms >= self.freeze_cool_ms
                and self.drives["looming"] > 0.55 and pops["dn_freeze"] > 14.0 and self.rng.random() < 0.25):
            self.freeze_until_ms = self.t_ms + self.rng.randrange(400, 1200)
            self.freeze_cool_ms = self.freeze_until_ms + 2500
        frozen = self.t_ms < self.freeze_until_ms
        if (not frozen and self.t_ms >= self.groom_until_ms and self.t_ms >= self.groom_cool_ms
                and mood in ("CRUISING", "ANXIOUS", "FEEDING")
                and (self.drives["odor"] > 0.45 or self.rng.random() < 0.01)):
            self.groom_until_ms = self.t_ms + self.rng.randrange(800, 2000)
            self.groom_cool_ms = self.groom_until_ms + 6000
        grooming = not frozen and self.t_ms < self.groom_until_ms
        if frozen:
            self.speed = 0.0
        elif grooming:
            self.speed = min(self.speed, 2.0)

        # guaranteed-motion floor (SPEC f.3): a fly that has barely moved for 1.5 s gets a push, once per 5 s
        if mood == "SLEEP" or frozen or grooming:
            self.stalled_ms = 0
        elif abs(self.speed) < 8.0:
            self.stalled_ms += int(round(self.tick_ms))
        else:
            self.stalled_ms = 0
        if self.stalled_ms >= 1500 and self.t_ms >= self.wander_cool_ms:
            self.wander_cool_ms = self.t_ms + 5000
            events.append({"kind": "wander_floor", "t_ms": self.t_ms,
                           "data": {"stalled_ms": self.stalled_ms}})
            self.speed = max(self.speed, 18.0)
            self.stalled_ms = 0

        # heading-integrated wander: steer toward a target point re-picked every 4..9 s plus an OU heading wobble.
        # Nothing brakes at the target, so a target near an edge walks the fly into the wall (d.8 wall_bump / wrap).
        self.noise_h += (-self.noise_h * 1.2 + 1.1 * self.rng.gauss(0.0, 1.0)) * dt
        if self.t_ms >= self.target_ms:
            self.target_ms = self.t_ms + self.rng.randrange(4000, 9000)
            self.tx = self.rng.uniform(10.0, self.w - 10.0)
            self.ty = self.rng.uniform(10.0, self.h - 10.0)
        turn = wrap_pi(math.atan2(self.ty - self.y, self.tx - self.x) - self.heading)
        self.omega = clamp(1.4 * turn + 0.6 * self.noise_h, -12.0, 12.0)
        self.heading = wrap_pi(self.heading + self.omega * dt)
        nx = self.x + self.speed * math.cos(self.heading) * dt
        ny = self.y + self.speed * math.sin(self.heading) * dt
        self.x, self.y = self._walls(nx, ny, events)
        self.leg_phase = (self.leg_phase + abs(self.speed) / 20.0 * dt) % 1.0
        if mood != "ESCAPE":
            self.jump_start_ms = None
            self.pending_jump = False
        if self.pending_jump:
            self.pending_jump = False
            self.jump_start_ms = self.t_ms
            events.append({"kind": "jump", "t_ms": self.t_ms,
                           "data": {"heading_out": r3(self.heading), "impulse": 220.0,
                                    "side": "L" if self.loom_side < 0 else "R", "forward": True}})
        if abs(self.omega) > 2.5 and self.rng.random() < 0.2:           # a fast turn = a saccade (d.8)
            events.append({"kind": "saccade", "t_ms": self.t_ms,
                           "data": {"side": "R" if self.omega > 0 else "L"}})

    def _walls(self, nx: float, ny: float, events: list[dict[str, Any]]) -> tuple[float, float]:
        """Apply `hello.canvas.walls` to a candidate position and emit the matching d.8 event."""
        w, h, m = self.w, self.h, 8.0
        if self.walls == "wrap":
            edge: str | None = None
            if nx < 0.0:
                nx, edge = nx + w, "left"
            elif nx > w:
                nx, edge = nx - w, "right"
            if ny < 0.0:
                ny, edge = ny + h, "top"
            elif ny > h:
                ny, edge = ny - h, "bottom"
            if edge is not None and self.t_ms - self.last_edge_ms >= 200:
                self.last_edge_ms = self.t_ms
                events.append({"kind": "wrap", "t_ms": self.t_ms, "data": {"edge": edge}})
            return clamp(nx, 0.0, w), clamp(ny, 0.0, h)
        hx, hy = math.cos(self.heading), math.sin(self.heading)       # heading BEFORE the reflection
        wall: tuple[float, float] | None = None                       # unit vector from the fly toward that wall
        if nx < m:
            nx, wall = m + (m - nx), (-1.0, 0.0)
            self.heading = wrap_pi(math.pi - self.heading)
        elif nx > w - m:
            nx, wall = (w - m) - (nx - (w - m)), (1.0, 0.0)
            self.heading = wrap_pi(math.pi - self.heading)
        if ny < m:
            ny = m + (m - ny)
            wall = wall or (0.0, -1.0)
            self.heading = wrap_pi(-self.heading)
        elif ny > h - m:
            ny = (h - m) - (ny - (h - m))
            wall = wall or (0.0, 1.0)
            self.heading = wrap_pi(-self.heading)
        if wall is None:
            return nx, ny
        # leave the wall instead of grinding along it: aim back into the middle third of the canvas
        self.target_ms = self.t_ms + self.rng.randrange(3000, 7000)
        self.tx = self.rng.uniform(0.3 * w, 0.7 * w)
        self.ty = self.rng.uniform(0.3 * h, 0.7 * h)
        if self.t_ms - self.last_edge_ms >= 200:
            self.last_edge_ms = self.t_ms
            # screen y points down and the heading is clockwise-positive, so cross > 0 = the wall is to the right
            cross = hx * wall[1] - hy * wall[0]
            events.append({"kind": "wall_bump", "t_ms": self.t_ms,
                           "data": {"side": "R" if cross > 0 else "L"}})
        return clamp(nx, m, w - m), clamp(ny, m, h - m)

    def _mode_events(self, mode: str, pops: dict[str, float], events: list[dict[str, Any]]) -> None:
        """The d.8 kinds that mark a body-mode transition: takeoff/landing, freeze/unfreeze, sleep/wake, groom, song."""
        prev = self.prev_fly_mode
        self.prev_fly_mode = mode
        if mode != prev:
            air_now, air_prev = mode in ("fly", "jump"), prev in ("fly", "jump")
            if air_now and not air_prev:
                events.append({"kind": "takeoff", "t_ms": self.t_ms, "data": {"wing_hz": r1(self._wing_hz(mode))}})
            elif air_prev and not air_now:
                events.append({"kind": "landing", "t_ms": self.t_ms, "data": {"reason": "mood %s" % self.mood}})
            if mode == "freeze" or prev == "freeze":
                events.append({"kind": "freeze" if mode == "freeze" else "unfreeze", "t_ms": self.t_ms,
                               "data": {"dnp09_hz": r2(pops["dn_freeze"]), "looming": r3(self.drives["looming"])}})
            if mode == "sleep":
                events.append({"kind": "sleep", "t_ms": self.t_ms, "data": {}})
            elif prev == "sleep":
                events.append({"kind": "wake", "t_ms": self.t_ms, "data": {"reason": "mood %s" % self.mood}})
            if mode == "groom":
                events.append({"kind": "groom", "t_ms": self.t_ms,
                               "data": {"side": "R" if self.rng.random() < 0.5 else "L"}})
        ext = self._wing_ext(mode)
        if ext != 0 and ext != self.last_wing_ext:                    # one wing held out = a song bout (d.8 `song`)
            events.append({"kind": "song", "t_ms": self.t_ms, "data": {"side": "R" if ext > 0 else "L"}})
        self.last_wing_ext = ext

    def _fly_mode(self) -> str:
        if self.jump_start_ms is not None and self.t_ms - self.jump_start_ms < 400:
            return "jump"
        if self.t_ms < self.freeze_until_ms:
            return "freeze"
        if self.t_ms < self.groom_until_ms:
            return "groom"
        return {"SLEEP": "sleep", "FEEDING": "feed", "EUPHORIA": "fly", "COURTSHIP": "court",
                "ESCAPE": "fly", "PANIC": "walk", "ANXIOUS": "walk", "CRUISING": "walk"}[self.mood]

    def _wing_hz(self, mode: str) -> float:
        return 0.0 if mode not in ("fly", "jump") else 200.0 + 20.0 * math.sin(self.t_ms / 900.0)

    def _wing_ext(self, mode: str) -> int:
        if mode != "court":
            return 0
        return 1 if (self.t_ms // 1500) % 2 == 0 else -1

    def _fly_wire(self) -> dict[str, Any]:
        mode = self._fly_mode()
        flying = mode in ("fly", "jump")
        wing_hz = self._wing_hz(mode)
        return {
            "x": r1(self.x), "y": r1(self.y),
            "vx": r1(self.speed * math.cos(self.heading)), "vy": r1(self.speed * math.sin(self.heading)),
            "heading": r3(self.heading), "speed": r1(self.speed), "omega": r3(clamp(self.omega, -12.0, 12.0)),
            "wing_hz": r1(wing_hz), "wing_amp": r3(0.85 if flying else 0.0),
            "wing_ext": self._wing_ext(mode),
            "mode": mode, "leg_phase": r3(self.leg_phase),
            "proboscis": r3(sat(self.drives["sugar"] * 1.2) if mode == "feed" else 0.0),
            "jump_t_ms": (self.t_ms - self.jump_start_ms) if mode == "jump" and self.jump_start_ms is not None else None,
        }

    def _ink(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        mood, m = self.mood, self.market
        color = "#00a800" if m.chg_m5 > 0.3 else "#a80000" if m.chg_m5 < -0.3 else "#000000"
        color = {"FEEDING": "#ff8c00", "COURTSHIP": "#ff69b4", "PANIC": "#ff0000", "ESCAPE": "#ff3300",
                 "SLEEP": "#6b6b6b", "EUPHORIA": "#ff00ff"}.get(mood, color)
        style = {"EUPHORIA": "rainbow", "PANIC": "zigzag", "ESCAPE": "zigzag", "FEEDING": "dotted",
                 "COURTSHIP": "hearts"}.get(mood, "solid")
        mode = self._fly_mode()
        alpha = {"fly": 0.4, "court": 0.7, "jump": 0.0}.get(mode, 1.0)
        stamp: str | None = None
        if mode == "jump":
            stamp = "dash"
        elif any(e["kind"] == "wall_bump" for e in events):
            stamp = "bump"
        elif mode == "feed" and (self.t_ms // 250) % 2 == 0:
            stamp = "blob"
        elif mode == "court" and self.t_ms % 500 < self.tick_ms:
            stamp = "heart"
        elif mode == "sleep" and self.t_ms % 500 < self.tick_ms:
            stamp = "zzz"
        return {"color": color, "width": r1(1 + round(3 * sat(abs(m.chg_m5) / 3.0))),
                "alpha": r3(alpha), "style": style, "stamp": stamp}

    def _tick_market(self) -> dict[str, Any]:
        snap = self.market.snapshot()
        snap["mode"] = self.market.mode
        snap["last_trade"] = self.market.last_trade
        return snap

    # ------------------------------------------------------------------ tweets

    def make_tweet(self, reason: str, snapshot_source: str) -> dict[str, Any]:
        self.tweet_n += 1
        m = self.market
        neurons = list(self.rng.sample(("LB3b", "MN9", "PAM", "DNp01", "LC4", "DNa02", "EPG", "PFL3"), 3))
        text = ("gm ser. %s at %.0f Hz, %s says %s. mock brain, %s candles, $FLY mcap %s. "
                % (neurons[0], 40.0 + 90.0 * self.drives["sugar"], self.rng.choice(VOCAB),
                   self.rng.choice(("wagmi", "ngmi", "cope", "wen")), m.regime.lower(),
                   ("%.2fM" % (m.price * m.supply / 1e6))))[:240]
        rec = {
            "type": "tweet", "seq": self.seq, "t_ms": self.t_ms, "wall": time.time(),
            "id": time.strftime("%Y%m%d-%H%M%S"), "reason": reason, "text": text,
            "model": "template", "dry_run": True, "posted": False, "url": None, "error": None,
            "snapshot_source": snapshot_source, "neurons": neurons, "mood": self.mood,
            "latency_ms": r1(1.0 + 5.0 * self.rng.random()),
        }
        self.tweets.insert(0, {k: v for k, v in rec.items() if k not in ("type", "seq")})
        del self.tweets[50:]
        return rec


# --------------------------------------------------------------------------------------------- clients

class Client:
    """One browser: a drop-oldest queue of 4 frames plus the d.7 poke rate limit."""

    _next_id = 0

    def __init__(self, conn: Any) -> None:
        Client._next_id += 1
        self.id = Client._next_id
        self.conn = conn
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=4)
        self.poke_times: list[float] = []
        self.last_frame_wall = time.time()
        self.dropped = 0

    def put(self, text: str) -> None:
        while True:
            try:
                self.queue.put_nowait(text)
                return
            except asyncio.QueueFull:
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except asyncio.QueueEmpty:
                    return

    def allow_poke(self) -> bool:
        now = time.time()
        self.poke_times = [t for t in self.poke_times if now - t < 1.0]
        if len(self.poke_times) >= 2:
            return False
        self.poke_times.append(now)
        return True


# --------------------------------------------------------------------------------------------- validation (d.7)

CLIENT_FIELDS: dict[str, tuple[str, ...]] = {
    "poke": ("stim", "strength", "side", "duration_ms"),
    "set_market_mode": ("mode",),
    "clear": (),
    "tweet_test": (),
    "ping": ("t",),
    "snapshot": ("id", "png_b64"),
}


def parse_client(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """`extra='forbid'` style validation of a client frame. Returns (message, error message)."""
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError) as exc:
        return None, "json: %s" % exc
    if not isinstance(msg, dict):
        return None, "frame must be a JSON object"
    t = msg.get("type")
    if not isinstance(t, str) or t not in CLIENT_FIELDS:
        return None, "type: Input should be one of %s" % ", ".join(sorted(CLIENT_FIELDS))
    allowed = set(CLIENT_FIELDS[t]) | {"type"}
    extra = sorted(set(msg) - allowed)
    if extra:
        return None, "%s: Extra inputs are not permitted (%s)" % (t, ", ".join(extra))
    missing = [f for f in CLIENT_FIELDS[t] if f not in msg]
    if missing:
        return None, "%s: Field required (%s)" % (t, ", ".join(missing))
    if t == "poke":
        if msg["stim"] not in CHANNELS:
            return None, "poke.stim: Input should be one of %s" % ", ".join(CHANNELS)
        if not isinstance(msg["strength"], (int, float)) or not 0.0 <= float(msg["strength"]) <= 1.0:
            return None, "poke.strength: Input should be between 0 and 1"
        if msg["side"] not in ("L", "R", "both"):
            return None, "poke.side: Input should be 'L', 'R' or 'both'"
        if not isinstance(msg["duration_ms"], (int, float)) or not 50 <= float(msg["duration_ms"]) <= 5000:
            return None, "poke.duration_ms: Input should be between 50 and 5000"
    elif t == "set_market_mode":
        if msg["mode"] not in MARKET_MODES:
            return None, "set_market_mode.mode: Input should be one of %s" % ", ".join(MARKET_MODES)
    elif t == "ping":
        if not isinstance(msg["t"], (int, float)):
            return None, "ping.t: Input should be a valid number"
    elif t == "snapshot":
        if not isinstance(msg["id"], str) or not isinstance(msg["png_b64"], str):
            return None, "snapshot: id and png_b64 must be strings"
    return msg, None


# --------------------------------------------------------------------------------------------- server

class MockServer:
    def __init__(self, brain: MockBrain, args: argparse.Namespace) -> None:
        self.brain = brain
        self.args = args
        self.clients: set[Client] = set()
        self.last_active: Client | None = None
        self.pending_snapshot: dict[str, asyncio.Future[int]] = {}
        self.snapshot_n = 0
        self.last_tick: dict[str, Any] | None = None
        self.verbose = not args.quiet

    # -------------------------------------------------------------- publishing

    def publish(self, frame: dict[str, Any]) -> None:
        if not self.clients:
            return
        text = json.dumps(frame, separators=(",", ":"))
        for c in list(self.clients):
            c.put(text)

    def publish_text(self, text: str) -> None:
        for c in list(self.clients):
            c.put(text)

    # -------------------------------------------------------------- the 20 Hz producer

    async def producer(self) -> None:
        brain = self.brain
        period = brain.tick_ms / 1000.0
        next_wall = time.perf_counter()
        last_market_seq_wall = 0.0
        next_tweet_wall = time.time() + self.args.tweet_s
        while True:
            next_wall += period
            sleep = next_wall - time.perf_counter()
            if sleep > 0:
                await asyncio.sleep(sleep)
            else:
                next_wall = time.perf_counter()
            try:
                frame, oob = brain.tick()
                self.last_tick = frame
                text = json.dumps(frame, separators=(",", ":"))
                self.publish_text(text)
                for ev in oob:
                    self.publish(ev)
                now = time.time()
                # mood changes about every 10 s (d.3)
                if brain.t_ms >= brain.mood_next_ms:
                    brain.mood_next_ms = brain.t_ms + int(self.args.mood_s * 1000 * (0.8 + 0.5 * brain.rng.random()))
                    dst, reason = brain.next_mood()
                    if dst != brain.mood:
                        self.publish(brain.force_mood(dst, reason))
                        if self.verbose:
                            log("mood %s -> %s (%s)" % (brain.prev_mood, dst, reason))
                # market frame roughly every second (d.5)
                if now - last_market_seq_wall >= self.args.market_s:
                    last_market_seq_wall = now
                    brain.market.bump_seq()
                    trades = list(brain.market.trades)
                    brain.market.trades.clear()
                    self.publish({"type": "market", "seq": brain.seq, "t_ms": brain.t_ms, "wall": now,
                                  "mode": brain.market.mode, "market": brain.market.snapshot(),
                                  "trades": trades[-200:]})
                # tweet every 30 s (d.4), preceded by a snapshot_request (d.9)
                if now >= next_tweet_wall:
                    next_tweet_wall = now + self.args.tweet_s
                    asyncio.get_running_loop().create_task(self.fire_tweet("mock_timer"))
            except Exception as exc:                                  # never stop the producer
                log("tick failed: %r" % (exc,))

    # -------------------------------------------------------------- snapshot + tweet

    async def request_snapshot(self, deadline_ms: int = 3000) -> str:
        """Ask the most recently active client for a PNG; returns "browser" or "server"."""
        c = self.last_active
        if c is None or c not in self.clients:
            return "server"
        self.snapshot_n += 1
        sid = "snap-%d" % self.snapshot_n
        fut: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self.pending_snapshot[sid] = fut
        c.put(json.dumps({"type": "snapshot_request", "id": sid, "deadline_ms": deadline_ms},
                         separators=(",", ":")))
        try:
            nbytes = await asyncio.wait_for(fut, timeout=deadline_ms / 1000.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self.pending_snapshot.pop(sid, None)
            return "server"
        self.pending_snapshot.pop(sid, None)
        if self.verbose:
            log("snapshot %s delivered: %d bytes" % (sid, nbytes))
        return "browser"

    async def fire_tweet(self, reason: str) -> dict[str, Any] | None:
        try:
            source = await self.request_snapshot()
            rec = self.brain.make_tweet(reason, source)
            self.publish(rec)
            if self.verbose:
                log("tweet (%s, snapshot %s): %s" % (reason, source, rec["text"][:70]))
            return rec
        except Exception as exc:
            log("tweet failed: %r" % (exc,))
            return None

    # -------------------------------------------------------------- per-connection

    async def handler(self, conn: Any) -> None:
        path = getattr(getattr(conn, "request", None), "path", "/ws") or "/ws"
        if path.split("?")[0] not in ("/ws", "/"):
            await conn.close(code=1008, reason="unknown path")
            return
        client = Client(conn)
        try:
            await conn.send(json.dumps(self.brain.hello(), separators=(",", ":")))
        except Exception as exc:
            log("client %d: hello failed: %r" % (client.id, exc))
            return
        self.clients.add(client)                       # only now may broadcasts reach this client
        self.last_active = client
        log("client %d connected (%d total)" % (client.id, len(self.clients)))
        sender = asyncio.get_running_loop().create_task(self._sender(client))
        try:
            async for raw in conn:
                if isinstance(raw, bytes):
                    await self._error(client, "bad_message", "binary frames are not accepted")
                    continue
                self.last_active = client
                await self._on_client_frame(client, raw)
        except Exception as exc:
            if self.verbose:
                log("client %d read ended: %r" % (client.id, exc))
        finally:
            sender.cancel()
            self.clients.discard(client)
            if self.last_active is client:
                self.last_active = next(iter(self.clients), None)
            log("client %d disconnected (%d dropped frames, %d total)"
                % (client.id, client.dropped, len(self.clients)))

    async def _sender(self, client: Client) -> None:
        try:
            while True:
                text = await client.queue.get()
                await client.conn.send(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.verbose:
                log("client %d send ended: %r" % (client.id, exc))

    async def _error(self, client: Client, code: str, msg: str) -> None:
        client.put(json.dumps({"type": "error", "code": code, "msg": msg}, separators=(",", ":")))

    async def _on_client_frame(self, client: Client, raw: str) -> None:
        msg, err = parse_client(raw)
        if msg is None:
            await self._error(client, "bad_message", err or "invalid frame")
            return
        brain = self.brain
        t = msg["type"]
        if t == "ping":
            client.put(json.dumps({"type": "pong", "t": msg["t"], "server_wall": time.time(),
                                   "seq": brain.seq}, separators=(",", ":")))
            return
        if t == "poke":
            if not client.allow_poke():
                await self._error(client, "rate_limited", "poke: 2 per second per client")
                return
            side = {"L": -1, "R": 1, "both": 0}[msg["side"]]
            brain.push_poke(msg["stim"], float(msg["strength"]), side, float(msg["duration_ms"]))
            self.publish(brain.event("poke", {"stim": msg["stim"], "strength": r3(float(msg["strength"])),
                                               "side": msg["side"], "duration_ms": int(msg["duration_ms"])}))
            if self.verbose:
                log("poke %s %s x%.2f for %d ms" % (msg["stim"], msg["side"], float(msg["strength"]),
                                                    int(msg["duration_ms"])))
            return
        if t == "set_market_mode":
            mode = msg["mode"]
            if mode == "dexscreener":
                await self._error(client, "forbidden", "FLY_TOKEN_ADDRESS is empty (mock server)")
                return
            what = brain.market.set_mode(mode)
            self.publish(brain.event("market_source", {"mode": brain.market.mode,
                                                        "reason": "client set %s" % what}))
            if self.verbose:
                log("market %s" % what)
            return
        if t == "clear":
            self.publish(brain.event("clear", {"by": "client"}))
            if self.verbose:
                log("clear")
            return
        if t == "tweet_test":
            asyncio.get_running_loop().create_task(self.fire_tweet("manual"))
            return
        if t == "snapshot":
            try:
                png = base64.b64decode(msg["png_b64"], validate=True)
            except (binascii.Error, ValueError):
                await self._error(client, "bad_message", "snapshot.png_b64: not valid base64")
                return
            if len(png) > MAX_SNAPSHOT_BYTES:
                await self._error(client, "bad_message", "snapshot: larger than 2 MB")
                return
            if not png.startswith(PNG_MAGIC):
                await self._error(client, "bad_message", "snapshot: not a PNG")
                return
            fut = self.pending_snapshot.get(msg["id"])
            if fut is not None and not fut.done():
                fut.set_result(len(png))
            return


# --------------------------------------------------------------------------------------------- plain HTTP

MAX_HTTP_HEAD = 64 * 1024
MAX_HTTP_BODY = 4 * 1024 * 1024
PHRASES: dict[int, str] = {200: "OK", 204: "No Content", 400: "Bad Request", 403: "Forbidden",
                           404: "Not Found", 405: "Method Not Allowed", 429: "Too Many Requests",
                           500: "Internal Server Error", 502: "Bad Gateway"}


def http_json(obj: Any) -> tuple[bytes, str]:
    return json.dumps(obj, separators=(",", ":")).encode("utf-8"), "application/json"


def http_response(status: int, body: bytes, ctype: str) -> bytes:
    head = (
        "HTTP/1.1 %d %s\r\n"
        "Date: %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: content-type, accept\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Max-Age: 600\r\n"
        "Connection: close\r\n"
        "\r\n"
    ) % (status, PHRASES.get(status, "Status"), time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()),
         ctype, len(body))
    return head.encode("ascii") + body


class HttpFront:
    """The plain-HTTP front door: it owns the advertised TCP port and pipes WebSocket upgrades to `ws_port`.

    `websockets.serve(process_request=...)` cannot serve the POST endpoints of `frontend/lib/api.ts`: in websockets
    16 `Request.parse` raises `unsupported HTTP method; expected GET; got POST` before the hook is reached, so a
    `postPoke` / `postMarketMode` / `postTweetTest` / `postSnapshot` fallback got a dropped connection and the mock
    console got a traceback. Serving HTTP here (and forwarding only handshakes) keeps both sides honest on one port.
    """

    def __init__(self, server: MockServer, ws_port: int) -> None:
        self.server = server
        self.ws_port = ws_port
        self.poke_times: list[float] = []

    # -------------------------------------------------------------- connection handling

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(self._read_head(reader), timeout=15.0)
            if head is None:
                return
            raw, method, target, headers, rest = head
            if method == "GET" and headers.get("upgrade", "").lower() == "websocket":
                await self._pipe(raw, reader, writer)
                return
            body = rest
            try:
                want = int(headers.get("content-length", "0") or "0")
            except ValueError:
                want = 0
            if want > MAX_HTTP_BODY:
                writer.write(http_response(400, *http_json({"ok": False, "error": "body too large"})))
            else:
                while len(body) < want:
                    chunk = await reader.read(min(65536, want - len(body)))
                    if not chunk:
                        break
                    body += chunk
                status, payload, ctype = await self.route(method, target, body)
                writer.write(http_response(status, payload, ctype))
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionResetError, BrokenPipeError):
            pass
        except Exception as exc:                                        # never let one request kill the listener
            log("http failed: %r" % (exc,))
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _read_head(self, reader: asyncio.StreamReader
                         ) -> tuple[bytes, str, str, dict[str, str], bytes] | None:
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = await reader.read(4096)
            if not chunk:
                return None
            raw += chunk
            if len(raw) > MAX_HTTP_HEAD:
                return None
        head, _, rest = raw.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        parts = lines[0].split()
        if len(parts) < 2:
            return None
        headers: dict[str, str] = {}
        for line in lines[1:]:
            k, sep, v = line.partition(":")
            if sep:
                headers[k.strip().lower()] = v.strip()
        return raw, parts[0].upper(), parts[1], headers, rest

    async def _pipe(self, first: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Forward one WebSocket connection byte-for-byte to the internal websockets listener."""
        try:
            ir, iw = await asyncio.open_connection("127.0.0.1", self.ws_port)
        except OSError as exc:
            writer.write(http_response(502, b"websocket backend unavailable\n", "text/plain"))
            await writer.drain()
            log("ws pipe failed: %r" % (exc,))
            return
        iw.write(first)
        await iw.drain()

        async def pump(r: asyncio.StreamReader, w: asyncio.StreamWriter) -> None:
            try:
                while True:
                    chunk = await r.read(65536)
                    if not chunk:
                        break
                    w.write(chunk)
                    await w.drain()
            except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
                pass
            finally:
                try:
                    w.close()
                except Exception:
                    pass

        await asyncio.gather(pump(reader, iw), pump(ir, writer), return_exceptions=True)

    # -------------------------------------------------------------- routing

    def _allow_poke(self) -> bool:
        now = time.time()
        self.poke_times = [t for t in self.poke_times if now - t < 1.0]
        if len(self.poke_times) >= 2:                                   # same 2/s budget as the d.7 poke frame
            return False
        self.poke_times.append(now)
        return True

    async def route(self, method: str, target: str, body: bytes) -> tuple[int, bytes, str]:
        path = target.split("?")[0]
        if method == "OPTIONS":
            return 204, b"", "text/plain"
        server, brain = self.server, self.server.brain
        if method == "GET":
            if path == "/api/health":
                return (200, *http_json({
                    "ok": True, "uptime_s": r1(time.time() - brain.t0_wall), "run_id": brain.run_id,
                    "rtf": 2.1, "speed": 1.0, "seq": brain.seq, "t_ms": brain.t_ms, "backend": "mock",
                    "connectome": {"name": "mock-20000-s1337-cal", "source": "synthetic", "n": 20000,
                                   "e": 548120, "gain": brain.gain, "license": "synthetic (no data)"},
                    "market": {"mode": brain.market.mode, "ok": True, "last_poll": time.time(), "failures": 0},
                    "agent": {"llm": "dryrun", "x": "dryrun", "tweets_today": len(brain.tweets),
                              "last_tweet_wall": brain.tweets[0]["wall"] if brain.tweets else None,
                              "disabled_reason": None},
                    "clients": len(server.clients), "log_tail": ["mock server: no brain is running"],
                }))
            if path == "/api/tweets":
                return (200, *http_json(brain.tweets[:20]))
            if path == "/api/groups":
                return (200, *http_json({name: 2 for name in READOUTS}))
            if path == "/api/state":
                return (200, *http_json({"hello": brain.hello(), "tick": server.last_tick, "trail": [],
                                         "mood_history": [[brain.t_ms, brain.mood]], "events": []}))
            if path in ("/", "/ws"):
                return (200, b"mock_ws.py: connect a WebSocket to /ws (SPEC section d)\n", "text/plain")
            return (404, *http_json({"ok": False, "error": "not found", "path": path}))
        if method != "POST":
            return (405, *http_json({"ok": False, "error": "method not allowed"}))
        return await self._post(path, body)

    async def _post(self, path: str, body: bytes) -> tuple[int, bytes, str]:
        """The four REST fallbacks of SPEC e.4 (+ /api/clear), validated exactly like their d.7 frames."""
        brain = self.server.brain
        if path not in ("/api/poke", "/api/market/mode", "/api/tweet/test", "/api/snapshot", "/api/clear"):
            return (404, *http_json({"ok": False, "error": "not found", "path": path}))
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (ValueError, UnicodeDecodeError) as exc:
            return (400, *http_json({"ok": False, "error": "bad_message", "msg": "json: %s" % exc}))
        if not isinstance(payload, dict):
            return (400, *http_json({"ok": False, "error": "bad_message", "msg": "body must be a JSON object"}))

        if path == "/api/tweet/test":
            rec = await self.server.fire_tweet("manual")
            if rec is None:
                return (500, *http_json({"ok": False, "error": "tweet failed"}))
            return (200, *http_json({k: v for k, v in rec.items() if k not in ("type", "seq")}))
        if path == "/api/clear":
            self.server.publish(brain.event("clear", {"by": "api"}))
            if self.server.verbose:
                log("clear (REST)")
            return (200, *http_json({"ok": True}))

        kind = {"/api/poke": "poke", "/api/market/mode": "set_market_mode", "/api/snapshot": "snapshot"}[path]
        msg, err = parse_client(json.dumps({"type": kind, **payload}))
        if msg is None:
            return (400, *http_json({"ok": False, "error": "bad_message", "msg": err or "invalid body"}))
        if kind == "poke":
            if not self._allow_poke():
                return (429, *http_json({"ok": False, "error": "rate_limited", "msg": "poke: 2 per second"}))
            side = {"L": -1, "R": 1, "both": 0}[msg["side"]]
            brain.push_poke(msg["stim"], float(msg["strength"]), side, float(msg["duration_ms"]))
            self.server.publish(brain.event("poke", {"stim": msg["stim"], "strength": r3(float(msg["strength"])),
                                                     "side": msg["side"],
                                                     "duration_ms": int(msg["duration_ms"])}))
            if self.server.verbose:
                log("poke %s %s (REST)" % (msg["stim"], msg["side"]))
            return (200, *http_json({"ok": True}))
        if kind == "set_market_mode":
            if msg["mode"] == "dexscreener":
                return (403, *http_json({"ok": False, "error": "forbidden",
                                         "msg": "FLY_TOKEN_ADDRESS is empty (mock server)"}))
            what = brain.market.set_mode(msg["mode"])
            self.server.publish(brain.event("market_source", {"mode": brain.market.mode,
                                                              "reason": "api set %s" % what}))
            if self.server.verbose:
                log("market %s (REST)" % what)
            # the real server (c.28) echoes the requested mode, not the resulting feed mode
            return (200, *http_json({"ok": True, "mode": msg["mode"]}))
        # snapshot
        try:
            png = base64.b64decode(msg["png_b64"], validate=True)
        except (binascii.Error, ValueError):
            return (400, *http_json({"ok": False, "error": "bad_message", "msg": "png_b64: not valid base64"}))
        if len(png) > MAX_SNAPSHOT_BYTES:                               # 413 like POST /api/snapshot of c.28
            return (413, *http_json({"ok": False, "error": "bad_message", "msg": "snapshot: larger than 2 MB"}))
        if not png.startswith(PNG_MAGIC):
            return (400, *http_json({"ok": False, "error": "bad_message", "msg": "snapshot: not a PNG"}))
        fut = self.server.pending_snapshot.get(msg["id"])
        if fut is not None and not fut.done():
            fut.set_result(len(png))
        return (200, *http_json({"ok": True, "bytes": len(png)}))


# --------------------------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mock_ws.py",
        description="Mock FlyBrain WebSocket server (SPEC section d) for frontend development.",
        epilog=("HTTP: GET /api/{health,state,tweets,groups}; POST /api/{poke,market/mode,tweet/test,snapshot,clear} "
                "(the REST fallbacks of frontend/lib/api.ts, validated like their d.7 frames). "
                "run_id is random per process - restart the mock to exercise the d.1 trail-clear message box, or "
                "pin it with --run-id. Use a small --cap (e.g. --cap 40) to see spikes.capped = true."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1", help="bind host (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=4000, help="port (default 4000)")
    p.add_argument("--run-id", default=None, help="pin hello.run_id (default: random per process)")
    p.add_argument("--walls", choices=("bounce", "wrap"), default="bounce",
                   help="hello.canvas.walls: reflect at the edges (wall_bump) or wrap (wrap)")
    p.add_argument("--tick-hz", type=int, default=20, help="tick rate, 5..50 (default 20)")
    p.add_argument("--dt-ms", type=float, default=1.0, help="advertised LIF step (default 1.0)")
    p.add_argument("--seed", type=int, default=1337, help="random seed (default 1337)")
    p.add_argument("--canvas", default="800x500", help="canvas WxH (default 800x500)")
    p.add_argument("--per-region", type=int, default=48, help="raster rows per region (default 48)")
    p.add_argument("--cap", type=int, default=2000, help="raster events per tick cap (default 2000)")
    p.add_argument("--mood-s", type=float, default=10.0, help="mean seconds between mood changes (default 10)")
    p.add_argument("--market-s", type=float, default=1.0, help="seconds between market frames (default 1)")
    p.add_argument("--tweet-s", type=float, default=30.0, help="seconds between tweet frames (default 30)")
    p.add_argument("--regime-s", type=float, default=90.0, help="mean market regime dwell (default 90)")
    p.add_argument("--quiet", action="store_true", help="only log connections and errors")
    return p


async def run(args: argparse.Namespace) -> int:
    try:
        from websockets.asyncio.server import serve
    except Exception as exc:                                        # pragma: no cover - install guard
        log("cannot import websockets (>= 14 asyncio API): %r" % (exc,))
        return 2

    try:
        cw, ch = (int(v) for v in args.canvas.lower().split("x", 1))
    except ValueError:
        log("bad --canvas %r (expected WxH, e.g. 800x500)" % args.canvas)
        return 2
    if not 5 <= args.tick_hz <= 50:
        log("bad --tick-hz %d (5..50)" % args.tick_hz)
        return 2

    brain = MockBrain(args.seed, cw, ch, args.tick_hz, args.dt_ms, args.per_region, args.cap, args.regime_s,
                      walls=args.walls, run_id=args.run_id)
    server = MockServer(brain, args)
    assert len(brain.rows) == 8 * args.per_region, "raster rows must be 8 * per_region"
    assert len(POP_KEYS) == 131, "pops must have 131 keys, got %d" % len(POP_KEYS)

    log("FlyBrain MOCK server (no neurons are simulated)")
    log("ws://%s:%d/ws  |  %d Hz ticks, %d raster rows, %d pops keys, run_id %s, walls %s"
        % (args.host, args.port, args.tick_hz, len(brain.rows), len(POP_KEYS), brain.run_id, brain.walls))
    log("http://%s:%d/api/health for the REST stand-ins (GET + POST); Ctrl+C to stop" % (args.host, args.port))

    # The websockets listener stays internal on 127.0.0.1:<ephemeral>; HttpFront owns the advertised port so that
    # POST /api/... can be answered at all (websockets parses and rejects non-GET before process_request).
    async with serve(server.handler, "127.0.0.1", 0,
                     ping_interval=20, ping_timeout=20, max_size=4 * 1024 * 1024) as ws_server:
        ws_port = ws_server.sockets[0].getsockname()[1]
        front = HttpFront(server, ws_port)
        try:
            http_server = await asyncio.start_server(front.handle, args.host, args.port)
        except OSError as exc:
            log("cannot bind %s:%d: %r" % (args.host, args.port, exc))
            return 2
        async with http_server:
            await server.producer()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        log("stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
