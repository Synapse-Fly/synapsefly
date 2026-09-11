"""Simulated token market: 5-regime Markov GBM with clustered trades and whales (SPEC section c.15).

Regime switching is a discrete-time Markov chain with switch probability ``dt / (dwell_s * dwell_scale)`` per
poll (uniform over the other regimes); ``ln P`` follows ``mu*dt + sigma*sqrt(dt)*Z`` plus a square-root price
impact per trade; trade arrivals are a self-exciting (Hawkes) Poisson process ``lam = lam0*(1 - eta) + exc(t)``
with ``exc`` jumping by ``eta*beta`` per trade and decaying at ``beta``; sizes are log-normal with occasional whales.

**Hawkes normalisation** (deviation from the literal c.15 formula, reported as a spec issue because it is
measurable): c.15
writes ``lam = lam0 + sum alpha*exp(-beta*(t - t_i))`` with ``alpha = 0.4, beta = 0.2``. That kernel integrates
to ``alpha/beta = 2`` excitations per trade, i.e. a *supercritical* process: every trade spawns two more on
average, the intensity diverges and no baseline survives (an earlier revision capped the excitation at
``3*lam0``, which simply pinned every regime at ``4*lam0`` and broke the f.1 ``activity`` numbers and the f.6
SLEEP gate). Here ``alpha`` is read as the **branching ratio** ``eta`` (expected offspring per trade, clipped to
``HAWKES_BRANCH_MAX``) and ``beta`` as the decay rate, so the kernel is ``eta*beta*exp(-beta*dt)`` and the
immigrant rate is ``lam0*(1 - eta)``. Then ``E[lam] == lam0`` exactly for every regime - which is what the
downstream numbers require (f.1 ``act_snap = sat(log1p(n_m5)/log1p(500))``: CALM ``lam0 = 0.3`` -> 90 trades per
5 min -> activity 0.73; DEAD ``lam0 = 0.02`` -> 6 -> activity 0.31, below the 0.35 SLEEP threshold of f.6) -
while bursts still ride well above the baseline (clustering: ``Var[N] ~ E[N]/(1 - eta)^2``).

Every number here is ``[E]`` (engineered stand-in for a meme-coin tape); the snapshot shape mirrors the
DexScreener pair object (RESEARCH section 8 ``[V]``) so the encoder cannot tell the two sources apart.
Deterministic by contract: the generator is ``SeedSequence(seed)`` child ``[2]`` (SPEC section 0.1) and the
draw order per poll is fixed (switch, Poisson count, trade sizes, whales, sides, diffusion).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .base import SIM_SOURCE, MarketSnapshot, Trade

__all__ = [
    "Regime",
    "REGIMES",
    "REGIME_NAMES",
    "SimulatedMarket",
    "SEED_CHILD_MARKET",
    "TRADE_USD_LN_MEDIAN",
    "TRADE_USD_LN_SIGMA",
    "IMPACT_COEF",
    "RUG_LIQ_DRAIN_PER_S",
    "HAWKES_BRANCH_MAX",
    "HAWKES_WINDOW_S",
]


@dataclass(frozen=True)
class Regime:
    """One market regime: per-second drift/vol of ``ln P``, Hawkes baseline trades/s, buy probability, mean dwell."""

    name: str
    mu: float
    sigma: float
    lam0: float
    p_buy: float
    dwell_s: float


#: SPEC section c.15 regime table ``[E]``.
REGIMES: tuple[Regime, ...] = (
    Regime("CALM", 0.0, 0.002, 0.30, 0.50, 90.0),
    Regime("PUMP", +0.0015, 0.006, 1.50, 0.82, 40.0),
    Regime("DUMP", -0.0020, 0.008, 1.50, 0.18, 30.0),
    Regime("CHOP", 0.0, 0.012, 0.80, 0.50, 45.0),
    Regime("RUG", -0.0060, 0.015, 2.00, 0.10, 20.0),  # liquidity also drains 2 %/s
    Regime("DEAD", 0.0, 0.0005, 0.02, 0.50, 120.0),
)
REGIME_NAMES: tuple[str, ...] = tuple(r.name for r in REGIMES)

#: SPEC section 0.1 spawn order: ``[2] market sim``.
SEED_CHILD_MARKET: int = 2

#: Trade size ``usd ~ LogNormal(ln 80, 1.2)`` ``[E]``.
TRADE_USD_LN_MEDIAN: float = math.log(80.0)
TRADE_USD_LN_SIGMA: float = 1.2
#: Price impact ``+-2e-4 * sqrt(usd / 100)`` on ``ln P`` per trade ``[E]``.
IMPACT_COEF: float = 2e-4
IMPACT_USD_REF: float = 100.0
#: RUG regime: ``liq *= (1 - 0.02 * dt)`` ``[E]``.
RUG_LIQ_DRAIN_PER_S: float = 0.02
#: Hawkes memory window of SPEC c.15 ("sum over the last 60 s"): a contribution is dropped once it is older
#: than this, which for the exponential kernel means zeroing the excitation below one kernel jump at 60 s ``[E]``.
HAWKES_WINDOW_S: float = 60.0
#: Largest branching ratio accepted from ``hawkes_alpha`` ``[E]``: at 1.0 the process is critical and the tape
#: explodes, so the constructor clips to this.
HAWKES_BRANCH_MAX: float = 0.95

_M5_S: float = 300.0
_H1_S: float = 3600.0
_H6_S: float = 6.0 * 3600.0
_H24_S: float = 24.0 * 3600.0
#: Price ring capacity in snapshots (~25 h at one snapshot per second); pruned in blocks.
_PRICE_RING_MAX: int = 90_000
_PRICE_RING_PRUNE: int = 3_600


class SimulatedMarket:
    """Deterministic simulated tape (SPEC section c.15). ``name == "sim"``; symbol ``FLY``, chain ``sim``."""

    name: str = SIM_SOURCE

    def __init__(
        self,
        seed: int,
        price0: float = 0.001,
        supply: float = 1e9,
        liq0: float = 50_000.0,
        regimes: Sequence[Regime] = REGIMES,
        dwell_scale: float = 1.0,
        hawkes_alpha: float = 0.4,
        hawkes_beta: float = 0.2,
        whale_p: float = 0.01,
        whale_mult: float = 30.0,
        snapshot_period_s: float = 1.0,
    ) -> None:
        if not regimes:
            raise ValueError("SimulatedMarket needs at least one regime")
        if price0 <= 0.0:
            raise ValueError("price0 must be > 0")
        self.seed = int(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(SEED_CHILD_MARKET + 1)[SEED_CHILD_MARKET])
        self.regimes: tuple[Regime, ...] = tuple(regimes)
        self._by_name: dict[str, int] = {r.name: i for i, r in enumerate(self.regimes)}
        self.price0 = float(price0)
        self.supply = float(supply)
        self.liq0 = float(liq0)
        self.dwell_scale = max(1e-6, float(dwell_scale))
        self.hawkes_alpha = float(hawkes_alpha)
        self.hawkes_beta = max(1e-9, float(hawkes_beta))
        #: Branching ratio (expected offspring per trade); ``< 1`` keeps the tape finite (module docstring).
        self.hawkes_branch = min(HAWKES_BRANCH_MAX, max(0.0, self.hawkes_alpha))
        #: Excitation jump per trade: the kernel ``eta*beta*exp(-beta*dt)`` integrates to ``eta``.
        self.hawkes_jump = self.hawkes_branch * self.hawkes_beta
        #: Immigrant rate per regime: ``lam0*(1 - eta)`` makes ``E[lam] == lam0``.
        self.hawkes_base_scale = 1.0 - self.hawkes_branch
        #: Excitation below the 60 s tail of a single trade is dropped (the c.15 "last 60 s" sum).
        self._exc_eps = self.hawkes_jump * math.exp(-self.hawkes_beta * HAWKES_WINDOW_S)
        self.whale_p = float(whale_p)
        self.whale_mult = float(whale_mult)
        self.snapshot_period_s = max(0.0, float(snapshot_period_s))
        # --- state ---
        self.ln_p: float = math.log(self.price0)
        self.liq: float = self.liq0
        self.regime_i: int = 0
        self._force_left_s: float = 0.0
        self._exc: float = 0.0  # Hawkes excitation (decayed to the last poll)
        self.lam: float = self.regimes[0].lam0  # intensity used by the last poll (trades/s)
        self._trades_h1: deque[tuple[float, bool, float]] = deque()
        self._trades_m5: deque[tuple[float, bool, float]] = deque()
        self._buys_h1 = 0
        self._sells_h1 = 0
        self._vol_h1 = 0.0
        self._buys_m5 = 0
        self._sells_m5 = 0
        self._vol_m5 = 0.0
        self._price_ts: list[float] = []
        self._price_v: list[float] = []
        self._last_snap_t: float | None = None
        self._seq: int = 0
        self._last_snapshot: MarketSnapshot | None = None
        self._n_polls: int = 0

    # ------------------------------------------------------------------ properties

    @property
    def regime(self) -> Regime:
        """The current regime."""
        return self.regimes[self.regime_i]

    @property
    def regime_name(self) -> str:
        return self.regimes[self.regime_i].name

    @property
    def price(self) -> float:
        return math.exp(self.ln_p)

    @property
    def forced_left_s(self) -> float:
        """Seconds left of a ``set_regime`` force (0 when Markov switching is active)."""
        return max(0.0, self._force_left_s)

    @property
    def last_snapshot(self) -> MarketSnapshot | None:
        return self._last_snapshot

    @property
    def seq(self) -> int:
        return self._seq

    # ------------------------------------------------------------------ MarketSource API

    def start(self) -> None:
        """No background work: the sim advances only inside ``poll``."""

    def stop(self) -> None:
        """Nothing to stop."""

    def force_snapshot(self) -> None:
        """Make the next ``poll`` emit a snapshot regardless of ``snapshot_period_s``.

        Used by ``MarketFeed`` when this source starts being served (fallback engaged / ``set_mode('sim')``) so
        that the feed can publish a body produced by *this* source instead of relabelling the previous source's
        (SPEC c.16 / d.2: ``regime`` must be non-null while the sim is the active source).
        """
        self._last_snap_t = None

    def set_regime(self, name: str, seconds: float) -> bool:
        """Force regime ``name`` for ``seconds`` of simulated time, then resume Markov switching. Unknown -> False."""
        idx = self._by_name.get(str(name).upper())
        if idx is None:
            return False
        self.regime_i = idx
        self._force_left_s = max(0.0, float(seconds))
        return True

    def poll(self, now: float, dt_s: float) -> tuple[MarketSnapshot | None, list[Trade]]:
        """Advance by ``dt_s`` seconds ending at wall time ``now`` (SPEC section c.15).

        Regime switch with ``P = dt/(dwell*dwell_scale)`` (uniform over the other regimes, suppressed while a
        ``set_regime`` force is active); ``ln P += mu*dt + sigma*sqrt(dt)*Z + impact``; trades ``~ Poisson(lam*dt)``
        with ``lam = lam0*(1 - eta) + exc`` (``eta = hawkes_alpha`` read as the branching ratio, so
        ``E[lam] == lam0``; module docstring); size ``~ LogNormal(ln 80, 1.2)`` USD (x ``whale_mult`` with
        probability ``whale_p``); impact ``+-2e-4*sqrt(usd/100)`` on ``ln P``; RUG: ``liq *= (1 - 0.02*dt)``.
        A new snapshot every ``snapshot_period_s`` from the 1-h trade ring (m5/h1 windows) and the price ring
        (``chg_*`` vs the price 5 min / 1 h / 6 h / 24 h ago or the earliest available).
        """
        now = float(now)
        dt = max(0.0, float(dt_s))
        self._n_polls += 1
        rng = self.rng

        # 1. regime switching -------------------------------------------------------------
        if self._force_left_s > 0.0:
            self._force_left_s -= dt
            if self._force_left_s <= 0.0:
                self._force_left_s = 0.0
            rng.random()  # keep the draw order identical whether or not a force is active
        else:
            reg = self.regimes[self.regime_i]
            p_switch = min(1.0, dt / (reg.dwell_s * self.dwell_scale)) if reg.dwell_s > 0 else 1.0
            u = rng.random()
            if len(self.regimes) > 1 and u < p_switch:
                others = [i for i in range(len(self.regimes)) if i != self.regime_i]
                self.regime_i = others[int(rng.integers(len(others)))]
        reg = self.regimes[self.regime_i]

        # 2. Hawkes intensity and trade count -------------------------------------------------
        self._exc *= math.exp(-self.hawkes_beta * dt)
        if self._exc < self._exc_eps:  # every contribution is older than HAWKES_WINDOW_S
            self._exc = 0.0
        lam = max(0.0, reg.lam0 * self.hawkes_base_scale + self._exc)
        self.lam = lam
        n_tr = int(rng.poisson(lam * dt)) if dt > 0.0 else 0

        # 3. trades ---------------------------------------------------------------------------
        trades: list[Trade] = []
        if n_tr > 0:
            ts = np.sort(now - dt + dt * rng.random(n_tr))
            usd = rng.lognormal(TRADE_USD_LN_MEDIAN, TRADE_USD_LN_SIGMA, n_tr)
            whale = rng.random(n_tr) < self.whale_p
            usd = np.where(whale, usd * self.whale_mult, usd)
            is_buy = rng.random(n_tr) < reg.p_buy
            self._exc += self.hawkes_jump * n_tr

        # 4. diffusion ------------------------------------------------------------------------
        z = float(rng.standard_normal())
        self.ln_p += reg.mu * dt + reg.sigma * math.sqrt(dt) * z

        if n_tr > 0:
            impact = IMPACT_COEF * np.sqrt(usd / IMPACT_USD_REF)
            for k in range(n_tr):
                buy = bool(is_buy[k])
                self.ln_p += float(impact[k]) if buy else -float(impact[k])
                price = math.exp(self.ln_p)
                u_k = float(usd[k])
                t_k = float(ts[k])
                trades.append(Trade(ts=t_k, kind="buy" if buy else "sell", usd=u_k, price=price, surrogate=False))
                rec = (t_k, buy, u_k)
                self._trades_h1.append(rec)
                self._trades_m5.append(rec)
                if buy:
                    self._buys_h1 += 1
                    self._buys_m5 += 1
                else:
                    self._sells_h1 += 1
                    self._sells_m5 += 1
                self._vol_h1 += u_k
                self._vol_m5 += u_k

        # 5. RUG liquidity drain --------------------------------------------------------------
        if reg.name == "RUG" and dt > 0.0:
            self.liq = self.liq * max(0.0, 1.0 - RUG_LIQ_DRAIN_PER_S * dt)

        # 6. ring maintenance -----------------------------------------------------------------
        self._prune(now)

        # 7. snapshot -------------------------------------------------------------------------
        snap: MarketSnapshot | None = None
        if self._last_snap_t is None or now - self._last_snap_t >= self.snapshot_period_s - 1e-9:
            snap = self._make_snapshot(now)
        return snap, trades

    # ------------------------------------------------------------------ internals

    def _prune(self, now: float) -> None:
        h1 = self._trades_h1
        lim = now - _H1_S
        while h1 and h1[0][0] <= lim:
            _, buy, u = h1.popleft()
            if buy:
                self._buys_h1 -= 1
            else:
                self._sells_h1 -= 1
            self._vol_h1 -= u
        m5 = self._trades_m5
        lim = now - _M5_S
        while m5 and m5[0][0] <= lim:
            _, buy, u = m5.popleft()
            if buy:
                self._buys_m5 -= 1
            else:
                self._sells_m5 -= 1
            self._vol_m5 -= u
        if not h1:
            self._vol_h1 = 0.0
        if not m5:
            self._vol_m5 = 0.0

    def _ref_price(self, target_ts: float) -> float:
        """Price at or before ``target_ts`` from the ring, or the earliest available."""
        i = bisect_right(self._price_ts, target_ts) - 1
        if i < 0:
            i = 0
        return self._price_v[i]

    def _make_snapshot(self, now: float) -> MarketSnapshot:
        self._seq += 1
        price = math.exp(self.ln_p)
        self._price_ts.append(now)
        self._price_v.append(price)
        if len(self._price_ts) > _PRICE_RING_MAX:
            del self._price_ts[:_PRICE_RING_PRUNE]
            del self._price_v[:_PRICE_RING_PRUNE]

        def chg(h: float) -> float:
            ref = self._ref_price(now - h)
            return (price / ref - 1.0) * 100.0 if ref > 0.0 else 0.0

        fdv = price * self.supply
        snap = MarketSnapshot(
            ts=now,
            source=SIM_SOURCE,
            chain="sim",
            dex="sim",
            pair="SIM",
            symbol="FLY",
            price_usd=price,
            price_native=price,
            buys_m5=int(self._buys_m5),
            sells_m5=int(self._sells_m5),
            buys_h1=int(self._buys_h1),
            sells_h1=int(self._sells_h1),
            chg_m5=chg(_M5_S),
            chg_h1=chg(_H1_S),
            chg_h6=chg(_H6_S),
            chg_h24=chg(_H24_S),
            vol_m5=max(0.0, float(self._vol_m5)),
            vol_h1=max(0.0, float(self._vol_h1)),
            liq_usd=float(self.liq),
            fdv=fdv,
            mcap=fdv,
            regime=self.regimes[self.regime_i].name,
            seq=self._seq,
        )
        self._last_snap_t = now
        self._last_snapshot = snap
        return snap
