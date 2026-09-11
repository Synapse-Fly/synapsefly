"""``MarketFeed``: the active market source with automatic DexScreener -> sim fallback (SPEC section c.16).

Not a thread itself (``DexScreenerSource`` owns the HTTP thread); ``poll`` runs on the sim thread and never
blocks. ``market='sim'`` serves ``SimulatedMarket(seed, dwell_scale=sim_regime_s/90)``; ``market='dexscreener'``
serves the DexScreener source and, after ``FALLBACK_FAILURES`` consecutive failed polls, the simulated market
labelled ``source='sim(fallback)'`` with a ``market_source`` event, resuming on the next success. The sim keeps
advancing in every mode so a fallback starts from a live tape. ``seq`` is re-stamped by the feed so that it
increases monotonically across source switches (the loop emits a ``market`` frame on every change).

A snapshot body is **never** relabelled: every published snapshot carries the source that produced it (a
DexScreener body relabelled ``sim`` would put the live chain / symbol / price / liquidity on the simulated
series, leave ``regime`` null while the sim is active - d.2 requires it non-null - and make the f.1
per-source reset capture the wrong ``liq_start``, which pins ``rug_snap`` at 1.0 forever). Instead, whenever the
served source changes, the incoming source is asked to emit a body immediately
(``SimulatedMarket.force_snapshot`` / ``DexScreenerSource.resend``); until it does, the previous snapshot keeps
its own label and ``mode`` reports the label of the snapshot actually being served.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from .base import DEX_SOURCE, FALLBACK_SOURCE, SIM_SOURCE, MarketSnapshot, Trade
from .dexscreener import DexScreenerSource
from .sim import REGIME_NAMES, SimulatedMarket

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings

__all__ = ["MarketFeed", "MARKET_MODES", "FALLBACK_FAILURES", "FORCED_REGIME_S", "SIM_REGIME_REF_S"]

log = logging.getLogger("flybrain.market.feed")

#: ``hello.market_modes`` / ``set_market_mode.mode`` (SPEC section d.1).
MARKET_MODES: tuple[str, ...] = ("sim", "dexscreener") + REGIME_NAMES
#: Consecutive DexScreener failures before the sim fallback is served ``[E]`` (SPEC section b, ``FLY_MARKET``).
FALLBACK_FAILURES: int = 3
#: A regime name via ``set_mode`` forces the sim regime for this long ``[E]`` (SPEC section d.7).
FORCED_REGIME_S: float = 60.0
#: ``dwell_scale = sim_regime_s / 90`` (90 s = the CALM dwell of the regime table) ``[E]``.
SIM_REGIME_REF_S: float = 90.0


def _event(source: str, reason: str) -> dict:
    """``market_source`` event in the uniform d.8 shape ``{kind, data: {mode, reason}}``.

    ``source`` / ``reason`` are kept as top-level aliases for the literal c.16 wording; a consumer doing the
    uniform ``bus.publish_event(kind=e['kind'], data=e['data'])`` (SPEC c.27 step 10) only needs ``data``.
    """
    return {"kind": "market_source", "data": {"mode": source, "reason": reason}, "source": source, "reason": reason}


class MarketFeed:
    """Owns the active source with fallback (SPEC section c.16)."""

    def __init__(self, settings: "Settings", seed: int) -> None:
        self.settings = settings
        self.seed = int(seed)
        dwell_scale = float(getattr(settings, "sim_regime_s", SIM_REGIME_REF_S)) / SIM_REGIME_REF_S
        self.sim = SimulatedMarket(int(seed), dwell_scale=dwell_scale)
        self.dex: DexScreenerSource | None = None
        self._want_dex = False
        self._fallback = False
        self._started = False
        self._seq = 0
        self._last: MarketSnapshot | None = None
        self._served_label: str | None = None  # source label of the snapshot currently served
        self._pending_events: list[dict] = []
        market = str(getattr(settings, "market", "sim"))
        if market == "dexscreener":
            token = str(getattr(settings, "token_address", "") or "")
            if token:
                self.dex = self._make_dex()
                self._want_dex = True
            else:
                log.warning("FLY_MARKET=dexscreener but FLY_TOKEN_ADDRESS is empty: serving the simulated market")
                self._pending_events.append(_event(SIM_SOURCE, "FLY_TOKEN_ADDRESS empty; dexscreener disabled"))

    # ------------------------------------------------------------------ lifecycle

    def _make_dex(self) -> DexScreenerSource:
        s = self.settings
        return DexScreenerSource(
            chain=str(getattr(s, "chain", "solana")),
            token=str(getattr(s, "token_address", "")),
            poll_s=int(getattr(s, "dex_poll_s", 60)),
            seed=self.seed,
        )

    def start(self) -> None:
        self._started = True
        self.sim.start()
        if self.dex is not None:
            self.dex.start()

    def stop(self) -> None:
        self._started = False
        if self.dex is not None:
            self.dex.stop()
        self.sim.stop()

    # ------------------------------------------------------------------ polling

    def _target_label(self) -> str:
        """The label the next published snapshot will carry (``"sim" | "dexscreener" | "sim(fallback)"``)."""
        if self._want_dex and self.dex is not None:
            if self._fallback or self.dex.latest is None:
                return FALLBACK_SOURCE
            return DEX_SOURCE
        return SIM_SOURCE

    @property
    def mode(self) -> str:
        """``"sim" | "dexscreener" | "sim(fallback)"`` - the source of the snapshot currently being served
        (before the first poll: the source that will be served)."""
        return self._served_label if self._served_label is not None else self._target_label()

    @property
    def last_snapshot(self) -> MarketSnapshot | None:
        return self._last

    @property
    def fallback_active(self) -> bool:
        return self._fallback

    def _stamp(self, snap: MarketSnapshot, label: str) -> MarketSnapshot:
        self._seq += 1
        return replace(snap, source=label, seq=self._seq)

    def poll(self, now: float, dt_s: float) -> tuple[MarketSnapshot, list[Trade], list[dict]]:
        """Always returns a snapshot (the last known one, never None after the first poll), new trades, events.

        The snapshot is always a body produced by the source named in ``snapshot.source``; a source switch asks
        the incoming source for a body on this very tick (``force_snapshot`` / ``resend``) and the
        ``market_source`` event is emitted together with the first body of the new source.
        """
        events: list[dict] = []
        if self._pending_events:
            events.extend(self._pending_events)
            self._pending_events.clear()

        reason: str | None = None
        announce = False  # a fallback engaged / resolved is always announced, even if the label is unchanged
        label = SIM_SOURCE
        snap: MarketSnapshot | None
        trades: list[Trade]
        if self._want_dex and self.dex is not None:
            fails = self.dex.failures
            if not self._fallback and fails >= FALLBACK_FAILURES:
                self._fallback = True
                announce = True
                reason = f"{fails} consecutive DexScreener failures: {self.dex.last_error or 'unknown error'}"
                self.sim.force_snapshot()  # publish a real sim body now, never a relabelled DexScreener one
            elif self._fallback and fails == 0 and self.dex.latest is not None:
                self._fallback = False
                announce = True
                reason = "DexScreener poll succeeded; resuming live data"
                self.dex.resend()
            sim_snap, sim_trades = self.sim.poll(now, dt_s)
            dex_snap, dex_trades = self.dex.poll(now, dt_s)
            if self._fallback or self.dex.latest is None:
                snap, trades, label = sim_snap, sim_trades, FALLBACK_SOURCE
                if reason is None:
                    reason = "DexScreener has no data yet; serving the simulated market"
            else:
                snap, trades, label = dex_snap, dex_trades, DEX_SOURCE
                if reason is None:
                    reason = "DexScreener data available; serving live data"
        else:
            snap, trades = self.sim.poll(now, dt_s)
            label = SIM_SOURCE
            if reason is None:
                reason = "simulated market selected"

        if snap is not None:
            self._last = self._stamp(snap, label)
            first = self._served_label is None
            changed = label != self._served_label
            self._served_label = label
            # the very first snapshot of a session is not a source *change*, but a fallback that engaged or
            # resolved is reported even when the served label itself did not move (c.16)
            if announce or (changed and not first):
                log.info("market source -> %s (%s)", label, reason)
                events.append(_event(label, reason or ""))
        elif announce:  # pragma: no cover - force_snapshot / resend always yield a body on the switching tick
            self._pending_events.append(_event(label, reason or ""))
        if self._last is None:  # pragma: no cover - the sim always yields a snapshot on its first poll
            raise RuntimeError("MarketFeed.poll produced no snapshot")
        return self._last, trades, events

    # ------------------------------------------------------------------ control

    def set_mode(self, mode: str) -> bool:
        """``'sim' | 'dexscreener' | CALM|PUMP|DUMP|CHOP|RUG|DEAD`` (regime names force the sim for 60 s)."""
        m = str(mode)
        if m == SIM_SOURCE:
            if self._want_dex:
                log.info("market mode -> sim")
                self.sim.force_snapshot()  # serve a sim body on the next poll (never a relabelled dex one)
                if self.dex is not None:
                    self.dex.stop()  # no point polling (and counting failures) while the sim is served
            self._want_dex = False
            self._fallback = False
            return True
        if m == DEX_SOURCE:
            token = str(getattr(self.settings, "token_address", "") or "")
            if not token:
                log.warning("market mode dexscreener refused: FLY_TOKEN_ADDRESS is empty")
                return False
            if self.dex is None:
                self.dex = self._make_dex()
                if self._started:
                    self.dex.start()
            else:
                if self._started:
                    self.dex.start()
                if self.dex.latest is not None:
                    self.dex.resend()  # serve the live body again instead of relabelling the sim's
            self._want_dex = True
            self._fallback = False
            log.info("market mode -> dexscreener")
            return True
        up = m.upper()
        if up in REGIME_NAMES:
            ok = self.sim.set_regime(up, FORCED_REGIME_S)
            if ok:
                log.info("sim regime forced -> %s for %.0f s", up, FORCED_REGIME_S)
            return ok
        return False

    def hello(self) -> dict:
        """The ``hello.market`` block (SPEC d.1): ``{mode, chain, symbol, token, poll_s}``.

        Describes the source actually being served: while the sim tape is on the wire (``sim`` or
        ``sim(fallback)``) the block reports the sim's own chain / symbol / snapshot period, never the
        configured DexScreener pair - otherwise the client ticker would label the simulated series with the
        real chain and a 60 s poll period.
        """
        s = self.settings
        mode = self.mode
        if mode == DEX_SOURCE and self.dex is not None:
            latest = self.dex.latest
            return {"mode": mode, "chain": str(getattr(s, "chain", "solana")),
                    "symbol": latest.symbol if latest is not None else "?",
                    "token": str(getattr(s, "token_address", "") or ""), "poll_s": int(self.dex.poll_s)}
        return {"mode": mode, "chain": "sim", "symbol": "FLY", "token": "",
                "poll_s": max(1, int(round(self.sim.snapshot_period_s)))}

    def status(self) -> dict:
        """``/api/health`` market block: ``{mode, ok, last_poll, failures}``."""
        if self.dex is not None and self._want_dex:
            d = self.dex.status()
            return {"mode": self.mode, "ok": bool(d["ok"]) and not self._fallback, "last_poll": d["last_poll"],
                    "failures": int(d["failures"])}
        last = None if self._last is None else float(self._last.ts)
        return {"mode": self.mode, "ok": True, "last_poll": last, "failures": 0}
