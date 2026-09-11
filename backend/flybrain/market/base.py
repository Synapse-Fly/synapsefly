"""Market data types and the ``MarketSource`` protocol (SPEC section c.14).

``Trade`` is one buy/sell event (real or synthesised between DexScreener polls), ``MarketSnapshot`` is the
DexScreener-shaped state of one pair (the ``market`` WebSocket message body of SPEC section d.5) and
``MarketSource`` is the polling protocol shared by ``SimulatedMarket`` (section c.15) and
``DexScreenerSource`` (section c.16). Nothing here is biological; every field mirrors the DexScreener pair
object of RESEARCH section 8 (``[V]`` schema) so that the simulated market and the real one are
interchangeable downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "Trade",
    "MarketSnapshot",
    "MarketSource",
    "SNAPSHOT_WIRE_KEYS",
    "TRADE_WIRE_KEYS",
    "SIM_SOURCE",
    "DEX_SOURCE",
    "FALLBACK_SOURCE",
]

#: ``MarketSnapshot.source`` values (SPEC section c.14).
SIM_SOURCE: str = "sim"
DEX_SOURCE: str = "dexscreener"
FALLBACK_SOURCE: str = "sim(fallback)"

#: Key order of ``MarketSnapshot.to_wire()`` (SPEC section d.5 ``market`` body; ``mode`` / ``last_trade`` are added by the loop).
SNAPSHOT_WIRE_KEYS: tuple[str, ...] = (
    "source", "ts", "seq", "chain", "dex", "pair", "symbol",
    "price_usd", "price_native",
    "buys_m5", "sells_m5", "buys_h1", "sells_h1",
    "chg_m5", "chg_h1", "chg_h6", "chg_h24",
    "vol_m5", "vol_h1",
    "liq_usd", "fdv", "mcap",
    "regime",
)

#: Key order of ``Trade.to_wire()`` (SPEC section d.5 ``trades`` entries).
TRADE_WIRE_KEYS: tuple[str, ...] = ("kind", "usd", "ts", "surrogate")


@dataclass(frozen=True)
class Trade:
    """One buy/sell event.

    ``ts`` wall seconds; ``kind`` ``"buy" | "sell"``; ``usd`` notional; ``price`` price_usd at the trade;
    ``surrogate`` True when synthesised between DexScreener polls (SPEC section c.16).
    """

    ts: float
    kind: str
    usd: float
    price: float
    surrogate: bool = False

    def to_wire(self) -> dict:
        """The ``trades[]`` entry of the ``market`` frame: ``{kind, usd, ts, surrogate}``."""
        return {"kind": self.kind, "usd": float(self.usd), "ts": float(self.ts), "surrogate": bool(self.surrogate)}


@dataclass(slots=True)
class MarketSnapshot:
    """DexScreener-shaped market state (SPEC section c.14).

    ``source`` ``"sim" | "dexscreener" | "sim(fallback)"``; ``chg_*`` are percent; ``price_usd`` /
    ``price_native`` / ``liq_usd`` / ``fdv`` / ``mcap`` may be ``None`` when the upstream pair omits them;
    ``regime`` is set by the simulated market only (``CALM|PUMP|DUMP|CHOP|RUG|DEAD``); ``seq`` increments
    on every new snapshot.
    """

    ts: float
    source: str
    chain: str
    dex: str
    pair: str
    symbol: str
    price_usd: float | None
    price_native: float | None
    buys_m5: int
    sells_m5: int
    buys_h1: int
    sells_h1: int
    chg_m5: float
    chg_h1: float
    chg_h6: float
    chg_h24: float
    vol_m5: float
    vol_h1: float
    liq_usd: float | None
    fdv: float | None
    mcap: float | None
    regime: str | None = None
    seq: int = 0

    @property
    def price(self) -> float | None:
        """``price_usd`` or, when missing, ``price_native`` (the ``P`` of SPEC section f.1)."""
        if self.price_usd is not None and self.price_usd > 0.0:
            return float(self.price_usd)
        if self.price_native is not None and self.price_native > 0.0:
            return float(self.price_native)
        return None

    def to_wire(self) -> dict:
        """The ``market`` WS message body (SPEC section d.5), keys in ``SNAPSHOT_WIRE_KEYS`` order."""
        return {
            "source": self.source,
            "ts": float(self.ts),
            "seq": int(self.seq),
            "chain": self.chain,
            "dex": self.dex,
            "pair": self.pair,
            "symbol": self.symbol,
            "price_usd": None if self.price_usd is None else float(self.price_usd),
            "price_native": None if self.price_native is None else float(self.price_native),
            "buys_m5": int(self.buys_m5),
            "sells_m5": int(self.sells_m5),
            "buys_h1": int(self.buys_h1),
            "sells_h1": int(self.sells_h1),
            "chg_m5": float(self.chg_m5),
            "chg_h1": float(self.chg_h1),
            "chg_h6": float(self.chg_h6),
            "chg_h24": float(self.chg_h24),
            "vol_m5": float(self.vol_m5),
            "vol_h1": float(self.vol_h1),
            "liq_usd": None if self.liq_usd is None else float(self.liq_usd),
            "fdv": None if self.fdv is None else float(self.fdv),
            "mcap": None if self.mcap is None else float(self.mcap),
            "regime": self.regime,
        }


@runtime_checkable
class MarketSource(Protocol):
    """Polling market source (SPEC section c.14). Implemented by ``SimulatedMarket`` and ``DexScreenerSource``."""

    name: str

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def poll(self, now: float, dt_s: float) -> tuple[MarketSnapshot | None, list[Trade]]:
        """Non-blocking. Returns (new snapshot or None if unchanged, trades since the last call)."""
        ...

    def set_regime(self, name: str, seconds: float) -> bool:
        """Sim only; other sources return False."""
        ...
