"""DexScreener market source (SPEC section c.16): ``/tokens/v1/{chain}/{token}`` polling + surrogate trades.

DexScreener has no trade-level endpoint (RESEARCH section 8 ``[V]``): the source polls the pair every
``poll_s`` seconds (edge cache ~60 s) on a background thread and ``SurrogateTrades`` synthesises a
rate-matched Poisson stream of buy/sell events between polls (``lam = buys_m5 / 300`` per s) so the encoder
sees the same kind of tape as with the simulated market. When a fresh snapshot arrives, the deltas of the
5-minute counters are emitted first as real (``surrogate=False``) trades.

``fetch_pairs`` is the only network call in the package (httpx with timeouts, ``User-Agent: synapsefly/0.1``);
tests monkeypatch it (or ``httpx.Client`` with an ``httpx.MockTransport``). All numbers here are ``[E]``.
"""

from __future__ import annotations

import logging
import math
import threading
import time

import httpx
import numpy as np

from .base import DEX_SOURCE, MarketSnapshot, Trade

__all__ = [
    "DEX_BASE",
    "USER_AGENT",
    "fetch_pairs",
    "parse_pair",
    "choose_pair",
    "SurrogateTrades",
    "DexScreenerSource",
    "SURROGATE_WINDOW_S",
    "SURROGATE_USD_LN_SIGMA",
    "FAIL_RETRY_S",
]

log = logging.getLogger("flybrain.market.dexscreener")

DEX_BASE: str = "https://api.dexscreener.com"
USER_AGENT: str = "synapsefly/0.1"
#: The ``m5`` counters cover 300 s: ``lam_buy = buys_m5 / 300`` per second ``[E]`` (rate matching, SPEC c.16).
SURROGATE_WINDOW_S: float = 300.0
#: Log-normal spread of surrogate trade sizes around the mean ``vol_m5 / n_m5`` ``[E]``.
SURROGATE_USD_LN_SIGMA: float = 0.8
#: Retry period after a failed poll (the feed falls back after 3 consecutive failures, so ~45 s worst case) ``[E]``.
FAIL_RETRY_S: float = 15.0


# ---------------------------------------------------------------------------------- HTTP + parsing


def fetch_pairs(chain: str, token: str, timeout_s: float = 10.0) -> list[dict]:
    """GET ``{DEX_BASE}/tokens/v1/{chain}/{token}`` via httpx with ``User-Agent 'synapsefly/0.1'``.

    Returns the bare list of pair objects (``[]`` for unknown tokens is a valid answer, HTTP 200).
    Raises ``httpx.HTTPError`` on transport/status failure and ``ValueError`` when the body is not a JSON list
    (the ``/latest/dex/tokens`` shape ``{"pairs": [...]}`` is accepted for tolerance). Exactly the c.16
    signature: tests either monkeypatch this function or ``httpx.Client`` (``httpx.MockTransport``).
    """
    url = f"{DEX_BASE}/tokens/v1/{chain}/{token}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    with httpx.Client(timeout=float(timeout_s), headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError as exc:  # json decode error
            raise ValueError(f"DexScreener returned non-JSON body: {exc}") from exc
    if data is None:
        return []
    if isinstance(data, dict) and isinstance(data.get("pairs"), list):
        return list(data["pairs"])
    if not isinstance(data, list):
        raise ValueError(f"DexScreener returned {type(data).__name__}, expected a list of pairs")
    return [p for p in data if isinstance(p, dict)]


def _to_float(x: object) -> float | None:
    """Tolerant float: strings -> float, missing / null / non-finite -> None."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _to_float0(x: object) -> float:
    v = _to_float(x)
    return 0.0 if v is None else v


def _to_int0(x: object) -> int:
    """Tolerant non-negative counter: missing / null / unparsable -> 0, fractional truncated, negative clamped to 0.

    A negative ``txns`` count would otherwise reach the d.5 ``market`` body and make f.1's
    ``bp = (buys - sells)/(n + 1)`` leave -1..1 and ``log1p(n)`` raise on a negative ``n``.
    """
    v = _to_float(x)
    return 0 if v is None else max(0, int(v))


def _sub(d: object, key: str) -> dict:
    if isinstance(d, dict):
        v = d.get(key)
        if isinstance(v, dict):
            return v
    return {}


def parse_pair(p: dict, now: float, seq: int) -> MarketSnapshot:
    """Tolerant DexScreener pair -> ``MarketSnapshot`` (RESEARCH section 8 parsing rules).

    ``priceUsd`` / ``priceNative`` are strings -> float or None; ``txns`` / ``volume`` / ``priceChange``
    sub-keys default 0; ``liquidity.usd`` / ``fdv`` / ``marketCap`` absent -> None. ``source='dexscreener'``,
    ``dex=p['dexId']``, ``pair=p['pairAddress']``, ``symbol=p['baseToken']['symbol']``, ``chain=p['chainId']``.
    Transaction counts are coerced to non-negative ints (``_to_int0``) so the snapshot can never carry a
    negative / fractional count into the wire body or the f.1 formulas.
    """
    if not isinstance(p, dict):
        raise ValueError("pair must be a dict")
    txns = _sub(p, "txns")
    m5 = _sub(txns, "m5")
    h1 = _sub(txns, "h1")
    vol = _sub(p, "volume")
    pc = _sub(p, "priceChange")
    liq = _sub(p, "liquidity")
    base = _sub(p, "baseToken")
    symbol = base.get("symbol")
    return MarketSnapshot(
        ts=float(now),
        source=DEX_SOURCE,
        chain=str(p.get("chainId") or ""),
        dex=str(p.get("dexId") or ""),
        pair=str(p.get("pairAddress") or ""),
        symbol=str(symbol) if symbol else "?",
        price_usd=_to_float(p.get("priceUsd")),
        price_native=_to_float(p.get("priceNative")),
        buys_m5=_to_int0(m5.get("buys")),
        sells_m5=_to_int0(m5.get("sells")),
        buys_h1=_to_int0(h1.get("buys")),
        sells_h1=_to_int0(h1.get("sells")),
        chg_m5=_to_float0(pc.get("m5")),
        chg_h1=_to_float0(pc.get("h1")),
        chg_h6=_to_float0(pc.get("h6")),
        chg_h24=_to_float0(pc.get("h24")),
        vol_m5=_to_float0(vol.get("m5")),
        vol_h1=_to_float0(vol.get("h1")),
        liq_usd=_to_float(liq.get("usd")) if liq else None,
        fdv=_to_float(p.get("fdv")),
        mcap=_to_float(p.get("marketCap")),
        regime=None,
        seq=int(seq),
    )


def choose_pair(pairs: list[dict]) -> dict | None:
    """The pair with the largest ``liquidity.usd`` (missing -> 0); ``None`` for an empty list."""
    best: dict | None = None
    best_liq = -1.0
    for p in pairs:
        if not isinstance(p, dict):
            continue
        liq = _to_float(_sub(p, "liquidity").get("usd"))
        v = 0.0 if liq is None else liq
        if v > best_liq:
            best, best_liq = p, v
    return best


# ---------------------------------------------------------------------------------- surrogate trades


class SurrogateTrades:
    """Rate-matched Poisson surrogate of the tape between DexScreener polls (SPEC section c.16)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self._last_seq: int | None = None

    def _sizes(self, mean_usd: float, n: int) -> np.ndarray:
        mu = math.log(max(1.0, mean_usd)) - 0.5 * SURROGATE_USD_LN_SIGMA**2  # mean-matched log-normal
        return self.rng.lognormal(mu, SURROGATE_USD_LN_SIGMA, n)

    def _stamps(self, now: float, dt_s: float, n: int) -> np.ndarray:
        return np.sort(now - dt_s + dt_s * self.rng.random(n))

    def step(self, snap: MarketSnapshot, prev: MarketSnapshot | None, dt_s: float, now: float) -> list[Trade]:
        """Trades for the interval ``(now - dt_s, now]``.

        On a fresh snapshot (``snap.seq`` differs from the previous call and from ``prev``) the
        ``max(0, delta buys/sells since prev)`` real-delta trades (``surrogate=False``) come **first**, and the
        ordinary surrogate stream of this step follows them (c.16 says "first", not "instead": a fresh step
        whose counter delta happens to be 0 still owes the tick its ``lam*dt`` surrogates, otherwise the tape
        goes silent for one tick on every poll). ``lam_buy = buys_m5/300``, ``lam_sell = sells_m5/300`` per
        second, counts ``~ Poisson(lam*dt)``, sizes log-normal around
        ``vol_m5 / max(1, buys_m5 + sells_m5)``, ``price = snap.price_usd`` (or ``price_native``),
        ``surrogate=True``. The first call after construction has no ``prev`` to diff, so it is
        surrogate-only (there is no baseline the counters could be a delta against).
        """
        dt = max(0.0, float(dt_s))
        price = snap.price or 0.0
        fresh = snap.seq != self._last_seq
        self._last_seq = snap.seq
        mean_usd = snap.vol_m5 / max(1, snap.buys_m5 + snap.sells_m5)
        out: list[Trade] = []
        if fresh and prev is not None and prev.seq != snap.seq:
            db = max(0, int(snap.buys_m5) - int(prev.buys_m5))
            ds = max(0, int(snap.sells_m5) - int(prev.sells_m5))
            n = db + ds
            if n:
                kinds = np.array(["buy"] * db + ["sell"] * ds)
                kinds = kinds[self.rng.permutation(n)]
                usd = self._sizes(mean_usd, n)
                ts = self._stamps(now, dt, n)
                for k in range(n):
                    out.append(Trade(ts=float(ts[k]), kind=str(kinds[k]), usd=float(usd[k]), price=price, surrogate=False))
        if dt <= 0.0:
            return out
        lam_b = snap.buys_m5 / SURROGATE_WINDOW_S
        lam_s = snap.sells_m5 / SURROGATE_WINDOW_S
        nb = int(self.rng.poisson(lam_b * dt)) if lam_b > 0 else 0
        ns = int(self.rng.poisson(lam_s * dt)) if lam_s > 0 else 0
        n = nb + ns
        if n == 0:
            return out
        kinds = np.array(["buy"] * nb + ["sell"] * ns)
        kinds = kinds[self.rng.permutation(n)]
        usd = self._sizes(mean_usd, n)
        ts = self._stamps(now, dt, n)
        for k in range(n):
            out.append(Trade(ts=float(ts[k]), kind=str(kinds[k]), usd=float(usd[k]), price=price, surrogate=True))
        return out


# ---------------------------------------------------------------------------------- polling source


class DexScreenerSource:
    """Background-polled DexScreener pair with surrogate trades between polls (SPEC section c.16).

    ``start()`` launches a daemon thread that fetches immediately, then every ``poll_s`` seconds (``FAIL_RETRY_S``
    after a failure) and stores the latest snapshot under a lock; ``poll()`` (sim thread) hands out the latest
    snapshot once (by ``seq``) plus surrogate trades; ``failures`` counts consecutive failed fetches
    (reset on success) and drives the feed's fallback.
    """

    name: str = DEX_SOURCE

    def __init__(self, chain: str, token: str, poll_s: int = 60, seed: int = 0) -> None:
        self.chain = str(chain)
        self.token = str(token)
        self.poll_s: int = max(1, int(poll_s))  # seconds between fetches (c.16 types it int)
        self.seed = int(seed)
        # SPEC 0.1 child [2] is the market stream; the dex surrogate uses a sub-stream distinct from the sim's.
        child = np.random.SeedSequence(int(seed)).spawn(3)[2]
        self._rng = np.random.default_rng(child.spawn(2)[1])
        self._surrogate = SurrogateTrades(self._rng)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: MarketSnapshot | None = None
        self._served: MarketSnapshot | None = None
        self._seq = 0
        self._failures = 0
        self._last_error: str | None = None
        self._last_poll_wall: float | None = None
        self._n_ok = 0

    # --- background thread ---------------------------------------------------------------

    def fetch_once(self, now: float | None = None) -> MarketSnapshot | None:
        """One synchronous fetch + parse (used by the thread; callable directly by tests). Never raises."""
        wall = time.time() if now is None else float(now)
        try:
            pairs = fetch_pairs(self.chain, self.token)
            pair = choose_pair(pairs)
            if pair is None:
                raise ValueError(f"no pairs for token {self.token!r} on {self.chain!r}")
            snap = parse_pair(pair, wall, 0)
        except Exception as exc:  # noqa: BLE001 - every failure is counted, never propagated (SPEC 0.1)
            with self._lock:
                self._failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._last_poll_wall = wall
                n = self._failures
            log.warning("dexscreener poll failed (%d consecutive): %s", n, self._last_error)
            return None
        with self._lock:
            self._seq += 1
            snap.seq = self._seq
            self._latest = snap
            self._failures = 0
            self._last_error = None
            self._last_poll_wall = wall
            self._n_ok += 1
        log.info("dexscreener %s %s price_usd=%s buys_m5=%d sells_m5=%d liq=%s", snap.dex, snap.symbol,
                 snap.price_usd, snap.buys_m5, snap.sells_m5, snap.liq_usd)
        return snap

    def _run(self) -> None:
        while not self._stop.is_set():
            snap = self.fetch_once()
            wait = self.poll_s if snap is not None else min(self.poll_s, FAIL_RETRY_S)
            if self._stop.wait(wait):
                break

    def start(self) -> None:
        """Start the polling thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dexscreener-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._thread = None

    # --- sim-thread API --------------------------------------------------------------------

    def poll(self, now: float, dt_s: float) -> tuple[MarketSnapshot | None, list[Trade]]:
        """(latest snapshot if its ``seq`` is new, else None) plus ``SurrogateTrades.step(...)`` for ``dt_s``."""
        with self._lock:
            latest = self._latest
            prev = self._served
        if latest is None:
            return None, []
        new = latest if (prev is None or latest.seq != prev.seq) else None
        trades = self._surrogate.step(latest, prev, dt_s, now)
        with self._lock:
            self._served = latest
        return new, trades

    def resend(self) -> None:
        """Make the next ``poll`` hand out the latest snapshot again even though its ``seq`` is not new.

        Used by ``MarketFeed`` when this source starts being served again (fallback resolved /
        ``set_mode('dexscreener')``) so the feed can publish a real DexScreener body immediately instead of
        relabelling the simulated one.
        """
        with self._lock:
            self._served = None

    def set_regime(self, name: str, seconds: float) -> bool:
        """Real data has no regimes."""
        return False

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    @property
    def latest(self) -> MarketSnapshot | None:
        with self._lock:
            return self._latest

    @property
    def last_poll_wall(self) -> float | None:
        with self._lock:
            return self._last_poll_wall

    @property
    def ok(self) -> bool:
        """True once at least one snapshot has been fetched and the last poll succeeded."""
        with self._lock:
            return self._latest is not None and self._failures == 0

    def status(self) -> dict:
        """``/api/health`` market block helper: ``{mode, ok, last_poll, failures}``."""
        with self._lock:
            return {
                "mode": self.name,
                "ok": self._latest is not None and self._failures == 0,
                "last_poll": self._last_poll_wall,
                "failures": self._failures,
                "error": self._last_error,
            }
