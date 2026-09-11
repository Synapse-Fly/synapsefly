"""SPEC section h.2 ``test_market.py``: simulated market invariants, DexScreener parsing (no network),
surrogate trades, feed fallback. Statistical tests assert invariants (regime dwell, buy share, price
positivity, trade rates), never exact values; every stream is seeded so the run is deterministic.

The ``sim_snapshot`` and ``dex_pair_json`` fixtures of SPEC h.1 (``tests/conftest.py``) are used as-is - this
file defines neither, so a fix to the canonical fixtures is picked up here.
"""

from __future__ import annotations

import copy
import math
import statistics
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from flybrain.market import dexscreener as dexmod
from flybrain.market.base import SNAPSHOT_WIRE_KEYS, MarketSnapshot, MarketSource, Trade
from flybrain.market.dexscreener import (
    DexScreenerSource,
    SurrogateTrades,
    choose_pair,
    fetch_pairs,
    parse_pair,
)
from flybrain.market.feed import FALLBACK_FAILURES, FORCED_REGIME_S, MARKET_MODES, MarketFeed
from flybrain.market.sim import REGIME_NAMES, REGIMES, SimulatedMarket

T0 = 1_789_051_563.0


def make_settings(**kw) -> SimpleNamespace:
    base = dict(market="sim", token_address="", chain="solana", dex_poll_s=60, sim_regime_s=90.0, seed=1337)
    base.update(kw)
    return SimpleNamespace(**base)


# ------------------------------------------------------------------------------------------- sim


def _run_sim(sim: SimulatedMarket, seconds: float, dt: float = 1.0, t0: float = T0):
    """Poll for ``seconds`` at ``dt``; returns (snapshots, trades, regime-per-poll)."""
    snaps, trades, regs = [], [], []
    n = int(round(seconds / dt))
    now = t0
    for _ in range(n):
        now += dt
        s, tr = sim.poll(now, dt)
        regs.append(sim.regime_name)
        if s is not None:
            snaps.append(s)
        trades.extend(tr)
    return snaps, trades, regs


def test_sim_regime_dwell_means():
    """Mean dwell of each regime within +-30 % of ``dwell_s`` (pooled over a long run and several seeds)."""
    dwells: dict[str, list[float]] = {r.name: [] for r in REGIMES}
    dt = 1.0
    for seed in (1, 2, 3, 4, 5):
        sim = SimulatedMarket(seed=seed, snapshot_period_s=1e9)  # no snapshots: pure regime/trade dynamics
        _, _, regs = _run_sim(sim, 10_000.0, dt=dt)
        cur, length = regs[0], 0
        for r in regs:
            if r == cur:
                length += 1
            else:
                dwells[cur].append(length * dt)
                cur, length = r, 1
        # the final open visit is censored -> not counted
    for reg in REGIMES:
        obs = dwells[reg.name]
        assert len(obs) >= 20, f"{reg.name}: only {len(obs)} completed visits"
        mean = statistics.fmean(obs)
        assert abs(mean - reg.dwell_s) <= 0.30 * reg.dwell_s, f"{reg.name}: mean dwell {mean:.1f} vs {reg.dwell_s}"


def test_sim_buy_probability_per_regime():
    for name, p_buy in (("PUMP", 0.82), ("DUMP", 0.18)):
        sim = SimulatedMarket(seed=7, snapshot_period_s=1e9)
        assert sim.set_regime(name, 1e9)
        _, trades, regs = _run_sim(sim, 4000.0, dt=1.0)
        assert set(regs) == {name}
        assert len(trades) > 1000
        share = sum(1 for t in trades if t.kind == "buy") / len(trades)
        assert abs(share - p_buy) <= 0.05, f"{name}: buy share {share:.3f}"
        assert all(t.price > 0 and t.usd > 0 and not t.surrogate for t in trades)


def test_sim_snapshot_windows():
    sim = SimulatedMarket(seed=11, snapshot_period_s=1.0)
    sim.set_regime("CHOP", 1e9)
    snaps, trades, _ = _run_sim(sim, 900.0, dt=0.5)
    assert snaps, "no snapshots"
    last = snaps[-1]
    now = last.ts
    win = [t for t in trades if t.ts > now - 300.0]
    assert last.buys_m5 == sum(1 for t in win if t.kind == "buy")
    assert last.sells_m5 == sum(1 for t in win if t.kind == "sell")
    h1 = [t for t in trades if t.ts > now - 3600.0]
    assert last.buys_h1 == sum(1 for t in h1 if t.kind == "buy")
    assert last.vol_m5 == pytest.approx(sum(t.usd for t in win), rel=1e-6, abs=1e-6)
    # chg_m5 from the price ring: price at or before now - 300 s
    ref = [s for s in snaps if s.ts <= now - 300.0][-1]
    assert last.chg_m5 == pytest.approx((last.price_usd / ref.price_usd - 1.0) * 100.0, rel=1e-9, abs=1e-9)
    assert all(s.price_usd > 0 for s in snaps)
    assert all(s.seq == i + 1 for i, s in enumerate(snaps))
    assert last.symbol == "FLY" and last.chain == "sim" and last.source == "sim" and last.regime == "CHOP"


def test_sim_trade_rate_matches_lam0_per_regime():
    """c.15 ``lam0`` is the regime's trades/s baseline: the Hawkes excitation rides on it, it does not scale it.

    Regression: with ``lam = lam0 + min(exc, 3*lam0)`` the (supercritical) excitation saturated the cap on
    ~99 % of polls, so every regime ran at ~4x ``lam0`` - CALM's f.1 ``activity`` became 0.94 instead of 0.73,
    DEAD's 0.39 instead of 0.31 (above the 0.35 SLEEP gate of f.6) and ``buys_m5`` was inflated 4x on the wire.
    """
    seconds, dt = 3000.0, 1.0
    for reg in REGIMES:
        sim = SimulatedMarket(seed=23, snapshot_period_s=1e9)
        assert sim.set_regime(reg.name, 1e9)
        _, trades, regs = _run_sim(sim, seconds, dt=dt)
        assert set(regs) == {reg.name}
        rate = len(trades) / seconds
        assert abs(rate - reg.lam0) <= 0.25 * reg.lam0, f"{reg.name}: {rate:.4f} trades/s vs lam0 {reg.lam0}"
    # clustering is still there: 20 s trade counts are overdispersed versus Poisson (Fano factor > 1)
    sim = SimulatedMarket(seed=11, snapshot_period_s=1e9)
    sim.set_regime("CALM", 1e9)
    _, trades, _ = _run_sim(sim, 20_000.0, dt=1.0)
    bins = [0] * 1000
    for tr in trades:
        k = int((tr.ts - T0) // 20)
        if 0 <= k < len(bins):
            bins[k] += 1
    mean, var = statistics.fmean(bins), statistics.pvariance(bins)
    assert var / mean > 1.25, f"Fano factor {var / mean:.2f}: the Hawkes clustering is gone"


def test_sim_dead_activity_below_sleep_threshold():
    """DEAD (``lam0 = 0.02``) must average ~6 trades per 5 min so f.1's ``activity`` lands near 0.31, below the
    0.35 that f.6's SLEEP gate requires (the count is noisy at 0.02/s, so the mean is pooled over seeds)."""
    per_seed = []
    for seed in range(1, 9):
        sim = SimulatedMarket(seed=seed, snapshot_period_s=1.0)
        assert sim.set_regime("DEAD", 1e9)
        snaps, _, _ = _run_sim(sim, 3000.0, dt=1.0)
        warm = [s.buys_m5 + s.sells_m5 for s in snaps if s.ts - T0 > 600.0]  # after the m5 window filled
        per_seed.append(statistics.fmean(warm))
    pooled = statistics.fmean(per_seed)
    assert 4.0 <= pooled <= 8.5, f"DEAD n_m5 mean {pooled:.2f} (spec lam0*300 = 6)"
    activity = math.log1p(pooled) / math.log1p(500.0)
    assert activity < 0.35, f"DEAD activity {activity:.3f} >= the f.6 SLEEP threshold"


def test_sim_rug_drains_liquidity():
    sim = SimulatedMarket(seed=3, liq0=50_000.0, snapshot_period_s=1.0)
    liq0 = sim.liq
    sim.set_regime("RUG", 1e9)
    snaps, _, _ = _run_sim(sim, 100.0, dt=1.0)
    # liq *= (1 - 0.02*dt) per second -> 0.98**100 ~ 0.133
    assert snaps[-1].liq_usd == pytest.approx(liq0 * 0.98 ** 100, rel=1e-6)
    assert snaps[-1].liq_usd < 0.2 * liq0
    liqs = [s.liq_usd for s in snaps]
    assert all(b <= a for a, b in zip(liqs, liqs[1:]))
    assert snaps[-1].price_usd < snaps[0].price_usd  # strong negative drift
    # CALM keeps liquidity
    sim2 = SimulatedMarket(seed=3)
    sim2.set_regime("CALM", 1e9)
    snaps2, _, _ = _run_sim(sim2, 100.0, dt=1.0)
    assert snaps2[-1].liq_usd == pytest.approx(sim2.liq0)


def test_sim_set_regime_forces_then_resumes():
    sim = SimulatedMarket(seed=5, snapshot_period_s=1e9)
    assert sim.set_regime("DEAD", 30.0) is True
    assert sim.set_regime("NOPE", 30.0) is False
    _, _, regs = _run_sim(sim, 30.0, dt=0.5)
    assert set(regs) == {"DEAD"}
    assert sim.forced_left_s == 0.0
    _, _, regs2 = _run_sim(sim, 3000.0, dt=1.0)
    assert len(set(regs2)) > 1, "Markov switching did not resume after the forced window"
    assert sim.set_regime("pump", 5.0) is True  # case-insensitive
    assert sim.regime_name == "PUMP"


def test_sim_deterministic():
    a = SimulatedMarket(seed=42)
    b = SimulatedMarket(seed=42)
    sa, ta, ra = _run_sim(a, 600.0, dt=0.05)
    sb, tb, rb = _run_sim(b, 600.0, dt=0.05)
    assert ra == rb
    assert [s.to_wire() for s in sa] == [s.to_wire() for s in sb]
    assert ta == tb
    assert len(ta) > 50
    c = SimulatedMarket(seed=43)
    sc, tc, _ = _run_sim(c, 600.0, dt=0.05)
    assert [s.price_usd for s in sc] != [s.price_usd for s in sa]
    assert all(s.price_usd > 0 for s in sa)


def test_sim_is_market_source():
    assert isinstance(SimulatedMarket(seed=1), MarketSource)
    assert isinstance(DexScreenerSource("solana", "x", seed=1), MarketSource)
    assert REGIME_NAMES == ("CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD")


# ------------------------------------------------------------------------------------------- dexscreener


def test_dex_parse_pair_sample(dex_pair_json):
    s = parse_pair(dex_pair_json, now=T0, seq=3)
    assert s.price_usd == pytest.approx(100.09)
    assert s.price_native == pytest.approx(100.09019)
    assert s.buys_m5 == 648 and s.sells_m5 == 656
    assert s.buys_h1 == 26649 and s.sells_h1 == 27601
    assert s.chg_m5 == pytest.approx(0.46) and s.chg_h24 == pytest.approx(-3.39)
    assert s.vol_m5 == pytest.approx(83071.16) and s.vol_h1 == pytest.approx(3444547.17)
    assert s.liq_usd == pytest.approx(25108908.33)
    assert s.fdv is None and s.mcap is None
    assert s.source == "dexscreener" and s.chain == "solana" and s.dex == "raydium"
    assert s.pair == "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2" and s.symbol == "SOL"
    assert s.regime is None and s.seq == 3 and s.ts == T0
    assert s.price == pytest.approx(100.09)


def test_dex_parse_pair_missing_fields(dex_pair_json):
    p = dex_pair_json
    del p["priceChange"]
    p["liquidity"] = None
    del p["priceUsd"]
    del p["txns"]["h1"]
    s = parse_pair(p, now=T0, seq=1)
    assert s.price_usd is None
    assert s.price_native == pytest.approx(100.09019)
    assert s.price == pytest.approx(100.09019)  # falls back to native
    assert s.chg_m5 == 0.0 and s.chg_h1 == 0.0 and s.chg_h6 == 0.0 and s.chg_h24 == 0.0
    assert s.liq_usd is None and s.fdv is None and s.mcap is None
    assert s.buys_h1 == 0 and s.sells_h1 == 0 and s.buys_m5 == 648
    # bare minimum object
    s2 = parse_pair({"chainId": "base"}, now=T0, seq=2)
    assert s2.price_usd is None and s2.buys_m5 == 0 and s2.vol_m5 == 0.0 and s2.symbol == "?"
    assert s2.dex == "" and s2.pair == "" and s2.chain == "base"
    with pytest.raises(ValueError):
        parse_pair("not a dict", now=T0, seq=1)  # type: ignore[arg-type]


def test_dex_parse_pair_clamps_bad_counts():
    """c.14 types the counters ``int`` and f.1 needs ``n >= 0``: negative / fractional ``txns`` are clamped.

    Regression: a negative count reached the d.5 body, where f.1 would compute ``bp = -7/-2 = 3.5`` (outside
    -1..1) and ``math.log1p(-3)`` raises.
    """
    s = parse_pair({"chainId": "x", "txns": {"m5": {"buys": -5, "sells": 2.9},
                                             "h1": {"buys": "-12", "sells": None}}}, now=T0, seq=1)
    assert s.buys_m5 == 0 and s.sells_m5 == 2
    assert s.buys_h1 == 0 and s.sells_h1 == 0
    w = s.to_wire()
    assert w["buys_m5"] == 0 and isinstance(w["sells_m5"], int)
    n = s.buys_m5 + s.sells_m5
    assert -1.0 <= (s.buys_m5 - s.sells_m5) / (n + 1) <= 1.0 and math.log1p(n) >= 0.0


def test_choose_pair_max_liquidity(dex_pair_json):
    a = dex_pair_json
    b = copy.deepcopy(a)
    b["pairAddress"] = "B"
    b["liquidity"] = {"usd": 99_999_999.0}
    c = copy.deepcopy(a)
    c["pairAddress"] = "C"
    del c["liquidity"]
    assert choose_pair([a, c, b])["pairAddress"] == "B"
    assert choose_pair([c])["pairAddress"] == "C"
    assert choose_pair([]) is None
    assert choose_pair([c, {"pairAddress": "D", "liquidity": None}])["pairAddress"] == "C"


def test_fetch_pairs_mock_transport(monkeypatch, dex_pair_json):
    """``fetch_pairs(chain, token, timeout_s)`` is exactly the c.16 signature, so the HTTP layer is mocked by
    patching ``httpx.Client`` with an ``httpx.MockTransport`` rather than by a test-only parameter."""
    seen: dict = {}
    real_client = httpx.Client

    def patch(handler) -> None:
        def client(**kwargs):
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(dexmod.httpx, "Client", client)

    def ok(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["ua"] = request.headers.get("user-agent")
        seen["timeout"] = request.extensions.get("timeout", {}).get("connect")
        return httpx.Response(200, json=[dex_pair_json])

    patch(ok)
    pairs = fetch_pairs("solana", "So111", timeout_s=3.0)
    assert len(pairs) == 1 and pairs[0]["dexId"] == "raydium"
    assert seen["url"] == "https://api.dexscreener.com/tokens/v1/solana/So111"
    assert seen["ua"] == "synapsefly/0.1"
    assert seen["timeout"] == pytest.approx(3.0)
    # unknown token -> [] with HTTP 200 is a valid answer
    patch(lambda r: httpx.Response(200, json=[]))
    assert fetch_pairs("solana", "nope") == []
    # the /latest/dex/tokens shape is tolerated
    patch(lambda r: httpx.Response(200, json={"pairs": [dex_pair_json]}))
    assert len(fetch_pairs("solana", "So111")) == 1
    patch(lambda r: httpx.Response(500, text="boom"))
    with pytest.raises(httpx.HTTPError):
        fetch_pairs("solana", "x")
    patch(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(ValueError):
        fetch_pairs("solana", "x")


def test_surrogate_trades_rate_matched(sim_snapshot):
    rng = np.random.default_rng(0)
    sur = SurrogateTrades(rng)
    snap = sim_snapshot(1, buys_m5=300, sells_m5=30, vol_m5=33_000.0, source="dexscreener")
    dt = 0.05
    trades: list[Trade] = []
    now = T0
    for _ in range(int(100.0 / dt)):
        now += dt
        trades.extend(sur.step(snap, snap, dt, now))
    buys = [t for t in trades if t.kind == "buy"]
    sells = [t for t in trades if t.kind == "sell"]
    assert abs(len(buys) - 100) <= 20, f"{len(buys)} buys in 100 s (expected ~100)"
    assert abs(len(sells) - 10) <= 8
    assert all(t.surrogate for t in trades)
    assert all(t.price == pytest.approx(snap.price_usd) for t in trades)
    assert all(T0 < t.ts <= now + 1e-9 for t in trades)
    assert all(t.usd > 0 for t in trades)
    mean_usd = np.mean([t.usd for t in trades])
    assert 30.0 < mean_usd < 300.0  # log-normal around 33000/330 = 100 USD
    # a fresh snapshot with +5 buys emits exactly 5 real-delta trades *first* (surrogates of that step follow)
    fresh = sim_snapshot(2, buys_m5=305, sells_m5=30, vol_m5=33_500.0, source="dexscreener")
    out = sur.step(fresh, snap, dt, now + dt)
    real = [t for t in out if not t.surrogate]
    assert len(real) == 5 and all(t.kind == "buy" for t in real)
    assert out[:5] == real, "the real-delta trades must come before the surrogates of the same step"
    # the next step with the same snapshot is surrogate again
    out2 = []
    for k in range(200):
        out2.extend(sur.step(fresh, fresh, dt, now + (k + 2) * dt))
    assert out2 and all(t.surrogate for t in out2)
    # counters that dropped (window rolled) produce no negative deltas
    fewer = sim_snapshot(3, buys_m5=100, sells_m5=10, source="dexscreener")
    assert [t for t in sur.step(fewer, fresh, dt, now + 300) if not t.surrogate] == []


def test_surrogate_fresh_step_with_zero_delta_still_emits(sim_snapshot):
    """A fresh snapshot whose counter delta is 0 still owes the tick its ``lam*dt`` surrogates (c.16 "first")."""
    sur = SurrogateTrades(np.random.default_rng(5))
    dt = 0.05
    prev = sim_snapshot(1, buys_m5=600, sells_m5=600, vol_m5=60_000.0, source="dexscreener")
    sur.step(prev, None, dt, T0)  # prime
    n = 0
    now = T0
    for k in range(400):  # 400 fresh snapshots with identical counters: delta 0 every time
        now += dt
        fresh = sim_snapshot(2 + k, buys_m5=600, sells_m5=600, vol_m5=60_000.0, source="dexscreener")
        out = sur.step(fresh, prev, dt, now)
        assert all(t.surrogate for t in out), "zero delta must not be reported as real trades"
        n += len(out)
        prev = fresh
    # lam = (600 + 600)/300 = 4 trades/s -> ~80 trades over 20 s of ticks, never 0
    assert 40 <= n <= 130, f"{n} surrogate trades over 20 s of fresh snapshots (expected ~80)"


def test_dex_source_poll_without_network(monkeypatch, dex_pair_json):
    calls = {"n": 0}

    def fake_fetch(chain, token, timeout_s=10.0, **kw):
        calls["n"] += 1
        p = copy.deepcopy(dex_pair_json)
        p["txns"]["m5"]["buys"] = 648 + calls["n"]
        return [p]

    monkeypatch.setattr(dexmod, "fetch_pairs", fake_fetch)
    src = DexScreenerSource("solana", "So111", poll_s=60, seed=1)
    assert src.poll(T0, 0.05) == (None, [])  # nothing fetched yet
    s1 = src.fetch_once(T0)
    assert s1 is not None and s1.seq == 1 and src.failures == 0 and src.ok
    new, tr = src.poll(T0 + 0.05, 0.05)
    assert new is s1 and all(t.surrogate for t in tr)
    new2, _ = src.poll(T0 + 0.1, 0.05)
    assert new2 is None  # same seq -> None
    s2 = src.fetch_once(T0 + 60)
    new3, tr3 = src.poll(T0 + 60.05, 0.05)
    assert new3 is s2 and new3.seq == 2 and new3.buys_m5 == 650
    assert len(tr3) == 1 and tr3[0].kind == "buy" and not tr3[0].surrogate  # +1 buy real-delta trade
    assert src.set_regime("PUMP", 60) is False
    st = src.status()
    assert st["mode"] == "dexscreener" and st["ok"] and st["failures"] == 0


# ------------------------------------------------------------------------------------------- feed


def test_feed_fallback_after_3_failures(monkeypatch, dex_pair_json):
    def boom(chain, token, timeout_s=10.0, **kw):
        raise httpx.ConnectTimeout("ConnectTimeout")

    monkeypatch.setattr(dexmod, "fetch_pairs", boom)
    feed = MarketFeed(make_settings(market="dexscreener", token_address="So111"), seed=1337)
    assert feed.dex is not None
    events_all: list[dict] = []
    now = T0
    snap, trades, events = feed.poll(now, 0.05)
    events_all += events
    assert snap is not None and snap.source == "sim(fallback)"  # nothing fetched yet -> sim serves, no event
    for k in range(FALLBACK_FAILURES):
        assert feed.dex.fetch_once(now + k) is None
    assert feed.dex.failures == FALLBACK_FAILURES
    for k in range(40):
        now += 0.05
        snap, trades, events = feed.poll(now, 0.05)
        events_all += events
        assert snap is not None
        assert snap.source == "sim(fallback)"
    assert feed.mode == "sim(fallback)"
    src_events = [e for e in events_all if e["kind"] == "market_source"]
    assert len(src_events) == 1
    # SPEC d.8: every event is {kind, data}; market_source data is {mode, reason} (c.16 keeps source/reason aliases)
    ev = src_events[0]
    assert set(ev["data"]) == {"mode", "reason"}
    assert ev["data"]["mode"] == "sim(fallback)" == ev["source"]
    assert ev["data"]["reason"] == ev["reason"]
    assert "3 consecutive DexScreener failures" in ev["data"]["reason"] and "ConnectTimeout" in ev["data"]["reason"]
    st = feed.status()
    assert st["mode"] == "sim(fallback)" and st["ok"] is False and st["failures"] == 3
    # seq keeps increasing across the served snapshots
    seqs = []
    for k in range(40):
        now += 0.05
        s, _, _ = feed.poll(now, 0.05)
        seqs.append(s.seq)
    assert all(b >= a for a, b in zip(seqs, seqs[1:]))

    # a later success resumes live data with a second market_source event
    def ok(chain, token, timeout_s=10.0, **kw):
        return [copy.deepcopy(dex_pair_json)]

    monkeypatch.setattr(dexmod, "fetch_pairs", ok)
    assert feed.dex.fetch_once(now) is not None
    now += 0.05
    snap, _, events = feed.poll(now, 0.05)
    assert snap.source == "dexscreener" and snap.symbol == "SOL" and snap.regime is None
    assert [e["data"]["mode"] for e in events if e["kind"] == "market_source"] == ["dexscreener"]
    assert feed.mode == "dexscreener"


def _dex_feed(monkeypatch, pair: dict, seed: int = 1337) -> MarketFeed:
    """A ``MarketFeed`` in dexscreener mode whose HTTP layer returns ``pair`` (no network, no thread)."""
    monkeypatch.setattr(dexmod, "fetch_pairs", lambda chain, token, timeout_s=10.0, **kw: [copy.deepcopy(pair)])
    return MarketFeed(make_settings(market="dexscreener", token_address="So111"), seed=seed)


def test_feed_fallback_never_relabels_the_live_body(monkeypatch, dex_pair_json):
    """A served-source change must publish a body *produced by the new source*, never the old one relabelled.

    Regression: relabelling the DexScreener body as ``sim(fallback)`` put the live chain / symbol / price /
    liquidity on the simulated series with ``regime`` null (d.2 requires it non-null while the sim is active)
    and made the f.1 per-source reset capture ``liq_start`` from the DexScreener liquidity, which pins
    ``rug_snap`` at 1.0 for the rest of the session.
    """
    dex_pair_json["baseToken"]["symbol"] = "SOL"
    dex_pair_json["liquidity"] = {"usd": 1_000_000.0}
    feed = _dex_feed(monkeypatch, dex_pair_json)
    now = T0
    assert feed.dex.fetch_once(now) is not None
    live, _, _ = feed.poll(now, 0.05)
    assert live.source == "dexscreener" and live.symbol == "SOL" and live.liq_usd == 1_000_000.0

    def boom(chain, token, timeout_s=10.0, **kw):
        raise httpx.ConnectTimeout("ConnectTimeout")

    monkeypatch.setattr(dexmod, "fetch_pairs", boom)
    for k in range(FALLBACK_FAILURES):
        assert feed.dex.fetch_once(now + k) is None
    seen_labels: list[str] = []
    for k in range(40):
        now += 0.05
        snap, _, _ = feed.poll(now, 0.05)
        seen_labels.append(snap.source)
        if snap.source == "sim(fallback)":  # every fallback body is a *sim* body
            assert snap.symbol == "FLY" and snap.chain == "sim" and snap.dex == "sim"
            assert snap.regime in REGIME_NAMES, "d.2: regime is non-null while the sim is the active source"
            assert 1e-4 < snap.price_usd < 1e-2, "the live 100.09 price must not reach the sim series"
            assert snap.liq_usd is not None and snap.liq_usd <= 50_000.0 + 1e-6
        else:  # a DexScreener body is never relabelled
            assert snap.source == "dexscreener" and snap.symbol == "SOL"
    assert seen_labels[0] == "sim(fallback)", "the sim must publish its own body on the very tick of the switch"
    assert set(seen_labels) == {"sim(fallback)"}


def test_feed_set_mode_sim_serves_a_sim_body_and_stops_polling(monkeypatch, dex_pair_json):
    """``set_mode('sim')`` from a live DexScreener feed: sim body immediately, and the HTTP thread stops."""
    feed = _dex_feed(monkeypatch, dex_pair_json)
    feed.start()
    try:
        assert feed.dex.fetch_once(T0) is not None
        live, _, _ = feed.poll(T0, 0.05)
        assert live.source == "dexscreener"
        assert feed.set_mode("sim") is True
        snap, _, events = feed.poll(T0 + 0.05, 0.05)
        assert snap.source == "sim" and snap.symbol == "FLY" and snap.regime in REGIME_NAMES
        assert [e["data"]["mode"] for e in events if e["kind"] == "market_source"] == ["sim"]
        assert feed.mode == "sim"
        # the dexscreener thread no longer polls (and no longer counts failures) while the sim is served
        assert not any(t.name == "dexscreener-poll" and t.is_alive() for t in threading.enumerate())
        # switching back serves the live body again on the next poll
        assert feed.set_mode("dexscreener") is True
        snap, _, events = feed.poll(T0 + 0.1, 0.05)
        assert snap.source == "dexscreener" and snap.symbol == "SOL"
        assert [e["data"]["mode"] for e in events if e["kind"] == "market_source"] == ["dexscreener"]
    finally:
        feed.stop()


def test_feed_hello_describes_the_served_source(monkeypatch, dex_pair_json):
    """d.1 ``hello.market``: while the sim tape is served the block is the sim's, not the configured pair's."""
    feed = _dex_feed(monkeypatch, dex_pair_json)
    h = feed.hello()  # no successful fetch yet -> the sim is what poll() serves
    assert h == {"mode": "sim(fallback)", "chain": "sim", "symbol": "FLY", "token": "",
                 "poll_s": max(1, int(round(feed.sim.snapshot_period_s)))}
    assert feed.dex.fetch_once(T0) is not None
    feed.poll(T0, 0.05)
    h = feed.hello()
    assert h == {"mode": "dexscreener", "chain": "solana", "symbol": "SOL", "token": "So111",
                 "poll_s": feed.dex.poll_s}
    assert isinstance(feed.dex.poll_s, int)  # c.16 types poll_s int


def test_feed_set_mode():
    feed = MarketFeed(make_settings(market="sim", sim_regime_s=90.0), seed=1337)
    feed.start()
    assert feed.mode == "sim"
    assert feed.set_mode("PUMP") is True
    assert feed.sim.regime_name == "PUMP"
    assert feed.sim.forced_left_s == pytest.approx(FORCED_REGIME_S)
    snap, trades, events = feed.poll(T0, 0.05)
    assert snap.regime == "PUMP" and snap.source == "sim" and events == []
    assert feed.set_mode("dexscreener") is False  # refused without FLY_TOKEN_ADDRESS
    assert feed.mode == "sim"
    assert feed.set_mode("sim") is True
    assert feed.set_mode("bogus") is False
    assert feed.set_mode("dead") is True and feed.sim.regime_name == "DEAD"
    # the forced regime holds for 60 s of polling then Markov switching resumes
    now = T0
    for k in range(int(FORCED_REGIME_S / 0.05)):
        now += 0.05
        s, _, _ = feed.poll(now, 0.05)
        assert feed.sim.regime_name == "DEAD"
        if now - T0 >= 1.0:  # snapshots are produced once per second; older ones may still say PUMP
            assert s.regime == "DEAD"
    assert feed.sim.forced_left_s == pytest.approx(0.0, abs=1e-6)
    feed.stop()
    assert set(MARKET_MODES) == {"sim", "dexscreener", "CALM", "PUMP", "DUMP", "CHOP", "RUG", "DEAD"}


def test_feed_sim_never_none_and_dwell_scale():
    feed = MarketFeed(make_settings(market="sim", sim_regime_s=45.0), seed=1)
    assert feed.sim.dwell_scale == pytest.approx(0.5)
    feed.start()
    now = T0
    seen = 0
    for _ in range(200):
        now += 0.05
        snap, trades, events = feed.poll(now, 0.05)
        assert isinstance(snap, MarketSnapshot) and snap.source == "sim"
        seen += 1
    assert seen == 200
    st = feed.status()
    assert st["mode"] == "sim" and st["ok"] is True and st["failures"] == 0
    feed.stop()


def test_feed_dexscreener_without_token_degrades_to_sim():
    feed = MarketFeed(make_settings(market="dexscreener", token_address=""), seed=1)
    assert feed.dex is None and feed.mode == "sim"
    snap, _, events = feed.poll(T0, 0.05)
    assert snap.source == "sim"
    src = [e for e in events if e["kind"] == "market_source"]
    assert len(src) == 1
    # every event a loop can publish is {kind, data} (SPEC d.8 / c.26 EventMsg.data)
    assert src[0]["data"] == {"mode": "sim", "reason": src[0]["reason"]}
    assert "FLY_TOKEN_ADDRESS" in src[0]["data"]["reason"]
    assert feed.hello() == {"mode": "sim", "chain": "sim", "symbol": "FLY", "token": "", "poll_s": 1}


def test_snapshot_to_wire_keys(sim_snapshot):
    s = sim_snapshot(7)
    w = s.to_wire()
    expected = ["source", "ts", "seq", "chain", "dex", "pair", "symbol", "price_usd", "price_native", "buys_m5",
                "sells_m5", "buys_h1", "sells_h1", "chg_m5", "chg_h1", "chg_h6", "chg_h24", "vol_m5", "vol_h1",
                "liq_usd", "fdv", "mcap", "regime"]
    assert list(w.keys()) == expected == list(SNAPSHOT_WIRE_KEYS)
    assert w["seq"] == 7 and w["regime"] == "CALM" and w["symbol"] == "FLY"
    s2 = sim_snapshot(8, liq_usd=None)
    assert s2.to_wire()["liq_usd"] is None
    t = Trade(ts=T0, kind="buy", usd=123.4, price=0.001, surrogate=False)
    assert t.to_wire() == {"kind": "buy", "usd": 123.4, "ts": T0, "surrogate": False}


# --------------------------------------------------------------------------------- scripts/demo_market.py


def _load_demo_market():
    """Import ``scripts/demo_market.py`` by path (SPEC h.4 contract; it is not a package module)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts" / "demo_market.py"
    assert path.is_file(), path
    spec = importlib.util.spec_from_file_location("demo_market_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class _FakeServer:
    """Stands in for the running backend: answers ``/api/state`` with a mood chosen per forced regime."""

    MOODS = {"PUMP": "EUPHORIA", "RUG": "PANIC", "CALM": "CRUISING", "DEAD": "SLEEP", "CHOP": "CRUISING"}

    def __init__(self, url: str, moods: dict | None = None) -> None:
        self.url = url
        self.moods = dict(self.MOODS if moods is None else moods)
        self.mode = "sim"
        self.posted: list[str] = []

    def close(self) -> None:
        pass

    def health(self) -> dict:
        return {"ok": True, "run_id": "test", "connectome": {"name": "tiny"}, "market": {"mode": "sim"}, "rtf": 2.0}

    def set_mode(self, mode: str) -> dict:
        self.posted.append(mode)
        self.mode = mode
        return {"ok": True, "mode": mode}

    def state(self) -> dict:
        return {"tick": {"mood": {"state": self.moods.get(self.mode, "CRUISING")},
                         "fly": {"mode": "walk"},
                         "market": {"regime": self.mode, "price_usd": 0.001, "chg_m5": 0.5, "buys_m5": 3,
                                    "sells_m5": 2, "symbol": "FLY"},
                         "drives": {"sugar": 0.1, "looming": 0.0, "sleep_pressure": 0.2}}}


def test_demo_market_exit_codes_and_ascii(monkeypatch, capsys):
    """h.4: ASCII only, ``-h``, and **non-zero on failure** - a stage that never observed its expected mood must
    fail the run by default (regression: the miss branch returned 0 unless an extra --strict flag was passed).
    The advisory DEAD/SLEEP stage (m5 windows are 300 s wide) never fails the run on its own.
    """
    demo = _load_demo_market()
    assert demo.ascii_only("bağlantı kurulamadı") == "ba?lant? kurulamad?"
    assert demo.ascii_only(OSError("WinError 10061: ğ")) == "WinError 10061: ?"
    assert [s[0] for s in demo.STAGES] == ["PUMP", "RUG", "CALM", "DEAD", "CHOP"]
    assert demo.ADVISORY_STAGES == frozenset({"DEAD"})

    # every stage hits its expected mood -> 0
    servers: list[_FakeServer] = []

    def ok_server(url):
        s = _FakeServer(url)
        servers.append(s)
        return s

    monkeypatch.setattr(demo, "Server", ok_server)
    assert demo.main(["--scale", "0.001"]) == 0
    out = capsys.readouterr().out
    assert out.encode("ascii", "strict")  # no non-ASCII byte anywhere
    assert servers[0].posted[-1] == "sim"  # the regime is handed back to the Markov chain at the end

    # nothing but CRUISING: PUMP and RUG (hard stages) miss -> 1; DEAD's SLEEP miss alone is advisory
    def sad_server(url):
        return _FakeServer(url, moods={})

    monkeypatch.setattr(demo, "Server", sad_server)
    assert demo.main(["--scale", "0.001"]) == 1
    out = capsys.readouterr().out
    assert "PUMP" in out and "FAILED" in out
    assert demo.main(["--scale", "0.001", "--no-strict"]) == 0

    # only DEAD/SLEEP missing: advisory by default, hard with --require-sleep
    def sleepless(url):
        return _FakeServer(url, moods={**_FakeServer.MOODS, "DEAD": "CRUISING"})

    monkeypatch.setattr(demo, "Server", sleepless)
    assert demo.main(["--scale", "0.001"]) == 0
    assert "DEAD (advisory)" in capsys.readouterr().out
    assert demo.main(["--scale", "0.001", "--require-sleep"]) == 1
    assert demo.main(["--dry-run"]) == 0
