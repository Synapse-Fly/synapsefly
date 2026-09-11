"""SPEC section h.2 ``test_encoder.py``: ``hill``, ``FeatureExtractor`` (section f.1) and ``SensoryEncoder``
(the 20-row drive table of section f.2). Offline and deterministic (seeded). The ``small_synthetic`` fixture of
SPEC h.1 (``tests/conftest.py``, session-scoped) is used as-is - this file does not rebuild it."""

from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain import encoder as enc_mod
from flybrain.encoder import (
    COMPASS_WEDGES,
    DRIVE_CHANNELS,
    FLASH_MS,
    POKE_CHANNELS,
    POKE_TABLE,
    ChannelDrive,
    FeatureExtractor,
    Features,
    Poke,
    SensoryEncoder,
    hill,
)
from flybrain.market.base import MarketSnapshot, Trade
from flybrain.snn.engine import Drive, LIFEngine
from flybrain.snn.params import current_from_rate

T0 = 1_789_051_563.0
DT = 0.05


def make_settings(**kw):
    """``flybrain.config.Settings`` when constructible with these keywords, else a namespace with the same fields."""
    base = dict(explore_baseline=0.25, mood_feedback=True, easter_eggs=True, drive_mode="poisson", seed=1337,
                tick_hz=20, tick_ms=50.0, market="sim")
    base.update(kw)
    try:
        from flybrain.config import Settings

        return Settings(**{k: v for k, v in base.items() if k in Settings.__dataclass_fields__})
    except Exception:  # pragma: no cover - only if E6's Settings diverges
        return SimpleNamespace(**base)


def snap_of(seq: int, *, buys_m5: int = 41, sells_m5: int = 27, price: float = 0.0012345,
            vol_m5: float = 5120.5, liq: float | None = 50_210.0, chg_m5: float = 1.2,
            ts: float | None = None, source: str = "sim") -> MarketSnapshot:
    """f.1 shorthand builder (one ``price`` / ``liq`` knob for both wire fields); the h.1 ``sim_snapshot``
    fixture of ``conftest.py`` is used where a plain sane snapshot is enough."""
    return MarketSnapshot(
        ts=T0 + seq if ts is None else ts, source=source, chain="sim", dex="sim", pair="SIM", symbol="FLY",
        price_usd=price, price_native=price, buys_m5=buys_m5, sells_m5=sells_m5, buys_h1=buys_m5 * 10,
        sells_h1=sells_m5 * 10, chg_m5=chg_m5, chg_h1=-3.4, chg_h6=5.1, chg_h24=12.7, vol_m5=vol_m5,
        vol_h1=vol_m5 * 12, liq_usd=liq, fdv=price * 1e9, mcap=price * 1e9, regime="CALM", seq=seq,
    )


def buy(usd: float, ts: float, price: float = 0.0012345) -> Trade:
    return Trade(ts=ts, kind="buy", usd=usd, price=price)


def sell(usd: float, ts: float, price: float = 0.0012345) -> Trade:
    return Trade(ts=ts, kind="sell", usd=usd, price=price)


class Runner:
    """Drives a ``FeatureExtractor`` tick by tick with a fixed wall clock."""

    def __init__(self, fx: FeatureExtractor, snap: MarketSnapshot | None = None, dt: float = DT) -> None:
        self.fx = fx
        self.snap = snap
        self.dt = dt
        self.now = T0
        self.f: Features | None = None

    def tick(self, trades=(), *, mood="CRUISING", feeding=False, mn9=0.0, snap=None, pokes=None) -> Features:
        self.now += self.dt
        if snap is not None:
            self.snap = snap
        self.f = self.fx.update(self.snap, list(trades), self.now, self.dt, mood, feeding, mn9, pokes=pokes)
        return self.f

    def run(self, seconds: float, **kw) -> Features:
        n = int(round(seconds / self.dt))
        for _ in range(n):
            self.tick(**kw)
        return self.f  # type: ignore[return-value]


@pytest.fixture
def fx() -> FeatureExtractor:
    return FeatureExtractor(make_settings(), seed=1337)


@pytest.fixture
def encoder(small_synthetic) -> SensoryEncoder:
    return SensoryEncoder(small_synthetic, make_settings(), seed=1337)


def rich_features(**kw) -> Features:
    base = dict(t_ms=1000, sugar=0.5, sweet_gain=1.0, looming=0.7, loom_t_ms=150, loom_side=-1, loom_episode=3,
                activity=0.4, explore=0.25 + 0.25 * 0.4, up=0.3, down=0.0, bitter=0.1, water=0.2, odor=0.5,
                chop=0.2, courtship=0.0, sleep_pressure=0.1, hunger=1.0, feeding=False)
    base.update(kw)
    return Features(**base)


def by(drives, group=None, channel=None):
    out = []
    for d in drives:
        if group is not None and d.group != group:
            continue
        if channel is not None and getattr(d, "channel", None) != channel:
            continue
        out.append(d)
    return out


# --------------------------------------------------------------------------------------------- hill


def test_hill():
    assert hill(0.25, 0.25, 1.5) == pytest.approx(0.5)
    assert hill(0.0, 0.25, 1.5) == 0.0
    assert hill(-1.0, 0.25, 1.5) == 0.0
    xs = np.linspace(0, 2, 200)
    ys = [hill(x, 0.3, 2.0) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))
    assert 0.0 <= hill(100.0, 0.25) <= 1.0
    # SPEC f.2 sugar anchors: 1 -> 133 Hz, 0.5 -> 111, 0.2 -> 62, 0.05 -> 12
    assert 150 * hill(1.0, 0.25, 1.5) == pytest.approx(133.3, abs=0.5)
    assert 150 * hill(0.5, 0.25, 1.5) == pytest.approx(111, abs=1)
    assert 150 * hill(0.2, 0.25, 1.5) == pytest.approx(62, abs=1)
    assert 150 * hill(0.05, 0.25, 1.5) == pytest.approx(12, abs=1)


# --------------------------------------------------------------------------------------------- features


def test_features_ranges(fx):
    rng = np.random.default_rng(3)
    r = Runner(fx)
    price = 0.001
    snap = snap_of(1, price=price)
    unit = ["val_abs", "activity", "rug", "up", "down", "sugar", "bitter", "water", "looming", "flash", "vol_norm",
            "odor", "chop", "courtship", "sleep_pressure", "hunger", "explore"]
    for k in range(600):
        if k % 20 == 0:  # a new snapshot every second
            price *= math.exp(rng.normal(0, 0.03))
            n_b, n_s = int(rng.integers(0, 400)), int(rng.integers(0, 400))
            liq = None if rng.random() < 0.1 else float(50_000 * math.exp(rng.normal(0, 0.2)))
            snap = snap_of(k // 20 + 1, buys_m5=n_b, sells_m5=n_s, price=price, chg_m5=float(rng.normal(0, 8)),
                                vol_m5=float(rng.exponential(5000)), liq=liq)
        trades = []
        for _ in range(int(rng.poisson(0.6))):
            kind = "buy" if rng.random() < 0.5 else "sell"
            trades.append(Trade(ts=r.now + DT * rng.random(), kind=kind, usd=float(rng.lognormal(4.4, 1.2)), price=price))
        mood = "SLEEP" if 300 < k < 320 else "CRUISING"
        f = r.tick(trades, snap=snap, mood=mood, feeding=(k % 7 == 0), mn9=40.0)
        assert -1.0 <= f.bp <= 1.0 and -1.0 <= f.val <= 1.0
        assert -1.0 <= f.mom <= 1.0 and -1.0 <= f.tick <= 1.0
        for name in unit:
            v = abs(f.val) if name == "val_abs" else getattr(f, name)
            assert 0.0 <= v <= 1.0, f"{name}={v} out of [0,1] at tick {k}"
        assert 0.6 <= f.sweet_gain <= 1.0
        assert f.candle in ("up", "down", "flat")
        assert f.loom_side in (-1, 0, 1)
        assert f.loom_t_ms is None or 0 <= f.loom_t_ms <= 300  # closed ramp window: the 300 ms peak is reachable
        assert f.since_last_trade_s >= 0.0
        assert f.usd_ref >= 20.0
        assert f.t_ms == (k + 1) * 50
        assert 0.0 <= f.any_max <= 1.0
        w = f.to_wire()
        assert set(w) == {"sugar", "bitter", "water", "looming", "loom_side", "flash", "odor", "chop", "courtship",
                          "sleep_pressure", "explore", "up", "down", "activity", "hunger", "candle", "any_max"}
        if mood == "SLEEP":
            assert f.explore == 0.0
    assert fx.usd_ref > 20.0


def test_features_buy_raises_sugar_sell_raises_looming():
    fx_b = FeatureExtractor(make_settings(), seed=1)
    r = Runner(fx_b, snap_of(1, buys_m5=30, sells_m5=30, chg_m5=0.0))
    f0 = r.tick()
    assert f0.sugar == 0.0 and f0.looming == 0.0 and f0.loom_side == 0
    f1 = r.tick([buy(100.0, r.now + 0.01)])
    assert f1.sugar > 0.1 and f1.looming == 0.0 and f1.loom_t_ms is None
    assert f1.since_last_trade_s == pytest.approx(0.04, abs=1e-6)
    f2 = r.run(3.0)
    assert 0.0 < f2.sugar < f1.sugar  # tau 1.5 s decay

    fx_s = FeatureExtractor(make_settings(), seed=1)
    r = Runner(fx_s, snap_of(1, buys_m5=30, sells_m5=30, chg_m5=0.0))
    r.tick()
    f1 = r.tick([sell(100.0, r.now + 0.01)])
    assert f1.looming > 0.1 and f1.sugar == 0.0
    assert f1.loom_side in (-1, 1) and f1.loom_episode == 1 and f1.loom_t_ms == 0
    f2 = r.tick()
    assert f2.loom_t_ms == 50 and f2.loom_side == f1.loom_side
    r.run(0.25)
    assert r.f.loom_t_ms == 300  # the ramp window is closed: the 220*looming peak is sampled at 300 ms
    r.tick()
    assert r.f.loom_t_ms is None  # ramp over after 300 ms
    r.run(2.0)
    assert r.f.loom_side == 0  # episode ends 2 s after the last sell
    # a strongly negative snapshot also raises looming without trades (1.2 * relu(-val))
    fx_v = FeatureExtractor(make_settings(), seed=1)
    r = Runner(fx_v, snap_of(1, buys_m5=0, sells_m5=200, chg_m5=-10.0))
    f = r.run(6.0)
    assert f.val < -0.5 and f.looming > 0.5 and f.down > 0.5 and f.bitter > 0.2 and f.up == 0.0
    fx_u = FeatureExtractor(make_settings(), seed=1)
    r = Runner(fx_u, snap_of(1, buys_m5=200, sells_m5=0, chg_m5=10.0))
    f = r.run(6.0)
    assert f.val > 0.5 and f.sugar > 0.5 and f.up > 0.5 and f.looming == 0.0 and f.down == 0.0


def test_loom_episode_and_side():
    # sim: two sells 100 ms apart -> same episode, one non-zero side; 3 s gap -> new episode
    fx = FeatureExtractor(make_settings(), seed=5)
    r = Runner(fx, snap_of(1))
    r.tick()
    f1 = r.tick([sell(50.0, r.now)])
    r.tick()
    f2 = r.tick([sell(50.0, r.now)])
    assert f1.loom_episode == 1 == f2.loom_episode
    assert f1.loom_side != 0 and f2.loom_side == f1.loom_side
    assert f2.loom_t_ms == 0  # each sell restarts the ramp
    r.run(3.0)
    assert r.f.loom_side == 0 and r.f.loom_t_ms is None
    f3 = r.tick([sell(50.0, r.now)])
    assert f3.loom_episode == 2 and f3.loom_side != 0
    # sim side comes from the seeded rng: identical seeds agree, many episodes cover both sides
    sides_a, sides_b = [], []
    for seed, acc in ((9, sides_a), (9, sides_b)):
        fx2 = FeatureExtractor(make_settings(), seed=seed)
        r2 = Runner(fx2, snap_of(1))
        for _ in range(12):
            r2.run(2.5)
            acc.append(r2.tick([sell(50.0, r2.now)]).loom_side)
    assert sides_a == sides_b and set(sides_a) == {-1, 1}
    # dex: side from the trade timestamp parity (+1 odd ms, -1 even ms)
    fx3 = FeatureExtractor(make_settings(), seed=5)
    r3 = Runner(fx3, snap_of(1, source="dexscreener"))
    r3.tick()
    f = r3.tick([sell(50.0, 1_700_000_000.001)])
    assert f.loom_side == +1
    r3.run(3.0)
    f = r3.tick([sell(50.0, 1_700_000_000.002)])
    assert f.loom_side == -1 and f.loom_episode == 2
    # a loom poke starts an episode on the poke side (looks like a sell wall)
    fx4 = FeatureExtractor(make_settings(), seed=5)
    r4 = Runner(fx4, snap_of(1))
    r4.tick()
    f = r4.tick(pokes=[Poke("loom", 1.0, +1, until_ms=10_000)])
    assert f.loom_side == +1 and f.loom_episode == 1 and f.loom_t_ms == 0 and f.looming >= 0.25
    f = r4.tick(pokes=[Poke("loom", 1.0, +1, until_ms=10_000)])  # the same poke again is ignored
    assert f.loom_t_ms == 50 and f.loom_episode == 1


def test_flash_on_whale_sell(fx):
    r = Runner(fx, snap_of(1))
    r.tick()
    for _ in range(20):
        r.tick([buy(50.0, r.now)])
    assert fx.usd_ref == pytest.approx(50.0)
    f = r.tick([sell(600.0, r.now)])  # 600 >= 10 * 50
    assert f.flash == 1.0 and f.whale is not None and f.whale.usd == 600.0
    ev = fx.drain_events()
    assert [e["kind"] for e in ev] == ["whale"] and ev[0]["data"]["kind"] == "sell" and ev[0]["data"]["ratio"] >= 10
    ticks_on = 1
    while True:
        f = r.tick()
        if f.flash == 0.0:
            break
        ticks_on += 1
        assert f.whale is None
    assert ticks_on == FLASH_MS // 50  # 300 ms
    assert r.run(1.0).flash == 0.0
    # a whale BUY is a whale (event) but does not flash
    f = r.tick([buy(700.0, r.now)])
    assert f.whale is not None and f.flash == 0.0
    assert fx.drain_events()[0]["data"]["kind"] == "buy"
    assert fx.drain_events() == []


def test_sustained_loom():
    fx = FeatureExtractor(make_settings(), seed=1)
    r = Runner(fx, snap_of(1, buys_m5=0, sells_m5=100, chg_m5=-10.0))
    seen_false = False
    t_true = None
    for k in range(200):
        f = r.tick()
        if f.looming >= 0.2 and not f.sustained_loom:
            seen_false = True
        if f.sustained_loom and t_true is None:
            t_true = k
    assert seen_false and t_true is not None
    assert fx.sustained_timer >= 3.0
    # calm snapshot -> looming decays -> sustained flag drops
    r.tick(snap=snap_of(2, buys_m5=50, sells_m5=50, chg_m5=0.0))
    f = r.run(10.0)
    assert f.looming < 0.2 and not f.sustained_loom


def test_sleep_pressure_rises_when_quiet(fx):
    r = Runner(fx, snap_of(1, buys_m5=0, sells_m5=0, vol_m5=0.0, chg_m5=0.0))
    f = r.run(90.0)
    assert f.activity < 0.35 and f.sleep_pressure >= 0.95
    assert f.explore == pytest.approx(0.25 + 0.25 * f.activity)
    # a trade burst (busy snapshot) raises activity above 0.35 and decays the pressure at dt/10
    busy = snap_of(2, buys_m5=300, sells_m5=300, vol_m5=60_000.0)
    f = r.run(20.0, snap=busy)
    assert f.activity >= 0.35 and f.sleep_pressure < 0.2


def test_hunger_and_sweet_gain(fx):
    r = Runner(fx, snap_of(1))
    f = r.tick()
    assert f.hunger == pytest.approx(0.5, abs=0.001)
    assert f.sweet_gain == pytest.approx(0.6 + 0.4 * f.hunger)
    f = r.run(60.0)
    assert f.hunger == pytest.approx(0.6, abs=0.002)  # +dt/600
    assert f.sweet_gain == pytest.approx(0.84, abs=0.002)
    f = r.run(10.0, feeding=True, mn9=60.0)  # -0.06 * sat(60/60) * dt
    assert f.hunger == pytest.approx(0.6 - 0.6 + 10 / 600, abs=0.003)
    f = r.run(60.0, feeding=True, mn9=120.0)
    assert f.hunger == 0.0 and f.sweet_gain == pytest.approx(0.6)
    f = r.run(2000.0)
    assert f.hunger == 1.0 and f.sweet_gain == pytest.approx(1.0)


def test_easter_egg(monkeypatch):
    fx = FeatureExtractor(make_settings(easter_eggs=True), seed=1)
    normal = snap_of(1, buys_m5=41)
    assert fx.easter_egg(normal, T0) is None
    assert fx.easter_egg(snap_of(2, buys_m5=420), T0) == "buys_m5 == 420"
    assert fx.easter_egg(snap_of(3, buys_m5=69), T0) == "buys_m5 == 69"
    assert fx.easter_egg(snap_of(4, price=0.000420123), T0) == "price 0.000420"
    assert fx.easter_egg(snap_of(5, price=69.5), T0) == "price 69"
    assert fx.easter_egg(snap_of(6, price=0.0691), T0) == "price 0.069"
    assert fx.easter_egg(snap_of(7, price=0.000421), T0) is None
    # clock 04:20 via monkeypatched time.localtime, once per minute
    fixed = time.struct_time((2026, 9, 10, 4, 20, 7, 3, 253, 0))
    monkeypatch.setattr(time, "localtime", lambda secs=None: fixed)
    assert fx.easter_egg(normal, T0) == "clock 04:20"
    assert fx.easter_egg(normal, T0 + 1.0) is None  # same minute -> once
    monkeypatch.setattr(time, "localtime", lambda secs=None: time.struct_time((2026, 9, 10, 16, 20, 0, 3, 253, 0)))
    assert fx.easter_egg(normal, T0 + 60.0) == "clock 16:20"
    monkeypatch.setattr(time, "localtime", lambda secs=None: time.struct_time((2026, 9, 10, 16, 21, 0, 3, 253, 0)))
    assert fx.easter_egg(normal, T0 + 120.0) is None
    # inside update: courtship pinned to 1.0 and one easter_egg event
    r = Runner(fx, snap_of(8, buys_m5=420))
    f = r.tick()
    assert f.courtship == 1.0
    ev = fx.drain_events()
    assert ev == [{"kind": "easter_egg", "data": {"reason": "buys_m5 == 420"}}]
    r.tick()
    assert fx.drain_events() == []  # same reason within 60 s -> no repeat
    # disabled by FLY_EASTER_EGGS=0
    off = FeatureExtractor(make_settings(easter_eggs=False), seed=1)
    assert off.easter_egg(snap_of(9, buys_m5=420), T0) is None
    assert off.easter_egg(normal, T0) is None
    r2 = Runner(off, snap_of(10, buys_m5=420))
    assert r2.tick().courtship == 0.0


def test_courtship_triggers(fx):
    r = Runner(fx, snap_of(1, price=0.001))
    r.tick()
    f = r.tick(snap=snap_of(2, price=0.001021))  # new session high by >= 2 %
    assert f.courtship == 1.0
    f = r.run(8.0)
    assert 0.3 < f.courtship < 0.4  # exp(-8/8)
    r.run(60.0)
    for k in range(3):  # three consecutive bull snapshots
        f = r.tick(snap=snap_of(10 + k, price=0.001021, buys_m5=90, sells_m5=10, chg_m5=6.0))
    assert f.courtship == 1.0
    r.run(60.0)
    assert r.f.courtship < 0.01
    f = r.tick(pokes=[Poke("pheromone", 1.0, 0, until_ms=fx.t_ms + 2000)])
    assert f.courtship == 1.0
    f = r.run(1.0)
    assert f.courtship == 1.0  # pinned while the pheromone poke is active
    f = r.run(3.0)
    assert f.courtship < 1.0


# --------------------------------------------------------------------------------------------- encoder


def test_encode_drive_table(encoder, small_synthetic):
    f = rich_features()
    drives = encoder.encode(f, heading=0.3, mood="CRUISING", pokes=[], t_ms=1000)
    assert drives and all(isinstance(d, Drive) and isinstance(d, ChannelDrive) for d in drives)
    assert all(d.rate_hz >= 1.0 and 0.0 <= d.recruit <= 1.0 and d.side in (-1, 0, 1) for d in drives)
    assert all(d.channel in DRIVE_CHANNELS for d in drives)
    assert all(d.mode == "poisson" and d.current_mv is None for d in drives)
    sugar = by(drives, "grn_sugar", "sugar")
    assert len(sugar) == 1 and sugar[0].rate_hz == pytest.approx(111, abs=1) and sugar[0].recruit == pytest.approx(0.7)
    assert sugar[0].tag == "grn_sugar:sugar" and SensoryEncoder.tag_of(sugar[0]) == "grn_sugar:sugar"
    ramp = by(drives, "lc_loom", "loom_ramp")
    assert len(ramp) == 1
    assert ramp[0].rate_hz == pytest.approx(220 * 0.7 * 0.25) and ramp[0].recruit == pytest.approx(0.6)
    assert ramp[0].side == f.loom_side == -1 and ramp[0].episode == 3
    tonic = by(drives, "lc_loom", "loom_tonic")
    assert len(tonic) == 1 and tonic[0].rate_hz == pytest.approx(12 * ((0.7 - 0.3) / 0.7) ** 2)
    assert tonic[0].recruit == 1.0 and tonic[0].side == -1 and tonic[0].episode == 0
    aux = by(drives, "lc_loom2", "loom_aux")
    assert len(aux) == 1 and aux[0].rate_hz == pytest.approx(0.5 * (38.5 + tonic[0].rate_hz))
    assert aux[0].recruit == pytest.approx(0.6) and aux[0].episode == 3
    assert by(drives, "lc_loom", "flash") == [] and by(drives, "lc_freeze") == []
    dng = by(drives, "dng100", "explore")
    assert len(dng) == 1 and dng[0].rate_hz >= 7.5
    assert dng[0].rate_hz == pytest.approx(30 * 0.35 + 20 * 0.5)
    epg = by(drives, "epg", "compass")
    assert len(epg) == 1 and epg[0].rate_hz == 40.0
    w = epg[0].weights
    assert w is not None and w.shape[0] == small_synthetic.groups["epg"].shape[0]
    ww = encoder.wedge_weights(0.3)
    assert ww.shape == (COMPASS_WEDGES,) and int(np.count_nonzero(ww)) == 3
    k = int(math.floor(0.3 / (2 * math.pi) * 16))
    assert ww[k] == 1.0 and ww[k - 1] == 0.375 and ww[k + 1] == 0.375
    assert set(np.unique(w)).issubset({0.0, 0.375, 1.0})
    # heading wraps: -0.1 rad and 2 pi - 0.1 give the same wedge; wedge 15 neighbours 0
    assert np.array_equal(encoder.wedge_weights(-0.1), encoder.wedge_weights(2 * math.pi - 0.1))
    assert encoder.wedge_weights(2 * math.pi - 0.01)[0] == 0.375
    light = by(drives, "photoreceptor", "light")
    assert len(light) == 1 and light[0].rate_hz == pytest.approx(15 + 45 * 0.4)
    lw = light[0].weights
    assert lw.shape[0] == small_synthetic.groups["photoreceptor"].shape[0] and lw.min() >= 0.5 and lw.max() <= 1.5
    odor = by(drives, "orn", "odor")
    assert len(odor) == 1 and odor[0].rate_hz == pytest.approx(80 * hill(0.5, 0.3, 1.5))
    assert set(np.unique(odor[0].weights)) == {0.25, 1.0} and encoder.n_food_orn > 0
    assert by(drives, "pam", "reward")[0].rate_hz == pytest.approx(40 * 0.3)
    assert by(drives, "ppl1", "punish") == []  # down = 0 -> skipped
    assert by(drives, "jo_groom", "dust")[0].rate_hz == pytest.approx(60 * hill(0.2, 0.3, 1.5))
    assert by(drives, "grn_bitter", "bitter")[0].recruit == pytest.approx(0.4 + 0.6 * 0.1)
    assert by(drives, "grn_water", "water")[0].rate_hz == pytest.approx(12.0)
    assert by(drives, "dfb_sleep", "sleep")[0].rate_hz == pytest.approx(1.0)  # 10 * 0.1 = 1 Hz: on the floor, kept
    assert by(drives, "grn_pher") == [] and by(drives, "jo_aud") == []  # courtship 0
    assert by(drives, channel="mood") == [] and by(drives, channel="poke") == []
    # flash / freeze rows
    f2 = rich_features(flash=1.0, sustained_loom=True, loom_t_ms=None)
    d2 = encoder.encode(f2, 0.0, "CRUISING", [], 1050)
    fl = by(d2, "lc_loom", "flash")
    assert len(fl) == 1 and fl[0].rate_hz == 220.0 and fl[0].side == 0 and fl[0].recruit == 1.0 and fl[0].episode == 3
    fr = by(d2, "lc_freeze", "freeze")
    assert len(fr) == 1 and fr[0].rate_hz == pytest.approx(80 * hill(0.7, 0.3, 2.0)) and fr[0].side == -1
    assert by(d2, "lc_loom", "loom_ramp") == []
    assert by(d2, "lc_loom2", "loom_aux")[0].rate_hz == pytest.approx(0.5 * tonic[0].rate_hz)
    # feeding removes the sugar term of the explore row
    d3 = encoder.encode(rich_features(feeding=True), 0.0, "CRUISING", [], 1100)
    assert by(d3, "dng100")[0].rate_hz == pytest.approx(30 * 0.35)
    assert encoder.last_drives == d3


def test_sleep_row_threshold(encoder):
    d = encoder.encode(Features(sleep_pressure=0.1), 0.0, "CRUISING", [], 0)
    assert by(d, "dfb_sleep")[0].rate_hz == pytest.approx(1.0)  # exactly the 1 Hz floor -> kept
    d = encoder.encode(Features(sleep_pressure=0.05), 0.0, "CRUISING", [], 0)
    assert by(d, "dfb_sleep") == []  # 0.5 Hz -> skipped


def test_explore_baseline_zero_disables(small_synthetic):
    enc0 = SensoryEncoder(small_synthetic, make_settings(explore_baseline=0.0), seed=1)
    f = rich_features(explore=0.0 + 0.25 * 0.4)
    d = enc0.encode(f, 0.0, "CRUISING", [], 1000)
    assert by(d, "dng100") == []  # the whole explore row is off (no sugar term either)
    enc1 = SensoryEncoder(small_synthetic, make_settings(explore_baseline=0.25), seed=1)
    d = enc1.encode(rich_features(sugar=0.0, activity=0.0, explore=0.25), 0.0, "CRUISING", [], 1000)
    assert by(d, "dng100")[0].rate_hz == pytest.approx(7.5)
    # an explore poke still works with the baseline off
    d = enc0.encode(f, 0.0, "CRUISING", [Poke("explore", 1.0, 0, 5000)], 1000)
    assert by(d, "dng100", "poke")[0].rate_hz == 40.0
    fx0 = FeatureExtractor(make_settings(explore_baseline=0.0), seed=1)
    assert Runner(fx0, snap_of(1, buys_m5=0, sells_m5=0)).tick().explore == 0.0


def test_mood_feedback_flag(small_synthetic):
    on = SensoryEncoder(small_synthetic, make_settings(mood_feedback=True), seed=1)
    off = SensoryEncoder(small_synthetic, make_settings(mood_feedback=False), seed=1)
    f = rich_features()
    d = on.encode(f, 0.0, "EUPHORIA", [], 1000)
    mood = by(d, channel="mood")
    assert {(m.group, m.rate_hz) for m in mood} == {("pam", 60.0), ("flight_dn", 40.0 * 0.7)}
    d = on.encode(f, 0.0, "EUPHORIA", [], 1050, scores={"euphoria": 0.5})
    assert by(d, "flight_dn", "mood")[0].rate_hz == pytest.approx(20.0)
    d = on.encode(f, 0.0, "PANIC", [], 1100)
    assert {(m.group, m.rate_hz) for m in by(d, channel="mood")} == {("ppl1", 60.0), ("dn_freeze", 20.0)}
    d = on.encode(f, 0.0, "ANXIOUS", [], 1150)
    assert {(m.group, m.rate_hz) for m in by(d, channel="mood")} == {("ppl1", 20.0)}
    # both the punish row and the mood row target ppl1 under different tags
    tags = [x.tag for x in by(d, "ppl1")]
    assert "ppl1:mood" in tags and len(set(tags)) == len(tags)
    d = on.encode(f, 0.0, "CRUISING", [], 1200)
    assert by(d, channel="mood") == []
    for m in ("EUPHORIA", "PANIC", "ANXIOUS"):
        d = off.encode(f, 0.0, m, [], 1300)
        assert by(d, channel="mood") == [] and by(d, channel="court") == []
    assert all(x.channel != "mood" for x in off.encode(f, 0.0, "EUPHORIA", [], 1400))
    # courtship upward crossing of 0.5 -> p1 80 Hz for 6 s (tag p1:court)
    t = 2000
    assert by(on.encode(rich_features(courtship=0.2), 0.0, "CRUISING", [], t), channel="court") == []
    d = on.encode(rich_features(courtship=0.9), 0.0, "CRUISING", [], t + 50)
    c = by(d, "p1", "court")
    assert len(c) == 1 and c[0].rate_hz == 80.0 and c[0].tag == "p1:court"
    assert by(on.encode(rich_features(courtship=0.9), 0.0, "CRUISING", [], t + 50 + 5990), channel="court")
    assert by(on.encode(rich_features(courtship=0.9), 0.0, "CRUISING", [], t + 50 + 6000), channel="court") == []
    # a pheromone poke also triggers it
    d = on.encode(rich_features(courtship=0.0), 0.0, "CRUISING", [Poke("pheromone", 1.0, 0, 20_000)], 9000)
    assert by(d, "p1", "court") and by(d, "grn_pher", "poke")


def test_sleep_scales_rates(encoder):
    f = rich_features(sleep_pressure=0.8)
    awake = encoder.encode(f, 0.0, "CRUISING", [Poke("sugar", 1.0, 0, 9000)], 1000)
    asleep = encoder.encode(f, 0.0, "SLEEP", [], 1050)
    a = {d.tag: d.rate_hz for d in awake}
    s = {d.tag: d.rate_hz for d in asleep}
    assert s and set(s) <= set(a)
    for tag, rate in s.items():
        assert rate == pytest.approx(0.3 * a[tag])
    assert s["grn_sugar:poke"] == pytest.approx(0.3 * 150.0)  # pokes are scaled too ("every other row")
    assert s["dfb_sleep:sleep"] == pytest.approx(0.3 * 8.0)
    # grogginess: for 2 s after leaving SLEEP every rate is halved, then back to normal
    groggy = encoder.encode(f, 0.0, "CRUISING", [], 1100)
    g = {d.tag: d.rate_hz for d in groggy}
    for tag, rate in g.items():
        assert rate == pytest.approx(0.5 * a[tag])
    still = {d.tag: d.rate_hz for d in encoder.encode(f, 0.0, "CRUISING", [], 1100 + 1999)}
    assert still["grn_sugar:sugar"] == pytest.approx(0.5 * a["grn_sugar:sugar"])
    back = {d.tag: d.rate_hz for d in encoder.encode(f, 0.0, "CRUISING", [], 1100 + 2000)}
    assert back["grn_sugar:sugar"] == pytest.approx(a["grn_sugar:sugar"])
    # mood feedback rows are not scaled by sleep (a fresh encoder so the poke is not active)
    d = encoder.encode(f, 0.0, "SLEEP", [], 20_000)
    assert by(d, channel="mood") == []


def test_pokes_become_drives(encoder):
    assert list(POKE_TABLE) == ["sugar", "bitter", "loom", "water", "dust", "pheromone", "sleep", "reward", "punish",
                                "explore"]
    assert SensoryEncoder.POKE_TABLE is POKE_TABLE
    # one channel per side (``POKE_CHANNELS``) so an L and an R poke of the same stim keep distinct engine tags
    assert POKE_CHANNELS == {0: "poke", -1: "poke_L", 1: "poke_R"}
    f = Features()  # everything quiet: only light / compass rows are on
    t = 5000
    for stim, (group, rate) in POKE_TABLE.items():
        for side in (-1, 0, 1):
            ch = POKE_CHANNELS[side]
            e = SensoryEncoder(encoder.conn, make_settings(), seed=1)
            d = e.encode(f, 0.0, "CRUISING", [Poke(stim, 0.5, side, until_ms=t + 500)], t)
            pk = by(d, group, ch)
            assert len(pk) == 1, f"{stim}/{side}"
            assert pk[0].rate_hz == pytest.approx(rate * 0.5) and pk[0].side == side and pk[0].recruit == 1.0
            assert pk[0].tag == f"{group}:{ch}"
            # the poke stays active on later ticks without being resent, and expires at until_ms
            assert by(e.encode(f, 0.0, "CRUISING", [], t + 450), group, ch)
            assert by(e.encode(f, 0.0, "CRUISING", [], t + 500), group, ch) == []
            assert e.active_pokes == []
    # an expired poke is ignored immediately
    d = encoder.encode(f, 0.0, "CRUISING", [Poke("sugar", 1.0, 0, until_ms=t)], t)
    assert by(d, channel="poke") == []
    # two pokes of the same stim/side: the strongest wins; different sides -> two drives with different tags
    d = encoder.encode(f, 0.0, "CRUISING", [Poke("bitter", 0.2, 0, t + 100), Poke("bitter", 0.9, 0, t + 100),
                                            Poke("bitter", 0.5, -1, t + 100)], t)
    pk = sorted([x for x in d if x.group == "grn_bitter" and x.channel.startswith("poke")], key=lambda x: x.side)
    assert [(x.side, x.rate_hz) for x in pk] == [(-1, pytest.approx(60.0)), (0, pytest.approx(108.0))]
    assert [x.tag for x in pk] == ["grn_bitter:poke_L", "grn_bitter:poke"]
    # unknown stims are ignored, strength is clipped to 1 (fresh encoder: the bitter pokes above are still active)
    e = SensoryEncoder(encoder.conn, make_settings(), seed=1)
    d = e.encode(f, 0.0, "CRUISING", [Poke("laser", 1.0, 0, t + 100), Poke("water", 3.0, 0, t + 100)], t)
    assert by(d, channel="poke")[0].rate_hz == 60.0 and len(by(d, channel="poke")) == 1
    # a loom poke is forwarded to a bound FeatureExtractor and its ramp starts at loom_t_ms == 0
    fx = FeatureExtractor(make_settings(), seed=1)
    e = SensoryEncoder(encoder.conn, make_settings(), seed=1, features=fx)
    assert e.features is fx and e.bound_via == "explicit"
    fx.update(snap_of(1), [], T0, 0.05, "CRUISING", False, 0.0)
    e.encode(f, 0.0, "CRUISING", [Poke("loom", 1.0, -1, until_ms=1000)], 50)
    f2 = fx.update(snap_of(1), [], T0 + 0.1, 0.05, "CRUISING", False, 0.0)
    assert f2.loom_side == -1 and f2.loom_episode == 1 and f2.loom_t_ms == 0 and f2.looming > 0.2


def test_poke_ramp_matches_a_sell_wall():
    """f.2: "a loom poke ... starts a ramp episode in the FeatureExtractor (so pokes look exactly like a sell
    wall)". The c.27 loop order is update (step 3) then encode (step 4), so a poke forwarded by the encoder must
    be applied at the start of the *next* update - not stamped with the clock of the update that already ran,
    which skipped the ``loom_t_ms == 0`` sample and made the poke's ramp 50 ms shorter than a sell's.
    """
    def ramp_samples(fx: FeatureExtractor, deliver) -> list[int]:
        r = Runner(fx, snap_of(1, buys_m5=30, sells_m5=30, chg_m5=0.0))
        r.tick()  # the encoder's step-4 poke of this tick is delivered by ``deliver`` below
        out: list[int] = []
        for k in range(9):
            f = r.tick(deliver(r, k))
            if f.loom_t_ms is not None:
                out.append(f.loom_t_ms)
        return out

    def market_sell(r: Runner, k: int):
        """A sell wall seen by update() itself (c.27 step 3)."""
        return [sell(50.0, r.now + DT)] if k == 0 else ()

    poke = Poke("loom", 1.0, +1, until_ms=10_000)

    def encoder_poke(r: Runner, k: int):
        """Exactly what SensoryEncoder.encode does at step 4 - after update() has already run for that tick."""
        if k == 0:
            r.fx.note_poke(poke)
        return ()

    sell_ramp = ramp_samples(FeatureExtractor(make_settings(), seed=1), market_sell)
    poke_ramp = ramp_samples(FeatureExtractor(make_settings(), seed=1), encoder_poke)
    assert sell_ramp == [0, 50, 100, 150, 200, 250, 300]
    assert poke_ramp == sell_ramp, f"poke ramp {poke_ramp} != sell ramp {sell_ramp}"


def test_sleep_scales_the_court_feedback_row(small_synthetic):
    """f.2 row 19: "SLEEP -> every other row's rate x 0.3" - the p1 court row is the one mood-feedback row that
    can be alive during SLEEP, and at the full 80 Hz it would re-enter COURTSHIP after 500 ms (f.6)."""
    e = SensoryEncoder(small_synthetic, make_settings(), seed=1)
    e.encode(rich_features(courtship=0.2), 0.0, "CRUISING", [], 0)  # arm the 0.5 upward crossing
    awake = by(e.encode(rich_features(courtship=0.9), 0.0, "CRUISING", [], 50), "p1", "court")
    assert len(awake) == 1 and awake[0].rate_hz == pytest.approx(80.0)
    asleep = by(e.encode(rich_features(courtship=0.9), 0.0, "SLEEP", [], 100), "p1", "court")
    assert len(asleep) == 1 and asleep[0].rate_hz == pytest.approx(0.3 * 80.0)
    sensory = {d.tag: d.rate_hz for d in e.encode(rich_features(courtship=0.9), 0.0, "SLEEP", [], 150)}
    assert sensory["grn_pher:pheromone"] == pytest.approx(0.3 * 100.0 * hill(0.9, 0.3, 1.5))


def test_since_last_trade_never_negative(fx):
    """f.1 reads ``since_last_trade_s`` as an age: a stamp from the future (surrogate stamps, a clock step)
    must clamp to 0, not go negative."""
    r = Runner(fx, snap_of(1))
    f = r.tick([buy(100.0, r.now + 10.0)])  # 10 s in the future
    assert f.since_last_trade_s == 0.0
    assert r.tick().since_last_trade_s == 0.0  # still clamped on the next tick
    f = r.run(11.0)
    assert f.since_last_trade_s > 0.0


def test_source_switch_rebases_liq_start(fx, sim_snapshot):
    """A source switch must re-capture ``liq_start`` from the new source's liquidity scale: keeping the
    DexScreener ``liq_start`` (1e6) while the sim serves 50k would satisfy ``liq < 0.7*liq_start`` forever and
    pin ``rug`` at 1.0 (with ``bitter`` at 0.5) for the rest of the session (the downstream half of the
    ``MarketFeed`` relabelling bug). Uses the h.1 ``sim_snapshot`` fixture."""
    dex = sim_snapshot(1, source="dexscreener", chain="solana", dex="raydium", symbol="SOL", regime=None,
                       liq_usd=1_000_000.0, price_usd=100.0, price_native=100.0, chg_m5=0.5)
    r = Runner(fx, dex)
    f = r.run(2.0)
    assert fx.liq_start == pytest.approx(1_000_000.0) and f.rug == 0.0
    sim = sim_snapshot(2, source="sim(fallback)", liq_usd=50_000.0, price_usd=0.001, price_native=0.001, chg_m5=0.5)
    f = r.run(5.0, snap=sim)
    assert fx.liq_start == pytest.approx(50_000.0)
    assert f.rug == 0.0 and f.bitter < 0.2


def test_encoder_does_not_autobind_when_ambiguous(small_synthetic):
    """Auto-binding is a convenience for a spec-literal three-argument construction; it must never guess
    between two extractors that happen to share one ``Settings`` object (two loops in one process)."""
    settings = make_settings()
    one = SensoryEncoder(small_synthetic, settings, seed=1)
    assert one.features is None and one.bound_via is None  # no extractor exists yet
    fx_a = FeatureExtractor(settings, seed=1)
    sole = SensoryEncoder(small_synthetic, settings, seed=1)
    assert sole.features is fx_a and sole.bound_via == "auto"
    fx_b = FeatureExtractor(settings, seed=2)
    ambiguous = SensoryEncoder(small_synthetic, settings, seed=1)
    assert ambiguous.features is None and ambiguous.bound_via is None
    assert FeatureExtractor.bindable_for(settings) is None
    assert set(FeatureExtractor.live_for(settings)) == {fx_a, fx_b}
    ambiguous.bind(fx_b)
    assert ambiguous.features is fx_b and ambiguous.bound_via == "explicit"
    # a poke through the ambiguous encoder reaches only the extractor it was bound to
    ambiguous.encode(Features(), 0.0, "CRUISING", [Poke("loom", 1.0, +1, 9_999)], 50)
    assert fx_b.update(snap_of(1), [], T0, 0.05, "CRUISING", False, 0.0).loom_episode == 1
    assert fx_a.update(snap_of(1), [], T0, 0.05, "CRUISING", False, 0.0).loom_episode == 0


def test_current_mode(small_synthetic):
    e = SensoryEncoder(small_synthetic, make_settings(drive_mode="current"), seed=1)
    d = e.encode(rich_features(), 0.0, "EUPHORIA", [Poke("sugar", 1.0, 0, 5000)], 1000)
    assert len(d) > 10
    for x in d:
        assert x.mode == "current"
        assert x.current_mv == pytest.approx(current_from_rate(x.rate_hz))
        assert x.current_mv >= 7.0  # v_th - v_rest = 7 mV is the rheobase; 1 Hz sits right on it
    with pytest.raises(ValueError):
        SensoryEncoder(small_synthetic, make_settings(drive_mode="voltage"), seed=1)


def test_all_drive_groups_exist(small_synthetic):
    groups = small_synthetic.groups
    for g in SensoryEncoder.DRIVE_GROUPS:
        assert g in groups and groups[g].shape[0] > 0, g
    for stim, (g, _) in POKE_TABLE.items():
        assert g in groups and groups[g].shape[0] > 0, stim
    e = SensoryEncoder(small_synthetic, make_settings(), seed=1)
    emitted: set[str] = set()
    pokes = [Poke(s, 1.0, 0, 99_999) for s in POKE_TABLE]
    for mood in ("CRUISING", "EUPHORIA", "PANIC", "ANXIOUS", "SLEEP", "FEEDING", "COURTSHIP", "ESCAPE"):
        f = rich_features(flash=1.0, sustained_loom=True, courtship=0.9, sleep_pressure=0.5, down=0.5, water=0.5)
        for d in e.encode(f, 1.0, mood, pokes, 1000):
            emitted.add(d.group)
    assert emitted <= set(groups)
    assert emitted == set(SensoryEncoder.DRIVE_GROUPS)


def test_apply_injects_into_engine(tiny_connectome):
    engine = LIFEngine(tiny_connectome, seed=0, dt_ms=1.0, noise_sigma=0.0)
    e = SensoryEncoder(tiny_connectome, make_settings(), seed=1)
    d = e.encode(rich_features(), 0.5, "CRUISING", [Poke("sugar", 1.0, 0, 500)], 0)
    n = e.apply(engine, d, duration_ms=50.0)
    assert n == len(d)
    tags = {inj.tag for inj in engine.injections}
    assert tags == {x.tag for x in d}
    stats = engine.step(50)
    assert stats.total_spikes > 0
    assert stats.spike_counts_by_group.get("grn_sugar", 0) > 0
    assert stats.spike_counts_by_group.get("lc4", 0) + stats.spike_counts_by_group.get("lplc2", 0) > 0
    # injections expire at the end of the window; re-applying replaces by tag (no accumulation)
    e.apply(engine, e.encode(rich_features(), 0.5, "CRUISING", [], 50), 50.0)
    assert len({inj.tag for inj in engine.injections}) == len(engine.injections)
    # side-filtered lc_loom ramp only recruits the left side
    ramp = [inj for inj in engine.injections if inj.tag == "lc_loom:loom_ramp"][0]
    assert ramp.drive.side == -1 and ramp.drive.recruit == pytest.approx(0.6)


def test_encoder_deterministic(small_synthetic):
    a = SensoryEncoder(small_synthetic, make_settings(), seed=7)
    b = SensoryEncoder(small_synthetic, make_settings(), seed=7)
    c = SensoryEncoder(small_synthetic, make_settings(), seed=8)
    assert np.array_equal(a.pr_phase, b.pr_phase) and not np.array_equal(a.pr_phase, c.pr_phase)
    assert np.array_equal(a.epg_wedge, b.epg_wedge)
    da = a.encode(rich_features(), 0.7, "CRUISING", [], 1234)
    db = b.encode(rich_features(), 0.7, "CRUISING", [], 1234)
    assert [(x.tag, x.rate_hz, x.recruit, x.side, x.episode) for x in da] == \
           [(x.tag, x.rate_hz, x.recruit, x.side, x.episode) for x in db]
    for x, y in zip(da, db):
        if x.weights is not None:
            assert np.array_equal(x.weights, y.weights)
    # the extractor uses SeedSequence child [6]: identical seeds -> identical feature streams
    fa, fb = FeatureExtractor(make_settings(), seed=3), FeatureExtractor(make_settings(), seed=3)
    ra, rb = Runner(fa, snap_of(1)), Runner(fb, snap_of(1))
    for k in range(40):
        tr = [sell(30.0, ra.now)] if k % 10 == 0 else []
        xa, xb = ra.tick(tr), rb.tick(tr)
        assert xa.to_wire() == xb.to_wire() and xa.loom_side == xb.loom_side
    assert enc_mod.SEED_CHILD_ENCODER == 6
