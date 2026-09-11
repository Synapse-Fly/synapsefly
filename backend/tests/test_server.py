"""Server tests (SPEC h.2 ``test_server.py``): REST surface, WebSocket protocol, loop behaviour.

Everything runs on ``small_synthetic`` (4,000 neurons) through a monkeypatched ``load_connectome`` with
``FLY_REALTIME=0``, so a whole app start costs tens of milliseconds and the suite stays far below its
30 s budget. The WebSocket tests use ``fastapi.testclient.TestClient``, which drives the real ``/ws``
endpoint (a real ASGI websocket, a real per-client queue, the real frame serialisers).

Two facts shape the WS tests: with ``FLY_REALTIME=0`` the simulation publishes ticks as fast as the CPU
allows, and every client queue is ``maxsize=4`` drop-oldest (SPEC d.10). A test therefore never expects
the *next* frame to be the interesting one - it scans a bounded number of frames for the frame it wants.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import sys
import time
import types
import zlib
from pathlib import Path
from typing import Any, Callable, Iterator


def _main_ipv6_loopback() -> bool:
    """Skip-guard for the dual-stack boot test: is an ``::1`` bind actually possible here?"""
    from flybrain.__main__ import _ipv6_loopback_ok

    return _ipv6_loopback_ok()

import pytest
from fastapi.testclient import TestClient

from flybrain.config import load_settings
from flybrain.encoder import Poke
from flybrain.market.base import Trade
from flybrain.server import app as app_mod
from flybrain.server.loop import MAX_MARKET_TRADES, SPEED_MAX, SPEED_MIN, SimulationLoop
from flybrain.server.protocol import (
    ErrorMsg,
    EventMsg,
    HelloMsg,
    MarketMsg,
    MoodChangeMsg,
    PongMsg,
    TickMsg,
    tick_to_json,
)
from flybrain.server.session_log import read_session

# --------------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------------

#: Hermetic, fast, offline settings for every test (SPEC h.1 ``settings_tmp`` plus the server knobs).
BASE_ENV: dict[str, str] = {
    "FLY_N_NEURONS": "4000",
    "FLY_REALTIME": "0",
    "FLY_SESSION_LOG": "0",
    "FLY_MARKET": "sim",
    "FLY_LLM": "dryrun",
    "FLY_X": "dryrun",
    "FLY_LOG_LEVEL": "INFO",
}


def make_settings(tmp_path: Path, **extra: str):
    """``load_settings`` with the offline defaults plus ``extra`` (no ``.env``, no process env)."""
    env = dict(BASE_ENV)
    env["FLY_DATA_DIR"] = str(tmp_path / "data")
    env["FLY_OUT_DIR"] = str(tmp_path / "out")
    env.update({k: str(v) for k, v in extra.items()})
    return load_settings(env=env, dotenv=None)


@pytest.fixture(autouse=True)
def _no_leaked_app() -> Iterator[None]:
    """Every test starts and ends with no live simulation (the SPEC c.28 double-start guard)."""
    assert app_mod.app_running() is None, "a previous test leaked a running app"
    yield
    if app_mod.app_running() is not None:  # pragma: no cover - only on a failing test
        app_mod.reset_start_guard()


@pytest.fixture
def patched_connectome(monkeypatch: pytest.MonkeyPatch, small_synthetic: Any) -> Any:
    """``load_connectome`` serves the session-scoped 4k synthetic graph (SPEC h.2)."""
    monkeypatch.setattr("flybrain.connectome.loaders.load_connectome", lambda settings: small_synthetic)
    return small_synthetic


@pytest.fixture
def client(tmp_path: Path, patched_connectome: Any) -> Iterator[TestClient]:
    """A started app (lifespan entered, first tick published) plus its context on ``app.state.ctx``."""
    settings = make_settings(tmp_path)
    app = app_mod.create_app(settings, boot_gates=False)
    with TestClient(app) as c:
        c.app.state.ctx.loop.wait_ready(5.0)
        yield c


@pytest.fixture
def headless(tmp_path: Path, patched_connectome: Any) -> Iterator[Callable[..., Any]]:
    """Factory for a context driven tick-by-tick on the calling thread (no server, no thread)."""
    built: list[Any] = []

    def make(**extra: str) -> Any:
        ctx = app_mod.build_context(make_settings(tmp_path, **extra))
        built.append(ctx)
        ctx.loop.startup()
        return ctx

    try:
        yield make
    finally:
        for ctx in built:
            ctx.close()


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------


def scan(ws: Any, want: str, limit: int = 600, timeout_s: float = 10.0,
         match: Callable[[dict], bool] | None = None) -> dict | None:
    """Read up to ``limit`` frames looking for ``type == want`` (drop-oldest may eat some)."""
    deadline = time.monotonic() + timeout_s
    for _ in range(limit):
        if time.monotonic() > deadline:  # pragma: no cover - only on a stalled server
            return None
        msg = json.loads(ws.receive_text())
        if msg.get("type") == want and (match is None or match(msg)):
            return msg
    return None


def tiny_png() -> bytes:
    """A minimal valid 1x1 greyscale PNG (magic + IHDR + IDAT + IEND), built without PIL."""
    import struct

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\x0a" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(b"\x00\x00", 6)) + chunk(b"IEND", b""))


def run_ticks(ctx: Any, n: int) -> list[dict]:
    return [ctx.loop.tick_once() for _ in range(int(n))]


# --------------------------------------------------------------------------------------------------
# REST
# --------------------------------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    """Every key of SPEC c.28, ``ok`` true, and the nested blocks validate against HealthResponse."""
    from flybrain.server.protocol import HealthResponse

    body = client.get("/api/health").json()
    assert set(body) == {"ok", "uptime_s", "run_id", "rtf", "speed", "seq", "t_ms", "backend",
                         "connectome", "market", "agent", "clients", "log_tail"}
    assert body["ok"] is True
    assert body["seq"] >= 1 and body["t_ms"] >= 50
    assert body["backend"] == "numpy"
    assert SPEED_MIN <= body["speed"] <= SPEED_MAX
    assert body["rtf"] > 0.0
    assert set(body["connectome"]) == {"name", "source", "n", "e", "gain", "license"}
    assert body["connectome"]["n"] == 4000 and body["connectome"]["source"] == "synthetic"
    assert set(body["market"]) == {"mode", "ok", "last_poll", "failures", "token_live"}
    assert body["market"]["mode"] == "sim" and body["market"]["ok"] is True
    assert body["market"]["token_live"] is False   # FLY_TOKEN_LIVE defaults off until the token launches
    assert set(body["agent"]) == {"llm", "x", "tweets_today", "last_tweet_wall", "disabled_reason"}
    assert body["agent"]["llm"] == "dryrun" and body["agent"]["x"] == "dryrun"
    assert body["clients"] == 0
    assert isinstance(body["log_tail"], list) and body["log_tail"]
    HealthResponse.model_validate(body)


def test_health_log_tail_is_opt_out(tmp_path: Path, patched_connectome: Any) -> None:
    """``FLY_HEALTH_LOG_TAIL=0`` empties the ring in ``/api/health`` without changing the SPEC c.28 shape.

    The endpoint is unauthenticated and public in production (api.synapsefly.com), and the ring carries uvicorn
    access lines with client addresses plus internal loop diagnostics.
    """
    app = app_mod.create_app(make_settings(tmp_path, FLY_HEALTH_LOG_TAIL="0"))
    with TestClient(app) as c:
        body = c.get("/api/health").json()
        assert "log_tail" in body and body["log_tail"] == []
        assert body["run_id"] and body["uptime_s"] >= 0.0          # the summary fields are always there


def test_state_has_hello_and_tick(client: TestClient) -> None:
    """``GET /api/state`` is hello + latest tick + trail + mood history + events, all model-valid."""
    body = client.get("/api/state").json()
    assert set(body) == {"hello", "tick", "trail", "mood_history", "events"}
    hello = HelloMsg.model_validate(body["hello"])
    assert hello.v == 1 and hello.run_id
    assert len(hello.raster.rows) == 8 * hello.raster.per_region
    assert all(row.slot == i for i, row in enumerate(hello.raster.rows))
    assert len(hello.pops) == len(body["tick"]["rates"]["pops"])
    tick = TickMsg.model_validate(body["tick"])
    assert tick.seq >= 1 and tick.t_ms == tick.seq * 50
    assert 0.0 <= tick.fly.x <= hello.canvas.w and 0.0 <= tick.fly.y <= hello.canvas.h
    assert isinstance(body["trail"], list) and len(body["trail"]) <= 4000
    assert all(len(p) == 4 for p in body["trail"])
    assert isinstance(body["mood_history"], list) and isinstance(body["events"], list)
    assert len(body["events"]) <= 50


def test_config_redacted(client: TestClient) -> None:
    """``GET /api/config`` never leaks a credential and keeps the Settings field names."""
    body = client.get("/api/config").json()
    assert body["connectome_source"] == "synthetic" and body["market"] == "sim"
    assert body["realtime"] is False and body["tick_ms"] == 50.0
    for key in body:
        low = key.lower()
        assert "x_api" not in low and "x_access" not in low and "anthropic" not in low
        assert "secret" not in low and "token_secret" not in low
    assert "X_API_KEY" not in body and "ANTHROPIC_API_KEY" not in body
    assert isinstance(body["data_dir"], str) and isinstance(body["cors_origins"], list)


def test_groups_endpoint(client: TestClient) -> None:
    """``{group: {n, types <= 20, sided}}`` over every resolved group key."""
    body = client.get("/api/connectome/groups").json()
    assert "gf" in body and "gf_L" in body and "grn_sugar" in body
    assert body["gf"]["n"] == 2 and body["gf"]["types"] == ["DNp01"] and body["gf"]["sided"] is True
    assert body["gf_L"]["sided"] is True
    for name, row in body.items():
        assert set(row) == {"n", "types", "sided"}
        assert row["n"] >= 0 and len(row["types"]) <= 20
    assert body["lc4"]["n"] > 0


def test_neurons_endpoint(client: TestClient) -> None:
    """``?group=gf`` yields the two DNp01 giant fibres with ids, regions and sides."""
    rows = client.get("/api/neurons", params={"group": "gf", "limit": 100}).json()
    assert len(rows) == 2
    assert all(set(r) == {"id", "body_id", "type", "region", "side"} for r in rows)
    assert all(r["type"] == "DNp01" for r in rows)
    assert all(r["region"] == "descending_motor" for r in rows)
    assert sorted(r["side"] for r in rows) == ["L", "R"]
    assert all(r["body_id"] >= 1_000_000 for r in rows)
    assert len(client.get("/api/neurons", params={"group": "lamina", "limit": 3}).json()) == 3
    assert client.get("/api/neurons", params={"group": "nope"}).status_code == 404


def test_poke_rest_and_rate_limit(client: TestClient) -> None:
    """Two pokes per second per client IP; the third is 429 and never reaches the bus."""
    assert client.post("/api/poke", json={"stim": "sugar"}).json() == {"ok": True}
    assert client.post("/api/poke", json={"stim": "loom", "side": "L", "duration_ms": 300}).status_code == 200
    third = client.post("/api/poke", json={"stim": "sugar"})
    assert third.status_code == 429 and "rate_limited" in third.json()["detail"]
    assert client.post("/api/poke", json={"stim": "nope"}).status_code == 422
    assert client.post("/api/poke", json={"stim": "sugar", "strength": 2.0}).status_code == 422
    assert client.post("/api/poke", json={"stim": "sugar", "duration_ms": 10}).status_code == 422


def test_market_mode_rest(client: TestClient) -> None:
    """A regime name is accepted and reaches the sim; ``dexscreener`` is refused without a token."""
    ctx = client.app.state.ctx
    assert client.post("/api/market/mode", json={"mode": "PUMP"}).json() == {"ok": True, "mode": "PUMP"}
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and ctx.feed.sim.regime_name != "PUMP":
        time.sleep(0.01)
    assert ctx.feed.sim.regime_name == "PUMP"
    refused = client.post("/api/market/mode", json={"mode": "dexscreener"})
    assert refused.status_code == 403 and "forbidden" in refused.json()["detail"]
    assert client.post("/api/market/mode", json={"mode": "NOPE"}).status_code == 422


def test_snapshot_rest(client: TestClient) -> None:
    """A PNG posted for a pending id reaches the waiting broker; junk is refused."""
    ctx = client.app.state.ctx
    png = tiny_png()
    ctx.bus.request_snapshot("snap-rest")           # registers the waiter (no client connected)
    body = client.post("/api/snapshot", json={"id": "snap-rest",
                                              "png_b64": base64.b64encode(png).decode()}).json()
    assert body == {"ok": True, "bytes": len(png)}
    assert ctx.bus.wait_snapshot("snap-rest", 1.0) == png
    assert client.post("/api/snapshot", json={"id": "x", "png_b64": "!!!not base64!!!"}).status_code == 400
    not_png = base64.b64encode(b"GIF89a-nope").decode()
    assert client.post("/api/snapshot", json={"id": "x", "png_b64": not_png}).status_code == 400


def test_tweet_test_rest(client: TestClient) -> None:
    """``POST /api/tweet/test`` fires the agent once with reason ``manual`` and returns the record."""
    record = client.post("/api/tweet/test").json()
    assert record["reason"] == "manual" and record["dry_run"] is True and record["posted"] is False
    assert record["model"] == "template" and record["text"]
    assert len(record["text"]) <= 280 and "http" not in record["text"]
    assert record["snapshot_source"] in ("browser", "server")
    assert record["mood"] in ("SLEEP", "CRUISING", "FEEDING", "EUPHORIA", "ANXIOUS", "PANIC", "ESCAPE",
                              "COURTSHIP")
    listed = client.get("/api/tweets", params={"limit": 5}).json()
    assert listed and listed[-1]["id"] == record["id"]


# --------------------------------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------------------------------


def test_ws_hello_then_tick(client: TestClient) -> None:
    """First frame is ``hello``, a ``tick`` follows within a second, both validate (SPEC d.1/d.2)."""
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        model = HelloMsg.model_validate(hello)
        assert model.tick_hz == 20 and model.steps_per_tick == 50 and model.realtime is False
        assert model.connectome.note and "synthetic" in model.connectome.note.lower()
        assert model.channels[0] == "sugar" and "dexscreener" in model.market_modes
        tick = scan(ws, "tick", limit=20, timeout_s=5.0)
        assert tick is not None
        validated = TickMsg.model_validate(tick)
        assert validated.sim.steps == 50 and validated.sim.backend == "numpy"
        assert validated.sim.speed == 1.0            # FLY_REALTIME=0 pins the brain clock
        assert len(validated.rates.regions) == 8 and len(validated.rates.pops) == len(model.pops)
        assert list(validated.rates.pops) == model.pops
        assert len(validated.spikes.slots) == len(validated.spikes.dt)
        assert all(0 <= s < len(model.raster.rows) for s in validated.spikes.slots)
        assert client.get("/api/health").json()["clients"] == 1


def test_ws_client_frames(client: TestClient) -> None:
    """``ping``/``poke``/``clear``/``set_market_mode`` and a bad frame, on one live connection."""
    with client.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"

        ws.send_text(json.dumps({"type": "ping", "t": 1789051563.1}))
        pong = scan(ws, "pong")
        assert pong is not None and pong["t"] == 1789051563.1
        PongMsg.model_validate(pong)
        assert pong["server_wall"] > 0.0

        ws.send_text(json.dumps({"type": "poke", "stim": "sugar", "strength": 0.8, "side": "L",
                                 "duration_ms": 400}))
        poke = scan(ws, "event", match=lambda m: m.get("kind") == "poke")
        assert poke is not None
        EventMsg.model_validate(poke)
        assert poke["data"] == {"stim": "sugar", "strength": 0.8, "side": "L", "duration_ms": 400}

        ws.send_text(json.dumps({"type": "clear"}))
        clear = scan(ws, "event", match=lambda m: m.get("kind") == "clear")
        assert clear is not None and clear["data"]["by"] == "client"

        ws.send_text(json.dumps({"type": "set_market_mode", "mode": "PUMP"}))
        market = scan(ws, "market", match=lambda m: m["market"].get("regime") == "PUMP")
        assert market is not None
        model = MarketMsg.model_validate(market)
        assert model.mode == "sim" and model.market.symbol == "FLY"
        assert all(t.kind in ("buy", "sell") for t in model.trades)

        ws.send_text(json.dumps({"type": "poke", "stim": "sugar", "oops": 1}))
        err = scan(ws, "error")
        assert err is not None and err["code"] == "bad_message" and "oops" in err["msg"]
        ErrorMsg.model_validate(err)

        ws.send_text(json.dumps({"type": "nonsense"}))
        err2 = scan(ws, "error")
        assert err2 is not None and err2["code"] == "bad_message" and "type" in err2["msg"]

        ws.send_text("{not json at all")
        err3 = scan(ws, "error")
        assert err3 is not None and err3["code"] == "bad_message"

        # the connection survived every bad frame
        assert scan(ws, "tick", limit=10) is not None


def test_ws_poke_rate_limit_and_forbidden(client: TestClient) -> None:
    """Per-connection poke limit answers ``rate_limited``; ``dexscreener`` answers ``forbidden``."""
    with client.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        for _ in range(3):
            ws.send_text(json.dumps({"type": "poke", "stim": "bitter"}))
        err = scan(ws, "error")
        assert err is not None and err["code"] == "rate_limited"
        ws.send_text(json.dumps({"type": "set_market_mode", "mode": "dexscreener"}))
        err = scan(ws, "error", match=lambda m: m["code"] == "forbidden")
        assert err is not None and "FLY_TOKEN_ADDRESS" in err["msg"]


def test_ws_snapshot_roundtrip(client: TestClient) -> None:
    """``SnapshotBroker.request`` from a worker thread -> ``snapshot_request`` -> ``"browser"``."""
    import threading

    ctx = client.app.state.ctx
    out: dict[str, Any] = {}
    with client.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        worker = threading.Thread(target=lambda: out.update(zip(("png", "source"),
                                                               ctx.snapshots.request(timeout_s=8.0))))
        worker.start()
        req = scan(ws, "snapshot_request")
        assert req is not None and req["id"] and req["deadline_ms"] >= 1000
        png = tiny_png()
        ws.send_text(json.dumps({"type": "snapshot", "id": req["id"],
                                 "png_b64": base64.b64encode(png).decode()}))
        worker.join(10.0)
    assert out.get("source") == "browser"
    assert out.get("png") == png
    assert ctx.snapshots.last_source == "browser"


def test_ws_queue_overflow_drops_oldest(client: TestClient) -> None:
    """A slow client loses the OLDEST frames, never the newest (SPEC d.10, maxsize 4)."""
    ctx = client.app.state.ctx
    ctx.loop.stop(2.0)                      # freeze the simulation so only our frames are in flight
    base = ctx.bus.dropped_frames
    queue = ctx.bus.subscribe(4)
    try:
        for i in range(1, 8):
            ctx.bus.publish_event({"type": "event", "seq": i, "t_ms": i * 50, "wall": 1.0,
                                   "kind": "poke", "data": {"i": i}})
        # the puts land on the event loop thread: wait until all seven were processed (7 - 4 = 3 drops)
        deadline = time.monotonic() + 5.0
        while ctx.bus.dropped_frames - base < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert queue.qsize() == 4
        got = [json.loads(queue.get_nowait())["data"]["i"] for _ in range(4)]
        assert got == [4, 5, 6, 7]           # 1-3 were dropped, the newest survived
        assert ctx.bus.dropped_frames - base == 3
    finally:
        ctx.bus.unsubscribe(queue)


# --------------------------------------------------------------------------------------------------
# loop behaviour
# --------------------------------------------------------------------------------------------------


def test_double_start_guard(tmp_path: Path, patched_connectome: Any) -> None:
    """A second ``create_app`` while one runs returns the same app; a second lifespan raises."""
    app = app_mod.create_app(make_settings(tmp_path), boot_gates=False)
    assert app_mod.app_running() is None     # nothing runs until the lifespan starts
    rival = app_mod.create_app(make_settings(tmp_path / "b"), boot_gates=False)
    with TestClient(app) as c:
        assert app_mod.app_running() is app
        again = app_mod.create_app(make_settings(tmp_path / "c"), boot_gates=False)
        assert again is app                  # the running app, not a second brain
        assert c.get("/api/health").json()["ok"] is True
        with pytest.raises(RuntimeError, match="already running"):
            with TestClient(rival):          # a second lifespan is refused outright
                pass                         # pragma: no cover
    assert app_mod.app_running() is None
    fresh = app_mod.create_app(make_settings(tmp_path / "d"), boot_gates=False)
    assert fresh is not None and fresh is not app   # buildable again once the first one stopped
    with TestClient(fresh) as c2:
        assert c2.get("/api/health").json()["seq"] >= 1


def test_session_log_written(tmp_path: Path, headless: Callable[..., Any]) -> None:
    """``FLY_SESSION_LOG=1`` writes a header line plus one line per tick with every input."""
    ctx = headless(FLY_SESSION_LOG="1")
    assert ctx.session_log is not None
    ctx.bus.push_poke(Poke(stim="sugar", strength=1.0, side=0, until_ms=ctx.loop.t_ms + 500))
    ctx.bus.push_command({"name": "set_market_mode", "mode": "CHOP"})
    ticks = run_ticks(ctx, 6)
    ctx.session_log.close()

    rows = list(read_session(ctx.session_log.path))
    assert rows[0]["kind"] == "header"
    assert rows[0]["run_id"] == ctx.settings.run_id
    assert rows[0]["settings"]["n_neurons"] == 4000
    assert "X_API_KEY" not in rows[0]["settings"]
    body = rows[1:]
    assert len(body) == len(ticks)
    for row, tick in zip(body, ticks):
        assert row["kind"] == "tick" and row["seq"] == tick["seq"] and row["t_ms"] == tick["t_ms"]
        assert row["total_spikes"] == tick["sim"]["spikes"]
        assert row["market"]["source"] == "sim"
    assert body[0]["pokes"] == [{"stim": "sugar", "strength": 1.0, "side": 0,
                                 "until_ms": body[0]["pokes"][0]["until_ms"]}]
    assert body[0]["commands"] == [{"name": "set_market_mode", "mode": "CHOP"}]


def test_poke_reaches_the_engine(headless: Callable[..., Any]) -> None:
    """A queued ``Poke`` becomes an engine injection and lifts the target population's rate."""
    ctx = headless()
    before = run_ticks(ctx, 4)[-1]["rates"]["pops"]["grn_sugar"]
    ctx.bus.push_poke(Poke(stim="sugar", strength=1.0, side=0, until_ms=ctx.loop.t_ms + 2000))
    ticks = run_ticks(ctx, 8)
    tags = [inj.tag for inj in ctx.engine.injections]
    assert any(tag.startswith("grn_sugar") for tag in tags), tags
    after = ticks[-1]["rates"]["pops"]["grn_sugar"]
    assert after > before + 20.0, (before, after)
    assert ticks[-1]["rates"]["pops"]["feed_mn"] > 0.0


def test_set_market_mode_switches_the_sim_regime(headless: Callable[..., Any]) -> None:
    """``set_market_mode`` reaches ``MarketFeed`` on the next tick and shows up in ``tick.market``."""
    ctx = headless()
    run_ticks(ctx, 2)
    ctx.bus.push_command({"name": "set_market_mode", "mode": "RUG"})
    ticks = run_ticks(ctx, 25)               # a sim snapshot lands once per simulated second
    assert ctx.feed.sim.regime_name == "RUG"
    assert ticks[-1]["market"]["regime"] == "RUG"
    assert ctx.feed.set_mode("dexscreener") is False      # no token address configured
    assert ctx.feed.set_mode("NOPE") is False


def test_wall_bump_injects_counter_steering(headless: Callable[..., Any]) -> None:
    """A wall bump pulses the DNa02 on the side away from the wall (SPEC c.27 step 6)."""
    ctx = headless()
    run_ticks(ctx, 1)
    ctx.body.teleport(ctx.body.w - 2.0, ctx.body.h / 2.0)
    ctx.body.heading = 0.0
    ctx.body.v = 200.0
    bumped = None
    for _ in range(20):
        tick = ctx.loop.tick_once()
        hit = [e for e in tick["events"] if e["kind"] == "wall_bump"]
        if hit:
            bumped = hit[0]
            break
    assert bumped is not None, "the fly never reached the wall"
    assert bumped["data"]["side"] in ("L", "R")
    expected = "steer_a02_R" if bumped["data"]["side"] == "L" else "steer_a02_L"
    wall = [inj for inj in ctx.engine.injections if inj.tag == "wall"]
    assert wall and wall[0].drive.group == expected
    assert wall[0].drive.rate_hz == pytest.approx(40.0)


def test_tick_events_and_mood_change_frame(headless: Callable[..., Any]) -> None:
    """In-tick events carry d.8 kinds; a transition publishes ``mood_change`` and an in-tick ``mood``."""
    ctx = headless()
    frames: list[dict] = []
    original = ctx.bus.publish_event

    def spy(msg: dict) -> None:
        frames.append(dict(msg))
        original(msg)

    ctx.bus.publish_event = spy              # type: ignore[method-assign]
    try:
        ticks = run_ticks(ctx, 120)
    finally:
        ctx.bus.publish_event = original     # type: ignore[method-assign]

    kinds = {e["kind"] for t in ticks for e in t["events"]}
    assert kinds, "a 6 s run must produce at least one in-pipeline event"
    assert kinds <= set(
        "jump takeoff landing freeze unfreeze feed_start feed_stop groom song saccade wander_floor "
        "sleep wake wall_bump wrap gf_spike mood".split())
    changes = [f for f in frames if f["type"] == "mood_change"]
    mood_events = [e for t in ticks for e in t["events"] if e["kind"] == "mood"]
    assert len(changes) == len(mood_events)
    for frame in changes:
        model = MoodChangeMsg.model_validate(frame)
        assert model.from_ != model.to and model.reason
        assert frame["mood"]["state"] == model.to
    assert all(set(e["data"]) == {"from", "to", "reason"} for e in mood_events)
    assert all(f["type"] != "tick" for f in frames)          # ticks go through publish_tick only


def test_loop_speed_adapts_to_compute(tmp_path: Path, patched_connectome: Any) -> None:
    """SPEC c.27 step 12: over-budget compute slows the brain clock, a long fast run speeds it up."""
    ctx = app_mod.build_context(make_settings(tmp_path, FLY_REALTIME="1"))
    try:
        loop = ctx.loop
        assert loop.realtime is True and loop.speed == SPEED_MAX
        loop._adapt_speed(loop.tick_ms * 0.9)
        assert loop.speed == pytest.approx(0.85)
        assert loop._steps_for_tick() == round(50 * 0.85)
        for _ in range(12):
            loop._adapt_speed(loop.tick_ms * 0.95)
        assert loop.speed == SPEED_MIN        # floors at 0.25
        assert loop._steps_for_tick() >= 5
        for _ in range(39):
            loop._adapt_speed(loop.tick_ms * 0.1)
        assert loop.speed == SPEED_MIN        # 39 fast ticks are not enough
        loop._adapt_speed(loop.tick_ms * 0.1)
        assert loop.speed == pytest.approx(0.275)
        for _ in range(40 * 20):
            loop._adapt_speed(0.0)
        assert loop.speed == SPEED_MAX        # and ceilings at 1.0
    finally:
        ctx.close()


def test_realtime_loop_keeps_the_tick_grid(tmp_path: Path, patched_connectome: Any) -> None:
    """A real 20 Hz run publishes ~20 ticks/s, never skips a ``seq`` and reports a sane ``rtf``.

    The *wall* grid is the invariant here, not the brain clock: SPEC c.27 step 12 lets a tick whose
    compute exceeds ``0.8 * tick_ms`` drop ``speed`` (and with it ``steps = round(50 * speed)``), which
    is exactly what a busy machine triggers. Asserting ``t_ms == 50 * seq`` therefore asserted that the
    designed back-off never fires; the assertions below bound the brain clock by ``speed`` instead.
    """
    app = app_mod.create_app(make_settings(tmp_path, FLY_REALTIME="1"), boot_gates=False)
    with TestClient(app) as c:
        ctx = c.app.state.ctx
        ctx.loop.wait_ready(5.0)
        time.sleep(1.0)
        tick = ctx.bus.latest_tick()                # read the tick first: the loop keeps ticking
        seq = int(tick["seq"])
        # ~20 wall ticks in a second: the lower bound leaves room for a slow CI box (a drop below 10 Hz
        # means a tick costs > 100 ms, a real pacing regression), the upper one catches a loop that
        # forgot to sleep.
        assert 10 <= seq <= 28, seq
        seqs = [int(t["seq"]) for t in ctx.history.last(10.0)]
        assert seqs[:seq] == list(range(1, seq + 1)), seqs[:seq]
        speed = float(tick["sim"]["speed"])
        assert SPEED_MIN <= speed <= SPEED_MAX
        # brain time is sum(steps * dt) with steps = max(5, round(50 * speed)): never ahead of the wall
        # grid, never below the 5-step floor, whatever the machine's load did to ``speed``.
        assert 5 * seq <= int(tick["t_ms"]) <= 50 * seq
        assert int(tick["sim"]["steps"]) == max(5, round(50 * speed))
        rtf = float(tick["sim"]["rtf"])
        assert math.isfinite(rtf) and rtf > 0.0
        assert ctx.loop.n_errors == 0


def test_deterministic_200_ticks(tmp_path: Path, patched_connectome: Any) -> None:
    """Two 200-tick ``FLY_REALTIME=0`` runs with the same seed are identical (SPEC 0.1)."""
    def run(tag: str) -> list[tuple]:
        ctx = app_mod.build_context(make_settings(tmp_path / tag, FLY_TWEETS_PER_DAY="0"))
        try:
            ctx.loop.startup()
            out = []
            for tick in run_ticks(ctx, 200):
                out.append((tick["seq"], tick["t_ms"], tick["sim"]["spikes"], tick["fly"]["x"],
                            tick["fly"]["y"], tick["fly"]["heading"], tick["mood"]["state"],
                            tick["drives"]["sugar"], round(tick["rates"]["pops"]["dng100"], 6),
                            tuple(tick["spikes"]["slots"][:8])))
            return out
        finally:
            ctx.close()

    first, second = run("a"), run("b")
    assert first == second
    assert sum(row[2] for row in first) > 0
    assert len({row[6] for row in first}) >= 1


@pytest.mark.slow
def test_replay_bit_exact(tmp_path: Path, patched_connectome: Any) -> None:
    """200 logged ticks replayed through ``FLY_REPLAY`` reproduce every spike count and the final pose."""
    live = app_mod.build_context(make_settings(tmp_path / "live", FLY_SESSION_LOG="1",
                                              FLY_TWEETS_PER_DAY="0"))
    try:
        live.loop.startup()
        live.bus.push_poke(Poke(stim="loom", strength=1.0, side=-1, until_ms=live.loop.t_ms + 600))
        ticks = run_ticks(live, 200)
        path = Path(live.session_log.path)
        pose = (live.body.x, live.body.y, live.body.heading)
        live.session_log.close()
    finally:
        live.close()

    replayed = app_mod.build_context(make_settings(tmp_path / "replay", FLY_SESSION_LOG="0",
                                                  FLY_TWEETS_PER_DAY="0", FLY_REPLAY=str(path)))
    try:
        assert replayed.loop.replay is not None
        replayed.loop.startup()
        out: list[dict] = []
        with pytest.raises(StopIteration):
            while True:
                out.append(replayed.loop.tick_once())
        assert len(out) == len(ticks) == replayed.loop.replay_ticks
        assert replayed.loop.replay_diffs == []
        assert [t["sim"]["spikes"] for t in out] == [t["sim"]["spikes"] for t in ticks]
        assert [t["t_ms"] for t in out] == [t["t_ms"] for t in ticks]
        assert (replayed.body.x, replayed.body.y, replayed.body.heading) == pose
        assert [t["fly"]["x"] for t in out] == [t["fly"]["x"] for t in ticks]
        assert [t["mood"]["state"] for t in out] == [t["mood"]["state"] for t in ticks]
    finally:
        replayed.close()


def test_boot_gates_are_logged(tmp_path: Path, patched_connectome: Any) -> None:
    """``boot_gates=True`` runs the SPEC g.6 gates once and logs them (no effect on the sim)."""
    app = app_mod.create_app(make_settings(tmp_path), boot_gates=True)
    with TestClient(app) as c:
        ctx = c.app.state.ctx
        deadline = time.monotonic() + 20.0
        while ctx.gates is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ctx.gates is not None
        assert set(ctx.gates.passed) >= {"rest_rate", "mn9", "gf_latency", "a02_diff", "runaway"}
        assert "gates:" in "\n".join(ctx.ring.tail(200))
        assert c.get("/api/health").json()["log_tail"]
        assert ctx.loop.seq > 0 and ctx.loop.n_errors == 0


def test_loop_survives_a_broken_collaborator(headless: Callable[..., Any]) -> None:
    """SPEC 0.1: an exploding market feed becomes an event plus a log line, never a dead sim."""
    ctx = headless()
    run_ticks(ctx, 2)

    def boom(now: float, dt_s: float):
        raise RuntimeError("dex exploded")

    original = ctx.feed.poll
    ctx.feed.poll = boom                     # type: ignore[assignment]
    try:
        tick = ctx.loop.tick_once()
    finally:
        ctx.feed.poll = original             # type: ignore[assignment]
    assert tick["seq"] == 3 and tick["sim"]["spikes"] >= 0
    assert ctx.loop.n_errors == 1 and "dex exploded" in str(ctx.loop.last_error)
    kinds = [e["kind"] for e in ctx.loop.recent_events()]
    assert "market_source" in kinds
    assert ctx.loop.tick_once()["seq"] == 4


def test_market_event_normalises_feed_shape_to_d8() -> None:
    """``SimulationLoop._market_event`` yields d.8 ``data == {mode, reason}`` for a real feed event.

    ``MarketFeed._event`` already emits ``{kind, data:{mode,reason}, source, reason}`` (its own d.8 shape
    with top-level aliases). The loop must forward ``data`` verbatim, never nest it under a spurious
    second ``data`` key (regression for the reviewer's d.8 finding).
    """
    from flybrain.market.feed import _event

    feed_ev = _event("sim(fallback)", "3 consecutive DexScreener failures: ConnectTimeout")
    out = SimulationLoop._market_event(feed_ev)
    assert out["kind"] == "market_source"
    assert set(out["data"]) == {"mode", "reason"}
    assert out["data"] == {"mode": "sim(fallback)",
                           "reason": "3 consecutive DexScreener failures: ConnectTimeout"}
    # a flat/legacy event without a nested ``data`` still normalises cleanly via the aliases
    flat = SimulationLoop._market_event({"kind": "market_source", "source": "dexscreener",
                                         "reason": "engaged"})
    assert set(flat["data"]) == {"mode", "reason"}
    assert flat["data"] == {"mode": "dexscreener", "reason": "engaged"}


def test_market_source_frame_shape_end_to_end(headless: Callable[..., Any]) -> None:
    """A real dexscreener->sim fallback publishes a ``market_source`` event with exactly ``{mode, reason}``.

    ``FLY_MARKET=dexscreener`` with an empty ``FLY_TOKEN_ADDRESS`` makes ``MarketFeed`` queue a fallback
    ``market_source`` event on its first poll; it flows through ``_market_event`` and out via
    ``bus.publish_event``. This is the ONLY path that exercises ``_market_event`` (SPEC d.8).
    """
    ctx = headless(FLY_MARKET="dexscreener", FLY_TOKEN_ADDRESS="")
    frames: list[dict] = []
    original = ctx.bus.publish_event

    def spy(msg: dict) -> None:
        frames.append(dict(msg))
        original(msg)

    ctx.bus.publish_event = spy              # type: ignore[method-assign]
    try:
        ctx.loop.tick_once()
    finally:
        ctx.bus.publish_event = original     # type: ignore[method-assign]

    sources = [f for f in frames if f.get("kind") == "market_source"]
    assert sources, "the dexscreener fallback must publish a market_source event"
    for frame in sources:
        EventMsg.model_validate(frame)
        assert set(frame["data"]) == {"mode", "reason"}
        assert isinstance(frame["data"]["mode"], str) and frame["data"]["mode"]
        assert isinstance(frame["data"]["reason"], str) and frame["data"]["reason"]
    # and the same clean shape is what GET /api/state.events would return
    stored = [e for e in ctx.loop.recent_events() if e["kind"] == "market_source"]
    assert stored and all(set(e["data"]) == {"mode", "reason"} for e in stored)


# --------------------------------------------------------------------------------------------------
# out-of-band feature events (SPEC d.8) - the FeatureExtractor queue is the only source
# --------------------------------------------------------------------------------------------------


def test_feature_events_are_drained_and_published(headless: Callable[..., Any],
                                                  sim_snapshot: Callable[..., Any]) -> None:
    """``whale`` / ``easter_egg`` reach the clients from ``features.drain_events()`` and nowhere else.

    ``FeatureExtractor.update`` derives both and queues them (it owns the whale threshold, the 60 s
    per-reason cooldown and the once-per-minute clock key). The loop must publish that queue verbatim:
    re-deriving them here published duplicates *and* left the queue growing for the whole process.
    """
    ctx = headless()
    run_ticks(ctx, 2)
    frames: list[dict] = []
    original = ctx.bus.publish_event

    def spy(msg: dict) -> None:
        frames.append(dict(msg))
        original(msg)

    small = [Trade(ts=100.0 + i, kind="buy", usd=10.0, price=0.001) for i in range(12)]
    whale = [Trade(ts=200.0, kind="sell", usd=250_000.0, price=0.001)]
    scripted = [(sim_snapshot(900, buys_m5=420), small, []),
                (sim_snapshot(901, buys_m5=420), whale, [])]
    polls = iter(scripted)
    real_poll = ctx.feed.poll
    ctx.bus.publish_event = spy                  # type: ignore[method-assign]
    ctx.feed.poll = lambda now, dt_s: next(polls)        # type: ignore[assignment]
    try:
        run_ticks(ctx, len(scripted))
    finally:
        ctx.feed.poll = real_poll                # type: ignore[assignment]
        ctx.bus.publish_event = original         # type: ignore[method-assign]

    eggs = [f for f in frames if f.get("kind") == "easter_egg"]
    assert len(eggs) == 1, eggs                  # one per reason per 60 s, never once per tick
    EventMsg.model_validate(eggs[0])
    assert eggs[0]["data"] == {"reason": "buys_m5 == 420"}
    whales = [f for f in frames if f.get("kind") == "whale"]
    assert len(whales) == 1, whales
    EventMsg.model_validate(whales[0])
    assert set(whales[0]["data"]) == {"kind", "usd", "ratio"}
    assert whales[0]["data"]["kind"] == "sell" and whales[0]["data"]["ratio"] >= 10.0
    assert ctx.features.drain_events() == []     # the loop owns the hand-off; nothing is left behind
    kinds = [e["kind"] for e in ctx.loop.recent_events()]
    assert "easter_egg" in kinds and "whale" in kinds


def test_clock_easter_egg_reaches_every_client(headless: Callable[..., Any],
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``clock 04:20`` egg is published once, not eaten by a second ``easter_egg()`` call.

    ``FeatureExtractor.easter_egg`` self-dedupes the clock branch on ``(year, yday, hour, minute)``, so
    any caller after ``update`` gets ``None``: the loop must take the queued event instead of asking
    again (SPEC d.8 ``{"reason": "clock 04:20"}``).
    """
    import flybrain.encoder as enc_mod

    at_0420 = types.SimpleNamespace(tm_year=2026, tm_yday=250, tm_hour=4, tm_min=20)
    monkeypatch.setattr(enc_mod, "time", types.SimpleNamespace(localtime=lambda *a: at_0420))

    ctx = headless()
    frames: list[dict] = []
    original = ctx.bus.publish_event

    def spy(msg: dict) -> None:
        frames.append(dict(msg))
        original(msg)

    ctx.bus.publish_event = spy                  # type: ignore[method-assign]
    try:
        run_ticks(ctx, 4)
    finally:
        ctx.bus.publish_event = original         # type: ignore[method-assign]

    eggs = [f for f in frames if f.get("kind") == "easter_egg"]
    assert len(eggs) == 1, eggs
    assert eggs[0]["data"] == {"reason": "clock 04:20"}
    EventMsg.model_validate(eggs[0])
    assert ctx.features.drain_events() == []


def test_pending_trades_are_bounded_between_market_frames(headless: Callable[..., Any],
                                                          sim_snapshot: Callable[..., Any]) -> None:
    """SPEC d.5 ``trades <= 200, oldest dropped`` holds while the buffer fills, not only when it flushes."""
    ctx = headless()
    run_ticks(ctx, 1)
    frozen = sim_snapshot(500)                   # the same seq every tick -> no ``market`` frame
    real_poll = ctx.feed.poll

    def stalled(now: float, dt_s: float) -> Any:
        batch = [Trade(ts=float(i), kind="buy", usd=12.0, price=0.001) for i in range(40)]
        return frozen, batch, []

    ctx.feed.poll = stalled                      # type: ignore[assignment]
    frames: list[dict] = []
    original = ctx.bus.publish_event

    def spy(msg: dict) -> None:
        frames.append(dict(msg))
        original(msg)

    try:
        run_ticks(ctx, 20)                       # 19 silent ticks x 40 trades = 760 queued
        assert len(ctx.loop._pending_trades) == MAX_MARKET_TRADES
        moved = sim_snapshot(501)
        ctx.feed.poll = lambda now, dt_s: (moved, [], [])       # type: ignore[assignment]
        ctx.bus.publish_event = spy              # type: ignore[method-assign]
        ctx.loop.tick_once()
    finally:
        ctx.bus.publish_event = original         # type: ignore[method-assign]
        ctx.feed.poll = real_poll                # type: ignore[assignment]

    markets = [f for f in frames if f.get("type") == "market"]
    assert markets and len(markets[-1]["trades"]) == MAX_MARKET_TRADES
    MarketMsg.model_validate(markets[-1])
    assert ctx.loop._pending_trades == []


# --------------------------------------------------------------------------------------------------
# loop error reporting
# --------------------------------------------------------------------------------------------------


def test_a_failing_tick_is_logged_with_the_seq_it_attempted(tmp_path: Path, patched_connectome: Any,
                                                            caplog: pytest.LogCaptureFixture) -> None:
    """SPEC 0.1: a failed tick is reported under the seq it was computing, never the last good one."""
    ctx = app_mod.build_context(make_settings(tmp_path))
    try:
        real_step = ctx.engine.step
        calls = {"n": 0}

        def step(n: int) -> Any:
            calls["n"] += 1
            if calls["n"] in (3, 4):
                raise RuntimeError("engine exploded")
            return real_step(n)

        ctx.engine.step = step                   # type: ignore[assignment]
        with caplog.at_level(logging.ERROR, logger="flybrain.server.loop"):
            ctx.loop.start()
            # wait for BOTH failures *and* for the sim to have carried on past them before stopping:
            # ``n_errors`` is bumped before the log line and before the next tick exists.
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and (ctx.loop.n_errors < 2 or ctx.loop.seq < 5):
                time.sleep(0.005)
            ctx.loop.stop(5.0)
        reported = [int(m.group(1)) for m in
                    (re.search(r"tick (\d+) failed", r.getMessage()) for r in caplog.records) if m]
        # both attempts were at tick 3 (tick 2 had succeeded, tick 3 never completed)
        assert reported == [3, 3], reported
        assert ctx.loop.n_errors == 2
        assert ctx.loop.seq >= 5                                     # and the sim carried on
    finally:
        ctx.close()


# --------------------------------------------------------------------------------------------------
# REST / WS conformance
# --------------------------------------------------------------------------------------------------


def test_state_tick_is_rounded_like_the_ws_frame(client: TestClient) -> None:
    """``GET /api/state.tick`` is the SPEC d rounding pass, byte for byte the WS ``tick`` payload."""
    ctx = client.app.state.ctx
    ctx.loop.stop(3.0)                           # freeze so REST and WS describe the same tick
    rest = client.get("/api/state").json()["tick"]
    wire = json.loads(tick_to_json(ctx.loop.latest()))
    assert rest == wire
    assert rest["rates"]["pops"] == {k: round(v, 2) for k, v in rest["rates"]["pops"].items()}
    assert rest["fly"]["x"] == round(rest["fly"]["x"], 1)
    TickMsg.model_validate(rest)


def test_clear_rest_publishes_an_api_event(client: TestClient) -> None:
    """``POST /api/clear`` is the only way to reach SPEC d.8 ``{"by": "api"}`` (mock_ws serves it too)."""
    with client.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        assert client.post("/api/clear").json() == {"ok": True}
        ev = scan(ws, "event", match=lambda m: m.get("kind") == "clear")
        assert ev is not None and ev["data"] == {"by": "api"}
        EventMsg.model_validate(ev)


def test_ws_binary_frame_answers_bad_message(client: TestClient) -> None:
    """A binary frame is a client mistake, not a disconnect (SPEC d preamble / d.7)."""
    with client.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        ws.send_bytes(b"\xff\xfe\x00not json")
        err = scan(ws, "error")
        assert err is not None and err["code"] == "bad_message"
        ErrorMsg.model_validate(err)
        assert scan(ws, "tick", limit=10) is not None      # the connection stayed open
        ws.send_text(json.dumps({"type": "ping", "t": 12.5}))
        pong = scan(ws, "pong")
        assert pong is not None and pong["t"] == 12.5


def test_poke_limiter_forgets_stale_and_closed_clients(client: TestClient) -> None:
    """The rate-limit table never grows without bound and a budget is never inherited (SPEC c.28)."""
    ctx = client.app.state.ctx
    for i in range(50):
        assert ctx.allow_poke(f"ws:{i}", now=100.0) is True
    assert len(ctx._poke_hits) == 50
    assert ctx.allow_poke("ws:later", now=200.0) is True
    assert set(ctx._poke_hits) == {"ws:later"}         # every key outside the window was swept
    assert ctx.allow_poke("ws:later", now=200.0) is True
    assert ctx.allow_poke("ws:later", now=200.0) is False      # 2 per second, per key
    ctx.forget_poke_key("ws:later")
    assert "ws:later" not in ctx._poke_hits
    assert ctx.allow_poke("ws:later", now=200.0) is True       # a fresh client, a fresh budget

    # The lines above drove the sweep with a FAKE clock (now=200.0); the WS handler below uses the real
    # time.monotonic(), which on a freshly-booted CI runner can be < 200, so a fake-future 'ws:later'
    # would never fall out of its window. Clear the table so this block tests only what it names: a
    # CLOSED socket's key is forgotten.
    ctx._poke_hits.clear()

    with client.websocket_connect("/ws") as ws:
        assert json.loads(ws.receive_text())["type"] == "hello"
        for _ in range(3):
            ws.send_text(json.dumps({"type": "poke", "stim": "bitter"}))
        assert scan(ws, "error", match=lambda m: m["code"] == "rate_limited") is not None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(k.startswith("ws:") for k in ctx._poke_hits):
        time.sleep(0.01)
    assert not [k for k in ctx._poke_hits if k.startswith("ws:")], ctx._poke_hits


# --------------------------------------------------------------------------------------------------
# flybrain/__main__.py
# --------------------------------------------------------------------------------------------------


def test_main_reports_a_bad_config_value_as_one_ascii_line(monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture) -> None:
    """A bad ``FLY_*`` value is an operator message plus exit 2, never a traceback (SPEC 0.1 / h.4)."""
    import flybrain.config as config_mod
    from flybrain.__main__ import main

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("FLY_MARKET: expected one of sim|dexscreener (got 'bogus')")

    monkeypatch.setattr(config_mod, "load_settings", boom)
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.err.strip() == "flybrain: FLY_MARKET: expected one of sim|dexscreener (got 'bogus')"
    assert "Traceback" not in captured.err and captured.err.isascii()
    assert captured.out == ""


def test_main_cli_contract(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """``--help`` exits 0, an unknown flag exits 2, and every flag overrides its ``FLY_*`` variable."""
    import flybrain.config as config_mod
    from flybrain.__main__ import ARG_ENV, main

    with pytest.raises(SystemExit) as helped:
        main(["--help"])
    assert helped.value.code == 0
    assert capsys.readouterr().out.isascii()
    with pytest.raises(SystemExit) as bad:
        main(["--nope"])
    assert bad.value.code == 2
    capsys.readouterr()

    seen: dict[str, str] = {}

    def capture(*args: Any, **kwargs: Any) -> Any:
        seen.update({var: os.environ.get(var, "") for var in ARG_ENV.values()})
        raise ValueError("FLY_SEED: stop before uvicorn")

    for var in ARG_ENV.values():
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config_mod, "load_settings", capture)
    assert main(["--port", "4399", "--seed", "7", "--n", "4000", "--market", "sim",
                 "--no-realtime", "--source", "synthetic"]) == 2
    assert seen["FLY_PORT"] == "4399" and seen["FLY_SEED"] == "7"
    assert seen["FLY_N_NEURONS"] == "4000" and seen["FLY_MARKET"] == "sim"
    assert seen["FLY_REALTIME"] == "0" and seen["FLY_CONNECTOME_SOURCE"] == "synthetic"


def test_bind_hosts_covers_both_loopback_families(monkeypatch: pytest.MonkeyPatch) -> None:
    """``bind_hosts`` broadens the loopback default to IPv4 + IPv6 so ``localhost`` connects first try.

    SPEC section b pins ``FLY_HOST=127.0.0.1`` yet the frontend default connects to ``localhost``, which
    Windows resolves ``::1`` before ``127.0.0.1``; binding both keeps the documented host and kills the
    ``ws://localhost:4000/ws failed`` console noise. Non-loopback hosts are never touched, and an
    IPv6-less machine keeps the single IPv4 address (never crashes on start).
    """
    import flybrain.__main__ as main_mod
    from flybrain.__main__ import bind_hosts

    monkeypatch.setattr(main_mod, "_ipv6_loopback_ok", lambda: True)
    assert bind_hosts("127.0.0.1") == ["127.0.0.1", "::1"]
    assert bind_hosts("localhost") == ["127.0.0.1", "::1"]
    assert bind_hosts("0.0.0.0") == "0.0.0.0"          # explicit all-interfaces is untouched
    assert bind_hosts("192.168.1.5") == "192.168.1.5"  # a LAN bind is untouched

    monkeypatch.setattr(main_mod, "_ipv6_loopback_ok", lambda: False)
    assert bind_hosts("127.0.0.1") == "127.0.0.1"      # no IPv6 loopback -> single IPv4, no crash


def test_main_binds_dual_stack_loopback(monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture) -> None:
    """``main`` hands uvicorn the dual-stack loopback list for the default ``FLY_HOST=127.0.0.1``."""
    import flybrain.__main__ as main_mod
    import flybrain.config as config_mod

    fake = types.SimpleNamespace(
        run_id="deadbeef", host="127.0.0.1", port=4000, connectome_source="synthetic",
        n_neurons=4000, seed=1337, dt_ms=1.0, tick_hz=20, realtime=True, market="sim",
        llm="dryrun", x_mode="dryrun", data_dir=Path("data"),
    )
    seen: dict[str, Any] = {}
    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = lambda app, **kw: seen.update(kw)  # type: ignore[attr-defined]

    monkeypatch.setattr(config_mod, "load_settings", lambda: fake)
    monkeypatch.setattr(app_mod, "create_app", lambda settings, boot_gates=True: object())
    monkeypatch.setattr(main_mod, "_ipv6_loopback_ok", lambda: True)
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    assert main_mod.main([]) == 0
    assert seen["host"] == ["127.0.0.1", "::1"] and seen["port"] == 4000
    assert capsys.readouterr().out.isascii()          # banner stays ASCII (SPEC 0.1)


@pytest.mark.skipif(not _main_ipv6_loopback(), reason="IPv6 loopback (::1) is not available on this host")
def test_uvicorn_actually_accepts_ipv6_loopback_when_bound_dual_stack() -> None:
    """End-to-end: a server bound via ``bind_hosts('127.0.0.1')`` accepts a real ``::1`` TCP connect.

    This is the exact symptom the integrator saw (``localhost`` -> ``::1`` refused): here the connect on
    both families must succeed, proving the fix at the socket layer, not just in the return value.
    """
    import contextlib
    import socket
    import threading

    import uvicorn

    from flybrain.__main__ import bind_hosts

    async def _tiny_app(scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            while True:
                msg = await receive()
                if msg["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif msg["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    hosts = bind_hosts("127.0.0.1")
    assert hosts == ["127.0.0.1", "::1"]
    server = uvicorn.Server(uvicorn.Config(_tiny_app, host=hosts, port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(200):
            if server.started:
                break
            time.sleep(0.02)
        assert server.started, "uvicorn did not start in time"

        def _connects(family: int, addr: str) -> bool:
            with contextlib.closing(socket.socket(family, socket.SOCK_STREAM)) as c:
                c.settimeout(2.0)
                try:
                    c.connect((addr, port))
                    return True
                except OSError:
                    return False

        assert _connects(socket.AF_INET, "127.0.0.1"), "IPv4 loopback refused"
        assert _connects(socket.AF_INET6, "::1"), "IPv6 loopback refused (the reported symptom)"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_state_mood_history_spans_more_than_the_tick_ring(client: TestClient) -> None:
    """``GET /api/state.mood_history`` is not bounded by the 10 s tick ring (SPEC e.5 draws 5 minutes).

    ``TickHistory(seconds=10.0)`` sizes the *full-tick* ring; ``mood_history()`` reads the compact event
    deque instead, whose window is ``COUNTER_WINDOW_S`` (15 min), so a client hydrating from REST gets a
    mood timeline far longer than the ticks the ring still holds.
    """
    from flybrain.server.state import COUNTER_WINDOW_S

    ctx = client.app.state.ctx
    ctx.loop.stop(3.0)                           # freeze: only our synthetic ticks move the history on
    base = float(ctx.history.latest()["wall"])
    for i in range(1, 2001):                     # 100 s of brain/wall time at 20 Hz
        mood = [{"kind": "mood", "t_ms": i * 50,
                 "data": {"from": "CRUISING", "to": "PANIC"}}] if i % 400 == 0 else []
        ctx.history.push({"seq": i, "t_ms": i * 50, "wall": base + i * 0.05, "events": mood})
    assert COUNTER_WINDOW_S >= 300.0
    assert len(ctx.history.last(10.0)) <= int(10.0 / 0.05) + 2     # the tick ring really is 10 s
    rows = client.get("/api/state").json()["mood_history"]
    assert [t for t, _ in rows] == [400 * 50, 800 * 50, 1200 * 50, 1600 * 50, 2000 * 50]
    assert all(state == "PANIC" for _, state in rows)
