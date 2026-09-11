"""Regression tests for workstream E7 (frontend core): ``scripts/mock_ws.py`` and narrow ``frontend/`` source pins.

``frontend/`` has no test runner - SPEC e.0 forbids new npm dependencies - so the browser half of this workstream is
guarded the way ``tests/test_docs_e8.py`` guards E8's: source assertions narrow enough to name the bug they prevent,
each skipping (never failing) when the file is absent, exactly like ``test_protocol::test_types_ts_mirror`` does for
``lib/types.ts``. The mock server half is driven for real (no sockets are opened: the tick loop and the HTTP router
are called directly).

Every test here pins a defect a review actually found:

* the 15 s dead-connection watchdog in ``lib/ws.ts`` was refreshed by *any* inbound frame, so a server that stopped
  ticking was never detected (SPEC e.2 / d.10 arm it on ``tick`` / ``pong`` only);
* the status bar mood badge waited for the next 4 Hz snapshot instead of repainting on ``mood_change`` (SPEC d.3);
* the mock emitted 4 of the 17 ``tick.events`` kinds of SPEC d.8 - ``wall_bump`` in particular was unreachable, so
  the ``bump`` stamp of e.4 and the wall handling of e.5 could not be exercised at all;
* the mock served no POST route, so the REST fallbacks of ``lib/api.ts`` hit a websockets handshake parser error;
* the raster sampler emitted ~4x the events its own ``rates.regions`` imply (SPEC d.10 size budget);
* ``hello.run_id`` was the constant ``"mock0001"``, so the changed-run_id trail-clear confirmation of d.1 could
  never be seen.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import re
import sys

from collections import Counter
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
MOCK_PATH = REPO / "scripts" / "mock_ws.py"
FRONTEND = REPO / "frontend"

#: every in-tick event kind of the SPEC d.8 table (``wrap`` only exists when walls == "wrap").
D8_TICK_KINDS = (
    "jump", "takeoff", "landing", "freeze", "unfreeze", "feed_start", "feed_stop", "groom", "song",
    "saccade", "wander_floor", "sleep", "wake", "wall_bump", "wrap", "gf_spike", "mood",
)


def load_mock() -> Any:
    """Import ``scripts/mock_ws.py`` by path (it is a script, not part of the ``flybrain`` package)."""
    if not MOCK_PATH.exists():                                    # pragma: no cover - the file is part of E7
        pytest.skip("scripts/mock_ws.py is missing")
    cached = sys.modules.get("_mock_ws_under_test")
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location("_mock_ws_under_test", MOCK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_mock_ws_under_test"] = module
    spec.loader.exec_module(module)
    return module


def read_frontend(rel: str) -> str:
    """Source of one frontend file, or skip when the frontend is not checked out."""
    path = FRONTEND / rel
    if not path.exists():                                         # pragma: no cover - the file is part of E7
        pytest.skip(f"frontend/{rel} is missing")
    return path.read_text(encoding="utf-8")


def drive(mock: Any, ticks: int, walls: str = "bounce") -> tuple[Counter, list[dict[str, Any]]]:
    """Run the mock brain through every mood with every poke channel; returns (event kinds, frames)."""
    brain = mock.MockBrain(1337, 800, 500, 20, 1.0, 48, 2000, 90.0, walls=walls, run_id="mockTEST")
    kinds: Counter = Counter()
    frames: list[dict[str, Any]] = []
    for i in range(ticks):
        if i % 150 == 0:                                          # force each mood in rotation
            dst = mock.MOOD_STATES[(i // 150) % len(mock.MOOD_STATES)]
            if dst != brain.mood:
                brain.force_mood(dst, "test sweep")
        if i % 400 == 0:
            for channel in mock.CHANNELS:
                brain.push_poke(channel, 1.0, 0, 900.0)
        frame, _oob = brain.tick()
        frames.append(frame)
        for event in frame["events"]:
            kinds[event["kind"]] += 1
    return kinds, frames


# ------------------------------------------------------------------------------------- mock: in-tick events (d.8)

@pytest.fixture(scope="module")
def swept() -> tuple[Counter, list[dict[str, Any]]]:
    """One 7.5 minute sweep (forced moods + every poke channel), shared by the tests below."""
    return drive(load_mock(), 9000)


@pytest.mark.slow
def test_mock_emits_every_d8_tick_event_kind(swept: tuple[Counter, list[dict[str, Any]]]) -> None:
    """A review found only 4 of the 17 kinds ever produced; `wall_bump` and `mood` in particular were dead code."""
    mock = load_mock()
    bounce, _ = swept
    missing = [k for k in D8_TICK_KINDS if k != "wrap" and k not in bounce]
    assert not missing, f"kinds never emitted with walls=bounce: {missing} (saw {sorted(bounce)})"
    assert "wrap" not in bounce, "wrap belongs to walls=wrap only"
    wrapped, _ = drive(mock, 4000, walls="wrap")
    assert wrapped["wrap"] > 0, "walls=wrap must emit the d.8 `wrap` event"
    assert "wall_bump" not in wrapped, "wall_bump belongs to walls=bounce only"


@pytest.mark.slow
def test_mock_wall_bump_reaches_the_wall_and_stamps_bump(swept: tuple[Counter, list[dict[str, Any]]]) -> None:
    """`hello.canvas.walls == "bounce"` must be true of the body too: the fly reaches the margin and bounces off.

    The old ellipse path clamped at +-8 px but could not get near a wall, so `ink.stamp == "bump"` (SPEC e.4) and
    the `wall_bump` branch of FlyCanvas were unreachable through the only dev server in the repo.
    """
    kinds, frames = swept
    assert kinds["wall_bump"] > 0
    xs = [f["fly"]["x"] for f in frames]
    ys = [f["fly"]["y"] for f in frames]
    # the old body stayed inside x 227..574 / y 143..357 forever; the fly must now roam the whole document
    assert min(xs) <= 20.0 and max(xs) >= 780.0, f"fly never reached a side wall: x {min(xs)}..{max(xs)}"
    assert min(ys) <= 20.0 and max(ys) >= 460.0, f"fly never reached a top/bottom wall: y {min(ys)}..{max(ys)}"
    assert all(8.0 <= x <= 792.0 for x in xs) and all(8.0 <= y <= 492.0 for y in ys), "bounce must stay inside"
    bumped = [f for f in frames if any(e["kind"] == "wall_bump" for e in f["events"])]
    assert any(f["ink"]["stamp"] == "bump" for f in bumped), "a wall_bump tick must carry the `bump` stamp"
    for frame in bumped:
        for event in frame["events"]:
            if event["kind"] == "wall_bump":
                assert event["data"]["side"] in ("L", "R")


@pytest.mark.slow
def test_mock_mood_event_duplicates_the_mood_change_frame() -> None:
    """SPEC d.8: the `mood` in-tick event is the rAF-consumer duplicate of the d.3 frame; it was never emitted."""
    mock = load_mock()
    brain = mock.MockBrain(1337, 800, 500, 20, 1.0, 48, 2000, 90.0, run_id="mockTEST")
    brain.tick()
    frame = brain.force_mood("PANIC", "anxiety 0.80 >= 0.70 for 2.0 s")
    assert frame["type"] == "mood_change" and frame["to"] == "PANIC"
    tick, _ = brain.tick()
    moods = [e for e in tick["events"] if e["kind"] == "mood"]
    assert len(moods) == 1, "exactly one `mood` event per transition"
    assert moods[0]["data"] == {"from": frame["from"], "to": "PANIC", "reason": frame["reason"]}


def test_mock_emits_all_three_d8_easter_egg_reasons(monkeypatch: Any) -> None:
    """SPEC d.8: the mock must be able to produce every easter_egg reason, not only `buys_m5 == 420`.

    A review found `price 0.000420` and `clock 04:20` were dead in the dev stand-in.
    """
    import time as _time
    mock = load_mock()

    def fresh() -> Any:
        b = mock.MockBrain(1337, 800, 500, 20, 1.0, 8, 2000, 90.0, run_id="mockTEST")
        b.easter_egg_ms = -999999
        b.t_ms = 100000                                   # well past the 30 s cooldown floor
        return b

    # sig_digits helper the price branch relies on
    assert mock.sig_digits(0.00042013).startswith("420")
    assert mock.sig_digits(6.912e-5).startswith("69")

    # buys_m5 in (69, 420) -> "buys_m5 == 420"
    b = fresh(); b.market.buys_m5, b.market.price = 420, 0.0012345
    assert b._easter_egg(1.0) == "buys_m5 == 420"
    b = fresh(); b.market.buys_m5, b.market.price = 69, 0.0012345
    assert b._easter_egg(1.0) == "buys_m5 == 420"

    # first three significant price digits 420 (or first two 69) -> "price 0.000420"
    b = fresh(); b.market.buys_m5, b.market.price = 40, 0.00042013
    assert b._easter_egg(1.0) == "price 0.000420"
    b = fresh(); b.market.buys_m5, b.market.price = 40, 6.912e-5
    assert b._easter_egg(1.0) == "price 0.000420"

    # local clock 04:20 -> "clock 04:20" (monkeypatched, per the SPEC h `test_easter_egg`)
    at_0420 = _time.struct_time((2026, 9, 11, 4, 20, 0, 0, 0, -1))
    monkeypatch.setattr(mock.time, "localtime", lambda *a: at_0420)
    b = fresh(); b.market.buys_m5, b.market.price = 40, 0.0012345
    assert b._easter_egg(1.0) == "clock 04:20"

    # a fire arms the 30 s cooldown: the next call is suppressed
    b = fresh(); b.market.buys_m5 = 420
    assert b._easter_egg(1.0) == "buys_m5 == 420"
    assert b._easter_egg(1.0) is None


# ------------------------------------------------------------------------------------- mock: size budget (d.10)

@pytest.mark.slow
def test_mock_raster_matches_its_own_published_region_rates(swept: tuple[Counter, list[dict[str, Any]]]) -> None:
    """SPEC d.10 sizes the tick from `rates.regions`; the sampler used to ignore them and emit ~4x too many events."""
    _kinds, frames = swept
    rows = 8 * 48
    events = [len(f["spikes"]["slots"]) for f in frames]
    implied = [rows * (sum(f["rates"]["regions"]) / 8.0) * 0.05 for f in frames]
    measured_mean = sum(events) / len(events)
    implied_mean = sum(implied) / len(implied)
    assert implied_mean > 0
    ratio = measured_mean / implied_mean
    assert 0.85 <= ratio <= 1.15, (
        f"raster emits {measured_mean:.0f} events/tick but the published rates imply {implied_mean:.0f}")
    assert measured_mean <= 110.0, "d.10 expects 20-100 raster events per tick at the advertised rest rates"


def test_mock_tick_frame_shape_is_wire_legal() -> None:
    """Guards the parallel/sorted spike arrays, the 131 pops keys and the absence of NaN in any frame."""
    mock = load_mock()
    brain = mock.MockBrain(4242, 800, 500, 20, 1.0, 48, 2000, 90.0, run_id="mockTEST")
    hello = brain.hello()
    assert hello["run_id"] == "mockTEST"
    assert len(hello["raster"]["rows"]) == 8 * hello["raster"]["per_region"]
    assert [r["slot"] for r in hello["raster"]["rows"]] == list(range(len(hello["raster"]["rows"])))
    assert hello["canvas"]["walls"] == "bounce"
    for _ in range(200):
        frame, _oob = brain.tick()
        spikes = frame["spikes"]
        assert len(spikes["slots"]) == len(spikes["dt"])
        assert spikes["dt"] == sorted(spikes["dt"])
        assert all(0 <= d < brain.steps_per_tick for d in spikes["dt"])
        assert all(0 <= s < len(brain.rows) for s in spikes["slots"])
        assert spikes["t0_ms"] == frame["t_ms"] - int(round(brain.tick_ms))
        assert list(frame["rates"]["pops"]) == list(hello["pops"]) and len(hello["pops"]) == 131
        assert frame["fly"]["mode"] in ("walk", "fly", "jump", "feed", "freeze", "groom", "court", "sleep")
        assert frame["ink"]["stamp"] in (None, "blob", "heart", "zzz", "dash", "bump")
        text = json.dumps(frame)
        assert "NaN" not in text and "Infinity" not in text
        for value in frame["rates"]["regions"]:
            assert math.isfinite(value)


# ------------------------------------------------------------------------------------- mock: run_id (d.1)

def test_mock_run_id_is_random_per_process_and_pinnable() -> None:
    """A constant "mock0001" made the d.1 changed-run_id trail-clear confirmation impossible to exercise."""
    mock = load_mock()
    same_seed = {mock.MockBrain(1337, 800, 500, 20, 1.0, 8, 2000, 90.0).run_id for _ in range(8)}
    assert len(same_seed) > 1, "run_id must not be derived from --seed (a restart has to look like a new session)"
    assert all(re.fullmatch(r"mock\d{4}", r) for r in same_seed)
    pinned = mock.MockBrain(1337, 800, 500, 20, 1.0, 8, 2000, 90.0, run_id="mockPIN")
    assert pinned.hello()["run_id"] == "mockPIN"
    parser = mock.build_parser()
    assert parser.parse_args([]).run_id is None
    assert parser.parse_args(["--run-id", "mockPIN"]).run_id == "mockPIN"
    assert parser.parse_args(["--walls", "wrap"]).walls == "wrap"


# ------------------------------------------------------------------------------------- mock: HTTP routes (e.4)

def http_front(mock: Any) -> Any:
    args = mock.build_parser().parse_args(["--quiet"])
    brain = mock.MockBrain(1337, 800, 500, 20, 1.0, 8, 2000, 90.0, run_id="mockTEST")
    server = mock.MockServer(brain, args)
    return mock.HttpFront(server, ws_port=0)


def route(front: Any, method: str, path: str, body: dict | None = None) -> tuple[int, Any]:
    payload = json.dumps(body).encode() if body is not None else b""
    status, raw, ctype = asyncio.run(front.route(method, path, payload))
    assert ctype in ("application/json", "text/plain")
    return status, (json.loads(raw) if raw and ctype == "application/json" else raw)


def test_mock_post_routes_answer_the_rest_fallbacks() -> None:
    """`lib/api.ts` postPoke / postMarketMode / postTweetTest / postSnapshot are the paths taken when the socket is
    down; websockets' handshake parser rejects POST before `process_request`, so they used to return a dropped
    connection plus a traceback. They are served by `HttpFront` now."""
    mock = load_mock()
    front = http_front(mock)
    assert route(front, "OPTIONS", "/api/poke")[0] == 204
    assert route(front, "POST", "/api/poke", {"stim": "sugar", "strength": 1.0, "side": "L",
                                              "duration_ms": 500}) == (200, {"ok": True})
    assert front.server.brain.pokes and front.server.brain.pokes[-1].stim == "sugar"
    status, body = route(front, "POST", "/api/poke", {"stim": "nope", "strength": 1.0, "side": "L",
                                                      "duration_ms": 500})
    assert status == 400 and body["error"] == "bad_message"
    status, body = route(front, "POST", "/api/poke", {"stim": "sugar", "strength": 2.0, "side": "L",
                                                      "duration_ms": 500})
    assert status == 400 and "strength" in body["msg"]
    status, body = route(front, "POST", "/api/market/mode", {"mode": "PUMP"})
    assert (status, body) == (200, {"ok": True, "mode": "PUMP"})
    status, body = route(front, "POST", "/api/market/mode", {"mode": "dexscreener"})
    assert status == 403 and body["error"] == "forbidden"
    png = "iVBORw0KGgo="                                           # the 8 PNG magic bytes, base64
    status, body = route(front, "POST", "/api/snapshot", {"id": "snap-1", "png_b64": png})
    assert (status, body) == (200, {"ok": True, "bytes": 8})
    status, body = route(front, "POST", "/api/snapshot", {"id": "snap-1", "png_b64": "bm90IGEgcG5n"})
    assert status == 400 and "PNG" in body["msg"]
    status, body = route(front, "POST", "/api/tweet/test", {})
    assert status == 200 and body["reason"] == "manual" and body["snapshot_source"] == "server"
    assert "type" not in body and "seq" not in body, "GET /api/tweets shape (TweetRecord = Omit<TweetMsg,type|seq>)"
    assert route(front, "POST", "/api/clear", {})[0] == 200
    assert route(front, "POST", "/api/nope", {})[0] == 404
    assert route(front, "PUT", "/api/poke", {})[0] == 405


def test_mock_get_routes_and_cors_header_match_what_is_served() -> None:
    mock = load_mock()
    front = http_front(mock)
    status, body = route(front, "GET", "/api/health")
    assert status == 200 and body["run_id"] == "mockTEST" and body["ok"] is True
    assert route(front, "GET", "/api/state")[0] == 200
    assert route(front, "GET", "/api/tweets")[0] == 200
    assert route(front, "GET", "/api/groups")[0] == 200
    assert route(front, "GET", "/nope")[0] == 404
    head = mock.http_response(200, b"{}", "application/json").decode("ascii")
    assert "Access-Control-Allow-Methods: GET, POST, OPTIONS" in head
    assert "Access-Control-Allow-Origin: *" in head


def test_mock_rest_poke_keeps_the_two_per_second_budget() -> None:
    mock = load_mock()
    front = http_front(mock)
    body = {"stim": "sugar", "strength": 1.0, "side": "both", "duration_ms": 500}
    codes = [route(front, "POST", "/api/poke", body)[0] for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[2:] == [429, 429], codes


# ------------------------------------------------------------------------------------- frontend source pins

def test_ws_watchdog_is_armed_by_tick_and_pong_only() -> None:
    """SPEC e.2/d.10: ping every 10 s, close after 15 s without a `pong` or a `tick`.

    The bug: `lastActivity` was stamped inside `sock.onmessage`, so any low-rate out-of-band frame (an `event` every
    2 s, say) kept a dead tick stream alive forever and the client never reconnected.
    """
    src = read_frontend("lib/ws.ts")
    assert "const PING_MS = 10_000;" in src and "const DEAD_MS = 15_000;" in src
    onmessage = src[src.index("sock.onmessage"):src.index("sock.onerror")]
    assert "lastActivity" not in onmessage.replace("// NOTE: `lastActivity`", ""), (
        "the generic onmessage path must NOT refresh the dead-connection watchdog")
    dispatch = src[src.index("const dispatch ="):src.index("const scheduleReconnect")]
    for case in ('case "hello":', 'case "tick":', 'case "pong":'):
        branch = dispatch[dispatch.index(case):]
        branch = branch[:branch.index("break;")]
        assert "lastActivity = performance.now()" in branch, f"{case} must arm the watchdog"
    mood = dispatch[dispatch.index('case "mood_change":'):]
    mood = mood[:mood.index("break;")]
    assert "lastActivity" not in mood, "an out-of-band frame must not count as a live tick stream"
    assert "performance.now() - lastActivity > DEAD_MS" in src


def test_status_bar_repaints_the_mood_badge_on_mood_change() -> None:
    """SPEC d.3: the badge repaints on the frame, not on the next 4 Hz snapshot."""
    status = read_frontend("components/StatusBar.tsx")
    paint = read_frontend("components/PaintWindow.tsx")
    assert "moodNow" in status and "moodNow?: { to: Mood; t_ms: number } | null" in status
    assert 'on("mood_change"' in paint, "PaintWindow (which owns the socket) must feed the badge"
    assert "moodNow={moodNow}" in paint


def test_status_bar_and_snapshot_caption_use_the_spec_strings() -> None:
    """e.5 prints `${n} n / ${e} e` (no thousands separators) and e.6's caption has no `$` before the price."""
    status = read_frontend("components/StatusBar.tsx")
    assert "toLocaleString" not in status
    assert "n / ${hello.connectome.e} e" in status
    canvas = read_frontend("components/FlyCanvas.tsx")
    assert "`FlyBrain | ${mood} | ${sym} ${price} ${chg} | t=${secs}s`" in canvas


def test_interp_has_only_the_two_snap_conditions_of_e3() -> None:
    """e.3 snaps when `prev` is null or the gap exceeds 4 ticks - an extra distance guard was undocumented."""
    src = read_frontend("lib/interp.ts")
    assert "SNAP_DIST_PX" not in src
    assert "gap > 4 * tickMs" in src


def test_interp_lerps_exactly_the_e3_fields_and_takes_leg_phase_from_latest() -> None:
    """e.3 enumerates x/y/heading/speed as lerped; leg_phase (like mode/wing_*/proboscis/jump_t_ms) is from latest.

    A review found interp.ts short-arc-interpolating leg_phase, an interpolation e.3 never lists.
    """
    src = read_frontend("lib/interp.ts")
    for assigned in ("pose.x =", "pose.y =", "pose.heading =", "pose.speed ="):
        assert assigned in src, f"e.3 field {assigned!r} must still be interpolated"
    assert "pose.leg_phase" not in src, "e.3 does not lerp leg_phase; poseOf(latest) already carries it"


def test_color_palette_fore_square_is_ink_color_verbatim() -> None:
    """e.5: the fore square of the current-colour box shows ink.color, not the animated EUPHORIA rainbow."""
    src = read_frontend("components/ColorPalette.tsx")
    assert "const fore = ink.color;" in src
    assert 'moodColor("EUPHORIA"' not in src, "fore is ink.color verbatim, never moodColor(EUPHORIA)"
    assert "moodColor(mood" in src, "the back square still shows moodColor(mood)"


def test_drawing_libs_guard_nan_and_keep_state_per_canvas() -> None:
    """e.8: a malformed frame never breaks the loop; two canvases must not share the ink / sparkle cadence."""
    ink = read_frontend("lib/ink.ts")
    assert "Number.isFinite(x0)" in ink and "Number.isFinite(y1)" in ink
    assert "new WeakMap<CanvasRenderingContext2D, InkState>()" in ink
    assert "export function resetInkState(ctx: CanvasRenderingContext2D)" in ink
    sprite = read_frontend("lib/flySprite.ts")
    assert "let frameCounter" not in sprite and "let sparkle" not in sprite
    assert "new WeakMap<CanvasRenderingContext2D, SpriteState>()" in sprite


def test_frontend_package_json_is_the_scaffold_plus_typecheck() -> None:
    """SPEC i.3 / e.0: the only allowed package.json change is the `typecheck` script.

    Documented deviation: `three` / `@types/three` were added for the `BrainView3D` window (the live 3D fly-brain
    view, commit 7e849f0), which needs WebGL and cannot be done with the 2-D canvas stack of SPEC e.4. Nothing else
    may appear here - the rest of the frontend stays dependency-free.
    """
    raw = read_frontend("package.json")
    pkg = json.loads(raw)
    assert set(pkg["dependencies"]) == {"next", "react", "react-dom", "three", "@types/three"}
    assert {k: pkg["dependencies"][k] for k in ("next", "react", "react-dom")} == {
        "next": "16.3.4", "react": "19.2.8", "react-dom": "19.2.8"}
    assert pkg["scripts"] == {"dev": "next dev", "build": "next build", "start": "next start",
                              "lint": "eslint", "typecheck": "tsc --noEmit"}
    env = read_frontend(".env.local.example").strip().splitlines()
    assert env == ["NEXT_PUBLIC_WS_URL=ws://localhost:4000/ws", "NEXT_PUBLIC_API_URL=http://localhost:4000"]


def test_mock_ws_help_documents_the_post_routes() -> None:
    """The advertised `Access-Control-Allow-Methods: ... POST` must not be a promise the server cannot keep."""
    mock = load_mock()
    text = " ".join(mock.build_parser().format_help().split())          # argparse re-wraps the epilog
    assert "POST /api/{poke,market/mode,tweet/test,snapshot,clear}" in text
    assert "--run-id" in text and "--walls" in text and "--cap" in text
