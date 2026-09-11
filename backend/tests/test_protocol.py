"""Tests for the wire contract and the server core (SPEC h.2 ``test_protocol.py``).

The SPEC section d examples are not retyped here: they are parsed straight out of ``docs/SPEC.md``
(every ```json fenced block) so the models are validated against the normative text itself and a spec
edit cannot silently drift away from the code. ``test_types_ts_mirror`` does the same for
``frontend/lib/types.ts``.

Beyond the normative names this file also covers the rest of the E6a surface: ``flybrain.config``
(SPEC b / c.1), ``flybrain.log`` (c.29), ``server/state.py``, ``server/session_log.py`` and
``server/png.py`` (c.25).
"""

from __future__ import annotations

import json
import logging
import re
import struct
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

from flybrain import config as cfg
from flybrain import log as flog
from flybrain.config import Settings, gain_value, load_settings, parse_bool, parse_dotenv, redacted
from flybrain.server import protocol as P
from flybrain.server.png import write_png
from flybrain.server.session_log import SessionLog, read_session
from flybrain.server.state import StateBus, TickHistory

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "docs" / "SPEC.md"
TYPES_TS = REPO / "frontend" / "lib" / "types.ts"


# =====================================================================================================
# SPEC example loading helpers
# =====================================================================================================


def _strip_line_comments(text: str) -> str:
    """Remove ``//`` comments (SPEC d.9 annotates the error frame with one)."""
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    return re.sub(r"(?m)\s//\s[^\n\"]*$", "", text)


def _json_objects(block: str) -> list[dict]:
    """Every JSON object in one fenced block (whole block first, then line by line)."""
    body = _strip_line_comments(block)
    try:
        parsed = json.loads(body)
        return [parsed] if isinstance(parsed, dict) else []
    except ValueError:
        pass
    out: list[dict] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


@pytest.fixture(scope="module")
def spec_frames() -> dict[str, dict]:
    """``{frame type: example dict}`` for every ```json block of SPEC section d."""
    if not SPEC_PATH.is_file():  # pragma: no cover - the spec ships with the repo
        pytest.skip(f"{SPEC_PATH} not found")
    text = SPEC_PATH.read_text(encoding="utf-8")
    frames: dict[str, dict] = {}
    for block in re.findall(r"```json\n(.*?)```", text, flags=re.S):
        for obj in _json_objects(block):
            kind = obj.get("type")
            if isinstance(kind, str) and kind not in frames:
                frames[kind] = obj
    missing = {"hello", "tick", "mood_change", "tweet", "market", "event", "snapshot_request", "pong",
               "error"} - set(frames)
    assert not missing, f"SPEC section d examples missing from {SPEC_PATH.name}: {sorted(missing)}"
    return frames


def _hello_example(spec_frames: dict[str, dict], *, fill: bool = True) -> dict:
    """The SPEC d.1 hello with its ``"..."`` placeholders replaced by generated rows / pop keys."""
    hello = json.loads(json.dumps(spec_frames["hello"]))
    rows = [r for r in hello["raster"]["rows"] if isinstance(r, dict)]
    hello["pops"] = [p for p in hello["pops"] if isinstance(p, str) and not p.startswith("...")]
    if fill:
        per_region = int(hello["raster"]["per_region"])
        total = per_region * len(P.REGION_NAMES)
        generated: list[dict] = []
        for slot in range(total):
            region = slot // per_region
            existing = next((r for r in rows if int(r["slot"]) == slot), None)
            if existing is not None:
                generated.append(existing)
                continue
            generated.append({"region": region, "slot": slot, "neuron": 1000 + slot,
                              "label": f"T{region}_{slot}", "side": "L" if slot % 2 == 0 else "R",
                              "star": False})
        hello["raster"]["rows"] = generated
    else:
        hello["raster"]["rows"] = rows
    return hello


def _pop_keys(spec_frames: dict[str, dict]) -> list[str]:
    return list(spec_frames["tick"]["rates"]["pops"].keys())


def _synthetic_tick(spec_frames: dict[str, dict], n_events: int) -> dict:
    """A rest-state tick with the full 131-key ``pops`` table and ``n_events`` raster events."""
    tick = json.loads(json.dumps(spec_frames["tick"]))
    tick["rates"]["pops"] = {k: 0.0 for k in _pop_keys(spec_frames)}
    tick["spikes"] = {"t0_ms": 617300, "win_ms": 50, "total": n_events, "capped": n_events >= 2000,
                      "slots": [(i * 7) % 384 for i in range(n_events)],
                      "dt": sorted((i * 3) % 50 for i in range(n_events))}
    tick["events"] = []
    return tick


# =====================================================================================================
# h.2: the normative protocol tests
# =====================================================================================================


def test_hello_model_validates_example(spec_frames: dict[str, dict]) -> None:
    """SPEC d.1 JSON (pops / rows filled programmatically) validates and round-trips."""
    hello = _hello_example(spec_frames)
    msg = P.HelloMsg.model_validate(hello)
    assert msg.type == "hello" and msg.v == P.PROTOCOL_VERSION
    assert msg.run_id == "9f3a2c1b"
    assert len(msg.raster.rows) == msg.raster.per_region * len(P.REGION_NAMES)
    assert [r.slot for r in msg.raster.rows] == list(range(len(msg.raster.rows)))  # slot == index (d.1)
    assert tuple(msg.regions) == P.REGION_NAMES
    assert list(msg.mood_states) == list(P.MOOD_STATES)
    assert list(msg.channels) == list(P.POKE_STIMS)
    assert list(msg.market_modes) == list(P.MARKET_MODES)
    assert set(msg.connectome.region_counts) == set(P.REGION_NAMES)
    assert msg.connectome.source == "synthetic" and "synthetic" in msg.connectome.note

    # the example's key set is exactly the model's wire key set (no drift in either direction)
    assert set(hello) == P.wire_fields(P.HelloMsg)
    round_tripped = json.loads(P.server_json(msg))
    assert set(round_tripped) == set(hello)


def test_tick_model_validates_example(spec_frames: dict[str, dict]) -> None:
    """SPEC d.2 JSON verbatim validates; every nested object is typed and complete."""
    tick = spec_frames["tick"]
    msg = P.TickMsg.model_validate(tick)
    assert msg.seq == 12345 and msg.t_ms == 617350
    assert set(tick) == P.wire_fields(P.TickMsg)
    for key, model in (("sim", P.SimStats), ("fly", P.FlyState), ("ink", P.InkStyle), ("mood", P.MoodState),
                       ("market", P.TickMarket), ("drives", P.Drives), ("spikes", P.Spikes)):
        assert set(tick[key]) == P.wire_fields(model), key
    assert set(tick["rates"]) == P.wire_fields(P.Rates)
    assert len(msg.rates.regions) == len(P.REGION_NAMES)
    assert len(msg.rates.pops) == 131  # SPEC d.2: 131 keys for the synthetic readout table
    assert len(msg.spikes.slots) == len(msg.spikes.dt)
    assert msg.fly.mode in P.FLY_MODES and msg.ink.style in P.INK_STYLE_NAMES
    assert msg.mood.state in P.MOOD_STATES and msg.drives.candle in P.CANDLES
    assert msg.market.mode in P.MARKET_SOURCE_NAMES and msg.market.last_trade is not None
    assert [e.kind for e in msg.events] == ["feed_stop"]
    assert all(e.kind in P.TICK_EVENT_KINDS for e in msg.events)


def test_mood_change_tweet_market_event_examples(spec_frames: dict[str, dict]) -> None:
    """SPEC d.3, d.4, d.5, d.8 and d.9 examples validate against their models."""
    mood = P.MoodChangeMsg.model_validate(spec_frames["mood_change"])
    assert mood.from_ == "FEEDING" and mood.to == "CRUISING" and mood.mood.state == "CRUISING"
    assert set(spec_frames["mood_change"]) == P.wire_fields(P.MoodChangeMsg)
    assert json.loads(P.server_json(mood))["from"] == "FEEDING"  # the wire key is "from", not "from_"

    tweet = P.TweetMsg.model_validate(spec_frames["tweet"])
    assert tweet.dry_run is True and tweet.posted is False and tweet.snapshot_source == "browser"
    assert tweet.model == "template" and tweet.mood == "EUPHORIA" and tweet.neurons
    assert set(spec_frames["tweet"]) == P.wire_fields(P.TweetMsg)

    market = P.MarketMsg.model_validate(spec_frames["market"])
    assert market.mode == "sim" and len(market.trades) == 2
    assert market.market.regime == "PUMP" and market.market.source == "sim"
    assert set(spec_frames["market"]) == P.wire_fields(P.MarketMsg)
    assert set(spec_frames["market"]["trades"][0]) == P.wire_fields(P.Trade)

    event = P.EventMsg.model_validate(spec_frames["event"])
    assert event.kind == "market_source" and "mode" in event.data
    assert event.kind in P.OOB_EVENT_KINDS
    assert set(spec_frames["event"]) == P.wire_fields(P.EventMsg)

    snap = P.SnapshotRequestMsg.model_validate(spec_frames["snapshot_request"])
    assert snap.id == "snap-17" and snap.deadline_ms == 3000
    pong = P.PongMsg.model_validate(spec_frames["pong"])
    assert pong.seq == 12345 and pong.t > 0
    err = P.ErrorMsg.model_validate(spec_frames["error"])
    assert err.code == "bad_message" and err.msg
    for example, model in ((spec_frames["snapshot_request"], P.SnapshotRequestMsg),
                           (spec_frames["pong"], P.PongMsg), (spec_frames["error"], P.ErrorMsg)):
        assert set(example) == P.wire_fields(model)

    # the small builders produce exactly those key sets
    assert set(P.error_frame("rate_limited", "slow down")) == P.wire_fields(P.ErrorMsg)
    assert set(P.pong_frame(1.0, 3)) == P.wire_fields(P.PongMsg)
    assert set(P.snapshot_request_frame("snap-1")) == P.wire_fields(P.SnapshotRequestMsg)
    assert set(P.event_frame("clear", {"by": "api"})) == P.wire_fields(P.EventMsg)


def test_parse_client_all_types() -> None:
    """Every SPEC d.7 frame parses; bad values and extra keys raise ``ValueError`` (extra='forbid')."""
    poke = P.parse_client('{"type":"poke","stim":"sugar","strength":1.0,"side":"both","duration_ms":500}')
    assert isinstance(poke, P.PokeMsg) and poke.stim == "sugar" and poke.side_int == 0
    assert P.parse_client('{"type":"poke","stim":"loom","side":"L"}').side_int == -1
    assert P.parse_client('{"type":"poke","stim":"loom","side":"R"}').side_int == 1
    mode = P.parse_client('{"type":"set_market_mode","mode":"PUMP"}')
    assert isinstance(mode, P.SetMarketModeMsg) and mode.mode == "PUMP"
    assert isinstance(P.parse_client('{"type":"clear"}'), P.ClearMsg)
    assert isinstance(P.parse_client('{"type":"tweet_test"}'), P.TweetTestMsg)
    ping = P.parse_client('{"type":"ping","t":1789051563.1}')
    assert isinstance(ping, P.PingMsg) and ping.t == pytest.approx(1789051563.1)
    snap = P.parse_client('{"type":"snapshot","id":"snap-17","png_b64":"iVBORw0="}')
    assert isinstance(snap, P.SnapshotMsg) and snap.id == "snap-17"
    assert isinstance(P.parse_client(b'{"type":"clear"}'), P.ClearMsg)          # bytes frame
    assert isinstance(P.parse_client({"type": "clear"}), P.ClearMsg)            # already decoded
    assert set(P.CLIENT_MODELS) == {"poke", "set_market_mode", "clear", "tweet_test", "ping", "snapshot"}

    # every poke stim of hello.channels is accepted
    for stim in P.POKE_STIMS:
        assert P.parse_client(json.dumps({"type": "poke", "stim": stim})).stim == stim

    bad = [
        '{"type":"poke","stim":"nope"}',                        # unknown stim
        '{"type":"poke","stim":"sugar","strength":2}',          # strength out of range
        '{"type":"poke","stim":"sugar","strength":-0.1}',
        '{"type":"poke","stim":"sugar","duration_ms":10}',      # below 50 ms
        '{"type":"poke","stim":"sugar","duration_ms":99999}',   # above 5000 ms
        '{"type":"poke","stim":"sugar","side":"middle"}',
        '{"type":"poke","stim":"sugar","oops":1}',              # extra key
        '{"type":"clear","oops":1}',
        '{"type":"set_market_mode","mode":"MOON"}',
        '{"type":"snapshot","id":"x"}',                         # missing png_b64
        '{"type":"nope"}',                                      # unknown type
        '{"seq":1}',                                            # no type
        "[]",                                                   # not an object
        "not json at all",
    ]
    for raw in bad:
        with pytest.raises(ValueError) as excinfo:
            P.parse_client(raw)
        err = excinfo.value
        assert isinstance(err, P.BadClientFrame)
        frame = err.to_frame()
        assert frame["type"] == "error" and frame["code"] == "bad_message" and frame["msg"]
        P.ErrorMsg.model_validate(frame)

    # the poke frame converts to an encoder Poke and to its event payload
    p = P.parse_client('{"type":"poke","stim":"bitter","strength":0.5,"side":"R","duration_ms":800}')
    poke_obj = p.to_poke(t_ms=1_000)
    assert poke_obj.stim == "bitter" and poke_obj.side == 1 and poke_obj.until_ms == 1_800
    assert p.to_event_data() == {"stim": "bitter", "strength": 0.5, "side": "R", "duration_ms": 800}


def test_tick_to_json_rounding(spec_frames: dict[str, dict]) -> None:
    """Positions 1 dp, angles 3 dp, rates 2 dp, drives / mood 3 dp; NaN raises ``ValueError``."""
    tick = json.loads(json.dumps(spec_frames["tick"]))
    tick["fly"].update(x=412.34567, y=233.98765, vx=-38.24999, speed=40.0999, heading=2.8345678,
                       omega=-0.4149999)
    tick["rates"]["regions"] = [3.21456] * 8
    tick["rates"]["pops"]["gf"] = 1.23456
    tick["drives"]["sugar"] = 0.6123456
    tick["mood"]["euphoria"] = 0.1234567
    tick["ink"]["alpha"] = 0.9999999
    tick["market"]["price_usd"] = 0.00123456789          # prices keep full precision
    tick["wall"] = 1789051563.1234567

    out = json.loads(P.tick_to_json(tick))
    assert out["fly"]["x"] == 412.3 and out["fly"]["y"] == 234.0
    assert out["fly"]["vx"] == -38.2 and out["fly"]["speed"] == 40.1
    assert out["fly"]["heading"] == 2.835 and out["fly"]["omega"] == -0.415
    assert out["rates"]["regions"] == [3.21] * 8
    assert out["rates"]["pops"]["gf"] == 1.23
    assert out["drives"]["sugar"] == 0.612
    assert out["mood"]["euphoria"] == 0.123
    assert out["ink"]["alpha"] == 1.0
    assert out["market"]["price_usd"] == 0.00123456789
    assert out["wall"] == 1789051563.123
    assert out["spikes"] == tick["spikes"] and out["events"] == tick["events"]

    # the rounding pass never mutates the caller's dict
    assert tick["fly"]["x"] == 412.34567
    # minified: no spaces after separators
    raw = P.tick_to_json(tick)
    assert '", "' not in raw and '": ' not in raw
    # the rounded tick still validates
    P.TickMsg.model_validate(out)

    for bad_value in (float("nan"), float("inf"), float("-inf")):
        broken = json.loads(json.dumps(spec_frames["tick"]))
        broken["rates"]["pops"]["gf"] = bad_value
        with pytest.raises(ValueError):
            P.tick_to_json(broken)


def test_tick_json_size_budget(spec_frames: dict[str, dict]) -> None:
    """SPEC d.10 budget: a 100-event tick stays small, a capped 2000-event tick stays under 24 KB.

    Measured with the full 131-key ``pops`` table: fixed part ~3.4 KB (``pops`` ~2.0 KB, ``market``
    ~0.45 KB), raster ~6.5 bytes/event. SPEC d.10's "2.2 KB fixed part" is optimistic, so the 100-event
    bound here is 4.5 KB; the per-event cost (the part that actually scales) is asserted directly.
    """
    small = P.tick_to_json(_synthetic_tick(spec_frames, 100))
    big = P.tick_to_json(_synthetic_tick(spec_frames, 2000))
    assert len(small) < 4_608, len(small)
    assert len(big) < 24_576, len(big)
    per_event = (len(big) - len(small)) / 1900.0
    assert per_event < 9.0, per_event
    assert json.loads(big)["spikes"]["capped"] is True
    P.TickMsg.model_validate(json.loads(big))


def test_types_ts_mirror() -> None:
    """``frontend/lib/types.ts`` interface field names equal the pydantic models' wire fields (SPEC e.1)."""
    if not TYPES_TS.is_file():
        pytest.skip(f"{TYPES_TS} absent; frontend types cannot be mirrored")
    interfaces = _parse_ts_interfaces(TYPES_TS.read_text(encoding="utf-8"))
    assert interfaces, "no 'export interface' block parsed from types.ts"

    models = {
        "HelloMsg": P.HelloMsg, "TickMsg": P.TickMsg, "SimStats": P.SimStats, "FlyState": P.FlyState,
        "InkStyle": P.InkStyle, "MoodState": P.MoodState, "MarketSnapshot": P.MarketSnapshot,
        "TickMarket": P.TickMarket, "Drives": P.Drives, "Spikes": P.Spikes,
        "MoodChangeMsg": P.MoodChangeMsg, "TweetMsg": P.TweetMsg, "MarketMsg": P.MarketMsg,
        "EventMsg": P.EventMsg, "SnapshotRequestMsg": P.SnapshotRequestMsg, "PongMsg": P.PongMsg,
        "ErrorMsg": P.ErrorMsg,
        # not in the normative list, but they have models and must not drift either
        "RasterRow": P.RasterRow, "Trade": P.Trade, "TickEvent": P.TickEvent,
        "StateResponse": P.StateResponse, "HealthResponse": P.HealthResponse,
    }
    problems: list[str] = []
    for name, model in models.items():
        if name not in interfaces:
            problems.append(f"{name}: no 'export interface {name}' in types.ts")
            continue
        ts_fields = interfaces[name]
        py_fields = P.wire_fields(model)
        missing = sorted(py_fields - ts_fields)
        extra = sorted(ts_fields - py_fields)
        if missing or extra:
            problems.append(f"{name}: missing in types.ts {missing}; missing in {model.__name__} {extra}")
    assert not problems, "frontend/lib/types.ts and server/protocol.py drifted:\n" + "\n".join(problems)


def _parse_ts_interfaces(text: str) -> dict[str, set[str]]:
    """``{InterfaceName: {field names}}`` from ``export interface X [extends Y] { ... }`` blocks.

    Deliberately tolerant (SPEC e.1 is hand-maintained TypeScript): comments are stripped, optional
    markers (``?``) and ``readonly`` ignored, nested object literals skipped (only depth-0 fields count),
    index signatures ignored, and ``extends`` bases are merged when they are local interfaces.
    """
    body = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    body = re.sub(r"(?m)//.*$", "", body)
    own: dict[str, set[str]] = {}
    bases: dict[str, list[str]] = {}
    for match in re.finditer(r"export\s+interface\s+([A-Za-z_$][\w$]*)\s*(extends\s+([^{]+?))?\s*\{", body):
        name = match.group(1)
        base_text = match.group(3) or ""
        bases[name] = [b.strip() for b in base_text.split(",") if b.strip()]
        inner = _balanced(body, match.end() - 1)
        own[name] = _ts_field_names(inner)
    out: dict[str, set[str]] = {}
    for name in own:
        fields = set(own[name])
        for base in bases.get(name, ()):
            if base in own:  # generic helpers such as Omit<...> are ignored on purpose
                fields |= own[base]
        out[name] = fields
    return out


def _balanced(text: str, open_index: int) -> str:
    """The text between ``text[open_index] == '{'`` and its matching ``'}'``."""
    depth = 0
    for i in range(open_index, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_index + 1:i]
    return ""  # pragma: no cover - unbalanced types.ts


def _ts_field_names(inner: str) -> set[str]:
    """Field names declared at brace depth 0 of an interface body."""
    fields: set[str] = set()
    depth = 0
    segment: list[str] = []
    segments: list[str] = []
    for ch in inner:
        if ch in "{[(":
            depth += 1
        elif ch in "}])":
            depth -= 1
        if depth == 0 and ch in ";\n,":
            segments.append("".join(segment))
            segment = []
        else:
            segment.append(ch)
    segments.append("".join(segment))
    for seg in segments:
        s = seg.strip()
        if not s or s.startswith("["):
            continue
        m = re.match(r"^(?:readonly\s+)?([A-Za-z_$][\w$]*)\s*\??\s*:", s)
        if m:
            fields.add(m.group(1))
    return fields


def test_rest_models() -> None:
    """REST bodies and responses of SPEC c.28 (``PokeRequest`` defaults, ``HealthResponse``)."""
    req = P.PokeRequest.model_validate({"stim": "sugar"})
    assert (req.strength, req.side, req.duration_ms) == (1.0, "both", 500)
    assert req.as_frame().to_poke(100).until_ms == 600
    with pytest.raises(Exception):
        P.PokeRequest.model_validate({"stim": "sugar", "nope": 1})      # extra='forbid'
    with pytest.raises(Exception):
        P.PokeRequest.model_validate({"stim": "unknown"})
    assert P.MarketModeRequest.model_validate({"mode": "RUG"}).mode == "RUG"
    assert P.SnapshotUpload.model_validate({"id": "s1", "png_b64": "iVBORw0="}).id == "s1"

    health = P.HealthResponse.model_validate({
        "ok": True, "uptime_s": 12.5, "run_id": "9f3a2c1b", "rtf": 2.3, "speed": 1.0, "seq": 10,
        "t_ms": 500, "backend": "numpy",
        "connectome": {"name": "synthetic-20000-s1337-cal", "source": "synthetic", "n": 20000, "e": 548120,
                       "gain": 1.0, "license": "synthetic (no data)"},
        "market": {"mode": "sim", "ok": True, "last_poll": None, "failures": 0},
        "agent": {"llm": "dryrun", "x": "dryrun", "tweets_today": 0, "last_tweet_wall": None,
                  "disabled_reason": None},
        "clients": 1, "log_tail": ["12:00:00 INFO flybrain: up"],
    })
    assert health.ok and health.connectome.n == 20000 and health.market.mode == "sim"
    assert health.agent.llm == "dryrun" and health.log_tail
    dumped = json.loads(P.server_json(health))
    assert set(dumped) == P.wire_fields(P.HealthResponse)


# =====================================================================================================
# c.1 / section b: config
# =====================================================================================================


def test_settings_defaults_are_offline(tmp_path: Path) -> None:
    """``Settings()`` with zero env is the offline / dry-run default set of SPEC section b."""
    s = Settings()
    assert (s.connectome_source, s.n_neurons, s.mean_outdeg, s.synth_weights) == ("synthetic", 20_000, 25,
                                                                                 "calibrated")
    assert (s.dt_ms, s.seed, s.backend, s.gain) == (1.0, 1337, "numpy", "auto")
    assert (s.noise_mu, s.noise_sigma, s.drive_mode) == (0.5, 3.5, "poisson")
    assert (s.tick_hz, s.realtime, s.market, s.llm, s.x_mode) == (20, True, "sim", "dryrun", "dryrun")
    assert (s.llm_model, s.llm_json, s.llm_fallbacks, s.tweet_lang) == ("claude-opus-5", True, False, "en")
    assert (s.port, s.host, s.cors_origins) == (4000, "127.0.0.1", ("http://localhost:3000",))
    assert (s.canvas_w, s.canvas_h, s.walls) == (800, 500, "bounce")
    assert (s.raster_per_region, s.raster_cap) == (48, 2000)
    assert (s.mood_feedback, s.explore_baseline, s.wander_sigma, s.easter_eggs) == (True, 0.25, 0.6, True)
    assert (s.session_log, s.replay, s.log_level) == (True, "", "INFO")
    assert (s.tweet_cooldown_s, s.tweet_reason_cooldown_s, s.tweets_per_day) == (900, 2700, 12)
    assert (s.dex_poll_s, s.sim_regime_s, s.chain, s.token_address) == (60, 90.0, "solana", "")

    loaded = load_settings(env={"FLY_DATA_DIR": str(tmp_path / "d"), "FLY_OUT_DIR": str(tmp_path / "o")},
                           dotenv=None)
    assert loaded.connectome_source == "synthetic" and loaded.market == "sim"
    assert loaded.data_dir.is_absolute() and loaded.out_dir.is_absolute()
    assert loaded.data_dir.is_dir() and loaded.out_dir.is_dir()      # created on demand
    assert loaded.tick_ms == 50.0 and loaded.steps_per_tick == 50
    assert re.fullmatch(r"[0-9a-f]{8}", loaded.run_id) and loaded.connectome_key


def test_load_settings_default_dirs_come_from_file_not_cwd() -> None:
    """``data_dir`` / ``out_dir`` default to ``<repo>/data`` and ``<repo>/out`` (SPEC section a)."""
    s = load_settings(env={}, dotenv=None)
    assert s.data_dir == cfg.repo_root() / "data"
    assert s.out_dir == cfg.repo_root() / "out"
    assert cfg.repo_root() == REPO
    # a relative FLY_DATA_DIR resolves against the repo root, never the CWD
    rel = load_settings(env={"FLY_DATA_DIR": "data"}, dotenv=None)
    assert rel.data_dir == REPO / "data"
    assert rel.sessions_dir == rel.data_dir / "sessions"
    # deeper relative paths resolve the same way (checked without creating anything)
    assert cfg._resolve_dir("data/alt", REPO / "data") == REPO / "data" / "alt"
    assert cfg._resolve_dir("", REPO / "data") == REPO / "data"


def test_load_settings_dotenv_then_environ(tmp_path: Path) -> None:
    """``.env`` is parsed first, the environment wins; blank values fall back to the default."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment line\n"
        "\n"
        "FLY_N_NEURONS=8000\n"
        "FLY_SEED=7\n"
        "export FLY_BACKEND=auto\n"
        "FLY_CONNECTOME_NAME='quoted name'\n"
        "FLY_DATA_DIR=\n"
        "not a setting line\n"
        "FLY_TICK_HZ = 10 \n",
        encoding="utf-8")
    parsed = parse_dotenv(dotenv)
    assert parsed["FLY_N_NEURONS"] == "8000" and parsed["FLY_BACKEND"] == "auto"
    assert parsed["FLY_CONNECTOME_NAME"] == "quoted name" and parsed["FLY_DATA_DIR"] == ""
    assert parsed["FLY_TICK_HZ"] == "10"

    s = load_settings(env={"FLY_SEED": "9", "FLY_OUT_DIR": str(tmp_path / "out")}, dotenv=dotenv)
    assert s.n_neurons == 8000          # from .env
    assert s.seed == 9                  # environ wins
    assert s.backend == "auto"
    assert s.connectome_name == "quoted name"
    assert s.tick_hz == 10 and s.tick_ms == 100.0 and s.steps_per_tick == 100
    assert s.data_dir == cfg.repo_root() / "data"   # blank -> default
    assert parse_dotenv(tmp_path / "missing.env") == {}


@pytest.mark.parametrize("env,var", [
    ({"FLY_CONNECTOME_SOURCE": "brain"}, "FLY_CONNECTOME_SOURCE"),
    ({"FLY_N_NEURONS": "10"}, "FLY_N_NEURONS"),
    ({"FLY_N_NEURONS": "banana"}, "FLY_N_NEURONS"),
    ({"FLY_MEAN_OUTDEG": "1"}, "FLY_MEAN_OUTDEG"),
    ({"FLY_SYNTH_WEIGHTS": "guessed"}, "FLY_SYNTH_WEIGHTS"),
    ({"FLY_SUBSET": "some"}, "FLY_SUBSET"),
    ({"FLY_MIN_WEIGHT": "0"}, "FLY_MIN_WEIGHT"),
    ({"FLY_DT_MS": "0.3"}, "FLY_DT_MS"),
    ({"FLY_BACKEND": "cuda"}, "FLY_BACKEND"),
    ({"FLY_GAIN": "-1"}, "FLY_GAIN"),
    ({"FLY_GAIN": "soft"}, "FLY_GAIN"),
    ({"FLY_DRIVE_MODE": "spike"}, "FLY_DRIVE_MODE"),
    ({"FLY_TICK_HZ": "1"}, "FLY_TICK_HZ"),
    ({"FLY_TICK_HZ": "60"}, "FLY_TICK_HZ"),
    ({"FLY_REALTIME": "maybe"}, "FLY_REALTIME"),
    ({"FLY_MARKET": "binance"}, "FLY_MARKET"),
    ({"FLY_DEX_POLL_S": "5"}, "FLY_DEX_POLL_S"),
    ({"FLY_LLM": "gpt"}, "FLY_LLM"),
    ({"FLY_TWEET_LANG": "de"}, "FLY_TWEET_LANG"),
    ({"FLY_X": "broadcast"}, "FLY_X"),
    ({"FLY_PORT": "0"}, "FLY_PORT"),
    ({"FLY_WALLS": "sticky"}, "FLY_WALLS"),
    ({"FLY_RASTER_PER_REGION": "4"}, "FLY_RASTER_PER_REGION"),
    ({"FLY_EXPLORE_BASELINE": "1.5"}, "FLY_EXPLORE_BASELINE"),
    ({"FLY_CORS_ORIGINS": " , "}, "FLY_CORS_ORIGINS"),
    ({"FLY_LOG_LEVEL": "CHATTY"}, "FLY_LOG_LEVEL"),
])
def test_load_settings_rejects_bad_values(env: dict, var: str) -> None:
    """Every invalid value raises ``ValueError`` naming the offending variable (SPEC c.1)."""
    with pytest.raises(ValueError) as excinfo:
        load_settings(env=env, dotenv=None)
    assert str(excinfo.value).startswith(var), str(excinfo.value)


def test_parse_bool_accepts_the_documented_spellings() -> None:
    """``1/0/true/false/yes/no`` (SPEC section b), case-insensitive, plus on/off."""
    for raw in ("1", "true", "TRUE", "yes", "Y", "on", "t", True, 3):
        assert parse_bool(raw) is True
    for raw in ("0", "false", "No", "off", "f", False, 0):
        assert parse_bool(raw) is False
    with pytest.raises(ValueError) as excinfo:
        parse_bool("perhaps", "FLY_REALTIME")
    assert "FLY_REALTIME" in str(excinfo.value)


def test_derived_fields_and_gain_value() -> None:
    """``tick_ms``, ``steps_per_tick``, ``run_id`` and ``gain_value`` (SPEC section b / c.1)."""
    s = load_settings(env={"FLY_TICK_HZ": "25", "FLY_DT_MS": "0.5"}, dotenv=None)
    assert s.tick_ms == 40.0 and s.steps_per_tick == 80 and s.brain_ms_per_tick == 40.0
    assert s.tick_s == pytest.approx(0.04)
    assert gain_value(s) is None                                  # "auto"
    assert gain_value(load_settings(env={"FLY_GAIN": "0.65"}, dotenv=None)) == pytest.approx(0.65)

    same = load_settings(env={"FLY_TICK_HZ": "25", "FLY_DT_MS": "0.5"}, dotenv=None)
    assert same.run_id == s.run_id                                # deterministic
    other = load_settings(env={"FLY_SEED": "1338"}, dotenv=None)
    assert other.run_id != s.run_id
    assert load_settings(env={"FLY_N_NEURONS": "8000"}, dotenv=None).connectome_key != s.connectome_key


def test_redacted_never_exposes_secrets() -> None:
    """``redacted()`` is an ``asdict`` with string paths and no credential-shaped key (SPEC c.1)."""
    s = load_settings(env={"FLY_LLM": "anthropic", "FLY_X": "post"}, dotenv=None)
    out = redacted(s)
    assert set(out) == {f.name for f in __import__("dataclasses").fields(Settings)}
    assert isinstance(out["data_dir"], str) and isinstance(out["out_dir"], str)
    assert isinstance(out["connectome_dir"], str) and isinstance(out["cors_origins"], list)
    joined = json.dumps(out).upper()
    for secret in cfg.SECRET_ENV_VARS:
        assert secret not in joined
    assert not any(cfg._is_secret_field(k) for k in out)
    assert "connectome_key" in out            # a cache key is not a credential
    assert json.dumps(out)            # JSON serialisable for GET /api/config
    # Settings itself stores no secret at all
    assert not any(cfg._is_secret_field(f.name) for f in __import__("dataclasses").fields(Settings))
    masked = cfg.mask_secrets({"ANTHROPIC_API_KEY": "sk-ant-secret"})
    assert masked["ANTHROPIC_API_KEY"] == "<set>" and "secret" not in json.dumps(masked)


def test_env_registry_covers_section_b_and_env_example() -> None:
    """Every ``Settings`` field has an env var, and every ``.env.example`` key is accepted (SPEC i.4)."""
    import dataclasses

    derived = {"tick_ms", "steps_per_tick", "run_id", "connectome_key"}
    fields = {f.name for f in dataclasses.fields(Settings)} - derived
    covered = {v.field for v in cfg.ENV_VARS if v.field}
    assert fields == covered, f"uncovered fields: {sorted(fields - covered)}"
    assert all(v.name.isupper() for v in cfg.ENV_VARS)
    assert len({v.name for v in cfg.ENV_VARS}) == len(cfg.ENV_VARS)

    example = REPO / ".env.example"
    if not example.is_file():
        pytest.skip(".env.example not written yet (E6c owns it)")
    keys = set(parse_dotenv(example))
    unknown = sorted(keys - set(cfg.ENV_BY_NAME))
    assert not unknown, f".env.example keys unknown to Settings: {unknown}"
    documented = {v.name for v in cfg.ENV_VARS if v.field or v.kind == "secret"}
    assert documented <= keys | {"FLY_CONNECTOME_NAME"}, sorted(documented - keys)
    # the shipped example loads as-is (blank values fall back to defaults)
    loaded = load_settings(env={k: v for k, v in parse_dotenv(example).items()}, dotenv=None)
    assert loaded.connectome_source in cfg.CONNECTOME_SOURCES


# =====================================================================================================
# c.29: logging
# =====================================================================================================


def test_setup_logging_ring_and_ascii(capsys: pytest.CaptureFixture[str]) -> None:
    """``setup_logging`` is idempotent, ASCII-forcing, and the ring keeps the last lines (SPEC c.29)."""
    ring = flog.setup_logging("DEBUG")
    again = flog.setup_logging("INFO")
    assert again is ring and flog.get_ring() is ring
    root = logging.getLogger()
    assert sum(1 for h in root.handlers if isinstance(h, flog.RingHandler)) == 1
    assert root.level == logging.INFO

    ring.clear()
    log = logging.getLogger("flybrain.test")
    log.info("ascii only please")
    log.warning("fly moods: sevin\u00e7 ve \u00fcz\u00fcnt\u00fc")   # Turkish -> '?' in the ring
    tail = ring.tail(10)
    assert any("ascii only please" in line for line in tail)
    assert any("sevin?" in line for line in tail)
    assert all(line.isascii() for line in tail)
    assert "flybrain.test" in tail[-1] and "WARNING" in tail[-1]
    assert ring.tail(0) == [] and len(ring.tail(1)) == 1

    capacity = flog.RingHandler(capacity=3)
    capacity.setFormatter(logging.Formatter("%(message)s"))
    for i in range(10):
        capacity.emit(logging.LogRecord("n", logging.INFO, __file__, i, "line %d", (i,), None))
    assert capacity.tail(50) == ["line 7", "line 8", "line 9"] and len(capacity) == 3

    stream = flog.AsciiStream(__import__("io").StringIO())
    assert stream.write("caf\u00e9") == 4
    assert stream.stream.getvalue() == "caf?"


# =====================================================================================================
# c.25: StateBus, TickHistory, SessionLog, write_png
# =====================================================================================================


def _tick(seq: int, wall: float, events: list[dict] | None = None, t_ms: int | None = None) -> dict:
    return {"type": "tick", "seq": seq, "t_ms": seq * 50 if t_ms is None else t_ms, "wall": wall,
            "events": events or []}


def test_state_bus_publishes_json_and_drops_oldest() -> None:
    """Frames reach every subscriber as ready-to-send JSON text; a full queue loses its oldest."""
    bus = StateBus()
    q1 = bus.subscribe(maxsize=2)
    q2 = bus.subscribe(maxsize=2)
    assert bus.client_count() == 2

    bus.publish_tick(_tick(1, 100.0))
    frame = q1.get_nowait()
    assert isinstance(frame, str) and frame.kind == "tick" and frame.data["seq"] == 1
    assert json.loads(frame)["type"] == "tick"
    assert q2.get_nowait().data["seq"] == 1
    assert bus.latest_tick()["seq"] == 1

    for seq in range(2, 7):                       # 5 frames into a 2-slot queue
        bus.publish_tick(_tick(seq, 100.0 + seq))
    seqs = [q1.get_nowait().data["seq"] for _ in range(q1.qsize())]
    assert seqs == [5, 6]                         # newest kept, oldest dropped
    assert bus.stats()["dropped"] >= 3

    bus.publish_event(P.event_frame("clear", {"by": "client"}, seq=7, t_ms=350))
    ev = q2.get_nowait()
    while q2.qsize() and ev.kind != "event":
        ev = q2.get_nowait()
    assert ev.kind == "event" and ev.data["kind"] == "clear"

    bus.unsubscribe(q1)
    bus.unsubscribe(q1)                           # idempotent
    assert bus.client_count() == 1

    bus.set_hello({"type": "hello", "run_id": "abc"})
    assert bus.hello()["run_id"] == "abc"
    # a NaN tick is dropped with a log line instead of raising into the sim thread
    bad = _tick(8, 200.0)
    bad["sim"] = {"rtf": float("nan")}
    bus.publish_tick(bad)
    assert bus.latest_tick()["seq"] == 6


@pytest.mark.asyncio
async def test_state_bus_publish_from_thread_with_running_loop() -> None:
    """The sim thread publishes through ``loop.call_soon_threadsafe`` into asyncio queues (SPEC 0.1)."""
    import asyncio

    bus = StateBus(asyncio.get_running_loop())
    q = bus.subscribe()
    worker = threading.Thread(target=lambda: bus.publish_tick(_tick(1, 100.0)))
    worker.start()
    worker.join(2.0)
    frame = await asyncio.wait_for(q.get(), 2.0)
    assert frame.kind == "tick" and frame.data["seq"] == 1

    threading.Thread(target=lambda: bus.publish_event(
        P.event_frame("poke", {"stim": "sugar"}, seq=1, t_ms=50))).start()
    event = await asyncio.wait_for(q.get(), 2.0)
    assert event.kind == "event" and event.data["kind"] == "poke"
    P.EventMsg.model_validate(event.data)
    bus.unsubscribe(q)


def test_state_bus_client_to_sim_queues() -> None:
    """Pokes and commands cross into the sim thread through bounded drain-once queues."""
    bus = StateBus()
    poke = P.parse_client('{"type":"poke","stim":"sugar"}').to_poke(0)
    bus.push_poke(poke)
    bus.push_command({"name": "set_market_mode", "mode": "PUMP"})
    bus.push_command({"name": "clear"})
    assert bus.stats()["pending_pokes"] == 1
    assert [p.stim for p in bus.drain_pokes()] == ["sugar"]
    assert bus.drain_pokes() == []
    cmds = bus.drain_commands()
    assert [c["name"] for c in cmds] == ["set_market_mode", "clear"]
    assert bus.drain_commands() == []


def test_state_bus_snapshot_roundtrip() -> None:
    """``request_snapshot`` -> client frame -> ``deliver_snapshot`` unblocks ``wait_snapshot``."""
    bus = StateBus()
    q = bus.subscribe()
    bus.note_active_client(q)
    png = write_png(np.zeros((2, 2, 3), dtype=np.uint8))

    bus.request_snapshot("snap-1")
    frame = q.get_nowait()
    assert frame.kind == "snapshot_request" and frame.data["id"] == "snap-1"
    P.SnapshotRequestMsg.model_validate(frame.data)
    assert bus.pending_snapshots() == ["snap-1"]

    got: list[bytes | None] = []
    waiter = threading.Thread(target=lambda: got.append(bus.wait_snapshot("snap-1", 2.0)))
    waiter.start()
    time.sleep(0.05)
    bus.deliver_snapshot("snap-1", png)
    waiter.join(3.0)
    assert got == [png]

    assert bus.wait_snapshot("snap-2", 0.05) is None          # nobody answers -> server fallback
    bus.deliver_snapshot("snap-2", png)                       # late answer is ignored, no raise


def test_tick_history_windows_and_counters() -> None:
    """``last``/``events`` use the tick ring; ``gf_spikes``/``jumps`` answer 10-minute windows (c.25)."""
    history = TickHistory(seconds=10.0, tick_s=0.05)
    t0 = 1_000.0
    for i in range(400):                                       # 20 s of ticks at 20 Hz
        wall = t0 + i * 0.05
        events: list[dict] = []
        if i % 100 == 0:
            events.append({"kind": "gf_spike", "t_ms": i * 50, "data": {"side": "R", "count": 2}})
        if i == 390:
            events.append({"kind": "jump", "t_ms": i * 50, "data": {"side": "R"}})
        if i == 395:
            events.append({"kind": "mood", "t_ms": i * 50, "data": {"from": "CRUISING", "to": "PANIC"}})
        history.push(_tick(i + 1, wall, events, t_ms=i * 50))

    assert len(history) <= 202                                 # 10 s ring
    recent = history.last(1.0)
    assert 18 <= len(recent) <= 22 and recent[-1]["seq"] == 400
    assert history.latest()["seq"] == 400
    assert history.jumps(60.0) == 1 and history.jumps(0.1) == 0
    assert history.gf_spikes(600.0) == 8                       # 4 events x count 2, all within 10 min
    assert history.gf_spikes(1.0) == 0
    kinds = [e["kind"] for e in history.events(10.0)]
    assert "jump" in kinds and "mood" in kinds
    assert history.mood_history()[-1] == (395 * 50, "PANIC")
    assert history.events(0.0) == []


def test_session_log_header_and_tick_lines(tmp_path: Path) -> None:
    """``SessionLog`` writes a header plus one replayable line per tick; ``read_session`` reads it back."""
    settings = load_settings(env={"FLY_DATA_DIR": str(tmp_path / "data"),
                                 "FLY_OUT_DIR": str(tmp_path / "out")}, dotenv=None)
    path = settings.sessions_dir / f"{settings.run_id}.jsonl"

    from flybrain.encoder import Poke
    from flybrain.market.base import MarketSnapshot, Trade

    snap = MarketSnapshot(ts=1.0, source="sim", chain="sim", dex="sim", pair="SIM", symbol="FLY",
                          price_usd=0.001, price_native=0.001, buys_m5=1, sells_m5=2, buys_h1=3,
                          sells_h1=4, chg_m5=0.1, chg_h1=0.2, chg_h6=0.3, chg_h24=0.4, vol_m5=10.0,
                          vol_h1=20.0, liq_usd=None, fdv=None, mcap=None, regime="PUMP")
    trades = [Trade(ts=1.0, kind="buy", usd=12.5, price=0.001),
              Trade(ts=1.1, kind="sell", usd=3.0, price=0.001, surrogate=True)]
    pokes = [Poke(stim="sugar", strength=0.5, side=-1, until_ms=600)]

    with SessionLog(path, settings) as slog:
        assert slog.ok and path.is_file()
        slog.write_tick(1, 50, 1789051563.1234, snap, trades, pokes,
                        [{"name": "set_market_mode", "mode": "PUMP"}], 812)
        slog.write_tick(2, 100, 1789051563.2, None, [], [], [], 0)
    rows = list(read_session(path))
    assert len(rows) == 3
    header, first, second = rows
    assert header["kind"] == "header" and header["run_id"] == settings.run_id
    assert header["connectome_key"] == settings.connectome_key and header["created"]
    assert header["settings"]["n_neurons"] == settings.n_neurons
    for secret in cfg.SECRET_ENV_VARS:
        assert secret not in json.dumps(header).upper()
    assert first["kind"] == "tick" and first["seq"] == 1 and first["total_spikes"] == 812
    assert first["market"]["regime"] == "PUMP" and first["market"]["symbol"] == "FLY"
    assert [t["kind"] for t in first["trades"]] == ["buy", "sell"]
    assert first["trades"][1]["surrogate"] is True and first["trades"][0]["price"] == 0.001
    assert first["pokes"] == [{"stim": "sugar", "strength": 0.5, "side": -1, "until_ms": 600}]
    assert Poke(**first["pokes"][0]).until_ms == 600            # replay can rebuild the poke
    assert first["commands"][0]["name"] == "set_market_mode"
    assert second["market"] is None and second["trades"] == []

    # a corrupt tail line is skipped, not fatal
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind":"tick","seq":3,\n')
    assert len(list(read_session(path))) == 3

    broken = SessionLog(tmp_path / "nope" / "dir" / "x.jsonl", settings)
    broken.write_tick(1, 1, 1.0, None, [], [], [], 0)           # never raises
    broken.close()


def test_write_png_roundtrip() -> None:
    """``write_png`` emits a decodable 8-bit RGB PNG with filter-0 scanlines (SPEC c.25)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    blob = write_png(img)
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"

    # walk the chunks: IHDR, IDAT, IEND with valid CRCs
    pos, chunks, idat = 8, [], b""
    while pos < len(blob):
        (length,) = struct.unpack(">I", blob[pos:pos + 4])
        tag = blob[pos + 4:pos + 8]
        data = blob[pos + 8:pos + 8 + length]
        (crc,) = struct.unpack(">I", blob[pos + 8 + length:pos + 12 + length])
        assert crc == zlib.crc32(tag + data) & 0xFFFFFFFF
        chunks.append(tag)
        if tag == b"IDAT":
            idat += data
        pos += 12 + length
    assert chunks == [b"IHDR", b"IDAT", b"IEND"]

    w, h, depth, colour, comp, filt, interlace = struct.unpack(">IIBBBBB", blob[16:16 + 13])
    assert (w, h, depth, colour, comp, filt, interlace) == (7, 5, 8, 2, 0, 0, 0)
    raw = zlib.decompress(idat)
    stride = 1 + w * 3
    assert len(raw) == h * stride
    for y in range(h):
        assert raw[y * stride] == 0                              # filter type 0 on every scanline
        row = np.frombuffer(raw[y * stride + 1:(y + 1) * stride], dtype=np.uint8)
        assert np.array_equal(row, img[y].reshape(-1))
    assert write_png(img) == blob                                # deterministic

    grey = write_png(np.full((3, 4), 128, dtype=np.uint8))       # convenience shapes
    assert grey[:8] == b"\x89PNG\r\n\x1a\n"
    assert write_png(np.zeros((2, 2, 4), dtype=np.uint8))[:8] == b"\x89PNG\r\n\x1a\n"
    for bad in (np.zeros((2, 2, 2), dtype=np.uint8), np.zeros((0, 4, 3), dtype=np.uint8),
                np.zeros((4,), dtype=np.uint8)):
        with pytest.raises(ValueError):
            write_png(bad)
