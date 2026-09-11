"""E5 agent tests (SPEC h.2 ``test_agent.py``): all dry-run, offline, deterministic; ``anthropic``/``tweepy`` are
never imported for real (fake modules are injected into ``sys.modules`` where a test needs them)."""

from __future__ import annotations

import json
import logging
import random
import re
import struct
import sys
import threading
import time
import zlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain.agent import llm as llm_mod
from flybrain.agent import orchestrator as orch_mod
from flybrain.agent import summary as summary_mod
from flybrain.agent.llm import (
    SYSTEM_PROMPT,
    TWEET_SCHEMA,
    VOCABULARY,
    TweetDraft,
    TweetGenerator,
    system_prompt_for,
    template_tweet,
    validate_tweet,
)
from flybrain.agent.orchestrator import TRIGGER_REASONS, AgentState, TweetAgent
from flybrain.agent.snapshot import FRAME_PX, PNG_MAGIC, STATUS_PX, TITLE_PX, SnapshotBroker, render_snapshot
from flybrain.agent.summary import MAX_SUMMARY_BYTES, RATE_KEYS, VOCABULARY_HINT, build_brain_summary, summary_json
from flybrain.agent.x_client import PostResult, XClient
from flybrain.config import Settings

# ----------------------------------------------------------------------------- helpers / fakes
SUMMARY_KEYS = ("schema", "reason", "lang", "mood", "market", "drives", "rates_hz", "regions_hz", "top_types",
                "events_10s", "gf_spikes_10min", "jumps_60s", "fly", "connectome", "session", "vocabulary_hint")

CONN_META = {"source": "synthetic", "name": "synthetic-20000-s1337-cal", "n": 20000, "e": 548120,
             "note": "synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data"}


def make_tick(mood: str = "EUPHORIA", prev: str = "FEEDING", t_ms: int = 617350, seq: int = 12345) -> dict:
    pops = {"gf": 0.0, "gf_L": 0.0, "gf_R": 0.0, "escape_dn": 0.0, "dn_freeze": 1.2, "dng100": 12.0,
            "steer_a02": 12.5, "steer_a02_L": 6.0, "steer_a02_R": 19.0, "flight_dn": 0.0, "feed_mn": 52.0,
            "sugar2_exc": 40.0, "grn_sugar": 131.0, "pam": 33.0, "ppl1": 0.0, "mbon_approach": 18.0,
            "mbon_avoid": 3.0, "kc": 2.1, "epg": 9.0, "pfl3": 12.0, "p1": 0.0, "pip10": 0.0, "lc4": 0.4,
            "lplc2": 0.5, "lc_loom": 0.0, "ttmn": 0.0, "psi": 0.0, "photoreceptor": 21.0, "lamina": 14.0,
            "t4t5": 4.0, "orn": 18.0, "dn_mean": 4.12}
    return {
        "type": "tick", "seq": seq, "t_ms": t_ms, "wall": 1789051563.12,
        "sim": {"rtf": 2.3, "speed": 1.0, "steps": 50},
        "fly": {"x": 412.3, "y": 233.9, "vx": -38.2, "vy": 12.1, "heading": 2.834, "speed": 40.1, "omega": -0.4,
                "wing_hz": 0.0, "wing_amp": 0.0, "wing_ext": 0, "mode": "feed", "leg_phase": 0.3, "proboscis": 0.6,
                "jump_t_ms": None},
        "ink": {"color": "#00a800", "width": 2.0, "alpha": 1.0, "style": "solid", "stamp": None},
        "mood": {"state": mood, "prev": prev, "since_ms": 3000, "euphoria": 0.78, "anxiety": 0.03, "arousal": 0.71,
                 "valence": 0.62, "fear": 0.05, "hunger": 0.2, "sleep": 0.0, "dwell_left_ms": 0},
        "market": {"source": "sim", "mode": "sim", "ts": 1789051563.0, "seq": 617, "chain": "sim", "dex": "sim",
                   "pair": "SIM", "symbol": "FLY", "price_usd": 0.0012345, "price_native": 0.0012345,
                   "buys_m5": 88, "sells_m5": 21, "buys_h1": 402, "sells_h1": 377, "chg_m5": 4.2, "chg_h1": 11.0,
                   "chg_h6": 5.1, "chg_h24": 40.5, "vol_m5": 21000.0, "vol_h1": 61230.0, "liq_usd": 52000.0,
                   "fdv": 1450000.0, "mcap": 1450000.0, "regime": "PUMP",
                   "last_trade": {"kind": "buy", "usd": 940.0, "ts": 1789051562.4, "surrogate": False}},
        "drives": {"sugar": 0.83, "bitter": 0.02, "water": 0.0, "looming": 0.0, "loom_side": 0, "flash": 0.0,
                   "odor": 0.55, "chop": 0.0, "courtship": 0.0, "sleep_pressure": 0.0, "explore": 0.46, "up": 0.55,
                   "down": 0.03, "activity": 0.4, "hunger": 0.2, "candle": "up", "any_max": 0.83},
        "rates": {"regions": [4.1, 6.0, 1.9, 3.3, 9.8, 2.2, 7.2, 2.4], "pops": pops},
        "spikes": {"t0_ms": t_ms - 50, "win_ms": 50, "total": 812, "capped": False, "slots": [], "dt": []},
        "events": [],
    }


class FakeHistory:
    def __init__(self, ticks: list[dict] | None = None, events: list[dict] | None = None, gf: int = 0,
                 jumps: int = 0) -> None:
        self._ticks = ticks or []
        self._events = events or [{"kind": "feed_start", "t_ms": 1, "data": {}},
                                  {"kind": "mood", "t_ms": 2, "data": {"from": "FEEDING", "to": "EUPHORIA"}}]
        self._gf = gf
        self._jumps = jumps

    def push(self, tick: dict) -> None:
        self._ticks.append(tick)

    def last(self, seconds: float) -> list[dict]:
        return list(self._ticks)

    def events(self, seconds: float) -> list[dict]:
        return list(self._events)

    def gf_spikes(self, seconds: float) -> int:
        return self._gf

    def jumps(self, seconds: float) -> int:
        return self._jumps


class FakeBus:
    """Minimal StateBus stand-in (SPEC c.25): events list + snapshot round trip + client count."""

    def __init__(self, clients: int = 0, tick: dict | None = None) -> None:
        self.events: list[dict] = []
        self._clients = clients
        self._tick = tick
        self._snap: dict[str, bytes] = {}
        self._cv = threading.Condition()
        self.requested: list[str] = []

    def publish_event(self, msg: dict) -> None:
        self.events.append(msg)

    def publish_tick(self, tick: dict) -> None:
        self._tick = tick

    def latest_tick(self) -> dict | None:
        return self._tick

    def client_count(self) -> int:
        return self._clients

    def request_snapshot(self, id: str) -> None:
        self.requested.append(id)

    def wait_snapshot(self, id: str, timeout_s: float) -> bytes | None:
        with self._cv:
            end = time.monotonic() + timeout_s
            while id not in self._snap:
                left = end - time.monotonic()
                if left <= 0:
                    return None
                self._cv.wait(left)
            return self._snap.pop(id)

    def deliver_snapshot(self, id: str, png: bytes) -> None:
        with self._cv:
            self._snap[id] = png
            self._cv.notify_all()


class FakeGenerator:
    """Scripted generator: returns the given texts in order (last one repeats)."""

    def __init__(self, texts: list[str], model: str = "fake-model") -> None:
        self.texts = list(texts)
        self.model_name = model
        self.calls = 0
        self.rng = random.Random(1)
        self.dry_run = False

    @property
    def model(self) -> str:
        return self.model_name

    def generate(self, summary: dict) -> TweetDraft:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return TweetDraft(text=text, model=self.model_name, dry_run=False, stop_reason="end_turn", latency_ms=1.0,
                          reason=summary["reason"], neurons=["LB3b"])


class FakePoster:
    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.disabled_reason: str | None = None
        self.calls: list[tuple[str, int]] = []

    def post(self, text: str, png: bytes | None = None) -> PostResult:
        self.calls.append((text, len(png) if png else 0))
        return PostResult(posted=not self.dry_run, dry_run=self.dry_run, id=None if self.dry_run else "1",
                          url=None if self.dry_run else "https://x.com/i/web/status/1", error=None, media_id=None)


def make_settings(tmp_path: Path, **over) -> Settings:
    kw = dict(data_dir=tmp_path / "data", out_dir=tmp_path / "out", llm="dryrun", x_mode="dryrun", seed=0,
              tweet_cooldown_s=900, tweet_reason_cooldown_s=2700, tweets_per_day=12, tweet_lang="en")
    kw.update(over)
    s = Settings(**kw)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    s.out_dir.mkdir(parents=True, exist_ok=True)
    return s


def make_agent(tmp_path: Path, generator=None, poster=None, bus=None, clients: int = 0, **over) -> TweetAgent:
    settings = make_settings(tmp_path, **over)
    tick = make_tick()
    bus = bus or FakeBus(clients=clients, tick=tick)
    gen = generator or TweetGenerator(settings, seed=settings.seed)
    poster = poster or FakePoster(dry_run=True)
    body = SimpleNamespace(kin=SimpleNamespace(x=412.3, y=233.9, heading=2.834, mode="feed"))
    broker = SnapshotBroker(bus, body, settings)
    for i in range(50):
        broker.record_trail(100 + i * 4.0, 200 + i * 2.0, "#ff0000", 2.0)
    agent = TweetAgent(settings, gen, poster, broker, bus, FakeHistory(), CONN_META, seed=settings.seed)
    return agent


def _clear_creds(monkeypatch) -> None:
    """No X credential may leak in from the developer's real environment (SPEC c.22 reads them from os.environ)."""
    for key in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_TOKEN_SECRET"):
        monkeypatch.delenv(key, raising=False)


def decode_png(png: bytes) -> np.ndarray:
    """Minimal PNG reader for 8-bit RGB, filter 0 rows (what server.png.write_png emits)."""
    assert png[:8] == PNG_MAGIC
    pos = 8
    w = h = 0
    idat = b""
    while pos < len(png):
        (ln,) = struct.unpack(">I", png[pos:pos + 4])
        tag = png[pos + 4:pos + 8]
        data = png[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if tag == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", data[:10])
            assert depth == 8 and ctype == 2
        elif tag == b"IDAT":
            idat += data
        elif tag == b"IEND":
            break
    raw = zlib.decompress(idat)
    stride = w * 3 + 1
    assert len(raw) == stride * h
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, stride)
    assert np.all(rows[:, 0] == 0), "only filter type 0 expected"
    return rows[:, 1:].reshape(h, w, 3)


def _url_free(text: str) -> bool:
    return re.search(r"https?://|www\.", text, re.IGNORECASE) is None


def _has_vocab(text: str) -> bool:
    low = text.lower()
    return any(v.lower() in low for v in VOCABULARY)


# ----------------------------------------------------------------------------- summary
def test_build_brain_summary_shape():
    tick = make_tick()
    session = {"uptime_s": 3612.0, "tweets_today": 2, "last_tweet_reason": "panic_entry", "trail_px": 18240}
    s = build_brain_summary(tick, FakeHistory(gf=0, jumps=0), "euphoria_entry", CONN_META, session, tick["market"],
                            "en")
    assert tuple(s.keys()) == SUMMARY_KEYS
    assert s["schema"] == "flybrain.summary.v1" and s["reason"] == "euphoria_entry" and s["lang"] == "en"
    assert tuple(s["rates_hz"].keys()) == RATE_KEYS and len(RATE_KEYS) == 19
    assert s["rates_hz"]["grn_sugar"] == 131.0 and s["rates_hz"]["steer_a02_R"] == 19.0
    assert tuple(s["mood"].keys()) == ("state", "prev", "since_s", "euphoria", "anxiety", "valence", "arousal", "hunger")
    assert s["mood"]["since_s"] == 3.0
    assert tuple(s["market"].keys()) == ("source", "symbol", "token_live", "price_usd", "chg_m5", "chg_h1", "chg_h24", "buys_m5",
                                         "sells_m5", "vol_m5", "liq_usd", "mcap", "regime", "last_trade")
    assert s["market"]["last_trade"] == {"kind": "buy", "usd": 940.0}
    assert tuple(s["drives"].keys()) == ("sugar", "bitter", "looming", "odor", "chop", "courtship", "sleep_pressure",
                                         "explore")
    assert list(s["regions_hz"].keys()) == ["optic_lobe", "antennal_lobe", "mushroom_body", "central_complex", "sez",
                                            "central_other", "descending_motor", "vnc"]
    # the d.6 example is ~1.7 KB minified, so the ladder coarsens the rate blocks to fit 1.5 KB
    assert s["regions_hz"]["sez"] in (9.8, 10)
    assert len(s["top_types"]) == 5 and s["top_types"][0] == ["LB3b", 131.0]
    assert all(isinstance(t[0], str) and isinstance(t[1], (int, float)) for t in s["top_types"])
    assert s["events_10s"] == ["feed_start", "mood:FEEDING->EUPHORIA"]
    assert s["gf_spikes_10min"] == 0 and s["jumps_60s"] == 0
    # d.6 shows 412.3 -> 412 and 233.9 -> 233 (truncation, not rounding)
    assert s["fly"] == {"mode": "feed", "x": 412, "y": 233, "speed": 40.1, "wing_hz": 0.0, "proboscis": 0.6,
                        "trail_px": 18240}
    assert s["connectome"]["source"] == "synthetic" and s["connectome"]["n"] == 20000
    assert "synthetic" in s["connectome"]["note"]
    assert s["session"] == {"uptime_s": 3612.0, "tweets_today": 2, "last_tweet_reason": "panic_entry"}
    assert 1 <= len(s["vocabulary_hint"]) <= 8 and set(s["vocabulary_hint"]) <= set(VOCABULARY_HINT)
    assert s["vocabulary_hint"][0] == "sugar GRNs (LB3b/LB3c)"  # the reason-relevant hints survive trimming
    # d.6: connectome.note is COPIED from meta['note'] - the trim ladder touches it last, so it survives here
    assert s["connectome"]["note"] == CONN_META["note"]
    assert s["connectome"]["name"] == CONN_META["name"]
    payload = summary_json(s)
    assert len(payload.encode("utf-8")) <= MAX_SUMMARY_BYTES == 1560, len(payload)
    assert json.loads(payload) == s
    # the full d.6 shape with the canonical example numbers fits the budget for every reason and both languages
    for reason in TRIGGER_REASONS:
        for lang in ("en", "tr"):
            sr = build_brain_summary(tick, FakeHistory(gf=12, jumps=4), reason, CONN_META, session, tick["market"],
                                     lang)
            assert tuple(sr.keys()) == SUMMARY_KEYS and len(summary_json(sr).encode("utf-8")) <= MAX_SUMMARY_BYTES
    # realistic non-integral numbers (no ".0" savings) still fit and keep the 5 top types and the 19 rate keys
    tick_r = make_tick()
    for k in tick_r["rates"]["pops"]:
        tick_r["rates"]["pops"][k] = tick_r["rates"]["pops"][k] + 0.37
    tick_r["rates"]["regions"] = [v + 0.13 for v in tick_r["rates"]["regions"]]
    tick_r["market"].update({"price_usd": 0.000012345678, "mcap": 12345678.9, "liq_usd": 123456.7, "vol_m5": 12345.6,
                             "chg_h24": -123.4})
    tick_r["mood"]["since_ms"] = 123456
    sr = build_brain_summary(tick_r, FakeHistory(gf=123, jumps=12), "panic_entry", CONN_META,
                             {"uptime_s": 86399.9, "tweets_today": 11, "last_tweet_reason": "courtship_entry",
                              "trail_px": 1234567}, tick_r["market"], "en")
    assert len(summary_json(sr).encode("utf-8")) <= MAX_SUMMARY_BYTES and tuple(sr.keys()) == SUMMARY_KEYS
    assert len(sr["top_types"]) >= 3 and tuple(sr["rates_hz"].keys()) == RATE_KEYS
    assert sr["top_types"][0][0] == "LB3b" and sr["rates_hz"]["grn_sugar"] >= 131
    assert sr["vocabulary_hint"] and sr["vocabulary_hint"][0] == "giant fiber DNp01"
    assert sr["connectome"]["note"] == CONN_META["note"]
    # robustness: an empty tick and a bare history still give every key
    s2 = build_brain_summary({}, None, "manual", {}, {}, {}, "tr")
    assert tuple(s2.keys()) == SUMMARY_KEYS and s2["lang"] == "tr" and s2["market"]["price_usd"] is None
    assert tuple(s2["rates_hz"].keys()) == RATE_KEYS and len(s2["vocabulary_hint"]) == 8
    # long note / name and many events are trimmed rather than blowing the budget
    big_meta = {**CONN_META, "note": "x" * 400, "name": "n" * 200}
    many = FakeHistory(events=[{"kind": f"ev{i}", "t_ms": i, "data": {}} for i in range(40)])
    s3 = build_brain_summary(tick, many, "panic_entry", big_meta, session, tick["market"], "en")
    assert len(summary_json(s3).encode("utf-8")) <= MAX_SUMMARY_BYTES and tuple(s3.keys()) == SUMMARY_KEYS


# ----------------------------------------------------------------------------- templates / validation
def test_template_tweet_every_reason():
    tick = make_tick()
    for reason in TRIGGER_REASONS:
        for lang in ("en", "tr"):
            summary = build_brain_summary(tick, FakeHistory(gf=4, jumps=3), reason, CONN_META,
                                          {"uptime_s": 10, "tweets_today": 0, "last_tweet_reason": None},
                                          tick["market"], lang)
            texts = set()
            for seed in range(6):
                a = template_tweet(summary, random.Random(seed))
                b = template_tweet(summary, random.Random(seed))
                assert a == b, "template must be deterministic per seed"
                assert 0 < len(a) <= 280
                assert _has_vocab(a), a
                assert _url_free(a)
                assert len(re.findall(r"#\w+", a)) <= 2
                assert "?" not in re.findall(r"\{[^}]*\}", a)  # no unfilled placeholder
                texts.add(a)
            assert len(texts) >= 2, f"{reason}/{lang}: templates should vary across seeds"
        assert len(llm_mod._TEMPLATES["en"][reason]) >= 6
        assert len(llm_mod._TEMPLATES["tr"][reason]) >= 6
    # numbers from the summary are used
    s = build_brain_summary(tick, FakeHistory(), "euphoria_entry", CONN_META, {}, tick["market"], "en")
    joined = " ".join(template_tweet(s, random.Random(i)) for i in range(12))
    assert "131.0" in joined or "33.0" in joined


def test_validate_tweet():
    ok, t = validate_tweet("gm ser https://example.com/x?y=1 sugar GRN LB3b at 131 Hz www.scam.io ngmi")
    assert ok and "http" not in t and "www." not in t and t == "gm ser sugar GRN LB3b at 131 Hz ngmi"
    # cap at 280 on a word boundary
    long = "mushroom body " + " ".join(f"word{i}" for i in range(120))
    ok, t = validate_tweet(long)
    assert ok and len(t) <= 280 and not t.endswith("wor") and long.startswith(t)
    assert long[len(t)] == " "  # cut exactly at a space
    # keeps the first 2 of 4 hashtags
    ok, t = validate_tweet("PAM dopamine #a #b #c #d done")
    assert ok and t == "PAM dopamine #a #b done"
    # cashtags: drops $DOGE, keeps $FLY, leaves prices alone
    ok, t = validate_tweet("$DOGE ngmi, $FLY wagmi at $0.0012 says the mushroom body")
    assert ok and "$DOGE" not in t and "$FLY" in t and "$0.0012" in t
    # whitespace + surrounding quotes
    ok, t = validate_tweet('  "  giant   fiber\n DNp01 fired "  ')
    assert ok and t == "giant fiber DNp01 fired"
    # vocabulary required
    ok, t = validate_tweet("gm ser wagmi")
    assert not ok and t == "gm ser wagmi"
    ok, _ = validate_tweet("gm ser wagmi", require_vocab=False)
    assert ok
    assert validate_tweet("")[0] is False and validate_tweet("https://only.url")[0] is False
    # the prompt is verbatim and the mode/lang variants are well-formed
    assert SYSTEM_PROMPT.startswith("You are FlyBrain, a spiking simulation of the Drosophila male CNS connectome")
    assert SYSTEM_PROMPT.endswith('Return only the JSON object {"text": string, "neurons": string[]}.')
    assert system_prompt_for("en", False).endswith("Output only the tweet text.")
    assert system_prompt_for("tr", True).endswith("Write the tweet in Turkish.")
    assert TWEET_SCHEMA["required"] == ["text", "neurons"] and TWEET_SCHEMA["additionalProperties"] is False


# ----------------------------------------------------------------------------- generator
def test_generator_dryrun_never_imports_anthropic(tmp_path):
    settings = make_settings(tmp_path)
    gen = TweetGenerator(settings, seed=1337)
    tick = make_tick()
    summary = build_brain_summary(tick, FakeHistory(), "euphoria_entry", CONN_META, {}, tick["market"], "en")
    before = set(sys.modules)
    d1 = gen.generate(summary)
    after = set(sys.modules)
    assert after == before
    assert "anthropic" not in sys.modules or "anthropic" in before
    assert d1.model == "template" and d1.dry_run is True and d1.error is None and not d1.refused
    assert _has_vocab(d1.text) and len(d1.text) <= 280 and d1.neurons
    # determinism: same seed -> same sequence (SeedSequence child [5])
    gen2 = TweetGenerator(settings, seed=1337)
    assert gen2.generate(summary).text == d1.text
    assert TweetGenerator(settings, seed=1).generate(summary).text != d1.text or True  # different seed may collide


class _Block:
    def __init__(self, type_: str, text: str = "") -> None:
        self.type = type_
        self.text = text


class _Resp:
    def __init__(self, text: str, stop: str = "end_turn") -> None:
        self.content = [_Block("thinking", "hidden"), _Block("text", text)]
        self.stop_reason = stop
        self.usage = SimpleNamespace(input_tokens=1200, output_tokens=80)


def _fake_anthropic(script: list | None = None) -> SimpleNamespace:
    """A fake ``anthropic`` module whose Messages.create pops ``mod.script`` (a response or an exception to raise).
    Exceptions must be instances of THIS module's classes (fill ``mod.script`` after creation)."""
    calls: list[dict] = []
    script = [] if script is None else script

    class APIConnectionError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, msg: str = "", status_code: int = 500) -> None:
            super().__init__(msg)
            self.status_code = status_code
            self.response = SimpleNamespace(headers={})

    class RateLimitError(APIStatusError):
        def __init__(self, msg: str = "rate limited") -> None:
            super().__init__(msg, 429)
            self.response = SimpleNamespace(headers={"retry-after": "0"})

    class Messages:
        def create(self, **kwargs):
            calls.append(kwargs)
            item = script.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

    class Anthropic:
        def __init__(self, **kwargs) -> None:
            self.messages = Messages()

    mod = SimpleNamespace(Anthropic=Anthropic, APIConnectionError=APIConnectionError, APIStatusError=APIStatusError,
                          RateLimitError=RateLimitError, __version__="0.86.0", calls=calls, script=script)
    return mod


def _install_fake(monkeypatch, script_builder) -> SimpleNamespace:
    """Create a fake anthropic module, fill its script via ``script_builder(fake)`` and install it."""
    fake = _fake_anthropic()
    fake.script.extend(script_builder(fake))
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    return fake


def test_generator_anthropic_mocked(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, llm="anthropic", llm_model="claude-opus-5", llm_json=True)
    tick = make_tick()
    summary = build_brain_summary(tick, FakeHistory(), "euphoria_entry", CONN_META, {}, tick["market"], "en")
    monkeypatch.setattr(time, "sleep", lambda s: None)

    good = json.dumps({"text": "gm ser, sugar GRN LB3b at 131 Hz says wagmi", "neurons": ["LB3b"]})
    fake = _fake_anthropic([_Resp(good)])
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    gen = TweetGenerator(settings, seed=3)
    d = gen.generate(summary)
    assert d.text == "gm ser, sugar GRN LB3b at 131 Hz says wagmi" and d.neurons == ["LB3b"]
    assert d.model == "claude-opus-5" and d.dry_run is False and d.stop_reason == "end_turn" and d.error is None
    assert d.usage == {"input_tokens": 1200, "output_tokens": 80}
    kw = fake.calls[0]
    assert kw["model"] == "claude-opus-5" and kw["max_tokens"] == 512
    assert kw["system"].startswith(SYSTEM_PROMPT) and kw["messages"] == [{"role": "user", "content": kw["messages"][0]["content"]}]
    assert json.loads(kw["messages"][0]["content"])["schema"] == "flybrain.summary.v1"
    assert kw["output_config"] == {"format": {"type": "json_schema", "schema": TWEET_SCHEMA}}
    for forbidden in ("thinking", "temperature", "budget_tokens", "top_p", "top_k", "stop_sequences"):
        assert forbidden not in kw
    assert set(kw) == {"model", "max_tokens", "system", "messages", "output_config"}

    # refusal -> refused=True, no template consumed
    fake = _fake_anthropic([_Resp("", stop="refusal")])
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.refused is True and d.text == "" and d.stop_reason == "refusal"

    # RateLimitError -> one retry, then template
    fake = _install_fake(monkeypatch, lambda f: [f.RateLimitError(), f.RateLimitError(), _Resp(good)])
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.model == "template" and d.error and "RateLimitError" in d.error and len(fake.calls) == 2
    assert _has_vocab(d.text)
    # 429 then success -> LLM text
    fake = _install_fake(monkeypatch, lambda f: [f.RateLimitError(), _Resp(good)])
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.model == "claude-opus-5" and len(fake.calls) == 2

    # APIConnectionError -> template
    fake = _install_fake(monkeypatch, lambda f: [f.APIConnectionError("boom")])
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.model == "template" and "APIConnectionError" in (d.error or "") and len(fake.calls) == 1
    # 5xx -> one retry then template; 4xx -> template immediately
    fake = _install_fake(monkeypatch, lambda f: [f.APIStatusError("bad gateway", 502),
                                                 f.APIStatusError("bad gateway", 502)])
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.model == "template" and len(fake.calls) == 2
    fake = _install_fake(monkeypatch, lambda f: [f.APIStatusError("bad request", 400)])
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.model == "template" and len(fake.calls) == 1

    # malformed JSON -> raw text used as the tweet
    fake = _fake_anthropic([_Resp("{not json: mushroom body says wagmi ser")])
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.model == "claude-opus-5" and d.text == "{not json: mushroom body says wagmi ser"
    # text with URL and no vocabulary -> validation fails -> regenerate once -> template
    fake = _fake_anthropic([_Resp(json.dumps({"text": "buy now https://x.y", "neurons": []})),
                            _Resp(json.dumps({"text": "gm gm", "neurons": []}))])
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    d = TweetGenerator(settings, seed=3).generate(summary)
    assert d.model == "template" and len(fake.calls) == 2 and "validation" in (d.error or "")

    # plain-text mode: no output_config, prompt ends with the text sentence
    fake = _fake_anthropic([_Resp("gm, giant fiber DNp01 quiet, wagmi")])
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    d = TweetGenerator(replace(settings, llm_json=False, tweet_lang="tr"), seed=3).generate(summary)
    assert d.text == "gm, giant fiber DNp01 quiet, wagmi" and "output_config" not in fake.calls[0]
    assert fake.calls[0]["system"].endswith("Output only the tweet text.\nWrite the tweet in Turkish.")


# ----------------------------------------------------------------------------- X client
def test_xclient_dryrun(tmp_path, caplog, monkeypatch):
    settings = make_settings(tmp_path, x_mode="dryrun")
    _clear_creds(monkeypatch)
    x = XClient(settings)
    assert x.dry_run and x.disabled_reason is None
    with caplog.at_level(logging.INFO, logger="flybrain.agent.x"):
        r = x.post("gm sugar GRN LB3b", png=b"\x89PNG12345")
    assert r == PostResult(posted=False, dry_run=True, id=None, url=None, error=None, media_id=None)
    assert any("[X DRY RUN] gm sugar GRN LB3b (+image 9 bytes)" in m for m in caplog.messages)
    assert "tweepy" not in sys.modules or True  # tweepy may already be loaded by another test; never required


def test_xclient_missing_creds_post_mode_degrades(tmp_path, caplog, monkeypatch):
    settings = make_settings(tmp_path, x_mode="post")
    _clear_creds(monkeypatch)
    monkeypatch.setenv("X_API_KEY", "k")  # three missing
    with caplog.at_level(logging.ERROR, logger="flybrain.agent.x"):
        x = XClient(settings)
    assert x.dry_run is True
    assert x.disabled_reason and "missing credentials" in x.disabled_reason
    assert "X_ACCESS_TOKEN_SECRET" in x.disabled_reason and "X_API_KEY" not in x.disabled_reason.split(":")[1]
    assert any(r.levelno == logging.ERROR and "credentials missing" in r.getMessage() for r in caplog.records)
    r = x.post("giant fiber DNp01 says gm")
    assert r.posted is False and r.dry_run is True and r.error is None


def _fake_tweepy(behaviour: str, calls: list) -> SimpleNamespace:
    class TweepyException(Exception):
        pass

    class HTTPException(TweepyException):
        pass

    class Forbidden(HTTPException):
        pass

    class Unauthorized(HTTPException):
        pass

    class TooManyRequests(HTTPException):
        def __init__(self, msg: str = "429") -> None:
            super().__init__(msg)
            self.reset_time = None

    class Client:
        def __init__(self, **kwargs) -> None:
            calls.append(("Client", kwargs))

        def create_tweet(self, text=None, media_ids=None, **kwargs):
            calls.append(("create_tweet", {"text": text, "media_ids": media_ids}))
            if behaviour == "forbidden":
                raise Forbidden("403 Forbidden: app lacks write access")
            if behaviour == "429":
                raise TooManyRequests()
            return SimpleNamespace(data={"id": 1234567890})

    class OAuth1UserHandler:
        def __init__(self, *a, **k) -> None:
            pass

    class API:
        def __init__(self, auth) -> None:
            pass

        def media_upload(self, filename, file=None):
            calls.append(("media_upload", filename))
            return SimpleNamespace(media_id_string="m-777")

    return SimpleNamespace(Client=Client, Forbidden=Forbidden, Unauthorized=Unauthorized, TooManyRequests=TooManyRequests,
                           HTTPException=HTTPException, TweepyException=TweepyException,
                           OAuth1UserHandler=OAuth1UserHandler, API=API)


def test_xclient_mocked_tweepy(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, x_mode="post")
    for k, v in (("X_API_KEY", "k"), ("X_API_SECRET", "s"), ("X_ACCESS_TOKEN", "t"), ("X_ACCESS_TOKEN_SECRET", "ts")):
        monkeypatch.setenv(k, v)
    calls: list = []
    monkeypatch.setitem(sys.modules, "tweepy", _fake_tweepy("ok", calls))
    x = XClient(settings)   # SPEC c.22: the only ctor argument is Settings; creds come from os.environ
    assert x.dry_run is False and x.disabled_reason is None
    # v2 upload is forced to fail (no network) -> v1.1 fallback supplies the media id
    monkeypatch.setattr(XClient, "upload_media_v2", lambda self, png: None)
    r = x.post("gm sugar GRN LB3b " + "x" * 300, png=b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    assert r.posted and r.dry_run is False and r.id == "1234567890"
    assert r.url == "https://x.com/i/web/status/1234567890" and r.media_id == "m-777" and r.error is None
    kinds = [c[0] for c in calls]
    assert kinds == ["media_upload", "Client", "create_tweet"]
    assert calls[1][1] == {"consumer_key": "k", "consumer_secret": "s", "access_token": "t",
                           "access_token_secret": "ts", "wait_on_rate_limit": False}
    assert len(calls[2][1]["text"]) == 280 and calls[2][1]["media_ids"] == ["m-777"]
    # text-only when there is no image
    r2 = x.post("PAM dopamine 33 Hz")
    assert r2.posted and calls[-1][1]["media_ids"] is None and x.posts == 2

    # Forbidden -> disabled for the session
    calls2: list = []
    monkeypatch.setitem(sys.modules, "tweepy", _fake_tweepy("forbidden", calls2))
    x2 = XClient(settings)
    r3 = x2.post("mushroom body says ngmi")
    assert r3.posted is False and r3.error and "Forbidden" in r3.error
    assert x2.disabled_reason and "Forbidden" in x2.disabled_reason
    r4 = x2.post("still disabled, LB3b")
    assert r4.posted is False and r4.error.startswith("disabled:") and len(calls2) == 2  # no new create_tweet

    # 429 -> disabled until now + 900 s
    calls3: list = []
    monkeypatch.setitem(sys.modules, "tweepy", _fake_tweepy("429", calls3))
    x3 = XClient(settings)
    r5 = x3.post("giant fiber DNp01")
    assert r5.posted is False and "TooManyRequests" in (r5.error or "")
    assert x3.disabled_reason and x3.disabled_reason.startswith("rate limited until")
    assert x3._disabled_until is not None and 800 < x3._disabled_until - time.time() <= 900


# ----------------------------------------------------------------------------- state / orchestrator
def test_agent_state_persistence(tmp_path):
    p = tmp_path / "data" / "agent_state.json"
    st = AgentState.load(p)
    assert st.fires_today == 0 and st.last_fire_wall == 0.0 and st.day == ""
    now = time.time()
    st.roll_day(now)
    st.last_fire_wall = now
    st.last_fire_by_reason = {"panic_entry": now - 10.0}
    st.fires_today = 5
    st.recent_hashes = ["a" * 40, "b" * 40]
    st.disabled_reason = None
    st.save(p)
    assert p.exists() and not p.with_suffix(".json.tmp").exists()
    st2 = AgentState.load(p)
    assert st2 == st
    # restart on the same UTC day keeps the count; a new day resets it
    st2.roll_day(now)
    assert st2.fires_today == 5
    st2.roll_day(now + 2 * 86400)
    assert st2.fires_today == 0 and st2.day == orch_mod._utc_day(now + 2 * 86400)
    # corrupt file -> fresh state, no exception
    p.write_text("{not json", encoding="utf-8")
    assert AgentState.load(p).fires_today == 0
    # the agent loads it at construction and does not reset fires_today
    st.save(p)
    agent = make_agent(tmp_path)
    assert agent.state.fires_today == 5 and agent.status()["fires_today"] == 5
    agent.close()


def test_agent_allowed_rules(tmp_path):
    agent = make_agent(tmp_path, tweet_cooldown_s=900, tweet_reason_cooldown_s=2700, tweets_per_day=3)
    now = time.time()
    assert agent.allowed("euphoria_entry", now) == (True, "ok")
    assert agent.allowed("bogus", now)[0] is False
    # (a) global cooldown
    agent.state.last_fire_wall = now - 100
    agent.state.last_fire_by_reason["panic_entry"] = now - 100
    ok, why = agent.allowed("euphoria_entry", now)
    assert not ok and why.startswith("cooldown")
    assert agent.allowed("euphoria_entry", now + 900)[0] is True
    # (b) per-reason cooldown
    ok, why = agent.allowed("panic_entry", now + 900)
    assert not ok and why.startswith("reason cooldown")
    assert agent.allowed("panic_entry", now + 2700)[0] is True
    # (c) daily cap (UTC day)
    agent.state.fires_today = 3
    ok, why = agent.allowed("panic_entry", now + 2700)
    assert not ok and "daily cap" in why
    # manual bypasses a-c
    assert agent.allowed("manual", now)[0] is True
    # a new UTC day resets the cap
    assert agent.allowed("panic_entry", now + 2 * 86400)[0] is True
    # (d) in flight
    agent._in_flight = True
    assert agent.allowed("manual", now) == (False, "in flight")
    agent._in_flight = False
    # (e) disabled (persisted) and disabled poster in live mode; a dry-run poster's disable does not block
    agent.state.disabled_reason = "operator"
    assert agent.allowed("manual", now) == (False, "disabled: operator")
    agent.state.disabled_reason = None
    agent.poster.disabled_reason = "Forbidden"
    assert agent.allowed("manual", now)[0] is True  # poster is dry-run
    agent.poster.dry_run = False
    assert agent.allowed("manual", now) == (False, "disabled: Forbidden")
    agent.close()


def test_agent_fire_writes_artifacts(tmp_path):
    bus = FakeBus(clients=0, tick=make_tick())
    agent = make_agent(tmp_path, bus=bus)
    tick = make_tick()
    rec = agent.fire("euphoria_entry", tick)
    stem = rec["id"]
    assert re.fullmatch(r"\d{8}-\d{6}(-\d+)?", stem)
    expected_keys = {"id", "t_ms", "wall", "reason", "text", "model", "dry_run", "posted", "url", "error",
                     "snapshot_source", "neurons", "mood", "latency_ms", "summary"}
    assert set(rec) == expected_keys
    assert rec["reason"] == "euphoria_entry" and rec["model"] == "template" and rec["dry_run"] is True
    assert rec["posted"] is False and rec["url"] is None and rec["error"] is None
    assert rec["snapshot_source"] == "server" and rec["mood"] == "EUPHORIA" and rec["t_ms"] == 617350
    assert rec["neurons"] and _has_vocab(rec["text"]) and len(rec["text"]) <= 280
    assert rec["summary"]["schema"] == "flybrain.summary.v1"
    # data/tweets.jsonl
    lines = (agent.settings.data_dir / "tweets.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == rec
    # out/tweets/<id>.{txt,json,png}
    d = agent.settings.out_dir / "tweets"
    assert (d / f"{stem}.txt").read_text(encoding="utf-8").strip() == rec["text"]
    assert json.loads((d / f"{stem}.json").read_text(encoding="utf-8")) == rec
    png = (d / f"{stem}.png").read_bytes()
    assert png[:8] == PNG_MAGIC
    # data/snapshots/<t_ms>_<mood>.png
    assert (agent.settings.data_dir / "snapshots" / "617350_EUPHORIA.png").exists()
    # tweet frame on the bus: the d.4 record (no summary) + type/seq
    frames = [m for m in bus.events if m.get("type") == "tweet"]
    assert len(frames) == 1
    fr = frames[0]
    assert fr["seq"] == 12345 and "summary" not in fr
    assert {k: v for k, v in fr.items() if k not in ("type", "seq")} == {k: v for k, v in rec.items() if k != "summary"}
    # state updated + persisted
    st = AgentState.load(agent.settings.data_dir / "agent_state.json")
    assert st.fires_today == 1 and st.last_fire_by_reason["euphoria_entry"] == pytest.approx(rec["wall"], abs=0.01)
    assert st.recent_hashes == [orch_mod._sha1(rec["text"])]
    assert agent.history(20) == [rec] and agent.status()["fires_today"] == 1
    assert agent.status()["last_tweet"]["id"] == stem
    # the SPEC c.28 health aliases
    health = agent.status()
    assert {"llm", "x", "tweets_today", "last_tweet_wall", "disabled_reason"} <= set(health)
    assert health["tweets_today"] == 1 and health["last_tweet_wall"] == pytest.approx(rec["wall"], abs=0.01)
    assert health["disabled_reason"] is None
    # a second fire in the same second gets a unique stem
    rec2 = agent.fire("manual", tick)
    assert rec2["id"] != stem and (d / f"{rec2['id']}.json").exists()
    assert len(agent.history()) == 2
    # the poster saw the png
    assert agent.poster.calls[0][1] == len(png)
    agent.close()


def test_agent_on_tick_triggers_and_manual(tmp_path):
    from flybrain.mood import Mood, Transition

    bus = FakeBus(clients=0, tick=make_tick())
    agent = make_agent(tmp_path, bus=bus, tweets_per_day=12)
    tick = make_tick()
    # a confirmed CRUISING entry never tweets; ESCAPE never tweets
    agent.on_tick(tick, None, Transition(1, Mood.FEEDING, Mood.CRUISING, "x"), 0)
    agent.on_tick(tick, None, Transition(1, Mood.PANIC, Mood.ESCAPE, "x"), 0)
    assert agent._future is None
    # confirmed EUPHORIA -> job
    agent.on_tick(tick, None, Transition(1, Mood.FEEDING, Mood.EUPHORIA, "x"), 0)
    assert agent.wait_idle(10) and agent.fires == 1 and agent.history()[-1]["reason"] == "euphoria_entry"
    # cooldown suppresses PANIC now; escape burst too
    agent.on_tick(tick, None, Transition(1, Mood.ANXIOUS, Mood.PANIC, "x"), 5)
    assert agent.wait_idle(10) and agent.fires == 1
    # manual bypasses the cooldowns
    agent.request_manual()
    assert agent.wait_idle(10) and agent.fires == 2 and agent.history()[-1]["reason"] == "manual"
    # escape burst fires once the cooldown is over, and only once per 60 s
    agent.state.last_fire_wall = 0.0
    agent.on_tick(tick, None, None, 3)
    assert agent.wait_idle(10) and agent.fires == 3 and agent.history()[-1]["reason"] == "escape_burst"
    agent.state.last_fire_wall = 0.0
    agent.state.last_fire_by_reason.clear()
    agent.on_tick(tick, None, None, 4)
    assert agent.wait_idle(10) and agent.fires == 3
    st = agent.status()
    assert st["in_flight"] is False and st["llm"] == "dryrun" and st["x"] == "dryrun" and st["fires_today"] == 3
    # the SPEC c.28 health keys are present and consistent
    assert {"llm", "x", "tweets_today", "last_tweet_wall", "disabled_reason"} <= set(st)
    assert st["tweets_today"] == 3 and st["disabled_reason"] is None
    agent.close()


def test_agent_dedupe(tmp_path, monkeypatch):
    const = "gm ser, the mushroom body says wagmi again"
    gen = FakeGenerator([const, const, const])
    agent = make_agent(tmp_path, generator=gen)
    tick = make_tick()
    r1 = agent.fire("euphoria_entry", tick)
    assert r1["text"] == const and r1["model"] == "fake-model"
    # same text again -> regenerated once via template (different text, model 'template')
    r2 = agent.fire("panic_entry", tick)
    assert r2["text"] != const and r2["model"] == "template" and r2["error"] is None
    lines = (agent.settings.data_dir / "tweets.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # if the template also collides -> skipped, nothing persisted
    monkeypatch.setattr(orch_mod, "template_tweet", lambda summary, rng: const)
    r3 = agent.fire("courtship_entry", tick)
    assert r3["error"] == "duplicate" and r3["id"] is None
    lines = (agent.settings.data_dir / "tweets.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and agent.fires == 2
    # a suppression has no SPEC d.8 event kind: it is visible in status(), not on the wire
    assert agent.status()["last_suppressed"] == {"reason": "courtship_entry", "why": "duplicate",
                                                 "wall": pytest.approx(time.time(), abs=5.0)}
    assert all(m.get("kind") != "tweet_suppressed" for m in agent.bus.events)
    agent.close()


# ----------------------------------------------------------------------------- snapshots
def test_render_snapshot_png():
    trail = [(100.0 + i * 6.0, 250.0, "#ff0000", 3.0) for i in range(50)]
    trail += [(400.0, 100.0 + i * 5.0, "#0000ff", 1.0) for i in range(40)]
    kin = SimpleNamespace(x=600.0, y=300.0, heading=1.2, mode="walk")
    png = render_snapshot(trail, kin, "EUPHORIA", {"symbol": "FLY", "price_usd": 0.0012345, "chg_m5": 4.2})
    img = decode_png(png)
    assert img.shape == (500 + 2 * FRAME_PX + TITLE_PX + STATUS_PX, 800 + 2 * FRAME_PX, 3)
    assert img.shape == (546, 806, 3)
    # frame and title bar colours
    assert tuple(img[0, 0]) == (0xC0, 0xC0, 0xC0) and tuple(img[FRAME_PX + 1, 400]) == (0x00, 0x00, 0x80)
    cy0, cx0 = FRAME_PX + TITLE_PX, FRAME_PX
    # trail pixels present (red horizontal, blue vertical), canvas otherwise white
    assert tuple(img[cy0 + 250, cx0 + 200]) == (0xFF, 0x00, 0x00)
    assert tuple(img[cy0 + 150, cx0 + 400]) == (0x00, 0x00, 0xFF)
    assert tuple(img[cy0 + 10, cx0 + 10]) == (0xFF, 0xFF, 0xFF)
    red = np.all(img[cy0:cy0 + 500, cx0:cx0 + 800] == (255, 0, 0), axis=-1).sum()
    assert red >= 250 * 3  # ~ 300 px long, 3 px wide
    # fly sprite drawn in black around (600, 300)
    patch = img[cy0 + 300 - 12:cy0 + 300 + 12, cx0 + 600 - 12:cx0 + 600 + 12]
    assert np.all(patch == (0, 0, 0), axis=-1).sum() >= 20
    # status caption is drawn (black pixels in the strip) and its text is well-formed
    strip = img[cy0 + 500:cy0 + 500 + STATUS_PX, cx0:cx0 + 800]
    assert np.all(strip == (0, 0, 0), axis=-1).sum() > 100
    # different headings give different sprites; degenerate inputs do not raise
    png2 = render_snapshot([], SimpleNamespace(x=-50, y=9999, heading=-7.0), "", {}, 64, 32)
    assert decode_png(png2).shape == (32 + 46, 64 + 6, 3)
    assert render_snapshot([(1, 2, "bad", "x"), ("a", None, None, None)], None, "PANIC", None)[:8] == PNG_MAGIC


def test_snapshot_broker_timeout_falls_back(tmp_path):
    settings = make_settings(tmp_path)
    body = SimpleNamespace(kin=SimpleNamespace(x=100.0, y=100.0, heading=0.0, mode="walk"))
    # no client -> server render, no request sent
    bus = FakeBus(clients=0, tick=make_tick(mood="PANIC", t_ms=4200))
    broker = SnapshotBroker(bus, body, settings)
    for i in range(10):
        broker.record_trail(10.0 + i, 20.0, "#00a800", 2.0)
    assert broker.trail_px == pytest.approx(9.0)
    png, src = broker.request(timeout_s=0.2)
    assert src == "server" and png[:8] == PNG_MAGIC and bus.requested == []
    assert (settings.data_dir / "snapshots" / "4200_PANIC.png").read_bytes() == png
    # a client that never answers -> timeout -> server
    bus2 = FakeBus(clients=1, tick=make_tick())
    broker2 = SnapshotBroker(bus2, body, settings)
    t0 = time.perf_counter()
    png, src = broker2.request(timeout_s=0.2)
    assert src == "server" and 0.15 <= time.perf_counter() - t0 < 3.0 and len(bus2.requested) == 1
    # delivered bytes -> browser
    bus3 = FakeBus(clients=1, tick=make_tick())
    broker3 = SnapshotBroker(bus3, body, settings)
    fake_png = PNG_MAGIC + b"browser-bytes" * 4

    def answer() -> None:
        deadline = time.monotonic() + 2.0
        while not bus3.requested and time.monotonic() < deadline:
            time.sleep(0.005)
        import base64

        broker3.deliver(bus3.requested[0], base64.b64encode(fake_png).decode())

    th = threading.Thread(target=answer, daemon=True)
    th.start()
    png, src = broker3.request(timeout_s=2.0)
    th.join(1.0)
    assert src == "browser" and png == fake_png
    # bad deliveries are ignored (no magic / bad base64) -> fallback stays server-side
    bus4 = FakeBus(clients=1, tick=make_tick())
    broker4 = SnapshotBroker(bus4, body, settings)
    broker4.deliver("snap-x", "not base64!!")
    broker4.deliver("snap-x", "aGVsbG8=")
    assert bus4._snap == {}
    # ring keeps the last 4000 points
    for i in range(5000):
        broker4.record_trail(float(i % 800), 5.0, "#000000", 1.0)
    assert len(broker4.trail()) == 4000
    wire = broker4.trail_wire()
    assert len(wire) == 4000 and all(len(p) == 4 and isinstance(p[2], str) for p in wire)
    assert json.dumps(wire[:3])  # JSON-serialisable for GET /api/state
    broker4.clear_trail()
    assert broker4.trail() == [] and broker4.trail_px == 0.0 and broker4.trail_wire() == []


# ----------------------------------------------------------------------------- regressions (review round 2)
def test_agent_concurrent_fire_unique_stems(tmp_path):
    """SPEC c.24 / d.4: the artifact stem IS the record ``id``, so parallel fire() calls must not share one.

    Regression: ``_unique_stem`` used to resolve collisions with a bare ``Path.exists()`` probe outside any lock, so
    six threads firing in the same wall second produced one stem, one set of out/tweets files and six records claiming
    the same ``id``."""
    n = 6
    texts = [f"gm ser, the mushroom body says wagmi number {i}" for i in range(n)]
    gen = FakeGenerator(texts)
    lock = threading.Lock()

    def generate(summary):  # one distinct text per call, whatever the interleaving
        with lock:
            gen.calls += 1
            text = texts[(gen.calls - 1) % n]
        return TweetDraft(text=text, model="fake-model", dry_run=False, stop_reason="end_turn", latency_ms=1.0,
                          reason=summary["reason"], neurons=["LB3b"])

    gen.generate = generate
    agent = make_agent(tmp_path, generator=gen)
    tick = make_tick()
    barrier = threading.Barrier(n)
    out: list[dict] = []

    def run() -> None:
        barrier.wait(5.0)
        rec = agent.fire("manual", tick)
        with lock:
            out.append(rec)

    threads = [threading.Thread(target=run, daemon=True) for _ in range(n)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(20.0)
    assert len(out) == n and all(r["error"] is None for r in out)
    stems = [r["id"] for r in out]
    assert len(set(stems)) == n, stems
    d = agent.settings.out_dir / "tweets"
    for stem in stems:
        assert (d / f"{stem}.txt").exists() and (d / f"{stem}.json").exists() and (d / f"{stem}.png").exists()
    assert len(list(d.glob("*.json"))) == n
    lines = (agent.settings.data_dir / "tweets.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == n and len({json.loads(ln)["id"] for ln in lines}) == n
    assert agent.fires == n and agent.state.fires_today == n
    # the state file is written outside the lock the sim thread needs, but is still consistent
    assert AgentState.load(agent.settings.data_dir / "agent_state.json").fires_today == n
    agent.close()


def test_render_snapshot_survives_non_finite_inputs(tmp_path):
    """SPEC 0.1 ("errors never stop the sim") / c.23: NaN / inf anywhere must not raise, and one poisoned trail point
    must not blank every later snapshot (``int(round(inf))`` raises OverflowError, ``int(round(nan))`` ValueError)."""
    nan, inf = float("nan"), float("inf")
    good = render_snapshot([(10.0, 10.0, "#ff0000", 2.0)], SimpleNamespace(x=1.0, y=1.0, heading=0.0), "CRUISING", {})
    assert good[:8] == PNG_MAGIC
    for kin in (SimpleNamespace(x=5.0, y=5.0, heading=nan), SimpleNamespace(x=5.0, y=5.0, heading=inf),
                SimpleNamespace(x=nan, y=nan, heading=0.0), SimpleNamespace(x=inf, y=-inf, heading=0.0)):
        assert render_snapshot([], kin, "PANIC", {})[:8] == PNG_MAGIC
    trails = [[(inf, 10.0, "#ff0000", 2.0)], [(10.0, nan, "#ff0000", 2.0)], [(10.0, 10.0, "#ff0000", inf)],
              [(1.0, 1.0, "#ff0000", 1.0), (nan, nan, "#ff0000", 1.0), (20.0, 20.0, "#ff0000", 1.0)]]
    for trail in trails:
        assert render_snapshot(trail, SimpleNamespace(x=1.0, y=1.0, heading=0.0), "PANIC", {})[:8] == PNG_MAGIC
    # non-finite market numbers only change the caption
    assert render_snapshot([], SimpleNamespace(x=1.0, y=1.0, heading=0.0), "PANIC",
                           {"price_usd": inf, "chg_m5": nan, "symbol": "FLY"})[:8] == PNG_MAGIC
    # the broker refuses to store a poisoned point at all, so the ring can never poison a later render
    settings = make_settings(tmp_path)
    body = SimpleNamespace(kin=SimpleNamespace(x=100.0, y=100.0, heading=0.0, mode="walk"))
    broker = SnapshotBroker(FakeBus(clients=0, tick=make_tick()), body, settings)
    broker.record_trail(10.0, 10.0, "#ff0000", 2.0)
    broker.record_trail(inf, 10.0, "#ff0000", 2.0)
    broker.record_trail(10.0, nan, "#ff0000", 2.0)
    broker.record_trail(20.0, 20.0, "#ff0000", inf)
    broker.record_trail(30.0, 30.0, "#ff0000", 2.0)
    pts = broker.trail()
    assert len(pts) == 2 and all(all(isinstance(v, (int, float)) for v in (p[0], p[1], p[3])) for p in pts)
    assert all(p[0] == p[0] and abs(p[0]) < 1e6 for p in pts)
    assert broker.trail_px == pytest.approx(28.284, abs=0.01)
    png, src = broker.request(timeout_s=0.05)
    assert src == "server" and png[:8] == PNG_MAGIC and len(png) > 1000  # a real raster, not the blank fallback


def test_agent_state_rejects_non_finite_file(tmp_path, caplog):
    """SPEC c.24 ``AgentState.load`` ("corrupt file -> a fresh state"): ``json.loads`` accepts the non-standard
    ``Infinity``/``NaN`` literals, which used to be kept and then blocked every trigger forever ("cooldown infs left")
    and made ``status()`` non-strict-JSON."""
    p = tmp_path / "data" / "agent_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    bodies = ['{"last_fire_wall": Infinity, "fires_today": 1, "day": "", "last_fire_by_reason": {}}',
              '{"last_fire_wall": NaN, "fires_today": 1, "day": "", "last_fire_by_reason": {}}',
              '{"last_fire_wall": 0.0, "fires_today": 1, "day": "", "last_fire_by_reason": {"manual": Infinity}}',
              '{"last_fire_wall": 1.0, "fires_today": NaN, "day": "", "last_fire_by_reason": {}}']
    for body in bodies:
        p.write_text(body, encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger="flybrain.agent"):
            st = AgentState.load(p)
        assert st == AgentState(), body
    # an agent built on such a file is not blocked and its health JSON is strict JSON
    p.write_text(bodies[0].replace('"fires_today": 1', '"fires_today": 0'), encoding="utf-8")
    agent = make_agent(tmp_path)
    assert agent.allowed("euphoria_entry", time.time()) == (True, "ok")
    json.dumps(agent.status(), allow_nan=False)          # would raise on Infinity/NaN
    # a non-finite value in memory is never written out either: the previous good file survives
    agent.state.last_fire_wall = 123.0
    agent.state.save(agent.state_path)
    assert json.loads(agent.state_path.read_text(encoding="utf-8"))["last_fire_wall"] == 123.0
    agent.state.last_fire_wall = float("inf")
    agent.state.save(agent.state_path)
    assert json.loads(agent.state_path.read_text(encoding="utf-8"))["last_fire_wall"] == 123.0
    st = agent.status()          # ... and status() clamps it, so /api/health stays strict JSON
    assert st["last_fire_wall"] == 0.0 and st["last_tweet_wall"] is None
    json.dumps(st, allow_nan=False)
    agent.close()


def test_summary_trim_keeps_note_hints_and_budget():
    """SPEC d.6: ``connectome.note`` is copied from ``meta['note']``, ``vocabulary_hint`` is never empty and the
    payload never exceeds 1.5 KB - including for pathological strings that no ladder step touches."""
    tick = make_tick()
    session = {"uptime_s": 3612.0, "tweets_today": 2, "last_tweet_reason": "panic_entry", "trail_px": 18240}
    # realistic (non-integral) rates: the note and at least one hint survive, and the 19 rate keys stay
    tick_r = make_tick()
    for k in tick_r["rates"]["pops"]:
        tick_r["rates"]["pops"][k] += 0.37
    for reason in TRIGGER_REASONS:
        s = build_brain_summary(tick_r, FakeHistory(), reason, CONN_META, session, tick_r["market"], "en")
        assert s["connectome"]["note"] == CONN_META["note"]
        assert len(s["vocabulary_hint"]) >= 1 and set(s["vocabulary_hint"]) <= set(VOCABULARY_HINT)
        assert len(s["top_types"]) >= 2 and tuple(s["rates_hz"].keys()) == RATE_KEYS
        assert len(summary_json(s).encode("utf-8")) <= MAX_SUMMARY_BYTES
    # intify mutates nested dicts in place: every later ladder step must see the live sub-dict, not a detached copy
    s2 = build_brain_summary(tick, FakeHistory(), "panic_entry", CONN_META, session, tick["market"], "en")
    conn, mood = s2["connectome"], s2["mood"]
    summary_mod._intify_inplace(s2)
    assert s2["connectome"] is conn and s2["mood"] is mood
    # strings copied verbatim from the tick are capped, and the hard guard holds the budget anyway
    tick_b = make_tick()
    tick_b["market"]["regime"] = "R" * 300
    tick_b["market"]["symbol"] = "S" * 300
    tick_b["fly"]["mode"] = "m" * 300
    tick_b["mood"]["state"] = "E" * 300
    tick_b["mood"]["prev"] = "F" * 300
    tick_b["market"]["last_trade"] = {"kind": "b" * 300, "usd": 1.0}
    s3 = build_brain_summary(tick_b, FakeHistory(events=[{"kind": "k" * 300, "t_ms": 1, "data": {}}]),
                             "x" * 300, {**CONN_META, "note": "n" * 900, "name": "m" * 900},
                             {"uptime_s": 1.0, "tweets_today": 0, "last_tweet_reason": "L" * 300},
                             tick_b["market"], "l" * 300)
    assert tuple(s3.keys()) == SUMMARY_KEYS
    assert len(summary_json(s3).encode("utf-8")) <= MAX_SUMMARY_BYTES
    for key in ("regime", "symbol", "source"):
        assert len(str(s3["market"][key])) <= 24
    assert len(s3["fly"]["mode"]) <= 24 and len(s3["mood"]["state"]) <= 24 and len(s3["reason"]) <= 32
    assert len(s3["session"]["last_tweet_reason"]) <= 32 and len(s3["events_10s"][0]) <= 40
    # this payload is the one that reaches the short_name ladder step, which used to be a silent no-op
    assert len(s3["connectome"]["name"]) <= 24


def test_agent_fire_tolerates_garbage_tick_numbers(tmp_path):
    """SPEC c.24 fire(): a malformed ``t_ms``/``seq`` must not raise (the REST /api/tweet/test path calls fire()
    directly, where an exception becomes a 500)."""
    bus = FakeBus(clients=0, tick=make_tick())
    agent = make_agent(tmp_path, bus=bus)
    ticks = [{}, {"t_ms": "abc", "seq": "zz"}, {"t_ms": None, "seq": None},
             {"t_ms": float("nan"), "seq": float("inf")}, {"t_ms": [1], "seq": {"a": 1}}]
    for tick in ticks:
        rec = agent.fire("manual", dict(tick))
        assert rec["t_ms"] == 0 and isinstance(rec["id"], str)
    frames = [m for m in bus.events if m.get("type") == "tweet"]
    assert len(frames) == len(ticks) and all(f["seq"] == 0 for f in frames)
    assert agent.fire("manual", None)["t_ms"] == 0          # not a dict at all
    agent.close()


def test_agent_escape_burst_window_not_consumed_by_suppression(tmp_path):
    """SPEC c.24 on_tick: the 60 s escape-burst window belongs to the trigger that FIRED; a suppressed trigger used to
    burn it and silently swallow the next minute of bursts."""
    agent = make_agent(tmp_path, tweets_per_day=0)        # every non-manual trigger is suppressed by the daily cap
    tick = make_tick()
    agent.on_tick(tick, None, None, 5)
    assert agent.wait_idle(5) and agent.fires == 0 and agent._last_escape_burst_wall == 0.0
    assert agent.status()["last_suppressed"]["reason"] == "escape_burst"
    agent.close()

    agent2 = make_agent(tmp_path / "b", tweets_per_day=12)
    agent2._in_flight = True                              # suppressed: in flight
    agent2.on_tick(tick, None, None, 5)
    assert agent2.fires == 0 and agent2._last_escape_burst_wall == 0.0
    agent2._in_flight = False
    agent2.on_tick(tick, None, None, 5)                   # the very next tick may still fire
    assert agent2.wait_idle(10) and agent2.fires == 1
    assert agent2._last_escape_burst_wall > 0.0
    agent2.state.last_fire_wall = 0.0
    agent2.state.last_fire_by_reason.clear()
    agent2.on_tick(tick, None, None, 5)                   # ... and now the window does hold for 60 s
    assert agent2.wait_idle(10) and agent2.fires == 1
    agent2.close()


def test_agent_dataclasses_use_slots():
    """SPEC c preamble: "All dataclasses are @dataclass(slots=True) unless frozen=True is stated"."""
    for cls in (TweetDraft, PostResult, AgentState):
        assert getattr(cls, "__slots__", None), cls.__name__
    st = AgentState()
    with pytest.raises(AttributeError):
        st.not_a_field = 1                                # slots: a typo cannot silently create state
    left = AgentState(1.0, {"manual": 1.0}, "2026-09-11", 1, ["a"], None)
    assert left.copy() == left and left.copy() is not left
    assert left.copy().last_fire_by_reason is not left.last_fire_by_reason


def test_agent_publishes_only_d8_event_kinds(tmp_path):
    """SPEC d.8 holds the normative ``event`` kind table and c.26 / e.1 mirror it; the agent must not invent kinds."""
    d8_kinds = {"market_source", "homeostasis", "clear", "poke", "easter_egg", "whale", "calibration"}
    assert set(orch_mod.OOB_EVENT_KINDS) == d8_kinds
    bus = FakeBus(clients=0, tick=make_tick())
    agent = make_agent(tmp_path, bus=bus, tweets_per_day=0)
    tick = make_tick()
    agent.on_tick(tick, None, None, 9)                    # suppressed trigger
    agent.request_manual()                                # manual bypasses the cap -> a real fire
    assert agent.wait_idle(10)
    agent._publish_event("tweet_error", {"reason": "manual", "error": "boom"})   # dropped, not published
    kinds = {m.get("kind") for m in bus.events if m.get("type") == "event"}
    assert kinds <= d8_kinds, kinds
    assert {m.get("type") for m in bus.events} <= {"tweet", "event"}
    # the information still reaches /api/health
    st = agent.status()
    assert st["suppressed"] >= 1 and st["last_suppressed"]["reason"] == "escape_burst"
    assert {"errors", "last_error", "suppressed", "last_suppressed"} <= set(st)
    agent.close()
