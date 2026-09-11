"""Tweet agent orchestration (SPEC section c.24).

``TweetAgent`` sits on the sim thread through ``on_tick`` (cheap, never blocks) and runs every tweet job on a
1-worker ``ThreadPoolExecutor``: summary -> text (template or LLM) -> validation -> dedupe -> snapshot -> post ->
persistence (``data/tweets.jsonl``, ``data/agent_state.json``, ``out/tweets/<id>.{txt,json,png}``) -> ``tweet``
frame on the bus. Triggers: a confirmed EUPHORIA / PANIC / COURTSHIP entry (3 s hold, ``MoodMachine.confirmed``) or an
escape burst (>= 3 jumps in 60 s); ``request_manual`` (the ``tweet_test`` frame) bypasses the cooldowns.
Rate limits: global cooldown, per-reason cooldown and a UTC daily cap, all persisted so a restart cannot reset them.
Every failure produces a log line and an event, never an exception on the sim thread.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import itertools
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from flybrain.agent.llm import TweetGenerator, _neurons_from_text, template_tweet, validate_tweet
from flybrain.agent.snapshot import SnapshotBroker
from flybrain.agent.summary import build_brain_summary
from flybrain.agent.x_client import XClient

if TYPE_CHECKING:  # pragma: no cover - typing only
    from flybrain.config import Settings
    from flybrain.mood import Transition
    from flybrain.server.state import StateBus, TickHistory

__all__ = ["TRIGGER_REASONS", "TWEET_MOODS", "AgentState", "TweetAgent", "JOB_TIMEOUT_S"]

log = logging.getLogger("flybrain.agent")

TRIGGER_REASONS: tuple[str, ...] = ("euphoria_entry", "panic_entry", "courtship_entry", "escape_burst", "manual")
TWEET_MOODS: tuple[str, ...] = ("EUPHORIA", "PANIC", "COURTSHIP")
JOB_TIMEOUT_S = 30.0
_ESCAPE_BURST_GAP_S = 60.0
_RECENT_HASHES = 20
_SUPPRESS_LOG_GAP_S = 30.0
_STEM_MEMORY = 512

#: Out-of-band ``event`` kinds of SPEC d.8 - the only kinds the agent is allowed to publish on the bus. Suppressions
#: and job failures have no d.8 kind, so they live in the log and in ``status()`` instead of inventing a wire kind.
OOB_EVENT_KINDS: tuple[str, ...] = ("market_source", "homeostasis", "clear", "poke", "easter_egg", "whale",
                                    "calibration")


def _utc_day(now: float) -> str:
    return dt.datetime.fromtimestamp(now, tz=dt.timezone.utc).strftime("%Y-%m-%d")


def _ascii(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _int(x: Any, default: int = 0) -> int:
    """``int(x)`` that never raises (a non-numeric ``t_ms``/``seq`` in a malformed tick must not kill a tweet)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return int(v)


def _finite(x: Any, default: float = 0.0) -> float:
    """``float(x)`` clamped to a finite value; ``None``/NaN/inf/garbage -> ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _strict_float(x: Any) -> float:
    """``float(x)`` that REJECTS NaN/Infinity (``json.loads`` accepts those literals) - used by ``AgentState.load``."""
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"non-finite value {v!r}")
    return v


@dataclass(slots=True)
class AgentState:
    """Persisted as data/agent_state.json."""

    last_fire_wall: float = 0.0
    last_fire_by_reason: dict[str, float] = field(default_factory=dict)
    day: str = ""
    fires_today: int = 0
    recent_hashes: list[str] = field(default_factory=list)
    disabled_reason: str | None = None

    @classmethod
    def load(cls, path: Path) -> "AgentState":
        """Missing or corrupt file -> a fresh state (never raises)."""
        path = Path(path)
        st = cls()
        try:
            if not path.exists():
                return st
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("agent: could not read %s (%s); starting fresh", path, exc)
            return st
        if not isinstance(raw, dict):
            return st
        try:
            # json.loads() accepts the non-standard Infinity/NaN literals, so every float is checked for finiteness:
            # an infinite last_fire_wall would block every trigger forever and make status() non-JSON-encodable.
            st.last_fire_wall = _strict_float(raw.get("last_fire_wall") or 0.0)
            lfr = raw.get("last_fire_by_reason")
            st.last_fire_by_reason = ({str(k): _strict_float(v) for k, v in lfr.items()}
                                      if isinstance(lfr, dict) else {})
            st.day = str(raw.get("day") or "")
            st.fires_today = int(_strict_float(raw.get("fires_today") or 0))
            rh = raw.get("recent_hashes")
            st.recent_hashes = [str(h) for h in rh][-_RECENT_HASHES:] if isinstance(rh, list) else []
            dr = raw.get("disabled_reason")
            st.disabled_reason = None if dr in (None, "") else str(dr)
        except (TypeError, ValueError, AttributeError) as exc:
            log.warning("agent: malformed %s (%s); starting fresh", path, exc)
            return cls()
        return st

    def save(self, path: Path) -> None:
        """Atomic write (tmp + replace); errors are logged, never raised."""
        path = Path(path)
        try:
            # allow_nan=False: a non-finite counter must fail here rather than produce a file that load() rejects.
            body = json.dumps(asdict(self), indent=1, allow_nan=False)
        except ValueError as exc:
            log.error("agent: refusing to save non-finite state (%s)", exc)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(body, encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            log.error("agent: could not save %s (%s)", path, exc)

    def copy(self) -> "AgentState":
        """A detached snapshot, so ``save()`` can run outside the lock the sim thread needs (SPEC 0.1)."""
        return AgentState(self.last_fire_wall, dict(self.last_fire_by_reason), self.day, self.fires_today,
                          list(self.recent_hashes), self.disabled_reason)

    def roll_day(self, now: float) -> None:
        day = _utc_day(now)
        if day != self.day:
            self.day = day
            self.fires_today = 0


class TweetAgent:
    """The agent state machine (SPEC c.24). Constructed always (dry-run included) so the trail ring exists."""

    def __init__(self, settings: "Settings", generator: TweetGenerator, poster: XClient, snapshots: SnapshotBroker,
                 bus: "StateBus", history: "TickHistory", conn_meta: dict, seed: int) -> None:
        self.settings = settings
        self.generator = generator
        self.poster = poster
        self.snapshots = snapshots
        self.bus = bus
        # SPEC c.24 names both the ctor argument and the records accessor ``history``; the TickHistory is kept as
        # ``tick_history`` so that ``TweetAgent.history(limit)`` stays the method of the contract.
        self.tick_history = history
        self.conn_meta = dict(conn_meta or {})
        self.seed = int(seed)

        self.data_dir = Path(getattr(settings, "data_dir", "data"))
        self.out_dir = Path(getattr(settings, "out_dir", "out"))
        self.state_path = self.data_dir / "agent_state.json"
        self.tweets_path = self.data_dir / "tweets.jsonl"
        self.tweet_dir = self.out_dir / "tweets"

        self.cooldown_s = float(getattr(settings, "tweet_cooldown_s", 900))
        self.reason_cooldown_s = float(getattr(settings, "tweet_reason_cooldown_s", 2700))
        self.tweets_per_day = int(getattr(settings, "tweets_per_day", 12))
        self.lang = str(getattr(settings, "tweet_lang", "en") or "en")

        self._lock = threading.RLock()
        self._stem_lock = threading.Lock()   # guards the artifact-stem allocation (fire() may be called concurrently)
        self._save_lock = threading.Lock()   # serialises agent_state.json writes (done outside self._lock)
        self._append_lock = threading.Lock()  # serialises data/tweets.jsonl appends
        self._stem_seen: list[str] = []
        self._stem_count = itertools.count(1)
        self.state = AgentState.load(self.state_path)
        self.state.roll_day(time.time())
        self._executor = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tweet-agent")
        self._future: cf.Future | None = None
        self._in_flight = False
        self._last_tick: dict | None = None
        self._last_escape_burst_wall = 0.0
        self._suppress_log: dict[str, float] = {}
        self._records: list[dict] = self._load_records()
        self.started_wall = time.time()
        self.fires = 0
        self.errors = 0
        self.suppressed = 0
        self.last_suppressed: dict | None = None
        self.last_error: str | None = None
        self.last_record: dict | None = None
        self._closed = False

    # ------------------------------------------------------------------ persistence helpers
    def _load_records(self, limit: int = 100) -> list[dict]:
        try:
            if not self.tweets_path.exists():
                return []
            lines = self.tweets_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out: list[dict] = []
        for line in lines[-limit:]:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def _append_record(self, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        except ValueError as exc:  # a non-finite number would make the line unparseable for GET /api/tweets
            log.error("agent: record for %s is not strict JSON (%s)", record.get("id"), exc)
            return
        try:
            self.tweets_path.parent.mkdir(parents=True, exist_ok=True)
            with self._append_lock:   # concurrent fire() calls must not interleave inside one line
                with self.tweets_path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError as exc:
            log.error("agent: could not append %s (%s)", self.tweets_path, exc)

    def _write_artifacts(self, stem: str, text: str, record: dict, png: bytes | None) -> None:
        try:
            self.tweet_dir.mkdir(parents=True, exist_ok=True)
            (self.tweet_dir / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
            (self.tweet_dir / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
            if png:
                (self.tweet_dir / f"{stem}.png").write_bytes(png)
        except OSError as exc:
            log.error("agent: could not write out/tweets/%s.* (%s)", stem, exc)

    def _unique_stem(self, wall: float) -> str:
        """A unique ``out/tweets/<stem>`` / d.4 ``id``. Two ``fire()`` calls in the same wall second (or in parallel)
        must not share a stem, so the allocation is serialised and remembers what it handed out: the filesystem probe
        alone loses the race because the artifacts are written after the stem is chosen."""
        base = time.strftime("%Y%m%d-%H%M%S", time.gmtime(_finite(wall)))
        with self._stem_lock:
            stem = base
            while stem in self._stem_seen or (self.tweet_dir / f"{stem}.json").exists():
                stem = f"{base}-{next(self._stem_count)}"
            self._stem_seen.append(stem)
            if len(self._stem_seen) > _STEM_MEMORY:
                del self._stem_seen[:-_STEM_MEMORY]
            return stem

    # ------------------------------------------------------------------ sim-thread entry points
    def on_tick(self, tick: dict, transition: "Transition | None", confirmed: "Transition | None",
                jumps_60s: int) -> None:
        """Sim thread; never blocks. Trigger when confirmed.dst in {EUPHORIA, PANIC, COURTSHIP} (reason '<mood>_entry')
        or jumps_60s >= 3 (reason 'escape_burst', once per 60 s). allowed(reason) -> submit fire() to the 1-worker
        executor; otherwise log the suppression reason."""
        self._last_tick = tick
        now = time.time()
        reason: str | None = None
        if confirmed is not None:
            dst = getattr(confirmed, "dst", None)
            dst_name = str(getattr(dst, "value", dst) or "")
            if dst_name in TWEET_MOODS:
                reason = f"{dst_name.lower()}_entry"
        if reason is None and _int(jumps_60s) >= 3 and now - self._last_escape_burst_wall >= _ESCAPE_BURST_GAP_S:
            reason = "escape_burst"
        if reason is None:
            return
        ok, why = self.allowed(reason, now)
        if ok and self._submit(reason, tick):
            # the 60 s escape-burst window belongs to the trigger that actually fired: a suppressed one leaves it
            # open so the next burst is still considered.
            if reason == "escape_burst":
                self._last_escape_burst_wall = now
        else:
            self._note_suppressed(reason, why if not ok else "submit refused")
            last = self._suppress_log.get(reason, 0.0)
            if now - last >= _SUPPRESS_LOG_GAP_S:
                self._suppress_log[reason] = now
                log.info("agent: trigger %s suppressed (%s)", reason, why)

    def request_manual(self) -> None:
        """The ``tweet_test`` frame: reason 'manual' (bypasses the cooldowns; never posts unless FLY_X=post)."""
        tick = self._last_tick
        if tick is None:
            try:
                tick = self.bus.latest_tick() if self.bus is not None else None
            except Exception:
                tick = None
        ok, why = self.allowed("manual", time.time())
        if not ok:
            log.info("agent: manual tweet suppressed (%s)", why)
            self._note_suppressed("manual", why)
            return
        if not self._submit("manual", tick or {}):
            self._note_suppressed("manual", "submit refused")

    def allowed(self, reason: str, now: float) -> tuple[bool, str]:
        """(a) now - last_fire >= tweet_cooldown_s; (b) now - last_fire_by_reason[reason] >= tweet_reason_cooldown_s;
        (c) fires_today < tweets_per_day (UTC day); (d) not in flight; (e) not disabled. reason == 'manual' bypasses a-c."""
        with self._lock:
            st = self.state
            st.roll_day(now)
            if self._in_flight:
                return False, "in flight"
            if st.disabled_reason:
                return False, f"disabled: {st.disabled_reason}"
            if not getattr(self.poster, "dry_run", True):
                x_disabled = getattr(self.poster, "disabled_reason", None)
                if x_disabled:
                    return False, f"disabled: {x_disabled}"
            if reason == "manual":
                return True, "manual"
            if reason not in TRIGGER_REASONS:
                return False, f"unknown reason {reason!r}"
            last_fire = _finite(st.last_fire_wall)
            left = self.cooldown_s - (now - last_fire)
            if last_fire > 0 and left > 0:
                return False, f"cooldown {left:.0f}s left"
            last_r = _finite(st.last_fire_by_reason.get(reason, 0.0))
            left_r = self.reason_cooldown_s - (now - last_r)
            if last_r > 0 and left_r > 0:
                return False, f"reason cooldown {left_r:.0f}s left"
            if st.fires_today >= self.tweets_per_day:
                return False, f"daily cap {self.tweets_per_day} reached"
            return True, "ok"

    # ------------------------------------------------------------------ job machinery
    def _submit(self, reason: str, tick: dict) -> bool:
        payload = dict(tick) if isinstance(tick, dict) else {}
        with self._lock:
            if self._in_flight or self._closed:
                return False
            self._in_flight = True
            try:
                self._future = self._executor.submit(self._job, reason, payload)
            except RuntimeError as exc:  # executor shut down
                self._in_flight = False
                log.error("agent: cannot submit tweet job (%s)", exc)
                return False
        return True

    def _job(self, reason: str, tick: dict) -> dict | None:
        """Executor worker: runs fire() on a helper thread with a hard 30 s bound on the in-flight state."""
        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["result"] = self.fire(reason, tick)
            except Exception as exc:  # noqa: BLE001
                box["error"] = exc

        th = threading.Thread(target=_run, name="tweet-fire", daemon=True)
        th.start()
        th.join(JOB_TIMEOUT_S)
        try:
            if th.is_alive():
                self._note_error(reason, f"tweet job {reason} exceeded {JOB_TIMEOUT_S:.0f}s")
                log.error("agent: %s", self.last_error)
                return None
            if "error" in box:
                exc = box["error"]
                self._note_error(reason, f"{type(exc).__name__}: {exc}"[:200])
                log.exception("agent: tweet job %s failed", reason, exc_info=exc)
                return None
            return box.get("result")
        finally:
            with self._lock:
                self._in_flight = False

    def wait_idle(self, timeout_s: float = 35.0) -> bool:
        """Block until the current job (if any) has finished; True when idle. For tests and shutdown."""
        fut = self._future
        if fut is None:
            return True
        try:
            fut.result(timeout=timeout_s)
        except cf.TimeoutError:
            return False
        except Exception:  # noqa: BLE001 - the job logs its own failure
            pass
        return not self._in_flight

    def close(self) -> None:
        self._closed = True
        self.wait_idle(5.0)
        self._executor.shutdown(wait=False)

    def _note_suppressed(self, reason: str, why: str) -> None:
        """A trigger that did not fire. SPEC d.8 has no event kind for this, so it stays local: log + ``status()``."""
        with self._lock:
            self.suppressed += 1
            self.last_suppressed = {"reason": reason, "why": why, "wall": round(time.time(), 3)}

    def _note_error(self, reason: str, error: str) -> None:
        """A job that failed. Like a suppression this has no SPEC d.8 kind: log + ``status()`` only."""
        with self._lock:
            self.errors += 1
            self.last_error = error

    def _publish_event(self, kind: str, data: dict) -> None:
        """Publish an out-of-band SPEC d.8 ``event`` frame. A kind outside the normative table is dropped (a frontend
        type union / pydantic Literal over d.8 must never see an unknown kind)."""
        if self.bus is None:
            return
        if kind not in OOB_EVENT_KINDS:
            log.debug("agent: event kind %r is not in the SPEC d.8 table; not published", kind)
            return
        tick = self._last_tick if isinstance(self._last_tick, dict) else {}
        msg = {"type": "event", "seq": _int(tick.get("seq")), "t_ms": _int(tick.get("t_ms")),
               "wall": time.time(), "kind": kind, "data": data}
        try:
            self.bus.publish_event(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent: publish_event failed (%s)", exc)

    # ------------------------------------------------------------------ the job itself
    def _session(self) -> dict:
        with self._lock:
            st = self.state
            last_reason = None
            if st.last_fire_by_reason:
                last_reason = max(st.last_fire_by_reason.items(), key=lambda kv: kv[1])[0]
            return {"uptime_s": round(time.time() - self.started_wall, 1), "tweets_today": int(st.fires_today),
                    "last_tweet_reason": last_reason,
                    "trail_px": float(getattr(self.snapshots, "trail_px", 0.0) or 0.0)}

    def fire(self, reason: str, tick: dict) -> dict:
        """summary -> generator.generate -> validate -> snapshot -> poster.post -> record (SPEC c.24).

        Returns the SPEC d.4 record (plus ``summary``); on a dedupe skip the record carries ``error='duplicate'`` and
        nothing is persisted. Safe to call synchronously (tests, selftest)."""
        t0 = time.perf_counter()
        wall = time.time()
        tick = tick if isinstance(tick, dict) else {}
        market = tick.get("market") if isinstance(tick.get("market"), dict) else {}
        mood = tick.get("mood") if isinstance(tick.get("mood"), dict) else {}
        mood_name = str(mood.get("state") or "CRUISING")
        summary = build_brain_summary(tick, self.tick_history, reason, self.conn_meta, self._session(), market,
                                      self.lang)

        # -- text
        draft = self.generator.generate(summary)
        text, model, neurons, error = draft.text, draft.model, list(draft.neurons), draft.error
        refused = bool(draft.refused)
        if refused or not text:
            text = template_tweet(summary, self.generator.rng)
            model = "template"
            neurons = []
            error = error or ("refusal" if refused else "empty text")
        ok, text = validate_tweet(text)
        if not ok:
            text = validate_tweet(template_tweet(summary, self.generator.rng))[1]
            model = "template"
            neurons = []
        if not neurons:
            neurons = _neurons_from_text(text)

        # -- dedupe: same text as one of the last 20 -> regenerate once via template, then skip
        with self._lock:
            recent = list(self.state.recent_hashes[-_RECENT_HASHES:])
        if _sha1(text) in recent:
            alt = validate_tweet(template_tweet(summary, self.generator.rng))[1]
            if _sha1(alt) in recent or not alt:
                log.info("agent: duplicate tweet for %s skipped", reason)
                self._note_suppressed(reason, "duplicate")
                return {"id": None, "t_ms": _int(tick.get("t_ms")), "wall": wall, "reason": reason,
                        "text": text, "model": model, "dry_run": True, "posted": False, "url": None,
                        "error": "duplicate", "snapshot_source": None, "neurons": neurons, "mood": mood_name,
                        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1), "summary": summary}
            text, model, neurons = alt, "template", _neurons_from_text(alt)

        # -- snapshot
        png: bytes | None = None
        snapshot_source = "server"
        try:
            png, snapshot_source = self.snapshots.request(timeout_s=3.0)
        except Exception as exc:  # noqa: BLE001
            log.warning("agent: snapshot failed (%s); posting without image", exc)
            png = None

        # -- post
        result = self.poster.post(text, png)
        if result.error and not error:
            error = result.error
        dry_run = bool(result.dry_run)

        # -- record (the stem allocation is serialised in _unique_stem; the state mutation takes self._lock)
        stem = self._unique_stem(wall)
        record = {
            "id": stem, "t_ms": _int(tick.get("t_ms")), "wall": round(wall, 3), "reason": reason,
            "text": text, "model": model, "dry_run": dry_run, "posted": bool(result.posted), "url": result.url,
            "error": error, "snapshot_source": snapshot_source, "neurons": neurons, "mood": mood_name,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        full = dict(record)
        full["summary"] = summary
        self._append_record(full)
        self._write_artifacts(stem, text, full, png)
        with self._lock:
            st = self.state
            st.roll_day(wall)
            st.fires_today += 1
            if not refused:  # a refusal consumes no cooldown (SPEC c.21)
                st.last_fire_wall = wall
                st.last_fire_by_reason[reason] = wall
            st.recent_hashes = (st.recent_hashes + [_sha1(text)])[-_RECENT_HASHES:]
            snap = st.copy()
            self._records.append(full)
            self._records = self._records[-200:]
            self.fires += 1
            self.last_record = full
        # the state file is written OUTSIDE self._lock: allowed() runs on the sim thread and must never wait on I/O.
        with self._save_lock:
            snap.save(self.state_path)

        frame = {"type": "tweet", "seq": _int(tick.get("seq"))}
        frame.update(record)
        if self.bus is not None:  # None only in the bus-less paths (scripts/selftest.py, tests)
            try:
                self.bus.publish_event(frame)
            except Exception as exc:  # noqa: BLE001
                log.warning("agent: publish tweet frame failed (%s)", exc)
        log.info("agent: fired %s (%s, %s, %.0f ms): %s", reason, model, "posted" if result.posted else "dry-run",
                 record["latency_ms"], _ascii(text))
        return full

    # ------------------------------------------------------------------ introspection
    def history_records(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._records[-max(0, int(limit)):])

    def history(self, limit: int = 20) -> list[dict]:
        """The last ``limit`` tweet records (newest last) for GET /api/tweets."""
        return self.history_records(limit)

    def status(self) -> dict:
        """For /api/health. Every float is finite so the health JSON is strict-JSON (``Infinity`` would break
        ``JSON.parse`` in the browser)."""
        now = time.time()
        with self._lock:
            st = self.state
            st.roll_day(now)
            last_fire = _finite(st.last_fire_wall)
            cooldown_left = max(0.0, self.cooldown_s - (now - last_fire)) if last_fire > 0 else 0.0
            return {
                "llm": "dryrun" if self.generator.dry_run else str(getattr(self.settings, "llm", "anthropic")),
                "model": self.generator.model if not self.generator.dry_run else "template",
                "x": "dryrun" if self.poster.dry_run else "post",
                "x_disabled_reason": self.poster.disabled_reason,
                "disabled_reason": st.disabled_reason or self.poster.disabled_reason,
                "in_flight": self._in_flight,
                "fires_today": int(st.fires_today),
                # SPEC c.28 spells the two health keys 'tweets_today' / 'last_tweet_wall'; both are aliases of the
                # persisted counters so GET /api/health can read agent.status() straight through.
                "tweets_today": int(st.fires_today),
                "last_tweet_wall": last_fire or None,
                "tweets_per_day": self.tweets_per_day,
                "fires_session": self.fires,
                "errors": self.errors,
                "last_error": self.last_error,
                "suppressed": self.suppressed,
                "last_suppressed": dict(self.last_suppressed) if self.last_suppressed else None,
                "last_fire_wall": last_fire,
                "last_fire_by_reason": {k: _finite(v) for k, v in st.last_fire_by_reason.items()},
                "cooldown_left_s": round(cooldown_left, 1),
                "cooldown_s": self.cooldown_s,
                "reason_cooldown_s": self.reason_cooldown_s,
                "lang": self.lang,
                "last_tweet": ({k: v for k, v in self.last_record.items() if k != "summary"}
                               if self.last_record else None),
            }
