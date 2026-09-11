"""Cross-thread state: ``StateBus`` (sim thread <-> asyncio) and ``TickHistory`` (short-term memory).

SPEC c.25. Three threads meet here and never share anything else (SPEC 0.1): ``SimulationLoop``
publishes ticks and events, uvicorn's asyncio loop fans them out to WebSocket clients, and the agent's
worker thread asks for browser snapshots. Every hand-off is either a ``threading.Lock``-guarded deque
or ``loop.call_soon_threadsafe``; the sim thread never blocks on I/O and never awaits.

**What a subscriber queue carries**: ready-to-send JSON **text** (a ``Frame``, which *is* a ``str``, with
``.data`` / ``.kind`` for introspection). The tick JSON is built exactly once per tick
(``protocol.tick_to_json``) and the same string goes to every client, as SPEC d.10 requires, so a WS
handler is just ``await ws.send_text(frame)``. Queues are ``maxsize=4`` and drop the **oldest** frame
when full: a slow browser loses old ticks, it never stalls the simulation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Final, Iterable, Mapping

from .protocol import SNAPSHOT_DEADLINE_MS, server_json, snapshot_request_frame, tick_to_json

__all__ = ["Frame", "StateBus", "TickHistory", "QUEUE_MAXSIZE", "MAX_PENDING_POKES", "MAX_PENDING_COMMANDS",
           "HISTORY_SECONDS", "COUNTER_WINDOW_S"]

log = logging.getLogger("flybrain.server.state")

QUEUE_MAXSIZE: Final[int] = 4            #: [E] per-client queue depth (SPEC d.10)
MAX_PENDING_POKES: Final[int] = 256      #: [E] bound on client -> sim pokes between two ticks
MAX_PENDING_COMMANDS: Final[int] = 256   #: [E] bound on client -> sim commands between two ticks
HISTORY_SECONDS: Final[float] = 10.0     #: [E] default full-tick ring (SPEC c.25)
COUNTER_WINDOW_S: Final[float] = 900.0   #: [E] how long compact event counters live (gf_spikes(600) works)


class Frame(str):
    """A wire frame: the JSON text (so ``ws.send_text(frame)`` just works) plus ``.data`` / ``.kind``."""

    __slots__ = ("_data",)

    def __new__(cls, text: str, data: Mapping[str, Any] | None = None) -> "Frame":
        obj = super().__new__(cls, text)
        obj._data = dict(data) if data is not None else None  # type: ignore[attr-defined]
        return obj

    @property
    def data(self) -> dict[str, Any]:
        """The frame as a dict (parsed lazily when it was built from text only)."""
        if self._data is None:  # type: ignore[attr-defined]
            try:
                parsed = json.loads(str(self))
            except ValueError:
                parsed = {}
            self._data = parsed if isinstance(parsed, dict) else {}  # type: ignore[attr-defined]
        return self._data  # type: ignore[attr-defined,return-value]

    @property
    def kind(self) -> str:
        """The frame ``type`` (``"tick"``, ``"event"``, ...)."""
        value = self.data.get("type")
        return str(value) if isinstance(value, str) else ""


class StateBus:
    """Sim thread -> asyncio bridge, plus the client -> sim command path (SPEC c.25).

    Created in the app lifespan with the running loop (``StateBus(asyncio.get_running_loop())``). A
    ``loop=None`` bus is fully functional for tests and headless scripts: publishing then puts frames
    into the queues synchronously instead of through ``call_soon_threadsafe``.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop
        self._lock = threading.Lock()
        self._queues: list[asyncio.Queue] = []
        self._active: asyncio.Queue | None = None
        self._latest_tick: dict | None = None
        self._hello: dict | None = None
        self._pokes: deque[Any] = deque(maxlen=MAX_PENDING_POKES)
        self._commands: deque[dict] = deque(maxlen=MAX_PENDING_COMMANDS)
        self._snap_waits: dict[str, threading.Event] = {}
        self._snap_data: dict[str, bytes] = {}
        # counters (diagnostics for /api/health and the tests)
        self.published_ticks = 0
        self.published_events = 0
        self.dropped_frames = 0

    # ---------------------------------------------------------------- subscriptions (asyncio thread)
    def subscribe(self, maxsize: int = QUEUE_MAXSIZE) -> asyncio.Queue:
        """Register a new client queue (drop-oldest, ``maxsize`` frames) and return it."""
        q: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(maxsize)))
        with self._lock:
            self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Forget a client queue (idempotent)."""
        with self._lock:
            if q in self._queues:
                self._queues.remove(q)
            if self._active is q:
                self._active = None

    def client_count(self) -> int:
        """Number of connected clients (= registered queues)."""
        with self._lock:
            return len(self._queues)

    def note_active_client(self, q: asyncio.Queue) -> None:
        """Mark ``q`` as the most recently active client (SPEC d.9 routes ``snapshot_request`` there).

        Call it from the WS handler after every frame received from that client. Optional: with no
        active client a snapshot request is broadcast to everyone instead.
        """
        with self._lock:
            if q in self._queues:
                self._active = q

    # ---------------------------------------------------------------- publish (sim / worker threads)
    def publish_tick(self, tick: dict) -> None:
        """Keep ``tick`` as the latest state and fan its JSON out once (SPEC d.10)."""
        try:
            text = tick_to_json(tick)
        except ValueError as exc:  # NaN / inf in the tick: never kill the sim thread
            log.warning("state: tick %s not serialisable (%s)", tick.get("seq"), exc)
            return
        with self._lock:
            self._latest_tick = dict(tick)
            self.published_ticks += 1
        self._fanout(Frame(text, tick), targets=None)

    def publish_event(self, msg: dict) -> None:
        """Fan out any non-tick server -> client frame (``event``, ``mood_change``, ``tweet``, ...)."""
        try:
            text = server_json(msg)
        except (TypeError, ValueError) as exc:
            log.warning("state: frame %r not serialisable (%s)", msg.get("type"), exc)
            return
        with self._lock:
            self.published_events += 1
        self._fanout(Frame(text, msg), targets=None)

    def _fanout(self, frame: Frame, targets: Iterable[asyncio.Queue] | None) -> None:
        with self._lock:
            queues = list(self._queues) if targets is None else [q for q in targets if q in self._queues]
        if not queues:
            return
        loop = self.loop
        if loop is None:
            for q in queues:
                self._put(q, frame)
            return
        try:
            loop.call_soon_threadsafe(self._put_many, queues, frame)
        except RuntimeError as exc:  # loop closed during shutdown
            log.debug("state: loop unavailable (%s); frame dropped", exc)

    def _put_many(self, queues: list[asyncio.Queue], frame: Frame) -> None:
        for q in queues:
            self._put(q, frame)

    def _put(self, q: asyncio.Queue, frame: Frame) -> None:
        """Drop-oldest put (SPEC d.10): a full queue loses its oldest frame, never the newest."""
        try:
            q.put_nowait(frame)
            return
        except asyncio.QueueFull:
            pass
        try:
            q.get_nowait()
            self.dropped_frames += 1
        except asyncio.QueueEmpty:  # pragma: no cover - raced with the consumer
            pass
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:  # pragma: no cover - raced with another producer
            self.dropped_frames += 1

    # ---------------------------------------------------------------- latest state
    def latest_tick(self) -> dict | None:
        """The most recent tick dict (``None`` before the first tick)."""
        with self._lock:
            return None if self._latest_tick is None else dict(self._latest_tick)

    def hello(self) -> dict | None:
        """The ``hello`` frame of this session (``None`` until the loop published it)."""
        with self._lock:
            return None if self._hello is None else dict(self._hello)

    def set_hello(self, hello: dict) -> None:
        """Store the ``hello`` frame; new clients get it on accept, ``GET /api/state`` echoes it."""
        with self._lock:
            self._hello = dict(hello)

    # ---------------------------------------------------------------- client -> sim
    def push_poke(self, poke: Any) -> None:
        """Queue a ``flybrain.encoder.Poke`` for the next tick (drops the oldest past the bound)."""
        with self._lock:
            self._pokes.append(poke)

    def push_command(self, cmd: dict) -> None:
        """Queue ``{"name": "set_market_mode"|"clear"|"tweet_test", ...}`` for the next tick."""
        with self._lock:
            self._commands.append(dict(cmd))

    def drain_pokes(self) -> list[Any]:
        """Take every queued poke (sim thread, once per tick)."""
        with self._lock:
            out = list(self._pokes)
            self._pokes.clear()
        return out

    def drain_commands(self) -> list[dict]:
        """Take every queued command (sim thread, once per tick)."""
        with self._lock:
            out = list(self._commands)
            self._commands.clear()
        return out

    # ---------------------------------------------------------------- snapshot round trip
    def request_snapshot(self, id: str, deadline_ms: int = SNAPSHOT_DEADLINE_MS) -> None:
        """Ask the most recently active client for a canvas PNG (SPEC d.9).

        Registers ``id`` as pending and sends one ``snapshot_request`` frame: to the active client when
        one is known, otherwise to every client (the first valid answer wins).
        """
        sid = str(id)
        with self._lock:
            self._snap_waits.setdefault(sid, threading.Event())
            self._snap_data.pop(sid, None)
            targets = [self._active] if self._active is not None else list(self._queues)
        frame = snapshot_request_frame(sid, deadline_ms)
        try:
            self._fanout(Frame(server_json(frame), frame), targets=targets or None)
        except Exception as exc:  # noqa: BLE001 - snapshots are best effort
            log.warning("state: snapshot request %s failed (%s)", sid, exc)

    def wait_snapshot(self, id: str, timeout_s: float) -> bytes | None:
        """Block the calling (worker) thread until the client answers, or ``None`` on timeout."""
        sid = str(id)
        with self._lock:
            ev = self._snap_waits.get(sid)
            if ev is None:
                ev = threading.Event()
                self._snap_waits[sid] = ev
        ev.wait(max(0.0, float(timeout_s)))
        with self._lock:
            self._snap_waits.pop(sid, None)
            data = self._snap_data.pop(sid, None)
        return data  # None on timeout (deliver_snapshot never ran)

    def deliver_snapshot(self, id: str, png: bytes) -> None:
        """Hand a decoded PNG to the waiter (asyncio thread); an unknown / expired id is ignored."""
        sid = str(id)
        with self._lock:
            ev = self._snap_waits.get(sid)
            if ev is None:
                log.debug("state: snapshot %s arrived too late", sid)
                return
            self._snap_data[sid] = bytes(png)
        ev.set()

    def pending_snapshots(self) -> list[str]:
        """Ids currently awaited (diagnostics / tests)."""
        with self._lock:
            return sorted(self._snap_waits)

    def stats(self) -> dict:
        """Counters for ``GET /api/health`` and the tests."""
        with self._lock:
            return {"clients": len(self._queues), "ticks": self.published_ticks,
                    "events": self.published_events, "dropped": self.dropped_frames,
                    "pending_pokes": len(self._pokes), "pending_commands": len(self._commands)}


class TickHistory:
    """Recent ticks plus long-lived event counters (SPEC c.25).

    ``last``/``events`` read a ring sized for ``seconds`` of ticks (10 s by default); ``gf_spikes`` and
    ``jumps`` answer windows up to ``COUNTER_WINDOW_S`` (15 min) from compact ``(wall, n)`` deques,
    because ``agent/summary.py`` asks for ``gf_spikes(600)`` while the tick ring only holds 10 s.
    "Now" is the wall clock of the newest pushed tick (so replays and tests are deterministic).
    """

    def __init__(self, seconds: float = HISTORY_SECONDS, tick_s: float = 0.05) -> None:
        self.seconds = max(0.1, float(seconds))
        self.tick_s = max(1e-3, float(tick_s))
        self._maxlen = int(self.seconds / self.tick_s) + 2
        self._lock = threading.Lock()
        self._ticks: deque[dict] = deque(maxlen=self._maxlen)
        self._events: deque[tuple[float, dict]] = deque(maxlen=20_000)
        self._gf: deque[tuple[float, int]] = deque(maxlen=20_000)
        self._jumps: deque[float] = deque(maxlen=5_000)
        self._now: float = 0.0

    # ---------------------------------------------------------------- write (sim thread)
    def push(self, tick: dict) -> None:
        """Append one tick and fold its events into the counters."""
        wall = _as_float(tick.get("wall"), time.time())
        with self._lock:
            self._ticks.append(tick)
            self._now = max(self._now, wall)
            for ev in tick.get("events") or ():
                if not isinstance(ev, Mapping):
                    continue
                kind = str(ev.get("kind", ""))
                self._events.append((wall, dict(ev)))
                if kind == "gf_spike":
                    data = ev.get("data") if isinstance(ev.get("data"), Mapping) else {}
                    count = int(_as_float(data.get("count"), 1.0)) if data else 1
                    self._gf.append((wall, max(1, count)))
                elif kind == "jump":
                    self._jumps.append(wall)
            self._prune_locked()

    def _prune_locked(self) -> None:
        cutoff = self._now - COUNTER_WINDOW_S
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        while self._gf and self._gf[0][0] < cutoff:
            self._gf.popleft()
        while self._jumps and self._jumps[0] < cutoff:
            self._jumps.popleft()

    # ---------------------------------------------------------------- read (any thread)
    def last(self, seconds: float) -> list[dict]:
        """Ticks of the last ``seconds`` wall seconds, oldest first."""
        window = max(0.0, float(seconds))
        with self._lock:
            cutoff = self._now - window
            return [t for t in self._ticks if _as_float(t.get("wall"), self._now) >= cutoff]

    def events(self, seconds: float) -> list[dict]:
        """In-tick events of the last ``seconds`` wall seconds, oldest first."""
        window = max(0.0, float(seconds))
        with self._lock:
            cutoff = self._now - window
            return [dict(ev) for wall, ev in self._events if wall >= cutoff]

    def gf_spikes(self, seconds: float) -> int:
        """Total DNp01 spikes reported by ``gf_spike`` events in the window."""
        window = max(0.0, float(seconds))
        with self._lock:
            cutoff = self._now - window
            return int(sum(n for wall, n in self._gf if wall >= cutoff))

    def jumps(self, seconds: float) -> int:
        """Number of ``jump`` events in the window (the agent's ``ESCAPE_BURST`` input)."""
        window = max(0.0, float(seconds))
        with self._lock:
            cutoff = self._now - window
            return int(sum(1 for wall in self._jumps if wall >= cutoff))

    def latest(self) -> dict | None:
        """The newest tick, or ``None``."""
        with self._lock:
            return self._ticks[-1] if self._ticks else None

    def mood_history(self, seconds: float | None = None) -> list[tuple[int, str]]:
        """``[(t_ms, state), ...]`` from the ``mood`` events (``GET /api/state.mood_history``)."""
        window = COUNTER_WINDOW_S if seconds is None else max(0.0, float(seconds))
        out: list[tuple[int, str]] = []
        with self._lock:
            cutoff = self._now - window
            for wall, ev in self._events:
                if wall < cutoff or str(ev.get("kind")) != "mood":
                    continue
                data = ev.get("data") if isinstance(ev.get("data"), Mapping) else {}
                to = str(data.get("to", "")) if data else ""
                if to:
                    out.append((int(_as_float(ev.get("t_ms"), 0.0)), to))
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._ticks)


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out else default
