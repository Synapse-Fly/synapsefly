"""``create_app``: the FastAPI application, its lifespan and every route (SPEC section c.28).

The lifespan wires the whole system together once per process::

    setup_logging -> load_connectome -> (gain) -> SpikeMonitor + LIFEngine -> MarketFeed.start()
    -> FeatureExtractor / SensoryEncoder / MotorDecoder / FlyBody / MoodMachine -> StateBus(loop)
    -> TickHistory -> SnapshotBroker -> TweetAgent -> SessionLog -> SimulationLoop.start()

and unwinds it in reverse on shutdown (``loop.stop()``, ``feed.stop()``, ``session_log.close()``,
agent executor ``shutdown(wait=False)``). A module-level lock makes a second concurrent start impossible
(``uvicorn --reload`` spawns two processes; ``scripts/dev.ps1`` never uses ``--reload``) and
``create_app`` hands back the already running application instead of building a second brain.

REST is the read/control surface for the UI and the scripts; ``/ws`` is the live one: ``hello`` once,
then one ``tick`` per wall tick plus the out-of-band frames of SPEC d.3-d.9. Every client gets its own
bounded queue (drop-oldest, size 4, SPEC d.10) and a client whose socket stalls for more than a second
is dropped rather than allowed to slow the simulation down.

Every route is a plain ``def`` so Starlette runs it in its threadpool: the simulation thread and the
agent executor are the only other writers, so a blocking read of their state can never stall the event
loop that feeds the WebSockets.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..config import Settings, gain_value, load_settings, redacted
from ..connectome import loaders as _loaders
from ..connectome.groups import SIDED
from ..connectome.schema import REGIONS
from ..decoder import FlyBody, MotorDecoder
from ..encoder import FeatureExtractor, SensoryEncoder
from ..log import RingHandler, setup_logging
from ..market.feed import MarketFeed
from ..mood import MoodMachine
from ..snn.engine import LIFEngine
from ..snn.monitor import SpikeMonitor
from ..snn.params import LIFParams
from . import session_log as _session_log
from .loop import SimulationLoop
from .protocol import (
    MarketModeRequest,
    PokeRequest,
    SnapshotUpload,
    BadClientFrame,
    error_frame,
    event_frame,
    parse_client,
    pong_frame,
    round_tick,
    server_json,
)
from .state import QUEUE_MAXSIZE, StateBus, TickHistory

__all__ = ["create_app", "build_context", "SimContext", "POKE_RATE_LIMIT", "POKE_RATE_WINDOW_S",
           "SEND_TIMEOUT_S", "MAX_SNAPSHOT_BYTES", "app_running", "reset_start_guard"]

log = logging.getLogger("flybrain.server.app")

#: ``POST /api/poke`` / the WS ``poke`` frame: at most this many pokes per window per client (SPEC c.28/d.7).
POKE_RATE_LIMIT: int = 2
POKE_RATE_WINDOW_S: float = 1.0
#: A WebSocket send that stalls longer than this drops the client (SPEC c.28).
SEND_TIMEOUT_S: float = 1.0
#: Startup gates (SPEC g.6) run in a background thread so the first tick is never delayed.
GATE_THREAD_JOIN_S: float = 2.0
#: SPEC 0.1 ``SeedSequence(FLY_SEED)`` children: [0] connectome [1] engine [2] market [3] decoder wander
#: [4] raster [5] agent [6] encoder episodes.
SEED_CHILDREN: int = 7
#: ``POST /api/snapshot`` / the WS ``snapshot`` frame: 2 MB decoded (SPEC d.7).
MAX_SNAPSHOT_BYTES: int = 2_000_000
_PNG_MAGIC: bytes = b"\x89PNG\r\n\x1a\n"

# Double-start guard (SPEC c.28): one live simulation per process.
_GUARD = threading.Lock()
_LIFESPAN_LOCK = threading.Lock()
_LIVE_APP: FastAPI | None = None


def app_running() -> FastAPI | None:
    """The application whose lifespan is currently live (``None`` when nothing runs)."""
    with _GUARD:
        return _LIVE_APP


def reset_start_guard() -> None:
    """Forget the live application (only for tests that crashed out of a lifespan)."""
    global _LIVE_APP
    with _GUARD:
        _LIVE_APP = None
    if _LIFESPAN_LOCK.locked():  # pragma: no cover - crash recovery
        try:
            _LIFESPAN_LOCK.release()
        except RuntimeError:
            pass


# ==================================================================================================
# context
# ==================================================================================================


@dataclass
class SimContext:
    """Everything one running simulation owns; reachable as ``app.state.ctx`` (never a global)."""

    settings: Settings
    conn: Any
    engine: LIFEngine
    feed: MarketFeed
    features: FeatureExtractor
    encoder: SensoryEncoder
    decoder: MotorDecoder
    body: FlyBody
    mood: MoodMachine
    bus: StateBus
    history: TickHistory
    snapshots: Any
    agent: Any
    session_log: Any | None
    loop: SimulationLoop
    ring: RingHandler | None = None
    started_wall: float = field(default_factory=time.time)
    calibration: dict | None = None
    gates: Any | None = None
    _poke_hits: dict[str, deque] = field(default_factory=dict)
    _poke_lock: threading.Lock = field(default_factory=threading.Lock)

    # -- poke rate limit (SPEC c.28: 2/s per client) ----------------------------------------------
    def allow_poke(self, key: str, now: float | None = None) -> bool:
        """True when ``key`` may spend a poke now; the table only ever holds keys inside the window.

        Every key whose newest hit fell out of ``POKE_RATE_WINDOW_S`` is forgotten on the way through:
        such a key would allow a poke anyway, so dropping it changes no decision and keeps a long-lived
        server from accumulating one dict entry per client IP / per connection forever.
        """
        now = time.monotonic() if now is None else float(now)
        key = str(key)
        with self._poke_lock:
            for stale in [k for k, d in self._poke_hits.items()
                          if k != key and (not d or now - d[-1] > POKE_RATE_WINDOW_S)]:
                del self._poke_hits[stale]
            hits = self._poke_hits.setdefault(key, deque(maxlen=POKE_RATE_LIMIT * 8))
            while hits and now - hits[0] > POKE_RATE_WINDOW_S:
                hits.popleft()
            if len(hits) >= POKE_RATE_LIMIT:
                return False
            hits.append(now)
            return True

    def forget_poke_key(self, key: str) -> None:
        """Drop one client's poke budget (the ``/ws`` handler calls it when the socket closes)."""
        with self._poke_lock:
            self._poke_hits.pop(str(key), None)

    def close(self) -> None:
        """Shut the simulation down in reverse order; every step is independently guarded."""
        for label, fn in (("loop", lambda: self.loop.stop(2.0)),
                          ("feed", self.feed.stop),
                          ("agent", getattr(self.agent, "close", lambda: None)),
                          ("session_log", getattr(self.session_log, "close", lambda: None))):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - shutdown never raises
                log.warning("shutdown: %s failed (%s)", label, exc)


# ==================================================================================================
# build
# ==================================================================================================


def _seed_children(seed: int) -> list[int]:
    """The seven fixed ``SeedSequence`` children of SPEC 0.1 as plain ints."""
    ss = np.random.SeedSequence(int(seed) & 0xFFFFFFFF)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(SEED_CHILDREN)]


def _resolve_gain(settings: Settings, conn: Any) -> tuple[float, dict | None]:
    """``FLY_GAIN`` float, else ``meta['gain_default']`` for calibrated synthetic, else ``calibrate_gain``."""
    explicit = gain_value(settings)
    if explicit is not None:
        return float(explicit), None
    default = float((getattr(conn, "meta", {}) or {}).get("gain_default", 1.0) or 1.0)
    literature = str(getattr(settings, "synth_weights", "calibrated")) == "literature"
    real = str(getattr(settings, "connectome_source", "synthetic")) != "synthetic"
    if not (literature or real):
        return default, None
    try:
        from ..snn.calibrate import calibrate_gain

        doc = calibrate_gain(conn, settings)
        return float(doc.get("gain", default) or default), doc
    except Exception as exc:  # noqa: BLE001 - calibration must never block a boot
        log.warning("calibration failed (%s); using gain %.3f", exc, default)
        return default, None


def _open_session_log(settings: Settings) -> Any | None:
    if not bool(getattr(settings, "session_log", True)):
        return None
    run_id = str(getattr(settings, "run_id", "") or "run")
    path = Path(getattr(settings, "sessions_dir", Path(settings.data_dir) / "sessions"))
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return _session_log.SessionLog(path / f"{run_id}-{stamp}.jsonl", settings)


def _open_replay(settings: Settings) -> Iterator[dict] | None:
    raw = str(getattr(settings, "replay", "") or "")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        from ..config import repo_root

        path = repo_root() / path
    if not path.is_file():
        log.warning("replay: %s does not exist; running live inputs", path)
        return None
    log.info("replay: feeding inputs from %s", path)
    return _session_log.read_session(path)


def _agent_conn_meta(conn: Any) -> dict:
    """``conn_meta`` for the agent (SPEC c.20/c.24): ``Connectome.meta`` plus the identity fields.

    ``meta`` (SPEC c.2) carries ``e``/``license``/``note``/... but **not** ``name``, ``n`` or ``source`` - those are
    attributes of the ``Connectome`` itself. Handing the bare ``meta`` to ``build_brain_summary`` made every tweet
    summary report ``connectome {"name": "", "n": 0}`` to the LLM, while SPEC d.6 shows the real name and size (the
    provenance line that keeps the model from claiming the graph is a real connectome). Attributes win over meta.
    """
    meta = dict(getattr(conn, "meta", None) or {})
    meta["name"] = str(getattr(conn, "name", "") or "")
    meta["n"] = int(getattr(conn, "n", 0) or 0)
    meta["source"] = str(getattr(conn, "source", "") or "synthetic")
    try:
        meta["e"] = int(getattr(conn, "e", meta.get("e", 0)) or 0)
    except Exception:  # pragma: no cover - defensive: a stub connectome without edges
        meta["e"] = int(meta.get("e", 0) or 0)
    return meta


def build_context(settings: Settings) -> SimContext:
    """Everything of SPEC c.28's lifespan list, in order, on the asyncio thread at startup."""
    ring = setup_logging(str(getattr(settings, "log_level", "INFO")))
    seeds = _seed_children(int(getattr(settings, "seed", 1337)))

    t0 = time.perf_counter()
    conn = _loaders.load_connectome(settings)
    log.info("connectome: %s source=%s n=%d e=%d synapses=%.0f (%.2f s)", conn.name, conn.source,
             conn.n, conn.e, float((conn.meta or {}).get("synapses", 0.0) or 0.0),
             time.perf_counter() - t0)

    gain, calib = _resolve_gain(settings, conn)
    monitor = SpikeMonitor(conn, per_region=int(getattr(settings, "raster_per_region", 48)),
                           cap=int(getattr(settings, "raster_cap", 2000)), seed=seeds[4],
                           dt_ms=float(getattr(settings, "dt_ms", 1.0)),
                           tick_s=float(getattr(settings, "tick_s", 0.05)))
    engine = LIFEngine(conn, LIFParams(), seed=int(getattr(settings, "seed", 1337)),
                       backend=str(getattr(settings, "backend", "numpy")),
                       dt_ms=float(getattr(settings, "dt_ms", 1.0)), gain=gain,
                       noise_mu=float(getattr(settings, "noise_mu", 0.5)),
                       noise_sigma=float(getattr(settings, "noise_sigma", 3.5)),
                       drive_mode=str(getattr(settings, "drive_mode", "poisson")), monitor=monitor)
    log.info("engine: backend=%s dt=%.2f ms gain=%.3f noise=(%.2f, %.2f) drive=%s", engine.backend,
             engine.dt_ms, engine.gain, engine.noise_mu, engine.noise_sigma, engine.drive_mode)

    feed = MarketFeed(settings, seed=seeds[2])
    feed.start()
    features = FeatureExtractor(settings, seed=seeds[6])
    encoder = SensoryEncoder(conn, settings, seed=seeds[6], features=features)
    decoder = MotorDecoder(settings, seed=seeds[3])
    body = FlyBody(int(getattr(settings, "canvas_w", 800)), int(getattr(settings, "canvas_h", 500)),
                   str(getattr(settings, "walls", "bounce")), seed=seeds[3])
    mood = MoodMachine(tick_s=float(getattr(settings, "tick_s", 0.05)))

    try:
        running = asyncio.get_running_loop()
    except RuntimeError:  # pragma: no cover - headless use (scripts)
        running = None
    bus = StateBus(running)
    history = TickHistory(seconds=10.0, tick_s=float(getattr(settings, "tick_s", 0.05)))

    from ..agent.llm import TweetGenerator
    from ..agent.orchestrator import TweetAgent
    from ..agent.snapshot import SnapshotBroker
    from ..agent.x_client import XClient

    snapshots = SnapshotBroker(bus, body, settings)
    agent = TweetAgent(settings, TweetGenerator(settings, seed=seeds[5]), XClient(settings), snapshots,
                       bus, history, _agent_conn_meta(conn), seed=seeds[5])

    replay = _open_replay(settings)
    slog = None if replay is not None else _open_session_log(settings)
    loop = SimulationLoop(settings, conn, engine, feed, features, encoder, decoder, body, mood, agent,
                          bus, history, slog, replay=replay)
    ctx = SimContext(settings=settings, conn=conn, engine=engine, feed=feed, features=features,
                     encoder=encoder, decoder=decoder, body=body, mood=mood, bus=bus, history=history,
                     snapshots=snapshots, agent=agent, session_log=slog, loop=loop, ring=ring,
                     calibration=calib)
    if calib is not None:
        report = calib.get("report") if isinstance(calib, dict) else None
        bus.publish_event(event_frame("calibration", {"gain": float(engine.gain), "report": report}))
        log.info("calibration: gain %.3f (%s)", float(engine.gain),
                 "cached" if calib.get("cached") else "bisected")
    return ctx


def _log_boot_gates(ctx: SimContext) -> None:
    """Run ``snn.calibrate.run_gates`` once at boot and log the five gates of SPEC g.6 as info lines."""
    try:
        from ..snn.calibrate import GATE_NAMES, run_gates

        report = run_gates(ctx.conn, ctx.settings, float(ctx.engine.gain), tonic=True)
        ctx.gates = report
        log.info("gates: %s", " ".join(f"{name}={'PASS' if report.passed.get(name) else 'FAIL'}"
                                      for name in GATE_NAMES))
        log.info("gates: rest %.2f Hz (active_frac %.4f), mn9 %.1f Hz, gf_latency %s ms, "
                 "a02_diff %.1f Hz, rtf %.2f, gain %.3f", report.rest_rate_hz, report.rest_active_frac,
                 report.mn9_hz, "none" if report.gf_latency_ms is None else f"{report.gf_latency_ms:.1f}",
                 report.a02_diff_hz, report.rtf, report.gain)
        for note in report.notes:
            log.info("gates: note: %s", note)
        if not report.ok():
            log.warning("gates: %d gate(s) FAILED - see scripts/selftest.py",
                        sum(1 for v in report.passed.values() if not v))
    except Exception as exc:  # noqa: BLE001 - diagnostics never break a boot
        log.warning("gates: could not run boot gates (%s)", exc)


# ==================================================================================================
# the app
# ==================================================================================================


def create_app(settings: Settings | None = None, *, boot_gates: bool = True) -> FastAPI:
    """Build the FastAPI app (SPEC c.28); the simulation starts with the lifespan.

    Returns the already running application when one exists in this process (the double-start guard):
    one process owns exactly one brain. ``boot_gates=False`` skips the background SPEC g.6 gate run.
    """
    live = app_running()
    if live is not None:
        log.warning("create_app: a simulation is already running in this process; returning that app")
        return live

    cfg = settings if settings is not None else load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _LIVE_APP
        if not _LIFESPAN_LOCK.acquire(blocking=False):
            raise RuntimeError("a FlyBrain simulation is already running in this process "
                               "(never start uvicorn with --reload)")
        ctx: SimContext | None = None
        gate_thread: threading.Thread | None = None
        try:
            with _GUARD:
                _LIVE_APP = app
            ctx = build_context(cfg)
            app.state.ctx = ctx
            ctx.loop.start()
            ctx.loop.wait_ready(5.0)
            if boot_gates:
                gate_thread = threading.Thread(target=_log_boot_gates, args=(ctx,), name="flybrain-gates",
                                               daemon=True)
                gate_thread.start()
            log.info("server: up run_id=%s port=%s", cfg.run_id, getattr(cfg, "port", "?"))
            yield
        finally:
            app.state.ctx = None
            if ctx is not None:
                ctx.close()
            if gate_thread is not None and gate_thread.is_alive():
                gate_thread.join(GATE_THREAD_JOIN_S)
            with _GUARD:
                _LIVE_APP = None
            _LIFESPAN_LOCK.release()
            log.info("server: down")

    app = FastAPI(title="SynapseFly / FlyBrain", version="0.1.0", lifespan=lifespan)
    app.state.ctx = None
    app.state.settings = cfg
    origins = list(getattr(cfg, "cors_origins", ("http://localhost:3000",)))
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
    _register_routes(app)
    return app


def _ctx_of(app: FastAPI) -> SimContext:
    ctx = getattr(app.state, "ctx", None)
    if ctx is None:
        raise HTTPException(status_code=503, detail="simulation not started")
    return ctx


def _register_routes(app: FastAPI) -> None:
    # ---------------------------------------------------------------- GET /api/health
    @app.get("/api/health")
    def health() -> dict:
        """Liveness plus everything an operator needs in one call (SPEC c.28)."""
        ctx = _ctx_of(app)
        conn, eng, lp = ctx.conn, ctx.engine, ctx.loop
        meta = dict(conn.meta or {})
        agent_status: dict = {}
        try:
            agent_status = ctx.agent.status() if ctx.agent is not None else {}
        except Exception as exc:  # noqa: BLE001
            log.warning("health: agent status failed (%s)", exc)
        try:
            market = dict(ctx.feed.status())
        except Exception as exc:  # noqa: BLE001
            log.warning("health: market status failed (%s)", exc)
            market = {"mode": ctx.feed.mode, "ok": False, "last_poll": None, "failures": 0}
        return {
            "ok": bool(lp.is_alive() and not lp.stopped and lp.seq > 0),
            "uptime_s": round(time.time() - ctx.started_wall, 3),
            "run_id": str(getattr(ctx.settings, "run_id", "")),
            "rtf": round(float(lp.rtf), 3),
            "speed": round(float(lp.speed), 3),
            "seq": int(lp.seq),
            "t_ms": int(lp.t_ms),
            "backend": str(eng.backend),
            "connectome": {"name": str(conn.name), "source": str(conn.source), "n": int(conn.n),
                           "e": int(conn.e), "gain": float(eng.gain),
                           "license": str(meta.get("license", "") or "")},
            "market": {"mode": str(market.get("mode", ctx.feed.mode)), "ok": bool(market.get("ok", True)),
                       "last_poll": market.get("last_poll"), "failures": int(market.get("failures", 0))},
            "agent": {"llm": str(agent_status.get("llm", "dryrun")),
                      "x": str(agent_status.get("x", "dryrun")),
                      "tweets_today": int(agent_status.get("tweets_today", 0)),
                      "last_tweet_wall": agent_status.get("last_tweet_wall"),
                      "disabled_reason": agent_status.get("disabled_reason")},
            "clients": int(ctx.bus.client_count()),
            "log_tail": ctx.ring.tail(50) if ctx.ring is not None else [],
        }

    # ---------------------------------------------------------------- GET /api/state
    @app.get("/api/state")
    def state() -> dict:
        """``hello`` + the latest tick + the trail ring + the mood history + the last 50 events."""
        ctx = _ctx_of(app)
        hello = ctx.bus.hello() or ctx.loop.hello()
        if hello is None:
            raise HTTPException(status_code=503, detail="simulation not ready")
        # The tick goes through the SPEC d rounding pass, exactly like the WS frame (protocol.tick_to_json
        # does the same): a client that hydrates from REST and then switches to /ws must not see the
        # float precision change mid-stream, and the rounded payload is a third of the size.
        tick = ctx.bus.latest_tick() or ctx.loop.latest()
        return {"hello": hello, "tick": None if tick is None else round_tick(tick),
                "trail": ctx.snapshots.trail_wire() if ctx.snapshots is not None else [],
                "mood_history": [[int(t), str(s)] for t, s in ctx.history.mood_history()],
                "events": ctx.loop.recent_events(50)}

    # ---------------------------------------------------------------- GET /api/config
    @app.get("/api/config")
    def config() -> dict:
        """``redacted(settings)``: never a credential (SPEC c.1)."""
        ctx = getattr(app.state, "ctx", None)
        return redacted(ctx.settings if ctx is not None else app.state.settings)

    # ---------------------------------------------------------------- GET /api/connectome/groups
    @app.get("/api/connectome/groups")
    def groups() -> dict:
        """``{group: {n, types: [<=20], sided}}`` over every resolved group key."""
        ctx = _ctx_of(app)
        conn = ctx.conn
        out: dict[str, dict] = {}
        for name in sorted(conn.groups):
            idx = np.asarray(conn.groups[name], dtype=np.int64)
            labels: list[str] = []
            if idx.size:
                seen = np.unique(np.asarray(conn.type_idx)[idx])
                labels = [conn.types[int(t)] for t in seen[:20] if conn.types[int(t)]]
            base = name[:-2] if name.endswith(("_L", "_R")) else name
            out[name] = {"n": int(idx.size), "types": labels, "sided": bool(base in SIDED)}
        return out

    # ---------------------------------------------------------------- GET /api/neurons
    @app.get("/api/neurons")
    def neurons(group: str = Query(default="gf"), limit: int = Query(default=100, ge=1, le=5000)) -> list:
        """``[{id, body_id, type, region, side}]`` for one group (``region``/``side`` as names).

        SPEC c.28 does not type the row values. This route is the human/introspection surface (a curl or
        a debug panel), so ``region`` is the ``REGIONS`` *name* and ``side`` the letter ``L``/``R``/``M``
        - unlike ``hello.raster.rows[].region``, which is the index into ``hello.regions`` because it is
        read 384 times per frame by the raster drawing code (SPEC d.1/e.4).
        """
        ctx = _ctx_of(app)
        conn = ctx.conn
        idx = conn.groups.get(group)
        if idx is None:
            raise HTTPException(status_code=404, detail=f"unknown group {group!r}")
        side_name = {-1: "L", 1: "R", 0: "M"}
        rows: list[dict] = []
        for i in np.asarray(idx, dtype=np.int64)[:int(limit)]:
            j = int(i)
            rows.append({"id": j, "body_id": int(conn.body_id[j]), "type": conn.type_of(j),
                         "region": REGIONS[int(conn.region[j])],
                         "side": side_name.get(int(conn.side[j]), "M")})
        return rows

    # ---------------------------------------------------------------- GET /api/tweets
    @app.get("/api/tweets")
    def tweets(limit: int = Query(default=20, ge=1, le=500)) -> list:
        """The last tweet records, dry-run included (SPEC d.4 + ``summary``)."""
        ctx = _ctx_of(app)
        if ctx.agent is None:
            return []
        try:
            return list(ctx.agent.history(int(limit)))
        except Exception as exc:  # noqa: BLE001
            log.warning("tweets: history failed (%s)", exc)
            return []

    # ---------------------------------------------------------------- POST /api/poke
    @app.post("/api/poke")
    def poke(body: PokeRequest, request: Request) -> dict:
        """Queue one sensory poke for the next tick; 2 per second per client IP (SPEC c.28)."""
        ctx = _ctx_of(app)
        client = request.client.host if request.client is not None else "unknown"
        if not ctx.allow_poke(f"ip:{client}"):
            raise HTTPException(status_code=429, detail="rate_limited: at most 2 pokes per second")
        frame = body.as_frame()
        ctx.bus.push_poke(frame.to_poke(ctx.loop.t_ms))
        ctx.bus.publish_event(event_frame("poke", frame.to_event_data(), ctx.loop.seq, ctx.loop.t_ms))
        return {"ok": True}

    # ---------------------------------------------------------------- POST /api/market/mode
    @app.post("/api/market/mode")
    def market_mode(body: MarketModeRequest) -> dict:
        """``sim`` / ``dexscreener`` / a forced sim regime for 60 s (SPEC d.7)."""
        ctx = _ctx_of(app)
        if body.mode == "dexscreener" and not str(getattr(ctx.settings, "token_address", "") or ""):
            raise HTTPException(status_code=403, detail="forbidden: FLY_TOKEN_ADDRESS is empty")
        ctx.bus.push_command({"name": "set_market_mode", "mode": body.mode})
        return {"ok": True, "mode": body.mode}

    # ---------------------------------------------------------------- POST /api/clear
    @app.post("/api/clear")
    def clear() -> dict:
        """Reset the trail ring and publish ``clear`` with ``{"by": "api"}`` (SPEC d.8).

        SPEC d.8 types the ``clear`` event as ``{by: "client"|"api"}``; ``"api"`` is only reachable
        through a REST route, which the c.28 list omits while ``scripts/mock_ws.py`` (the mock the
        frontend is developed against) serves it. Same body and reply as the mock: ``{"ok": true}``.
        """
        ctx = _ctx_of(app)
        ctx.bus.push_command({"name": "clear", "by": "api"})
        return {"ok": True}

    # ---------------------------------------------------------------- POST /api/snapshot
    @app.post("/api/snapshot")
    def snapshot(body: SnapshotUpload) -> dict:
        """REST alternative to the WS ``snapshot`` reply: hand a browser PNG to the waiting agent."""
        ctx = _ctx_of(app)
        try:
            raw = base64.b64decode(body.png_b64, validate=False)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="png_b64 is not valid base64") from None
        if len(raw) > MAX_SNAPSHOT_BYTES:
            raise HTTPException(status_code=413, detail=f"png too large ({len(raw)} bytes)")
        if raw[:8] != _PNG_MAGIC:
            raise HTTPException(status_code=400, detail="payload is not a PNG")
        ctx.bus.deliver_snapshot(body.id, raw)
        return {"ok": True, "bytes": len(raw)}

    # ---------------------------------------------------------------- POST /api/tweet/test
    @app.post("/api/tweet/test")
    def tweet_test() -> dict:
        """Fire the agent once with reason ``manual`` (honours the dry-run flags, bypasses cooldowns)."""
        ctx = _ctx_of(app)
        if ctx.agent is None:
            raise HTTPException(status_code=503, detail="agent not available")
        before = int(getattr(ctx.agent, "fires", 0))
        ctx.agent.request_manual()
        ctx.agent.wait_idle(30.0)
        record = getattr(ctx.agent, "last_record", None)
        if record is None or int(getattr(ctx.agent, "fires", 0)) == before:
            raise HTTPException(status_code=429, detail="agent suppressed the manual tweet "
                                                       "(daily cap, in flight or disabled)")
        return dict(record)

    # ---------------------------------------------------------------- WS /ws
    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        """``hello`` once, then ticks and out-of-band frames; accepts the client frames of SPEC d.7."""
        ctx = getattr(app.state, "ctx", None)
        if ctx is None:
            await socket.close(code=1013)  # try again later
            return
        await socket.accept()
        queue = ctx.bus.subscribe(QUEUE_MAXSIZE)
        # One fresh identity per connection for the poke budget: ``id(queue)`` is recycled by CPython
        # once the queue is collected, so a new client could inherit a previous one's spent budget.
        client_key = f"ws:{uuid.uuid4().hex}"
        try:
            hello = ctx.bus.hello() or ctx.loop.hello()
            if hello is not None:
                await socket.send_text(server_json(hello))
            sender = asyncio.create_task(_send_loop(socket, queue), name="ws-send")
            receiver = asyncio.create_task(_recv_loop(socket, ctx, queue, client_key), name="ws-recv")
            done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                    log.info("ws: %s ended with %s: %s", task.get_name(), type(exc).__name__, exc)
        finally:
            ctx.bus.unsubscribe(queue)
            ctx.forget_poke_key(client_key)
            try:
                await socket.close()
            except Exception:  # noqa: BLE001 - already closed
                pass


async def _send_loop(socket: WebSocket, queue: asyncio.Queue) -> None:
    """Drain this client's queue; a send that stalls > 1 s drops the client (SPEC c.28)."""
    while True:
        frame = await queue.get()
        try:
            await asyncio.wait_for(socket.send_text(str(frame)), SEND_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.info("ws: client send stalled > %.1fs; dropping it", SEND_TIMEOUT_S)
            return
        except (WebSocketDisconnect, RuntimeError):
            return


async def _recv_loop(socket: WebSocket, ctx: SimContext, queue: asyncio.Queue,
                     limiter_key: str) -> None:
    """Parse and apply client frames; a bad frame answers ``error`` and keeps the connection open."""
    while True:
        # ``receive()`` rather than ``receive_text()``: a binary frame must answer ``bad_message`` with
        # the connection open (SPEC d preamble), and ``receive_text`` raises KeyError on one, which
        # would kill this task and drop the client silently. ``parse_client`` accepts bytes (SPEC c.26).
        try:
            message = await socket.receive()
        except WebSocketDisconnect:  # pragma: no cover - receive() returns the disconnect message
            return
        except RuntimeError:  # disconnect message already consumed
            return
        if str(message.get("type", "")) != "websocket.receive":
            return                                    # websocket.disconnect
        raw: Any = message.get("text")
        if raw is None:
            raw = message.get("bytes")
        if raw is None:  # pragma: no cover - ASGI always carries one of the two
            await _reply(socket, error_frame("bad_message", "empty frame"))
            continue
        ctx.bus.note_active_client(queue)
        try:
            msg = parse_client(raw)
        except BadClientFrame as exc:
            await _reply(socket, exc.to_frame())
            continue
        except ValueError as exc:  # pragma: no cover - parse_client always raises BadClientFrame
            await _reply(socket, error_frame("bad_message", str(exc)))
            continue
        kind = getattr(msg, "type", "")
        try:
            if kind == "ping":
                await _reply(socket, pong_frame(float(msg.t), int(ctx.loop.seq)))
            elif kind == "poke":
                if not ctx.allow_poke(limiter_key):
                    await _reply(socket, error_frame("rate_limited", "at most 2 pokes per second"))
                    continue
                ctx.bus.push_poke(msg.to_poke(ctx.loop.t_ms))
                ctx.bus.publish_event(event_frame("poke", msg.to_event_data(), ctx.loop.seq,
                                                  ctx.loop.t_ms))
            elif kind == "set_market_mode":
                if msg.mode == "dexscreener" and not str(getattr(ctx.settings, "token_address", "") or ""):
                    await _reply(socket, error_frame("forbidden", "FLY_TOKEN_ADDRESS is empty"))
                    continue
                ctx.bus.push_command({"name": "set_market_mode", "mode": msg.mode})
            elif kind == "clear":
                ctx.bus.push_command({"name": "clear", "by": "client"})
            elif kind == "tweet_test":
                ctx.bus.push_command({"name": "tweet_test"})
            elif kind == "snapshot":
                ctx.snapshots.deliver(msg.id, msg.png_b64)
            else:  # pragma: no cover - parse_client rejects unknown types
                await _reply(socket, error_frame("bad_message", f"unhandled type {kind!r}"))
        except Exception as exc:  # noqa: BLE001 - a client can never break the server
            log.warning("ws: handling %r failed (%s)", kind, exc)
            await _reply(socket, error_frame("bad_message", f"{type(exc).__name__}: {exc}"))


async def _reply(socket: WebSocket, frame: dict) -> None:
    try:
        await asyncio.wait_for(socket.send_text(server_json(frame)), SEND_TIMEOUT_S)
    except (asyncio.TimeoutError, WebSocketDisconnect, RuntimeError):  # pragma: no cover
        pass
