"""``SimulationLoop``: the integration spine (SPEC section c.27).

One thread owns the whole brain pipeline and runs it at a fixed wall-clock rate::

    market feed -> FeatureExtractor -> SensoryEncoder -> LIFEngine -> MotorDecoder -> FlyBody
                -> MoodMachine -> TweetAgent trigger -> StateBus (-> WebSocket) / SessionLog

Timing contract (SPEC section 0 "Scheduler" and c.27 step 12): wall-clock ticks at ``tick_hz`` (20 Hz by
default) that are **never skipped**; the brain time simulated per tick is ``steps_per_tick * dt_ms *
speed`` with ``speed`` in ``[0.25, 1.0]`` adapting to the measured compute cost; ``speed`` and ``rtf`` are
broadcast in every tick. ``FLY_REALTIME=0`` pins ``speed = 1.0``, runs exactly ``steps_per_tick`` steps
per tick and never sleeps (tests, replay, headless scripts).

Threading (SPEC 0.1): this thread never blocks on I/O. Market HTTP lives in ``MarketFeed``'s own thread,
agent work on the ``TweetAgent``'s 1-worker executor, WebSocket fan-out on uvicorn's asyncio loop; the
only hand-off is the ``StateBus``. Every external failure becomes an ``event`` frame plus a log line and
never stops the simulation: the body of one tick is wrapped, and a failing tick is counted and skipped.

Determinism: with ``FLY_MARKET=sim`` and the numpy backend the run is a pure function of ``FLY_SEED``.
Wall time would break that, so a non-realtime run uses a **virtual clock** aligned to whole
milliseconds (``wall0_ms + seq * tick_ms``); the session log can therefore round-trip it losslessly and
``replay=read_session(path)`` reproduces a run tick for tick (``scripts/replay.py``, SPEC h.4).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..connectome.schema import REGIONS
from ..decoder import FlyBody, InkStyle, Kinematics, MotorDecoder, Readouts
from ..encoder import POKE_TABLE, FeatureExtractor, Features, Poke, SensoryEncoder
from ..market.base import MarketSnapshot, Trade
from ..market.feed import MARKET_MODES, MarketFeed
from ..mood import MoodInputs, MoodMachine, MoodState, Transition
from ..snn.engine import LIFEngine, apply_tonic_table
from ..snn.monitor import StepStats
from .protocol import MOOD_STATES, PROTOCOL_VERSION
from .state import StateBus, TickHistory

__all__ = ["SimulationLoop", "SPEED_MIN", "SPEED_MAX", "SPEED_DOWN_FRAC", "SPEED_UP_FRAC",
           "SPEED_UP_TICKS", "MIN_STEPS", "MAX_MARKET_TRADES", "EVENT_RING", "WALL_BUMP_RATE_HZ",
           "WALL_BUMP_MS"]

log = logging.getLogger("flybrain.server.loop")

#: ``speed`` bounds: brain ms simulated per wall ms (SPEC section 0 "Scheduler").
SPEED_MIN: float = 0.25
SPEED_MAX: float = 1.0
#: Compute above this fraction of the tick budget slows the brain clock down (SPEC c.27 step 12).
SPEED_DOWN_FRAC: float = 0.8
SPEED_DOWN_FACTOR: float = 0.85
#: ... and below this fraction, for ``SPEED_UP_TICKS`` consecutive ticks, speeds it back up.
SPEED_UP_FRAC: float = 0.45
SPEED_UP_FACTOR: float = 1.1
SPEED_UP_TICKS: int = 40
#: Never run fewer than this many LIF steps per tick, whatever ``speed`` says (SPEC c.27 step 5).
MIN_STEPS: int = 5
#: ``market`` frame trade budget (SPEC d.5: "<= 200, oldest dropped").
MAX_MARKET_TRADES: int = 200
#: ``GET /api/state.events`` keeps the last 50 events (SPEC c.28).
EVENT_RING: int = 50
#: Wall-bump steering pulse ``[E]`` (SPEC c.27 step 6).
WALL_BUMP_RATE_HZ: float = 40.0
WALL_BUMP_MS: float = 100.0
#: ``mood_change`` / in-tick ``mood`` events and the agent trigger share these names.
_TWEET_MOODS: frozenset[str] = frozenset({"EUPHORIA", "PANIC", "COURTSHIP"})


def _side_away(bump_side: str) -> str:
    """Wall on the left -> steer right (SPEC c.27 step 6: ``steer_a02_<side away from wall>``)."""
    return "steer_a02_R" if str(bump_side).upper() == "L" else "steer_a02_L"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and out not in (float("inf"), float("-inf")) else default


class SimulationLoop(threading.Thread):
    """The simulation thread (SPEC c.27).

    Every collaborator is injected (the app lifespan builds them, SPEC c.28) so the loop can also be
    driven headless by ``scripts/selftest.py`` / ``scripts/replay.py`` with a ``loop=None`` ``StateBus``.
    ``replay`` is an iterator of ``session_log.read_session`` rows: when given, the market snapshot,
    trades, pokes and commands of every tick come from the log instead of the live feed and the loop
    stops when the log ends.
    """

    def __init__(self, settings: Any, conn: Any, engine: LIFEngine, feed: MarketFeed,
                 features: FeatureExtractor, encoder: SensoryEncoder, decoder: MotorDecoder,
                 body: FlyBody, mood: MoodMachine, agent: Any | None, bus: StateBus,
                 history: TickHistory, session_log: Any | None,
                 replay: Iterator[dict] | None = None) -> None:
        super().__init__(name="flybrain-sim", daemon=True)
        self.settings = settings
        self.conn = conn
        self.engine = engine
        self.feed = feed
        self.features = features
        self.encoder = encoder
        self.decoder = decoder
        self.body = body
        self.mood = mood
        self.agent = agent
        self.bus = bus
        self.history = history
        self.session_log = session_log
        self.replay = replay

        self.realtime: bool = bool(getattr(settings, "realtime", True)) and replay is None
        self.tick_ms: float = float(getattr(settings, "tick_ms", 50.0))
        self.tick_s: float = self.tick_ms / 1000.0
        self.tick_ms_int: int = max(1, int(round(self.tick_ms)))
        self.dt_ms: float = float(getattr(engine, "dt_ms", 1.0))
        self.steps_per_tick: int = max(1, int(getattr(settings, "steps_per_tick", 50)))

        #: sim-ms per wall-ms, ``[0.25, 1.0]`` (SPEC c.27). Pinned to 1.0 when ``FLY_REALTIME=0``.
        self.speed: float = SPEED_MAX
        self.seq: int = 0
        #: Sequence number of the tick currently being computed (``seq + 1`` once it succeeds). A tick
        #: that raises never reaches ``self.seq``, so this is the only honest number to report for it.
        self.attempted_seq: int = 0
        self.rtf: float = 0.0
        self.step_ms: float = 0.0
        self.compute_ms: float = 0.0
        self.started_wall: float = time.time()
        self.n_errors: int = 0
        self.last_error: str | None = None
        self.replay_diffs: list[tuple[int, int, int]] = []
        self.replay_ticks: int = 0

        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._hello: dict | None = None
        self._fast_run: int = 0
        self._wall0_ms: int = int(time.time() * 1000.0)
        self._readouts: Readouts = Readouts.zeros()
        self._last_trade: Trade | None = None
        self._pending_trades: list[Trade] = []
        self._last_market_seq: int = -1
        self._events: deque[dict] = deque(maxlen=EVENT_RING)

    # ================================================================= introspection

    @property
    def t_ms(self) -> int:
        """Brain time at the end of the last completed tick (ms)."""
        return int(self.engine.t_ms)

    @property
    def brain_ms_per_tick(self) -> float:
        """Brain ms simulated by the next tick at the current ``speed``."""
        return self._steps_for_tick() * self.dt_ms

    def latest(self) -> dict | None:
        """The most recent tick dict (``None`` before the first tick)."""
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def hello(self) -> dict | None:
        """The ``hello`` frame of this session (built once at startup)."""
        with self._lock:
            return None if self._hello is None else dict(self._hello)

    def recent_events(self, limit: int = EVENT_RING) -> list[dict]:
        """The last events of this session, in-tick and out-of-band (``GET /api/state.events``)."""
        with self._lock:
            rows = list(self._events)
        return rows[-max(0, int(limit)):]

    def wait_ready(self, timeout_s: float = 5.0) -> bool:
        """Block until the first tick has been published (tests / ``GET /api/state``)."""
        return self._ready.wait(max(0.0, float(timeout_s)))

    def stats(self) -> dict:
        """Counters for ``GET /api/health``."""
        return {"uptime_s": round(time.time() - self.started_wall, 3), "rtf": round(self.rtf, 3),
                "speed": round(self.speed, 3), "seq": int(self.seq), "t_ms": int(self.t_ms),
                "errors": int(self.n_errors), "last_error": self.last_error,
                "backend": str(getattr(self.engine, "backend", "numpy"))}

    # ================================================================= thread lifecycle

    def stop(self, timeout_s: float = 2.0) -> None:
        """Ask the thread to finish the current tick and join it (idempotent)."""
        self._stop_event.set()
        if self.is_alive():
            self.join(max(0.0, float(timeout_s)))
            if self.is_alive():  # pragma: no cover - a wedged tick must not wedge shutdown
                log.warning("loop: thread still alive after %.1fs", timeout_s)

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:
        """Startup (tonic table + hello), then one iteration per wall tick until ``stop()``."""
        try:
            self.startup()
        except Exception as exc:  # noqa: BLE001 - a broken startup must not kill the process
            self.n_errors += 1
            self.last_error = f"startup: {type(exc).__name__}: {exc}"
            log.exception("loop: startup failed (%s)", exc)
            return
        deadline = time.perf_counter()
        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                self.tick_once()
            except StopIteration:
                log.info("loop: replay exhausted after %d tick(s)", self.replay_ticks)
                self._stop_event.set()
                break
            except Exception as exc:  # noqa: BLE001 - SPEC 0.1: errors never stop the sim
                self._note_tick_error(exc)
            self.compute_ms = (time.perf_counter() - t0) * 1000.0
            if not self.realtime:
                continue
            self._adapt_speed(self.compute_ms)
            deadline += self.tick_s
            now = time.perf_counter()
            if now < deadline:
                self._stop_event.wait(deadline - now)
            elif now - deadline > self.tick_s:
                # chronically late: resync the grid instead of accumulating a backlog. Ticks are never
                # skipped (SPEC c.27 step 12), the brain clock simply slows down via ``speed``.
                deadline = now
        broker = getattr(self.agent, "snapshots", None) if self.agent is not None else None
        if broker is not None:
            try:
                broker.save_trail(force=True)  # final flush so a graceful stop persists the latest painting
            except Exception as exc:  # noqa: BLE001
                log.debug("loop: final trail save failed (%s)", exc)
        log.info("loop: stopped after %d tick(s), t_ms=%d, %d error(s)", self.seq, self.t_ms, self.n_errors)

    def _note_tick_error(self, exc: BaseException) -> None:
        """Count and report a failed tick under the sequence number it was *attempting* (SPEC 0.1).

        ``self.seq`` only advances on a tick that completed, so reporting it here would name a tick that
        *succeeded* and make consecutive failures indistinguishable in ``/api/health.log_tail``.
        """
        self.n_errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        log.exception("loop: tick %d failed (%s)", self.attempted_seq, exc)

    # ================================================================= startup

    def startup(self) -> dict:
        """Apply the tonic-current table (SPEC c.10) and publish ``hello`` (SPEC c.27)."""
        applied = apply_tonic_table(self.engine)
        log.info("loop: tonic currents applied: %s",
                 ", ".join(f"{k}={v}" for k, v in sorted(applied.items())) or "none")
        missing = sorted(k for k, v in applied.items() if v == 0)
        if missing:
            log.warning("loop: tonic groups empty in this connectome: %s", ", ".join(missing))
        hello = self.build_hello()
        with self._lock:
            self._hello = hello
        self.bus.set_hello(hello)
        try:
            if self.bus.client_count() > 0:  # clients that connected before the loop started
                self.bus.publish_event(hello)
        except Exception as exc:  # noqa: BLE001
            log.debug("loop: could not publish hello (%s)", exc)
        log.info("loop: ready run_id=%s n=%d e=%d backend=%s dt=%.2fms tick=%.1fms realtime=%s",
                 str(getattr(self.settings, "run_id", "")), int(self.conn.n), int(self.conn.e),
                 self.engine.backend, self.dt_ms, self.tick_ms, self.realtime)
        return hello

    # ================================================================= one tick

    def _steps_for_tick(self) -> int:
        if not self.realtime:
            return self.steps_per_tick
        return max(MIN_STEPS, int(round(self.steps_per_tick * self.speed)))

    def _now(self) -> float:
        """Wall seconds for this tick: the real clock in realtime mode, else a ms-aligned virtual clock.

        Deliberate deviation from SPEC c.27 step 1 (``now = time.time()``), scoped to ``FLY_REALTIME=0``:
        ``now`` feeds ``feed.poll`` and every wall-clock decay in ``FeatureExtractor``, so a real clock
        would make a headless run non-reproducible and the session log impossible to replay bit-exactly
        (SPEC h.4 / ``test_replay_bit_exact``). The consequence to know about is that ``tick.wall`` of a
        non-realtime run advances one ``tick_ms`` per tick regardless of how fast the CPU produced it, so
        it runs ahead of ``hello.server_wall`` in proportion to the RTF. Realtime runs - every run a
        browser ever sees - use ``time.time()`` and are unaffected.
        """
        if self.realtime:
            return time.time()
        return (self._wall0_ms + self.seq * self.tick_ms_int) / 1000.0

    def tick_once(self) -> dict:
        """Run exactly one tick and return its dict (also the unit of ``scripts/replay.py``)."""
        body_t0 = time.perf_counter()
        seq = self.attempted_seq = self.seq + 1
        t0_ms = int(self.engine.t_ms)
        row: dict | None = None
        if self.replay is not None:
            row = self._next_replay_row()          # raises StopIteration at the end of the log
            now = _as_float(row.get("wall"), self._now())
        else:
            now = self._now()

        # -- 1. client -> sim -------------------------------------------------------------------
        pokes = list(self.bus.drain_pokes())
        commands = list(self.bus.drain_commands())
        if row is not None:
            pokes = self._replay_pokes(row)
            commands = [dict(c) for c in (row.get("commands") or ())]
        oob: list[dict] = []
        for cmd in commands:
            self._apply_command(cmd, t0_ms, now, oob)

        # -- 2. market --------------------------------------------------------------------------
        snap, trades, mkt_events = self._poll_market(now, row)
        oob.extend(mkt_events)
        if trades:
            self._last_trade = trades[-1]

        # -- 3. features ------------------------------------------------------------------------
        r_prev = self._readouts
        mood_name = self.mood.state.state.value
        f = self.features.update(snap, trades, now, self.tick_s, mood_name, bool(self.decoder.feeding),
                                 float(r_prev.mn9), pokes=pokes, t_ms=t0_ms)
        oob.extend(self._drain_feature_events())

        # -- 4. encode + inject -----------------------------------------------------------------
        steps = self._steps_for_tick()
        brain_ms = steps * self.dt_ms
        brain_s = brain_ms / 1000.0
        drives = self.encoder.encode(f, float(self.body.kin.heading), mood_name, pokes, t0_ms,
                                     scores=self.mood.state.to_wire())
        self.encoder.apply(self.engine, drives, brain_ms)

        # -- 5. step the brain ------------------------------------------------------------------
        stats = self.engine.step(steps)
        t_ms = int(self.engine.t_ms)
        self.step_ms = float(stats.step_ms_mean)

        # -- 6. decode + integrate --------------------------------------------------------------
        r = Readouts.from_stats(stats)
        self._readouts = r
        cmd = self.decoder.update(r, f, mood_name, t_ms, brain_s)
        kin, wall_events = self.body.integrate(cmd, brain_s, t_ms)
        self.decoder.observe(kin)
        events: list[dict] = list(cmd.events) + list(wall_events)
        for ev in wall_events:
            if ev.get("kind") == "wall_bump":
                group = _side_away(str((ev.get("data") or {}).get("side", "L")))
                try:
                    self.engine.inject(group, rate_hz=WALL_BUMP_RATE_HZ, duration_ms=WALL_BUMP_MS, tag="wall")
                except Exception as exc:  # noqa: BLE001 - a missing group must not stop the sim
                    log.warning("loop: wall bump inject %s failed (%s)", group, exc)

        # -- 7. ink + trail ---------------------------------------------------------------------
        ink = self.body.ink(kin, f, mood_name, t_ms)
        broker = getattr(self.agent, "snapshots", None) if self.agent is not None else None
        if broker is not None:
            try:
                broker.record_trail(kin.x, kin.y, ink.color, ink.width)
                broker.save_trail()  # throttled internally (~every 10s): keeps the shared painting across a restart
            except Exception as exc:  # noqa: BLE001
                log.debug("loop: record_trail failed (%s)", exc)

        # -- 8. mood ----------------------------------------------------------------------------
        ms, transition = self._update_mood(f, r, stats, t_ms, events, pokes)
        confirmed = self.mood.confirmed()
        if transition is not None:
            events.append({"kind": "mood", "t_ms": t_ms,
                           "data": {"from": transition.src.value, "to": transition.dst.value,
                                    "reason": transition.reason}})

        # -- 9. homeostasis ---------------------------------------------------------------------
        homeo = self.engine.apply_homeostasis(stats)
        if homeo is not None:
            oob.append({"kind": "homeostasis", "data": dict(homeo)})

        # -- 10. publish ------------------------------------------------------------------------
        self.seq = seq
        compute_ms = max((time.perf_counter() - body_t0) * 1000.0, 1e-3)
        self.compute_ms = compute_ms
        tick = self.build_tick(seq=seq, t_ms=t_ms, wall=now, stats=stats, steps=steps, kin=kin, ink=ink,
                               mood_state=ms, snap=snap, drives=f, events=events, compute_ms=compute_ms)
        with self._lock:
            self._latest = tick
            for ev in events:
                self._events.append({"kind": str(ev.get("kind", "")), "t_ms": int(ev.get("t_ms", t_ms)),
                                     "data": dict(ev.get("data") or {})})
        self.bus.publish_tick(tick)
        self.history.push(tick)
        self._ready.set()

        for ev in oob:
            self._publish_event(str(ev.get("kind", "event")), dict(ev.get("data") or {}), seq, t_ms, now)
        if transition is not None:
            self._publish_mood_change(transition, ms, seq, t_ms, now)
        self._pending_trades.extend(trades)
        if len(self._pending_trades) > MAX_MARKET_TRADES:
            # SPEC d.5 "<= 200, oldest dropped": bound the buffer as it fills, not only when a snapshot
            # finally arrives (a stalled feed would otherwise grow it without limit).
            del self._pending_trades[:-MAX_MARKET_TRADES]
        if snap is not None and int(snap.seq) != self._last_market_seq:
            self._last_market_seq = int(snap.seq)
            self._publish_market(snap, seq, t_ms, now)

        # -- 11. agent + session log ------------------------------------------------------------
        if self.agent is not None:
            try:
                self.agent.on_tick(tick, transition, confirmed, int(self.history.jumps(60.0)))
            except Exception as exc:  # noqa: BLE001 - the agent never stops the sim
                self.n_errors += 1
                self.last_error = f"agent: {type(exc).__name__}: {exc}"
                log.warning("loop: agent.on_tick failed (%s)", exc)
        if self.session_log is not None:
            try:
                self.session_log.write_tick(seq, t_ms, now, snap, trades, pokes, commands,
                                            int(stats.total_spikes))
            except Exception as exc:  # noqa: BLE001
                log.warning("loop: session log write failed (%s)", exc)

        if row is not None:
            self._check_replay(row, seq, int(stats.total_spikes))
        return tick

    # ================================================================= tick helpers

    def _apply_command(self, cmd: Mapping[str, Any], t_ms: int, now: float, oob: list[dict]) -> None:
        """``set_market_mode`` / ``clear`` / ``tweet_test`` (SPEC c.27 step 1)."""
        name = str(cmd.get("name", ""))
        try:
            if name == "set_market_mode":
                mode = str(cmd.get("mode", ""))
                ok = self.feed.set_mode(mode)
                if ok:
                    oob.append({"kind": "market_source", "data": {"mode": self.feed.mode,
                                                                  "reason": f"set_market_mode {mode}"}})
                else:
                    log.warning("loop: set_market_mode %s refused", mode)
            elif name == "clear":
                broker = getattr(self.agent, "snapshots", None) if self.agent is not None else None
                if broker is not None:
                    broker.clear_trail()
                oob.append({"kind": "clear", "data": {"by": str(cmd.get("by", "client"))}})
            elif name == "tweet_test":
                if self.agent is not None:
                    self.agent.request_manual()
            else:
                log.warning("loop: unknown command %r ignored", name)
        except Exception as exc:  # noqa: BLE001 - SPEC 0.1
            log.warning("loop: command %s failed (%s)", name, exc)

    def _poll_market(self, now: float, row: Mapping[str, Any] | None
                     ) -> tuple[MarketSnapshot | None, list[Trade], list[dict]]:
        if row is not None:
            return self._replay_market(row)
        try:
            snap, trades, events = self.feed.poll(now, self.tick_s)
        except Exception as exc:  # noqa: BLE001 - a broken feed must not stop the sim
            self.n_errors += 1
            self.last_error = f"market: {type(exc).__name__}: {exc}"
            log.warning("loop: market poll failed (%s)", exc)
            return self.feed.last_snapshot, [], [{"kind": "market_source",
                                                  "data": {"mode": self.feed.mode, "reason": str(exc)}}]
        return snap, list(trades), [self._market_event(e) for e in events]

    @staticmethod
    def _market_event(ev: Mapping[str, Any]) -> dict:
        """Normalise a ``MarketFeed`` event to the SPEC d.8 shape ``{kind, data: {mode, reason}}``.

        ``MarketFeed._event`` already emits that exact shape (with ``source``/``reason`` kept as top-level
        aliases for the literal c.16 wording), so forward its ``data`` verbatim; only synthesise ``data``
        from the aliases when a feed hands over an event without one. Either way the published ``data`` has
        exactly ``mode``/``reason`` keys - never a spurious nested ``data``.
        """
        raw = ev.get("data")
        if isinstance(raw, Mapping):
            data = {"mode": raw.get("mode", ev.get("source", "")), "reason": raw.get("reason", "")}
        else:
            data = {"mode": ev.get("source", ""), "reason": ev.get("reason", "")}
        return {"kind": str(ev.get("kind", "market_source")), "data": data}

    def _drain_feature_events(self) -> list[dict]:
        """Take the ``whale`` / ``easter_egg`` events ``FeatureExtractor.update`` queued this tick.

        The extractor is the only place these are derived (SPEC d.8): it owns the whale threshold, the
        60 s per-reason easter-egg cooldown and the once-per-minute clock key. Re-deriving them here
        would consume nothing (the clock key is already spent) *and* leave ``features._events`` growing
        for the lifetime of the process, so the queue is drained once per tick and published as is.
        """
        try:
            queued = self.features.drain_events()
        except Exception as exc:  # noqa: BLE001 - SPEC 0.1: never stop the sim over diagnostics
            log.warning("loop: draining feature events failed (%s)", exc)
            return []
        out: list[dict] = []
        for ev in queued:
            if not isinstance(ev, Mapping):  # pragma: no cover - defensive
                continue
            out.append({"kind": str(ev.get("kind", "event")), "data": dict(ev.get("data") or {})})
        return out

    def _update_mood(self, f: Features, r: Readouts, stats: StepStats, t_ms: int, events: list[dict],
                     pokes: Sequence[Poke]) -> tuple[MoodState, Transition | None]:
        rates = stats.rates or {}
        jumped = any(str(e.get("kind")) == "jump" for e in events)
        # SPEC d.2 ``drives.any_max`` includes sleep_pressure; the mood machine's ``any_drive_max`` must
        # not, or a sleepy-but-quiet market would keep arousal high and SLEEP unreachable.
        any_drive = max(f.sugar, f.bitter, f.water, f.looming, f.flash, f.odor, f.chop, f.courtship)
        inputs = MoodInputs(
            t_ms=t_ms,
            sugar=float(f.sugar), looming=float(f.looming), bitter=float(f.bitter),
            activity=float(f.activity), sleep_pressure=float(f.sleep_pressure),
            mn9=float(r.mn9), dnp09=_as_float(rates.get("dn_freeze")), lc_loom=float(r.lc_loom),
            pam=_as_float(rates.get("pam")), ppl1=_as_float(rates.get("ppl1")),
            mbon_approach=_as_float(rates.get("mbon_approach")),
            mbon_avoid=_as_float(rates.get("mbon_avoid")),
            p1=float(r.p1), pip10=float(r.pip10), dn_mean=float(r.dn_mean),
            feeding=bool(self.decoder.feeding), jumped=jumped,
            gf_spikes_10s=int(self.history.gf_spikes(10.0)) + int(r.gf_spikes_l + r.gf_spikes_r),
            any_drive_max=float(any_drive), poked=bool(pokes or self.encoder.active_pokes),
            forced=None, hunger=float(f.hunger),
        )
        return self.mood.update(inputs)

    def _adapt_speed(self, compute_ms: float) -> None:
        """SPEC c.27 step 12: back off fast, recover slowly."""
        budget = self.tick_ms
        if compute_ms > SPEED_DOWN_FRAC * budget:
            before = self.speed
            self.speed = max(SPEED_MIN, self.speed * SPEED_DOWN_FACTOR)
            self._fast_run = 0
            if self.speed < before - 1e-9:
                log.info("loop: speed %.3f -> %.3f (compute %.1f ms of %.1f ms)", before, self.speed,
                         compute_ms, budget)
        elif compute_ms < SPEED_UP_FRAC * budget:
            self._fast_run += 1
            if self._fast_run >= SPEED_UP_TICKS:
                self._fast_run = 0
                before = self.speed
                self.speed = min(SPEED_MAX, self.speed * SPEED_UP_FACTOR)
                if self.speed > before + 1e-9:
                    log.info("loop: speed %.3f -> %.3f (compute %.1f ms)", before, self.speed, compute_ms)
        else:
            self._fast_run = 0

    # ================================================================= publishing

    def _publish_event(self, kind: str, data: Mapping[str, Any], seq: int, t_ms: int, wall: float) -> None:
        msg = {"type": "event", "seq": int(seq), "t_ms": int(t_ms), "wall": float(wall),
               "kind": str(kind), "data": dict(data)}
        with self._lock:
            self._events.append({"kind": str(kind), "t_ms": int(t_ms), "data": dict(data)})
        try:
            self.bus.publish_event(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("loop: publish_event %s failed (%s)", kind, exc)

    def _publish_mood_change(self, tr: Transition, ms: MoodState, seq: int, t_ms: int, wall: float) -> None:
        msg = {"type": "mood_change", "seq": int(seq), "t_ms": int(t_ms), "wall": float(wall),
               "from": tr.src.value, "to": tr.dst.value, "reason": tr.reason, "mood": ms.to_wire()}
        log.info("mood: %s -> %s (%s)", tr.src.value, tr.dst.value, tr.reason)
        try:
            self.bus.publish_event(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("loop: publish mood_change failed (%s)", exc)

    def _publish_market(self, snap: MarketSnapshot, seq: int, t_ms: int, wall: float) -> None:
        trades = self._pending_trades[-MAX_MARKET_TRADES:]
        self._pending_trades = []
        msg = {"type": "market", "seq": int(seq), "t_ms": int(t_ms), "wall": float(wall),
               "mode": self.feed.mode, "market": snap.to_wire(),
               "trades": [t.to_wire() for t in trades]}
        try:
            self.bus.publish_event(msg)
        except Exception as exc:  # noqa: BLE001
            log.warning("loop: publish market failed (%s)", exc)

    # ================================================================= frame builders

    def build_hello(self) -> dict:
        """The ``hello`` frame (SPEC d.1): the single source of truth for the session's invariants."""
        s = self.settings
        conn = self.conn
        meta = dict(getattr(conn, "meta", {}) or {})
        monitor = self.engine.monitor
        source = str(getattr(conn, "source", "") or "")
        if source not in ("synthetic", "csv", "neuprint"):
            source = str(getattr(s, "connectome_source", "synthetic"))
        agent = self._hello_agent()
        return {
            "type": "hello",
            "v": PROTOCOL_VERSION,
            "run_id": str(getattr(s, "run_id", "") or ""),
            "server_wall": time.time(),
            "dt_ms": float(self.dt_ms),
            "tick_ms": float(self.tick_ms),
            "steps_per_tick": int(self.steps_per_tick),
            "tick_hz": int(getattr(s, "tick_hz", 20)),
            "realtime": bool(self.realtime),
            "backend": str(self.engine.backend),
            "canvas": {"w": int(getattr(s, "canvas_w", 800)), "h": int(getattr(s, "canvas_h", 500)),
                       "walls": str(getattr(s, "walls", "bounce"))},
            "connectome": {
                "name": str(getattr(conn, "name", "")),
                "source": source,
                "n": int(conn.n),
                "e": int(conn.e),
                "synapses": float(meta.get("synapses", 0.0) or 0.0),
                "gain": float(self.engine.gain),
                "weights_mode": meta.get("weights_mode"),
                "license": str(meta.get("license", "") or ""),
                "citation": meta.get("citation"),
                "note": str(meta.get("note", "") or ""),
                "region_counts": {str(k): int(v) for k, v in (meta.get("region_counts") or {}).items()},
            },
            "regions": list(REGIONS),
            "raster": {"per_region": int(monitor.per_region), "cap": int(monitor.cap),
                       "rows": monitor.rows_wire()},
            "pops": list(monitor.rates.keys()),
            "mood_states": list(MOOD_STATES),
            "channels": list(POKE_TABLE),
            "market_modes": list(MARKET_MODES),
            "market": self._hello_market(),
            "agent": agent,
            "features": {
                "explore_baseline": float(getattr(s, "explore_baseline", 0.25)),
                "wander_sigma": float(getattr(s, "wander_sigma", 0.6)),
                "mood_feedback": bool(getattr(s, "mood_feedback", True)),
                "easter_eggs": bool(getattr(s, "easter_eggs", True)),
                "drive_mode": str(getattr(s, "drive_mode", "poisson")),
                "noise_mu": float(getattr(s, "noise_mu", 0.5)),
                "noise_sigma": float(getattr(s, "noise_sigma", 3.5)),
            },
        }

    def _hello_market(self) -> dict:
        """``hello.market`` (SPEC d.1) plus the launch disclosure of ``FLY_TOKEN_LIVE``.

        ``self.feed.hello()`` describes the source being served; this adds the two things a client needs in
        order to be honest about it. ``token_live`` is false unless the operator has explicitly set
        ``FLY_TOKEN_LIVE=1`` together with the project's own pair: while it is false the numbers on the wire
        belong to a third-party pair that is only the brain's sensory input, and no surface may present them as
        this project's own price / market cap / liquidity. ``pair`` and ``dex`` complete the identity of that
        pair (``chain``, ``symbol`` and ``token`` are already in the block) so the UI can name it; they are
        ``None`` until the first snapshot lands, which is why ``token`` - the configured address - stays the
        authoritative identity.
        """
        block = dict(self.feed.hello())
        snap = getattr(self.feed, "last_snapshot", None)
        block["token_live"] = bool(getattr(self.settings, "token_live", False))
        block["pair"] = str(getattr(snap, "pair", "") or "") or None
        block["dex"] = str(getattr(snap, "dex", "") or "") or None
        return block

    def _hello_agent(self) -> dict:
        s = self.settings
        llm = str(getattr(s, "llm", "dryrun"))
        x = str(getattr(s, "x_mode", "dryrun"))
        if self.agent is not None:
            gen = getattr(self.agent, "generator", None)
            poster = getattr(self.agent, "poster", None)
            if gen is not None and getattr(gen, "dry_run", False):
                llm = "dryrun"
            if poster is not None and getattr(poster, "dry_run", True):
                x = "dryrun"
        return {"llm": llm, "x": x, "cooldown_s": int(getattr(s, "tweet_cooldown_s", 900)),
                "reason_cooldown_s": int(getattr(s, "tweet_reason_cooldown_s", 2700)),
                "tweets_per_day": int(getattr(s, "tweets_per_day", 12)),
                "lang": str(getattr(s, "tweet_lang", "en"))}

    def build_tick(self, *, seq: int, t_ms: int, wall: float, stats: StepStats, steps: int,
                   kin: Kinematics, ink: InkStyle, mood_state: MoodState,
                   snap: MarketSnapshot | None, drives: Features, events: Iterable[Mapping[str, Any]],
                   compute_ms: float) -> dict:
        """The ``tick`` frame (SPEC d.2): the single source of truth for the tick shape.

        ``rtf`` is SPEC c.27 step 12's ``(steps*dt) / max(compute_ms, 1e-3)`` measured over the tick
        *body* - the frame is built at step 10, so the fan-out, agent trigger and session-log write of
        the same tick (steps 10-11) cannot be inside it. ``_adapt_speed`` and the pacing sleep of step 12
        use the full tick duration instead, and both numbers stay readable as ``loop.rtf`` /
        ``loop.compute_ms``.
        """
        rtf = (steps * self.dt_ms) / max(float(compute_ms), 1e-3)
        self.rtf = rtf
        market = self._market_block(snap)
        return {
            "type": "tick",
            "seq": int(seq),
            "t_ms": int(t_ms),
            "wall": float(wall),
            "sim": {"rtf": float(rtf), "speed": float(self.speed), "steps": int(steps),
                    "step_ms": float(stats.step_ms_mean), "spikes": int(stats.total_spikes),
                    "active_frac": float(stats.active_frac_max), "gain": float(stats.gain),
                    "edge_visits": int(stats.edge_visits), "forced": int(stats.forced_events),
                    "noise": bool(stats.noise_on), "backend": str(self.engine.backend)},
            "fly": kin.to_wire(),
            "ink": ink.to_wire(),
            "mood": mood_state.to_wire(),
            "market": market,
            "drives": drives.to_wire(),
            "rates": {"regions": [float(x) for x in stats.region_rates.tolist()],
                      "pops": {k: float(v) for k, v in (stats.rates or {}).items()}},
            "spikes": {"t0_ms": int(stats.t0_ms), "win_ms": int(stats.t1_ms - stats.t0_ms),
                       "total": int(stats.total_spikes), "capped": bool(stats.capped),
                       "slots": [int(v) for v in stats.spike_indices_sample.tolist()],
                       "dt": [int(v) for v in stats.spike_dt_sample.tolist()]},
            "events": [{"kind": str(e.get("kind", "")), "t_ms": int(e.get("t_ms", t_ms)),
                        "data": dict(e.get("data") or {})} for e in events],
        }

    def _market_block(self, snap: MarketSnapshot | None) -> dict:
        """``tick.market``: the snapshot plus the feed mode, the last trade of the session and ``token_live``.

        The disclosure rides on every tick, not only on ``hello``: a client that connects late, drops a hello or
        rehydrates from ``GET /api/state`` must never render this feed as this project's own token by default.
        """
        body: dict[str, Any]
        if snap is None:
            body = {"source": "sim", "ts": 0.0, "seq": 0, "chain": "sim", "dex": "sim", "pair": "SIM",
                    "symbol": "FLY", "price_usd": None, "price_native": None, "buys_m5": 0, "sells_m5": 0,
                    "buys_h1": 0, "sells_h1": 0, "chg_m5": 0.0, "chg_h1": 0.0, "chg_h6": 0.0,
                    "chg_h24": 0.0, "vol_m5": 0.0, "vol_h1": 0.0, "liq_usd": None, "fdv": None,
                    "mcap": None, "regime": None}
        else:
            body = snap.to_wire()
        body["mode"] = self.feed.mode
        body["last_trade"] = None if self._last_trade is None else self._last_trade.to_wire()
        body["token_live"] = bool(getattr(self.settings, "token_live", False))
        return body

    # ================================================================= replay

    def _next_replay_row(self) -> dict:
        """The next ``kind == "tick"`` row of the session log (``StopIteration`` at the end)."""
        assert self.replay is not None
        while True:
            row = next(self.replay)                       # StopIteration ends the run
            if isinstance(row, Mapping) and str(row.get("kind", "")) == "tick":
                return dict(row)

    @staticmethod
    def _replay_pokes(row: Mapping[str, Any]) -> list[Poke]:
        out: list[Poke] = []
        for raw in row.get("pokes") or ():
            try:
                out.append(Poke(stim=str(raw["stim"]), strength=float(raw["strength"]),
                                side=int(raw["side"]), until_ms=int(raw["until_ms"])))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("loop: replay poke %r skipped (%s)", raw, exc)
        return out

    def _replay_market(self, row: Mapping[str, Any]) -> tuple[MarketSnapshot | None, list[Trade], list[dict]]:
        snap: MarketSnapshot | None = self.feed.last_snapshot
        raw = row.get("market")
        if isinstance(raw, Mapping):
            try:
                snap = MarketSnapshot(**{k: v for k, v in raw.items()})
            except TypeError as exc:
                log.warning("loop: replay snapshot skipped (%s)", exc)
        trades: list[Trade] = []
        for t in row.get("trades") or ():
            try:
                trades.append(Trade(ts=float(t["ts"]), kind=str(t["kind"]), usd=float(t["usd"]),
                                    price=float(t.get("price", 0.0) or 0.0),
                                    surrogate=bool(t.get("surrogate", False))))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("loop: replay trade %r skipped (%s)", t, exc)
        return snap, trades, []

    def _check_replay(self, row: Mapping[str, Any], seq: int, total_spikes: int) -> None:
        self.replay_ticks += 1
        expected = row.get("total_spikes")
        if expected is None:
            return
        if int(expected) != int(total_spikes):
            self.replay_diffs.append((int(seq), int(expected), int(total_spikes)))
            if len(self.replay_diffs) == 1:
                log.warning("replay: tick %d spikes %d != logged %d", seq, total_spikes, int(expected))
