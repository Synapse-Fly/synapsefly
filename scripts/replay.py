#!/usr/bin/env python
"""Replay a recorded session and prove the brain is deterministic (SPEC h.4, section 0 "Replay").

Usage::

    py -3 scripts/replay.py [data\\sessions\\<run_id>.jsonl] [--assert] [--out replay.jsonl]
                            [--ticks N] [--quiet]

With no path the newest ``<data_dir>/sessions/*.jsonl`` is used; when there is no session log at all the
script prints ``REPLAY SKIP: ...`` and exits 0 (a fresh checkout has not run the server yet -- set
``FLY_SESSION_LOG=1`` and start ``backend\\run.py`` once to record one).

What it does: reads the header's settings (forcing ``FLY_REALTIME=0``, ``FLY_MARKET=sim``,
``FLY_SESSION_LOG=0``, ``FLY_REPLAY=``), rebuilds the identical connectome and asserts ``connectome_key``
matches the header, then feeds every logged tick's inputs (market snapshot, trades, pokes, commands) back
through the SPEC c.27 pipeline with a fresh engine. ``--assert`` compares ``total_spikes`` per tick and the
final fly pose against the log and prints ``REPLAY OK <n> ticks`` or the first diverging tick.

Two things a reader of SPEC h.4 should know:

* **Stepping.** A log recorded the way SPEC i.5 tells a user to (``FLY_REALTIME=1``, the section-b
  default) does **not** carry a uniform number of LIF steps per tick: ``SimulationLoop`` scales them by
  its adaptive ``speed``, and ``SessionLog.write_tick`` (SPEC c.25) persists neither ``steps`` nor
  ``speed``. The step count of every tick is therefore reconstructed from the log's own brain clock
  (``steps = (t_ms - t_ms_prev) / dt_ms``, see ``steps_plan``) and pushed into the loop before each tick,
  which is what makes such a log replay bit-exactly. A log whose ``t_ms`` column cannot be turned into
  whole steps is refused under ``--assert`` with a one-line reason instead of silently diverging, and a
  paced run at a sub-millisecond ``dt`` is flagged as ambiguous (an int brain clock cannot express it).
* **Pose.** SPEC h.4 asks ``--assert`` to compare the final fly pose too, but the ``write_tick``
  signature of SPEC c.25 has no pose argument, so no server-written log contains one. The pose is
  compared whenever the log has ``x``/``y`` (i.e. a log re-recorded by ``--out``) and the verdict says
  loudly when it was skipped -- it is never quietly counted as a pass.

``--out FILE`` re-records the replay in the same session-log format (header + one ``tick`` line per tick,
``total_spikes`` taken from this run), so ``replay.py FILE --assert`` replays it again -- that round trip is
the determinism proof that needs no server.

The replay always runs through the server's own wiring -- ``server.app.build_context`` plus
``SimulationLoop(replay=...)`` driven by ``startup()`` + ``tick_once()`` (SPEC h.4). There is deliberately
no second in-script pipeline: a parallel re-implementation of SPEC c.27 drifts from the loop (mood
feedback scores, homeostasis, the 10 s GF window) and a fallback that silently produces different spike
counts is worse than a clear "install the runtime pins" failure. ASCII only, no network, exit 0/1 (2 for a
usage problem).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from flybrain.config import (ENV_BY_NAME, ENV_VARS, gain_value, load_settings,   # noqa: E402
                             parse_dotenv)
from flybrain.connectome.loaders import load_connectome                     # noqa: E402
from flybrain.server.session_log import read_session                        # noqa: E402
from flybrain.snn.calibrate import load_calibration                         # noqa: E402

POSE_TOL_PX = 0.05          # the log rounds the pose; this is the comparison tolerance
STEP_EPS = 1e-6             # t_ms / dt_ms must land on a whole step this closely


def ambient_env() -> dict[str, str]:
    """``<repo>/.env`` then the process environment (the environment wins), known keys only.

    SPEC section b precedence, the same helper ``selftest.py`` uses: a user who put ``FLY_DATA_DIR``
    in ``.env`` (and nowhere else) must see the same ``data/sessions`` directory the server wrote to.
    """
    env: dict[str, str] = {}
    dotenv = REPO / ".env"
    if dotenv.is_file():
        env.update(parse_dotenv(dotenv))
    for key, value in os.environ.items():
        if key in ENV_BY_NAME:
            env[key] = value
    return {k: v for k, v in env.items() if k in ENV_BY_NAME}


# ------------------------------------------------------------------------------ log reading
def header_key(path: Path) -> str:
    """``connectome_key`` of a log's header line, or ``""`` (never raises)."""
    try:
        with path.open("r", encoding="utf-8") as fh:
            row = json.loads(fh.readline() or "{}")
    except (OSError, ValueError):
        return ""
    return str(row.get("connectome_key") or "") if isinstance(row, dict) else ""


def newest_session(data_dir: Path, want_key: str = "") -> tuple[Path | None, str]:
    """The log to replay when no path was given, plus a one-line note about the choice.

    The newest log wins, except that a log recorded against *this* configuration's graph beats a newer
    one recorded against another: an old ``data/sessions`` full of logs from a different ``FLY_N_NEURONS``
    would otherwise make the no-argument invocation of SPEC i.5 fail on a ``connectome_key`` mismatch.
    """
    sessions = sorted((data_dir / "sessions").glob("*.jsonl"),
                      key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    if not sessions:
        return None, ""
    if want_key:
        for i, candidate in enumerate(sessions):
            if header_key(candidate) == want_key:
                if i == 0:
                    return candidate, ""
                return candidate, (f"picked the newest of {len(sessions)} logs whose connectome_key is "
                                   f"{want_key} ({i} newer log(s) were recorded against another graph)")
        return sessions[0], (f"none of the {len(sessions)} logs in {data_dir / 'sessions'} was recorded "
                             f"against this configuration's graph ({want_key}); using the newest one, "
                             f"which will fail the connectome_key check. Record a fresh log (run the "
                             f"server once with FLY_SESSION_LOG=1) or pass a path explicitly.")
    return sessions[0], ""


def load_log(path: Path) -> tuple[dict, list[dict]]:
    """(header, tick rows). Raises ValueError when the file carries no header line."""
    header: dict | None = None
    ticks: list[dict] = []
    for row in read_session(path):
        kind = row.get("kind")
        if kind == "header" and header is None:
            header = row
        elif kind == "tick":
            ticks.append(row)
    if header is None:
        raise ValueError(f"{path.name} has no 'header' line (not a session log?)")
    return header, ticks


def settings_from_header(header: dict, forced: dict[str, str] | None = None):
    """Rebuild ``Settings`` from the header's redacted settings dict via the env registry.

    A header without a usable ``settings`` block would silently fall back to the section-b defaults for
    every non-graph setting (``tick_hz``, ``explore_baseline``, noise, walls, canvas) -- the
    ``connectome_key`` guard only catches graph-affecting drift -- so say so loudly instead.
    """
    raw = header.get("settings")
    src = dict(raw) if isinstance(raw, dict) else {}
    if not src:
        print("[WARN] header    this log has no usable 'settings' block, so every setting falls back to "
              "the section-b defaults (tick_hz, noise, explore_baseline, walls, canvas): a replay of a "
              "run that used other values will diverge even when the connectome_key matches.")
    env: dict[str, str] = {}
    for spec in ENV_VARS:
        if not spec.field or spec.field not in src:
            continue
        value = src[spec.field]
        if isinstance(value, bool):
            env[spec.name] = "1" if value else "0"
        elif isinstance(value, (list, tuple)):
            env[spec.name] = ",".join(str(x) for x in value)
        elif value is None:
            env[spec.name] = ""
        else:
            env[spec.name] = str(value)
    env.update({"FLY_REALTIME": "0", "FLY_MARKET": "sim", "FLY_SESSION_LOG": "0", "FLY_REPLAY": ""})
    ambient = ambient_env()
    # The report is this script's output, so the log stays at WARNING unless the operator asks for more
    # *on this invocation* (exported FLY_LOG_LEVEL); the .env value steers the server, not the replay.
    env["FLY_LOG_LEVEL"] = os.environ.get("FLY_LOG_LEVEL") or "WARNING"
    for key in ("FLY_DATA_DIR", "FLY_OUT_DIR"):      # a relocated checkout keeps working
        if ambient.get(key):
            env[key] = ambient[key]
    if forced:
        env.update(forced)
    return load_settings(env={k: v for k, v in env.items() if k in ENV_BY_NAME}, dotenv=None)


# ------------------------------------------------------------------------------ stepping
def steps_plan(ticks: list[dict], dt_ms: float,
               default_steps: int) -> tuple[list[int] | None, str]:
    """Per-tick LIF step counts reconstructed from the log's ``t_ms`` column.

    A recorded run does **not** always step ``steps_per_tick`` times per tick: with ``FLY_REALTIME=1``
    (the section-b default, so this is the normal case for a log a user recorded) ``SimulationLoop``
    runs ``max(MIN_STEPS, round(steps_per_tick * speed))`` steps, and ``speed`` drops as soon as one
    tick costs more than 80 % of the tick budget. ``SessionLog.write_tick`` (SPEC c.25) persists
    neither ``steps`` nor ``speed``, so the only record of what was stepped is the brain clock itself:
    ``t_ms`` advances by exactly ``steps * dt_ms`` per tick. Replaying with a fixed ``steps_per_tick``
    therefore diverges on every tick after the first speed change -- hence this reconstruction.

    Returns ``(plan, note)``. ``plan is None`` means the column cannot be turned into whole steps (a
    hand-edited log, or ``dt_ms`` changed since the recording): the caller must not claim bit-exactness.
    """
    dt = float(dt_ms)
    if dt <= 0.0:
        return None, f"dt_ms is {dt:g}"
    plan: list[int] = []
    prev = 0
    for i, row in enumerate(ticks, start=1):
        t_ms = int(row.get("t_ms") or 0)
        delta = t_ms - prev
        prev = t_ms
        exact = delta / dt
        steps = int(round(exact))
        if delta <= 0 or steps < 1 or abs(exact - steps) > STEP_EPS:
            return None, (f"tick {i} advances the brain clock by {delta} ms, which is not a whole "
                          f"number of {dt:g} ms steps (t_ms must grow by steps*dt_ms every tick)")
        plan.append(steps)
    if not plan:
        return [], "no ticks"
    lo, hi = min(plan), max(plan)
    if lo == hi:
        return plan, (f"{lo} steps/tick ({lo * dt:g} ms of brain time per tick, uniform: the run was "
                      f"recorded with FLY_REALTIME=0 or never left speed 1.0)")
    return plan, (f"{lo}-{hi} steps/tick, reconstructed per tick from the log's t_ms column (the "
                  f"recorded run was paced: FLY_REALTIME=1 scales the steps per tick by 'speed', which "
                  f"SessionLog does not persist; nominal steps_per_tick={int(default_steps)})")


def plan_is_ambiguous(plan: list[int] | None, dt_ms: float) -> bool:
    """True when the reconstruction can be off by a step: sub-ms ``dt`` plus a paced run.

    ``t_ms`` is an integer (SPEC 0.1 "brain time ms int"), so at ``dt_ms < 1`` an odd step count rounds
    the brain clock and the per-tick split cannot be recovered exactly -- a paced ``dt 0.5`` run may
    replay one step short on some ticks. Uniform stepping is unaffected (every tick gets the nominal
    count) and ``dt_ms >= 1`` is exact, which covers the documented default.
    """
    if plan is None or len(set(plan)) <= 1:
        return False
    return abs(float(dt_ms) - round(float(dt_ms))) > 1e-9


# ------------------------------------------------------------------------------ drivers
def resolve_gain(conn, settings) -> tuple[float, str]:
    explicit = gain_value(settings)
    if explicit is not None:
        return float(explicit), "FLY_GAIN"
    key = str(settings.connectome_key or "")
    cached = load_calibration(settings, key) if key else None
    if cached is not None:
        return float(cached.get("gain", 1.0)), "calib json"
    return float(conn.meta.get("gain_default", 1.0) or 1.0), "gain_default"


def replay_via_loop(settings, path: Path, ticks: list[dict], quiet: bool,
                    plan: list[int] | None = None) -> tuple[list[dict] | None, str | None, object | None]:
    """Replay through the server's own wiring (SPEC h.4): ``build_context`` + ``SimulationLoop``.

    ``settings.replay`` already points at the log, so ``server.app.build_context`` builds exactly what the
    lifespan builds -- the same seed children, the same gain resolution, the same collaborators -- with the
    loop's ``replay`` iterator wired up. The loop is driven with ``startup()`` + ``tick_once()`` instead of
    ``start()``: no thread and no wall clock, so the replay is bit-exact against a log the server wrote and
    ends deterministically when ``tick_once`` raises ``StopIteration``.

    ``plan`` (from ``steps_plan``) is pushed into ``loop.steps_per_tick`` before each tick: a replay is
    never realtime, so ``_steps_for_tick()`` returns that value verbatim and the tick steps the brain
    exactly as many times as the recorded tick did -- including a run that was paced by ``speed``.

    Returns (rows, None, ctx) or (None, reason, ctx); a reason is fatal (there is no second driver, by
    design -- see the module docstring). ``ctx`` (when not None) must be shut down by the caller.
    """
    ctx = None
    try:
        from flybrain.server.app import build_context
        from flybrain.server.loop import SimulationLoop
    except Exception as exc:  # noqa: BLE001
        return None, f"flybrain.server is not importable ({exc.__class__.__name__}: {exc})", None
    if not callable(getattr(SimulationLoop, "tick_once", None)):
        return None, "SimulationLoop has no tick_once()", None
    try:
        ctx = build_context(settings)
    except Exception as exc:  # noqa: BLE001
        return None, f"build_context failed ({exc.__class__.__name__}: {exc})", None
    loop = ctx.loop
    if loop.replay is None:
        return None, f"the loop has no replay iterator (FLY_REPLAY={settings.replay!r} was not wired)", ctx
    loop.agent = None                 # a replay must not tweet

    rows: list[dict] = []
    limit = len(ticks)
    gain = float(ctx.engine.gain)
    try:
        loop.startup()
        nominal = int(loop.steps_per_tick)
        while len(rows) < limit:
            loop.steps_per_tick = (int(plan[len(rows)]) if plan is not None and len(rows) < len(plan)
                                   else nominal)
            try:
                tick = loop.tick_once()
            except StopIteration:
                break
            fly = tick.get("fly") if isinstance(tick.get("fly"), dict) else {}
            sim = tick.get("sim") if isinstance(tick.get("sim"), dict) else {}
            spikes = tick.get("spikes") if isinstance(tick.get("spikes"), dict) else {}
            src = ticks[len(rows)]
            rows.append({"seq": int(tick.get("seq") or len(rows) + 1),
                         "t_ms": int(tick.get("t_ms") or 0), "wall": float(tick.get("wall") or 0.0),
                         "total_spikes": int(spikes.get("total", sim.get("spikes", 0)) or 0),
                         "x": float(fly.get("x") or 0.0), "y": float(fly.get("y") or 0.0),
                         "mood": str((tick.get("mood") or {}).get("state") or ""),
                         "mode": str(fly.get("mode") or ""), "gain": float(sim.get("gain") or gain),
                         "market": src.get("market"), "trades": list(src.get("trades") or ()),
                         "pokes": list(src.get("pokes") or ()), "commands": list(src.get("commands") or ())})
            if not quiet and len(rows) % 200 == 0:
                print(f"  ... {len(rows)}/{limit} ticks", flush=True)
    except Exception as exc:  # noqa: BLE001
        return None, f"SimulationLoop failed at tick {len(rows) + 1} ({exc.__class__.__name__}: {exc})", ctx
    if not rows:
        return None, "SimulationLoop produced no ticks", ctx
    if not quiet:
        print(f"driver    server.app.build_context + SimulationLoop.tick_once ({len(rows)} ticks, "
              f"{len(loop.replay_diffs)} spike diff(s) reported by the loop itself)")
    return rows, None, ctx


def shutdown(ctx) -> None:
    """Stop whatever ``build_context`` started (never raises)."""
    if ctx is None:
        return
    for stop in (lambda: ctx.loop.stop(0.2), lambda: ctx.feed.stop(),
                 lambda: ctx.session_log.close() if ctx.session_log is not None else None):
        try:
            stop()
        except Exception:  # noqa: BLE001
            pass


# ------------------------------------------------------------------------------ comparison / output
def compare(ticks: list[dict], rows: list[dict]) -> tuple[bool, str]:
    """Compare ``total_spikes`` per tick and the final pose; returns (ok, message)."""
    if len(rows) != len(ticks):
        return False, f"replayed {len(rows)} ticks but the log has {len(ticks)}"
    for i, (logged, got) in enumerate(zip(ticks, rows)):
        want = int(logged.get("total_spikes") or 0)
        have = int(got.get("total_spikes") or 0)
        if want != have:
            return False, (f"tick {i} (seq {logged.get('seq')}, t_ms {logged.get('t_ms')}): "
                           f"total_spikes {have} != {want} logged")
    return True, f"{len(rows)} ticks, total_spikes identical"


def compare_pose(ticks: list[dict], rows: list[dict]) -> tuple[bool | None, str]:
    """Compare the final fly pose; ``None`` means "the log has no pose, nothing was compared".

    SPEC h.4 asks ``--assert`` to compare the final pose as well as ``total_spikes``, but the
    ``SessionLog.write_tick`` signature of SPEC c.25 has no pose argument, so no log the server writes
    carries one (see the ``spec_issues`` note in the module docstring). The comparison therefore runs
    only against a log re-recorded by ``--out``, and the caller must say so out loud rather than fold a
    skipped check into a success message.
    """
    if not ticks or not rows:
        return None, "no ticks"
    last_log, last_row = ticks[-1], rows[-1]
    if "x" not in last_log or "y" not in last_log:
        return None, ("this session log carries no fly pose, so only total_spikes was compared "
                      "(SPEC c.25 SessionLog.write_tick records no x/y; re-record with --out to get a "
                      "log that does)")
    dx = abs(float(last_log["x"]) - float(last_row["x"]))
    dy = abs(float(last_log["y"]) - float(last_row["y"]))
    ok = math.isfinite(dx) and math.isfinite(dy) and dx <= POSE_TOL_PX and dy <= POSE_TOL_PX
    return ok, (f"final pose ({last_row['x']:.1f}, {last_row['y']:.1f}) vs "
                f"({float(last_log['x']):.1f}, {float(last_log['y']):.1f})")


def write_out(path: Path, header: dict, rows: list[dict]) -> int:
    """Re-record the replay as a session log (header + one tick line per tick)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        out_header = dict(header)
        out_header["replayed"] = True
        fh.write(json.dumps(out_header, separators=(",", ":"), default=str) + "\n")
        for row in rows:
            line = {"kind": "tick", "seq": row["seq"], "t_ms": row["t_ms"], "wall": row["wall"],
                    "total_spikes": row["total_spikes"], "market": row.get("market"),
                    "trades": row.get("trades") or [], "pokes": row.get("pokes") or [],
                    "commands": row.get("commands") or [],
                    "x": round(float(row["x"]), 1), "y": round(float(row["y"]), 1),
                    "mood": row.get("mood"), "mode": row.get("mode")}
            fh.write(json.dumps(line, separators=(",", ":"), default=str) + "\n")
            written += 1
    return written


# ------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="replay a FlyBrain session log through a fresh engine")
    ap.add_argument("path", nargs="?", default=None, help="session log (default: the newest one)")
    ap.add_argument("--assert", action="store_true", dest="do_assert",
                    help="compare total_spikes per tick, and the final pose when the log has one "
                         "(a server-written log has none; the verdict says so)")
    ap.add_argument("--out", default=None, help="re-record the replay as a session log")
    ap.add_argument("--ticks", type=int, default=0, help="replay at most N ticks (0 = all)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.path:
        path = Path(args.path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.is_file():
            print(f"REPLAY FAILED: no such session log: {path}")
            return 2
    else:
        data_dir = REPO / "data"
        want = ""
        try:
            probe = load_settings(env=ambient_env(), dotenv=None)
            data_dir = Path(probe.data_dir)
            want = str(probe.connectome_key or "")
        except Exception:  # noqa: BLE001
            pass
        found, why = newest_session(data_dir, want)
        if found is None:
            print(f"REPLAY SKIP: no session log in {data_dir / 'sessions'}. Record one with "
                  f"FLY_SESSION_LOG=1 (the default) and a single run of 'py -3 backend\\run.py', "
                  f"then re-run this script.")
            return 0
        path = found
        if why and not args.quiet:
            print(f"note      {why}")

    try:
        header, ticks = load_log(path)
    except (OSError, ValueError) as exc:
        print(f"REPLAY FAILED: {exc}")
        return 2
    if not ticks:
        print(f"REPLAY SKIP: {path.name} has a header but no tick lines (the run was stopped "
              f"before the first tick).")
        return 0
    if args.ticks > 0:
        ticks = ticks[: args.ticks]

    try:
        settings = settings_from_header(header, forced={"FLY_REPLAY": str(path)})
    except ValueError as exc:
        print(f"REPLAY FAILED: the header settings are not loadable ({exc})")
        return 2
    if not args.quiet:
        print(f"log       {path}")
        print(f"header    run_id={header.get('run_id')} created={header.get('created')} "
              f"version={header.get('version')} ticks={len(ticks)}")
        print(f"settings  source={settings.connectome_source} n={settings.n_neurons} "
              f"seed={settings.seed} dt={settings.dt_ms:g} ms tick={settings.tick_hz} Hz "
              f"backend={settings.backend}")

    # Both remaining preconditions need only `settings`, so they run *before* the graph is built: a
    # stale log must not cost the operator a 20k connectome build before it is rejected.
    want_key = str(header.get("connectome_key") or "")
    have_key = str(settings.connectome_key or "")
    if want_key and have_key and want_key != have_key:
        print(f"REPLAY FAILED: connectome_key mismatch: the log was recorded with {want_key} but this "
              f"configuration builds {have_key}. The graph changed (different source, N, seed, weights "
              f"or data files), so a bit-exact replay is impossible.")
        return 1

    plan, plan_note = steps_plan(ticks, settings.dt_ms, settings.steps_per_tick)
    if plan is None:
        print(f"[WARN] stepping  {plan_note}")
        if args.do_assert:
            print(f"REPLAY FAILED: this log's brain clock does not describe whole LIF steps, so the "
                  f"number of steps each tick ran cannot be recovered and --assert cannot be honest "
                  f"about bit-exactness. Re-record the run with FLY_REALTIME=0 (a paced FLY_REALTIME=1 "
                  f"run is replayable, but only while its t_ms column is intact) and dt_ms="
                  f"{settings.dt_ms:g}.")
            return 1
        print("[WARN] stepping  falling back to a fixed "
              f"{int(settings.steps_per_tick)} steps/tick; spike counts will not match the log")
    elif not args.quiet:
        print(f"stepping  {plan_note}")
    ambiguous = plan_is_ambiguous(plan, settings.dt_ms)
    if ambiguous:
        print(f"[WARN] stepping  dt is {settings.dt_ms:g} ms and the run was paced, but the logged brain "
              f"clock is whole milliseconds: a tick that ran an odd number of steps cannot be recovered "
              f"exactly, so a mismatch below may be a recording artefact rather than a determinism bug.")

    try:
        conn = load_connectome(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"REPLAY FAILED: could not rebuild the connectome ({exc.__class__.__name__}: {exc})")
        return 1
    gain, gain_src = resolve_gain(conn, settings)
    if not args.quiet:
        print(f"brain     {conn.name}: n={conn.n} e={conn.e} gain={gain:.3f} ({gain_src}) "
              f"key={have_key or 'n/a'}")

    t0 = time.perf_counter()
    rows: list[dict] | None = None
    ctx = None
    try:
        rows, why, ctx = replay_via_loop(settings, path, ticks, args.quiet, plan)
    finally:
        shutdown(ctx)
    if rows is None:
        print(f"REPLAY FAILED: the SPEC h.4 driver (server.app.build_context + SimulationLoop) is "
              f"unavailable: {why}. Install the runtime pins: "
              f"py -3 -m pip install -r backend\\requirements.txt")
        return 1
    wall_s = time.perf_counter() - t0

    replayed_spikes = sum(int(r["total_spikes"]) for r in rows)
    logged_spikes = sum(int(t.get("total_spikes") or 0) for t in ticks)
    if not args.quiet:
        print(f"replayed  {len(rows)} ticks in {wall_s:.2f} s "
              f"({len(rows) * settings.tick_ms / 1000.0:.1f} s of sim)")
        print(f"spikes    replay {replayed_spikes} vs log {logged_spikes}"
              f" (delta {replayed_spikes - logged_spikes})")
        print(f"final     pose ({rows[-1]['x']:.1f}, {rows[-1]['y']:.1f}) mood {rows[-1]['mood']} "
              f"mode {rows[-1]['mode']}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        try:
            n = write_out(out_path, header, rows)
            print(f"wrote     {out_path} ({n} tick lines, replayable with --assert)")
        except OSError as exc:
            print(f"REPLAY FAILED: could not write {out_path} ({exc})")
            return 1

    if args.do_assert:
        ok, message = compare(ticks, rows)
        if not ok:
            print(f"REPLAY MISMATCH: {message}")
            if ambiguous:
                print(f"hint: the log's whole-millisecond brain clock cannot express this run's "
                      f"{settings.dt_ms:g} ms stepping exactly (see the WARN above). Re-record with "
                      f"FLY_REALTIME=0, or with FLY_DT_MS=1.0, for a comparison that can be bit-exact.")
            return 1
        pose_ok, pose_msg = compare_pose(ticks, rows)
        if pose_ok is None:
            print(f"[WARN] pose      {pose_msg}")
            print(f"REPLAY OK {len(rows)} ticks ({message}; fly pose NOT compared, see the WARN above)")
            return 0
        if not pose_ok:
            print(f"REPLAY MISMATCH: {pose_msg}")
            return 1
        print(f"REPLAY OK {len(rows)} ticks ({message}; {pose_msg})")
        return 0

    print(f"REPLAY DONE {len(rows)} ticks (use --assert to compare against the log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
