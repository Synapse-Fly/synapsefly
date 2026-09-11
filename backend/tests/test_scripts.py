"""The operator scripts a new user runs first: ``smoke.py``, ``selftest.py``, ``replay.py``, ``dev.ps1``,
and the repo-root ``.env.example`` (SPEC h.3, h.4, i.4, i.5).

Every script is exercised as a real subprocess (``py -3 scripts/<name>.py``) from a directory that is *not*
the repo root, with a hermetic environment: ``FLY_DATA_DIR`` / ``FLY_OUT_DIR`` point into ``tmp_path`` and
every secret is stripped, so the tests prove the scripts are offline, CWD-independent and never touch the
developer's ``data/`` or ``out/``. ``--fast`` keeps ``smoke`` and ``selftest`` at N = 4000 with
``FLY_REALTIME=0``.

Markers: every test whose subprocess builds or loads a connectome is ``slow``. A fresh interpreter plus a
4k graph is ~0.4-1.5 s and there are enough of them to matter against the SPEC h budget of 25 s for
``-m "not slow"`` (shared with 600+ other tests) and 90 s for the full run. What stays unmarked is the
cheap surface: ``-h``, a bad option, a missing file, ``.env.example`` and ``dev.ps1`` (text + PowerShell
parse + the health predicate, no Python subprocess at all).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
SMOKE = SCRIPTS / "smoke.py"
SELFTEST = SCRIPTS / "selftest.py"
REPLAY = SCRIPTS / "replay.py"
DEV_PS1 = SCRIPTS / "dev.ps1"
ENV_EXAMPLE = REPO / ".env.example"

#: names that must never leak into a script run (the scripts must work with zero accounts)
SECRET_KEYS = ("ANTHROPIC_API_KEY", "X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN",
               "X_ACCESS_TOKEN_SECRET", "NEUPRINT_APPLICATION_CREDENTIALS")

TIMEOUT_S = 180.0


# ==================================================================================================
# helpers
# ==================================================================================================


def script_env(tmp: Path, **extra: str) -> dict[str, str]:
    """A hermetic environment: tmp data/out, offline switches, no secrets, ASCII-safe stdout."""
    import os

    env = {k: v for k, v in os.environ.items() if k not in SECRET_KEYS and not k.startswith("FLY_")}
    env.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "FLY_DATA_DIR": str(tmp / "data"),
        "FLY_OUT_DIR": str(tmp / "out"),
        "FLY_N_NEURONS": "4000",
        "FLY_REALTIME": "0",
        "FLY_MARKET": "sim",
        "FLY_LLM": "dryrun",
        "FLY_X": "dryrun",
        "FLY_SESSION_LOG": "0",
        "FLY_LOG_LEVEL": "WARNING",
    })
    env.update(extra)
    return env


def run_script(script: Path, args: list[str], tmp: Path, cwd: Path | None = None,
               **extra_env: str) -> subprocess.CompletedProcess:
    """Run ``py -3 <script> <args>`` from ``cwd`` (default: the user's home, i.e. never the repo)."""
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.run(cmd, cwd=str(cwd or Path.home()), env=script_env(tmp, **extra_env),
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=TIMEOUT_S)
    return proc


def explain(proc: subprocess.CompletedProcess) -> str:
    tail_out = "\n".join(proc.stdout.splitlines()[-30:])
    tail_err = "\n".join(proc.stderr.splitlines()[-15:])
    return f"rc={proc.returncode}\n--- stdout ---\n{tail_out}\n--- stderr ---\n{tail_err}"


@pytest.fixture(scope="module")
def script_tmp(tmp_path_factory) -> Path:
    """One data/out sandbox for the whole module, so the 4k connectome is built once and cached."""
    return tmp_path_factory.mktemp("scripts")


@pytest.fixture(scope="module")
def smoke_run(script_tmp: Path) -> subprocess.CompletedProcess:
    return run_script(SMOKE, ["--fast"], script_tmp)


@pytest.fixture(scope="module")
def selftest_run(script_tmp: Path) -> subprocess.CompletedProcess:
    return run_script(SELFTEST, ["--fast"], script_tmp)


def make_session_log(path: Path, env_overrides: dict[str, str], n_ticks: int = 20,
                     total_spikes: int = 0, t_ms: list[int] | None = None) -> Path:
    """A minimal but valid session log: header + ``n_ticks`` tick lines with no market and no inputs.

    ``t_ms`` replaces the default uniform brain clock (``i * tick_ms``) so a test can hand the script a
    clock that no real run could have produced.
    """
    from flybrain.config import load_settings, redacted

    settings = load_settings(env=dict(env_overrides), dotenv=None)
    path.parent.mkdir(parents=True, exist_ok=True)
    wall = 1_789_000_000.0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        header = {"kind": "header", "version": 1, "created": "2026-09-11T00:00:00+00:00",
                  "run_id": settings.run_id, "connectome_key": settings.connectome_key,
                  "settings": redacted(settings)}
        fh.write(json.dumps(header, default=str) + "\n")
        for i in range(1, n_ticks + 1):
            row = {"kind": "tick", "seq": i,
                   "t_ms": (t_ms[i - 1] if t_ms is not None else i * int(settings.tick_ms)),
                   "wall": round(wall + i * settings.tick_ms / 1000.0, 3),
                   "total_spikes": int(total_spikes), "market": None, "trades": [], "pokes": [],
                   "commands": []}
            fh.write(json.dumps(row) + "\n")
    return path


def record_paced_log(tmp: Path, plan: tuple[int, ...] = (50, 50, 42, 36, 44, 50),
                     n_ticks: int = 24) -> tuple[Path, tuple[float, float]]:
    """Record a *real* session log whose per-tick LIF step count varies, and return (path, final pose).

    This is what ``FLY_REALTIME=1`` -- the section-b default, i.e. every log a user records by following
    SPEC i.5 -- produces: ``SimulationLoop`` runs ``round(steps_per_tick * speed)`` steps per tick and
    drops ``speed`` as soon as a tick costs more than 80 % of the tick budget, so the logged ``t_ms``
    column advances in uneven jumps. Driving ``tick_once()`` with ``steps_per_tick`` set per tick
    reproduces exactly that shape deterministically and in a second, instead of needing a 25 s realtime
    run at N = 20000 to make the pacer move.
    """
    from flybrain.config import load_settings
    from flybrain.server.app import build_context

    env = {k: v for k, v in script_env(tmp).items() if k.startswith("FLY_")}
    env["FLY_SESSION_LOG"] = "1"
    settings = load_settings(env=env, dotenv=None)
    ctx = build_context(settings)
    ctx.loop.agent = None                       # a recording must not tweet
    try:
        ctx.loop.startup()
        for i in range(n_ticks):
            ctx.loop.steps_per_tick = int(plan[i % len(plan)])
            ctx.loop.tick_once()
        path = Path(ctx.session_log.path)
        pose = (round(float(ctx.body.x), 1), round(float(ctx.body.y), 1))
    finally:
        ctx.close()
    return path, pose


@pytest.fixture(scope="module")
def paced_log(script_tmp: Path) -> tuple[Path, tuple[float, float]]:
    """One paced recording for the whole module (building it twice buys nothing)."""
    return record_paced_log(script_tmp, n_ticks=24)


def replay_module():
    """``scripts/replay.py`` imported as a module (for unit tests of its helpers)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("flybrain_replay_script", REPLAY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ==================================================================================================
# scripts/smoke.py (SPEC h.3)
# ==================================================================================================


@pytest.mark.slow
def test_smoke_fast_exits_zero(smoke_run: subprocess.CompletedProcess) -> None:
    """``smoke.py --fast`` passes every SPEC h.3 assertion on a 4k brain."""
    assert smoke_run.returncode == 0, explain(smoke_run)
    assert "SMOKE OK" in smoke_run.stdout, explain(smoke_run)
    assert "FAIL" not in smoke_run.stdout, explain(smoke_run)


@pytest.mark.slow
def test_smoke_prints_the_four_phase_rows(smoke_run: subprocess.CompletedProcess) -> None:
    """One table row per SPEC h.3 phase, with the documented columns."""
    out = smoke_run.stdout
    # SPEC h.3 names the columns: a reader grepping the spec's words must find them in the output
    for column in ("phase", "steps", "mean Hz", "active_frac_max", "MN9 Hz", "DNp01 spikes",
                   "first GF ms", "ms/step"):
        assert column in out, f"missing column {column!r}\n{explain(smoke_run)}"
    rows = {line.split()[0] for line in out.splitlines() if line[:1].isalpha() and len(line.split()) >= 8}
    for phase in ("rest", "sugar", "loom", "recover"):
        assert phase in rows, f"missing phase row {phase!r}\n{explain(smoke_run)}"


@pytest.mark.slow
def test_smoke_prints_rtf_and_every_gate(smoke_run: subprocess.CompletedProcess) -> None:
    """RTF plus the named checks of SPEC h.3 / g.6, each with its measured value."""
    out = smoke_run.stdout
    assert "RTF = 2000*1" in out, explain(smoke_run)
    for needle in ("rest mean rate 1-5 Hz", "rest active_frac_max <= 0.02", "rest DNp01 spikes == 0",
                   "sugar feed_mn >= 20 Hz", "sugar feed_mn >= 5x rest", "sugar sugar2_exc >= 20 Hz",
                   "loom first DNp01 spike <= 20 ms", "loom ttmn spike <= 30 ms",
                   "recover mean rate back at rest", "no runaway", "RTF >= 1.5x"):
        assert needle in out, f"missing check {needle!r}\n{explain(smoke_run)}"
    assert out.count("[PASS]") >= 11, explain(smoke_run)


@pytest.mark.slow
def test_smoke_rest_gf_gate_is_the_literal_zero_in_both_weight_modes(
        smoke_run: subprocess.CompletedProcess, script_tmp: Path) -> None:
    """SPEC h.3 phase 1 asks for ``0`` DNp01 spikes at rest. The bound is enforced literally and hard
    everywhere -- small brain, default brain, calibrated and literature weights -- with no ``<= 2``
    allowance and no ``[WARN]`` escape hatch (finding: smoke.py weakened this gate)."""
    assert "[PASS] rest DNp01 spikes == 0" in smoke_run.stdout, explain(smoke_run)
    assert "DNp01 spikes <=" not in smoke_run.stdout, explain(smoke_run)

    lit = run_script(SMOKE, ["--fast", "--weights", "literature", "--no-rtf-check"], script_tmp)
    assert lit.returncode == 0, explain(lit)
    assert "[PASS] rest DNp01 spikes == 0" in lit.stdout, explain(lit)
    assert "DNp01 spikes <=" not in lit.stdout, explain(lit)
    assert "[WARN]" not in lit.stdout, explain(lit)


@pytest.mark.slow
def test_smoke_full_n_enforces_the_literal_h3_bounds(script_tmp: Path) -> None:
    """At the SPEC default N (20000, calibrated) the only remaining tolerance is gone: the recover mean
    rate must be back below a flat 5.0 Hz with no small-N headroom, and the resting DNp01 gate is the
    literal ``== 0`` (it is that at every N). This pins the strict path that ``--fast`` (N=4000)
    deliberately relaxes by +1.0 Hz for the recover gate only."""
    proc = run_script(SMOKE, ["--n", "20000", "--no-rtf-check"], script_tmp)
    assert proc.returncode == 0, explain(proc)
    assert "SMOKE OK" in proc.stdout and "FAIL" not in proc.stdout, explain(proc)
    assert "2000 steps" in proc.stdout, explain(proc)
    assert "[PASS] rest DNp01 spikes == 0" in proc.stdout, explain(proc)
    recover_line = next((ln for ln in proc.stdout.splitlines()
                         if "recover mean rate back at rest" in ln), "")
    assert recover_line, explain(proc)
    assert "limit 5.00 Hz" in recover_line, recover_line          # the literal SPEC 5.0 Hz, not rest+slack
    assert "small-N headroom" not in recover_line, recover_line


@pytest.mark.slow
def test_smoke_ran_2000_steps(smoke_run: subprocess.CompletedProcess) -> None:
    assert "2000 steps" in smoke_run.stdout, explain(smoke_run)


def test_smoke_help_exits_zero(script_tmp: Path) -> None:
    proc = run_script(SMOKE, ["-h"], script_tmp)
    assert proc.returncode == 0, explain(proc)
    for flag in ("--n", "--seed", "--dt", "--weights", "--fast", "--no-rtf-check"):
        assert flag in proc.stdout, f"{flag} not documented\n{explain(proc)}"


def test_smoke_rejects_a_bad_option(script_tmp: Path) -> None:
    """A typo fails fast with a one-line reason naming the variable, not a traceback."""
    proc = run_script(SMOKE, ["--fast", "--dt", "0.37"], script_tmp)
    assert proc.returncode == 1, explain(proc)
    assert "SMOKE FAILED" in proc.stdout, explain(proc)
    assert "FLY_DT_MS" in proc.stdout, explain(proc)
    assert "Traceback" not in proc.stderr, explain(proc)


def test_smoke_rejects_a_bad_environment_value(script_tmp: Path) -> None:
    """A broken ``FLY_*`` value in the environment / ``.env`` is a clear failure, not silence.

    Finding: the script used to pass only FLY_DATA_DIR/FLY_OUT_DIR through, so ``FLY_N_NEURONS=abc``
    (and ``FLY_N_NEURONS=8000`` on a weak laptop, per the i.4 comment) were both ignored.
    """
    proc = run_script(SMOKE, [], script_tmp, FLY_N_NEURONS="abc")
    assert proc.returncode == 1, explain(proc)
    assert "SMOKE FAILED" in proc.stdout and "FLY_N_NEURONS" in proc.stdout, explain(proc)
    assert "Traceback" not in proc.stderr, explain(proc)


@pytest.mark.slow
def test_smoke_honours_fly_n_neurons_from_the_environment(script_tmp: Path) -> None:
    """``FLY_N_NEURONS`` steers the run when no flag is given; an explicit flag still wins."""
    from_env = run_script(SMOKE, ["--no-rtf-check"], script_tmp, FLY_N_NEURONS="5000")
    assert from_env.returncode == 0, explain(from_env)
    assert "smoke  n=5000" in from_env.stdout, explain(from_env)
    assert "FLY_N_NEURONS=5000" in from_env.stdout, explain(from_env)    # the provenance is printed

    flag_wins = run_script(SMOKE, ["--fast", "--no-rtf-check"], script_tmp, FLY_N_NEURONS="5000")
    assert flag_wins.returncode == 0, explain(flag_wins)
    assert "smoke  n=4000" in flag_wins.stdout, explain(flag_wins)


# ==================================================================================================
# scripts/selftest.py (SPEC h.3, i.6)
# ==================================================================================================


@pytest.mark.slow
def test_selftest_fast_exits_zero(selftest_run: subprocess.CompletedProcess) -> None:
    """``selftest.py --fast`` is green on a machine with no accounts and no network."""
    assert selftest_run.returncode == 0, explain(selftest_run)
    assert "SELFTEST OK" in selftest_run.stdout, explain(selftest_run)


@pytest.mark.slow
def test_selftest_prints_the_five_gates(selftest_run: subprocess.CompletedProcess) -> None:
    """The SPEC g.6 gates, each PASS with its number, inside the Win95 ASCII box."""
    out = selftest_run.stdout
    for gate in ("1. rest 1-5 Hz", "2. sugar -> MN9 >= 20 Hz", "3. loom -> GF <= 20 ms",
                 "4. no runaway", "5. PFL3 contralateral"):
        assert f"[PASS] {gate}" in out, f"gate not PASS: {gate!r}\n{explain(selftest_run)}"
    assert "+==[ SynapseFly / FlyBrain  selftest ]" in out, explain(selftest_run)
    assert "[FAIL]" not in out, explain(selftest_run)


@pytest.mark.slow
def test_selftest_prints_rtf_gain_and_motion(selftest_run: subprocess.CompletedProcess) -> None:
    """RTF, the gain used, and the headless motion measurements of SPEC h.3."""
    out = selftest_run.stdout
    assert "RTF >= 2x (floor 1x)" in out, explain(selftest_run)
    assert "gain" in out and "gain_default" in out, explain(selftest_run)
    assert "[PASS] 6. wander_floor == 0" in out, explain(selftest_run)
    assert "[PASS] 7. mean speed >= 15 px/s" in out, explain(selftest_run)
    assert "8. visited >= 3 quadrants" in out, explain(selftest_run)
    # the quadrant line carries the travelled path box, so no separate span metric is needed ...
    assert "path box" in out, explain(selftest_run)
    # ... and nothing outside SPEC h.3 may hold the exit code (finding: "explored canvas span")
    assert "explored canvas span" not in out, explain(selftest_run)


@pytest.mark.slow
def test_selftest_prints_a_dry_run_tweet(selftest_run: subprocess.CompletedProcess) -> None:
    """The full agent path runs with ``reason='manual'`` and nothing is posted."""
    out = selftest_run.stdout
    assert "[PASS] 9. dry-run tweet" in out, explain(selftest_run)
    assert "dry-run tweet (reason=manual, nothing was posted)" in out, explain(selftest_run)
    assert "dry_run=True" in out, explain(selftest_run)


@pytest.mark.slow
def test_selftest_writes_only_into_the_sandbox(selftest_run: subprocess.CompletedProcess,
                                               script_tmp: Path) -> None:
    """FLY_DATA_DIR / FLY_OUT_DIR are honoured, so a run never touches the repo's data/ or out/."""
    assert selftest_run.returncode == 0, explain(selftest_run)
    assert (script_tmp / "data").is_dir()
    assert (script_tmp / "out").is_dir()
    assert str(script_tmp) in selftest_run.stdout


@pytest.mark.slow
def test_selftest_output_is_ascii(selftest_run: subprocess.CompletedProcess) -> None:
    """SPEC 0.1: the cp1254 console must never see a non-ASCII byte."""
    assert selftest_run.stdout.isascii(), "non-ASCII in stdout: " + repr(
        [c for c in selftest_run.stdout if not c.isascii()][:10])


@pytest.mark.slow
def test_selftest_is_green_at_another_seed(script_tmp: Path) -> None:
    """The exit code must not depend on the seed.

    Finding: an undocumented hard "explored canvas span" gate failed at seeds 7 and 2024 (and at 42 with
    ``--fast``) while every SPEC g.6 gate passed. Only checks SPEC h.3 defines may fail a run, so a
    correct brain at an ordinary seed is green -- with the quadrant count reported as an advisory.
    """
    proc = run_script(SELFTEST, ["--fast", "--no-tweet", "--seed", "7", "--motion-s", "6"],
                      script_tmp)
    assert proc.returncode == 0, explain(proc)
    assert "SELFTEST OK" in proc.stdout, explain(proc)
    assert "[FAIL]" not in proc.stdout, explain(proc)
    assert "8. visited >= 3 quadrants" in proc.stdout, explain(proc)


def test_selftest_help_exits_zero(script_tmp: Path) -> None:
    proc = run_script(SELFTEST, ["-h"], script_tmp)
    assert proc.returncode == 0, explain(proc)
    for flag in ("--n", "--seed", "--source", "--dir", "--fast", "--no-motion", "--no-tweet"):
        assert flag in proc.stdout, f"{flag} not documented\n{explain(proc)}"


@pytest.mark.slow
def test_selftest_missing_csv_dir_fails_with_a_reason(script_tmp: Path) -> None:
    """A missing connectome directory is a one-line failure, not a traceback."""
    proc = run_script(SELFTEST, ["--fast", "--source", "csv", "--dir",
                                 str(script_tmp / "not-there")], script_tmp)
    assert proc.returncode == 1, explain(proc)
    assert "SELFTEST FAILED" in proc.stdout, explain(proc)
    assert "Traceback" not in proc.stderr, explain(proc)


# ==================================================================================================
# scripts/replay.py (SPEC h.4)
# ==================================================================================================


def test_replay_skips_when_there_is_no_session_log(tmp_path: Path) -> None:
    """A fresh checkout has no session log: say so clearly and exit 0 (nothing is broken)."""
    proc = run_script(REPLAY, [], tmp_path, FLY_DATA_DIR=str(tmp_path / "data"))
    assert proc.returncode == 0, explain(proc)
    assert "REPLAY SKIP" in proc.stdout, explain(proc)
    assert "FLY_SESSION_LOG=1" in proc.stdout, explain(proc)


def test_replay_missing_file_is_a_usage_error(tmp_path: Path) -> None:
    proc = run_script(REPLAY, [str(tmp_path / "nope.jsonl")], tmp_path)
    assert proc.returncode == 2, explain(proc)
    assert "REPLAY FAILED" in proc.stdout, explain(proc)


@pytest.mark.slow
def test_replay_round_trip_is_bit_exact_and_compares_the_pose(script_tmp: Path) -> None:
    """``--out`` re-records the replay; replaying that file with ``--assert`` must be identical.

    This is the determinism proof that needs no running server: two independent runs of the whole tick
    pipeline over the same inputs produce the same ``total_spikes`` per tick and the same pose. The
    re-recorded log is also the only kind that carries ``x``/``y``, so it is where the SPEC h.4 pose
    comparison actually runs -- asserted here so the comparison cannot rot into a permanent no-op.
    """
    env = script_env(script_tmp)
    log = make_session_log(script_tmp / "sessions" / "stub.jsonl", env, n_ticks=20)
    rec = script_tmp / "sessions" / "rec.jsonl"

    first = run_script(REPLAY, [str(log), "--out", str(rec)], script_tmp)
    assert first.returncode == 0, explain(first)
    assert "REPLAY DONE 20 ticks" in first.stdout, explain(first)
    assert rec.is_file()

    rows = [json.loads(line) for line in rec.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "header"
    ticks = [r for r in rows if r.get("kind") == "tick"]
    assert len(ticks) == 20
    assert sum(t["total_spikes"] for t in ticks) > 0, "the replay produced no spikes at all"
    assert all("x" in t and "y" in t for t in ticks), "the re-recording must carry the pose"

    second = run_script(REPLAY, [str(rec), "--assert"], script_tmp)
    assert second.returncode == 0, explain(second)
    assert "REPLAY OK 20 ticks" in second.stdout, explain(second)
    assert "total_spikes identical" in second.stdout, explain(second)
    # the pose IS compared against a log that has one, and no warning is printed
    assert "final pose" in second.stdout, explain(second)
    assert "[WARN] pose" not in second.stdout, explain(second)


@pytest.mark.slow
def test_replay_assert_reports_the_first_diverging_tick(script_tmp: Path) -> None:
    """The stub log claims 0 spikes per tick, so ``--assert`` must fail and name the tick."""
    env = script_env(script_tmp)
    log = make_session_log(script_tmp / "sessions" / "zero.jsonl", env, n_ticks=6, total_spikes=0)
    proc = run_script(REPLAY, [str(log), "--assert"], script_tmp)
    assert proc.returncode == 1, explain(proc)
    assert "REPLAY MISMATCH" in proc.stdout, explain(proc)
    assert "total_spikes" in proc.stdout and "!= 0 logged" in proc.stdout, explain(proc)


def test_replay_detects_a_connectome_key_mismatch(script_tmp: Path) -> None:
    """A log recorded against a different graph can never replay bit-exactly: say why."""
    env = script_env(script_tmp)
    log = make_session_log(script_tmp / "sessions" / "otherkey.jsonl", env, n_ticks=3)
    rows = log.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header["connectome_key"] = "deadbeefcafe"
    rows[0] = json.dumps(header, default=str)
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = run_script(REPLAY, [str(log), "--assert"], script_tmp)
    assert proc.returncode == 1, explain(proc)
    assert "connectome_key mismatch" in proc.stdout, explain(proc)


@pytest.mark.slow
def test_replay_reproduces_a_log_recorded_with_a_paced_brain_clock(
        script_tmp: Path, paced_log: tuple[Path, tuple[float, float]]) -> None:
    """THE regression test for the replay blocker.

    A log recorded the documented way (``FLY_REALTIME=1``, the section-b default) does not step the brain
    the same number of times every tick: ``SimulationLoop`` scales the steps by its adaptive ``speed`` and
    ``SessionLog`` persists neither the steps nor the speed. ``replay.py`` used to assume a fixed
    ``steps_per_tick``, so every tick after the first speed change diverged and ``--assert`` exited 1 on
    any log a user actually recorded. It must now recover the per-tick stepping from the log's ``t_ms``
    column and reproduce the run exactly -- including the final pose, checked here against the live body
    because the session log itself carries none.
    """
    log, pose = paced_log
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    ticks = [r for r in rows if r.get("kind") == "tick"]
    assert len(ticks) == 24
    deltas = {b["t_ms"] - a["t_ms"] for a, b in zip(ticks, ticks[1:])}
    assert len(deltas) > 1, f"the recording is not paced, the test would be vacuous: {deltas}"

    proc = run_script(REPLAY, [str(log), "--assert"], script_tmp)
    assert proc.returncode == 0, explain(proc)
    assert "REPLAY OK 24 ticks" in proc.stdout, explain(proc)
    assert "total_spikes identical" in proc.stdout, explain(proc)
    assert "reconstructed per tick" in proc.stdout, explain(proc)      # and it says how
    assert f"final     pose ({pose[0]:.1f}, {pose[1]:.1f})" in proc.stdout, explain(proc)
    # SPEC h.4 wants the final pose compared, SPEC c.25's write_tick records none: the verdict of a
    # server-written log must say the pose was skipped instead of folding it into the success message.
    assert "[WARN] pose" in proc.stdout, explain(proc)
    assert "NOT compared" in proc.stdout, explain(proc)


def test_replay_steps_plan_recovers_the_recorded_stepping() -> None:
    """``steps_plan`` unit test: uniform, paced, and a brain clock that cannot be whole steps."""
    mod = replay_module()
    uniform = [{"t_ms": 50 * i} for i in range(1, 6)]
    plan, note = mod.steps_plan(uniform, 1.0, 50)
    assert plan == [50] * 5 and "uniform" in note

    paced = [{"t_ms": t} for t in (50, 100, 142, 178, 222)]
    plan, note = mod.steps_plan(paced, 1.0, 50)
    assert plan == [50, 50, 42, 36, 44], plan
    assert "reconstructed per tick" in note and "FLY_REALTIME=1" in note

    half_dt = [{"t_ms": t} for t in (25, 50, 71)]
    plan, _note = mod.steps_plan(half_dt, 0.5, 50)
    assert plan == [50, 50, 42], plan
    # brain time is an int, so a paced sub-ms run cannot be split exactly: the script must say so
    assert mod.plan_is_ambiguous(plan, 0.5) is True
    assert mod.plan_is_ambiguous(plan, 1.0) is False            # whole-ms dt is exact
    assert mod.plan_is_ambiguous([50, 50, 50], 0.5) is False    # uniform stepping is exact too

    for broken in ([{"t_ms": 50}, {"t_ms": 50}], [{"t_ms": 50}, {"t_ms": 40}], [{"t_ms": 0}]):
        plan, note = mod.steps_plan(broken, 1.0, 50)
        assert plan is None, broken
        assert "brain clock" in note, note


def test_replay_refuses_to_assert_on_a_broken_brain_clock(script_tmp: Path) -> None:
    """A ``t_ms`` column that cannot be whole LIF steps is refused with a reason, never "REPLAY OK"."""
    env = script_env(script_tmp)
    log = make_session_log(script_tmp / "sessions" / "badclock.jsonl", env, n_ticks=4,
                           t_ms=[50, 100, 100, 150])         # a tick that advanced nothing
    proc = run_script(REPLAY, [str(log), "--assert"], script_tmp)
    assert proc.returncode == 1, explain(proc)
    assert "REPLAY FAILED" in proc.stdout, explain(proc)
    assert "FLY_REALTIME" in proc.stdout, explain(proc)
    assert "REPLAY OK" not in proc.stdout, explain(proc)


def test_replay_pose_comparison_is_not_silently_skipped() -> None:
    """``compare_pose`` returns ``None`` (= "not compared"), never a bare pass, on a poseless log.

    Finding: it used to return ``True`` with an explanatory string that the verdict swallowed, so a pose
    divergence with identical spike counts printed REPLAY OK.
    """
    mod = replay_module()
    poseless = [{"t_ms": 50, "total_spikes": 1}]
    ok, msg = mod.compare_pose(poseless, [{"x": 1.0, "y": 2.0}])
    assert ok is None and "carries no fly pose" in msg

    ok, _msg = mod.compare_pose([{"x": 10.0, "y": 20.0}], [{"x": 10.0, "y": 20.0}])
    assert ok is True
    ok, msg = mod.compare_pose([{"x": 10.0, "y": 20.0}], [{"x": 11.0, "y": 20.0}])
    assert ok is False and "final pose" in msg


@pytest.mark.slow
def test_replay_warns_when_the_header_has_no_settings(script_tmp: Path) -> None:
    """A damaged header silently replayed with section-b defaults used to pass as REPLAY OK."""
    env = script_env(script_tmp)
    log = make_session_log(script_tmp / "sessions" / "noheadersettings.jsonl", env, n_ticks=3)
    rows = log.read_text(encoding="utf-8").splitlines()
    header = json.loads(rows[0])
    header.pop("settings", None)
    rows[0] = json.dumps(header, default=str)
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")

    proc = run_script(REPLAY, [str(log), "--ticks", "1"], script_tmp)
    assert "[WARN] header" in proc.stdout, explain(proc)
    assert "section-b defaults" in proc.stdout, explain(proc)


def test_replay_discovery_uses_the_dotenv_precedence(tmp_path: Path, monkeypatch) -> None:
    """``ambient_env`` reads ``<repo>/.env`` and lets the real environment win (SPEC section b).

    Finding: the no-path probe used ``os.environ`` only, so a ``FLY_DATA_DIR`` set in ``.env`` (and
    nowhere else) made the script scan the wrong ``sessions`` directory.
    """
    mod = replay_module()
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    (fake_repo / ".env").write_text("FLY_DATA_DIR=" + str(tmp_path / "from_dotenv") + "\n"
                                    "FLY_SEED=4242\n", encoding="utf-8")
    monkeypatch.setattr(mod, "REPO", fake_repo)
    monkeypatch.delenv("FLY_DATA_DIR", raising=False)
    monkeypatch.delenv("FLY_SEED", raising=False)

    env = mod.ambient_env()
    assert env["FLY_DATA_DIR"] == str(tmp_path / "from_dotenv")
    assert env["FLY_SEED"] == "4242"

    monkeypatch.setenv("FLY_DATA_DIR", str(tmp_path / "from_environ"))
    assert mod.ambient_env()["FLY_DATA_DIR"] == str(tmp_path / "from_environ")


def test_replay_discovery_prefers_a_log_of_this_configuration(tmp_path: Path) -> None:
    """When no path is given, a newer log recorded against another graph is skipped, not failed on."""
    mod = replay_module()
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True)
    older = sessions / "a-older.jsonl"
    newer = sessions / "b-newer.jsonl"
    older.write_text(json.dumps({"kind": "header", "connectome_key": "mine"}) + "\n", encoding="utf-8")
    newer.write_text(json.dumps({"kind": "header", "connectome_key": "other"}) + "\n", encoding="utf-8")
    import os as _os
    _os.utime(older, (1_000_000, 1_000_000))
    _os.utime(newer, (2_000_000, 2_000_000))

    found, why = mod.newest_session(tmp_path, "mine")
    assert found == older and "connectome_key is mine" in why
    found, why = mod.newest_session(tmp_path, "")
    assert found == newer and why == ""
    assert mod.newest_session(tmp_path / "empty", "mine") == (None, "")


def test_replay_help_exits_zero(script_tmp: Path) -> None:
    proc = run_script(REPLAY, ["-h"], script_tmp)
    assert proc.returncode == 0, explain(proc)
    for flag in ("--assert", "--out", "--ticks"):
        assert flag in proc.stdout, f"{flag} not documented\n{explain(proc)}"


# ==================================================================================================
# .env.example (SPEC i.4)
# ==================================================================================================


def test_env_example_exists_and_is_ascii() -> None:
    assert ENV_EXAMPLE.is_file(), f"{ENV_EXAMPLE} is missing (SPEC i.4)"
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert text.isascii(), "SPEC 0.1: .env.example must be ASCII-only"
    assert "\r\n" not in text or text.count("\r\n") == text.count("\n"), "mixed line endings"


def test_env_example_covers_every_documented_variable() -> None:
    """Every section-b variable and every secret has a line; nothing unknown is present."""
    from flybrain.config import ENV_BY_NAME, ENV_VARS, parse_dotenv

    keys = set(parse_dotenv(ENV_EXAMPLE))
    unknown = sorted(keys - set(ENV_BY_NAME))
    assert not unknown, f"keys unknown to Settings: {unknown}"
    documented = {v.name for v in ENV_VARS if v.field or v.kind == "secret"}
    missing = sorted(documented - keys)
    assert not missing, f"missing from .env.example: {missing}"


def test_env_example_loads_as_is_with_the_section_b_defaults() -> None:
    """Copying the file to .env must reproduce the defaults exactly -- including no inline comments.

    ``config.parse_dotenv`` takes everything after the first ``=`` as the value (SPEC section b: no shell
    semantics), so a trailing ``# comment`` would become part of the value and blow up on the first run.
    """
    from flybrain.config import load_settings, parse_dotenv

    values = parse_dotenv(ENV_EXAMPLE)
    for key, value in values.items():
        assert "#" not in value, f"{key} has an inline comment in its value: {value!r}"
    loaded = load_settings(env=dict(values), dotenv=None)
    default = load_settings(env={}, dotenv=None)
    for field in ("connectome_source", "n_neurons", "mean_outdeg", "synth_weights", "subset",
                  "min_weight", "dt_ms", "seed", "backend", "gain", "noise_mu", "noise_sigma",
                  "drive_mode", "tick_hz", "realtime", "market", "chain", "dex_poll_s", "sim_regime_s",
                  "llm", "llm_model", "llm_json", "llm_fallbacks", "tweet_lang", "x_mode",
                  "tweet_cooldown_s", "tweet_reason_cooldown_s", "tweets_per_day", "port", "host",
                  "cors_origins", "canvas_w", "canvas_h", "walls", "raster_per_region", "raster_cap",
                  "mood_feedback", "explore_baseline", "wander_sigma", "easter_eggs", "session_log",
                  "log_level"):
        assert getattr(loaded, field) == getattr(default, field), (
            f".env.example changes the default of {field}: "
            f"{getattr(loaded, field)!r} != {getattr(default, field)!r}")


def test_env_example_documents_every_variable_with_a_comment() -> None:
    """Each ``KEY=`` line is preceded by a ``#`` line (SPEC i.4: a one-line comment per variable)."""
    lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    undocumented: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        previous = lines[i - 1].strip() if i else ""
        if not previous.startswith("#"):
            undocumented.append(stripped.split("=")[0])
    assert not undocumented, f"variables without a comment line above them: {undocumented}"


# ==================================================================================================
# scripts/dev.ps1 (SPEC h.4)
# ==================================================================================================


def test_dev_ps1_exists_and_is_ascii() -> None:
    assert DEV_PS1.is_file(), f"{DEV_PS1} is missing (SPEC h.4)"
    assert DEV_PS1.read_text(encoding="utf-8").isascii()


def test_dev_ps1_contract() -> None:
    """The SPEC h.4 contract: PYTHONUTF8, two jobs, both logs, the health wait, the browser, no reload."""
    text = DEV_PS1.read_text(encoding="utf-8")
    assert "PYTHONUTF8" in text
    assert "py -3 backend\\run.py" in text
    assert "npm run dev" in text
    assert "out" in text and "backend.log" in text and "frontend.log" in text
    assert "/api/health" in text
    assert "http://localhost:" in text
    assert "NoFrontend" in text
    assert "Start-Job" in text
    # SPEC h.4: never --reload. The command line handed to the job is a literal, so pin it exactly;
    # --reload may still be *mentioned* ("no --reload", the help text) but never passed.
    assert "Start-DevJob 'backend' $Repo 'py -3 backend\\run.py' $BackendLog" in text
    for lineno, line in enumerate(text.splitlines(), start=1):
        if "--reload" not in line:
            continue
        assert line.lstrip().startswith(("#", ".", "<#")) or "no --reload" in line \
            or "Never uses" in line, f"dev.ps1:{lineno} passes --reload: {line.strip()!r}"


def test_dev_ps1_parses() -> None:
    """PowerShell can parse the script (a syntax error would only show up at 'first five minutes')."""
    import shutil

    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:  # pragma: no cover - non-Windows CI
        pytest.skip("powershell is not on PATH")
    code = (
        "$errs = $null; "
        f"[System.Management.Automation.Language.Parser]::ParseFile('{DEV_PS1}', [ref]$null, [ref]$errs) "
        "| Out-Null; "
        "if ($errs.Count -gt 0) { $errs | ForEach-Object { $_.ToString() }; exit 1 } else { 'PARSE OK' }"
    )
    proc = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", code],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=60.0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PARSE OK" in proc.stdout, proc.stdout + proc.stderr


def powershell_or_skip() -> str:
    import shutil

    exe = shutil.which("powershell") or shutil.which("pwsh")
    if exe is None:  # pragma: no cover - non-Windows CI
        pytest.skip("powershell is not on PATH")
    return exe


def test_dev_ps1_accepts_h() -> None:
    """SPEC h.4 final line: every script accepts ``-h``. It used to be a parameter-binding error."""
    powershell = powershell_or_skip()
    for flag in ("-h", "-Help"):
        proc = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                               "Bypass", "-File", str(DEV_PS1), flag],
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=60.0)
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert "dev.ps1" in proc.stdout and "Usage:" in proc.stdout, out
        assert "-NoFrontend" in proc.stdout, out
        assert proc.stdout.isascii(), out
        assert "cannot be found that matches parameter" not in out, out


def test_dev_ps1_health_predicate_requires_ok_true() -> None:
    """The wait loop must break on ``ok: true``, not on any HTTP 200.

    Finding: a 200 carrying ``{"ok": false}`` let dev.ps1 announce "=== up" and open the browser on a
    backend that reports itself not ok. The predicate is extracted from the shipped file and exercised.
    """
    powershell = powershell_or_skip()
    code = (
        "$errs = $null; "
        f"$ast = [System.Management.Automation.Language.Parser]::ParseFile('{DEV_PS1}', "
        "[ref]$null, [ref]$errs); "
        "$fn = $ast.FindAll({ param($n) "
        "$n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and "
        "$n.Name -eq 'Test-HealthOk' }, $true); "
        "if ($fn.Count -ne 1) { 'NO PREDICATE'; exit 1 }; "
        "Set-StrictMode -Version 2.0; "
        "Invoke-Expression $fn[0].Extent.Text; "
        "'ok_true=' + (Test-HealthOk ('{\"ok\":true,\"rtf\":1.2}' | ConvertFrom-Json)); "
        "'ok_false=' + (Test-HealthOk ('{\"ok\":false,\"rtf\":0}' | ConvertFrom-Json)); "
        "'no_key=' + (Test-HealthOk ('{\"rtf\":1}' | ConvertFrom-Json)); "
        "'null=' + (Test-HealthOk $null)"
    )
    proc = subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-Command", code],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=60.0)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "ok_true=True" in proc.stdout, out
    assert "ok_false=False" in proc.stdout, out
    assert "no_key=False" in proc.stdout, out
    assert "null=False" in proc.stdout, out

    text = DEV_PS1.read_text(encoding="utf-8")
    assert "if (Test-HealthOk $answer) { $healthOk = $true; break }" in text, \
        "the wait loop must use the ok predicate, not the bare response"


# ==================================================================================================
# housekeeping
# ==================================================================================================


@pytest.mark.parametrize("script", [SMOKE, SELFTEST, REPLAY])
def test_scripts_are_ascii_and_compile(script: Path) -> None:
    """ASCII-only source (SPEC 0.1) that byte-compiles without a SyntaxError."""
    source = script.read_text(encoding="utf-8")
    assert source.isascii(), f"{script.name} has non-ASCII characters"
    compile(source, str(script), "exec")


@pytest.mark.parametrize("script", [SMOKE, SELFTEST, REPLAY])
def test_scripts_resolve_paths_from_file(script: Path) -> None:
    """SPEC 0.1: never depend on the CWD -- the repo root comes from ``__file__``."""
    source = script.read_text(encoding="utf-8")
    assert "Path(__file__).resolve().parents[1]" in source
    assert "sys.path.insert(0, str(BACKEND))" in source
