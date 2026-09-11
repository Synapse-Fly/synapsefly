#!/usr/bin/env python
"""Brain smoke test: 2000 steps in four phases with the SPEC h.3 assertions (offline, no server).

Usage::

    py -3 scripts/smoke.py [--n 20000] [--seed 1337] [--dt 1.0] [--weights calibrated]
                           [--fast] [--no-rtf-check] [--no-tonic] [--backend numpy|torch|auto]
                           [--gain auto|FLOAT] [--quiet]

Builds (or loads from the cache) the synthetic connectome, creates a ``LIFEngine`` with noise on and the
SPEC c.10 tonic table applied (exactly what ``SimulationLoop`` does at startup), then runs **2000 steps of
dt** straight through in four phases, printing one table row per phase::

    phase | steps | mean Hz | active_frac_max | MN9 Hz | DNp01 spikes | first GF ms | ms/step

1. ``rest``    steps 0-999,     no drives                     -> 1.0 <= mean <= 5.0 Hz, af_max <= 0.02, GF quiet
2. ``sugar``   steps 1000-1499, ``grn_sugar_labellar`` 100 Hz  -> feed_mn >= 20 Hz and >= 5x rest, sugar2_exc >= 20 Hz
3. ``loom``    steps 1500-1699, ``lc_loom`` 150 Hz             -> first DNp01 spike <= 20 ms, a ttmn spike <= 30 ms
4. ``recover`` steps 1700-1999, no drives                      -> mean back at rest, active_frac <= 0.05 everywhere

Finally prints ``RTF = 2000*dt / wall_ms`` and fails when RTF < 1.5 at N <= 20k (a performance regression
guard; ``--no-rtf-check`` on slow machines). Every failed check prints the measured value and the sizes of
the groups involved. ASCII only, absolute paths resolved from ``__file__``, no network, no API keys, exit 0
on success and 1 on the first failed check (the reason is the last line).

Configuration: flag > environment / ``<repo>/.env`` > the SPEC h.3 default, for the six knobs a flag also
covers (``FLY_N_NEURONS``, ``FLY_SEED``, ``FLY_DT_MS``, ``FLY_SYNTH_WEIGHTS``, ``FLY_BACKEND``,
``FLY_GAIN``) plus ``FLY_DATA_DIR`` / ``FLY_OUT_DIR``; anything taken from the environment is printed. So
the ``FLY_N_NEURONS=8000`` of a weak laptop (the SPEC i.4 comment) is honoured, and a broken value fails
with the section-b message naming the variable instead of being ignored. Everything else is forced offline
-- ``synthetic`` source, ``FLY_REALTIME=0``, sim market, dry-run LLM/X, no session log -- so a populated
API key or a ``csv`` source can never change what a smoke run does.

Every SPEC h.3 bound is enforced literally, at every N and in both weight modes -- including the **0**
DNp01 spikes at rest (measured as 0 at N = 4000 and 20000, seeds 1337/7/2024/99999, calibrated and
literature). Exactly one tolerance exists, confined to the small ``--fast`` brain, and the measured value is
always printed so nothing is hidden:

* SPEC h.3's "mean rate back below 5 Hz" after the drives stop is the literal ``<= 5.0`` Hz at the SPEC
  default N (>= 20000), where the recover tail measures ~3.9-4.1 Hz. A small ``--fast`` brain (N < 20000)
  rests at 4.4-4.9 Hz and so has no headroom for the sampling noise of a 100 ms window; there, and only
  there, the limit becomes ``max(5.0, rest + 1.0)`` (the row says ``small-N headroom``) so the check still
  catches a brain that stays excited without flagging noise.

One deviation from the letter of SPEC h.3, which does not mention it: the engine starts with the SPEC c.10
tonic-current table applied, because ``SimulationLoop`` applies it at startup and a smoke test that does not
is not testing the brain the server runs. It is load-bearing -- without it the resting ``feed_mn`` sits at
~9.5 Hz and phase 2's "5 x the rest value" cannot be reached -- so ``--no-tonic`` is a diagnostic, not a
supported configuration, and says so when used.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from flybrain.config import ENV_BY_NAME, gain_value, load_settings, parse_dotenv   # noqa: E402
from flybrain.connectome.loaders import load_connectome                    # noqa: E402
from flybrain.snn.calibrate import CalibTargets, calibrate_gain, load_calibration   # noqa: E402
from flybrain.snn.engine import LIFEngine, LIFParams, apply_tonic_table    # noqa: E402
from flybrain.snn.monitor import SpikeMonitor                              # noqa: E402

# SPEC h.3 phase plan: (name, steps, drive group, rate Hz)
PHASES: tuple[tuple[str, int, str | None, float], ...] = (
    ("rest", 1000, None, 0.0),
    ("sugar", 500, "grn_sugar_labellar", 100.0),
    ("loom", 200, "lc_loom", 150.0),
    ("recover", 300, None, 0.0),
)
WINDOW_MS = 50.0                 # stepping window for the non-latency phases (one wall tick)
RTF_MIN = 1.5                    # SPEC h.3 performance guard at N = 20k
RTF_GUARD_MAX_N = 20_000         # above this the guard is reported but not enforced
SUGAR_MIN_HZ = 20.0              # SPEC g.6 gate 2
SUGAR_REST_FACTOR = 5.0
SUGAR_REST_FLOOR_HZ = 0.5        # matches tests/test_synthetic.py: 5x max(rest, 0.5)
SUGAR2_MIN_HZ = 20.0
LOOM_GF_MAX_MS = 20.0            # SPEC g.6 gate 3
LOOM_TTMN_MAX_MS = 30.0
RUNAWAY_AF = 0.05                # SPEC g.6 gate 4
REST_GF_MAX_SPIKES = 0           # SPEC h.3 phase 1, literally: "0 DNp01 spikes" (both weight modes)
STRICT_RECOVER_N = 20_000        # SPEC default N: at/above it the recover bound is the literal 5.0 Hz
RECOVER_SLACK_HZ = 1.0           # small-N (< STRICT_RECOVER_N) headroom: limit becomes rest + this slack


def ambient_env() -> dict[str, str]:
    """``<repo>/.env`` then the process environment (the environment wins), known keys only.

    SPEC section b precedence, the same helper ``selftest.py`` / ``replay.py`` use. Only the knobs this
    script also exposes as a flag are taken from it (brain shape and engine); the offline switches
    (market, LLM, X, session log, realtime, connectome source) stay forced, so a populated key or a
    ``csv`` source can never change what a smoke run does.
    """
    env: dict[str, str] = {}
    dotenv = REPO / ".env"
    if dotenv.is_file():
        env.update(parse_dotenv(dotenv))
    for key, value in os.environ.items():
        if key in ENV_BY_NAME:
            env[key] = value
    return {k: v for k, v in env.items() if k in ENV_BY_NAME}


class Check:
    """One named assertion with its measured value (kept so the summary can print every failure).

    Every SPEC h.3 assertion is constructed hard: a violation is a ``[FAIL]`` and exit code 1. ``hard``
    exists for the informational rows (``RTF check skipped``), never to soften a documented bound.
    """

    __slots__ = ("name", "ok", "detail", "hard")

    def __init__(self, name: str, ok: bool, detail: str, hard: bool = True) -> None:
        self.name = name
        self.ok = bool(ok)
        self.detail = detail
        self.hard = bool(hard)

    @property
    def label(self) -> str:
        return "PASS" if self.ok else ("FAIL" if self.hard else "WARN")


class PhaseResult:
    """Everything measured during one phase."""

    __slots__ = ("name", "steps", "brain_ms", "wall_ms", "mean_hz", "af_max", "mn9_hz", "mn9_late_hz",
                 "sugar2_late_hz", "gf_spikes", "first_gf_ms", "first_ttmn_ms", "tail_mean_hz")

    def __init__(self, name: str, steps: int) -> None:
        self.name = name
        self.steps = steps
        self.brain_ms = 0.0
        self.wall_ms = 0.0
        self.mean_hz = 0.0
        self.af_max = 0.0
        self.mn9_hz = 0.0
        self.mn9_late_hz = 0.0
        self.sugar2_late_hz = 0.0
        self.gf_spikes = 0
        self.first_gf_ms: float | None = None
        self.first_ttmn_ms: float | None = None
        self.tail_mean_hz = 0.0

    @property
    def ms_per_step(self) -> float:
        return self.wall_ms / max(self.steps, 1)


def resolve_gain(conn, settings, mode: str) -> tuple[float, str]:
    """The gain the server would use: option > calibration cache > ``meta['gain_default']``.

    ``mode`` is ``auto`` (calibrate only when the weights are not the calibrated default and nothing is
    cached, exactly as SPEC g.6 describes first start), ``force`` (always bisect) or ``never``.
    """
    explicit = gain_value(settings)
    if explicit is not None:
        return float(explicit), "option"
    key = str(settings.connectome_key or "")
    cached = load_calibration(settings, key) if key else None
    default = float(conn.meta.get("gain_default", 1.0) or 1.0)
    if mode != "force" and cached is not None:
        return float(cached.get("gain", default)), "calibration cache"
    needs = mode == "force" or (mode == "auto" and cached is None and abs(default - 1.0) > 1e-9)
    if needs:
        print(f"calib  gain is 'auto' and the weights are not pre-calibrated (gain_default {default:.3f}): "
              f"bisecting once, cached in {settings.data_dir / 'cache'} ...", flush=True)
        try:
            doc = calibrate_gain(conn, settings, force=(mode == "force"))
            return float(doc.get("gain", default)), "calibrate_gain"
        except Exception as exc:  # noqa: BLE001 - fall back to the documented default, never crash
            print(f"calib  failed ({exc.__class__.__name__}: {exc}); using gain_default")
    return default, "meta gain_default"


def _group_size(conn, name: str) -> int:
    idx = conn.groups.get(name)
    return 0 if idx is None else int(idx.size)


def _rate(stats_list, group: str, size: int, ms: float) -> float:
    """Hz per neuron of ``group`` over the given StepStats list."""
    if size <= 0 or ms <= 0.0:
        return 0.0
    spikes = sum(int(s.spike_counts_by_group.get(group, 0)) for s in stats_list)
    return spikes / size / (ms / 1000.0)


def _gf(stats_list) -> int:
    return int(sum(int(s.gf_spikes.get("L", 0)) + int(s.gf_spikes.get("R", 0)) for s in stats_list))


def run_phase(eng: LIFEngine, name: str, steps: int, group: str | None, rate_hz: float,
              sizes: dict[str, int], per_step: bool) -> PhaseResult:
    """Run one phase; ``per_step`` gives 1-step resolution (needed for the loom latency)."""
    res = PhaseResult(name, steps)
    dt = float(eng.dt_ms)
    res.brain_ms = steps * dt
    if group is not None and sizes.get(group, 0) > 0:
        eng.inject(group, rate_hz=rate_hz, duration_ms=res.brain_ms, recruit=1.0, tag=f"smoke:{name}")
    chunk = 1 if per_step else max(1, int(round(WINDOW_MS / dt)))
    stats: list = []
    t0 = time.perf_counter()
    done = 0
    while done < steps:
        k = min(chunk, steps - done)
        stats.append(eng.step(k))
        done += k
    res.wall_ms = (time.perf_counter() - t0) * 1000.0
    if group is not None:
        eng.clear_injections(f"smoke:{name}")

    n = max(eng.n, 1)
    total = sum(int(s.total_spikes) for s in stats)
    res.mean_hz = total / n / (res.brain_ms / 1000.0)
    res.af_max = max((float(s.active_frac_max) for s in stats), default=0.0)
    res.mn9_hz = _rate(stats, "feed_mn", sizes.get("feed_mn", 0), res.brain_ms)
    res.gf_spikes = _gf(stats)

    # first GF / ttmn spike, relative to the phase start (window resolution unless per_step)
    t_start = stats[0].t0_ms if stats else eng.t_ms
    for s in stats:
        off = float(s.t0_ms - t_start)
        if res.first_gf_ms is None and (int(s.gf_spikes.get("L", 0)) + int(s.gf_spikes.get("R", 0))) > 0:
            res.first_gf_ms = off
        if res.first_ttmn_ms is None and int(s.spike_counts_by_group.get("ttmn", 0)) > 0:
            res.first_ttmn_ms = off

    # late window (second half of the phase): the sugar gate measures the last 250 ms of 500
    half = res.brain_ms / 2.0
    late = [s for s in stats if float(s.t0_ms - t_start) >= half - 1e-9]
    late_ms = sum(int(s.n_steps) for s in late) * dt
    res.mn9_late_hz = _rate(late, "feed_mn", sizes.get("feed_mn", 0), late_ms)
    res.sugar2_late_hz = _rate(late, "sugar2_exc", sizes.get("sugar2_exc", 0), late_ms)

    # tail window: the last 100 ms of brain time (the recover gate)
    tail_ms_target = min(100.0, res.brain_ms)
    tail: list = []
    acc = 0.0
    for s in reversed(stats):
        tail.append(s)
        acc += int(s.n_steps) * dt
        if acc >= tail_ms_target - 1e-9:
            break
    res.tail_mean_hz = sum(int(s.total_spikes) for s in tail) / n / (max(acc, dt) / 1000.0)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="FlyBrain smoke test: 2000 steps, four phases, the SPEC h.3 assertions (offline)")
    ap.add_argument("--n", type=int, default=None,
                    help="neurons (default FLY_N_NEURONS, else 20000; --fast uses 4000)")
    ap.add_argument("--seed", type=int, default=None, help="master seed (default FLY_SEED, else 1337)")
    ap.add_argument("--dt", type=float, default=None,
                    help="integration step in ms: 1.0|0.5|0.2|0.1 (default FLY_DT_MS, else 1.0)")
    ap.add_argument("--weights", choices=("calibrated", "literature"), default=None,
                    help="synthetic weight mode (default FLY_SYNTH_WEIGHTS, else calibrated)")
    ap.add_argument("--backend", choices=("numpy", "torch", "auto"), default=None,
                    help="propagation backend (default FLY_BACKEND, else numpy)")
    ap.add_argument("--gain", default=None,
                    help="'auto' (calibration / meta default) or a float (default FLY_GAIN, else auto)")
    ap.add_argument("--calibrate", choices=("auto", "force", "never"), default="auto",
                    help="run calibrate_gain when the weights are not pre-calibrated (default auto)")
    ap.add_argument("--fast", action="store_true", help="smaller brain (N=4000) for CI / a quick check")
    ap.add_argument("--no-rtf-check", action="store_true", dest="no_rtf_check",
                    help="report the RTF but never fail on it (slow machines)")
    ap.add_argument("--no-tonic", action="store_true", dest="no_tonic",
                    help="skip the SPEC c.10 tonic table (the server always applies it)")
    ap.add_argument("--quiet", action="store_true", help="only the table and the verdict")
    args = ap.parse_args(argv)

    ambient = ambient_env()
    from_env: list[str] = []

    def resolve(name: str, cli: str | None, default: str) -> str:
        """The flag when given, else the environment / ``.env`` value, else the documented default.

        The raw string is handed to ``load_settings``, so a bad ``.env`` value fails with the section-b
        message naming the variable instead of being silently ignored.
        """
        if cli is not None:
            return str(cli)
        raw = ambient.get(name)
        if raw:
            if raw.strip() != default:                # only a real override is worth a line
                from_env.append(f"{name}={raw}")
            return raw
        return default

    env = {
        # forced: a smoke run is always the offline synthetic brain, whatever the environment says
        "FLY_CONNECTOME_SOURCE": "synthetic",
        "FLY_REALTIME": "0",
        "FLY_MARKET": "sim",
        "FLY_LLM": "dryrun",
        "FLY_X": "dryrun",
        "FLY_SESSION_LOG": "0",
        "FLY_REPLAY": "",
        "FLY_LOG_LEVEL": "WARNING",
        # resolved: flag > environment / .env > SPEC h.3 default
        "FLY_N_NEURONS": ("4000" if args.n is None and args.fast
                          else resolve("FLY_N_NEURONS", None if args.n is None else str(int(args.n)),
                                       "20000")),
        "FLY_SEED": resolve("FLY_SEED", None if args.seed is None else str(int(args.seed)), "1337"),
        "FLY_DT_MS": resolve("FLY_DT_MS", None if args.dt is None else repr(float(args.dt)), "1.0"),
        "FLY_SYNTH_WEIGHTS": resolve("FLY_SYNTH_WEIGHTS", args.weights, "calibrated"),
        "FLY_BACKEND": resolve("FLY_BACKEND", args.backend, "numpy"),
        "FLY_GAIN": resolve("FLY_GAIN", args.gain, "auto"),
    }
    for key in ("FLY_DATA_DIR", "FLY_OUT_DIR"):      # where the connectome cache lives
        if ambient.get(key):
            env[key] = ambient[key]
    try:
        settings = load_settings(env=env, dotenv=None)
    except ValueError as exc:
        print(f"SMOKE FAILED: bad option ({exc})")
        return 1
    n_neurons = int(settings.n_neurons)

    print(f"smoke  n={n_neurons} seed={settings.seed} dt={settings.dt_ms:g} ms "
          f"weights={settings.synth_weights} backend={settings.backend} "
          f"tonic={'off' if args.no_tonic else 'on'}")
    if from_env and not args.quiet:
        print("env    from the environment / .env: " + ", ".join(from_env)
              + "  (market/LLM/X/session log/realtime stay forced offline)")
    t0 = time.perf_counter()
    try:
        conn = load_connectome(settings)
    except Exception as exc:  # noqa: BLE001 - a clear one-line reason is the contract
        print(f"SMOKE FAILED: could not build the connectome ({exc.__class__.__name__}: {exc})")
        return 1
    build_s = time.perf_counter() - t0

    gain, gain_src = resolve_gain(conn, settings, args.calibrate)
    if not args.quiet:
        print(f"brain  {conn.name}: n={conn.n} e={conn.e} gain={gain:.3f} ({gain_src}) "
              f"loaded in {build_s:.2f} s{' [cached]' if conn.meta.get('cached') else ''}")
        print(f"note   {conn.meta.get('note', '')}")

    groups = ("grn_sugar_labellar", "lc_loom", "feed_mn", "sugar2_exc", "ttmn", "gf")
    sizes = {name: _group_size(conn, name) for name in groups}
    if not args.quiet:
        print("sizes  " + "  ".join(f"{k}={v}" for k, v in sizes.items()))

    monitor = SpikeMonitor(conn, per_region=settings.raster_per_region, cap=settings.raster_cap,
                           seed=settings.seed, dt_ms=settings.dt_ms, tick_s=settings.tick_ms / 1000.0)
    eng = LIFEngine(conn, LIFParams(), seed=settings.seed, backend=settings.backend, dt_ms=settings.dt_ms,
                    gain=gain, noise_mu=settings.noise_mu, noise_sigma=settings.noise_sigma,
                    drive_mode="poisson", monitor=monitor)
    if args.no_tonic:
        # SPEC h.3 does not mention the tonic table, but SimulationLoop applies it at startup and the
        # phase-2 assertion depends on it (without it resting feed_mn is ~9.5 Hz and "5x rest" is out of
        # reach). Say so instead of failing mysteriously.
        print("WARN   --no-tonic: the SPEC c.10 tonic table is OFF, which the server never does. The "
              "SPEC h.3 phase-2 bounds assume it (resting feed_mn ~0.5 Hz with the table, ~9.5 Hz "
              "without), so 'sugar feed_mn >= 5x rest' can fail -- it does at N=20000. Diagnostic only.")
    else:
        applied = apply_tonic_table(eng)
        if not args.quiet:
            print("tonic  SPEC c.10 table applied (as SimulationLoop does at startup): "
                  + (", ".join(f"{k}={v}" for k, v in sorted(applied.items())) or "nothing matched"))

    results: list[PhaseResult] = []
    for name, steps, group, rate in PHASES:
        if group is not None and sizes.get(group, 0) == 0:
            print(f"SMOKE FAILED: group '{group}' is empty, phase '{name}' cannot run "
                  f"(sizes: {', '.join(f'{k}={v}' for k, v in sizes.items())})")
            return 1
        results.append(run_phase(eng, name, steps, group, rate, sizes, per_step=(name == "loom")))

    rest, sugar, loom, recover = results
    total_steps = sum(r.steps for r in results)
    wall_ms = sum(r.wall_ms for r in results)
    rtf = (total_steps * settings.dt_ms) / max(wall_ms, 1e-3)
    af_global = max(r.af_max for r in results)

    # ------------------------------------------------------------------ table
    # the column names are SPEC h.3's, verbatim
    head = (f"{'phase':<8} {'steps':>5} {'mean Hz':>8} {'active_frac_max':>15} {'MN9 Hz':>7} "
            f"{'DNp01 spikes':>12} {'first GF ms':>11} {'ms/step':>8}")
    bar = "-" * len(head)
    print(bar)
    print(head)
    print(bar)
    for r in results:
        mn9 = r.mn9_late_hz if r.name == "sugar" else r.mn9_hz
        gf_ms = "-" if r.first_gf_ms is None else f"{r.first_gf_ms:.1f}"
        print(f"{r.name:<8} {r.steps:>5} {r.mean_hz:>8.2f} {r.af_max:>15.4f} {mn9:>7.1f} "
              f"{r.gf_spikes:>12} {gf_ms:>11} {r.ms_per_step:>8.3f}")
    print(bar)
    if not args.quiet:
        print(f"('first GF ms' is exact in the loom phase and {WINDOW_MS:g} ms window resolution elsewhere; "
              f"'MN9 Hz' is the last {sugar.brain_ms / 2:.0f} ms for sugar, the whole phase otherwise)")
    print(f"RTF = {total_steps}*{settings.dt_ms:g} / {wall_ms:.1f} ms = {rtf:.2f}x  "
          f"(brain {total_steps * settings.dt_ms:.0f} ms in {wall_ms:.0f} ms wall)")

    # ------------------------------------------------------------------ checks
    targets = CalibTargets()
    sizes_txt = ", ".join(f"{k}={v}" for k, v in sizes.items())
    rest_floor = max(rest.mn9_hz, SUGAR_REST_FLOOR_HZ)
    # SPEC h.3 recover: "back below 5 Hz". Literal 5.0 Hz at the SPEC default N; small-N keeps headroom.
    recover_strict = conn.n >= STRICT_RECOVER_N
    recover_limit = (targets.rest_rate_max_hz if recover_strict
                     else max(targets.rest_rate_max_hz, rest.mean_hz + RECOVER_SLACK_HZ))
    checks: list[Check] = [
        Check("rest mean rate 1-5 Hz",
              targets.rest_rate_min_hz <= rest.mean_hz <= targets.rest_rate_max_hz,
              f"{rest.mean_hz:.2f} Hz over {rest.brain_ms:.0f} ms, n={eng.n}"),
        Check("rest active_frac_max <= 0.02",
              rest.af_max <= targets.rest_active_frac_max, f"{rest.af_max:.4f}"),
        Check("rest DNp01 spikes == 0", rest.gf_spikes <= REST_GF_MAX_SPIKES,
              f"{rest.gf_spikes} spikes, gf={sizes['gf']} cells (limit {REST_GF_MAX_SPIKES})"),
        Check("sugar feed_mn >= 20 Hz",
              sugar.mn9_late_hz >= SUGAR_MIN_HZ,
              f"{sugar.mn9_late_hz:.1f} Hz over the last {sugar.brain_ms / 2:.0f} ms, feed_mn={sizes['feed_mn']}"),
        Check("sugar feed_mn >= 5x rest",
              sugar.mn9_late_hz >= SUGAR_REST_FACTOR * rest_floor,
              f"{sugar.mn9_late_hz:.1f} Hz vs 5 x {rest_floor:.1f} Hz rest "
              f"(rest feed_mn {rest.mn9_hz:.1f} Hz, floor {SUGAR_REST_FLOOR_HZ:g})"),
        Check("sugar sugar2_exc >= 20 Hz",
              sugar.sugar2_late_hz >= SUGAR2_MIN_HZ,
              f"{sugar.sugar2_late_hz:.1f} Hz, sugar2_exc={sizes['sugar2_exc']}"),
        Check("loom first DNp01 spike <= 20 ms",
              loom.first_gf_ms is not None and loom.first_gf_ms <= LOOM_GF_MAX_MS,
              ("no DNp01 spike in the loom phase" if loom.first_gf_ms is None
               else f"{loom.first_gf_ms:.1f} ms, lc_loom={sizes['lc_loom']} gf={sizes['gf']}")),
        Check("loom ttmn spike <= 30 ms",
              loom.first_ttmn_ms is not None and loom.first_ttmn_ms <= LOOM_TTMN_MAX_MS,
              ("no ttmn spike in the loom phase" if loom.first_ttmn_ms is None
               else f"{loom.first_ttmn_ms:.1f} ms, ttmn={sizes['ttmn']}")),
        Check("recover mean rate back at rest",
              recover.tail_mean_hz <= recover_limit,
              f"{recover.tail_mean_hz:.2f} Hz over the last 100 ms, limit {recover_limit:.2f} Hz "
              f"(rest was {rest.mean_hz:.2f} Hz{'' if recover_strict else ', small-N headroom'})"),
        Check("no runaway (active_frac <= 0.05 everywhere)",
              af_global <= RUNAWAY_AF, f"max active_frac {af_global:.4f} over {total_steps} steps"),
    ]
    if args.no_rtf_check:
        checks.append(Check("RTF check skipped", True, f"{rtf:.2f}x (--no-rtf-check)"))
    elif conn.n > RTF_GUARD_MAX_N:
        checks.append(Check("RTF guard not enforced above N=20k", True, f"{rtf:.2f}x at n={conn.n}"))
    else:
        checks.append(Check(f"RTF >= {RTF_MIN:g}x", rtf >= RTF_MIN,
                            f"{rtf:.2f}x ({wall_ms / total_steps:.3f} ms/step at n={conn.n})"))

    print(bar)
    for c in checks:
        print(f"[{c.label}] {c.name:<42} {c.detail}")
    print(bar)
    failed = [c for c in checks if not c.ok and c.hard]
    warned = [c for c in checks if not c.ok and not c.hard]
    if failed:
        print(f"group sizes: {sizes_txt}")
        print(f"SMOKE FAILED: {len(failed)} check(s) failed; first: {failed[0].name} ({failed[0].detail})")
        return 1
    warn_txt = f", {len(warned)} warning(s)" if warned else ""
    print(f"SMOKE OK  {len(checks)} checks{warn_txt}, {total_steps} steps, RTF {rtf:.2f}x, "
          f"{build_s + wall_ms / 1000.0:.2f} s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
