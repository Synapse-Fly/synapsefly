"""Engine benchmark: ms/step and RTF on the synthetic connectome (SPEC sections a, h.4).

Usage (from anywhere; paths are resolved from this file, never from the CWD)::

    py -3 scripts/bench.py                                  # tiny fixture + synthetic N=20k, dt 1.0, active 1 % and 3 %
    py -3 scripts/bench.py --n 20000,40000,80000 --dt 1.0,0.5 --active 0.005,0.02
    py -3 scripts/bench.py --torch                          # adds a torch-backend row per combination (if importable)
    py -3 scripts/bench.py --random --edges-per-neuron 100   # extra stress row on a dense uniform-random graph
    py -3 scripts/bench.py --profile                        # cProfile of the first graph run (top 25 by tottime)

For every (n, dt, active) combination the script builds the synthetic graph
(``build_synthetic(n, seed=--seed)``, SPEC section g - the same graph the server runs, so the numbers
characterise real traffic), drives a random half of the neurons with Poisson kicks whose rate is
chosen so that the forced spikes alone hit the target active fraction, runs ``--steps`` steps
``--repeat`` times and prints n, e, dt, the measured active fraction, the median ms/step, the RTF
(brain ms / wall ms), the backend and the edge visits per step. ``--random`` replaces the synthetic
graph with a local uniform-random one of E = ``--edges-per-neuron`` * N edges (lognormal synapse
counts, 30 % inhibitory, untyped): a deliberately denser worst case (E = 2 M at N = 20k versus the
synthetic 0.5 M), not a model of the connectome. Output is ASCII only; exit code 1 when the SPEC
section 0 target (50 steps of 1 ms at N=20k in < 50 ms wall, numpy backend, 1-3 % active) is missed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from flybrain.connectome.csr import sum_duplicates  # noqa: E402
from flybrain.connectome.groups import resolve_groups  # noqa: E402
from flybrain.connectome.schema import (  # noqa: E402
    REGIONS,
    SYNTHETIC_BODY_BASE,
    Connectome,
    count_groups,
    count_regions,
    utc_now_iso,
)
from flybrain.snn.engine import LIFEngine  # noqa: E402

TARGET_MS_50 = 50.0   # SPEC section 0: 50 steps (1 ms each) at N=20k in < 50 ms wall


def synthetic_connectome(n: int, seed: int = 1337, mean_outdeg: int = 25) -> Connectome:
    """The SPEC section g synthetic connectome (``--random`` swaps in ``random_connectome`` instead)."""
    from flybrain.connectome.synthetic import build_synthetic  # imported here: ~1-10 s per build

    return build_synthetic(n_neurons=n, seed=seed, mean_outdeg=mean_outdeg)


def random_connectome(n: int, e: int, seed: int = 0, inhib_frac: float = 0.3) -> Connectome:
    """Random directed graph with lognormal synapse counts (``--random`` stress row only; not
    connectome data and not the graph the server runs - use it as an upper bound on density)."""
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, size=e)
    post = rng.integers(0, n, size=e)
    w = np.maximum(1.0, np.round(rng.lognormal(mean=np.log(4.0), sigma=0.9, size=e)))
    pre, post, weight = sum_duplicates(pre, post, w, n)
    sign = np.where(rng.random(n) < inhib_frac, -1.0, 1.0).astype(np.float32)
    region = (rng.integers(0, len(REGIONS), size=n)).astype(np.uint8)
    side = np.where(rng.random(n) < 0.5, -1, 1).astype(np.int8)
    types = [""]
    type_idx = np.zeros(n, dtype=np.int32)
    nt = np.where(sign < 0, "gaba", "acetylcholine").astype(object)
    groups = resolve_groups(types, type_idx, side)
    meta = {
        "e": int(pre.shape[0]), "synapses": float(weight.sum(dtype=np.float64)), "license": "synthetic (no data)",
        "citation": None, "seed": int(seed), "build_args": {"n": n, "e": e, "bench": True}, "patches_applied": [],
        "gain_default": 1.0, "weights_mode": "bench", "engineered_edges": [], "region_counts": count_regions(region),
        "group_counts": count_groups(groups), "created": utc_now_iso(),
    }
    return Connectome(
        name=f"bench-{n}-{e}", source="synthetic", n=n, pre=pre.astype(np.int32), post=post.astype(np.int32),
        weight=weight.astype(np.float32), sign=sign, region=region, side=side, types=types, type_idx=type_idx,
        body_id=(SYNTHETIC_BODY_BASE + np.arange(n, dtype=np.int64)), nt=nt, groups=groups, meta=meta,
    )


def bench_engine(conn: Connectome, dt: float, steps: int, repeat: int, backend: str, gain: float,
                 active: float | None, label: str, window: int = 50) -> dict:
    """Run ``steps`` steps ``repeat`` times on a fresh engine state; returns the median row."""
    t_build = time.perf_counter()
    eng = LIFEngine(conn, seed=1, backend=backend, dt_ms=dt, gain=gain)
    build_ms = (time.perf_counter() - t_build) * 1000.0
    if active is not None and active > 0:
        rng = np.random.default_rng(7)
        frac = 0.5
        idx = np.sort(rng.choice(conn.n, size=int(frac * conn.n), replace=False)).astype(np.int32)
        rate_hz = active / frac * 1000.0 / dt   # forced spikes per step = frac*n*rate*dt/1000 = active*n
        eng.inject(idx, rate_hz=rate_hz, duration_ms=1e12, tag="bench")
    per_rep: list[dict] = []
    for _ in range(repeat):
        eng.step(window)  # warm-up (caches, first-touch of buffers)
        spikes = visits = forced = 0
        af = 0.0
        wall0 = time.perf_counter()
        done = 0
        while done < steps:
            k = min(window, steps - done)
            s = eng.step(k)
            spikes += s.total_spikes
            visits += s.edge_visits
            forced += s.forced_events
            af = max(af, s.active_frac_max)
            done += k
        wall_ms = (time.perf_counter() - wall0) * 1000.0
        per_rep.append({"ms_step": wall_ms / steps, "spikes_step": spikes / steps, "visits_step": visits / steps,
                        "forced_step": forced / steps, "active_max": af})
    ms_step = statistics.median(r["ms_step"] for r in per_rep)
    rtf = dt / ms_step
    active_mean = statistics.median(r["spikes_step"] for r in per_rep) / conn.n
    row = {
        "label": label, "n": conn.n, "e": conn.e, "dt": dt, "backend": eng.backend, "steps": steps, "repeat": repeat,
        "ms_step": ms_step, "rtf": rtf, "active_mean": active_mean,
        "active_max": max(r["active_max"] for r in per_rep),
        "visits_step": statistics.median(r["visits_step"] for r in per_rep),
        "forced_step": statistics.median(r["forced_step"] for r in per_rep),
        "build_ms": build_ms, "ms_50steps": ms_step * 50.0,
        "concat_threshold": int(getattr(eng.prop, "concat_threshold", -1)),
    }
    print(
        f"{label:<14} n={conn.n:>7d} e={conn.e:>9d} dt={dt:<4} {eng.backend:<5} active={active_mean * 100:5.2f}% "
        f"(max {row['active_max'] * 100:5.2f}%)  {ms_step:7.3f} ms/step (median of {repeat})  rtf={rtf:7.2f}  "
        f"50 steps={row['ms_50steps']:7.2f} ms  visits/step={row['visits_step']:9.0f}"
        + (f"  concat_thr={row['concat_threshold']}" if row["concat_threshold"] >= 0 else "")
    )
    return row


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s: str) -> list[int]:
    return [int(float(x)) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FlyBrain engine benchmark (ASCII output)")
    ap.add_argument("--n", default="20000", help="comma list of neuron counts (default 20000)")
    ap.add_argument("--dt", default="1.0", help="comma list of dt in ms (default 1.0)")
    ap.add_argument("--active", default="0.01,0.03", help="comma list of target active fractions (default 0.01,0.03)")
    ap.add_argument("--seed", type=int, default=1337, help="build_synthetic seed (default 1337)")
    ap.add_argument("--mean-outdeg", type=int, default=25, help="build_synthetic mean background out-degree")
    ap.add_argument("--random", action="store_true",
                    help="bench a dense uniform-random graph instead of the synthetic connectome")
    ap.add_argument("--edges-per-neuron", type=int, default=100,
                    help="--random only: E = this * N (default 100 -> 2M at 20k)")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "torch", "auto"])
    ap.add_argument("--torch", action="store_true", help="add a torch-backend row per combination when importable")
    ap.add_argument("--gain", type=float, default=None,
                    help="synaptic gain (default: meta['gain_default'] of the synthetic graph, 0.25 for --random)")
    ap.add_argument("--profile", action="store_true", help="cProfile the first graph run")
    ap.add_argument("--no-tiny", action="store_true", help="skip the tiny fixture")
    args = ap.parse_args(argv)

    print(f"numpy {np.__version__}; python {sys.version.split()[0]}; steps={args.steps} x{args.repeat}")
    if not args.no_tiny:
        try:
            from tests.conftest import build_tiny_connectome  # needs pytest installed

            tiny = build_tiny_connectome()
        except Exception as exc:  # pragma: no cover - pytest missing
            print(f"tiny fixture unavailable ({exc.__class__.__name__}); using a random 400-neuron graph")
            tiny = random_connectome(400, 3300, seed=3)
        bench_engine(tiny, 1.0, max(args.steps, 2000), args.repeat, args.backend, 1.0, None, "tiny fixture")

    backends = [args.backend]
    if args.torch and "torch" not in backends:
        try:
            import torch  # noqa: F401

            backends.append("torch")
        except Exception:
            print("torch not importable: --torch ignored")

    rows: list[dict] = []
    profiled = False
    for n in _ints(args.n):
        t0 = time.perf_counter()
        if args.random:
            conn = random_connectome(n, args.edges_per_neuron * n, seed=1)
            gain = 0.25 if args.gain is None else args.gain
            kind = "random"
        else:
            conn = synthetic_connectome(n, seed=args.seed, mean_outdeg=args.mean_outdeg)
            gain = float(conn.meta.get("gain_default", 1.0)) if args.gain is None else args.gain
            kind = "synthetic"
        print(f"built {kind} n={conn.n} e={conn.e} gain={gain:.3f} in {(time.perf_counter() - t0) * 1000.0:.0f} ms")
        for dt in _floats(args.dt):
            for active in _floats(args.active):
                for backend in backends:
                    label = f"{kind} {n // 1000}k"
                    if args.profile and not profiled:
                        import cProfile
                        import pstats

                        prof = cProfile.Profile()
                        prof.enable()
                        rows.append(bench_engine(conn, dt, args.steps, args.repeat, backend, gain, active, label))
                        prof.disable()
                        pstats.Stats(prof).sort_stats("tottime").print_stats(25)
                        profiled = True
                    else:
                        rows.append(bench_engine(conn, dt, args.steps, args.repeat, backend, gain, active, label))

    ok = True
    for row in rows:
        if row["n"] == 20000 and row["dt"] == 1.0 and row["backend"] == "numpy" and 0.005 <= row["active_mean"] <= 0.035:
            hit = row["ms_50steps"] < TARGET_MS_50
            ok = ok and hit
            print(f"SPEC section 0 target: 50 steps < {TARGET_MS_50:.0f} ms wall at N=20k, active "
                  f"{row['active_mean'] * 100:.2f}% -> {row['ms_50steps']:.2f} ms ({'OK' if hit else 'MISSED'})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
