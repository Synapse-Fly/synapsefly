#!/usr/bin/env python
"""Export the synthetic connectome as neuPrint-schema CSV and prove loader parity (SPEC section g.7 / h.4).

Usage:
    py -3 scripts/export_synthetic.py --out data\\connectome\\synthetic_export [--n 20000] [--seed 1337]
                                      [--weights calibrated|literature] [--mean-outdeg 25]

Steps: ``build_synthetic`` -> ``export_csv`` (neurons.csv + connections.csv) -> ``load_csv_dir(out, min_weight=1,
subset='all')`` on the result, then compare ``n``, ``e``, ``sign``, ``region``, ``side``, every group and the CSR.
Prints one ASCII line per check and ``PARITY OK`` (exit 0) or ``PARITY FAILED: ...`` (exit 1). Never touches the
network. The export is a stand-in shaped like MaleCNS v1.0, not connectome data (``meta['note']``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402

from flybrain.connectome.loaders import load_csv_dir  # noqa: E402
from flybrain.connectome.synthetic import build_synthetic, export_csv  # noqa: E402


def compare(a, b) -> list[str]:
    """Return the list of parity violations between two connectomes (empty = identical)."""
    problems: list[str] = []
    if a.n != b.n:
        problems.append(f"n {a.n} != {b.n}")
    if a.e != b.e:
        problems.append(f"e {a.e} != {b.e}")
    if problems:
        return problems
    for name in ("sign", "region", "side", "type_idx", "body_id"):
        x, y = getattr(a, name), getattr(b, name)
        if not np.array_equal(np.asarray(x), np.asarray(y)):
            problems.append(f"{name} differs ({int((np.asarray(x) != np.asarray(y)).sum())} entries)")
    if list(a.types) != list(b.types):
        problems.append("types differ")
    if [str(x) for x in a.nt] != [str(x) for x in b.nt]:
        problems.append("nt differs")
    if set(a.groups) != set(b.groups):
        problems.append(f"group keys differ: {sorted(set(a.groups) ^ set(b.groups))[:10]}")
    else:
        bad = [k for k in a.groups if not np.array_equal(a.groups[k], b.groups[k])]
        if bad:
            problems.append(f"groups differ: {bad[:10]}")
    ca, cb = a.csr(), b.csr()
    if not (np.array_equal(ca.indptr, cb.indptr) and np.array_equal(ca.indices, cb.indices)
            and np.array_equal(ca.data, cb.data)):
        problems.append("CSR differs")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="synthetic connectome -> neuPrint-schema CSV (loader parity proof)")
    ap.add_argument("--out", required=True, help="output directory (neurons.csv + connections.csv)")
    ap.add_argument("--n", type=int, default=20_000, help="number of neurons (default 20000)")
    ap.add_argument("--seed", type=int, default=1337, help="generator seed (default 1337)")
    ap.add_argument("--weights", choices=("calibrated", "literature"), default="calibrated")
    ap.add_argument("--mean-outdeg", type=int, default=25, dest="mean_outdeg")
    args = ap.parse_args(argv)

    out = Path(args.out)
    t0 = time.perf_counter()
    conn = build_synthetic(n_neurons=args.n, seed=args.seed, mean_outdeg=args.mean_outdeg, weights=args.weights)
    t_build = time.perf_counter() - t0
    print(f"built  {conn.name}: n={conn.n} e={conn.e} synapses={conn.meta['synapses']:.0f} in {t_build:.2f} s")
    print(f"note   {conn.meta['note']}")
    # the engineered ([E]) rows the README / NOTICE must list are read from the graph, never transcribed by hand
    e_pids = list(conn.meta["engineered_edges"])
    print(f"E rows {len(e_pids)} of {len(conn.meta['projections'])}: {' '.join(e_pids)}")
    print(f"brakes {' '.join(conn.meta['rest_brakes'])} | retuned {' '.join(sorted(conn.meta['rest_tuning']))}")
    t1 = time.perf_counter()
    neurons_path, conns_path = export_csv(conn, out)
    print(f"wrote  {neurons_path} ({neurons_path.stat().st_size} bytes)")
    print(f"wrote  {conns_path} ({conns_path.stat().st_size} bytes) in {time.perf_counter() - t1:.2f} s")
    t2 = time.perf_counter()
    back = load_csv_dir(out, min_weight=1, subset="all", n_max=None)
    print(f"loaded {back.name}: n={back.n} e={back.e} in {time.perf_counter() - t2:.2f} s")
    print(f"n equal: {conn.n == back.n} ({conn.n} vs {back.n})")
    print(f"e equal: {conn.e == back.e} ({conn.e} vs {back.e})")
    problems = compare(conn, back)
    if problems:
        print("PARITY FAILED: " + "; ".join(problems))
        return 1
    print("PARITY OK (n, e, sign, region, side, types, nt, groups, CSR identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
