#!/usr/bin/env python3
"""Convert the public MaleCNS v1.0 GCS flat files into neuPrint-export CSVs (SPEC section a).

    py -3 scripts/prepare_malecns.py [--out data/connectome/malecns] [--download-dir data/connectome/malecns/raw]
                                     [--min-weight 1] [--all-status] [--gzip] [--skip-download]

Downloads three Arrow IPC (feather) files anonymously from gs://flyem-male-cns (RESEARCH section 2.1
[V], ~566 MB in total, CC-BY 4.0) into ``<out>/raw/`` (resumable via a ``.part`` file and a Range
header; every download is size-checked against the Content-Length header, and against the
RESEARCH section 2.1 sizes with a warning when the upstream file changed):
  body-annotations-male-cns-v1.0-minconf-0.5.feather               (14.5 MB)  bodyId,type,instance,somaSide,superclass,class,subclass,status,...
  body-neurotransmitters-male-cns-v1.0.feather                      (43.3 MB)  body,consensus_nt,predicted_nt,...
  connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather (508.0 MB) body_pre,body_post,weight,type_pre,type_post
and writes
  <out>/neurons.csv      bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt,predictedNt
  <out>/connections.csv  bodyId_pre,bodyId_post,weight
which flybrain.connectome.loaders.load_csv_dir reads with FLY_CONNECTOME_SOURCE=csv.

Needs pyarrow (prepare-time only; the runtime never imports it). Console output is ASCII (exception
text is escaped). Importing this module performs no I/O.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "data" / "connectome" / "malecns"
BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES: dict[str, str] = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
}
#: expected sizes in MB (RESEARCH section 2.1 [V], verified 2026-09-10); a mismatch beyond SIZE_TOLERANCE
#: only warns, because the upstream objects may legitimately be re-uploaded.
EXPECTED_MB: dict[str, float] = {"annotations": 14.5, "neurotransmitters": 43.3, "weights": 508.0}
SIZE_TOLERANCE = 0.05
RAW_DIRNAME = "raw"
NEURON_COLUMNS: tuple[str, ...] = (
    "bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "status", "consensusNt", "predictedNt",
)
LICENSE_NOTE = "FlyEM MaleCNS v1.0, CC-BY 4.0 (Berg et al. 2026, doi:10.1016/j.cell.2026.08.015)"


def ascii_safe(obj: object) -> str:
    """``str(obj)`` with non-ASCII characters escaped (Windows raises localized OSError text)."""
    return str(obj).encode("ascii", "backslashreplace").decode("ascii")


def check_size(dest: Path, expected_mb: float | None) -> bool:
    """True when ``dest`` is within SIZE_TOLERANCE of the RESEARCH 2.1 size (or no expectation); warns otherwise."""
    if expected_mb is None:
        return True
    size_mb = dest.stat().st_size / 1e6
    if abs(size_mb - expected_mb) <= SIZE_TOLERANCE * expected_mb:
        return True
    print(f"  WARNING: {dest.name} is {size_mb:.1f} MB, RESEARCH 2.1 lists {expected_mb:.1f} MB "
          "(upstream file changed, or a truncated copy: delete it to re-download)")
    return False


# --------------------------------------------------------------------------- download


def download(url: str, dest: Path, chunk: int = 1 << 20, progress_every_mb: int = 50,
             expected_mb: float | None = None) -> Path:
    """Stream ``url`` to ``dest`` via ``dest.part`` (resumes with a Range header); returns ``dest``.

    The byte count is checked against the ``Content-Length`` header (a short read raises
    ``RuntimeError`` and keeps the ``.part`` file for the next resume) and, when ``expected_mb`` is
    given, against the RESEARCH section 2.1 size (warning only).
    """
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"  present: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        check_size(dest, expected_mb)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    headers = {"User-Agent": "flybrain-prepare/0.1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if have else "wb"
    t0 = time.time()
    expected_total: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if have and resp.status != 206:  # server ignored the Range header: start over
                have, mode = 0, "wb"
            total = resp.headers.get("Content-Length")
            if total:
                expected_total = int(total) + have
            total_mb = expected_total / 1e6 if expected_total else 0.0
            done = have
            next_mark = done + progress_every_mb * 1e6
            with open(part, mode) as fh:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if done >= next_mark:
                        print(f"  {dest.name}: {done / 1e6:.0f} / {total_mb:.0f} MB ({time.time() - t0:.0f} s)")
                        next_mark += progress_every_mb * 1e6
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"download failed for {url}: HTTP {exc.code}") from exc
    got = part.stat().st_size
    if expected_total is not None and got != expected_total:
        raise RuntimeError(
            f"download of {dest.name} is incomplete ({got} of {expected_total} bytes); re-run to resume the .part file"
        )
    part.replace(dest)
    print(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.0f} s)")
    check_size(dest, expected_mb)
    return dest


# --------------------------------------------------------------------------- conversion


def _col_or_empty(table: Any, name: str, n: int) -> list[str]:
    if name in table.schema.names:
        return ["" if v is None else str(v) for v in table.column(name).to_pylist()]
    return [""] * n


def convert(annotations: Any, neurotransmitters: Any, weights: Any, out_dir: Path, *, min_weight: int = 1,
            traced_only: bool = True, gzip_out: bool = False) -> dict[str, Any]:
    """Write ``neurons.csv`` + ``connections.csv`` (optionally ``.csv.gz``) from the three pyarrow tables.

    ``annotations`` needs ``bodyId`` (+ type, instance, somaSide, superclass, class, subclass, status);
    ``neurotransmitters`` needs ``body`` (+ consensus_nt, predicted_nt); ``weights`` needs
    ``body_pre, body_post, weight``. Rows with status != 'Traced' are dropped when ``traced_only``;
    edges with ``weight < min_weight`` are dropped. Returns a stats dict.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ann = annotations.num_rows
    body = np.asarray(annotations.column("bodyId").to_numpy(zero_copy_only=False), dtype=np.int64)
    cols = {name: _col_or_empty(annotations, name, n_ann) for name in ("type", "instance", "somaSide", "superclass", "class", "subclass", "status")}

    # NT lookup by body id (searchsorted join; the NT table has one row per body)
    nt_body = np.asarray(neurotransmitters.column("body").to_numpy(zero_copy_only=False), dtype=np.int64)
    nt_order = np.argsort(nt_body, kind="stable")
    nt_sorted = nt_body[nt_order]
    consensus_all = _col_or_empty(neurotransmitters, "consensus_nt", neurotransmitters.num_rows)
    predicted_all = _col_or_empty(neurotransmitters, "predicted_nt", neurotransmitters.num_rows)
    pos = np.searchsorted(nt_sorted, body)
    pos_c = np.minimum(pos, max(nt_sorted.size - 1, 0))
    found = (nt_sorted[pos_c] == body) if nt_sorted.size else np.zeros(body.shape, dtype=bool)
    src_row = nt_order[pos_c]

    keep = np.ones(n_ann, dtype=bool)
    if traced_only:
        keep = np.asarray([s == "Traced" for s in cols["status"]], dtype=bool)
    order = np.flatnonzero(keep)
    order = order[np.argsort(body[order], kind="stable")]

    n_path = out_dir / ("neurons.csv.gz" if gzip_out else "neurons.csv")
    if gzip_out:
        import gzip

        fh: Any = gzip.open(n_path, "wt", encoding="utf-8", newline="")
    else:
        fh = open(n_path, "w", encoding="utf-8", newline="")
    with fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(NEURON_COLUMNS)
        for i in order.tolist():
            cnt = consensus_all[src_row[i]] if found[i] else ""
            pnt = predicted_all[src_row[i]] if found[i] else ""
            w.writerow([int(body[i]), cols["type"][i], cols["instance"][i], cols["superclass"][i], cols["class"][i],
                        cols["subclass"][i], cols["somaSide"][i], cols["status"][i], cnt, pnt])

    wt = weights.select(["body_pre", "body_post", "weight"])
    if min_weight > 1:
        wt = wt.filter(pc.greater_equal(wt.column("weight"), pa.scalar(int(min_weight))))
    wt = wt.rename_columns(["bodyId_pre", "bodyId_post", "weight"])
    e_path = out_dir / ("connections.csv.gz" if gzip_out else "connections.csv")
    if gzip_out:
        with pa.CompressedOutputStream(str(e_path), "gzip") as sink:
            pacsv.write_csv(wt, sink)
    else:
        pacsv.write_csv(wt, str(e_path))
    stats = {
        "neurons_in": int(n_ann), "neurons_out": int(order.size), "nt_matched": int(found[order].sum()),
        "edges_in": int(weights.num_rows), "edges_out": int(wt.num_rows), "min_weight": int(min_weight),
        "neurons_path": str(n_path), "connections_path": str(e_path),
    }
    return stats


# --------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="MaleCNS v1.0 GCS flat files -> neuPrint-export CSVs (needs pyarrow).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--download-dir", type=Path, default=None,
                    help=f"where the feather files are kept (default <out>/{RAW_DIRNAME}, SPEC h.4)")
    ap.add_argument("--min-weight", type=int, default=1, help="drop edges below this synapse count (default 1 = keep all)")
    ap.add_argument("--all-status", action="store_true", help="keep non-Traced bodies too (the loader filters anyway)")
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz instead of .csv")
    ap.add_argument("--skip-download", action="store_true", help="fail instead of downloading missing feathers")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import pyarrow.feather as pf
    except ImportError:
        print("prepare_malecns.py needs pyarrow: py -3 -m pip install pyarrow   (prepare-time only)")
        return 2
    out_dir = Path(args.out)
    dl_dir = Path(args.download_dir) if args.download_dir else out_dir / RAW_DIRNAME
    t0 = time.time()
    try:
        paths: dict[str, Path] = {}
        print(f"source: {BASE_URL}  ({LICENSE_NOTE})")
        for key, fname in FILES.items():
            dest = dl_dir / fname
            if args.skip_download:
                if not dest.is_file():
                    print(f"ERROR: missing {ascii_safe(dest)} (--skip-download)")
                    return 1
                check_size(dest, EXPECTED_MB.get(key))
                paths[key] = dest
            else:
                paths[key] = download(BASE_URL + fname, dest, expected_mb=EXPECTED_MB.get(key))
        print("reading feather files (pyarrow)")
        ann = pf.read_table(paths["annotations"])
        nt = pf.read_table(paths["neurotransmitters"])
        wt = pf.read_table(paths["weights"], columns=["body_pre", "body_post", "weight"])
        print(f"  annotations {ann.num_rows} rows, neurotransmitters {nt.num_rows} rows, weights {wt.num_rows} rows")
        stats = convert(ann, nt, wt, out_dir, min_weight=int(args.min_weight), traced_only=not args.all_status,
                        gzip_out=bool(args.gzip))
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {ascii_safe(exc)}")
        return 1
    print(f"wrote {ascii_safe(stats['neurons_path'])} ({stats['neurons_out']} neurons, {stats['nt_matched']} with NT) and "
          f"{ascii_safe(stats['connections_path'])} ({stats['edges_out']} edges) in {time.time() - t0:.0f} s")
    print(f"next: FLY_CONNECTOME_SOURCE=csv FLY_CONNECTOME_DIR={ascii_safe(out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
