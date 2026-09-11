#!/usr/bin/env python3
"""Fetch a MaleCNS core subset from neuPrint into neurons.csv + connections.csv (SPEC sections h.4 / i.5).

    py -3 scripts/fetch_neuprint.py [--out data/connectome/neuprint] [--n-random 25000] [--min-weight 3]
                                    [--chunk 2000] [--token ...] [--dataset male-cns:v1.0]
                                    [--server https://neuprint.janelia.org] [--no-fill] [--chunk-types 50]
                                    [--concurrency 3] [--fresh]

Token: ``--token`` or the environment variable NEUPRINT_APPLICATION_CREDENTIALS (obtain it at
https://neuprint.janelia.org/account) selects ``neuprint-python`` ``Client`` (urllib with a bearer
header when the package is missing). Without a token the script uses the stdlib ``urllib`` POST to
``/api/custom/custom`` WITHOUT an Authorization header (RESEARCH section 2.2 [V]: a dummy bearer
token is rejected with 401, omitting the header works). ``--anonymous`` is accepted as a no-op for
backwards compatibility.

Output schema (the "neuPrint export" of RESEARCH section 2.3 that flybrain.connectome.loaders reads):
  neurons.csv      bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt,predictedNt
  connections.csv  bodyId_pre,bodyId_post,weight

Selection: every Traced neuron whose type fullmatches any GROUP_REGEX of flybrain.connectome.groups
(the regex union is expanded to the literal type list via a first query of distinct types), plus
class in {gustatory, Kenyon_Cell, MBON, DAN, CX, ALPN, ALLN} or superclass in {descending_neuron,
cb_motor, vnc_motor, vnc_efferent} (the loader's "core" definition), random-filled with
``rand() < p`` to --n-random, then every ConnectsTo edge among the selected bodies with
weight >= --min-weight, fetched in chunks of --chunk presynaptic bodies with at most --concurrency
(default 3) requests in flight. The ``<= 8 type pairs per query`` rule of RESEARCH section 2.2
concerns type-pair joins, which this script never issues (it matches indexed bodyId lists).

Resumable per chunk: ``neurons.csv`` is written as soon as the selection is known and every edge
chunk is stored under ``<out>/chunks/edges_NNNNN.csv`` as it arrives; a re-run with the same
--dataset / --min-weight / --chunk reuses ``neurons.csv`` and the chunks already on disk
(``--fresh`` discards them). After the last chunk the pieces are merged into ``connections.csv`` and
the chunk directory is removed.

The runtime never imports this file and never fetches at start-up (SPEC section b). Importing this
module performs no network I/O and needs nothing but the standard library. Console output is ASCII
(exception text is escaped, SPEC section 0.1).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

REPO = Path(__file__).resolve().parents[1]
BACKEND = REPO / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

TOKEN_ENV = "NEUPRINT_APPLICATION_CREDENTIALS"
DEFAULT_SERVER = "https://neuprint.janelia.org"
DEFAULT_DATASET = "male-cns:v1.0"
DEFAULT_OUT = REPO / "data" / "connectome" / "neuprint"
DEFAULT_N_RANDOM = 25_000     # SPEC i.5 command line
DEFAULT_MIN_WEIGHT = 3        # SPEC h.4 / b (FLY_MIN_WEIGHT default)
DEFAULT_CHUNK = 2000          # <= 2000 pre bodies per edge query (RESEARCH section 2.2)
DEFAULT_CONCURRENCY = 3       # <= 3 concurrent requests (RESEARCH section 2.2)
TRACED_TOTAL_HINT = 165_122   # Traced neurons in male-cns:v1.0 (RESEARCH section 3) [V]
CHUNK_DIRNAME = "chunks"
STATE_FILENAME = "state.json"

NEURON_COLUMNS: tuple[str, ...] = (
    "bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "status", "consensusNt", "predictedNt",
)
EDGE_COLUMNS: tuple[str, ...] = ("bodyId_pre", "bodyId_post", "weight")
CORE_CLASSES: tuple[str, ...] = ("gustatory", "Kenyon_Cell", "MBON", "DAN", "CX", "ALPN", "ALLN")
CORE_SUPERCLASSES: tuple[str, ...] = ("descending_neuron", "cb_motor", "vnc_motor", "vnc_efferent")

TOKEN_HINT = f"""no neuPrint token: using anonymous HTTP (no Authorization header; RESEARCH 2.2).
  If the server answers 401, obtain a token at https://neuprint.janelia.org/account and set
  {TOKEN_ENV} (PowerShell: $env:{TOKEN_ENV} = "<token>") or pass --token.
  Alternative without any account: py -3 scripts/prepare_malecns.py (public GCS flat files, needs pyarrow)."""

_RETRY_STATUS = {429, 500, 502, 503, 504}


def ascii_safe(obj: object) -> str:
    """``str(obj)`` with non-ASCII characters escaped (Windows raises localized OSError text)."""
    return str(obj).encode("ascii", "backslashreplace").decode("ascii")


# --------------------------------------------------------------------------- transports


class HttpTransport:
    """Stdlib urllib POST /api/custom/custom; sends Authorization only when a token is given."""

    def __init__(self, server: str, dataset: str, token: str | None = None, timeout: float = 300.0,
                 retry_wait: float = 5.0) -> None:
        self.server = server.rstrip("/")
        self.dataset = dataset
        self.token = token.strip() if token else None
        self.timeout = timeout
        self.retry_wait = retry_wait

    def query(self, cypher: str, retries: int = 3) -> tuple[list[str], list[list[Any]]]:
        body = json.dumps({"cypher": cypher, "dataset": self.dataset}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        url = f"{self.server}/api/custom/custom"
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                return list(payload.get("columns", [])), list(payload.get("data", []))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code == 401:
                    detail = ascii_safe(exc.read().decode("utf-8", "replace")[:200])
                    raise RuntimeError(f"neuPrint rejected the request (401): {detail}") from exc
                if exc.code not in _RETRY_STATUS or attempt == retries:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                if attempt == retries:
                    raise
            wait = self.retry_wait * attempt
            print(f"  request failed ({ascii_safe(last)}); retrying in {wait:.0f} s ({attempt}/{retries})")
            time.sleep(wait)
        raise RuntimeError(f"neuPrint request failed: {ascii_safe(last)}")  # pragma: no cover


class NeuprintPythonTransport:
    """neuprint-python Client.fetch_custom(format='json'); used when the package is installed and a token exists."""

    def __init__(self, server: str, dataset: str, token: str, retry_wait: float = 5.0) -> None:
        from neuprint import Client  # optional dependency, imported here only

        self.client = Client(server, dataset=dataset, token=token)
        self.retry_wait = retry_wait
        self._lock = threading.Lock()  # one requests.Session behind the client: serialise calls

    def query(self, cypher: str, retries: int = 3) -> tuple[list[str], list[list[Any]]]:
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                with self._lock:
                    payload = self.client.fetch_custom(cypher, format="json")
                return list(payload.get("columns", [])), list(payload.get("data", []))
            except Exception as exc:  # network / server errors: retry a few times
                last = exc
                if attempt == retries:
                    raise
                wait = self.retry_wait * attempt
                print(f"  request failed ({ascii_safe(exc)}); retrying in {wait:.0f} s ({attempt}/{retries})")
                time.sleep(wait)
        raise RuntimeError(f"neuPrint request failed: {ascii_safe(last)}")  # pragma: no cover


def make_transport(server: str, dataset: str, token: str | None) -> Any:
    """Token -> neuprint-python (urllib + bearer when not installed); no token -> anonymous urllib."""
    if token:
        try:
            t = NeuprintPythonTransport(server, dataset, token)
            print("transport: neuprint-python (token)")
            return t
        except ImportError:
            print("transport: urllib with bearer token (neuprint-python not installed)")
            return HttpTransport(server, dataset, token)
    print("transport: urllib, anonymous (no Authorization header)")
    print(TOKEN_HINT)
    return HttpTransport(server, dataset, None)


# --------------------------------------------------------------------------- cypher helpers


def cypher_literal(values: Sequence[Any]) -> str:
    """JSON list literal, which Cypher accepts verbatim for lists of numbers / strings."""
    return json.dumps(list(values), ensure_ascii=True)


def chunks(seq: Sequence[Any], size: int) -> list[Sequence[Any]]:
    return [seq[i : i + size] for i in range(0, len(seq), max(1, size))]


NEURON_RETURN = (
    "RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance, n.superclass AS superclass, "
    "n.class AS class, n.subclass AS subclass, n.somaSide AS somaSide, n.status AS status, "
    "n.consensusNt AS consensusNt, n.predictedNt AS predictedNt"
)


def group_patterns() -> list[re.Pattern[str]]:
    from flybrain.connectome.groups import GROUP_REGEX  # backend/ is on sys.path (module top)

    return [re.compile(rx) for rx in GROUP_REGEX.values()]


def matched_types(all_types: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> list[str]:
    """Type strings that fullmatch at least one GROUP_REGEX pattern (sorted, unique)."""
    out = {t for t in all_types if t and any(p.fullmatch(t) is not None for p in patterns)}
    return sorted(out)


def rows_to_neurons(columns: list[str], rows: list[list[Any]], into: dict[int, dict[str, str]]) -> int:
    """Merge query rows into ``into`` keyed by bodyId; returns the number of NEW bodies."""
    idx = {c: i for i, c in enumerate(columns)}
    added = 0
    for row in rows:
        try:
            bid = int(row[idx["bodyId"]])
        except (KeyError, TypeError, ValueError):
            continue
        if bid in into:
            continue
        rec: dict[str, str] = {}
        for col in NEURON_COLUMNS:
            val = row[idx[col]] if col in idx else None
            rec[col] = "" if val is None else str(val)
        rec["bodyId"] = str(bid)
        into[bid] = rec
        added += 1
    return added


# --------------------------------------------------------------------------- fetch steps


def fetch_all_types(t: Any) -> list[str]:
    cols, rows = t.query("MATCH (n:Neuron) WHERE n.status = 'Traced' AND n.type IS NOT NULL RETURN DISTINCT n.type AS type")
    return [str(r[0]) for r in rows if r and r[0] is not None]


def fetch_neurons_by_types(t: Any, types: Sequence[str], into: dict[int, dict[str, str]], chunk: int) -> None:
    for i, part in enumerate(chunks(list(types), chunk), 1):
        cypher = f"MATCH (n:Neuron) WHERE n.status = 'Traced' AND n.type IN {cypher_literal(part)} {NEURON_RETURN}"
        cols, rows = t.query(cypher)
        added = rows_to_neurons(cols, rows, into)
        print(f"  types chunk {i}: {len(part)} types -> {len(rows)} rows ({added} new, total {len(into)})")


def fetch_neurons_by_class(t: Any, into: dict[int, dict[str, str]]) -> None:
    cypher = (
        "MATCH (n:Neuron) WHERE n.status = 'Traced' AND (n.class IN "
        f"{cypher_literal(CORE_CLASSES)} OR n.superclass IN {cypher_literal(CORE_SUPERCLASSES)}) {NEURON_RETURN}"
    )
    cols, rows = t.query(cypher)
    added = rows_to_neurons(cols, rows, into)
    print(f"  class/superclass core: {len(rows)} rows ({added} new, total {len(into)})")


def fetch_random_fill(t: Any, into: dict[int, dict[str, str]], n_random: int) -> None:
    need = n_random - len(into)
    if need <= 0:
        print(f"  core already holds {len(into)} >= {n_random} bodies; no random fill")
        return
    remaining = max(TRACED_TOTAL_HINT - len(into), need)
    p = min(1.0, 1.25 * need / remaining)
    cypher = f"MATCH (n:Neuron) WHERE n.status = 'Traced' AND rand() < {p:.6f} {NEURON_RETURN}"
    cols, rows = t.query(cypher)
    candidates: dict[int, dict[str, str]] = {}
    rows_to_neurons(cols, rows, candidates)
    picked = 0
    for bid, rec in candidates.items():
        if picked >= need:
            break
        if bid in into:
            continue
        into[bid] = rec
        picked += 1
    print(f"  random fill p={p:.4f}: {len(rows)} candidates -> {picked} added (total {len(into)})")


def _chunk_path(chunk_dir: Path, i: int) -> Path:
    return chunk_dir / f"edges_{i:05d}.csv"


def _read_edge_csv(path: Path) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0].strip():
                out.append((int(row[0]), int(row[1]), int(row[2])))
    return out


def _write_edge_csv(path: Path, edges: Sequence[tuple[int, int, int]]) -> None:
    """Atomic write (temp file + os.replace) so an interrupted run never leaves a half chunk behind."""
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(EDGE_COLUMNS)
        for pre, post, wt in edges:
            w.writerow((pre, post, wt))
    os.replace(tmp, path)


def fetch_edge_chunk(t: Any, part: Sequence[int], all_literal: str, min_weight: int) -> list[tuple[int, int, int]]:
    cypher = (
        "MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) "
        f"WHERE a.bodyId IN {cypher_literal(part)} AND b.bodyId IN {all_literal} AND c.weight >= {int(min_weight)} "
        "RETURN a.bodyId AS bodyId_pre, b.bodyId AS bodyId_post, c.weight AS weight"
    )
    cols, rows = t.query(cypher)
    idx = {c: j for j, c in enumerate(cols)}
    edges: list[tuple[int, int, int]] = []
    for r in rows:
        try:
            edges.append((int(r[idx["bodyId_pre"]]), int(r[idx["bodyId_post"]]), int(r[idx["weight"]])))
        except (KeyError, TypeError, ValueError):
            continue
    return edges


def fetch_edges(t: Any, body_ids: Sequence[int], min_weight: int, chunk: int, chunk_dir: Path,
                concurrency: int = DEFAULT_CONCURRENCY) -> list[tuple[int, int, int]]:
    """All ConnectsTo edges among ``body_ids`` with ``weight >= min_weight``; chunked by pre body, resumable.

    Chunk ``i`` is served from ``<chunk_dir>/edges_{i:05d}.csv`` when that file exists, otherwise
    fetched (at most ``concurrency`` requests in flight) and written there atomically. The first
    failure stops new submissions, waits for the running requests (their chunks are still stored)
    and re-raises, so a re-run only fetches what is missing.
    """
    ids = sorted(int(b) for b in body_ids)
    all_lit = cypher_literal(ids)
    parts = chunks(ids, chunk)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    edges_by_chunk: dict[int, list[tuple[int, int, int]]] = {}
    todo: list[int] = []
    for i in range(len(parts)):
        p = _chunk_path(chunk_dir, i)
        if p.is_file():
            edges_by_chunk[i] = _read_edge_csv(p)
        else:
            todo.append(i)
    if edges_by_chunk:
        print(f"  {len(edges_by_chunk)}/{len(parts)} edge chunks already on disk ({sum(len(v) for v in edges_by_chunk.values())} rows)")

    def work(i: int) -> tuple[int, list[tuple[int, int, int]]]:
        got = fetch_edge_chunk(t, parts[i], all_lit, min_weight)
        _write_edge_csv(_chunk_path(chunk_dir, i), got)
        return i, got

    workers = max(1, min(int(concurrency), len(todo) or 1))
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set[concurrent.futures.Future] = set()
        queue = list(todo)
        failure: BaseException | None = None
        while queue or pending:
            while queue and len(pending) < workers and failure is None:
                pending.add(pool.submit(work, queue.pop(0)))
            if not pending:
                break
            finished, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for fut in finished:
                try:
                    i, got = fut.result()
                except BaseException as exc:  # keep the other in-flight chunks, then re-raise
                    if failure is None:
                        failure = exc
                    continue
                edges_by_chunk[i] = got
                done += 1
                total = sum(len(v) for v in edges_by_chunk.values())
                print(f"  edges chunk {i + 1}/{len(parts)}: {len(got)} rows (fetched {done}/{len(todo)}, total {total})")
        if failure is not None:
            raise failure
    edges: list[tuple[int, int, int]] = []
    for i in range(len(parts)):
        edges.extend(edges_by_chunk[i])
    return edges


# --------------------------------------------------------------------------- files and resume state


def write_neurons_csv(path: Path, neurons: dict[int, dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        w.writerow(NEURON_COLUMNS)
        for bid in sorted(neurons):
            rec = neurons[bid]
            w.writerow([rec.get(c, "") for c in NEURON_COLUMNS])
    os.replace(tmp, path)
    return path


def read_neurons_csv(path: Path) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    with open(path, "r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                bid = int(row["bodyId"])
            except (KeyError, TypeError, ValueError):
                continue
            out[bid] = {c: (row.get(c) or "") for c in NEURON_COLUMNS}
    return out


def write_connections_csv(path: Path, edges: Sequence[tuple[int, int, int]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(EDGE_COLUMNS)
        for pre, post, wt in sorted(set(edges)):
            w.writerow((pre, post, wt))
    os.replace(tmp, path)
    return path


def write_csvs(out_dir: Path, neurons: dict[int, dict[str, str]], edges: Sequence[tuple[int, int, int]]) -> tuple[Path, Path]:
    """Write both CSVs (kept for callers of the previous API)."""
    return write_neurons_csv(out_dir / "neurons.csv", neurons), write_connections_csv(out_dir / "connections.csv", edges)


def _bodies_digest(neurons: dict[int, dict[str, str]]) -> str:
    h = hashlib.sha1()
    for bid in sorted(neurons):
        h.update(f"{bid}\n".encode("ascii"))
    return h.hexdigest()[:16]


def resume_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset": args.dataset, "server": args.server, "min_weight": int(args.min_weight), "chunk": int(args.chunk),
        "n_random": int(args.n_random), "no_fill": bool(args.no_fill),
    }


def load_resumable(out_dir: Path, state: dict[str, Any]) -> dict[int, dict[str, str]] | None:
    """The neuron selection of an interrupted run when ``<out>/chunks/state.json`` matches ``state``."""
    chunk_dir = out_dir / CHUNK_DIRNAME
    st_path = chunk_dir / STATE_FILENAME
    n_path = out_dir / "neurons.csv"
    if not (st_path.is_file() and n_path.is_file()):
        return None
    try:
        saved = json.loads(st_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if {k: saved.get(k) for k in state} != state:
        return None
    neurons = read_neurons_csv(n_path)
    if not neurons or saved.get("bodies") != _bodies_digest(neurons):
        return None
    return neurons


def save_state(out_dir: Path, state: dict[str, Any], neurons: dict[int, dict[str, str]]) -> None:
    chunk_dir = out_dir / CHUNK_DIRNAME
    chunk_dir.mkdir(parents=True, exist_ok=True)
    doc = dict(state)
    doc["bodies"] = _bodies_digest(neurons)
    doc["n_bodies"] = len(neurons)
    (chunk_dir / STATE_FILENAME).write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Fetch a MaleCNS core subset from neuPrint into neuPrint-export CSVs (anonymous urllib or token).",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default {DEFAULT_OUT})")
    ap.add_argument("--n-random", "--n-max", dest="n_random", type=int, default=DEFAULT_N_RANDOM,
                    help=f"target neuron count after random fill (default {DEFAULT_N_RANDOM})")
    ap.add_argument("--min-weight", type=int, default=DEFAULT_MIN_WEIGHT,
                    help=f"minimum ConnectsTo weight to fetch (default {DEFAULT_MIN_WEIGHT})")
    ap.add_argument("--chunk", "--chunk-ids", dest="chunk", type=int, default=DEFAULT_CHUNK,
                    help=f"presynaptic bodies per edge query (default {DEFAULT_CHUNK})")
    ap.add_argument("--token", default=None,
                    help=f"neuPrint token (default: {TOKEN_ENV}; omitted -> anonymous HTTP). Prefer the env var.")
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--no-fill", action="store_true", help="skip the random fill (core neurons only)")
    ap.add_argument("--chunk-types", type=int, default=50, help="type strings per neuron query (default 50)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"edge requests in flight (default {DEFAULT_CONCURRENCY}, RESEARCH 2.2)")
    ap.add_argument("--fresh", action="store_true", help="discard neurons.csv / chunks of an interrupted run")
    ap.add_argument("--anonymous", action="store_true", help="no-op (anonymous HTTP is the default without a token)")
    return ap


def main(argv: Sequence[str] | None = None, transport_factory: Callable[..., Any] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.n_random < 1 or args.min_weight < 1 or args.chunk < 1 or args.concurrency < 1:
        print("ERROR: --n-random, --min-weight, --chunk and --concurrency must be >= 1")
        return 2
    token = (args.token or "").strip() or os.environ.get(TOKEN_ENV, "").strip() or None
    out_dir = Path(args.out)
    chunk_dir = out_dir / CHUNK_DIRNAME
    state = resume_state(args)
    t0 = time.time()
    try:
        factory = transport_factory or make_transport
        transport = factory(args.server, args.dataset, token)
        print(f"dataset {args.dataset} at {args.server}; output {ascii_safe(out_dir)}")
        if args.fresh and chunk_dir.is_dir():
            shutil.rmtree(chunk_dir)
            print("  --fresh: discarded the chunks of a previous run")
        neurons = None if args.fresh else load_resumable(out_dir, state)
        if neurons is not None:
            n_have = sum(1 for i in range(len(chunks(sorted(neurons), int(args.chunk)))) if _chunk_path(chunk_dir, i).is_file())
            print(f"resuming: {len(neurons)} bodies from {ascii_safe(out_dir / 'neurons.csv')}, {n_have} edge chunks on disk")
        else:
            if chunk_dir.is_dir():
                shutil.rmtree(chunk_dir)  # chunks of a run with different arguments are useless
            print("step 1/5: distinct traced types")
            all_types = fetch_all_types(transport)
            types = matched_types(all_types, group_patterns())
            print(f"  {len(all_types)} types on the server, {len(types)} match GROUP_REGEX")
            neurons = {}
            print("step 2/5: neurons of matched types")
            fetch_neurons_by_types(transport, types, neurons, int(args.chunk_types))
            print("step 3/5: class / superclass core")
            fetch_neurons_by_class(transport, neurons)
            print("step 4/5: random fill")
            if args.no_fill:
                print("  skipped (--no-fill)")
            else:
                fetch_random_fill(transport, neurons, int(args.n_random))
            if not neurons:
                print("ERROR: the selection is empty (no Traced neurons matched)")
                return 1
            n_path = write_neurons_csv(out_dir / "neurons.csv", neurons)
            save_state(out_dir, state, neurons)
            print(f"  wrote {ascii_safe(n_path)} ({len(neurons)} neurons); edge chunks resume from here")
        print(f"step 5/5: edges among {len(neurons)} bodies (weight >= {args.min_weight}, chunk {args.chunk}, "
              f"<= {args.concurrency} in flight)")
        edges = fetch_edges(transport, list(neurons), int(args.min_weight), int(args.chunk), chunk_dir,
                            int(args.concurrency))
        e_path = write_connections_csv(out_dir / "connections.csv", edges)
        shutil.rmtree(chunk_dir, ignore_errors=True)
    except KeyboardInterrupt:
        print("interrupted (re-run the same command to resume from the stored chunks)")
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {ascii_safe(exc)}")
        if chunk_dir.is_dir():
            print("re-run the same command to resume from the stored chunks")
        return 1
    n_path = out_dir / "neurons.csv"
    print(f"wrote {ascii_safe(n_path)} ({len(neurons)} neurons) and {ascii_safe(e_path)} ({len(edges)} edges) "
          f"in {time.time() - t0:.0f} s")
    print("next: set FLY_CONNECTOME_SOURCE=neuprint (or FLY_CONNECTOME_SOURCE=csv FLY_CONNECTOME_DIR=<out>)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
