"""CSV loaders and the connectome dispatcher (SPEC section c.7).

Reads a neuPrint-style export (``neurons.csv`` + ``connections.csv``), a FlyWire-Codex download
(``classification`` + ``consolidated_cell_types`` + ``neurons`` + ``connections``, all ``.csv.gz``)
or the Codex MaleCNS mirror (``'Root ID'`` headers) into a ``Connectome``. Plain ``.csv`` and
``.csv.gz`` are read with the csv module (strings) and ``np.loadtxt`` (numeric columns);
``.parquet`` / ``.feather`` are accepted too and read with pyarrow, imported inside the reader only.

Provenance of the biology in this module: the NT -> sign convention is Shiu et al. 2024 with
histamine inhibitory (RESEARCH section 7 [L]); the region cascade is RESEARCH section 6 [V]; the
FlyWire -> MaleCNS superclass mapping is engineered [E]; the licences/citations are RESEARCH
section 1 / 2.4 [V]/[D].

Only numpy and the standard library are imported at module level; ``flybrain.connectome.synthetic``
is imported lazily by ``load_connectome`` for the synthetic source.
"""

from __future__ import annotations

import csv
import gzip
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Iterator

import numpy as np

from .cache import ascii_safe
from .csr import remap_ids, sum_duplicates
from .groups import GROUP_REGEX, region_of, resolve_groups, side_of, validate_groups
from .patches import apply_patches
from .schema import Connectome, count_groups, count_regions, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings

__all__ = [
    "Schema",
    "NT_ALIASES",
    "NT_SIGN",
    "CORE_CLASSES",
    "CORE_SUPERCLASSES",
    "CODEX_SUPERCLASS_MAP",
    "LICENSES",
    "CITATIONS",
    "nt_normalise",
    "nt_sign",
    "detect_schema",
    "load_csv_dir",
    "load_connectome",
    "find_file",
    "read_header",
]

log = logging.getLogger("flybrain.connectome.loaders")


class Schema(str, Enum):
    """Recognised on-disk layouts (SPEC section c.7)."""

    NEUPRINT = "neuprint"        # neurons.csv: bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt[,predictedNt]
                                 # connections.csv: bodyId_pre,bodyId_post,weight
    CODEX_FAFB = "codex_fafb"    # classification.csv.gz (root_id,flow,super_class,class,sub_class,hemilineage,side,nerve)
                                 # + consolidated_cell_types.csv.gz (root_id,primary_type,...) + neurons.csv.gz (root_id,...,nt_type,...)
                                 # + connections.csv.gz (pre_root_id,post_root_id,neuropil,syn_count,nt_type)
    CODEX_MCNS = "codex_mcns"    # 'Root ID','Super Class','Class','Soma side','Primary Cell Type','Predicted NT type' + connections_princeton.csv.gz


#: NT label aliases (SPEC section c.7, verbatim). Keys are compared after ``lower().strip()``.
NT_ALIASES: dict[str, str] = {"ach": "acetylcholine", "acetylcholine": "acetylcholine", "glut": "glutamate", "glu": "glutamate",
    "glutamate": "glutamate", "gaba": "gaba", "his": "histamine", "histamine": "histamine", "da": "dopamine", "dopamine": "dopamine",
    "oct": "octopamine", "octopamine": "octopamine", "ser": "serotonin", "5ht": "serotonin", "serotonin": "serotonin",
    "unc": "unknown", "unclear": "unknown", "unknown": "unknown", "": "unknown", "none": "unknown", "nan": "unknown"}
#: Shiu 2024 sign convention [L] with histamine inhibitory (ort chloride channel, Lappalainen 2024 [L]);
#: unknown / unclear -> +1 (RESEARCH section 7).
NT_SIGN: dict[str, float] = {"acetylcholine": 1, "dopamine": 1, "octopamine": 1, "serotonin": 1, "unknown": 1,
                             "gaba": -1, "glutamate": -1, "histamine": -1}

#: ``subset='core'`` keeps neurons of these classes / superclasses in addition to every GROUP_REGEX match.
CORE_CLASSES: frozenset[str] = frozenset({"gustatory", "Kenyon_Cell", "MBON", "DAN", "CX", "ALPN", "ALLN"})
CORE_SUPERCLASSES: frozenset[str] = frozenset({"descending_neuron", "cb_motor", "vnc_motor", "vnc_efferent"})

#: FlyWire / Codex ``super_class`` vocabulary -> MaleCNS superclass used by ``region_of`` [E] (RESEARCH section 6).
CODEX_SUPERCLASS_MAP: dict[str, str] = {
    "optic": "ol_intrinsic",
    "visual_projection": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "descending": "descending_neuron",
    "ascending": "ascending_neuron",
    "motor": "cb_motor",
    "sensory": "cb_sensory",
    "central": "cb_intrinsic",
    "endocrine": "cb_endocrine",
}

LICENSES: dict[Schema, str] = {
    Schema.NEUPRINT: "CC-BY 4.0 (MaleCNS)",
    Schema.CODEX_MCNS: "CC-BY 4.0 (MaleCNS)",
    Schema.CODEX_FAFB: "CC-BY-NC 4.0 (FlyWire)",
}
_MALECNS_CITATION = (
    "Berg et al. (2026) Sexual dimorphism in the complete Drosophila male central nervous system connectome. "
    "Cell. doi:10.1016/j.cell.2026.08.015 (data: FlyEM MaleCNS v1.0, CC-BY 4.0)"
)
_FLYWIRE_CITATION = (
    "Dorkenwald et al. (2024) Neuronal wiring diagram of an adult brain. Nature. doi:10.1038/s41586-024-07558-y; "
    "Schlegel et al. (2024) Whole-brain annotation and multi-connectome cell typing of Drosophila. Nature. "
    "doi:10.1038/s41586-024-07686-5 (data: FlyWire / Codex, CC-BY-NC 4.0)"
)
CITATIONS: dict[Schema, str] = {
    Schema.NEUPRINT: _MALECNS_CITATION,
    Schema.CODEX_MCNS: _MALECNS_CITATION,
    Schema.CODEX_FAFB: _FLYWIRE_CITATION,
}

#: gain used when the engine runs REAL synapse counts without a calibration file [?] (prior-art value,
#: RESEARCH section 12; the same default as literature-weight synthetic graphs, SPEC section c.6).
REAL_WEIGHTS_GAIN_DEFAULT: float = 0.65

_TRACED_STATUSES: frozenset[str] = frozenset({"Traced", ""})
#: largest per-row synapse count accepted (float32 keeps integers exact below 2**24; a real (pre, post)
#: pair never exceeds a few thousand synapses, RESEARCH section 5 [V]).
_MAX_WEIGHT: float = float(2**24)
_TABLE_SUFFIXES: tuple[str, ...] = (".csv", ".csv.gz", ".parquet", ".feather")

# normalised header aliases (lower-case, non-alphanumerics removed) per logical column
_NEURON_COLS: dict[str, tuple[str, ...]] = {
    "id": ("bodyid", "rootid", "id", "bodyidint", "root"),
    "type": ("type", "primarycelltype", "primarytype", "celltype", "cell_type"),
    "instance": ("instance", "name"),
    "superclass": ("superclass",),
    "class": ("class",),
    "subclass": ("subclass",),
    "side": ("somaside", "side"),
    "status": ("status",),
    "nt1": ("consensusnt",),          # priority 1: consensusNt
    "nt2": ("predictednt",),          # priority 2: predictedNt
    "nt3": ("nttype",),               # priority 3: nt_type
    "nt4": ("predictednttype",),      # priority 4: 'Predicted NT type' (normalised: predicted+nt+type)
}
_EDGE_COLS: dict[str, tuple[str, ...]] = {
    "pre": ("bodyidpre", "prerootid", "preptrootid", "prebodyid", "bodypre", "pre", "preid", "source", "presynaptic"),
    "post": ("bodyidpost", "postrootid", "postptrootid", "postbodyid", "bodypost", "post", "postid", "target", "postsynaptic"),
    "weight": ("weight", "syncount", "synapses", "synapsecount", "count", "nsyn", "n"),
    "nt": ("nttype", "nt", "predictednttype", "consensusnt"),
}
_NT_PRIORITY = ("nt1", "nt2", "nt3", "nt4")
_GROUP_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(rx) for rx in GROUP_REGEX.values())


# --------------------------------------------------------------------------- NT helpers


def nt_normalise(s: str | None) -> str:
    """``lower().strip()``; map through ``NT_ALIASES``; unrecognised -> ``'unknown'``."""
    if s is None:
        return "unknown"
    key = str(s).lower().strip()
    return NT_ALIASES.get(key, "unknown")


def nt_sign(s: str | None) -> float:
    """``NT_SIGN[nt_normalise(s)]`` as a float (+1.0 / -1.0)."""
    return float(NT_SIGN[nt_normalise(s)])


# --------------------------------------------------------------------------- file helpers


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_file(path: Path, stem: str) -> Path | None:
    """First existing ``<path>/<stem>{.csv,.csv.gz,.parquet,.feather}`` or ``None``."""
    path = Path(path)
    for suffix in _TABLE_SUFFIXES:
        p = path / f"{stem}{suffix}"
        if p.is_file():
            return p
    return None


def _is_arrow(p: Path) -> bool:
    return p.suffix.lower() in (".parquet", ".feather")


def _open_text(p: Path) -> IO[str]:
    if p.name.lower().endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace", newline="")
    return open(p, "r", encoding="utf-8", errors="replace", newline="")


def _read_arrow(p: Path, columns: list[str] | None = None) -> Any:
    """pyarrow Table of a .parquet/.feather file (pyarrow imported here only)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(f"reading {p.name} needs pyarrow (pip install pyarrow) or convert it to .csv") from exc
    if p.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(p, columns=columns)
    import pyarrow.feather as pf

    return pf.read_table(p, columns=columns)


def read_header(p: Path) -> list[str]:
    """Column names of a .csv/.csv.gz (first row) or .parquet/.feather (schema) file."""
    p = Path(p)
    if _is_arrow(p):
        return list(_read_arrow(p).schema.names)
    with _open_text(p) as fh:
        return [c.strip() for c in next(csv.reader(fh), [])]


def _column_map(header: list[str], aliases: dict[str, tuple[str, ...]]) -> dict[str, int]:
    """logical name -> column index for every logical column present in ``header``."""
    normed = [_norm_col(h) for h in header]
    out: dict[str, int] = {}
    for logical, names in aliases.items():
        for cand in names:
            if cand in normed:
                out[logical] = normed.index(cand)
                break
    return out


def _has_data_rows(p: Path) -> bool:
    with _open_text(p) as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if any(cell.strip() for cell in row):
                return True
    return False


def _read_string_columns(p: Path, cols: dict[str, int]) -> dict[str, list[str]]:
    """Read the listed columns of a table as lists of str (csv module; pyarrow for arrow files)."""
    out: dict[str, list[str]] = {k: [] for k in cols}
    if _is_arrow(p):
        table = _read_arrow(p)
        names = list(table.schema.names)
        for k, idx in cols.items():
            out[k] = ["" if v is None else str(v) for v in table.column(names[idx]).to_pylist()]
        return out
    with _open_text(p) as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        items = list(cols.items())
        for row in reader:
            if not row or not any(cell.strip() for cell in row):
                continue
            for k, idx in items:
                out[k].append(row[idx].strip() if idx < len(row) else "")
    return out


def _read_edge_table(p: Path, cols: dict[str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """(pre_id int64, post_id int64, weight float64, nt U16 or None) of a connections table.

    Numeric columns go through ``np.loadtxt`` with a structured dtype (int64 for the ids so 64-bit
    FlyWire root ids stay exact, float64 for the weight so integer and float exports both parse);
    an optional NT column is read in the same pass.
    """
    want_nt = "nt" in cols
    if _is_arrow(p):
        table = _read_arrow(p)
        names = list(table.schema.names)
        pre = np.asarray(table.column(names[cols["pre"]]).to_numpy(zero_copy_only=False), dtype=np.int64)
        post = np.asarray(table.column(names[cols["post"]]).to_numpy(zero_copy_only=False), dtype=np.int64)
        w = np.asarray(table.column(names[cols["weight"]]).to_numpy(zero_copy_only=False), dtype=np.float64)
        nt = None
        if want_nt:
            nt = np.asarray(["" if v is None else str(v) for v in table.column(names[cols["nt"]]).to_pylist()], dtype="U16")
        return pre, post, w, nt
    if not _has_data_rows(p):
        empty_nt = np.zeros(0, dtype="U16") if want_nt else None
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64), empty_nt
    usecols = [cols["pre"], cols["post"], cols["weight"]]
    dtype: list[tuple[str, str]] = [("pre", "i8"), ("post", "i8"), ("w", "f8")]
    if want_nt:
        usecols.append(cols["nt"])
        dtype.append(("nt", "U16"))
    with _open_text(p) as fh:
        try:
            arr = np.loadtxt(fh, delimiter=",", skiprows=1, usecols=usecols, dtype=dtype, comments=None,
                             quotechar='"', ndmin=1)
        except ValueError as exc:  # numpy's message names the row/column but never the file
            raise ValueError(
                f"{p.name}: cannot parse the numeric columns of the connection table "
                f"(pre/post must be integers, weight a number, rows must not be ragged): {exc}"
            ) from exc
    nt = np.asarray(arr["nt"]) if want_nt else None
    return np.asarray(arr["pre"], dtype=np.int64), np.asarray(arr["post"], dtype=np.int64), np.asarray(arr["w"], dtype=np.float64), nt


# --------------------------------------------------------------------------- schema detection


def detect_schema(path: Path) -> Schema:
    """Sniff the header of ``neurons.csv(.gz)`` / ``classification.csv.gz`` in ``path``; ``ValueError`` if none matches.

    Probe order follows SPEC section c.7 (``neurons`` before ``classification``): ``neurons`` with a
    ``bodyId`` column -> ``NEUPRINT`` (so a directory that holds BOTH a neuPrint export and a
    left-over Codex ``classification.csv.gz`` resolves to the richer, explicitly typed neuPrint
    schema instead of the Codex one); then ``classification`` with ``root_id`` + ``super_class`` ->
    ``CODEX_FAFB``, or ``CODEX_MCNS`` when its header is the MaleCNS mirror's title case
    (``'Root ID'`` / ``'Super Class'``); finally ``neurons`` with ``'Root ID'`` / ``'Primary Cell
    Type'`` -> ``CODEX_MCNS``.
    """
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"connectome directory does not exist: {path}")
    neu_file = find_file(path, "neurons")
    neu_raw: list[str] = [h.strip() for h in read_header(neu_file)] if neu_file is not None else []
    neu_normed = {_norm_col(h) for h in neu_raw}
    if neu_file is not None and "bodyid" in neu_normed:
        return Schema.NEUPRINT
    cls_file = find_file(path, "classification")
    if cls_file is not None:
        raw = [h.strip() for h in read_header(cls_file)]
        normed = {_norm_col(h) for h in raw}
        if "rootid" in normed and "superclass" in normed:
            if "Root ID" in raw or "Super Class" in raw:
                return Schema.CODEX_MCNS
            return Schema.CODEX_FAFB
    if neu_file is not None:
        raw, normed = neu_raw, neu_normed
        if "Root ID" in raw or "Super Class" in raw or "Primary Cell Type" in raw:
            return Schema.CODEX_MCNS
        if "rootid" in normed and ("superclass" in normed or "primarytype" in normed or "primarycelltype" in normed):
            return Schema.CODEX_MCNS
        if "rootid" in normed:
            raise ValueError(
                f"{neu_file.name} has FlyWire root_id columns but {path} lacks classification.csv(.gz) "
                "(Codex downloads need classification + consolidated_cell_types + neurons + connections)"
            )
    raise ValueError(
        f"no recognised connectome schema in {path}: expected neurons.csv(.gz) with a bodyId column "
        "(neuPrint export), classification.csv.gz with root_id/super_class (FlyWire Codex) or "
        "neurons.csv(.gz) with 'Root ID' (Codex MaleCNS mirror)"
    )


# --------------------------------------------------------------------------- neuron table


@dataclass(slots=True)
class _NeuronTable:
    """Per-neuron string columns in file order (before status filtering / sorting)."""

    ids: np.ndarray                       # int64
    type: list[str]
    instance: list[str]
    superclass: list[str]
    cls: list[str]
    subclass: list[str]
    side: list[str]
    status: list[str] | None
    nt: list[str]                         # raw label chosen by priority (may be '')
    files: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.ids.shape[0])


def _parse_ids(values: list[str], what: str) -> np.ndarray:
    try:
        return np.asarray([int(float(v)) if ("e" in v.lower() or "." in v) else int(v) for v in values], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"non-integer {what} id in neuron table: {exc}") from exc


def _pick_nt(columns: dict[str, list[str]], n: int) -> list[str]:
    """Per row, the first non-empty label in the priority consensusNt > predictedNt > nt_type > 'Predicted NT type'."""
    avail = [columns[k] for k in _NT_PRIORITY if k in columns]
    out: list[str] = []
    for i in range(n):
        label = ""
        for col in avail:
            v = col[i]
            if v and v.lower() not in ("none", "nan", "null"):
                label = v
                break
        out.append(label)
    return out


def _read_neuprint_neurons(p: Path) -> _NeuronTable:
    header = read_header(p)
    cols = _column_map(header, _NEURON_COLS)
    if "id" not in cols:
        raise ValueError(f"{p.name}: no bodyId column in header {header}")
    data = _read_string_columns(p, cols)
    n = len(data["id"])
    empty = [""] * n

    def col(k: str) -> list[str]:
        return data.get(k, empty)

    return _NeuronTable(
        ids=_parse_ids(data["id"], "bodyId"),
        type=col("type"), instance=col("instance"), superclass=col("superclass"), cls=col("class"),
        subclass=col("subclass"), side=col("side"), status=data.get("status"), nt=_pick_nt(data, n), files=[p.name],
    )


def _read_codex_fafb_neurons(path: Path) -> _NeuronTable:
    cls_file = find_file(path, "classification")
    if cls_file is None:
        raise ValueError(f"{path}: classification.csv(.gz) is missing")
    c_header = read_header(cls_file)
    c_cols = _column_map(c_header, _NEURON_COLS)
    if "id" not in c_cols:
        raise ValueError(f"{cls_file.name}: no root_id column in header {c_header}")
    cdata = _read_string_columns(cls_file, c_cols)
    n = len(cdata["id"])
    ids = _parse_ids(cdata["id"], "root")
    files = [cls_file.name]

    types = [""] * n
    ct_file = find_file(path, "consolidated_cell_types")
    if ct_file is not None:
        files.append(ct_file.name)
        t_cols = _column_map(read_header(ct_file), _NEURON_COLS)
        if "id" in t_cols and "type" in t_cols:
            tdata = _read_string_columns(ct_file, {"id": t_cols["id"], "type": t_cols["type"]})
            lookup = dict(zip(tdata["id"], tdata["type"]))
            types = [lookup.get(rid, "") for rid in cdata["id"]]

    nt = [""] * n
    instance = [""] * n
    neu_file = find_file(path, "neurons")
    if neu_file is not None:
        files.append(neu_file.name)
        n_cols = _column_map(read_header(neu_file), _NEURON_COLS)
        wanted = {k: n_cols[k] for k in ("id", "instance", *_NT_PRIORITY) if k in n_cols}
        if "id" in wanted:
            ndata = _read_string_columns(neu_file, wanted)
            nt_lab = _pick_nt(ndata, len(ndata["id"]))
            nt_lookup = dict(zip(ndata["id"], nt_lab))
            nt = [nt_lookup.get(rid, "") for rid in cdata["id"]]
            if "instance" in ndata:
                inst_lookup = dict(zip(ndata["id"], ndata["instance"]))
                instance = [inst_lookup.get(rid, "") for rid in cdata["id"]]
    if ct_file is None or all(t == "" for t in types):
        # without consolidated_cell_types every type string stays '' -> no functional group resolves and
        # validate_groups rejects the graph; say so here instead of only in that later failure.
        log.warning(
            "%s: no usable consolidated_cell_types.csv(.gz) -> every type label is empty (groups cannot "
            "resolve); a complete Codex download has classification + consolidated_cell_types + neurons + "
            "connections", ascii_safe(path.name or path),
        )
    if neu_file is None:
        log.warning(
            "%s: no neurons.csv(.gz) -> no per-neuron nt_type; neurotransmitters fall back to the "
            "connection table's NT column (majority per presynaptic cell)", ascii_safe(path.name or path),
        )
    empty = [""] * n
    return _NeuronTable(
        ids=ids, type=types, instance=instance, superclass=cdata.get("superclass", empty), cls=cdata.get("class", empty),
        subclass=cdata.get("subclass", empty), side=cdata.get("side", empty), status=cdata.get("status"), nt=nt, files=files,
    )


def _read_codex_mcns_neurons(path: Path) -> _NeuronTable:
    p = find_file(path, "neurons") or find_file(path, "classification")
    if p is None:
        raise ValueError(f"{path}: neurons.csv(.gz) is missing")
    table = _read_neuprint_neurons(p)  # the alias table covers 'Root ID', 'Super Class', 'Soma side', ...
    ct_file = find_file(path, "consolidated_cell_types")
    if ct_file is not None and all(t == "" for t in table.type):
        t_cols = _column_map(read_header(ct_file), _NEURON_COLS)
        if "id" in t_cols and "type" in t_cols:
            tdata = _read_string_columns(ct_file, {"id": t_cols["id"], "type": t_cols["type"]})
            lookup = dict(zip(tdata["id"], tdata["type"]))
            id_str = [str(v) for v in table.ids.tolist()]
            table.type = [lookup.get(rid, "") for rid in id_str]
            table.files.append(ct_file.name)
    return table


def _edge_file(path: Path, schema: Schema) -> Path:
    stems = ("connections_princeton", "connections") if schema is Schema.CODEX_MCNS else ("connections",)
    for stem in stems:
        p = find_file(path, stem)
        if p is not None:
            return p
    raise FileNotFoundError(f"{path}: connections.csv(.gz) is missing")


def _side_token(s: str) -> str:
    """FlyWire 'left'/'right'/'center' and MaleCNS 'L'/'R'/'M' -> the one-letter token ``side_of`` expects."""
    t = s.strip().lower()
    if t in ("l", "left"):
        return "L"
    if t in ("r", "right"):
        return "R"
    if t in ("m", "center", "centre", "middle", "midline"):
        return "M"
    return s


def _superclass_token(s: str) -> str:
    t = s.strip()
    return CODEX_SUPERCLASS_MAP.get(t.lower(), t)


# --------------------------------------------------------------------------- main loader


def load_csv_dir(path: Path, *, min_weight: int = 3, subset: str = "core", n_max: int | None = 20_000,
                 seed: int = 0, name: str | None = None) -> Connectome:
    """Read the schema's files (.csv or .csv.gz; .parquet/.feather via pyarrow) into a validated ``Connectome``.

    Rows with ``status`` outside {'Traced', ''} are dropped when a status column exists. Sides come
    from ``side_of(somaSide, instance)`` (FlyWire ``left``/``right``/``center`` accepted). NT per neuron:
    consensusNt > predictedNt > nt_type > 'Predicted NT type'; a neuron whose own label is unknown
    but whose outgoing edges carry an NT column receives the synapse-weighted majority of those
    edge labels (edges without an NT keep using the presynaptic neuron's NT, which is the only
    place the data model stores a sign). Duplicate (pre, post) rows are summed (Codex has one row
    per neuropil), then edges with ``weight < min_weight`` are pruned and self-loops dropped.
    ``subset='core'``: keep every neuron matching any GROUP_REGEX or class in ``CORE_CLASSES`` or
    superclass in ``CORE_SUPERCLASSES``, then random-fill with other neurons to ``n_max``
    (``np.random.default_rng(seed)``; ``n_max=None`` = core only); ``subset='all'``: everything.
    Neurons are ordered by ascending external id. ``source='csv'``; ``meta['license']`` is
    'CC-BY 4.0 (MaleCNS)' for NEUPRINT/CODEX_MCNS and 'CC-BY-NC 4.0 (FlyWire)' for CODEX_FAFB;
    ``meta['gain_default']`` = 0.65 [?] for real synapse counts; ``meta['weights_mode'] = 'real'``.
    """
    path = Path(path)
    if subset not in ("core", "all"):
        raise ValueError(f"subset must be 'core' or 'all', got {subset!r}")
    if min_weight < 1:
        raise ValueError(f"min_weight must be >= 1, got {min_weight}")
    schema = detect_schema(path)
    if schema is Schema.NEUPRINT:
        neu = find_file(path, "neurons")
        assert neu is not None
        table = _read_neuprint_neurons(neu)
    elif schema is Schema.CODEX_FAFB:
        table = _read_codex_fafb_neurons(path)
    else:
        table = _read_codex_mcns_neurons(path)
    n_file = table.n
    edge_path = _edge_file(path, schema)
    table.files.append(edge_path.name)

    # ---- status filter, duplicate ids, sort by external id ------------------------------
    keep = np.ones(n_file, dtype=bool)
    n_untraced = 0
    if table.status is not None:
        st = np.asarray([s.strip() for s in table.status], dtype=object)
        keep = np.fromiter((s in _TRACED_STATUSES for s in st), dtype=bool, count=n_file)
        n_untraced = int((~keep).sum())
    order = np.flatnonzero(keep)
    order = order[np.argsort(table.ids[order], kind="stable")]
    ids_sorted = table.ids[order]
    if ids_sorted.size > 1:
        first = np.ones(ids_sorted.size, dtype=bool)
        first[1:] = ids_sorted[1:] != ids_sorted[:-1]
        order = order[first]
        ids_sorted = ids_sorted[first]
    n_all = int(order.size)
    if n_all == 0:
        raise ValueError(f"{path}: no traced neurons found in the neuron table")

    def pick(col: list[str]) -> list[str]:
        return [col[i] for i in order.tolist()]

    type_lab = pick(table.type)
    instance = pick(table.instance)
    superclass = [_superclass_token(s) for s in pick(table.superclass)]
    cls = [s.strip() for s in pick(table.cls)]
    side_tok = pick(table.side)
    nt_lab = [nt_normalise(s) for s in pick(table.nt)]

    side = np.asarray([side_of(_side_token(s), inst) for s, inst in zip(side_tok, instance)], dtype=np.int8)
    region = np.asarray([region_of(sc, c, t) for sc, c, t in zip(superclass, cls, type_lab)], dtype=np.uint8)

    # ---- edges -----------------------------------------------------------------------------------
    e_header = read_header(edge_path)
    e_cols = _column_map(e_header, _EDGE_COLS)
    for k in ("pre", "post", "weight"):
        if k not in e_cols:
            raise ValueError(f"{edge_path.name}: no {k} column in header {e_header}")
    pre_id, post_id, w_all, edge_nt = _read_edge_table(edge_path, e_cols)
    e_file = int(pre_id.shape[0])

    # edge-NT majority for neurons without an NT of their own
    nt_from_edges = 0
    if edge_nt is not None and e_file and any(v == "unknown" for v in nt_lab):
        pos = np.searchsorted(ids_sorted, pre_id)
        pos_c = np.minimum(pos, n_all - 1)
        found = ids_sorted[pos_c] == pre_id
        unknown_mask = np.asarray([v == "unknown" for v in nt_lab], dtype=bool)
        sel = found & unknown_mask[pos_c]
        if sel.any():
            labels = np.asarray([nt_normalise(v) for v in edge_nt[sel].tolist()], dtype=object)
            uniq, codes = np.unique(labels.astype(str), return_inverse=True)
            k = int(uniq.size)
            acc = np.zeros(n_all * k, dtype=np.float64)
            np.add.at(acc, pos_c[sel].astype(np.int64) * k + codes.astype(np.int64), w_all[sel])
            acc = acc.reshape(n_all, k)
            for i in np.flatnonzero(unknown_mask).tolist():
                row = acc[i]
                if row.sum() > 0:
                    best = str(uniq[int(row.argmax())])
                    if best != "unknown":
                        nt_lab[i] = best
                        nt_from_edges += 1

    sign = np.asarray([nt_sign(v) for v in nt_lab], dtype=np.float32)

    # ---- subset selection ----------------------------------------------------------------------
    uniq_types = sorted(set(type_lab))
    t_index = {t: i for i, t in enumerate(uniq_types)}
    type_idx_all = np.asarray([t_index[t] for t in type_lab], dtype=np.int64)
    core_type = np.fromiter(
        (t != "" and any(p.fullmatch(t) is not None for p in _GROUP_PATTERNS) for t in uniq_types),
        dtype=bool, count=len(uniq_types),
    )
    core = core_type[type_idx_all]
    core |= np.fromiter((c in CORE_CLASSES for c in cls), dtype=bool, count=n_all)
    core |= np.fromiter((s in CORE_SUPERCLASSES for s in superclass), dtype=bool, count=n_all)
    n_core = int(core.sum())
    if subset == "all":
        sel_idx = np.arange(n_all, dtype=np.int64)
        n_fill = 0
    else:
        chosen = core.copy()
        n_fill = 0
        if n_max is not None and n_core < int(n_max):
            candidates = np.flatnonzero(~core)
            n_fill = min(int(n_max) - n_core, int(candidates.size))
            if n_fill > 0:
                rng = np.random.default_rng(int(seed))
                chosen[rng.choice(candidates, size=n_fill, replace=False)] = True
        sel_idx = np.flatnonzero(chosen)
    n = int(sel_idx.size)
    body_id = ids_sorted[sel_idx].astype(np.int64)  # still sorted (sel_idx increasing)

    pre_m, post_m, keep_mask = remap_ids(pre_id, post_id, body_id)
    pre_m, post_m, w_kept = pre_m[keep_mask], post_m[keep_mask], w_all[keep_mask]
    e_inside = int(pre_m.shape[0])
    if e_inside:
        w_max = float(np.abs(w_kept).max())
        if not np.isfinite(w_kept).all() or w_max > _MAX_WEIGHT:
            raise ValueError(
                f"{edge_path.name}: synapse counts must be finite and <= {_MAX_WEIGHT:g} (max |weight| {w_max:g})"
            )
        pre_s, post_s, w_s = sum_duplicates(pre_m, post_m, w_kept, n)  # summed, sorted, self-loops dropped
        if not np.isfinite(w_s).all():
            raise ValueError(f"{edge_path.name}: summed synapse counts of a (pre, post) pair overflow float32")
        prune = w_s >= np.float32(min_weight)
        pre_s, post_s, w_s = pre_s[prune], post_s[prune], w_s[prune]
    else:
        pre_s = np.zeros(0, dtype=np.int32)
        post_s = np.zeros(0, dtype=np.int32)
        w_s = np.zeros(0, dtype=np.float32)

    # ---- per-neuron arrays of the kept set -----------------------------------------------
    sel_list = sel_idx.tolist()
    kept_types = [type_lab[i] for i in sel_list]
    types = sorted(set(kept_types))
    t_index2 = {t: i for i, t in enumerate(types)}
    type_idx = np.asarray([t_index2[t] for t in kept_types], dtype=np.int32)
    side_k = np.ascontiguousarray(side[sel_idx], dtype=np.int8)
    region_k = np.ascontiguousarray(region[sel_idx], dtype=np.uint8)
    sign_k = np.ascontiguousarray(sign[sel_idx], dtype=np.float32)
    nt_k = np.asarray([nt_lab[i] for i in sel_list], dtype=object)
    groups = resolve_groups(types, type_idx, side_k)

    if name is None:
        name = f"{schema.value}-{path.name}-{n}-{subset}-w{int(min_weight)}"
    meta: dict[str, Any] = {
        "e": int(pre_s.shape[0]),
        "synapses": float(w_s.sum(dtype=np.float64)) if w_s.size else 0.0,
        "license": LICENSES[schema],
        "citation": CITATIONS[schema],
        "seed": int(seed),
        "build_args": {
            "path": str(path), "schema": schema.value, "files": list(table.files), "min_weight": int(min_weight),
            "subset": subset, "n_max": None if n_max is None else int(n_max), "seed": int(seed),
            "n_file": int(n_file), "n_untraced": int(n_untraced), "n_traced": int(n_all), "n_core": int(n_core),
            "n_fill": int(n_fill), "e_file": int(e_file), "e_inside": int(e_inside), "nt_from_edges": int(nt_from_edges),
        },
        "patches_applied": [],
        "gain_default": REAL_WEIGHTS_GAIN_DEFAULT,
        "weights_mode": "real",
        "engineered_edges": [],
        "region_counts": count_regions(region_k),
        "group_counts": count_groups(groups),
        "created": utc_now_iso(),
        "note": f"real connectome data loaded from a {schema.value} export ({subset} subset); synapse counts are unscaled",
    }
    conn = Connectome(
        name=str(name),
        source="csv",
        n=n,
        pre=np.ascontiguousarray(pre_s, dtype=np.int32),
        post=np.ascontiguousarray(post_s, dtype=np.int32),
        weight=np.ascontiguousarray(w_s, dtype=np.float32),
        sign=sign_k,
        region=region_k,
        side=side_k,
        types=types,
        type_idx=type_idx,
        body_id=np.ascontiguousarray(body_id, dtype=np.int64),
        nt=nt_k,
        groups=groups,
        meta=meta,
    )
    conn.validate()
    log.info(
        "loaded %s from %s: n=%d (of %d traced, %d core, %d fill) e=%d (of %d rows, min_weight=%d)",
        schema.value, path, n, n_all, n_core, n_fill, conn.e, e_file, min_weight,
    )
    return conn


# --------------------------------------------------------------------------- dispatcher


def _fetch_command(target: Path) -> str:
    return f"py -3 scripts/fetch_neuprint.py --out \"{target}\""


def _build(settings: Any, source: str) -> Connectome:
    from .cache import resolve_connectome_dir, settings_field

    n_neurons = int(settings_field(settings, "n_neurons"))
    seed = int(settings_field(settings, "seed"))
    name_override = str(getattr(settings, "connectome_name", "") or "") or None
    if source == "synthetic":
        try:
            from .synthetic import build_synthetic  # lazy: written by another engineer, may be absent
        except ImportError as exc:
            raise ImportError(
                "FLY_CONNECTOME_SOURCE=synthetic needs flybrain.connectome.synthetic (build_synthetic); "
                f"import failed: {exc}"
            ) from exc
        return build_synthetic(
            n_neurons=n_neurons,
            seed=seed,
            mean_outdeg=int(settings_field(settings, "mean_outdeg")),
            weights=str(settings_field(settings, "synth_weights")),
        )
    target = resolve_connectome_dir(settings)
    neurons = find_file(target, "neurons") or find_file(target, "classification")
    if neurons is None:
        if source == "neuprint":
            raise FileNotFoundError(
                f"neuPrint export not found in {target} (neurons.csv / connections.csv). Fetch it once with:\n  "
                + _fetch_command(target)
                + "\n(set NEUPRINT_APPLICATION_CREDENTIALS for token access; the sim never fetches at runtime)"
            )
        raise FileNotFoundError(
            f"no neurons.csv(.gz) / classification.csv.gz in FLY_CONNECTOME_DIR={target}. Produce the CSVs with\n  "
            "py -3 scripts/prepare_malecns.py   (GCS flat files, needs pyarrow)\nor\n  "
            + _fetch_command(target)
            + "\nor point FLY_CONNECTOME_DIR at a directory holding a neuPrint-export or Codex download"
        )
    return load_csv_dir(
        target,
        min_weight=int(settings_field(settings, "min_weight")),
        subset=str(settings_field(settings, "subset")),
        n_max=n_neurons,
        seed=seed,
        name=name_override,
    )


def load_connectome(settings: "Settings | Any") -> Connectome:
    """Dispatch on ``settings.connectome_source`` and serve from the cache when possible.

    synthetic -> ``build_synthetic(n_neurons, seed, mean_outdeg, synth_weights)`` (module imported
    lazily); csv -> ``load_csv_dir(connectome_dir, ...)``; neuprint ->
    ``load_csv_dir(data_dir/'connectome'/'neuprint', ...)`` (``FileNotFoundError`` with the exact
    fetch command when the files are missing). Always: ``cache.load_cached(key)`` first
    (memory-mapped for csv/neuprint with ``subset='all'``), else build, ``apply_patches`` and
    ``store_cached`` (so the cached graph is already patched and the mandatory ``apply_patches``
    afterwards is a no-op even on a memory-mapped full graph); then ``validate_groups``.
    ``connectome_name`` overrides the display name; ``meta['cache_key']`` is set. A cache write
    failure is logged, never fatal. Never performs network I/O.
    """
    from . import cache as _cache

    source = str(_cache.settings_field(settings, "connectome_source"))
    if source not in ("synthetic", "csv", "neuprint"):
        raise ValueError(f"FLY_CONNECTOME_SOURCE must be synthetic|csv|neuprint, got {source!r}")
    key = _cache.cache_key(settings)
    cache_dir = Path(_cache.settings_field(settings, "data_dir")) / "cache"
    subset = str(_cache.settings_field(settings, "subset"))
    mmap = source != "synthetic" and subset == "all"
    conn = _cache.load_cached(key, cache_dir, mmap=mmap)
    if conn is None:
        conn = _build(settings, source)
        conn = apply_patches(conn)
        try:
            _cache.store_cached(key, conn, cache_dir)
        except Exception as exc:  # errors never stop the sim (SPEC 0.1); localized OSError text is escaped
            log.warning("could not store connectome cache %s in %s: %s: %s", key, ascii_safe(cache_dir),
                        type(exc).__name__, ascii_safe(exc))
    conn = apply_patches(conn)  # idempotent
    name_override = str(getattr(settings, "connectome_name", "") or "")
    if name_override:
        conn.name = name_override
    conn.meta["cache_key"] = key
    validate_groups(conn)
    return conn


def iter_group_sizes(conn: Connectome) -> Iterator[tuple[str, int]]:
    """(group, size) pairs in ``GROUP_REGEX`` order (diagnostics for scripts)."""
    for gname in GROUP_REGEX:
        yield gname, int(np.asarray(conn.groups.get(gname, ())).size)
