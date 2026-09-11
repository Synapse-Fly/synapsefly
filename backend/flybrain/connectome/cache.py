"""Connectome cache: ``cache_key``, ``load_cached``, ``store_cached`` (SPEC section c.8).

A cache entry is ``<cache_dir>/<key>.npz`` (uncompressed ``np.savez`` of every array, written by
``Connectome.save``) plus the sidecar ``<key>.json`` ({name, source, n, types, group sizes, meta}).
Writes are atomic (temp files + ``os.replace``) so a crash never leaves a half-written entry, and
``load_cached`` returns ``None`` for anything unreadable so the caller simply rebuilds.

Only numpy and the standard library are imported at module level. No biological numbers live here.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from .. import __version__
from .schema import Connectome, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings

__all__ = [
    "GENERATOR_VERSION",
    "ascii_safe",
    "cache_key",
    "load_cached",
    "store_cached",
    "resolve_connectome_dir",
    "settings_field",
]

log = logging.getLogger("flybrain.connectome.cache")

#: Version stamp of the synthetic generator that participates in every cache key. Bump it whenever
#: ``flybrain.connectome.synthetic`` changes the graph it produces for the same arguments. When that
#: module defines its own ``GENERATOR_VERSION`` it takes precedence (looked up lazily).
GENERATOR_VERSION: str = "1"

#: file suffixes whose (name, size, mtime) enter the key for csv/neuprint sources.
_DATA_SUFFIXES: tuple[str, ...] = (".csv", ".csv.gz", ".parquet", ".feather")

_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: defaults of SPEC section b for every field ``cache_key`` reads (so a partial settings object works).
_FIELD_DEFAULTS: dict[str, Any] = {
    "connectome_source": "synthetic",
    "connectome_dir": Path("data/connectome/malecns"),
    "n_neurons": 20_000,
    "mean_outdeg": 25,
    "synth_weights": "calibrated",
    "subset": "core",
    "min_weight": 3,
    "seed": 1337,
    "data_dir": Path("data"),
}


# --------------------------------------------------------------------------- settings helpers


def ascii_safe(obj: object) -> str:
    """``str(obj)`` with every non-ASCII character escaped (backslash-u form) for the cp1254 console (SPEC 0.1).

    Windows raises localized ``OSError`` messages (e.g. Turkish "Erisim engellendi" with a dotted i);
    every exception text that reaches a log line goes through this helper.
    """
    return str(obj).encode("ascii", "backslashreplace").decode("ascii")


def settings_field(settings: Any, name: str) -> Any:
    """``getattr(settings, name)`` falling back to the SPEC section b default (mapping keys also accepted)."""
    if isinstance(settings, Mapping):
        if name in settings:
            return settings[name]
    else:
        val = getattr(settings, name, None)
        if val is not None:
            return val
    if name in _FIELD_DEFAULTS:
        return _FIELD_DEFAULTS[name]
    raise AttributeError(f"settings has no field {name!r}")


def resolve_connectome_dir(settings: Any) -> Path:
    """Directory holding the CSV export for the configured source.

    ``neuprint`` -> ``<data_dir>/connectome/neuprint`` (always). ``csv`` (and ``synthetic``, for key
    purposes) -> ``connectome_dir``: absolute paths are used as-is; a relative path is resolved
    against the repository root (the parent of ``data_dir``, SPEC section a), and when that does
    not exist but ``<data_dir>/<path>`` does, the latter is used. Never touches the network.
    """
    source = str(settings_field(settings, "connectome_source"))
    data_dir = Path(settings_field(settings, "data_dir"))
    if source == "neuprint":
        return data_dir / "connectome" / "neuprint"
    raw = Path(settings_field(settings, "connectome_dir"))
    if raw.is_absolute():
        return raw
    root_based = data_dir.parent / raw
    if root_based.is_dir():
        return root_based
    data_based = data_dir / raw
    if data_based.is_dir():
        return data_based
    return root_based


def _dir_signature(d: Path) -> list[tuple[str, int, int]]:
    """Sorted ``(name, size, mtime_ns)`` of every data file directly inside ``d`` (empty when absent)."""
    if not d.is_dir():
        return []
    out: list[tuple[str, int, int]] = []
    for p in sorted(d.iterdir()):
        if not p.is_file():
            continue
        lname = p.name.lower()
        if not any(lname.endswith(s) for s in _DATA_SUFFIXES):
            continue
        st = p.stat()
        out.append((p.name, int(st.st_size), int(st.st_mtime_ns)))
    return out


_SYNTHETIC_MODULE = "flybrain.connectome.synthetic"


def _generator_version() -> str:
    """``GENERATOR_VERSION`` of the synthetic generator, resolved through ``sys.modules`` on every call.

    The lookup goes through ``sys.modules`` first (not the package attribute bound by a previous
    ``from . import synthetic``) so a replaced module object is honoured; when the module is
    absent it is imported lazily, and an import failure falls back to this module's constant.
    """
    mod = sys.modules.get(_SYNTHETIC_MODULE)
    if mod is None:
        try:
            mod = importlib.import_module(_SYNTHETIC_MODULE)
        except Exception:  # ImportError, or anything raised by a broken module: fall back to ours
            return GENERATOR_VERSION
    return str(getattr(mod, "GENERATOR_VERSION", GENERATOR_VERSION))


# --------------------------------------------------------------------------- public API


def cache_key(settings: "Settings | Any", extra: Mapping[str, object] | None = None) -> str:
    """sha1 over the sorted repr of the inputs that determine the built graph, truncated to 12 hex chars.

    Inputs: ``connectome_source``, the resolved ``connectome_dir`` plus (name, size, mtime_ns) of
    every data file in it (csv / neuprint sources only -- a deliberate narrowing of SPEC c.8: the
    synthetic generator never reads the directory, so dropping CSVs there must not invalidate a
    synthetic cache or break ``scripts/replay.py``'s ``connectome_key`` check), ``n_neurons``,
    ``mean_outdeg``, ``synth_weights``, ``subset``, ``min_weight``, ``seed``,
    ``flybrain.__version__``, ``GENERATOR_VERSION`` and the optional ``extra`` mapping.
    Missing settings fields take their SPEC section b defaults.
    """
    source = str(settings_field(settings, "connectome_source"))
    parts: dict[str, Any] = {
        "connectome_source": source,
        "n_neurons": int(settings_field(settings, "n_neurons")),
        "mean_outdeg": int(settings_field(settings, "mean_outdeg")),
        "synth_weights": str(settings_field(settings, "synth_weights")),
        "subset": str(settings_field(settings, "subset")),
        "min_weight": int(settings_field(settings, "min_weight")),
        "seed": int(settings_field(settings, "seed")),
        "flybrain_version": str(__version__),
        "generator_version": _generator_version(),
    }
    if source == "synthetic":
        parts["connectome_dir"] = None
        parts["dir_files"] = []
    else:
        d = resolve_connectome_dir(settings)
        parts["connectome_dir"] = str(d.resolve()) if d.exists() else str(d)
        parts["dir_files"] = _dir_signature(d)
    if extra:
        parts["extra"] = sorted((str(k), repr(v)) for k, v in extra.items())
    blob = repr(sorted(parts.items()))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _check_key(key: str) -> str:
    key = str(key)
    if not _KEY_RE.match(key):
        raise ValueError(f"cache key must match {_KEY_RE.pattern}, got {key!r}")
    return key


def _paths(key: str, cache_dir: Path) -> tuple[Path, Path, Path]:
    stem = Path(cache_dir) / key
    return stem, stem.with_suffix(".npz"), stem.with_suffix(".json")


def load_cached(key: str, cache_dir: Path, mmap: bool = False) -> Connectome | None:
    """Return the cached ``Connectome`` for ``key`` or ``None`` (missing, unreadable or inconsistent).

    ``mmap=True`` memory-maps the numeric arrays (``Connectome.load(mmap=True)``, full-graph mode);
    on Windows a mapped entry cannot be replaced or deleted while references are alive.
    """
    key = _check_key(key)
    stem, npz, js = _paths(key, cache_dir)
    if not (npz.is_file() and js.is_file()):
        return None
    try:
        conn = Connectome.load(stem, mmap=mmap)
    except Exception as exc:  # corrupt / partial / foreign file: rebuild instead of crashing
        log.warning("connectome cache %s is unreadable (%s: %s); ignoring it", key, type(exc).__name__, ascii_safe(exc))
        return None
    try:
        if conn.sign.shape[0] != conn.n or conn.post.shape[0] != conn.e or conn.weight.shape[0] != conn.e:
            raise ValueError("array lengths disagree with n / e")
        if int(conn.meta.get("e", conn.e)) != conn.e:
            raise ValueError(f"meta['e']={conn.meta.get('e')} != {conn.e} edges")
        rc = conn.meta.get("region_counts")
        if not isinstance(rc, dict) or int(sum(int(v) for v in rc.values())) != conn.n:
            raise ValueError("meta['region_counts'] does not sum to n")
    except Exception as exc:
        log.warning("connectome cache %s is inconsistent (%s); ignoring it", key, ascii_safe(exc))
        return None
    log.info("connectome cache hit %s: %s n=%d e=%d mmap=%s", key, conn.name, conn.n, conn.e, mmap)
    return conn


def store_cached(key: str, conn: Connectome, cache_dir: Path) -> Path:
    """Write ``<cache_dir>/<key>.npz`` + ``<key>.json`` atomically; returns the ``.npz`` path.

    ``conn.meta`` gains ``cache_key`` and ``cached`` (ISO timestamp) before saving. The arrays are
    written to ``<key>-tmp<pid>.*`` first and renamed into place with ``os.replace`` (both files),
    so readers never observe a partial entry; temp files are removed on failure.
    """
    key = _check_key(key)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem, npz, js = _paths(key, cache_dir)
    tmp_stem = cache_dir / f"{key}-tmp{os.getpid()}"
    tmp_npz, tmp_js = tmp_stem.with_suffix(".npz"), tmp_stem.with_suffix(".json")
    conn.meta["cache_key"] = key
    conn.meta["cached"] = utc_now_iso()
    try:
        conn.save(tmp_stem)
        os.replace(tmp_npz, npz)
        os.replace(tmp_js, js)
    except BaseException:
        for p in (tmp_npz, tmp_js):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
        raise
    size_mb = npz.stat().st_size / 1e6
    log.info("connectome cache stored %s: %s n=%d e=%d (%.1f MB)", key, conn.name, conn.n, conn.e, size_mb)
    return npz


def cached_arrays_are_memmaps(conn: Connectome) -> bool:
    """True when the edge arrays of ``conn`` are ``np.memmap`` instances (diagnostics / tests)."""
    return isinstance(conn.pre, np.memmap) and isinstance(conn.post, np.memmap)
