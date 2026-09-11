"""Tests for flybrain.connectome.cache (SPEC section c.8)."""

from __future__ import annotations

import gc
import json
import logging
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain.connectome import cache as cache_mod
from flybrain.connectome.cache import (
    GENERATOR_VERSION,
    ascii_safe,
    cache_key,
    cached_arrays_are_memmaps,
    load_cached,
    resolve_connectome_dir,
    store_cached,
)
from flybrain.connectome.schema import Connectome

HEX12 = re.compile(r"^[0-9a-f]{12}$")


def make_settings(tmp_path: Path, **over) -> SimpleNamespace:
    base = dict(
        connectome_source="synthetic",
        connectome_dir=Path("data/connectome/malecns"),
        connectome_name="",
        n_neurons=20_000,
        mean_outdeg=25,
        synth_weights="calibrated",
        subset="core",
        min_weight=3,
        seed=1337,
        data_dir=tmp_path / "data",
    )
    base.update(over)
    return SimpleNamespace(**base)


# --------------------------------------------------------------------------- cache_key


def test_key_is_12_hex_and_deterministic(tmp_path):
    s = make_settings(tmp_path)
    k1, k2 = cache_key(s), cache_key(s)
    assert HEX12.match(k1) and k1 == k2


def test_key_accepts_partial_settings(settings):
    # the conftest placeholder only has data_dir / out_dir / seed: defaults of SPEC section b fill the rest
    k = cache_key(settings)
    assert HEX12.match(k)
    assert k == cache_key(SimpleNamespace(data_dir=settings.data_dir, seed=0))
    assert k != cache_key(SimpleNamespace(data_dir=settings.data_dir, seed=1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("n_neurons", 20_001),
        ("mean_outdeg", 26),
        ("synth_weights", "literature"),
        ("subset", "all"),
        ("min_weight", 4),
        ("seed", 1338),
        ("connectome_source", "csv"),
    ],
)
def test_key_changes_with_each_input(tmp_path, field, value):
    base = make_settings(tmp_path)
    assert cache_key(base) != cache_key(make_settings(tmp_path, **{field: value}))


def test_key_changes_with_extra(tmp_path):
    s = make_settings(tmp_path)
    assert cache_key(s) != cache_key(s, extra={"gain": 0.5})
    assert cache_key(s, extra={"gain": 0.5}) == cache_key(s, extra={"gain": 0.5})
    assert cache_key(s, extra={"gain": 0.5}) != cache_key(s, extra={"gain": 0.6})
    assert cache_key(s, extra={}) == cache_key(s)


def test_key_tracks_flybrain_version(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    k = cache_key(s)
    monkeypatch.setattr(cache_mod, "__version__", "9.9.9")
    assert cache_key(s) != k


def test_key_tracks_generator_version(tmp_path, monkeypatch):
    s = make_settings(tmp_path)
    fake = types.ModuleType("flybrain.connectome.synthetic")
    fake.GENERATOR_VERSION = "test-A"
    monkeypatch.setitem(sys.modules, "flybrain.connectome.synthetic", fake)
    k_a = cache_key(s)
    fake.GENERATOR_VERSION = "test-B"
    assert cache_key(s) != k_a
    assert isinstance(GENERATOR_VERSION, str) and GENERATOR_VERSION


def test_key_sees_replaced_generator_module_after_real_import(tmp_path, monkeypatch):
    # regression: once the real synthetic module had been imported (package attribute bound), a module
    # injected into sys.modules was ignored and the key no longer tracked GENERATOR_VERSION
    s = make_settings(tmp_path)
    real = pytest.importorskip("flybrain.connectome.synthetic")
    k_real = cache_key(s)
    assert cache_mod._generator_version() == str(real.GENERATOR_VERSION)
    fake = types.ModuleType("flybrain.connectome.synthetic")
    fake.GENERATOR_VERSION = "ZZZ"
    monkeypatch.setitem(sys.modules, "flybrain.connectome.synthetic", fake)
    assert cache_mod._generator_version() == "ZZZ"
    assert cache_key(s) != k_real
    monkeypatch.setitem(sys.modules, "flybrain.connectome.synthetic", real)
    assert cache_key(s) == k_real
    monkeypatch.setitem(sys.modules, "flybrain.connectome.synthetic", None)  # import failure -> local constant
    assert cache_mod._generator_version() == GENERATOR_VERSION


def test_key_tracks_csv_directory_contents(tmp_path):
    d = tmp_path / "csvdir"
    d.mkdir()
    (d / "neurons.csv").write_text("bodyId,type\n1,DNp01\n", encoding="utf-8")
    s = make_settings(tmp_path, connectome_source="csv", connectome_dir=d)
    k0 = cache_key(s)
    (d / "README.md").write_text("not data", encoding="utf-8")  # non-data files are ignored
    assert cache_key(s) == k0
    (d / "neurons.csv").write_text("bodyId,type\n1,DNp01\n2,DNa02\n", encoding="utf-8")  # size + mtime change
    assert cache_key(s) != k0
    (d / "connections.csv.gz").write_bytes(b"\x1f\x8b")  # a new data file changes the key too
    k2 = cache_key(s)
    assert k2 != k0
    # a synthetic key ignores the directory entirely
    syn = make_settings(tmp_path, connectome_source="synthetic", connectome_dir=d)
    k_syn = cache_key(syn)
    (d / "neurons.csv").write_text("bodyId,type\n3,MN9\n", encoding="utf-8")
    assert cache_key(syn) == k_syn


def test_key_for_neuprint_source_uses_data_dir(tmp_path):
    s = make_settings(tmp_path, connectome_source="neuprint")
    k0 = cache_key(s)
    d = tmp_path / "data" / "connectome" / "neuprint"
    d.mkdir(parents=True)
    (d / "neurons.csv").write_text("bodyId,type\n1,DNp01\n", encoding="utf-8")
    assert cache_key(s) != k0


# --------------------------------------------------------------------------- resolve_connectome_dir


def test_resolve_connectome_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    absolute = tmp_path / "abs"
    assert resolve_connectome_dir(make_settings(tmp_path, connectome_source="csv", connectome_dir=absolute)) == absolute
    s = make_settings(tmp_path, connectome_source="neuprint", connectome_dir=absolute)
    assert resolve_connectome_dir(s) == data_dir / "connectome" / "neuprint"
    (tmp_path / "rel").mkdir()
    s = make_settings(tmp_path, connectome_source="csv", connectome_dir=Path("rel"))
    assert resolve_connectome_dir(s) == tmp_path / "rel"  # repo root = parent of data_dir
    (data_dir / "under_data").mkdir()
    s = make_settings(tmp_path, connectome_source="csv", connectome_dir=Path("under_data"))
    assert resolve_connectome_dir(s) == data_dir / "under_data"
    s = make_settings(tmp_path, connectome_source="csv", connectome_dir=Path("missing/dir"))
    assert resolve_connectome_dir(s) == tmp_path / "missing" / "dir"


# --------------------------------------------------------------------------- store / load


def _assert_same(a: Connectome, b: Connectome) -> None:
    assert a.name == b.name and a.source == b.source and a.n == b.n and a.e == b.e
    for name in ("pre", "post", "weight", "sign", "region", "side", "type_idx", "body_id"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype, name
        assert np.array_equal(x, y), name
    assert [str(v) for v in a.nt] == [str(v) for v in b.nt]
    assert a.types == b.types
    assert set(a.groups) == set(b.groups)
    for k in a.groups:
        assert np.array_equal(a.groups[k], b.groups[k]) and b.groups[k].dtype == np.int32


def test_store_and_load_round_trip(tiny_connectome, tmp_data_dir):
    key = "abc123def456"
    path = store_cached(key, tiny_connectome, tmp_data_dir / "cache")
    assert path == tmp_data_dir / "cache" / f"{key}.npz" and path.is_file()
    assert (tmp_data_dir / "cache" / f"{key}.json").is_file()
    assert tiny_connectome.meta["cache_key"] == key and "cached" in tiny_connectome.meta
    assert not list((tmp_data_dir / "cache").glob("*-tmp*"))
    loaded = load_cached(key, tmp_data_dir / "cache")
    assert loaded is not None
    _assert_same(tiny_connectome, loaded)
    assert loaded.meta["cache_key"] == key
    assert loaded.meta["region_counts"] == tiny_connectome.meta["region_counts"]
    assert loaded.meta["e"] == loaded.e
    loaded.validate()
    doc = json.loads((tmp_data_dir / "cache" / f"{key}.json").read_text(encoding="utf-8"))
    assert doc["n"] == tiny_connectome.n and doc["meta"]["cache_key"] == key


def test_store_creates_cache_dir_and_overwrites(tiny_connectome, tmp_path):
    cache_dir = tmp_path / "new" / "cache"
    store_cached("k1", tiny_connectome, cache_dir)
    store_cached("k1", tiny_connectome, cache_dir)  # replace in place
    assert load_cached("k1", cache_dir) is not None
    assert sorted(p.name for p in cache_dir.iterdir()) == ["k1.json", "k1.npz"]


def test_load_mmap(tiny_connectome, tmp_data_dir):
    cache_dir = tmp_data_dir / "cache"
    store_cached("mm", tiny_connectome, cache_dir)
    loaded = load_cached("mm", cache_dir, mmap=True)
    assert loaded is not None and cached_arrays_are_memmaps(loaded)
    assert isinstance(loaded.weight, np.memmap) and isinstance(loaded.body_id, np.memmap)
    _assert_same(tiny_connectome, loaded)
    with pytest.raises(ValueError):
        loaded.weight[0] = 1.0  # read-only mapping
    csr = loaded.csr()
    assert csr.e == loaded.e
    del loaded, csr
    gc.collect()


def test_load_missing_returns_none(tmp_data_dir):
    assert load_cached("000000000000", tmp_data_dir / "cache") is None
    assert load_cached("000000000000", tmp_data_dir / "does-not-exist") is None


@pytest.mark.parametrize("bad", ["", "a.b", "a/b", "..", "x" * 65, "a b"])
def test_bad_keys_raise(tmp_data_dir, bad):
    with pytest.raises(ValueError):
        load_cached(bad, tmp_data_dir / "cache")


def test_corrupt_entries_return_none(tiny_connectome, tmp_data_dir, caplog):
    cache_dir = tmp_data_dir / "cache"
    store_cached("good", tiny_connectome, cache_dir)
    (cache_dir / "bad.npz").write_bytes(b"not a zip archive")
    (cache_dir / "bad.json").write_text("{}", encoding="utf-8")
    assert load_cached("bad", cache_dir) is None
    (cache_dir / "half.npz").write_bytes((cache_dir / "good.npz").read_bytes())  # json sidecar missing
    assert load_cached("half", cache_dir) is None
    doc = json.loads((cache_dir / "good.json").read_text(encoding="utf-8"))
    doc["meta"]["e"] = 999_999
    (cache_dir / "tampered.npz").write_bytes((cache_dir / "good.npz").read_bytes())
    (cache_dir / "tampered.json").write_text(json.dumps(doc), encoding="utf-8")
    assert load_cached("tampered", cache_dir) is None
    assert load_cached("good", cache_dir) is not None
    assert all(ord(ch) < 128 for rec in caplog.records for ch in rec.getMessage())


def test_unreadable_entry_log_is_ascii_for_localized_oserror(tiny_connectome, tmp_data_dir, monkeypatch, caplog):
    # regression: a localized Windows OSError text (cp1254 letters) reached the log verbatim
    cache_dir = tmp_data_dir / "cache"
    store_cached("loc", tiny_connectome, cache_dir)

    def boom(cls, stem, mmap=False):
        raise PermissionError("[WinError 5] Eri\u015fim engellendi: 'x\u00e7'")  # localized Windows text

    monkeypatch.setattr(Connectome, "load", classmethod(boom))
    with caplog.at_level(logging.WARNING, logger="flybrain.connectome.cache"):
        assert load_cached("loc", cache_dir) is None
    msgs = [rec.getMessage() for rec in caplog.records if "unreadable" in rec.getMessage()]
    assert len(msgs) == 1 and "PermissionError" in msgs[0] and "\\u015f" in msgs[0]
    assert all(ord(ch) < 128 for ch in msgs[0])
    assert ascii_safe("plain") == "plain" and ascii_safe(Path("\u00e7")) == "\\xe7"


def test_failed_store_leaves_no_temp_files(tiny_connectome, tmp_data_dir, monkeypatch):
    cache_dir = tmp_data_dir / "cache"

    def boom(self, stem):
        Path(stem).with_suffix(".npz").write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(Connectome, "save", boom)
    with pytest.raises(OSError):
        store_cached("fail", tiny_connectome, cache_dir)
    assert list(cache_dir.iterdir()) == []
