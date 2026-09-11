"""Propagator / backend tests (SPEC sections c.11, h.2).

The numpy tests always run; every torch test calls ``pytest.importorskip("torch")`` itself so the
file is collected without the optional dependency.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from flybrain.snn import propagators
from flybrain.snn.engine import LIFEngine
from flybrain.snn.monitor import SpikeMonitor
from flybrain.snn.propagators import NumpyPropagator, Propagator, make_propagator, torch_available
from tests.test_engine import quiet_engine, random_connectome, small_connectome


def _data(conn):
    csr = conn.csr()
    return csr, (csr.data * np.float32(0.275)).astype(np.float32)


def _bruteforce(csr, data, spk, n):
    out = np.zeros(n, dtype=np.float64)
    visits = 0
    for i in spk.tolist():
        for e in range(int(csr.indptr[i]), int(csr.indptr[i + 1])):
            out[int(csr.indices[e])] += float(data[e])
            visits += 1
    return out, visits


# --------------------------------------------------------------------------- numpy


@pytest.mark.parametrize("k", [0, 1, 7, 16, 17, 40, 150])
def test_numpy_propagator_bruteforce(k):
    """Random spikes -> ``out`` equals a Python loop over the CSR rows (both gather branches)."""
    n = 200
    conn = random_connectome(n, 3000, seed=k + 3)
    csr, data = _data(conn)
    rng = np.random.default_rng(k)
    spk = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int32)
    expect, visits = _bruteforce(csr, data, spk, n)
    prop = NumpyPropagator(csr, data)
    assert isinstance(prop, Propagator)
    out = np.zeros(n, dtype=np.float32)
    assert prop(spk, out) == visits
    np.testing.assert_allclose(out, expect.astype(np.float32), rtol=1e-5, atol=1e-5)
    # accumulates on top of existing values
    out2 = np.ones(n, dtype=np.float32)
    prop(spk, out2)
    np.testing.assert_allclose(out2, 1.0 + expect.astype(np.float32), rtol=1e-5, atol=1e-5)
    assert prop.visits == 2 * visits


def test_concat_vs_vectorised_gather():
    """Both branches of ``NumpyPropagator`` produce identical results (bit for bit)."""
    n = 300
    conn = random_connectome(n, 6000, seed=8)
    csr, data = _data(conn)
    concat = NumpyPropagator(csr, data, concat_threshold=10**6)
    vector = NumpyPropagator(csr, data, concat_threshold=0)
    rng = np.random.default_rng(1)
    for k in (1, 2, 5, 33, 100, 300):
        spk = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int32)
        o1 = np.zeros(n, dtype=np.float32)
        o2 = np.zeros(n, dtype=np.float32)
        assert concat(spk, o1) == vector(spk, o2)
        assert np.array_equal(o1, o2)
    # the growable index buffer survives a burst larger than its initial size
    small = NumpyPropagator(csr, data, concat_threshold=0)
    small._idx_buf = np.empty(8, dtype=np.int64)
    spk = np.arange(n, dtype=np.int32)
    o3 = np.zeros(n, dtype=np.float32)
    o4 = np.zeros(n, dtype=np.float32)
    small(spk, o3)
    vector(spk, o4)
    assert np.array_equal(o3, o4)
    with pytest.raises(ValueError):
        NumpyPropagator(csr, data.astype(np.float64))
    with pytest.raises(ValueError):
        vector.set_data(np.zeros(3, dtype=np.float32))


@pytest.mark.parametrize("k", [0, 1, 5, 16, 17, 50])
def test_event_propagation_equals_dense_matmul(k):
    """Event-driven scatter == dense (spk one-hot) @ W on a 50-neuron random graph (float32 tolerance)."""
    n = 50
    conn = random_connectome(n, 600, seed=k + 1)
    csr, data = _data(conn)
    dense = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a, b = csr.indptr[i], csr.indptr[i + 1]
        dense[i, csr.indices[a:b]] += data[a:b]
    rng = np.random.default_rng(k)
    spk = np.sort(rng.choice(n, size=k, replace=False)).astype(np.int32)
    prop = NumpyPropagator(csr, data)
    out = np.zeros(n, dtype=np.float32)
    visits = prop(spk, out)
    onehot = np.zeros(n)
    onehot[spk] = 1.0
    expect = onehot @ dense
    assert visits == (int((csr.indptr[spk + 1] - csr.indptr[spk]).sum()) if k else 0)
    np.testing.assert_allclose(out, expect.astype(np.float32), rtol=1e-5, atol=1e-5)
    # engine-level: propagate() delegates and accumulates (call twice -> doubled)
    eng = quiet_engine(conn)
    out2 = np.zeros(n, dtype=np.float32)
    eng.propagate(spk, out2)
    eng.propagate(spk, out2)
    np.testing.assert_allclose(out2, 2 * expect.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_event_driven_equals_dense():
    """Tiny net, 200 steps: the engine's spike trains are identical to a dense ``W @ spikes`` reference
    implementation of the 7-substep loop written here (no noise, 4 random kicks per step)."""
    n = 50
    conn = random_connectome(n, 500, seed=11)
    eng = quiet_engine(conn, gain=1.0, monitor=SpikeMonitor(conn, per_region=64, cap=10_000, seed=0))
    slot = eng.monitor.slot_of_neuron
    assert np.all(slot >= 0)
    c = eng.constants
    csr = conn.csr()
    W = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        a, b = csr.indptr[i], csr.indptr[i + 1]
        W[i, csr.indices[a:b]] = csr.data[a:b] * np.float32(0.275)
    rng = np.random.default_rng(3)
    v = np.full(n, -52.0, np.float32)
    g = np.zeros(n, np.float32)
    ref = np.zeros(n, np.int16)
    D = c.delay_steps
    ring = np.zeros((D + 1, n), np.float32)
    total_ref = 0
    for t in range(200):
        kicked = np.sort(rng.choice(n, size=4, replace=False))
        eng.kick(kicked)
        s = eng.step(1)
        # reference
        rslot = t % (D + 1)
        g_in = ring[rslot].copy()
        active = ref == 0
        v_new = np.float32(-52.0) + (v - np.float32(-52.0)) * np.float32(c.a_m) + np.float32(c.b) * g
        v = np.where(active, v_new, v).astype(np.float32)
        g = np.where(active, g * np.float32(c.a_s), g).astype(np.float32)
        ref = np.where(active, ref, ref - 1).astype(np.int16)
        fired = active & (v > -45.0)
        spk = np.flatnonzero(fired)
        g = g + g_in
        g[kicked] += np.float32(68.75)
        v[spk] = -52.0
        g[spk] = 0.0
        ref[spk] = c.ref_steps
        ring[rslot] = 0.0
        ring[(t + D) % (D + 1)] += W[spk].sum(axis=0, dtype=np.float32)
        total_ref += spk.shape[0]
        assert s.total_spikes == spk.shape[0], f"step {t}"
        assert set(s.spike_indices_sample.tolist()) == set(slot[spk].tolist()), f"step {t}"
        np.testing.assert_allclose(eng.v, v, atol=1e-3)
        assert np.array_equal(eng.ref, ref)
    assert total_ref > 100


def _sink_connectome(n: int, sinks: set[int], seed: int = 0):
    """Ring-ish graph in which every neuron of ``sinks`` has out-degree 0 (motor neurons / sensory
    terminals / ``min_weight`` pruning all produce such rows in real data)."""
    edges = []
    for i in range(n):
        if i in sinks:
            continue
        for d in (1, 2, 3):
            edges.append((i, (i + d) % n, float(1 + (i + d) % 5)))
    return small_connectome(n, edges, seed=seed)


@pytest.mark.parametrize("sinks", [{0}, {7}, {59}, {0, 59}, {0, 1, 2}, {20, 58, 59}, set(range(0, 60, 2))])
def test_propagator_zero_outdegree_rows(sinks):
    """Regression: spiking neurons with out-degree 0 must not shift or drop any downstream target.

    The vectorised gather builds one segment per spiking row; a zero-length row used to collide with
    the next row's boundary (silently misrouting every later edge) or write past the buffer end when it
    was the last spike (IndexError). Both numpy branches and torch must equal the python-loop reference.
    """
    n = 60
    conn = _sink_connectome(n, sinks)
    csr, data = _data(conn)
    assert int((np.diff(np.asarray(csr.indptr)) == 0).sum()) == len(sinks)
    spk = np.arange(n, dtype=np.int32)            # every neuron spikes: sinks first, middle and last
    expect, visits = _bruteforce(csr, data, spk, n)
    vector = NumpyPropagator(csr, data, concat_threshold=0)      # forced vectorised gather
    concat = NumpyPropagator(csr, data, concat_threshold=10**9)  # forced python-slice gather
    o_vec = np.zeros(n, dtype=np.float32)
    o_cat = np.zeros(n, dtype=np.float32)
    assert vector(spk, o_vec) == visits
    assert concat(spk, o_cat) == visits
    assert np.array_equal(o_vec, o_cat)
    np.testing.assert_allclose(o_vec, expect.astype(np.float32), rtol=1e-6, atol=1e-6)
    # only the sinks spike -> nothing is delivered, no crash
    if sinks:
        only = np.array(sorted(sinks), dtype=np.int32)
        o = np.zeros(n, dtype=np.float32)
        assert vector(only, o) == 0 and concat(only, o) == 0
        assert not o.any()
    if torch_available():
        import torch  # noqa: F401

        o_t = np.zeros(n, dtype=np.float32)
        assert propagators.TorchPropagator(csr, data)(spk, o_t) == visits
        np.testing.assert_allclose(o_t, o_vec, rtol=1e-6, atol=1e-6)


def test_engine_runs_on_connectome_with_sinks():
    """Engine-level regression for the same defect: 600 steps on a graph with sink rows, both gather
    branches, identical spike counts (the default threshold used to take the broken branch every step)."""
    conn = _sink_connectome(200, {3, 101, 199}, seed=4)
    counts = []
    for threshold in (4000, 0):
        eng = LIFEngine(conn, seed=9, dt_ms=1.0)
        eng.prop.concat_threshold = threshold
        eng.inject(np.arange(conn.n, dtype=np.int32), rate_hz=60.0, duration_ms=1e9, tag="all")
        counts.append([eng.step(50).total_spikes for _ in range(12)])
    assert counts[0] == counts[1] and sum(counts[0]) > 0


def test_numpy_propagator_default_threshold_is_spec_c11():
    """SPEC c.11 signature: ``concat_threshold: int = 4000`` (the branch real traffic takes)."""
    conn = random_connectome(40, 300, seed=2)
    csr, data = _data(conn)
    assert NumpyPropagator(csr, data).concat_threshold == 4000
    assert inspect.signature(NumpyPropagator.__init__).parameters["concat_threshold"].default == 4000


def test_make_propagator_auto(tiny_connectome, monkeypatch):
    csr, data = _data(tiny_connectome)
    assert isinstance(make_propagator("auto", csr, data), NumpyPropagator)      # e << 5M
    assert isinstance(make_propagator("numpy", csr, data), NumpyPropagator)
    assert isinstance(make_propagator("NUMPY", csr, data), NumpyPropagator)
    monkeypatch.setattr(propagators, "torch_available", lambda: False)
    monkeypatch.setattr(propagators, "AUTO_TORCH_EDGES", 0)
    assert isinstance(make_propagator("auto", csr, data), NumpyPropagator)     # torch "missing" -> numpy
    with pytest.raises(ValueError):
        make_propagator("cuda", csr, data)
    monkeypatch.undo()
    if torch_available():
        monkeypatch.setattr(propagators, "AUTO_TORCH_EDGES", 0)
        assert isinstance(make_propagator("auto", csr, data), propagators.TorchPropagator)
        assert isinstance(make_propagator("torch", csr, data), propagators.TorchPropagator)
    else:
        with pytest.raises(ImportError):
            make_propagator("torch", csr, data)


# --------------------------------------------------------------------------- torch (optional)


def test_torch_propagator_matches_numpy(tiny_connectome):
    """Torch scatter == numpy scatter (to 1e-5; on CPU float32 they are bit-identical) for random spike sets."""
    pytest.importorskip("torch", reason="torch not installed (optional backend)")
    csr, data = _data(tiny_connectome)
    npp = NumpyPropagator(csr, data)
    tpp = propagators.TorchPropagator(csr, data)
    assert isinstance(npp, Propagator) and isinstance(tpp, Propagator)
    rng = np.random.default_rng(0)
    for trial in range(60):
        k = int(rng.integers(0, 150))
        spk = np.sort(rng.choice(tiny_connectome.n, size=k, replace=False)).astype(np.int32)
        o1 = np.zeros(tiny_connectome.n, dtype=np.float32)
        o2 = np.zeros(tiny_connectome.n, dtype=np.float32)
        v1 = npp(spk, o1)
        v2 = tpp(spk, o2)
        assert v1 == v2
        np.testing.assert_allclose(o1, o2, rtol=1e-5, atol=1e-5, err_msg=f"trial {trial}")
    # a contiguous row of a 2-D buffer (the ring layout) is written in place; a strided view via the copy path
    big = np.zeros((2, tiny_connectome.n), dtype=np.float32)
    view = big[1]
    spk = np.arange(0, tiny_connectome.n, 7, dtype=np.int32)
    o = np.zeros(tiny_connectome.n, dtype=np.float32)
    npp(spk, o)
    tpp(spk, view)
    np.testing.assert_allclose(view, o, rtol=1e-5, atol=1e-5)
    strided = np.zeros(2 * tiny_connectome.n, dtype=np.float32)[::2]
    tpp(spk, strided)
    np.testing.assert_allclose(strided, o, rtol=1e-5, atol=1e-5)


def test_torch_set_data_and_gain(tiny_connectome):
    pytest.importorskip("torch", reason="torch not installed (optional backend)")
    csr, data = _data(tiny_connectome)
    tpp = propagators.TorchPropagator(csr, data)
    tpp.set_data((data * np.float32(0.5)).astype(np.float32))
    spk = np.array([int(np.flatnonzero(np.diff(csr.indptr) > 0)[0])], dtype=np.int32)
    o = np.zeros(tiny_connectome.n, dtype=np.float32)
    tpp(spk, o)
    a, b = csr.indptr[spk[0]], csr.indptr[spk[0] + 1]
    expect = np.zeros(tiny_connectome.n, dtype=np.float32)
    np.add.at(expect, csr.indices[a:b], data[a:b] * np.float32(0.5))
    np.testing.assert_allclose(o, expect, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError):
        tpp.set_data(np.zeros(3, dtype=np.float32))


def test_torch_engine_equals_numpy_engine_spike_for_spike(tiny_connectome):
    """Torch backend == numpy backend spike for spike on the tiny fixture (same seed, drives, 400 steps)."""
    pytest.importorskip("torch", reason="torch not installed (optional backend)")
    e1 = LIFEngine(tiny_connectome, seed=5, backend="numpy")
    e2 = LIFEngine(tiny_connectome, seed=5, backend="torch")
    assert e1.backend == "numpy" and e2.backend == "torch"
    for e in (e1, e2):
        e.inject("grn_sugar_labellar", rate_hz=100.0, duration_ms=300.0, tag="s")
        e.inject("lc_loom", rate_hz=80.0, duration_ms=300.0, tag="l")
        e.inject("kc", rate_hz=30.0, duration_ms=300.0, tag="k")
    for _ in range(8):
        s1 = e1.step(50)
        s2 = e2.step(50)
        assert s1.total_spikes == s2.total_spikes
        assert s1.edge_visits == s2.edge_visits
        assert s1.forced_events == s2.forced_events
        assert np.array_equal(s1.spike_indices_sample, s2.spike_indices_sample)
        assert np.array_equal(s1.spike_dt_sample, s2.spike_dt_sample)
        assert s1.spike_counts_by_group == s2.spike_counts_by_group
        np.testing.assert_allclose(e1.v, e2.v, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(e1.ring, e2.ring, rtol=1e-5, atol=1e-5)
    assert s1.total_spikes > 0
    e2.set_gain(0.5)
    np.testing.assert_array_equal(e2.prop.data_mv, e2.data_mv)
