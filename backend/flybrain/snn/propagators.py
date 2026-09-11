"""Spike propagation backends over the outgoing CSR (SPEC section c.11).

A ``Propagator`` scatters ``data_mv[edge]`` into ``out[post]`` for every outgoing edge of every
spiking neuron and returns the number of edge visits. Two backends:

* ``NumpyPropagator`` - pure numpy. Small spike sets use python slices + ``np.concatenate``;
  larger ones use a vectorised flat-slot gather (segment index built with a fill/cumsum trick,
  int64 fancy indexing) and ``np.add.at`` for the scatter. Measured on this CPU at N=20k/E=2M
  (numpy 2.4, 50k visits): ``np.add.at`` 105 us vs ``np.bincount(weights=...)`` 350 us; int64
  index arrays gather 3x faster than int32 ones; the fill/cumsum segment index is 10x faster than
  ``np.repeat + arange``; the python-slice path only wins below ~8 spikes (12 vs 16 us at 5 spikes,
  55 vs 35 us at 50, 559 vs 279 us at 600). The default ``concat_threshold`` is SPEC c.11's 4000,
  so real traffic (1-3 % active of 20k = 200-600 spikes/step) takes the python-slice branch; pass a
  small threshold (e.g. 16) to force the vectorised branch, which is ~2x cheaper above ~8 spikes.
  Both branches handle rows with out-degree 0 (motor neurons, sensory terminals, any ``min_weight``
  pruning): the vectorised one drops the empty rows before building the segment index.
* ``TorchPropagator`` - torch (imported inside the constructor only), ``repeat_interleave`` gather
  and ``index_add_`` scatter; used for E > 5,000,000 or when requested explicitly.

No biological numbers live in this module.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol, runtime_checkable

import numpy as np

from ..connectome.schema import CSR

__all__ = ["Propagator", "NumpyPropagator", "TorchPropagator", "make_propagator", "AUTO_TORCH_EDGES"]

log = logging.getLogger("flybrain.snn.propagators")

#: ``backend='auto'`` picks torch above this many edges (SPEC sections 0 / b).
AUTO_TORCH_EDGES: int = 5_000_000


@runtime_checkable
class Propagator(Protocol):
    """Edge-visit scatter: ``out[post] += data_mv[edge]`` for every outgoing edge of every index in ``spk``."""

    def __call__(self, spk: np.ndarray, out: np.ndarray) -> int:
        """``spk`` int32 (sorted), ``out`` float32[n] accumulated in place. Returns edge visits."""
        ...

    def set_data(self, data_mv: np.ndarray) -> None:
        """Replace the per-edge weights (float32[E], CSR order) after a gain change."""
        ...


def _check_data(csr: CSR, data_mv: np.ndarray) -> np.ndarray:
    data_mv = np.asarray(data_mv)
    if data_mv.dtype != np.float32 or data_mv.ndim != 1 or data_mv.shape[0] != csr.e:
        raise ValueError(f"data_mv must be float32[{csr.e}], got {data_mv.dtype}{data_mv.shape}")
    return data_mv


class NumpyPropagator:
    """Numpy backend.

    ``len(spk) <= concat_threshold``: python slices + ``np.concatenate`` (cheapest for a few spikes
    because the vectorised path has ~15 us of fixed numpy-call overhead). Otherwise: vectorised
    flat-slot gather ``ln = b - a; seg = cumsum(ln) - ln; idx = repeat(a - seg, ln) + arange(ln.sum())``
    implemented as an in-place fill/cumsum over a reusable int64 buffer (the ``repeat`` form is
    10x slower), then ``np.add.at(out, tg, wv)``. Spiking neurons with out-degree 0 are removed from
    ``a``/``ln`` before the segment index is built (a zero-length run would otherwise collide with the
    next row's boundary and shift every later target, or write past the end of the buffer); the
    returned visit count is the unfiltered ``ln.sum()``. Measured crossover on this CPU is ~8 spikes,
    well below the SPEC c.11 default of 4000 kept here.
    """

    name = "numpy"

    def __init__(self, csr: CSR, data_mv: np.ndarray, concat_threshold: int = 4000) -> None:
        self.csr = csr
        self.n = csr.n
        self.indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
        self.data_mv = _check_data(csr, data_mv)
        self.concat_threshold = int(concat_threshold)
        self._idx_buf = np.empty(max(1024, min(csr.e, 1 << 17)), dtype=np.int64)
        self.visits = 0  # cumulative edge visits (diagnostics)

    def set_data(self, data_mv: np.ndarray) -> None:
        self.data_mv = _check_data(self.csr, data_mv)

    def __call__(self, spk: np.ndarray, out: np.ndarray) -> int:
        k = int(spk.shape[0])
        if k == 0:
            return 0
        a = self.indptr[spk]
        b = self.indptr[spk + 1]
        ln = b - a
        tot = int(ln.sum())
        if tot == 0:
            return 0
        if k <= self.concat_threshold:
            al = a.tolist()
            bl = b.tolist()
            ind = self.indices
            dat = self.data_mv
            if k == 1:
                tg = ind[al[0]:bl[0]]
                wv = dat[al[0]:bl[0]]
            else:
                tg = np.concatenate([ind[x:y] for x, y in zip(al, bl)])
                wv = np.concatenate([dat[x:y] for x, y in zip(al, bl)])
        else:
            if tot > self._idx_buf.shape[0]:
                self._idx_buf = np.empty(int(tot * 1.5) + 1024, dtype=np.int64)
            if int(ln.min()) == 0:
                # drop out-degree-0 rows: a zero-length run would put two boundaries on the same slot
                nz = ln > 0
                a = a[nz]
                b = b[nz]
                ln = ln[nz]
            idx = self._idx_buf[:tot]
            idx.fill(1)
            idx[0] = a[0]
            # at every segment boundary jump from the end of one row to the start of the next
            ends = np.cumsum(ln)
            idx[ends[:-1]] = a[1:] - b[:-1] + 1
            np.cumsum(idx, out=idx)
            tg = self.indices[idx]
            wv = self.data_mv[idx]
        np.add.at(out, tg, wv)
        self.visits += tot
        return tot


class TorchPropagator:
    """Torch backend (CPU by default). ``torch`` is imported here, never at module import time.

    ``indices``/``data`` live as tensors (indices int64); gather via ``repeat_interleave``, scatter
    via ``out_t.index_add_(0, tg, wv)`` on a tensor that shares memory with ``out`` (``torch.from_numpy``)
    so no copy back is needed; a non-contiguous ``out`` falls back to an explicit copy.
    ``torch.set_num_threads(min(8, os.cpu_count() - 2))`` unless ``threads`` is given;
    ``deterministic=True`` -> ``torch.use_deterministic_algorithms(True)`` (~15 % slower).
    """

    name = "torch"

    def __init__(
        self,
        csr: CSR,
        data_mv: np.ndarray,
        threads: int | None = None,
        device: str = "cpu",
        deterministic: bool = True,
    ) -> None:
        import torch  # optional dependency, imported here only

        self.torch = torch
        self.csr = csr
        self.n = csr.n
        self.device = torch.device(device)
        if threads is None:
            threads = max(1, min(8, (os.cpu_count() or 4) - 2))
        try:
            torch.set_num_threads(int(threads))
        except Exception:  # pragma: no cover - some builds forbid changing threads after start
            pass
        self.threads = int(threads)
        if deterministic:
            torch.use_deterministic_algorithms(True)
        self.deterministic = bool(deterministic)
        self.indptr_t = torch.from_numpy(np.ascontiguousarray(csr.indptr, dtype=np.int64)).to(self.device)
        self.indices_t = torch.from_numpy(np.ascontiguousarray(csr.indices, dtype=np.int64)).to(self.device)
        self.data_mv = _check_data(csr, data_mv)
        self.data_t = torch.from_numpy(np.ascontiguousarray(self.data_mv)).to(self.device)
        self.visits = 0

    def set_data(self, data_mv: np.ndarray) -> None:
        self.data_mv = _check_data(self.csr, data_mv)
        self.data_t = self.torch.from_numpy(np.ascontiguousarray(self.data_mv)).to(self.device)

    def __call__(self, spk: np.ndarray, out: np.ndarray) -> int:
        k = int(spk.shape[0])
        if k == 0:
            return 0
        torch = self.torch
        spk_t = torch.from_numpy(np.ascontiguousarray(spk, dtype=np.int64)).to(self.device)
        a = self.indptr_t[spk_t]
        ln = self.indptr_t[spk_t + 1] - a
        tot = int(ln.sum().item())
        if tot == 0:
            return 0
        seg = torch.cumsum(ln, 0) - ln
        idx = torch.repeat_interleave(a - seg, ln) + torch.arange(tot, dtype=torch.int64, device=self.device)
        tg = self.indices_t[idx]
        wv = self.data_t[idx]
        if self.device.type == "cpu" and out.flags.c_contiguous and out.dtype == np.float32:
            out_t = torch.from_numpy(out)  # shares memory: index_add_ writes straight into out
            out_t.index_add_(0, tg, wv)
        else:
            out_t = torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32)).to(self.device)
            out_t.index_add_(0, tg, wv)
            out[...] = out_t.cpu().numpy()
        self.visits += tot
        return tot


def torch_available() -> bool:
    """True when ``import torch`` succeeds (import cost ~5 s on this machine, cached afterwards)."""
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def make_propagator(backend: str, csr: CSR, data_mv: np.ndarray) -> Propagator:
    """``'numpy' | 'torch' | 'auto'`` (torch if ``csr.e > 5_000_000`` and importable, else numpy; logs the choice).

    ``'torch'`` raises ``ImportError`` when torch is missing (the caller asked for it explicitly).
    """
    backend = (backend or "numpy").lower()
    if backend == "auto":
        if csr.e > AUTO_TORCH_EDGES and torch_available():
            log.info("propagator backend: torch (auto, e=%d > %d)", csr.e, AUTO_TORCH_EDGES)
            return TorchPropagator(csr, data_mv)
        log.info("propagator backend: numpy (auto, e=%d)", csr.e)
        return NumpyPropagator(csr, data_mv)
    if backend == "numpy":
        log.info("propagator backend: numpy (e=%d)", csr.e)
        return NumpyPropagator(csr, data_mv)
    if backend == "torch":
        log.info("propagator backend: torch (e=%d)", csr.e)
        return TorchPropagator(csr, data_mv)
    raise ValueError(f"unknown backend {backend!r} (expected numpy | torch | auto)")
