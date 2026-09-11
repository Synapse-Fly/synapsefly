"""LIF engine: Shiu-2024 neurons on the connectome CSR (SPEC section c.10).

Exact 2-D matrix-exponential update with Brian2 "unless refractory" semantics (``v`` and ``g``
frozen while refractory, incoming input still accumulates), reset ``v = v_reset, g = 0``, a ring
buffer of ``delay_steps + 1`` rows for the synaptic delay, Poisson forcing kicks of
``f_poi * w_syn = 68.75 mV`` or constant currents (``current_from_rate``), background noise
``g += N(mu*dt, sigma*sqrt(dt))`` per step and neuron (RESEARCH section 7 ``[V]``:
mu 0.5 mV/ms, sigma 3.5 mV/sqrt(ms) -> ~2.3 Hz isolated rest activity).

Per-step order (binding, see ``LIFEngine.step``):
  1. read the delay ring slot; expire injections
  2. update ``v``/``g`` of active neurons, ``ref -= 1`` for refractory ones (frozen)
  3. threshold -> ``spk``
  4. ``g += ring + noise + forced kicks``; current injections live in ``i_ext``
  5. reset the spiking neurons
  6. zero the read slot; propagate ``spk`` into slot ``(t + D) % (D + 1)``
  7. monitor.record; ``t_step += 1``

Latency contract ``[V]``: an isolated neuron receiving one 68.75 mV kick during step ``k`` (kick time
``t_k = k*dt``) crosses threshold during step ``k + ceil(3.0/dt)`` -> ``3.0 <= t_spike - t_k <= 3.0 + dt``.

Performance notes (numpy backend, this CPU, N=20k/E=2M, 1-3 % active): the update is 6 full-vector
float32 ops (~15 us); propagation 0.1-0.3 ms (event-driven gather + ``np.add.at``); the refractory
set is kept as ``ref_steps`` index buckets (one per remaining step) so no per-step scan of ``ref`` is
needed (save/restore of the frozen values costs ~7 us).

DISCLOSED DEVIATION from SPEC c.10 step 4 (performance, kept deliberately): the per-step noise is not
``rng.normal(mu*dt, sigma*sqrt(dt), n)`` but a random-offset slice of a seeded pool of ``NOISE_POOL``
float32 standard normals (4 MB) scaled to ``mu*dt + sigma*sqrt(dt)*z``, with ``NOISE_REFRESH`` pool
entries re-drawn per step (the pool is fully renewed every 512 steps). Measured on this CPU at
n = 20k: 5.4 us per step versus 148.7 us for ``standard_normal(n, out=...)`` and 232.9 us for
``rng.normal``, i.e. 0.3 ms versus 7.4 ms of the 50 ms budget that SPEC section 0 gives a 50-step
tick - at 3 % active the literal formula would leave almost no margin. The samples stay N(0, 1) and
independent across neurons within a step (a pool entry is re-used by another neuron at another step
at most every ``NOISE_POOL / n`` steps); verified: increments over 300 steps at n = 20k have mean
0.5047 / std 3.4965 (targets 0.5 / 3.5), lag-1 correlation between consecutive steps <= 0.019, and
the isolated rest rate is 2.35-2.41 Hz at n = 400..100000 and dt 1.0 / 0.5 (RESEARCH section 7: ~2.3 Hz).
``sigma <= 0`` disables the whole term (mean included), exactly as c.10 writes.

All per-step buffers are preallocated; the only per-step allocations are the small spike arrays.

Provenance: LIF constants ``[L]`` (params.py), noise defaults ``[V]``, tonic table ``[E]``,
homeostasis rule ``[E]`` (SPEC section 0). Only numpy is imported at module level.
"""

from __future__ import annotations

import copy
import logging
import math
import time
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .monitor import SpikeMonitor, StepStats
from .params import LIFParams, StepConstants, current_from_rate, rate_from_current
from .propagators import Propagator, make_propagator

if TYPE_CHECKING:  # pragma: no cover
    from ..connectome.schema import Connectome

__all__ = [
    "Drive",
    "Injection",
    "StepStats",
    "LIFEngine",
    "TONIC_TABLE_MV",
    "apply_tonic_table",
    "HOMEOSTASIS_ACTIVE_FRAC",
    "HOMEOSTASIS_STEPS",
    "HOMEOSTASIS_FACTOR",
    "HOMEOSTASIS_FLOOR",
    "NOISE_POOL",
    "NOISE_REFRESH",
]

log = logging.getLogger("flybrain.snn.engine")

#: Tonic-current table (mV offset of the resting potential) applied by ``SimulationLoop`` at start, all ``[E]``:
#: lamina / motion_in 7.3 mV (~15 Hz baseline so histaminergic photoreceptor input can modulate them),
#: feed_pre_inh 7.05 mV (~10 Hz tonic inhibition of MN9, released by GNG042->GNG015), mal 7.05 mV (tonic
#: inhibition of pC1). 7.05 / 7.3 mV = ``current_from_rate`` of 10 / ~15 Hz (RESEARCH section 7 ``[V]``).
TONIC_TABLE_MV: dict[str, float] = {"lamina": 7.3, "motion_in": 7.3, "feed_pre_inh": 7.05, "mal": 7.05}

#: Homeostasis rule ``[E]`` (SPEC section 0): active fraction > 5 % for 10 consecutive steps -> gain *= 0.9 (floor 0.3).
HOMEOSTASIS_ACTIVE_FRAC: float = 0.05
HOMEOSTASIS_STEPS: int = 10
HOMEOSTASIS_FACTOR: float = 0.9
HOMEOSTASIS_FLOOR: float = 0.3

_SEED_CHILD_ENGINE = 1   # SPEC section 0.1 spawn order: [1] engine (Poisson + noise)
_SEED_CHILD_RASTER = 4   # [4] raster sampling (default monitor only)

#: Gaussian noise pool: ``NOISE_POOL`` float32 standard normals (grown to >= 4 n), ``NOISE_REFRESH``
#: entries re-drawn per step (the pool is fully renewed every NOISE_POOL / NOISE_REFRESH = 512 steps).
NOISE_POOL: int = 1 << 20
NOISE_REFRESH: int = 2048


@dataclass(frozen=True)
class Drive:
    """A sensory drive request (SPEC section c.10).

    ``group``: key of ``Connectome.groups`` (sided keys allowed); ``rate_hz``: Poisson rate per recruited
    neuron (or the rate converted by ``current_from_rate`` in current mode); ``recruit``: fraction of the
    group driven; ``side``: 0 both / -1 L / +1 R (filters by ``Connectome.side``); ``episode``: deterministic
    recruitment permutation id (rng seeded from (seed, group, episode)); ``mode``: ``poisson|current``;
    ``weights``: optional per-neuron multiplier of ``rate_hz`` (len == group size, or == recruited size).
    """

    group: str
    rate_hz: float
    recruit: float = 1.0
    side: int = 0
    episode: int = 0
    mode: str = "poisson"
    weights: np.ndarray | None = None


@dataclass(frozen=True)
class Injection:
    """A ``Drive`` with an expiry, produced by ``inject()``; cleared automatically when ``t_ms >= until_ms``."""

    drive: Drive
    until_ms: int
    tag: str


class _Rec:
    """Internal active-injection record (resolved indices + per-step probabilities / currents)."""

    __slots__ = ("tag", "injection", "kind", "idx", "p", "cur", "until_f")

    def __init__(self, tag: str, injection: Injection, kind: str, idx: np.ndarray,
                 p: np.ndarray | float | None, cur: np.ndarray | float | None, until_f: float) -> None:
        self.tag = tag
        self.injection = injection
        self.kind = kind
        self.idx = idx
        self.p = p
        self.cur = cur
        self.until_f = until_f


def _group_hash(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _current_from_rate_vec(rate_hz: np.ndarray, p: LIFParams) -> np.ndarray:
    """Vectorised ``current_from_rate`` (float64 in, float64 out; rate <= 0 -> 0)."""
    r = np.asarray(rate_hz, dtype=np.float64)
    out = np.zeros_like(r)
    pos = r > 0
    if pos.any():
        isi = np.maximum(1000.0 / r[pos] - p.t_ref, 1e-9)
        out[pos] = (p.v_th - p.v_rest) / (1.0 - np.exp(-isi / p.tau_m))
    return out


class LIFEngine:
    """Shiu-2024 LIF network on a ``Connectome`` (SPEC section c.10).

    State: ``v`` float32[n] (= v_rest), ``g`` float32[n] (= 0), ``ref`` int16[n] (= 0), ``i_ext`` float32[n]
    (= 0), ``tonic`` float32[n] (constant currents, ``TONIC_TABLE_MV``), ``ring`` float32[delay_steps+1, n],
    ``t_step`` int. ``data_mv = csr.data * w_syn * gain`` (float32, cached; recomputed by ``set_gain``).
    ``rng`` = Generator from ``SeedSequence(seed)`` child [1]. ``backend 'auto'`` -> torch if
    ``csr.e > 5_000_000`` and torch imports, else numpy.
    """

    def __init__(
        self,
        connectome: "Connectome",
        params: LIFParams = LIFParams(),
        seed: int = 0,
        backend: str = "numpy",
        dt_ms: float = 1.0,
        gain: float = 1.0,
        noise_mu: float = 0.5,
        noise_sigma: float = 3.5,
        drive_mode: str = "poisson",
        monitor: "SpikeMonitor | None" = None,
    ) -> None:
        if drive_mode not in ("poisson", "current"):
            raise ValueError(f"drive_mode must be poisson|current, got {drive_mode!r}")
        self.conn = connectome
        self.params = params
        self.seed = int(seed)
        self.dt_ms = float(dt_ms)
        self.c: StepConstants = params.constants(self.dt_ms)
        self.drive_mode = drive_mode
        self.csr = connectome.csr()
        n = int(connectome.n)
        self._n = n
        self.gain = float(gain)

        # synaptic weights in mV (signed) and the propagator
        self._data_mv = np.empty(self.csr.e, dtype=np.float32)
        self._compute_data_mv()
        self.prop: Propagator = make_propagator(backend, self.csr, self._data_mv)
        self.backend = getattr(self.prop, "name", backend)

        # state -----------------------------------------------------------------------
        p = params
        self.v = np.full(n, p.v_rest, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.ref = np.zeros(n, dtype=np.int16)
        self.i_ext = np.zeros(n, dtype=np.float32)
        self.tonic = np.zeros(n, dtype=np.float32)
        self.D = int(self.c.delay_steps)
        self.ring = np.zeros((self.D + 1, n), dtype=np.float32)
        self.t_step = 0
        self._v_eq = np.full(n, p.v_rest, dtype=np.float32)   # v_rest + i_ext + tonic
        self._tmp = np.empty(n, dtype=np.float32)
        self._noise = np.empty(n, dtype=np.float32)
        self._fired = np.empty(n, dtype=bool)
        self._a_m = np.float32(self.c.a_m)
        self._a_s = np.float32(self.c.a_s)
        self._b = np.float32(self.c.b)
        self._v_th = np.float32(p.v_th)
        self._v_reset = np.float32(p.v_reset)
        self._kick = np.float32(self.c.kick_mv)
        self._ref_steps = np.int16(self.c.ref_steps)
        self._empty32 = np.zeros(0, dtype=np.int32)
        # refractory buckets: bucket j holds the neurons with ref == j + 1 (rotated every step)
        self._ref_buckets: list[np.ndarray] = [self._empty32] * int(self.c.ref_steps)

        # randomness ------------------------------------------------------------------
        pool = NOISE_POOL
        while pool < 4 * n:
            pool *= 2
        self._pool_size = pool
        self._pool = np.empty(pool, dtype=np.float32)
        self._pool_pos = 0
        self._pool_refresh = int(NOISE_REFRESH)
        self.rng = np.random.default_rng()  # replaced by reseed()
        self.reseed(self.seed)
        self.noise_mu = 0.0
        self.noise_sigma = 0.0
        self._mu_dt = np.float32(0.0)
        self._sig_dt = np.float32(0.0)
        self.set_noise(noise_mu, noise_sigma)

        # injections --------------------------------------------------------------------
        self._inj: dict[str, _Rec] = {}
        self._auto_tag = 0
        self._next_expiry = math.inf
        self._pending_kicks: list[tuple[np.ndarray, float]] = []

        # monitor / counters ------------------------------------------------------------
        if monitor is None:
            raster_seed = int(np.random.SeedSequence(self.seed).spawn(8)[_SEED_CHILD_RASTER].generate_state(1)[0])
            monitor = SpikeMonitor(connectome, seed=raster_seed, dt_ms=self.dt_ms)
        self.monitor = monitor
        self.monitor.dt_ms = self.dt_ms
        self.hot_steps = 0
        self._hot_max = 0
        self._hot_thr = HOMEOSTASIS_ACTIVE_FRAC * n
        self.forced_total = 0
        self.visits_total = 0
        self._forced_win = 0
        self._visits_win = 0

    # ------------------------------------------------------------------ properties

    @property
    def n(self) -> int:
        return self._n

    @property
    def t_ms(self) -> int:
        """Brain time in ms (floor of ``t_step * dt``)."""
        return int(math.floor(self.t_step * self.dt_ms + 1e-9))

    @property
    def t_ms_f(self) -> float:
        """Exact brain time in ms (float)."""
        return self.t_step * self.dt_ms

    @property
    def constants(self) -> StepConstants:
        return self.c

    @property
    def data_mv(self) -> np.ndarray:
        """float32[E] signed synaptic weights in mV (``csr.data * w_syn * gain``)."""
        return self._data_mv

    @property
    def injections(self) -> list[Injection]:
        """Active injections (insertion order)."""
        return [r.injection for r in self._inj.values()]

    # ------------------------------------------------------------------ configuration

    def _compute_data_mv(self) -> None:
        np.multiply(self.csr.data, np.float32(self.params.w_syn * self.gain), out=self._data_mv)

    def set_gain(self, gain: float) -> None:
        """Synaptic gain multiplier; recomputes ``data_mv`` and hands it to the propagator."""
        self.gain = float(gain)
        self._compute_data_mv()
        self.prop.set_data(self._data_mv)

    def set_noise(self, mu: float, sigma: float) -> None:
        """Background noise ``g += N(mu*dt, sigma*sqrt(dt))`` per step (mV/ms, mV/sqrt(ms)).

        ``sigma <= 0`` disables the whole term, mean included (SPEC c.10 step 4 gates the draw on
        ``sigma > 0``), so a ``noise_sigma=0`` engine rests exactly at ``v_rest`` and
        ``StepStats.noise_on`` is False. A tonic depolarisation independent of the noise belongs in
        ``set_tonic`` / ``TONIC_TABLE_MV``, not in ``noise_mu``."""
        self.noise_mu = float(mu)
        self.noise_sigma = max(0.0, float(sigma))
        self._mu_dt = np.float32(self.noise_mu * self.dt_ms)
        self._sig_dt = np.float32(self.noise_sigma * math.sqrt(self.dt_ms))

    def set_tonic(self, target: str | np.ndarray, current_mv: float) -> None:
        """Permanent constant current (mV offset of the resting potential), e.g. lamina/medulla 7.3 mV ``[E]``."""
        idx, _ = self._resolve(target, 0)
        if idx.size:
            self.tonic[idx] = np.float32(current_mv)
            self._rebuild_v_eq()

    def _rebuild_v_eq(self) -> None:
        np.add(self.i_ext, self.tonic, out=self._v_eq)
        self._v_eq += np.float32(self.params.v_rest)

    # ------------------------------------------------------------------ injections

    def _resolve(self, target: str | np.ndarray, side: int) -> tuple[np.ndarray, str]:
        if isinstance(target, str):
            if target not in self.conn.groups:
                raise ValueError(f"unknown group {target!r}")
            idx = np.asarray(self.conn.groups[target], dtype=np.int64)
            name = target
        else:
            idx = np.unique(np.asarray(target, dtype=np.int64))
            if idx.size and (idx[0] < 0 or idx[-1] >= self._n):
                raise ValueError(f"target indices out of range [0, {self._n})")
            name = "custom"
        if side:
            idx = idx[self.conn.side[idx] == np.int8(side)]
        return idx.astype(np.int32), name

    def inject(
        self,
        target: str | np.ndarray,
        *,
        rate_hz: float | None = None,
        current_mv: float | None = None,
        duration_ms: float = 50.0,
        recruit: float = 1.0,
        side: int = 0,
        episode: int = 0,
        tag: str = "",
        weights: np.ndarray | None = None,
    ) -> Injection:
        """Group name or explicit int32 indices. Exactly one of ``rate_hz`` / ``current_mv``. Replaces any
        active injection with the same tag (tags are how the encoder updates a channel every tick; an
        empty tag gets a unique auto tag). ``rate_hz`` is applied as Poisson kicks or, when the engine's
        ``drive_mode`` is ``'current'``, as the constant current ``current_from_rate(rate_hz)``.
        ``current_mv`` always sets a constant current. Returns the ``Injection``."""
        if (rate_hz is None) == (current_mv is None):
            raise ValueError("inject(): exactly one of rate_hz / current_mv is required")
        if not (0.0 <= recruit <= 1.0):
            raise ValueError(f"recruit must be in [0, 1], got {recruit}")
        full_idx, name = self._resolve(target, 0)
        group_size = int(full_idx.shape[0])
        w: np.ndarray | None = None
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).ravel()
        idx = full_idx
        if side:
            keep = self.conn.side[idx] == np.int8(side)
            idx = idx[keep]
            if w is not None and w.shape[0] == group_size:
                w = w[keep]
        if recruit < 1.0 and idx.size:
            k = int(math.ceil(recruit * idx.shape[0] - 1e-9))   # ceil(recruit * n_side) per SPEC h.2
            rng = np.random.default_rng([self.seed, _group_hash(name), int(episode)])
            perm = rng.permutation(idx.shape[0])[:k]
            perm.sort()
            if w is not None and w.shape[0] == idx.shape[0]:
                w = w[perm]
            idx = idx[perm]
        if w is not None and w.shape[0] != idx.shape[0]:
            raise ValueError(
                f"weights must have length {group_size} (group size) or {idx.shape[0]} (recruited), got {w.shape[0]}"
            )
        if not tag:
            self._auto_tag += 1
            tag = f"_inj{self._auto_tag}"
        t_now = self.t_ms_f
        until_f = t_now + float(duration_ms)
        until_ms = int(math.ceil(until_f - 1e-9))
        if rate_hz is not None:
            rate = float(rate_hz)
            mode = self.drive_mode
        else:
            rate = rate_from_current(float(current_mv), self.params)
            mode = "current"
        drive = Drive(group=name, rate_hz=rate, recruit=float(recruit), side=int(side), episode=int(episode),
                      mode=mode, weights=None if w is None else w.astype(np.float32))
        inj = Injection(drive=drive, until_ms=until_ms, tag=tag)
        p: np.ndarray | float | None = None
        cur: np.ndarray | float | None = None
        if rate_hz is not None and self.drive_mode == "poisson":
            kind = "poisson"
            base = rate * self.dt_ms / 1000.0
            if w is None:
                p = np.float32(min(1.0, max(0.0, base)))
            else:
                p = np.clip(base * w, 0.0, 1.0).astype(np.float32)
        else:
            kind = "current"
            if rate_hz is not None:
                if w is None:
                    cur = current_from_rate(rate, self.params)
                else:
                    cur = _current_from_rate_vec(rate * w, self.params).astype(np.float32)
            else:
                cur = float(current_mv) if w is None else (float(current_mv) * w).astype(np.float32)
        old = self._inj.get(tag)
        self._inj[tag] = _Rec(tag, inj, kind, idx, p, cur, until_f)
        if kind == "current" or (old is not None and old.kind == "current"):
            self._rebuild_i_ext()
        self._next_expiry = min(self._next_expiry, until_f)
        return inj

    def inject_drive(self, drive: Drive, duration_ms: float = 50.0, tag: str = "") -> Injection:
        """Apply a ``Drive`` object (the encoder's output) for ``duration_ms`` under ``tag``."""
        if drive.mode == "current":
            return self.inject(drive.group, current_mv=current_from_rate(drive.rate_hz, self.params),
                               duration_ms=duration_ms, recruit=drive.recruit, side=drive.side,
                               episode=drive.episode, tag=tag, weights=drive.weights)
        return self.inject(drive.group, rate_hz=drive.rate_hz, duration_ms=duration_ms, recruit=drive.recruit,
                           side=drive.side, episode=drive.episode, tag=tag, weights=drive.weights)

    def clear_injections(self, tag: str | None = None) -> None:
        """Remove one tagged injection (``tag``) or all of them (``None``)."""
        if tag is None:
            had_cur = any(r.kind == "current" for r in self._inj.values())
            self._inj.clear()
        else:
            r = self._inj.pop(tag, None)
            had_cur = r is not None and r.kind == "current"
        if had_cur:
            self._rebuild_i_ext()
        self._next_expiry = min((r.until_f for r in self._inj.values()), default=math.inf)

    def kick(self, target: str | np.ndarray, mv: float | None = None, side: int = 0) -> int:
        """Queue one forced kick (default ``kick_mv`` = 68.75 mV) into ``g`` of the target neurons, applied at
        step 4 of the NEXT step. Returns the number of neurons kicked."""
        idx, _ = self._resolve(target, side)
        if idx.size:
            self._pending_kicks.append((idx, self.c.kick_mv if mv is None else float(mv)))
        return int(idx.shape[0])

    def _rebuild_i_ext(self) -> None:
        self.i_ext.fill(0.0)
        for r in self._inj.values():
            if r.kind == "current" and r.idx.size:
                self.i_ext[r.idx] += np.asarray(r.cur, dtype=np.float32)
        self._rebuild_v_eq()

    def _expire(self, t_now: float) -> None:
        dead = [tag for tag, r in self._inj.items() if r.until_f <= t_now + 1e-9]
        if not dead:
            return
        had_cur = False
        for tag in dead:
            r = self._inj.pop(tag)
            had_cur = had_cur or r.kind == "current"
        if had_cur:
            self._rebuild_i_ext()
        self._next_expiry = min((r.until_f for r in self._inj.values()), default=math.inf)

    # ------------------------------------------------------------------ stepping

    def _rebuild_ref_buckets(self) -> None:
        """Re-derive the refractory index buckets from the ``ref`` array (after reset / load_state_dict)."""
        R = int(self.c.ref_steps)
        if R <= 0:
            self._ref_buckets = []
            self.ref.fill(0)
            return
        np.clip(self.ref, 0, R, out=self.ref)
        self._ref_buckets = [np.flatnonzero(self.ref == (j + 1)).astype(np.int32) for j in range(R)]

    def _refractory_indices(self) -> np.ndarray:
        buckets = [b for b in self._ref_buckets if b.shape[0]]
        if not buckets:
            return self._empty32
        if len(buckets) == 1:
            return buckets[0]
        return np.concatenate(buckets)

    def _step_one(self, step_i: int) -> None:
        n_slots = self.D + 1
        slot = self.t_step % n_slots
        g_in = self.ring[slot]
        t_now = self.t_step * self.dt_ms
        # 1. expire injections
        if self._next_expiry <= t_now + 1e-9:
            self._expire(t_now)
        v, g, ref, tmp = self.v, self.g, self.ref, self._tmp
        # 2. exact update for active neurons; refractory ones are frozen (save/restore) and count down
        rf = self._refractory_indices()
        n_rf = int(rf.shape[0])
        if n_rf:
            v_rf = v[rf]
            g_rf = g[rf]
        np.subtract(v, self._v_eq, out=tmp)
        np.multiply(tmp, self._a_m, out=tmp)
        np.add(tmp, self._v_eq, out=tmp)
        np.multiply(g, self._b, out=v)
        np.add(v, tmp, out=v)
        np.multiply(g, self._a_s, out=g)
        fired = self._fired
        np.greater(v, self._v_th, out=fired)
        if n_rf:
            v[rf] = v_rf
            g[rf] = g_rf
            ref[rf] -= 1
            fired[rf] = False
        # 3. threshold
        spk = np.flatnonzero(fired).astype(np.int32)
        # 4. delayed input + noise + forced kicks
        np.add(g, g_in, out=g)
        if self.noise_sigma > 0.0:
            rng = self.rng
            nb = self._noise
            off = int(rng.integers(0, self._pool_size - self._n))
            np.multiply(self._pool[off:off + self._n], self._sig_dt, out=nb)
            np.add(nb, self._mu_dt, out=nb)
            np.add(g, nb, out=g)
            pos = self._pool_pos
            rng.standard_normal(self._pool_refresh, dtype=np.float32, out=self._pool[pos:pos + self._pool_refresh])
            self._pool_pos = (pos + self._pool_refresh) % self._pool_size
        if self._inj:
            rng = self.rng
            kick = self._kick
            for r in self._inj.values():
                if r.kind != "poisson" or r.idx.size == 0:
                    continue
                hit = rng.random(r.idx.shape[0], dtype=np.float32) < r.p
                nh = int(np.count_nonzero(hit))
                if nh:
                    g[r.idx[hit]] += kick
                    self._forced_win += nh
        if self._pending_kicks:
            for idx, mv in self._pending_kicks:
                g[idx] += np.float32(mv)
                self._forced_win += int(idx.shape[0])
            self._pending_kicks.clear()
        # 5. reset
        k = int(spk.shape[0])
        if k:
            v[spk] = self._v_reset
            g[spk] = 0.0
            ref[spk] = self._ref_steps
        if self._ref_buckets:
            buckets = self._ref_buckets
            buckets.pop(0)                      # ref 1 -> 0: active again next step
            buckets.append(spk)                 # ref = ref_steps for the neurons that just fired
        # 6. zero the consumed slot, propagate into the delayed slot
        g_in.fill(0.0)
        if k:
            self._visits_win += self.prop(spk, self.ring[(self.t_step + self.D) % n_slots])
        # 7. record
        self.monitor.record(spk, step_i)
        self.t_step += 1
        if k > self._hot_thr:
            self.hot_steps += 1
            if self.hot_steps > self._hot_max:
                self._hot_max = self.hot_steps
        else:
            self.hot_steps = 0

    def step(self, n_steps: int = 1) -> StepStats:
        """Run ``n_steps`` of ``dt``; per step, in THIS order (SPEC section c.10):

          1. ``slot = t_step % (D+1)``; ``g_in = ring[slot]`` (view); expire injections whose ``until_ms <= t_ms``
          2. ``active = ref == 0``; ``v[active] = v_rest + I + (v - v_rest - I)*a_m + b*g`` (``I = i_ext + tonic``);
             ``g[active] *= a_s``; ``ref[~active] -= 1`` (refractory: v, g frozen)
          3. ``fired = active & (v > v_th)``; ``spk = flatnonzero(fired)``
          4. ``g += g_in + noise + forced kicks``: ``g += N(mu*dt, sigma*sqrt(dt))`` (if sigma > 0); for each
             Poisson injection ``p = rate*dt/1000`` per recruited neuron -> ``g[idx[hit]] += kick_mv``;
             current injections live in ``i_ext`` (rebuilt when they change)
          5. reset: ``v[spk] = v_reset``; ``g[spk] = 0``; ``ref[spk] = ref_steps``
          6. ``ring[slot] = 0``; ``propagate(spk)`` -> ``ring[(t_step + D) % (D+1)] += data_mv`` over outgoing edges
          7. ``monitor.record(spk, step_i)``; ``t_step += 1``

        Returns ``StepStats`` for the window (``monitor.flush()`` plus the engine-level fields).
        Latency contract: an isolated neuron receiving one 68.75 mV kick at step k fires at
        ``3.0 ms <= t <= 3.0 ms + dt`` (``test_engine::test_forced_kick_latency``)."""
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        t0 = self.t_ms
        self.monitor.begin(t0)
        self._forced_win = 0
        self._visits_win = 0
        self._hot_max = self.hot_steps
        wall0 = time.perf_counter()
        for i in range(n_steps):
            self._step_one(i)
        wall_ms = (time.perf_counter() - wall0) * 1000.0
        self.monitor.forced_events = self._forced_win
        self.monitor.edge_visits = self._visits_win
        stats = self.monitor.flush()
        stats.n_steps = n_steps
        stats.t0_ms = t0
        stats.t1_ms = self.t_ms
        stats.forced_events = self._forced_win
        stats.edge_visits = self._visits_win
        stats.step_ms_mean = wall_ms / n_steps if n_steps else 0.0
        stats.noise_on = self.noise_sigma > 0.0
        stats.gain = self.gain
        stats.hot_steps = self.hot_steps
        stats.hot_steps_max = self._hot_max
        self.forced_total += self._forced_win
        self.visits_total += self._visits_win
        return stats

    def propagate(self, spk: np.ndarray, out: np.ndarray) -> int:
        """Delegates to the ``Propagator``; returns edge visits."""
        return self.prop(np.ascontiguousarray(spk, dtype=np.int32), out)

    def apply_homeostasis(
        self,
        stats: StepStats,
        active_frac: float = HOMEOSTASIS_ACTIVE_FRAC,
        steps: int = HOMEOSTASIS_STEPS,
        factor: float = HOMEOSTASIS_FACTOR,
        floor: float = HOMEOSTASIS_FLOOR,
    ) -> dict | None:
        """Homeostasis rule ``[E]``: when the active fraction exceeded ``active_frac`` for ``steps`` consecutive
        LIF steps (``stats.hot_steps_max >= steps``), multiply the gain by ``factor`` (never below ``floor``),
        reset the hot counter and return the ``homeostasis`` event data ``{gain_before, gain_after,
        active_frac}``; otherwise return ``None``. ``active_frac`` other than the default only changes the
        reporting threshold if it matches the engine's counter threshold (``HOMEOSTASIS_ACTIVE_FRAC``)."""
        if stats.hot_steps_max < steps:
            return None
        before = self.gain
        after = max(float(floor), before * float(factor))
        self.hot_steps = 0
        self._hot_max = 0
        if after < before:
            self.set_gain(after)
        log.info("homeostasis: gain %.4f -> %.4f (active_frac_max %.4f)", before, after, stats.active_frac_max)
        return {"gain_before": before, "gain_after": after, "active_frac": float(stats.active_frac_max)}

    # ------------------------------------------------------------------ state

    def state_dict(self) -> dict[str, np.ndarray | int | float]:
        """Copies of ``v, g, ref, i_ext, tonic, ring`` plus ``t_step``, ``gain``, ``hot_steps``.

        Also carries the random stream so that a restored engine continues the original run bit for
        bit: ``rng_state`` (the ``Generator`` bit-generator state dict), ``noise_pool`` (a copy of the
        Gaussian pool) and ``noise_pool_pos``. Injections are NOT part of the state (the caller owns
        them); ``load_state_dict`` accepts a document without the random keys (then only the arrays
        match and the two futures diverge)."""
        return {
            "v": self.v.copy(),
            "g": self.g.copy(),
            "ref": self.ref.copy(),
            "i_ext": self.i_ext.copy(),
            "tonic": self.tonic.copy(),
            "ring": self.ring.copy(),
            "t_step": int(self.t_step),
            "gain": float(self.gain),
            "hot_steps": int(self.hot_steps),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "noise_pool": self._pool.copy(),
            "noise_pool_pos": int(self._pool_pos),
        }

    def load_state_dict(self, d: dict) -> None:
        """Inverse of ``state_dict`` (shapes must match; injections are left untouched). Restores the
        random stream when ``rng_state`` / ``noise_pool`` / ``noise_pool_pos`` are present."""
        for key, arr in (("v", self.v), ("g", self.g), ("ref", self.ref), ("i_ext", self.i_ext),
                         ("tonic", self.tonic), ("ring", self.ring)):
            if key in d:
                src = np.asarray(d[key])
                if src.shape != arr.shape:
                    raise ValueError(f"state_dict[{key!r}] has shape {src.shape}, expected {arr.shape}")
                arr[...] = src
        if "t_step" in d:
            self.t_step = int(d["t_step"])
        if "gain" in d:
            self.set_gain(float(d["gain"]))
        if "hot_steps" in d:
            self.hot_steps = int(d["hot_steps"])
        if d.get("rng_state") is not None:
            self.rng.bit_generator.state = copy.deepcopy(d["rng_state"])
        if d.get("noise_pool") is not None:
            src = np.asarray(d["noise_pool"], dtype=np.float32)
            if src.shape != self._pool.shape:
                raise ValueError(f"state_dict['noise_pool'] has shape {src.shape}, expected {self._pool.shape}")
            self._pool[...] = src
        if "noise_pool_pos" in d:
            self._pool_pos = int(d["noise_pool_pos"]) % self._pool_size
        self._rebuild_v_eq()
        self._rebuild_ref_buckets()

    def reseed(self, seed: int) -> None:
        """Re-seed the Poisson/noise generator (SeedSequence child [1] of ``seed``) and refill the noise pool."""
        self.seed = int(seed)
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(8)[_SEED_CHILD_ENGINE])
        self.rng.standard_normal(self._pool_size, dtype=np.float32, out=self._pool)
        self._pool_pos = 0

    def reset(self, reseed: bool = True) -> None:
        """Return to the initial state: ``v = v_rest``, ``g = 0``, ``ref = 0``, ``ring = 0``, ``t_step = 0``, all
        injections and pending kicks cleared, EMA rates zeroed; ``tonic`` and ``gain`` are kept.
        ``reseed=True`` also restarts the random stream from the engine's seed."""
        self.v.fill(self.params.v_rest)
        self.g.fill(0.0)
        self.ref.fill(0)
        self.ring.fill(0.0)
        self.t_step = 0
        self._inj.clear()
        self._pending_kicks.clear()
        self._next_expiry = math.inf
        self.hot_steps = 0
        self._hot_max = 0
        self._rebuild_i_ext()
        self._rebuild_ref_buckets()
        self.monitor.reset()
        if reseed:
            self.reseed(self.seed)

    def rates(self) -> dict[str, float]:
        """``monitor.rates.snapshot()``."""
        return self.monitor.rates.snapshot()


def apply_tonic_table(engine: LIFEngine, table: dict[str, float] = TONIC_TABLE_MV) -> dict[str, int]:
    """Apply ``TONIC_TABLE_MV`` (``[E]``) to ``engine``; returns ``{group: neurons affected}`` (missing groups -> 0)."""
    applied: dict[str, int] = {}
    for group, mv in table.items():
        idx = engine.conn.groups.get(group)
        n = int(idx.shape[0]) if idx is not None else 0
        if n:
            engine.set_tonic(group, float(mv))
        applied[group] = n
    return applied
