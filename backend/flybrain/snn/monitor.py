"""Spike bookkeeping: raster sample, per-population counts and EMA rates (SPEC section c.12).

``StepStats`` (the window summary of SPEC section c.10) is DEFINED here and re-exported by
``flybrain.snn.engine`` so that ``engine`` can import ``SpikeMonitor`` without a circular import.

Population layer (SPEC c.4 / c.12): ``groups.readout_partition`` assigns every neuron to at most one
readout id (first match in ``READOUTS`` order wins); sided readouts own three ids (``name``,
``name_L``, ``name_R``) and the unsuffixed count/size is folded as midline + L + R
(``groups.fold_sided_counts`` / ``groups.readout_sizes``). ``spike_counts_by_group`` and the
``RateEstimator`` use exactly those names; ``RateEstimator.snapshot()`` appends the nine derived keys
of SPEC d.2, so the wire order is ``readout_partition`` names + ``DERIVED_KEYS`` (131 keys for the
70-readout table of ``groups.py``).

No biological numbers live here except the EMA time constants of SPEC section c.12 ``[E]``.
Only numpy is imported at module level.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..connectome.groups import STAR_TYPES, fold_sided_counts, readout_partition, readout_sizes
from ..connectome.schema import REGION_ID, REGIONS

if TYPE_CHECKING:  # pragma: no cover
    from ..connectome.schema import Connectome

__all__ = ["RasterRow", "StepStats", "SpikeMonitor", "RateEstimator", "DERIVED_KEYS", "DT_SAMPLE_MAX"]

#: derived keys appended by ``RateEstimator.snapshot()`` in this order (SPEC d.2).
DERIVED_KEYS: tuple[str, ...] = (
    "lc_loom", "escape_vnc", "steer_a02_diff", "steer_a01_diff", "steer_g13_diff",
    "b1_diff", "i1_diff", "hg1_diff", "dn_mean",
)
_DIFF_BASES: tuple[str, ...] = ("steer_a02", "steer_a01", "steer_g13", "b1", "i1", "hg1")
_SIDE_CHAR = {-1: "L", 1: "R", 0: "M"}
log = logging.getLogger("flybrain.snn.monitor")

#: largest step offset representable in ``StepStats.spike_dt_sample`` (int16 per SPEC c.10).
DT_SAMPLE_MAX: int = 32767
_DN_REGION = REGION_ID["descending_motor"]


@dataclass(frozen=True)
class RasterRow:
    """One raster lane row (``hello.raster.rows[slot]``); ``side`` is ``'L'|'R'|'M'``; ``neuron`` -1 = empty slot."""

    region: int
    slot: int
    neuron: int
    label: str
    side: str
    star: bool

    def to_wire(self) -> dict:
        return {"region": self.region, "slot": self.slot, "neuron": self.neuron, "label": self.label,
                "side": self.side, "star": self.star}


@dataclass(slots=True)
class StepStats:
    """Summary of one ``LIFEngine.step`` window (SPEC section c.10).

    ``spike_counts_by_group``: readout partition names (incl. sided keys, folded) -> spikes in the
    window; ``region_counts`` int64[8]; ``spike_indices_sample`` int32 raster slots and
    ``spike_dt_sample`` int16 step offsets (parallel, sorted by dt, uniformly subsampled to ``cap``
    with ``capped=True``; a window longer than ``DT_SAMPLE_MAX + 1`` steps has its offsets clipped to
    ``DT_SAMPLE_MAX`` with a warning instead of wrapping negative - the production tick is 50 steps);
    ``rates`` = ``RateEstimator.snapshot()``; ``region_rates`` float32[8].
    ``hot_steps`` = consecutive steps with ``spikes/n`` above the homeostasis threshold at the end of
    the window and ``hot_steps_max`` the longest such run seen in the window (carry-in included) -
    the loop's runaway / homeostasis inputs (extra fields, defaulted).
    """

    n_steps: int
    t0_ms: int
    t1_ms: int
    spike_counts_by_group: dict[str, int]
    region_counts: np.ndarray
    total_spikes: int
    active_frac_max: float
    spike_indices_sample: np.ndarray
    spike_dt_sample: np.ndarray
    capped: bool
    rates: dict[str, float]
    region_rates: np.ndarray
    forced_events: int
    edge_visits: int
    step_ms_mean: float
    gf_spikes: dict[str, int]
    noise_on: bool
    gain: float
    hot_steps: int = 0
    hot_steps_max: int = 0


# --------------------------------------------------------------------------- rates


def _tau_for(name: str) -> float:
    base = name[:-2] if (name.endswith("_L") or name.endswith("_R")) else name
    tau = RateEstimator.TAU_S.get(base)
    if tau is not None:
        return tau
    for prefix, t in RateEstimator.TAU_PREFIX_S:
        if base.startswith(prefix):
            return t
    return RateEstimator.DEFAULT_TAU_S


class RateEstimator:
    """Exponential-moving-average firing rates in Hz per neuron (SPEC section c.12).

    ``r <- r*exp(-window/tau) + (counts/(size*window))*(1-exp(-window/tau))``; ``size 0 -> 0``.
    Time constants ``[E]``: fast escape channels 0.05 s, locomotion/steering 0.15 s, slow
    feeding/valence/song channels 0.30 s, default 0.20 s, regions 0.25 s. ``_L``/``_R`` names
    inherit the constant of their base readout.
    """

    TAU_S: dict[str, float] = {
        "gf": 0.05, "escape_dn": 0.05, "dn_saccade": 0.05, "ttmn": 0.05, "psi": 0.05, "gfc2": 0.05,
        "dn_fwd": 0.15, "dng100": 0.15, "dn_halt": 0.15, "dn_back": 0.15, "flight_dn": 0.15,
        "dn_freeze": 0.15, "groom_dn": 0.15, "wing_power": 0.15, "wing_steer": 0.15,
        "b1": 0.15, "i1": 0.15, "hg1": 0.15,
        "feed_mn": 0.30, "feed_dn": 0.30, "feed_pre_exc": 0.30, "feed_pre_inh": 0.30,
        "sugar2_exc": 0.30, "sugar2_inh": 0.30, "mbon_avoid": 0.30, "mbon_approach": 0.30,
        "pam": 0.30, "ppl1": 0.30, "p1": 0.30, "song_dn": 0.30, "song_vnc": 0.30,
        "pip10": 0.30, "dms2": 0.30,
    }
    TAU_PREFIX_S: tuple[tuple[str, float], ...] = (
        ("steer_", 0.15), ("wing_", 0.15), ("feed_", 0.30), ("sugar2_", 0.30), ("mbon_", 0.30), ("song_", 0.30),
    )
    DEFAULT_TAU_S: float = 0.20
    REGION_TAU_S: float = 0.25

    def __init__(self, names: list[str], sizes: np.ndarray, tick_s: float) -> None:
        self.names: list[str] = list(names)
        self.sizes = np.asarray(sizes, dtype=np.int64).copy()
        if self.sizes.shape != (len(self.names),):
            raise ValueError(f"sizes must have length {len(self.names)}, got {self.sizes.shape}")
        self.tick_s = float(tick_s)
        self.tau = np.array([_tau_for(n) for n in self.names], dtype=np.float64)
        self.r = np.zeros(len(self.names), dtype=np.float64)
        self.r_reg = np.zeros(len(REGIONS), dtype=np.float64)
        self._pos = {name: i for i, name in enumerate(self.names)}
        self._size_safe = np.where(self.sizes > 0, self.sizes, 1).astype(np.float64)
        self._has = (self.sizes > 0).astype(np.float64)
        self._decay_window = -1.0
        self._decay = np.ones_like(self.r)
        self._decay_reg = 1.0
        self._i_lc4 = self._pos.get("lc4")
        self._i_lplc2 = self._pos.get("lplc2")
        self._i_esc = [self._pos[n] for n in ("ttmn", "psi", "gfc2") if n in self._pos]
        self._diff_idx = {b: (self._pos.get(b + "_L"), self._pos.get(b + "_R")) for b in _DIFF_BASES}

    def _prepare(self, window_s: float) -> None:
        if window_s != self._decay_window:
            self._decay = np.exp(-window_s / self.tau)
            self._decay_reg = math.exp(-window_s / self.REGION_TAU_S)
            self._decay_window = window_s

    def update(self, counts: np.ndarray, region_counts: np.ndarray, region_sizes: np.ndarray,
               window_s: float | None = None) -> None:
        """EMA update with the spike ``counts`` (aligned with ``names``) of a window of ``window_s`` seconds
        (default ``tick_s``); ``region_counts``/``region_sizes`` int[8]. Hz per neuron; ``size 0 -> 0``;
        a non-positive window is ignored."""
        w = self.tick_s if window_s is None else float(window_s)
        if not (w > 0.0):
            return
        self._prepare(w)
        inst = np.asarray(counts, dtype=np.float64) / (self._size_safe * w) * self._has
        self.r *= self._decay
        self.r += inst * (1.0 - self._decay)
        rs = np.asarray(region_sizes, dtype=np.float64)
        inst_reg = np.divide(np.asarray(region_counts, dtype=np.float64), np.where(rs > 0, rs, 1.0) * w) * (rs > 0)
        self.r_reg *= self._decay_reg
        self.r_reg += inst_reg * (1.0 - self._decay_reg)

    def reset(self) -> None:
        self.r[...] = 0.0
        self.r_reg[...] = 0.0

    def get(self, name: str) -> float:
        i = self._pos.get(name)
        return float(self.r[i]) if i is not None else 0.0

    def size_of(self, name: str) -> int:
        i = self._pos.get(name)
        return int(self.sizes[i]) if i is not None else 0

    def keys(self) -> list[str]:
        """Wire key order of ``snapshot()`` (``names`` then ``DERIVED_KEYS``) = ``hello.pops``."""
        return self.names + list(DERIVED_KEYS)

    def derived(self) -> dict[str, float]:
        """The nine ``DERIVED_KEYS`` only (SPEC c.4 refers to this name, c.12 folds them into
        ``snapshot()``; both readings give the same numbers)."""
        snap = self.snapshot()
        return {k: snap[k] for k in DERIVED_KEYS}

    def snapshot(self) -> dict[str, float]:
        """All readouts + derived (``derived()`` returns the derived part alone): ``lc_loom`` =
        size-weighted mean of lc4/lplc2; ``escape_vnc`` = mean(ttmn, psi, gfc2);
        ``<x>_diff = <x>_R - <x>_L`` for steer_a02/a01/g13, b1, i1, hg1; ``dn_mean`` =
        descending_motor region rate."""
        r = self.r
        out: dict[str, float] = dict(zip(self.names, r.tolist()))
        s4 = int(self.sizes[self._i_lc4]) if self._i_lc4 is not None else 0
        s2 = int(self.sizes[self._i_lplc2]) if self._i_lplc2 is not None else 0
        tot = s4 + s2
        if tot > 0:
            lc = (r[self._i_lc4] * s4 if s4 else 0.0) + (r[self._i_lplc2] * s2 if s2 else 0.0)
            out["lc_loom"] = float(lc / tot)
        else:
            out["lc_loom"] = 0.0
        out["escape_vnc"] = float(sum(r[i] for i in self._i_esc) / 3.0)
        for base, (il, ir) in self._diff_idx.items():
            out[base + "_diff"] = float((r[ir] if ir is not None else 0.0) - (r[il] if il is not None else 0.0))
        out["dn_mean"] = float(self.r_reg[_DN_REGION])
        return out

    def regions(self) -> np.ndarray:
        """float32[8] Hz per neuron per region (EMA, tau 0.25 s)."""
        return self.r_reg.astype(np.float32)


# --------------------------------------------------------------------------- monitor


class SpikeMonitor:
    """Per-window spike bookkeeping (SPEC section c.12).

    ``pop_id`` / ``names`` from ``groups.readout_partition`` (sizes from ``groups.readout_sizes``);
    ``region`` uint8 from ``conn``. Raster sample: per region, first every ``STAR_TYPES`` neuron present
    in that region (``STAR_TYPES`` order, L then R then midline - the literal rule of SPEC c.12, i.e.
    ``star_per_side=None``), then a seeded random sample of the rest, up to ``per_region`` rows;
    ``slot_of_neuron`` int32[n] (-1 = unsampled); ``rows`` has exactly ``8 * per_region`` entries with
    ``rows[i].slot == i`` (empty slots carry ``neuron == -1``). ``star_per_side=k`` is an optional
    non-default cap of ``k`` cells per (star type, side) for callers that want a mixed lane even where
    one star population (e.g. 400 R1-R6 cells) would fill the whole optic-lobe lane.

    ``record`` only appends the spike array of the step (cheap); all counting happens in ``flush``.
    ``dt_ms`` is set by the owning ``LIFEngine`` and sizes the EMA window; ``tick_s`` is the default
    window of the ``RateEstimator``.
    """

    def __init__(self, conn: "Connectome", per_region: int = 48, cap: int = 2000, seed: int = 0,
                 dt_ms: float = 1.0, tick_s: float = 0.05, star_per_side: int | None = None) -> None:
        self.conn = conn
        self.n = int(conn.n)
        self.per_region = int(per_region)
        self.cap = int(cap)
        self.dt_ms = float(dt_ms)
        self.star_per_side = None if star_per_side is None else max(0, int(star_per_side))
        self.region = np.ascontiguousarray(conn.region, dtype=np.uint8)
        self.region_sizes = np.bincount(self.region.astype(np.int64), minlength=len(REGIONS)).astype(np.int64)

        # --- population partition (SPEC c.4) -----------------------------------------------
        self.pop_id, self.names = readout_partition(conn.groups, self.n)
        self.sizes = readout_sizes(self.pop_id, self.names)
        self._n_names = len(self.names)
        self._pos = {nm: i for i, nm in enumerate(self.names)}
        self.rates = RateEstimator(self.names, self.sizes, tick_s)

        # --- DNp01 per side ------------------------------------------------------------------
        empty = np.zeros(0, dtype=np.int32)
        self.gf_side = np.zeros(self.n, dtype=np.int8)
        gl = np.asarray(conn.groups.get("gf_L", empty), dtype=np.int64)
        gr = np.asarray(conn.groups.get("gf_R", empty), dtype=np.int64)
        if gl.size:
            self.gf_side[gl] = 1
        if gr.size:
            self.gf_side[gr] = 2

        # --- raster sample --------------------------------------------------------------------
        self.rng = np.random.default_rng(seed)
        self.rows: list[RasterRow] = []
        self.slot_of_neuron = np.full(self.n, -1, dtype=np.int32)
        self._build_raster(conn)

        self._spk: list[np.ndarray] = []
        self._lens: list[int] = []
        self._t0_ms = 0
        self._dt_clip_warned = False
        self.forced_events = 0
        self.edge_visits = 0

    # ---- construction helpers ----------------------------------------------------------------

    def _build_raster(self, conn: "Connectome") -> None:
        types = list(conn.types)
        type_idx = np.asarray(conn.type_idx, dtype=np.int64)
        side = np.asarray(conn.side, dtype=np.int64)
        star_rank = np.full(len(types), -1, dtype=np.int64)
        rank = 0
        for t in STAR_TYPES:
            if t in types:
                star_rank[types.index(t)] = rank
                rank += 1
        is_star = star_rank[type_idx] >= 0
        side_rank = np.where(side == -1, 0, np.where(side == 1, 1, 2))
        rows: list[RasterRow] = []
        for r in range(len(REGIONS)):
            members = np.flatnonzero(self.region == r)
            chosen: list[int] = []
            stars = members[is_star[members]]
            if stars.size:
                order = np.lexsort((stars, side_rank[stars], star_rank[type_idx[stars]]))
                stars = stars[order]
                if self.star_per_side is not None:
                    key = star_rank[type_idx[stars]] * 3 + side_rank[stars]
                    # ordinal of every star within its (type, side) run (runs are contiguous after the sort)
                    first = np.ones(stars.shape[0], dtype=bool)
                    first[1:] = key[1:] != key[:-1]
                    starts = np.flatnonzero(first)
                    ordinal = np.arange(stars.shape[0]) - np.repeat(starts, np.diff(np.append(starts, stars.shape[0])))
                    stars = stars[ordinal < self.star_per_side]
                chosen.extend(int(x) for x in stars[: self.per_region])
            room = self.per_region - len(chosen)
            if room > 0:
                taken = np.zeros(self.n, dtype=bool)
                if chosen:
                    taken[np.asarray(chosen, dtype=np.int64)] = True
                rest = members[~taken[members]]
                if rest.size:
                    take = min(room, int(rest.size))
                    sample = self.rng.choice(rest, size=take, replace=False)
                    chosen.extend(int(x) for x in np.sort(sample))
            for j in range(self.per_region):
                slot = r * self.per_region + j
                if j < len(chosen):
                    i = chosen[j]
                    self.slot_of_neuron[i] = slot
                    rows.append(RasterRow(region=r, slot=slot, neuron=i, label=types[int(type_idx[i])],
                                          side=_SIDE_CHAR[int(side[i])], star=bool(is_star[i])))
                else:
                    rows.append(RasterRow(region=r, slot=slot, neuron=-1, label="", side="M", star=False))
        self.rows = rows

    # ---- public API ------------------------------------------------------------------------------

    def size_of(self, name: str) -> int:
        """Population size behind a partition name (0 for unknown names)."""
        i = self._pos.get(name)
        return int(self.sizes[i]) if i is not None else 0

    def begin(self, t0_ms: int) -> None:
        self._t0_ms = int(t0_ms)
        self._spk = []
        self._lens = []
        self.forced_events = 0
        self.edge_visits = 0

    def record(self, spk: np.ndarray, step_i: int) -> None:
        """Append the spike indices of step ``step_i`` (counted in ``flush``)."""
        self._spk.append(spk)
        self._lens.append(int(spk.shape[0]))

    def reset(self) -> None:
        """Forget the current window and zero the EMA rates."""
        self._spk = []
        self._lens = []
        self.rates.reset()

    def flush(self) -> StepStats:
        """Build ``StepStats`` for the recorded window: ``np.bincount(pop_id[spk][pop_id[spk] >= 0])``
        folded into the sided slots, region bincount, DNp01 spikes per side, max active fraction, the
        raster events of sampled neurons uniformly subsampled to ``cap`` (``capped=True`` when
        truncated); then update the ``RateEstimator`` and reset the counters. Engine-level fields
        (forced_events, edge_visits, step_ms_mean, noise_on, gain, hot_steps) are filled in by
        ``LIFEngine.step``."""
        n_steps = len(self._lens)
        lens = np.asarray(self._lens, dtype=np.int64)
        total = int(lens.sum()) if n_steps else 0
        counts = np.zeros(self._n_names, dtype=np.int64)
        region_counts = np.zeros(len(REGIONS), dtype=np.int64)
        gf = {"L": 0, "R": 0}
        slots = np.zeros(0, dtype=np.int32)
        dts = np.zeros(0, dtype=np.int16)
        capped = False
        if total:
            all_spk = np.concatenate(self._spk).astype(np.int64, copy=False)
            pid = self.pop_id[all_spk]
            pid = pid[pid >= 0]
            if pid.size:
                counts = fold_sided_counts(np.bincount(pid.astype(np.int64), minlength=self._n_names), self.names)
            region_counts = np.bincount(self.region[all_spk].astype(np.int64), minlength=len(REGIONS)).astype(np.int64)
            gs = self.gf_side[all_spk]
            if gs.any():
                gf = {"L": int(np.count_nonzero(gs == 1)), "R": int(np.count_nonzero(gs == 2))}
            s = self.slot_of_neuron[all_spk]
            mask = s >= 0
            if mask.any():
                slots = s[mask].astype(np.int32, copy=False)
                # int32 while gathering: a window longer than 32768 steps would wrap int16 negative
                dt32 = np.repeat(np.arange(n_steps, dtype=np.int32), lens)[mask]
                if slots.shape[0] > self.cap:
                    pick = (np.arange(self.cap, dtype=np.int64) * slots.shape[0]) // self.cap
                    slots = slots[pick]
                    dt32 = dt32[pick]
                    capped = True
                if n_steps > DT_SAMPLE_MAX + 1:
                    np.clip(dt32, 0, DT_SAMPLE_MAX, out=dt32)
                    if not self._dt_clip_warned:
                        self._dt_clip_warned = True
                        log.warning("spike_dt_sample: window of %d steps exceeds the int16 range, "
                                    "offsets clipped to %d", n_steps, DT_SAMPLE_MAX)
                dts = dt32.astype(np.int16)
        window_s = n_steps * self.dt_ms / 1000.0
        if n_steps:
            self.rates.update(counts, region_counts, self.region_sizes, window_s)
        stats = StepStats(
            n_steps=n_steps,
            t0_ms=self._t0_ms,
            t1_ms=int(math.floor(self._t0_ms + n_steps * self.dt_ms + 1e-9)),
            spike_counts_by_group=dict(zip(self.names, counts.tolist())),
            region_counts=region_counts,
            total_spikes=total,
            active_frac_max=(float(lens.max()) / self.n) if (n_steps and self.n) else 0.0,
            spike_indices_sample=np.ascontiguousarray(slots, dtype=np.int32),
            spike_dt_sample=np.ascontiguousarray(dts, dtype=np.int16),
            capped=capped,
            rates=self.rates.snapshot(),
            region_rates=self.rates.regions(),
            forced_events=self.forced_events,
            edge_visits=self.edge_visits,
            step_ms_mean=0.0,
            gf_spikes=gf,
            noise_on=True,
            gain=1.0,
        )
        self._spk = []
        self._lens = []
        self.forced_events = 0
        self.edge_visits = 0
        return stats

    def rows_wire(self) -> list[dict]:
        """``hello.raster.rows`` (list of dicts, ``rows[i]['slot'] == i``)."""
        return [r.to_wire() for r in self.rows]
