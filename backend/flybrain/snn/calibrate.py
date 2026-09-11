"""Boot gates and gain calibration (SPEC section c.13, gates of section g.6).

``run_gates`` runs the five deterministic checks of the selftest on a fresh ``LIFEngine``:
(1) 1 s of rest with noise and no drives (mean rate 1-5 Hz, per-step active fraction <= 2 %),
(2) 1 s of labellar sugar at 100 Hz -> MN9 rate over the last 500 ms in [20, 90] Hz, (3) 200 ms of
looming at 150 Hz on all of ``lc_loom`` -> latency to the first DNp01 (giant fibre) spike <= 20 ms,
(4) 500 ms of ``pfl3_L`` at 60 Hz -> ``steer_a02_R - steer_a02_L`` over the last 250 ms >= 10 Hz
(contralateral PFL3 -> DNa02), (5) the runaway flag (active fraction > 5 % for 10 consecutive
steps) over all runs, plus (6) the real-time factor of run (1). ``calibrate_gain`` bisects the
synaptic gain on a log scale until MN9 lands in its band without runaway and caches the result
under ``data/cache/<connectome_key>.calib.json``.

Targets: rest 1-5 Hz ``[V]`` (task brief / RESEARCH section 7: 2.3 Hz isolated), MN9 20-90 Hz during
sugar ``[E]`` (feeding motor rates, literature order of magnitude), GF latency <= 20 ms ``[L]``
(LC4/LPLC2 -> DNp01 is a 2-synapse path with a 1.8 ms delay per synapse; g.6 expects 8-12 ms),
DNa02 asymmetry >= 10 Hz ``[V]`` pathway / ``[E]`` number. ``Settings`` is only used through
``getattr`` (``seed``, ``dt_ms``, ``backend``, ``noise_mu``, ``noise_sigma``, ``drive_mode``,
``data_dir``, ``connectome_key``) so the conftest placeholder namespace works too. The tonic table of
SPEC c.10 is NOT applied unless ``tonic=True`` (c.13 specifies a fresh engine; the loop applies the
table at start). Only numpy is imported at module level.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .engine import HOMEOSTASIS_STEPS, LIFEngine, apply_tonic_table
from .params import LIFParams

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Settings
    from ..connectome.schema import Connectome

__all__ = ["CalibTargets", "GateReport", "run_gates", "calibrate_gain", "calib_path", "load_calibration",
           "GATE_NAMES"]

log = logging.getLogger("flybrain.snn.calibrate")

_REST_MS = 1000
_SUGAR_MS = 1000
_SUGAR_MEASURE_MS = 500
_LOOM_MS = 200
_PFL3_MS = 500
_PFL3_MEASURE_MS = 250
_WINDOW_MS = 50.0   # tick-sized windows

#: keys of ``GateReport.passed`` in report order.
GATE_NAMES: tuple[str, ...] = ("rest_rate", "rest_active_frac", "mn9", "gf_latency", "a02_diff", "runaway")


@dataclass(frozen=True)
class CalibTargets:
    """Gate thresholds (SPEC section c.13)."""

    rest_rate_min_hz: float = 1.0
    rest_rate_max_hz: float = 5.0        # mean over all neurons, 1 s, noise on, no drives
    rest_active_frac_max: float = 0.02   # max per-step spikes/n at rest
    sugar_rate_hz: float = 100.0         # grn_sugar_labellar 100 Hz for 1 s
    mn9_min_hz: float = 20.0
    mn9_max_hz: float = 90.0
    loom_rate_hz: float = 150.0          # lc_loom 150 Hz, recruit 1.0 -> first DNp01 spike
    gf_latency_max_ms: float = 20.0
    runaway_active_frac: float = 0.05    # never exceeded for 10 consecutive steps
    pfl3_rate_hz: float = 60.0           # pfl3_L 60 Hz for 500 ms -> steer_a02_R - steer_a02_L
    a02_diff_min_hz: float = 10.0


@dataclass
class GateReport:
    """Result of ``run_gates``; ``passed`` has one bool per gate (``GATE_NAMES``), ``ok()`` is their conjunction.

    ``gf_latency_ms`` is ``None`` when no DNp01 spike occurred within the loom window; ``region_rates_hz``
    (extra, informative) holds the rest-run mean rate per region in ``REGIONS`` order.
    """

    rest_rate_hz: float
    rest_active_frac: float
    mn9_hz: float
    gf_latency_ms: float | None
    a02_diff_hz: float
    runaway: bool
    rtf: float
    gain: float
    passed: dict[str, bool]
    notes: list[str] = field(default_factory=list)
    region_rates_hz: list[float] = field(default_factory=list)

    def ok(self) -> bool:
        return all(self.passed.values())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok()
        return d


def _setting(settings: Any, name: str, default: Any) -> Any:
    val = getattr(settings, name, None)
    return default if val is None else val


def _make_engine(conn: "Connectome", settings: Any, gain: float) -> LIFEngine:
    return LIFEngine(
        conn,
        LIFParams(),
        seed=int(_setting(settings, "seed", 0)),
        backend=str(_setting(settings, "backend", "numpy")),
        dt_ms=float(_setting(settings, "dt_ms", 1.0)),
        gain=float(gain),
        noise_mu=float(_setting(settings, "noise_mu", 0.5)),
        noise_sigma=float(_setting(settings, "noise_sigma", 3.5)),
        drive_mode=str(_setting(settings, "drive_mode", "poisson")),
    )


def _run_ms(eng: LIFEngine, total_ms: float, window_ms: float = _WINDOW_MS) -> list:
    """Run ``total_ms`` of brain time in ``window_ms`` chunks; returns the list of StepStats."""
    steps_total = int(round(total_ms / eng.dt_ms))
    per = max(1, int(round(window_ms / eng.dt_ms)))
    out = []
    done = 0
    while done < steps_total:
        k = min(per, steps_total - done)
        out.append(eng.step(k))
        done += k
    return out


def _group_empty(conn: "Connectome", name: str) -> bool:
    idx = conn.groups.get(name)
    return idx is None or int(np.asarray(idx).size) == 0


def run_gates(conn: "Connectome", settings: "Settings", gain: float, targets: CalibTargets = CalibTargets(),
              tonic: bool = False) -> GateReport:
    """Fresh ``LIFEngine(seed=settings.seed)``. (1) rest 1000 ms; (2) reset state, sugar for 1000 ms, MN9 rate
    over the last 500 ms; (3) reset, loom, run 200 ms, latency to the first DNp01 spike; (4) reset, ``pfl3_L``
    Poisson ``pfl3_rate_hz`` for 500 ms, ``a02_diff_hz = steer_a02_R - steer_a02_L`` over the last 250 ms;
    (5) runaway flag from all runs; (6) rtf = brain ms / wall ms over run (1). The five gates are SPEC g.6.
    Deterministic for a given connectome, settings and gain (``rtf`` excepted). A gate whose groups are
    empty is reported as failed with an explanatory note (never an exception). ``tonic=True`` applies
    ``TONIC_TABLE_MV`` first (as the running server does; not part of c.13)."""
    eng = _make_engine(conn, settings, gain)
    notes: list[str] = []
    if tonic:
        applied = apply_tonic_table(eng)
        missing = [k for k, v in applied.items() if v == 0]
        if missing:
            notes.append("tonic groups empty: " + ", ".join(missing))
    n = eng.n
    hot_max = 0

    # (1) rest -----------------------------------------------------------------------------
    wall0 = time.perf_counter()
    stats1 = _run_ms(eng, _REST_MS)
    wall_ms = (time.perf_counter() - wall0) * 1000.0
    total = sum(s.total_spikes for s in stats1)
    rest_rate = total / n / (_REST_MS / 1000.0) if n else 0.0
    rest_af = max((s.active_frac_max for s in stats1), default=0.0)
    hot_max = max([hot_max] + [s.hot_steps_max for s in stats1])
    rtf = _REST_MS / max(wall_ms, 1e-3)
    reg_counts = np.zeros(8, dtype=np.int64)
    for s in stats1:
        reg_counts += s.region_counts
    reg_sizes = eng.monitor.region_sizes
    region_rates = [float(reg_counts[i]) / float(reg_sizes[i]) / (_REST_MS / 1000.0) if reg_sizes[i] else 0.0
                    for i in range(8)]

    # (2) sugar -> MN9 -----------------------------------------------------------------------
    eng.reset()
    mn9_size = eng.monitor.size_of("feed_mn")
    mn9_hz = 0.0
    if _group_empty(conn, "grn_sugar_labellar"):
        notes.append("grn_sugar_labellar is empty: sugar gate skipped")
    elif mn9_size == 0:
        notes.append("feed_mn (MN9) is empty: sugar gate skipped")
    else:
        eng.inject("grn_sugar_labellar", rate_hz=targets.sugar_rate_hz, duration_ms=_SUGAR_MS, tag="sugar")
        stats2 = _run_ms(eng, _SUGAR_MS)
        t_meas = _SUGAR_MS - _SUGAR_MEASURE_MS
        mn9 = sum(s.spike_counts_by_group.get("feed_mn", 0) for s in stats2 if s.t0_ms >= t_meas)
        mn9_hz = mn9 / mn9_size / (_SUGAR_MEASURE_MS / 1000.0)
        hot_max = max([hot_max] + [s.hot_steps_max for s in stats2])

    # (3) loom -> GF latency --------------------------------------------------------------------
    eng.reset()
    gf_latency: float | None = None
    if _group_empty(conn, "lc_loom"):
        notes.append("lc_loom is empty: loom gate skipped")
    elif _group_empty(conn, "gf"):
        notes.append("gf (DNp01) is empty: loom gate skipped")
    else:
        eng.inject("lc_loom", rate_hz=targets.loom_rate_hz, duration_ms=_LOOM_MS, recruit=1.0, tag="loom")
        steps = int(round(_LOOM_MS / eng.dt_ms))
        for i in range(steps):
            s = eng.step(1)
            hot_max = max(hot_max, s.hot_steps_max)
            if gf_latency is None and (s.gf_spikes["L"] + s.gf_spikes["R"]) > 0:
                gf_latency = i * eng.dt_ms
                # keep running so the runaway check covers the full loom window
        if gf_latency is None:
            notes.append(f"no DNp01 spike within {_LOOM_MS} ms of looming")

    # (4) PFL3 -> DNa02 contralateral ---------------------------------------------------------------
    eng.reset()
    a02_diff = 0.0
    size_l = eng.monitor.size_of("steer_a02_L")
    size_r = eng.monitor.size_of("steer_a02_R")
    if _group_empty(conn, "pfl3_L"):
        notes.append("pfl3_L is empty: steering gate skipped")
    elif size_l == 0 or size_r == 0:
        notes.append("steer_a02_L / steer_a02_R is empty: steering gate skipped")
    else:
        eng.inject("pfl3_L", rate_hz=targets.pfl3_rate_hz, duration_ms=_PFL3_MS, tag="pfl3")
        stats4 = _run_ms(eng, _PFL3_MS)
        t_meas = _PFL3_MS - _PFL3_MEASURE_MS
        c_l = sum(s.spike_counts_by_group.get("steer_a02_L", 0) for s in stats4 if s.t0_ms >= t_meas)
        c_r = sum(s.spike_counts_by_group.get("steer_a02_R", 0) for s in stats4 if s.t0_ms >= t_meas)
        win_s = _PFL3_MEASURE_MS / 1000.0
        a02_diff = c_r / size_r / win_s - c_l / size_l / win_s
        hot_max = max([hot_max] + [s.hot_steps_max for s in stats4])

    # (5) runaway, (6) rtf ------------------------------------------------------------------------------
    runaway = hot_max >= HOMEOSTASIS_STEPS
    passed = {
        "rest_rate": targets.rest_rate_min_hz <= rest_rate <= targets.rest_rate_max_hz,
        "rest_active_frac": rest_af <= targets.rest_active_frac_max,
        "mn9": targets.mn9_min_hz <= mn9_hz <= targets.mn9_max_hz,
        "gf_latency": gf_latency is not None and gf_latency <= targets.gf_latency_max_ms,
        "a02_diff": a02_diff >= targets.a02_diff_min_hz,
        "runaway": not runaway,
    }
    rep = GateReport(rest_rate_hz=float(rest_rate), rest_active_frac=float(rest_af), mn9_hz=float(mn9_hz),
                     gf_latency_ms=gf_latency, a02_diff_hz=float(a02_diff), runaway=bool(runaway),
                     rtf=float(rtf), gain=float(gain), passed=passed, notes=notes,
                     region_rates_hz=region_rates)
    log.info("gates gain=%.4f rest=%.2fHz af=%.4f mn9=%.1fHz gf=%s a02diff=%.1fHz runaway=%s rtf=%.1f ok=%s",
             gain, rest_rate, rest_af, mn9_hz, "none" if gf_latency is None else f"{gf_latency:.1f}ms",
             a02_diff, runaway, rtf, rep.ok())
    return rep


def calib_path(settings: Any, key: str) -> Path:
    """``<data_dir>/cache/<key>.calib.json``."""
    return Path(_setting(settings, "data_dir", Path("data"))) / "cache" / f"{key}.calib.json"


def load_calibration(settings: Any, key: str) -> dict | None:
    """The cached ``calibrate_gain`` result for ``key`` (a dict with at least ``gain``) or ``None``."""
    path = calib_path(settings, key)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or "gain" not in doc:
        return None
    return doc


def _key_of(conn: "Connectome", settings: Any) -> str:
    key = str(_setting(settings, "connectome_key", "") or "")
    return key or str(conn.name)


def _identity_of(conn: "Connectome") -> dict:
    """Graph identity stored in (and checked against) the cached document: a cache written for another
    connectome must never be handed back, even when ``settings.connectome_key`` is unchanged."""
    return {"connectome": str(conn.name), "n": int(conn.n), "e": int(conn.e)}


def _band_distance(mn9_hz: float, targets: CalibTargets) -> float:
    if mn9_hz < targets.mn9_min_hz:
        return targets.mn9_min_hz - mn9_hz
    if mn9_hz > targets.mn9_max_hz:
        return mn9_hz - targets.mn9_max_hz
    return 0.0


def calibrate_gain(conn: "Connectome", settings: "Settings", targets: CalibTargets = CalibTargets(),
                   lo: float = 0.05, hi: float = 1.5, iters: int = 10, force: bool = False) -> dict:
    """Bisection on gain (log scale): raise gain while MN9 < min and no runaway; lower while runaway or
    MN9 > max; stop early once MN9 is inside its band without runaway. Writes
    ``data/cache/<connectome_key>.calib.json`` ``{gain, report, targets, created, key, connectome, n, e,
    converged, history}`` and returns that dict. A cached document is returned without re-running only
    when its ``targets`` AND its graph identity (``connectome`` name, ``n``, ``e``) match the call, so a
    stale cache under an unchanged ``connectome_key`` cannot hand back a gain bisected on another graph;
    ``force=True`` always recalibrates. Literature-weight mode and real data only (calibrated synthetic uses
    gain 1.0 and merely runs ``run_gates`` in selftest). When no iterate lands in the band, the non-runaway
    iterate whose MN9 rate is closest to the band is chosen (``converged=False``); when every iterate ran
    away the lowest bracket end is evaluated and returned."""
    if not (0.0 < lo < hi):
        raise ValueError(f"need 0 < lo < hi, got lo={lo} hi={hi}")
    key = _key_of(conn, settings)
    ident = _identity_of(conn)
    if not force:
        cached = load_calibration(settings, key)
        if cached is not None and cached.get("targets") == asdict(targets):
            same = all(cached.get(k) == v for k, v in ident.items())
            if same:
                log.info("calibrate_gain: cached gain=%.4f for key %s", float(cached["gain"]), key)
                return cached
            log.info("calibrate_gain: cache for key %s was built on %s (n=%s e=%s), recalibrating for %s "
                     "(n=%d e=%d)", key, cached.get("connectome"), cached.get("n"), cached.get("e"),
                     ident["connectome"], ident["n"], ident["e"])
    lo_g, hi_g = float(lo), float(hi)
    history: list[dict] = []
    best: tuple[float, GateReport] | None = None
    converged = False
    for it in range(int(iters)):
        mid = math.sqrt(lo_g * hi_g)
        rep = run_gates(conn, settings, mid, targets)
        history.append({"iter": it, "gain": mid, "mn9_hz": rep.mn9_hz, "runaway": rep.runaway,
                        "rest_rate_hz": rep.rest_rate_hz, "ok": rep.ok()})
        if rep.runaway or rep.mn9_hz > targets.mn9_max_hz:
            hi_g = mid
        elif rep.mn9_hz < targets.mn9_min_hz:
            lo_g = mid
        else:
            best = (mid, rep)
            converged = True
            break
        if not rep.runaway:
            if best is None or _band_distance(rep.mn9_hz, targets) < _band_distance(best[1].mn9_hz, targets):
                best = (mid, rep)
    if best is None:
        # every iterate ran away: fall back to the lowest bracket end
        rep = run_gates(conn, settings, lo_g, targets)
        best = (lo_g, rep)
    gain, rep = best
    doc = {
        "gain": float(gain),
        "report": rep.to_dict(),
        "targets": asdict(targets),
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "connectome": ident["connectome"],
        "n": ident["n"],
        "e": ident["e"],
        "key": key,
        "converged": converged,
        "history": history,
    }
    path = calib_path(settings, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    log.info("calibrate_gain: gain=%.4f converged=%s (%d runs) -> %s", gain, converged, len(history), path)
    return doc
