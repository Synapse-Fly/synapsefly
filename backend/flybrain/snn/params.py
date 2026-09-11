"""LIF parameters and exact per-step constants (SPEC section c.9).

Neuron model: Shiu et al. 2024 (Nature 634:210, doi:10.1038/s41586-024-07763-9) leaky
integrate-and-fire with a current-based exponential synapse, integrated with the EXACT 2-D
matrix-exponential update of RESEARCH section 7 ``[V]``::

    dv/dt = (v_rest - v + g + I_ext) / tau_m        dg/dt = -g / tau_s
    v <- v_rest + I + (v - v_rest - I) * a_m + b * g_old
    g <- g_old * a_s
    a_m = exp(-dt/tau_m); a_s = exp(-dt/tau_s); b = tau_s/(tau_m - tau_s) * (a_m - a_s)

Provenance of every number: ``[L]`` literature (Shiu 2024 defaults), ``[V]`` verified numerically in
RESEARCH section 7, ``[D]`` derived from proposal 1's reading of Shiu's Brian2 code, ``[E]`` engineered.
Only ``math`` and ``dataclasses`` are imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["LIFParams", "StepConstants", "current_from_rate", "rate_from_current"]


@dataclass(frozen=True)
class LIFParams:
    """Shiu et al. 2024 (Nature 634:210) defaults ``[L]``.

    ``v_rest = v_reset = -52 mV``, ``v_th = -45 mV`` (7 mV threshold gap), ``tau_m = 20 ms``,
    ``tau_s = 5 ms``, ``t_ref = 2.2 ms`` (absolute refractory, "unless refractory" semantics),
    ``delay = 1.8 ms`` (axonal/synaptic), ``w_syn = 0.275 mV`` per synapse (Shiu's single free
    parameter). ``f_poi = 250`` so that one Poisson forcing event adds ``f_poi * w_syn = 68.75 mV``
    to ``g`` and produces exactly one spike ``[D]`` (RESEARCH section 7).
    """

    v_rest: float = -52.0
    v_reset: float = -52.0
    v_th: float = -45.0
    tau_m: float = 20.0
    tau_s: float = 5.0
    t_ref: float = 2.2
    delay: float = 1.8
    w_syn: float = 0.275
    f_poi: float = 250.0

    def constants(self, dt_ms: float) -> "StepConstants":
        """Exact matrix-exponential update constants for a step of ``dt_ms`` (RESEARCH section 7 ``[V]``).

        ``dt 1.0 -> a_m 0.951229 / a_s 0.818731 / b 0.044166 / ref 2 / delay 2``;
        ``dt 0.5 -> 0.975310 / 0.904837 / 0.023491 / 4 / 4``; ``dt 0.1 -> 0.995012 / 0.980199 / 0.004938 / 22 / 18``.
        """
        if not (dt_ms > 0.0):
            raise ValueError(f"dt_ms must be > 0, got {dt_ms}")
        if self.tau_m == self.tau_s:
            raise ValueError("tau_m must differ from tau_s (matrix-exponential update is singular)")
        a_m = math.exp(-dt_ms / self.tau_m)
        a_s = math.exp(-dt_ms / self.tau_s)
        b = self.tau_s / (self.tau_m - self.tau_s) * (a_m - a_s)
        return StepConstants(
            dt_ms=float(dt_ms),
            a_m=a_m,
            a_s=a_s,
            b=b,
            ref_steps=int(round(self.t_ref / dt_ms)),
            delay_steps=int(round(self.delay / dt_ms)),
            kick_mv=self.f_poi * self.w_syn,
        )


@dataclass(frozen=True)
class StepConstants:
    """Per-step constants of the exact 2-D update (SPEC section c.9, RESEARCH section 7 ``[V]``).

    ``a_m = exp(-dt/tau_m)``; ``a_s = exp(-dt/tau_s)``; ``b = tau_s/(tau_m - tau_s) * (a_m - a_s)``;
    ``ref_steps = round(t_ref/dt)``; ``delay_steps = round(delay/dt)``; ``kick_mv = f_poi * w_syn`` (68.75 mV).
    """

    dt_ms: float
    a_m: float
    a_s: float
    b: float
    ref_steps: int
    delay_steps: int
    kick_mv: float


def current_from_rate(rate_hz: float, p: LIFParams = LIFParams()) -> float:
    """Constant current ``I`` (mV) such that an ISOLATED neuron with steady offset ``I`` fires at ``rate_hz``.

    ``I = (v_th - v_rest) / (1 - exp(-((1000/rate_hz) - t_ref)/tau_m))``; 100 Hz -> 21.68 mV ``[V]``
    (RESEARCH section 7: 10 Hz -> 7.05, 20 -> 7.71, 50 -> 11.88, 150 -> 34.97). ``rate_hz <= 0 -> 0.0``.
    Rates at or above ``1000/t_ref`` (no time to integrate) are clamped to an ISI of ``t_ref + 1e-9`` ms.
    """
    if not (rate_hz > 0.0):
        return 0.0
    isi = max(1000.0 / rate_hz - p.t_ref, 1e-9)
    return (p.v_th - p.v_rest) / (1.0 - math.exp(-isi / p.tau_m))


def rate_from_current(i_mv: float, p: LIFParams = LIFParams()) -> float:
    """Inverse of ``current_from_rate``: ``0.0`` when ``i_mv <= v_th - v_rest`` (never reaches threshold).

    ``ISI = t_ref - tau_m * ln(1 - (v_th - v_rest)/I)``; ``rate = 1000 / ISI``.
    """
    gap = p.v_th - p.v_rest
    if not (i_mv > gap):
        return 0.0
    isi = p.t_ref - p.tau_m * math.log(1.0 - gap / i_mv)
    return 1000.0 / isi
