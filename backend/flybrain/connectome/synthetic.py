"""Synthetic MaleCNS-shaped connectome generator (SPEC sections c.6 and g).

The generator is a **structured stand-in shaped like MaleCNS v1.0**, not data: every core neuron
carries a real MaleCNS ``type`` string (so every ``GROUP_REGEX`` of SPEC c.4 resolves on it), the
8-region taxonomy, a real ``superclass``/``class`` (so ``region_of()`` reproduces the region column),
a neurotransmitter and therefore a sign, and 112 explicit projection rules carrying the verified
MaleCNS synapse counts (RESEARCH section 5) with correct laterality, plus the 2 engineered rest brakes
of ``REST_BRAKES`` (114 rows in all, see below). Everything else is a subcritical random background.
``meta['note']`` says so and every consumer repeats it.

Provenance tags (SPEC 0.1): population counts ``[V]`` RESEARCH section 4 where the real population
is small, scaled down ``[E]`` for the large sensory/relay populations (R1-R6 3,377 -> 400, KC 4,064
-> 800, T4/T5 13,580 -> 480, ORN 2,635 -> 320); projection rows ``V`` = verified pathway (RESEARCH
section 5), ``D`` = literature/digest, ``E`` = engineered stand-in (listed in
``meta['engineered_edges']``); region fractions ``[V]`` RESEARCH 3.1; background degree/weight
distribution ``[D]`` (MaleCNS mean 4.85 synapses per connection); the steady-state weight formula
``[V]`` RESEARCH section 7.

Rest-state tuning (``[E]``, auditable): ``PROJECTIONS_SPEC`` is the SPEC g.3 table transcribed verbatim
(112 rows); ``PROJECTIONS`` (what ``build_synthetic`` wires) is that table with the ``REST_TUNING``
overrides applied and the ``REST_BRAKES`` rows appended - 114 rows, 21 ``E`` pids, where SPEC c.6 says
112 rows and SPEC g.5 "the 19 ``E`` pids": the 19 ``E`` rows of g.3 are shipped unchanged and the two
extra rows (and ``E`` pids) are the engineered rest brakes P113/P114, listed in
``meta['engineered_edges']`` - which is what the README/NOTICE ``E``-row list must be generated from.
SPEC g.6 makes the five behaviour gates the arbiter ("an engineer changing a target_mv re-runs pytest
tests/test_synthetic.py"); with the table exactly as printed it cannot pass gate 1 at any N - measured
under the mandated noise and the SPEC c.10 tonic table, N=4000 gives mean 12.8 Hz with optic_lobe
22.7 Hz and descending_motor 20.9 Hz (gate-1 band 1-5 Hz, region ceiling 12 Hz) and 160 giant-fibre spikes in 600 ms (133 Hz per GF cell), N=20000 gives
mean 9.5 Hz, optic_lobe 13.5 Hz, descending_motor 23.1 Hz and 472 GF spikes in 1000 ms. Every override is
recorded in ``meta['rest_tuning']`` with the SPEC value next to the value used, and every brake pid in
``meta['rest_brakes']`` / ``meta['engineered_edges']``.

Cold start (``[E]``, a limit of the engine, not of the graph): a fresh ``LIFEngine`` starts every neuron at
``v_rest`` with an empty delay ring, so for the first ~10 ms *nothing in the graph has fired* and no synaptic
brake exists yet - the only populations that fire that early are the ones the tonic table of SPEC c.10 holds
just below threshold. With that table applied (what ``SimulationLoop`` does at startup, SPEC c.27) ``P114``
closes the window and the giant fibre is silent through 1000 ms of rest for every seed tested. Without it
(``run_gates`` builds a bare engine, SPEC c.13) the GF still fires once at t = 22-35 ms on roughly one cold
start in ten: that needs a hyperpolarising ``gf`` entry in the c.10 tonic table or a warm-up in
``scripts/smoke.py``, neither of which this module can provide.

Only numpy and the standard library are imported at module level.
"""

from __future__ import annotations

import csv
import fnmatch
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .csr import sum_duplicates
from .groups import GROUP_REGEX, region_of, resolve_groups, validate_groups
from .loaders import NT_SIGN, nt_normalise
from .patches import apply_patches
from .schema import REGION_ID, REGIONS, SYNTHETIC_BODY_BASE, Connectome, count_groups, count_regions, utc_now_iso

__all__ = [
    "GENERATOR_VERSION",
    "Pop",
    "Proj",
    "POPULATIONS",
    "PROJECTIONS_SPEC",
    "PROJECTIONS",
    "REST_TUNING",
    "REST_BRAKES",
    "NOISE_FLOOR_HZ",
    "TONIC_REST_HZ",
    "tune_projections",
    "REGION_FRAC",
    "FILLER_NT",
    "REGION_ADJACENCY",
    "FILLER_SUPERCLASS",
    "EXPORT_FILLER_CLASS",
    "SYNTHETIC_NOTE",
    "SYNTHETIC_LICENSE",
    "CORE_TOTAL",
    "calibrated_weight",
    "scale_populations",
    "region_plan",
    "resolve_selector",
    "build_synthetic",
    "mean_signed_in_weight",
    "export_csv",
]

#: Version stamp of the graph this module produces (participates in ``cache.cache_key``; bump on any change).
GENERATOR_VERSION: str = "3"   # bumped: REST_BRAKES P113/P114 + scale_k on P012/P013 change the graph

SYNTHETIC_NOTE: str = "synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data"
SYNTHETIC_LICENSE: str = "synthetic (no data)"


# --------------------------------------------------------------------------- rule DSL (SPEC g.0)


@dataclass(frozen=True)
class Pop:
    """One core population: ``n`` is both sides combined; sides split ceil(n/2) L, floor(n/2) R (indices
    interleaved L,R,L,R,... so that ``rank_within_side`` is well defined); ``n == 1`` -> side 0.
    ``sign = NT_SIGN[nt]`` (SPEC c.7)."""

    type: str
    region: str
    n: int
    nt: str
    superclass: str
    cls: str = ""


@dataclass(frozen=True)
class Proj:
    """One projection rule (SPEC g.0). ``pid``: ``"P001"``..``"P112"`` are the SPEC g.3 rows, ``"P113"``/``"P114"`` the
    engineered rest brakes of ``REST_BRAKES``. ``src``/``dst``: exact ``GROUP_REGEX`` key, or a ``'|'``-separated list
    of type patterns matched with ``fnmatch`` against the unique type list (``*`` allowed). ``k_in``:
    presynaptic partners per postsynaptic cell (per allowed side; informative for ``all``/``wedge``/``glomerular``).
    ``r_nom_hz``: nominal presynaptic rate; ``target_mv``: steady-state ``g`` at ``r_nom`` (calibrated mode);
    ``laterality``: ipsi|contra|both; ``topology``: random|all|retinotopic|glomerular|wedge|ring_shift;
    ``w_lit``: literature mean synapses per pair (0 = engineered edge, calibrated value used in both modes);
    ``provenance``: V|D|E; ``w_cv``: lognormal jitter of the per-edge weight (both modes).

    ``scale_k`` is an E1 extension of the SPEC c.6 field list (which ends at ``w_cv``): when True the calibrated
    weight is computed from the *realised* mean in-degree of the row instead of the table's ``k_in``. SPEC g.0 states
    that for ``all`` topology ``k_in`` "equals the per-side source count, so the two agree by construction" - true
    only at ``s == 1``; below it the realised count shrinks with the population scale and the row silently delivers
    ``s * target_mv``. ``meta['projections'][pid]['k_eff']`` records the count actually used for every row."""

    pid: str
    src: str
    dst: str
    k_in: int
    r_nom_hz: float
    target_mv: float
    laterality: str
    topology: str
    w_lit: float
    provenance: str
    w_cv: float = 0.5
    scale_k: bool = False


LATERALITIES: tuple[str, ...] = ("ipsi", "contra", "both")
TOPOLOGIES: tuple[str, ...] = ("random", "all", "retinotopic", "glomerular", "wedge", "ring_shift")

# --------------------------------------------------------------------------- tables (SPEC g.1 / g.3, verbatim)

POPULATIONS: tuple[Pop, ...] = (
    Pop('R1-R6', 'optic_lobe', 400, 'histamine', 'ol_sensory', 'visual'),  # 1
    Pop('R7p', 'optic_lobe', 30, 'histamine', 'ol_sensory', 'visual'),  # 2
    Pop('R7y', 'optic_lobe', 30, 'histamine', 'ol_sensory', 'visual'),  # 3
    Pop('R8p', 'optic_lobe', 30, 'histamine', 'ol_sensory', 'visual'),  # 4
    Pop('R8y', 'optic_lobe', 30, 'histamine', 'ol_sensory', 'visual'),  # 5
    Pop('L1', 'optic_lobe', 120, 'glutamate', 'ol_intrinsic', ''),  # 6
    Pop('L2', 'optic_lobe', 120, 'acetylcholine', 'ol_intrinsic', ''),  # 7
    Pop('L3', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 8
    Pop('L4', 'optic_lobe', 40, 'acetylcholine', 'ol_intrinsic', ''),  # 9
    Pop('L5', 'optic_lobe', 40, 'acetylcholine', 'ol_intrinsic', ''),  # 10
    Pop('C2', 'optic_lobe', 40, 'gaba', 'ol_intrinsic', ''),  # 11
    Pop('C3', 'optic_lobe', 40, 'gaba', 'ol_intrinsic', ''),  # 12
    Pop('Mi1', 'optic_lobe', 120, 'acetylcholine', 'ol_intrinsic', ''),  # 13
    Pop('Tm3', 'optic_lobe', 120, 'acetylcholine', 'ol_intrinsic', ''),  # 14
    Pop('Mi4', 'optic_lobe', 60, 'gaba', 'ol_intrinsic', ''),  # 15
    Pop('Mi9', 'optic_lobe', 60, 'glutamate', 'ol_intrinsic', ''),  # 16
    Pop('Tm1', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 17
    Pop('Tm2', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 18
    Pop('Tm4', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 19
    Pop('Tm9', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 20
    Pop('T4a', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 21
    Pop('T4b', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 22
    Pop('T4c', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 23
    Pop('T4d', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 24
    Pop('T5a', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 25
    Pop('T5b', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 26
    Pop('T5c', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 27
    Pop('T5d', 'optic_lobe', 60, 'acetylcholine', 'ol_intrinsic', ''),  # 28
    Pop('LC4', 'optic_lobe', 126, 'acetylcholine', 'visual_projection', ''),  # 29
    Pop('LPLC2', 'optic_lobe', 185, 'acetylcholine', 'visual_projection', ''),  # 30
    Pop('LC6', 'optic_lobe', 124, 'acetylcholine', 'visual_projection', ''),  # 31
    Pop('LPLC1', 'optic_lobe', 134, 'acetylcholine', 'visual_projection', ''),  # 32
    Pop('LPLC4', 'optic_lobe', 97, 'acetylcholine', 'visual_projection', ''),  # 33
    Pop('LC9', 'optic_lobe', 219, 'acetylcholine', 'visual_projection', ''),  # 34
    Pop('LC31a', 'optic_lobe', 32, 'acetylcholine', 'visual_projection', ''),  # 35
    Pop('ORN_DM1', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 36
    Pop('ORN_DM4', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 37
    Pop('ORN_DM2', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 38
    Pop('ORN_VM2', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 39
    Pop('ORN_VA2', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 40
    Pop('ORN_DL1', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 41
    Pop('ORN_DA1', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 42
    Pop('ORN_DL3', 'antennal_lobe', 40, 'acetylcholine', 'cb_sensory', 'olfactory'),  # 43
    Pop('DM1_lPN', 'antennal_lobe', 4, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 44
    Pop('DM4_adPN', 'antennal_lobe', 4, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 45
    Pop('DM2_lPN', 'antennal_lobe', 4, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 46
    Pop('VM2_adPN', 'antennal_lobe', 4, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 47
    Pop('VA2_adPN', 'antennal_lobe', 4, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 48
    Pop('DL1_adPN', 'antennal_lobe', 4, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 49
    Pop('DA1_lPN', 'antennal_lobe', 8, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 50
    Pop('DL3_lPN', 'antennal_lobe', 4, 'acetylcholine', 'cb_intrinsic', 'ALPN'),  # 51
    Pop('lLN1_a', 'antennal_lobe', 20, 'gaba', 'cb_intrinsic', 'ALLN'),  # 52
    Pop('lLN2T_a', 'antennal_lobe', 20, 'gaba', 'cb_intrinsic', 'ALLN'),  # 53
    Pop('KCg-m', 'mushroom_body', 340, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 54
    Pop('KCg-d', 'mushroom_body', 40, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 55
    Pop('KCab-c', 'mushroom_body', 130, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 56
    Pop('KCab-m', 'mushroom_body', 90, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 57
    Pop('KCab-s', 'mushroom_body', 60, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 58
    Pop('KCab-p', 'mushroom_body', 20, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 59
    Pop("KCa'b'-ap1", 'mushroom_body', 40, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 60
    Pop("KCa'b'-ap2", 'mushroom_body', 40, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 61
    Pop("KCa'b'-m", 'mushroom_body', 40, 'acetylcholine', 'cb_intrinsic', 'Kenyon_Cell'),  # 62
    Pop('APL', 'mushroom_body', 2, 'gaba', 'cb_intrinsic', ''),  # 63
    Pop('DPM', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', ''),  # 64
    Pop('MBON01', 'mushroom_body', 2, 'glutamate', 'cb_intrinsic', 'MBON'),  # 65
    Pop('MBON03', 'mushroom_body', 2, 'glutamate', 'cb_intrinsic', 'MBON'),  # 66
    Pop('MBON04', 'mushroom_body', 2, 'glutamate', 'cb_intrinsic', 'MBON'),  # 67
    Pop('MBON05', 'mushroom_body', 2, 'glutamate', 'cb_intrinsic', 'MBON'),  # 68
    Pop('MBON06', 'mushroom_body', 2, 'glutamate', 'cb_intrinsic', 'MBON'),  # 69
    Pop('MBON09', 'mushroom_body', 2, 'gaba', 'cb_intrinsic', 'MBON'),  # 70
    Pop('MBON11', 'mushroom_body', 2, 'gaba', 'cb_intrinsic', 'MBON'),  # 71
    Pop('MBON12', 'mushroom_body', 4, 'acetylcholine', 'cb_intrinsic', 'MBON'),  # 72
    Pop('MBON13', 'mushroom_body', 2, 'acetylcholine', 'cb_intrinsic', 'MBON'),  # 73
    Pop('MBON14', 'mushroom_body', 2, 'acetylcholine', 'cb_intrinsic', 'MBON'),  # 74
    Pop('MBON18', 'mushroom_body', 2, 'acetylcholine', 'cb_intrinsic', 'MBON'),  # 75
    Pop('PAM01', 'mushroom_body', 44, 'dopamine', 'cb_intrinsic', 'DAN'),  # 76
    Pop('PAM02', 'mushroom_body', 26, 'dopamine', 'cb_intrinsic', 'DAN'),  # 77
    Pop('PAM03', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 78
    Pop('PAM04', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 79
    Pop('PAM05', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 80
    Pop('PAM06', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 81
    Pop('PAM07', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 82
    Pop('PAM08', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 83
    Pop('PAM09', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 84
    Pop('PAM10', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 85
    Pop('PAM11', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 86
    Pop('PAM12', 'mushroom_body', 20, 'dopamine', 'cb_intrinsic', 'DAN'),  # 87
    Pop('PAM13', 'mushroom_body', 16, 'dopamine', 'cb_intrinsic', 'DAN'),  # 88
    Pop('PAM14', 'mushroom_body', 16, 'dopamine', 'cb_intrinsic', 'DAN'),  # 89
    Pop('PAM15', 'mushroom_body', 14, 'dopamine', 'cb_intrinsic', 'DAN'),  # 90
    Pop('PPL101', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 91
    Pop('PPL102', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 92
    Pop('PPL103', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 93
    Pop('PPL104', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 94
    Pop('PPL105', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 95
    Pop('PPL106', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 96
    Pop('PPL107', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 97
    Pop('PPL108', 'mushroom_body', 2, 'dopamine', 'cb_intrinsic', 'DAN'),  # 98
    Pop('EPG', 'central_complex', 46, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 99
    Pop('PEN_a(PEN1)', 'central_complex', 20, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 100
    Pop('PEN_b(PEN2)', 'central_complex', 22, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 101
    Pop('PEG', 'central_complex', 18, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 102
    Pop('Delta7', 'central_complex', 42, 'glutamate', 'cb_intrinsic', 'CX'),  # 103
    Pop('ER4d', 'central_complex', 26, 'gaba', 'cb_intrinsic', 'CX'),  # 104
    Pop('ER4m', 'central_complex', 18, 'gaba', 'cb_intrinsic', 'CX'),  # 105
    Pop('ER2_a', 'central_complex', 12, 'gaba', 'cb_intrinsic', 'CX'),  # 106
    Pop('PFL3', 'central_complex', 24, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 107
    Pop('PFL2', 'central_complex', 12, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 108
    Pop('PFL1', 'central_complex', 14, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 109
    Pop('hDeltaB', 'central_complex', 19, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 110
    Pop('PFNd', 'central_complex', 40, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 111
    Pop('PFNv', 'central_complex', 20, 'acetylcholine', 'cb_intrinsic', 'CX'),  # 112
    Pop('ExR1', 'central_complex', 4, 'gaba', 'cb_intrinsic', 'CX'),  # 113
    Pop('ExR2', 'central_complex', 4, 'gaba', 'cb_intrinsic', 'CX'),  # 114
    Pop('FB6A', 'central_complex', 10, 'glutamate', 'cb_intrinsic', 'CX'),  # 115
    Pop('FB6H', 'central_complex', 10, 'glutamate', 'cb_intrinsic', 'CX'),  # 116
    Pop('FB7A', 'central_complex', 10, 'glutamate', 'cb_intrinsic', 'CX'),  # 117
    Pop('FB7B', 'central_complex', 10, 'glutamate', 'cb_intrinsic', 'CX'),  # 118
    Pop('LB3b', 'sez', 11, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 119
    Pop('LB3c', 'sez', 23, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 120
    Pop('PhG1a', 'sez', 2, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 121
    Pop('PhG1b', 'sez', 2, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 122
    Pop('PhG1c', 'sez', 4, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 123
    Pop('LB3a', 'sez', 17, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 124
    Pop('LB3d', 'sez', 26, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 125
    Pop('LB1a', 'sez', 11, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 126
    Pop('LB1b', 'sez', 6, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 127
    Pop('LB1c', 'sez', 16, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 128
    Pop('LB1d', 'sez', 5, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 129
    Pop('LB1e', 'sez', 19, 'acetylcholine', 'cb_sensory', 'gustatory'),  # 130
    Pop('GNG215', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 131
    Pop('GNG232', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 132
    Pop('GNG132', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 133
    Pop('GNG089', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 134
    Pop('PRW046', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 135
    Pop('PRW047', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 136
    Pop('GNG042', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 137
    Pop('GNG038', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 138
    Pop('GNG551', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 139
    Pop('GNG108', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 140
    Pop('GNG120', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 141
    Pop('GNG117', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 142
    Pop('GNG234', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 143
    Pop('GNG015', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 144
    Pop('GNG095', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 145
    Pop('GNG130', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 146
    Pop('GNG180', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 147
    Pop('GNG184', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 148
    Pop('GNG016', 'sez', 2, 'unknown', 'cb_intrinsic', ''),  # 149
    Pop('GNG087', 'sez', 3, 'glutamate', 'cb_intrinsic', ''),  # 150
    Pop('GNG592', 'sez', 2, 'glutamate', 'cb_intrinsic', ''),  # 151
    Pop('MN9', 'sez', 2, 'acetylcholine', 'cb_motor', ''),  # 152
    Pop('MN1', 'sez', 2, 'acetylcholine', 'cb_motor', ''),  # 153
    Pop('MN6', 'sez', 2, 'acetylcholine', 'cb_motor', ''),  # 154
    Pop('SAD093', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 155
    Pop('GNG458', 'sez', 2, 'gaba', 'cb_intrinsic', ''),  # 156
    Pop('GNG532', 'sez', 2, 'acetylcholine', 'cb_intrinsic', ''),  # 157
    Pop('pC1_1a', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 158
    Pop('pC1_2a', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 159
    Pop('pC1_4a', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 160
    Pop('pC1_7b', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 161
    Pop('pC1_10a', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 162
    Pop('pC1_12a', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 163
    Pop('pC1_14a', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 164
    Pop('pC1_16a', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 165
    Pop('pC1_18b', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 166
    Pop('pC1_19', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 167
    Pop('mAL_m8', 'central_other', 16, 'gaba', 'cb_intrinsic', ''),  # 168
    Pop('mAL_m1', 'central_other', 12, 'gaba', 'cb_intrinsic', ''),  # 169
    Pop('aIPg7', 'central_other', 6, 'acetylcholine', 'cb_intrinsic', ''),  # 170
    Pop('AVLP732m', 'central_other', 4, 'acetylcholine', 'cb_intrinsic', ''),  # 171
    Pop('AVLP733m', 'central_other', 4, 'acetylcholine', 'cb_intrinsic', ''),  # 172
    Pop('JO-A1', 'central_other', 60, 'acetylcholine', 'cb_sensory', 'mechanosensory'),  # 173
    Pop('JO-A2', 'central_other', 40, 'acetylcholine', 'cb_sensory', 'mechanosensory'),  # 174
    Pop('JO-CM', 'central_other', 40, 'acetylcholine', 'cb_sensory', 'mechanosensory'),  # 175
    Pop('JO-EV1', 'central_other', 30, 'acetylcholine', 'cb_sensory', 'mechanosensory'),  # 176
    Pop('JO-FV', 'central_other', 30, 'acetylcholine', 'cb_sensory', 'mechanosensory'),  # 177
    Pop('BM', 'central_other', 9, 'acetylcholine', 'cb_sensory', 'mechanosensory_tactile'),  # 178
    Pop('LAL083', 'central_other', 4, 'glutamate', 'cb_intrinsic', ''),  # 179
    Pop('LAL126', 'central_other', 4, 'glutamate', 'cb_intrinsic', ''),  # 180
    Pop('LAL179', 'central_other', 4, 'acetylcholine', 'cb_intrinsic', ''),  # 181
    Pop('PS049', 'central_other', 4, 'gaba', 'cb_intrinsic', ''),  # 182
    Pop('PS059', 'central_other', 4, 'gaba', 'cb_intrinsic', ''),  # 183
    Pop('VES051', 'central_other', 4, 'glutamate', 'cb_intrinsic', ''),  # 184
    Pop('VES052', 'central_other', 4, 'glutamate', 'cb_intrinsic', ''),  # 185
    Pop('AOTU015', 'central_other', 4, 'acetylcholine', 'cb_intrinsic', ''),  # 186
    Pop('AOTU019', 'central_other', 4, 'gaba', 'cb_intrinsic', ''),  # 187
    Pop('PVLP020', 'central_other', 4, 'gaba', 'cb_intrinsic', ''),  # 188
    Pop('DNp01', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 189
    Pop('DNp02', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 190
    Pop('DNp04', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 191
    Pop('DNp11', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 192
    Pop('DNp03', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 193
    Pop('DNp07', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 194
    Pop('DNp10', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 195
    Pop('DNp09', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 196
    Pop('DNg100', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 197
    Pop('DNge053', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 198
    Pop('DNg97', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 199
    Pop('MDN', 'descending_motor', 4, 'acetylcholine', 'descending_neuron', ''),  # 200
    Pop('DNg60', 'descending_motor', 2, 'gaba', 'descending_neuron', ''),  # 201
    Pop('DNg74_a', 'descending_motor', 2, 'gaba', 'descending_neuron', ''),  # 202
    Pop('DNg74_b', 'descending_motor', 2, 'gaba', 'descending_neuron', ''),  # 203
    Pop('DNa01', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 204
    Pop('DNa02', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 205
    Pop('DNa03', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 206
    Pop('DNb01', 'descending_motor', 2, 'glutamate', 'descending_neuron', ''),  # 207
    Pop('DNg13', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 208
    Pop('DNg02_a', 'descending_motor', 10, 'acetylcholine', 'descending_neuron', ''),  # 209
    Pop('DNg02_b', 'descending_motor', 5, 'acetylcholine', 'descending_neuron', ''),  # 210
    Pop('DNg02_c', 'descending_motor', 4, 'acetylcholine', 'descending_neuron', ''),  # 211
    Pop('DNg02_d', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 212
    Pop('DNg02_e', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 213
    Pop('DNg02_f', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 214
    Pop('DNg02_g', 'descending_motor', 4, 'acetylcholine', 'descending_neuron', ''),  # 215
    Pop('DNg62', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 216
    Pop('DNge078', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 217
    Pop('DNge062', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 218
    Pop('DNge080', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 219
    Pop('DNg67', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 220
    Pop('pIP10', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 221
    Pop('pMP2', 'descending_motor', 2, 'acetylcholine', 'descending_neuron', ''),  # 222
    Pop('DNge129', 'descending_motor', 2, 'gaba', 'descending_neuron', ''),  # 223
    Pop('TTMn', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 224
    Pop('PSI', 'descending_motor', 2, 'unknown', 'vnc_efferent', ''),  # 225
    Pop('DLMn a, b', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 226
    Pop('DLMn c-f', 'descending_motor', 8, 'glutamate', 'vnc_motor', ''),  # 227
    Pop('DVMn 1a-c', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 228
    Pop('DVMn 2a, b', 'descending_motor', 4, 'glutamate', 'vnc_motor', ''),  # 229
    Pop('DVMn 3a, b', 'descending_motor', 4, 'glutamate', 'vnc_motor', ''),  # 230
    Pop('b1 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 231
    Pop('b2 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 232
    Pop('b3 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 233
    Pop('i1 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 234
    Pop('i2 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 235
    Pop('iii1 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 236
    Pop('iii3 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 237
    Pop('hg1 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 238
    Pop('hg2 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 239
    Pop('hg3 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 240
    Pop('hg4 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 241
    Pop('tp1 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 242
    Pop('tp2 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 243
    Pop('tpn MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 244
    Pop('ps1 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 245
    Pop('ps2 MN', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 246
    Pop('MNwm35', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 247
    Pop('MNwm36', 'descending_motor', 2, 'glutamate', 'vnc_motor', ''),  # 248
    Pop('Ti flexor MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 249
    Pop('Ti extensor MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 250
    Pop('Tr flexor MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 251
    Pop('Tr extensor MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 252
    Pop('Fe reductor MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 253
    Pop('Ta depressor MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 254
    Pop('Ta levator MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 255
    Pop('Sternotrochanter MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 256
    Pop('Sternal anterior rotator MN', 'descending_motor', 12, 'glutamate', 'vnc_motor', ''),  # 257
    Pop('Sternal posterior rotator MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 258
    Pop('Pleural remotor/abductor MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 259
    Pop('ltm MN', 'descending_motor', 6, 'glutamate', 'vnc_motor', ''),  # 260
    Pop('GFC2', 'vnc', 10, 'acetylcholine', 'vnc_intrinsic', ''),  # 261
    Pop('dPR1', 'vnc', 2, 'acetylcholine', 'vnc_intrinsic', ''),  # 262
    Pop('dMS2', 'vnc', 20, 'acetylcholine', 'vnc_intrinsic', ''),  # 263
    Pop('vPR6', 'vnc', 8, 'acetylcholine', 'vnc_intrinsic', ''),  # 264
    Pop('vPR9_a', 'vnc', 3, 'gaba', 'vnc_intrinsic', ''),  # 265
    Pop('vPR9_b', 'vnc', 3, 'gaba', 'vnc_intrinsic', ''),  # 266
    Pop('vPR9_c', 'vnc', 3, 'gaba', 'vnc_intrinsic', ''),  # 267
    Pop('AN13B002', 'vnc', 2, 'gaba', 'ascending_neuron', ''),  # 268
    Pop('AN05B023d', 'vnc', 2, 'gaba', 'ascending_neuron', ''),  # 269
    Pop('AN03A008', 'vnc', 2, 'acetylcholine', 'ascending_neuron', ''),  # 270
    Pop('AN04B003', 'vnc', 2, 'acetylcholine', 'ascending_neuron', ''),  # 271
    Pop('AN08B020', 'vnc', 2, 'acetylcholine', 'ascending_neuron', ''),  # 272
    Pop('AN05B102a', 'vnc', 4, 'acetylcholine', 'ascending_neuron', ''),  # 273
    Pop('IN13A001', 'vnc', 6, 'gaba', 'vnc_intrinsic', ''),  # 274
    Pop('IN13A022', 'vnc', 6, 'gaba', 'vnc_intrinsic', ''),  # 275
    Pop('IN08A002', 'vnc', 6, 'glutamate', 'vnc_intrinsic', ''),  # 276
    Pop('IN19A016', 'vnc', 6, 'gaba', 'vnc_intrinsic', ''),  # 277
    Pop('IN07B010', 'vnc', 6, 'acetylcholine', 'vnc_intrinsic', ''),  # 278
    Pop('IN19B043', 'vnc', 6, 'acetylcholine', 'vnc_intrinsic', ''),  # 279
    Pop('IN03B015', 'vnc', 6, 'gaba', 'vnc_intrinsic', ''),  # 280
    Pop('IN12B003', 'vnc', 6, 'gaba', 'vnc_intrinsic', ''),  # 281
    Pop('IN21A026', 'vnc', 6, 'glutamate', 'vnc_intrinsic', ''),  # 282
    Pop('LgLG3', 'vnc', 162, 'acetylcholine', 'vnc_sensory', 'gustatory'),  # 283
    Pop('LgLG4', 'vnc', 43, 'acetylcholine', 'vnc_sensory', 'gustatory'),  # 284
    Pop('WG2', 'vnc', 97, 'acetylcholine', 'vnc_sensory', 'gustatory'),  # 285
    Pop('LgAG1', 'vnc', 25, 'acetylcholine', 'vnc_sensory', 'gustatory'),  # 286
    Pop('LgLG1a', 'vnc', 136, 'acetylcholine', 'vnc_sensory', 'gustatory'),  # 287
    Pop('LgLG1b', 'vnc', 134, 'acetylcholine', 'vnc_sensory', 'gustatory'),  # 288
    Pop('LgLG2', 'vnc', 130, 'acetylcholine', 'vnc_sensory', 'gustatory'),  # 289
)

#: SPEC g.3 transcribed verbatim (112 rows; ``test_synthetic::test_projection_table_matches_spec`` parses the SPEC table).
PROJECTIONS_SPEC: tuple[Proj, ...] = (
    Proj('P001', 'R1-R6', 'L1', 6, 30, 8, 'ipsi', 'retinotopic', 27, 'V'),  # w_cal 32: histaminergic, inhibitory on L1
    Proj('P002', 'R1-R6', 'L2|L3', 6, 30, 8, 'ipsi', 'retinotopic', 27, 'V'),  # w_cal 32: L3 real mean is 5
    Proj('P003', 'L1', 'Mi1|Tm3', 2, 15, 8, 'ipsi', 'retinotopic', 65, 'V'),  # w_cal 194: Tm3 real mean is 12
    Proj('P004', 'L1', 'L5|C2|C3', 2, 15, 6, 'ipsi', 'retinotopic', 45, 'V'),  # w_cal 145: L1->L5 54.5, C3 44.7, C2 18.4
    Proj('P005', 'L2|L3', 'Tm1|Tm2|Tm4|Tm9', 2, 15, 8, 'ipsi', 'retinotopic', 30, 'D'),  # w_cal 194: OFF pathway relay
    Proj('P006', 'Mi1|Tm3', 'T4a|T4b|T4c|T4d', 7, 15, 8, 'ipsi', 'retinotopic', 10, 'V'),  # w_cal 55: Mi1->T4a 10.1 (7 Mi1 per T4a)
    Proj('P007', 'Mi4|Mi9', 'T4a|T4b|T4c|T4d', 4, 15, 3, 'ipsi', 'retinotopic', 4.5, 'V'),  # w_cal 36: inhibitory flanks (GABA / Glu)
    Proj('P008', 'Tm1|Tm2|Tm4|Tm9', 'T5a|T5b|T5c|T5d', 6, 15, 8, 'ipsi', 'retinotopic', 5, 'V'),  # w_cal 65: Tm9->T5a 4.8
    Proj('P009', 'T4a|T4b|T4c|T4d', 'LPLC2', 8, 20, 6, 'ipsi', 'retinotopic', 3.2, 'V'),  # w_cal 27: 1,483 T4a -> 184 LPLC2
    Proj('P010', 'T5a|T5b|T5c|T5d', 'LPLC2|LC4', 8, 20, 6, 'ipsi', 'retinotopic', 3.5, 'V'),  # w_cal 27: T5a->LC4 1.5
    Proj('P011', 'T4a|T4b|T4c|T4d|T5a|T5b|T5c|T5d', 'LC6|LPLC1|LPLC4|LC9|LC31a', 8, 20, 6, 'ipsi', 'retinotopic', 3, 'D'),  # w_cal 27: auxiliary + freezing LCs
    Proj('P012', 'LC4', 'DNp01', 63, 33, 7, 'ipsi', 'all', 50.5, 'V'),  # w_cal 2: every ipsi LC4 contacts the GF; 7 mV = half the 14 mV GF budget
    Proj('P013', 'LPLC2', 'DNp01', 92, 33, 7, 'ipsi', 'all', 26.3, 'V'),  # w_cal 2: other half of the GF budget
    Proj('P014', 'LC4', 'DNp04', 63, 33, 14, 'ipsi', 'all', 92, 'V'),  # w_cal 5: strongest LC4 target
    Proj('P015', 'LC4', 'DNp02|DNp11', 63, 33, 14, 'ipsi', 'all', 32, 'V'),  # w_cal 5: 33.7 / 30.0
    Proj('P016', 'LPLC2', 'DNp04', 92, 33, 6, 'ipsi', 'all', 18.4, 'V'),  # w_cal 1: 
    Proj('P017', 'DNp11', 'DNp01', 1, 50, 6, 'contra', 'all', 108, 'V'),  # w_cal 87: contralateral, 118 syn total
    Proj('P018', 'DNp01', 'GFC2', 1, 50, 14, 'ipsi', 'all', 12, 'V'),  # w_cal 204: GFC2 = 10 cells (5 per side)
    Proj('P019', 'GFC2', 'TTMn', 5, 50, 14, 'ipsi', 'all', 40, 'V'),  # w_cal 41: 
    Proj('P020', 'DNp01', 'TTMn|PSI', 1, 50, 14, 'ipsi', 'all', 25, 'V'),  # w_cal 204: chemical only (45 / 4); electrical proxies added by patches.py
    Proj('P021', 'PSI', 'DLMn c-f', 1, 50, 14, 'contra', 'all', 50.8, 'V'),  # w_cal 204: PSI crosses the midline
    Proj('P022', 'PSI', 'DLMn a, b|DVMn 3a, b', 1, 50, 10, 'both', 'all', 17, 'V'),  # w_cal 145: 21.5 ipsi / 12.8 contra
    Proj('P023', 'LC9|LC31a', 'DNp09', 125, 40, 14, 'ipsi', 'all', 17, 'V'),  # w_cal 2: DNp09's #1/#2 inputs (16.2 / 23.8)
    Proj('P024', 'PVLP020', 'DNp09', 2, 10, 3, 'ipsi', 'all', 148, 'V'),  # w_cal 109: GABA brake on DNp09
    Proj('P025', 'JO-A1|JO-A2', 'DNp01', 11, 30, 6, 'both', 'random', 48, 'V'),  # w_cal 13: auditory input to the GF (531 syn)
    Proj('P026', 'DNp02|DNp04|DNp11', 'TTMn', 3, 50, 14, 'ipsi', 'all', 0, 'E'),  # w_cal 68: takeoff DNs -> jump MN (no direct chemical path verified)
    Proj('P027', 'IN13A022|IN21A026', 'TTMn', 6, 5, 3, 'ipsi', 'all', 100, 'V'),  # w_cal 73: tonic inhibition of TTMn (987 GABA / 507 Glu)
    Proj('P028', 'DNp07|DNp10', 'Ti extensor MN|Tr extensor MN', 2, 40, 6, 'ipsi', 'random', 0, 'E'),  # w_cal 55: landing leg extension stand-in
    Proj('P029', 'PFL3', 'DNa02', 12, 20, 10, 'contra', 'all', 30.7, 'V'),  # w_cal 30: L->R 380, R->L 356; CONTRALATERAL
    Proj('P030', 'PFL3', 'DNa03|DNb01', 12, 20, 8, 'contra', 'all', 25, 'V'),  # w_cal 24: 19.5 / 31.4, contralateral
    Proj('P031', 'DNa03', 'DNa02', 1, 30, 6, 'ipsi', 'all', 276, 'V'),  # w_cal 145: 
    Proj('P032', 'DNa02', 'Sternal anterior rotator MN', 1, 30, 14, 'ipsi', 'all', 64.7, 'V'),  # w_cal 339: 776 syn ipsi
    Proj('P033', 'AN03A008|AN04B003', 'DNa02', 2, 10, 6, 'ipsi', 'all', 555, 'V'),  # w_cal 218: ascending steering input
    Proj('P034', 'PS049|PS059', 'DNa02', 4, 10, 4, 'ipsi', 'all', 252, 'V'),  # w_cal 73: GABA
    Proj('P035', 'LAL083|LAL126|VES051|VES052', 'DNa02', 8, 10, 4, 'both', 'all', 158, 'V'),  # w_cal 36: Glu; LAL contra, VES ipsi
    Proj('P036', 'LAL179|AOTU015|AOTU019', 'DNa02', 6, 10, 4, 'both', 'all', 145, 'V'),  # w_cal 48: ACh contra / ACh ipsi / GABA contra
    Proj('P037', 'DNg97|GNG532|LAL083', 'DNg13', 3, 20, 6, 'both', 'all', 125, 'V'),  # w_cal 73: 236 contra / 313 / 200
    Proj('P038', 'GNG458|DNge129', 'DNg100', 2, 5, 3, 'ipsi', 'all', 988, 'V'),  # w_cal 218: GABA brake; no verified excitatory brain input to DNg100
    Proj('P039', 'DNg100', 'IN13A001', 1, 30, 6, 'contra', 'all', 36, 'V'),  # w_cal 145: 
    Proj('P040', 'DNg100|DNge053|DNg97', 'IN07B010|IN19B043', 3, 30, 14, 'ipsi', 'random', 0, 'E'),  # w_cal 113: excitatory walking premotor pool
    Proj('P041', 'IN07B010|IN19B043', 'Ti flexor MN|Ti extensor MN|Tr flexor MN|Tr extensor MN|Fe reductor MN|Ta depressor MN|Ta levator MN|Sternotrochanter MN|ltm MN', 6, 40, 14, 'ipsi', 'random', 0, 'E'),  # w_cal 42: walking premotor -> leg MNs
    Proj('P042', 'IN13A001|IN03B015|IN12B003|IN19A016', 'Ti flexor MN|Ti extensor MN|Tr flexor MN|Tr extensor MN|Fe reductor MN|Ta depressor MN|Ta levator MN|Sternotrochanter MN|ltm MN', 6, 10, 3, 'ipsi', 'random', 0, 'E'),  # w_cal 36: inhibitory leg premotor
    Proj('P043', 'MDN', 'IN07B010|IN03B015|IN12B003', 2, 30, 10, 'ipsi', 'random', 466, 'V'),  # w_cal 121: backward walking (476 / 462 / 461)
    Proj('P044', 'DNg60|DNg74_a|DNg74_b', 'Sternal posterior rotator MN|Pleural remotor/abductor MN|Tr flexor MN|Sternotrochanter MN', 3, 30, 8, 'ipsi', 'all', 280, 'V'),  # w_cal 65: GABAergic halt DNs inhibit leg MNs directly
    Proj('P045', 'GNG232', 'DNg60', 1, 40, 8, 'ipsi', 'all', 0, 'E'),  # w_cal 145: feeding halts walking
    Proj('P046', 'MBON12|MBON13|MBON14|MBON18', 'AOTU015|LAL179', 4, 20, 8, 'both', 'random', 0, 'E'),  # w_cal 73: approach valence -> steering excitation
    Proj('P047', 'MBON01|MBON03|MBON05|MBON06', 'DNg100|DNge053', 4, 20, 6, 'both', 'random', 0, 'E'),  # w_cal 55: avoid valence (Glu) suppresses forward DNs
    Proj('P048', 'EPG', 'PFL3|PFL1|PFL2', 2, 20, 8, 'ipsi', 'wedge', 25, 'V'),  # w_cal 145: 968 ipsi of 1,182
    Proj('P049', 'Delta7', 'PFL3', 8, 10, 4, 'both', 'random', 29, 'D'),  # w_cal 36: Glu; 5,617 total
    Proj('P050', 'PFNd|PFNv', 'hDeltaB', 6, 15, 8, 'both', 'random', 15, 'D'),  # w_cal 65: 
    Proj('P051', 'hDeltaB', 'PFL3|PFL2|PFL1', 4, 15, 6, 'both', 'random', 10, 'D'),  # w_cal 73: 
    Proj('P052', 'DNg02_a|DNg02_b|DNg02_c|DNg02_d|DNg02_e|DNg02_f|DNg02_g', 'MNwm36', 14, 30, 14, 'ipsi', 'all', 84, 'V'),  # w_cal 24: 2,349 pooled
    Proj('P053', 'DNg02_a|DNg02_b|DNg02_c|DNg02_d|DNg02_e|DNg02_f|DNg02_g', 'tp2 MN|ps1 MN', 14, 30, 10, 'ipsi', 'all', 38, 'V'),  # w_cal 17: 1,362 / 759
    Proj('P054', 'DNg02_a|DNg02_b|DNg02_c|DNg02_d|DNg02_e|DNg02_f|DNg02_g', 'DVMn 1a-c|DVMn 3a, b', 14, 30, 8, 'both', 'all', 12, 'V'),  # w_cal 14: 436 / 326
    Proj('P055', 'DNg02_a|DNg02_b|DNg02_c|DNg02_d|DNg02_e|DNg02_f|DNg02_g', 'DLMn c-f|DLMn a, b', 14, 30, 4, 'both', 'all', 2.8, 'V'),  # w_cal 7: weak direct power-MN drive
    Proj('P056', 'DNg02_a|DNg02_b|DNg02_c|DNg02_d|DNg02_e|DNg02_f|DNg02_g', 'b1 MN|i1 MN|b2 MN|i2 MN', 14, 30, 6, 'ipsi', 'all', 0, 'E'),  # w_cal 10: flight context for steering MNs
    Proj('P057', 'DNp03', 'i1 MN', 1, 40, 10, 'contra', 'all', 37, 'V'),  # w_cal 182: 75 syn contra
    Proj('P058', 'DNp03', 'b1 MN|hg1 MN', 1, 40, 8, 'contra', 'all', 0, 'E'),  # w_cal 145: saccade stand-in (DNp03 -> b1 MN absent in v1.0)
    Proj('P059', 'LB3b|LB3c|PhG1a|PhG1b|PhG1c', 'GNG215|GNG232|GNG132|GNG089', 21, 100, 14, 'ipsi', 'all', 40, 'V'),  # w_cal 5: 741 / 495 / 426 / 196 pooled over ~16 pairs
    Proj('P060', 'LB3b|LB3c|PhG1a|PhG1b|PhG1c', 'GNG038|GNG042', 21, 100, 14, 'ipsi', 'all', 54, 'V'),  # w_cal 5: GABA 2nd order (1,361 / 902)
    Proj('P061', 'LB3b|LB3c|PhG1a|PhG1b|PhG1c', 'AN13B002|DNg67|DNge062', 21, 100, 8, 'ipsi', 'all', 22, 'V'),  # w_cal 3: AN13B002 464, DNg67 195; DNge062 hop is [E]
    Proj('P062', 'LgLG3|LgLG4|WG2', 'AN13B002|AN05B023d', 151, 60, 14, 'ipsi', 'all', 58, 'V'),  # w_cal 1: 8,770 + 6,936 + 7,964
    Proj('P063', 'LgLG3|WG2', 'PRW046|PRW047', 130, 60, 10, 'ipsi', 'all', 11, 'V'),  # w_cal 1: 1,495 / 1,522
    Proj('P064', 'GNG215|GNG232', 'GNG108|DNge080', 2, 60, 14, 'ipsi', 'all', 435, 'V'),  # w_cal 85: 588 / 283 / 283
    Proj('P065', 'GNG089', 'GNG108|GNG120', 1, 60, 8, 'ipsi', 'all', 80, 'V'),  # w_cal 97: 112 / 49
    Proj('P066', 'GNG215|GNG232|PRW046|PRW047', 'GNG117|GNG234', 4, 60, 14, 'both', 'all', 0, 'E'),  # w_cal 42: closes the loop onto the two remaining verified MN9 exciters
    Proj('P067', 'GNG108|GNG120|GNG117|GNG234|DNge080|DNge062', 'MN9', 6, 60, 14, 'ipsi', 'all', 376, 'V'),  # w_cal 28: 373 / 359 / 410 / 362 / 197 / 556
    Proj('P068', 'GNG042', 'GNG015', 1, 60, 14, 'ipsi', 'all', 595, 'V'),  # w_cal 170: disinhibition hop 1
    Proj('P069', 'GNG015|GNG095|GNG130|GNG180|GNG184', 'MN9', 5, 10, 3, 'ipsi', 'all', 372, 'V'),  # w_cal 44: tonic GABA onto MN9 (10 Hz tonic table)
    Proj('P070', 'GNG132', 'GNG130', 1, 60, 8, 'ipsi', 'all', 368, 'V'),  # w_cal 97: inhibitory branch
    Proj('P071', 'GNG038', 'GNG095|GNG180', 1, 60, 8, 'ipsi', 'all', 0, 'E'),  # w_cal 97: second disinhibition branch
    Proj('P072', 'LB1a|LB1b|LB1c|LB1d|LB1e|LgAG1', 'GNG016', 41, 100, 14, 'ipsi', 'all', 122, 'V'),  # w_cal 2: 4,990 syn
    Proj('P073', 'LB1a|LB1b|LB1c|LB1d|LB1e', 'GNG087|GNG592', 28, 100, 14, 'ipsi', 'all', 63, 'V'),  # w_cal 4: Glu 2nd order (2,148 / 1,365)
    Proj('P074', 'GNG016', 'LB1c|LB1e', 1, 40, 3, 'ipsi', 'all', 48, 'V'),  # w_cal 55: feedback onto bitter GRNs (NT unclear -> +1)
    Proj('P075', 'GNG087|GNG592', 'GNG108|GNG120|GNG117|GNG234', 2, 40, 8, 'both', 'all', 0, 'E'),  # w_cal 73: bitter suppression of feeding premotor (no 2-hop bitter->MN9 path in v1.0)
    Proj('P076', 'GNG016', 'PPL101|PPL102|PPL103', 1, 40, 8, 'both', 'all', 0, 'E'),  # w_cal 145: bitter -> punishment dopamine
    Proj('P077', 'GNG215|GNG232|PRW046|PRW047', 'PAM01|PAM02|PAM03', 4, 60, 8, 'both', 'random', 0, 'E'),  # w_cal 24: sugar -> reward dopamine
    Proj('P078', 'LB3a', 'GNG215|GNG132', 8, 60, 6, 'ipsi', 'all', 0, 'E'),  # w_cal 9: water GRNs share the sweet 2nd-order layer (stand-in)
    Proj('P079', 'ORN_*', '*PN', 20, 20, 14, 'ipsi', 'glomerular', 92, 'V'),  # w_cal 25: ORN_DM1 -> DM1_lPN: 73 -> 2 cells, 13,384 syn
    Proj('P080', 'ORN_*', 'lLN1_a|lLN2T_a', 20, 20, 8, 'ipsi', 'random', 5, 'D'),  # w_cal 15: 
    Proj('P081', 'lLN1_a|lLN2T_a', '*PN', 10, 10, 3, 'ipsi', 'random', 10, 'D'),  # w_cal 22: GABA gain control
    Proj('P082', '*PN', 'KC*', 6, 20, 14, 'both', 'random', 21, 'V'),  # w_cal 85: DM1_lPN -> KC mean 21.1; 6 PNs per KC
    Proj('P083', 'KC*', 'APL', 800, 2, 14, 'both', 'all', 59, 'V'),  # w_cal 6: 79,271 / 1,342 KCs
    Proj('P084', 'APL', 'KC*', 2, 20, 6, 'both', 'all', 59, 'V'),  # w_cal 109: GABA, all-to-all
    Proj('P085', 'KCg-m|KCg-d', 'MBON01|MBON11|MBON09', 200, 2, 14, 'both', 'random', 22.6, 'V'),  # w_cal 25: KCg-m -> MBON01 30,660
    Proj('P086', 'KCab-c|KCab-m|KCab-s|KCab-p', 'MBON06|MBON14|MBON18|MBON12|MBON13', 200, 2, 14, 'both', 'random', 18, 'V'),  # w_cal 25: KCab -> MBON06 12,634; MBON12/13 hops are D
    Proj('P087', "KCa'b'-ap1|KCa'b'-ap2|KCa'b'-m", 'MBON03|MBON04|MBON05', 100, 2, 14, 'both', 'random', 8, 'D'),  # w_cal 51: 
    Proj('P088', 'PAM*', 'KC*', 40, 20, 3, 'both', 'random', 1.3, 'V'),  # w_cal 3: PAM01 -> KCg-m 20,477 over 15,263 pairs
    Proj('P089', 'PPL10*', 'KC*', 4, 20, 3, 'both', 'random', 5.4, 'V'),  # w_cal 27: PPL101 -> KCg-m 7,224
    Proj('P090', 'FB6A|FB6H|FB7A|FB7B', 'ExR1|ExR2', 10, 10, 6, 'both', 'random', 20, 'D'),  # w_cal 44: dFB sleep -> ExR (Glu)
    Proj('P091', 'ExR1|ExR2', 'PFL3|hDeltaB|PFNd', 4, 10, 3, 'both', 'random', 0, 'E'),  # w_cal 55: GABA arousal gate (sleep mode)
    Proj('P092', 'EPG', 'PEN_a(PEN1)|PEN_b(PEN2)', 2, 20, 14, 'ipsi', 'wedge', 66, 'V'),  # w_cal 255: 6,100 syn
    Proj('P093', 'PEN_a(PEN1)', 'EPG', 2, 20, 14, 'ipsi', 'ring_shift', 156, 'V'),  # w_cal 255: 14,398 syn; shift +1 on R, -1 on L
    Proj('P094', 'PEN_b(PEN2)', 'EPG', 2, 20, 8, 'ipsi', 'ring_shift', 60, 'D'),  # w_cal 145: 
    Proj('P095', 'EPG', 'PEG', 2, 20, 8, 'ipsi', 'wedge', 40, 'D'),  # w_cal 145: 
    Proj('P096', 'EPG', 'Delta7', 46, 20, 14, 'both', 'all', 10, 'V'),  # w_cal 11: 19,896 syn
    Proj('P097', 'Delta7', 'EPG', 42, 20, 6, 'both', 'all', 2.2, 'V'),  # w_cal 5: Glu; 4,294 syn; implementation skips the source's own wedge
    Proj('P098', 'ER4d|ER4m|ER2_a', 'EPG', 28, 10, 6, 'both', 'all', 10, 'V'),  # w_cal 16: GABA ring input (12,335 for ER4d)
    Proj('P099', 'LgLG1a|LgLG1b|LgLG2', 'AN05B102a', 200, 40, 14, 'ipsi', 'all', 20, 'D'),  # w_cal 1: pheromone GRN target [?]
    Proj('P100', 'AN05B102a|AVLP732m|AVLP733m', 'pC1_*', 6, 30, 10, 'both', 'random', 0, 'E'),  # w_cal 40: engineered pheromone/auditory relay onto P1
    Proj('P101', 'JO-A1|JO-A2', 'AVLP732m|AVLP733m', 20, 30, 14, 'ipsi', 'random', 20, 'D'),  # w_cal 17: candidate auditory relay
    Proj('P102', 'mAL_m8|mAL_m1', 'pC1_*', 14, 10, 4, 'contra', 'all', 8, 'V'),  # w_cal 21: GABA; top pC1 inputs (2,617 / 2,468)
    Proj('P103', 'pC1_14a|pC1_7b|pC1_10a|aIPg7', 'pIP10', 6, 30, 14, 'both', 'random', 120, 'V'),  # w_cal 57: pC1_14a 565 (6 -> 2), aIPg7 1,280
    Proj('P104', 'pIP10|pMP2', 'dPR1', 2, 30, 14, 'both', 'all', 278, 'V'),  # w_cal 170: 1,112 / 1,750
    Proj('P105', 'dPR1', 'dMS2', 1, 30, 14, 'contra', 'all', 41, 'V'),  # w_cal 339: 1,646 mostly contra
    Proj('P106', 'dMS2', 'hg1 MN', 10, 40, 14, 'ipsi', 'all', 108, 'V'),  # w_cal 25: 3,657 syn
    Proj('P107', 'dMS2', 'b1 MN|hg3 MN|hg4 MN', 10, 40, 6, 'ipsi', 'all', 4.6, 'V'),  # w_cal 11: 155 to b1; hg3/hg4 D
    Proj('P108', 'vPR9_a|vPR9_c', 'pIP10', 3, 10, 3, 'contra', 'all', 20, 'D'),  # w_cal 73: GABA
    Proj('P109', 'AN08B020', 'pC1_*', 1, 20, 4, 'both', 'all', 23, 'V'),  # w_cal 145: 1,381 syn ascending
    Proj('P110', 'JO-CM|JO-EV1|JO-FV', 'SAD093', 20, 30, 14, 'ipsi', 'random', 14, 'V'),  # w_cal 17: 275 syn
    Proj('P111', 'SAD093|BM', 'DNg62|DNge078', 5, 30, 14, 'both', 'random', 100, 'V'),  # w_cal 68: 712 / 617
    Proj('P112', 'DNg62|DNge078', 'Ti flexor MN|Tr flexor MN', 2, 40, 8, 'ipsi', 'random', 0, 'E'),  # w_cal 73: front-leg grooming stand-in
)

#: Isolated-neuron rest rate under the default engine noise (mu 0.5 mV/ms, sigma 3.5 mV/sqrt(ms)) [V] RESEARCH 7:
#: the floor every population sits on when nothing drives it; the rest brakes are calibrated at this rate.
NOISE_FLOOR_HZ: float = 2.3

#: Measured rest rate of the populations the tonic-current table of SPEC c.10 holds just below threshold (``lamina``
#: and ``motion_in`` at 7.3 mV, ``mal`` and ``feed_pre_inh`` at 7.05 mV): 22-31 Hz over N in {4000, 20000} [E].
#: ``P114`` is calibrated at this rate, so switching the tonic table off weakens that brake instead of inverting it.
TONIC_REST_HZ: float = 25.0

#: Rest-state re-tuning of SPEC g.3 rows [E], applied on top of ``PROJECTIONS_SPEC`` (pid -> {field: value}). SPEC g.6
#: makes the five gates the arbiter and explicitly allows ``target_mv`` edits. Why each group is needed (measured on the
#: engine of SPEC c.10 with its noise and tonic table):
#: (1) P005/P006/P008/P009/P010/P011 - the optic-lobe relays are calibrated 5-10x above their own literature means
#:     (T4/T5 -> LC: 6 mV target vs the 0.7 mV that w_lit 3.2 implies), so they relay the tonic lamina/medulla DC
#:     (~27 Hz under noise) 1:1 into LC4/LPLC2 (89 Hz) and from there into the giant fibre. Set to the literature-
#:     equivalent target (w_cal == round(w_lit)); the flicker signal still reaches T4/T5.
#: (2) P046/P047 assume MBONs at 20 Hz nominal, but P085-P087 make MBONs fire ~45 Hz at KC rest by construction; r_nom
#:     is corrected to that rest rate and the target lowered to 1 mV so DNa02 is not driven (was 78 Hz) and DNg100 not
#:     suppressed (-9 mV of MBON-avoid glutamate silenced the explore baseline of SPEC f.2) at rest.
#: (3) P048/P051 (EPG / hDeltaB -> PFL) are 6-7x above literature and make PFL cells fire 13-17 Hz from single-spike
#:     fluctuations, driving DNa02 to ~30 Hz at rest; lowered to ~1.5x literature (compass bump still visible).
#: (4) P040/P041 (engineered walking pools) set the decorative leg-MN rest rate; lowered so the rest mean at N=4000
#:     with the tonic table stays inside the 1-5 Hz gate (4.7-4.9 Hz).
#: (5) P067's presynaptic pool realises ~36 Hz (not 60 Hz) during the 100 Hz sugar gate because every hop of the
#:     verified chain attenuates; r_nom is set to the realised rate so MN9 reaches the 14 mV the gate arithmetic of
#:     SPEC g.6 assumes (MN9 ~45 Hz instead of a seed-dependent 16-22 Hz against the 20 Hz gate).
#: (6) P012/P013 (LC4/LPLC2 -> DNp01) get ``scale_k`` and a raised target. ``scale_k``: SPEC g.0 says an ``all`` row's
#:     ``k_in`` "equals the per-side source count, so the two agree by construction", which only holds at ``s == 1``
#:     (at N=4000 LC4 is 16 cells per side against k_in 63), so the realised in-degree is used for the weight formula
#:     and the table's ``target_mv`` is honoured at every N - without it the looming gate-3 latency is N-dependent
#:     (10 ms at N=20000, 24 ms at N=4000 against the 20 ms limit). Target 7 -> 16 mV: the rest brake below
#:     hyperpolarises the giant fibre by ~32 mV (it must, see REST_BRAKES), so the looming drive has to cover that
#:     plus the 7 mV threshold gap inside the 20 ms of gate 3, while SPEC g.6's own gate-3 arithmetic assumes an
#:     unbraked GF sitting at ``v_rest`` (which SPEC h.3's "0 DNp01 spikes at rest" forbids). With the +6 electrical
#:     proxies of SPEC c.5 the per-edge weight is 12 instead of 8 and the measured latency is 12-14 ms at every N.
REST_TUNING: dict[str, dict[str, float | bool]] = {
    "P005": {"target_mv": 1.2},                      # w 194 -> 29 (w_lit 30)
    "P006": {"target_mv": 1.4},                      # w 55 -> 10 (w_lit 10)
    "P008": {"target_mv": 0.6},                      # w 65 -> 5 (w_lit 5)
    "P009": {"target_mv": 0.7},                      # w 27 -> 3 (w_lit 3.2)
    "P010": {"target_mv": 0.8},                      # w 27 -> 4 (w_lit 3.5)
    "P011": {"target_mv": 0.7},                      # w 27 -> 3 (w_lit 3)
    "P012": {"target_mv": 16.0, "scale_k": True},    # w 2 -> 6 at s=1 (+6 patch proxies = 12 per edge)
    "P013": {"target_mv": 16.0, "scale_k": True},    # w 2 -> 6 at s=1 (+6 patch proxies = 12 per edge)
    "P040": {"target_mv": 8.0},                      # w 113 -> 65
    "P041": {"target_mv": 6.0},                      # w 42 -> 18
    "P046": {"r_nom_hz": 45.0, "target_mv": 1.0},    # w 73 -> 4
    "P047": {"r_nom_hz": 45.0, "target_mv": 1.0},    # w 55 -> 4
    "P048": {"target_mv": 2.0},                      # w 145 -> 36 (w_lit 25)
    "P051": {"target_mv": 2.0},                      # w 73 -> 24 (w_lit 10)
    "P067": {"r_nom_hz": 36.0},                      # w 28 -> 47
}

#: Engineered rest brakes (provenance ``E``, listed in ``meta['engineered_edges']`` and ``meta['rest_brakes']``).
#: Under the mandated noise every neuron fires ~2.3 Hz, so the 155 ipsilateral LC4/LPLC2 inputs of each DNp01
#: (6 chemical + 6 patched synapses each, SPEC c.5) plus the cell's own noise deliver ~8 mV at rest against a 7 mV
#: threshold gap, and the giant fibre would fire 10-20 Hz with no stimulus (one GF spike is one jump, and SPEC h.3
#: requires 0 DNp01 spikes at rest). The brake gives the GF the high resting threshold it has in vivo [L] in two
#: parts, because suppressing a 2.3 Hz noise floor needs inhibition that is both deep and *continuous*
#: (``lambda * tau_s >> 1``; a deep but lumpy brake leaves gaps in which a single 26 mV DNp11 kick fires the GF):
#:
#: * ``P113`` - depth: a 32 mV steady-state GABA/Glu brake from the 159 inhibitory cells that have no inputs in the
#:   table and no encoder drive (ring neurons, PVLP/PS/AOTU/LAL/VES premotor cells, VNC inhibitory interneurons, vPR9,
#:   GNG458/DNge129, dFB), so it tracks the noise floor (zero when noise is off, larger during dFB-driven sleep). All
#:   159 come from populations below the ``n < 40`` scaling floor of SPEC g.2, so the pool - and the brake - is the
#:   same at every N. 0.37 events/ms: deep but lumpy.
#: * ``P114`` - continuity: an 8 mV brake from the inhibitory cells that the tonic-current table of SPEC c.10 holds
#:   just below threshold (``mal`` 28-31 Hz, and Mi4/Mi9 of ``motion_in`` 22 Hz; neither has an input in the
#:   projection table or an encoder channel, so their rate is set by the tonic table alone). At 25 Hz nominal these
#:   3.7 events/ms make the total brake continuous, and - unlike every noise-floor population - they fire within the
#:   first 10 ms of a cold start, which is what closes the startup window (see the module docstring).
#:
#: Measured with the tonic table applied (the state ``SimulationLoop`` starts in, SPEC c.27): 0 DNp01 spikes over
#: 1000 ms of rest for all 72 combinations of N in {4000, 8000, 20000} x build seed in {1, 7, 1337} x engine seed
#: 0..7, and the looming gate 3 fires the GF 12-14 ms after the drive starts at every N (limit 20 ms).
REST_BRAKES: tuple[Proj, ...] = (
    Proj("P113",
         "ER4d|ER4m|ER2_a|PVLP020|AOTU019|PS049|PS059|LAL083|LAL126|VES051|VES052|IN13A022|IN21A026|IN08A002"
         "|vPR9_a|vPR9_b|vPR9_c|GNG458|DNge129|FB6A|FB6H|FB7A|FB7B",
         "DNp01", 159, NOISE_FLOOR_HZ, 32, "both", "all", 0, "E"),                       # w_cal 64, 0.37 events/ms
    Proj("P114", "mAL_m8|mAL_m1|Mi4|Mi9",
         "DNp01", 148, TONIC_REST_HZ, 8, "both", "all", 0, "E", scale_k=True),           # w_cal 2 at s=1, 4 at s=0.25
)


def tune_projections(spec: Sequence[Proj] = PROJECTIONS_SPEC, tuning: dict[str, dict[str, float]] = REST_TUNING,
                     brakes: Sequence[Proj] = REST_BRAKES) -> tuple[Proj, ...]:
    """``spec`` with the per-pid field overrides of ``tuning`` applied, followed by ``brakes``.

    Raises ``ValueError`` for an unknown pid, an unknown ``Proj`` field or a duplicate pid, so a typo in the tuning
    table cannot silently leave a row untouched."""
    by_pid = {p.pid: p for p in spec}
    if len(by_pid) != len(spec):
        raise ValueError("duplicate pid in the projection table")
    fields = set(Proj.__dataclass_fields__) - {"pid"}
    out: list[Proj] = []
    for p in spec:
        ov = tuning.get(p.pid)
        if ov:
            bad = set(ov) - fields
            if bad:
                raise ValueError(f"{p.pid}: unknown Proj field(s) in REST_TUNING: {sorted(bad)}")
            p = replace(p, **ov)
        out.append(p)
    unknown = set(tuning) - set(by_pid)
    if unknown:
        raise ValueError(f"REST_TUNING names unknown pid(s): {sorted(unknown)}")
    for b in brakes:
        if b.pid in by_pid:
            raise ValueError(f"brake pid {b.pid} collides with a SPEC row")
        by_pid[b.pid] = b
        out.append(b)
    return tuple(out)


#: The table ``build_synthetic`` wires: SPEC g.3 + ``REST_TUNING`` + ``REST_BRAKES`` (114 rows, 21 ``E`` pids:
#: the 19 SPEC g.3 ``E`` rows plus the two engineered rest brakes P113/P114).
PROJECTIONS: tuple[Proj, ...] = tune_projections()

#: core total at s = 1 (N >= 20,000), SPEC g.1.
CORE_TOTAL: int = 6535

#: region fractions of the filler budget [V] RESEARCH 3.1 (sum = 1.000).
REGION_FRAC: dict[str, float] = {"optic_lobe": 0.605, "antennal_lobe": 0.023, "mushroom_body": 0.027,
    "central_complex": 0.018, "sez": 0.024, "central_other": 0.160, "descending_motor": 0.013, "vnc": 0.130}
#: filler neurotransmitter shares [V] RESEARCH 3.3 (rescaled by ``inhib_frac`` at build time).
FILLER_NT: tuple[tuple[str, float], ...] = (("acetylcholine", .62), ("glutamate", .18), ("gaba", .13), ("histamine", .04), ("unknown", .03))
#: background inter-region targets (SPEC g.4) [E].
REGION_ADJACENCY: dict[str, tuple[str, ...]] = {
    "optic_lobe":       ("central_other", "descending_motor"),               # VPN highways only
    "antennal_lobe":    ("mushroom_body", "central_other"),
    "mushroom_body":    ("central_other", "central_complex"),
    "central_complex":  ("central_other", "descending_motor"),
    "sez":              ("descending_motor", "central_other"),
    "central_other":    ("central_complex", "descending_motor", "sez", "mushroom_body"),
    "descending_motor": ("vnc",),                                            # brain -> VNC only via DNs
    "vnc":              ("descending_motor", "central_other"),               # VNC -> brain via ANs
}
#: filler ``superclass`` = the region's dominant MaleCNS superclass [V] RESEARCH 3.1 (export / provenance only).
FILLER_SUPERCLASS: dict[str, str] = {
    "optic_lobe": "ol_intrinsic", "antennal_lobe": "cb_sensory", "mushroom_body": "cb_intrinsic",
    "central_complex": "cb_intrinsic", "sez": "cb_intrinsic", "central_other": "cb_intrinsic",
    "descending_motor": "descending_neuron", "vnc": "vnc_intrinsic",
}
#: filler ``class`` written by ``export_csv`` [E]: a region marker that ``region_of(superclass, cls, "")`` maps back
#: to the region (the data model stores no class, so the CSV round trip needs it); "" where the superclass suffices.
EXPORT_FILLER_CLASS: dict[str, str] = {
    "optic_lobe": "", "antennal_lobe": "olfactory", "mushroom_body": "Kenyon_Cell", "central_complex": "CX",
    "sez": "SEZPN", "central_other": "", "descending_motor": "", "vnc": "",
}

# background graph constants (SPEC g.4)
_BG_SAME_REGION_P: float = 0.85       # [E]
_BG_OUTDEG_SHIFT: float = 0.18        # lognormal(ln(mean) - 0.18, 0.6) has mean == mean_outdeg [E]
_BG_OUTDEG_SIGMA: float = 0.6
_BG_OUTDEG_MIN: int = 3
_BG_OUTDEG_MAX: int = 200
_BG_WEIGHT_P: float = 0.206           # Geometric(p) mean 4.85 synapses = MaleCNS mean per connection [D]
_BG_WEIGHT_MAX: int = 40
_W_MAX: int = 400                     # per-edge cap of projection weights (SPEC g.0)
_N_WEDGES: int = 16                   # EPG/PEN wedges (SPEC g.0)
_SCALE_MIN_N: int = 40                # rows with n >= 40 are scaled, smaller rows never (SPEC g.2)
_SCALE_FLOOR: int = 4
_GAIN_DEFAULT: dict[str, float] = {"calibrated": 1.0, "literature": 0.65}

_ADJ_TAB = np.full((len(REGIONS), 4), -1, dtype=np.int64)
_ADJ_LEN = np.zeros(len(REGIONS), dtype=np.int64)
for _r, _adj in REGION_ADJACENCY.items():
    _ADJ_LEN[REGION_ID[_r]] = len(_adj)
    for _j, _t in enumerate(_adj):
        _ADJ_TAB[REGION_ID[_r], _j] = REGION_ID[_t]
del _r, _adj, _j, _t


# --------------------------------------------------------------------------- formulas (SPEC c.6 / g.2)


def calibrated_weight(k_in: int, r_nom_hz: float, target_mv: float, w_syn: float = 0.275, tau_s: float = 5.0,
                      gain: float = 1.0, w_max: int = 400) -> int:
    """``w = clip(round(target_mv / (k_in * w_syn * gain * r_nom_hz/1000 * tau_s)), 1, w_max)``.

    Steady-state ``g`` at the nominal presynaptic rate equals ``target_mv`` (RESEARCH section 7 [V]:
    ``g_ss = k_in * w * w_syn * gain * (r/1000) * tau_s``). Examples: (12, 100, 14) -> 8, (4, 60, 14) -> 42,
    (155, 33, 14) -> 2, (1, 30, 14) -> 339.
    """
    if k_in <= 0 or r_nom_hz <= 0 or w_syn <= 0 or tau_s <= 0 or gain <= 0:
        raise ValueError("calibrated_weight: k_in, r_nom_hz, w_syn, tau_s and gain must be > 0")
    denom = float(k_in) * float(w_syn) * float(gain) * (float(r_nom_hz) / 1000.0) * float(tau_s)
    w = int(round(float(target_mv) / denom))
    return int(min(max(w, 1), int(w_max)))


def scale_factor(n_target: int) -> float:
    """``s = clip((n_target - 2000)/18000, 0.25, 1.0)`` (SPEC g.2)."""
    return float(min(max((int(n_target) - 2000) / 18000.0, 0.25), 1.0))


def scale_populations(n_target: int, pops: Sequence[Pop] = POPULATIONS) -> tuple[float, list[Pop]]:
    """``s = clip((n_target - 2000)/18000, 0.25, 1.0)``. Populations with ``n >= 40`` are scaled to
    ``max(4, round(n*s))``; populations with ``n < 40`` are never scaled. At ``n_target >= 20000``, ``s == 1`` and the
    core is exactly the table (6,535 neurons). Returns ``(s, pops)``. ``round`` is Python's round-half-to-even, which
    the SPEC g.2 table was computed with.

    SPEC c.6 adds "(kept even when scaled)" to that sentence; that clause is NOT implemented because it contradicts
    the normative region-plan table of SPEC g.2: rounding every scaled row to an even count gives a core of 2,868
    (up) or 2,800 (down) at N=4000 instead of g.2's 2,830, and the same at N=8000. Plain ``round`` reproduces g.2
    exactly and leaves 22 odd scaled rows at N=4000 / 30 at N=8000 (L3 15, Mi4 15, KCg-m 85, LC9 55 ...), which the
    sides split as ceil/floor per SPEC g.0. Pinned by ``test_synthetic::test_region_plan``; reported as a spec
    contradiction."""
    s = scale_factor(n_target)
    out: list[Pop] = []
    for p in pops:
        if p.n >= _SCALE_MIN_N:
            out.append(replace(p, n=max(_SCALE_FLOOR, int(round(p.n * s)))))
        else:
            out.append(p)
    return s, out


def region_plan(n_target: int) -> dict[str, tuple[int, int, int]]:
    """``{region: (core, filler, total)}`` using ``REGION_FRAC`` on the remaining budget:
    ``filler_r = floor(FRAC_r * (N - core_total))``, remainder to ``optic_lobe``. Raises ``ValueError`` if
    ``N < core_total``. SPEC g.2 lists the expected values for N = 4000 / 8000 / 20000 / 166700."""
    n_target = int(n_target)
    _, pops = scale_populations(n_target)
    core = {r: 0 for r in REGIONS}
    for p in pops:
        core[p.region] += p.n
    core_total = sum(core.values())
    if n_target < core_total:
        raise ValueError(f"n_neurons={n_target} is below the core size {core_total} at this scale")
    budget = n_target - core_total
    filler = {r: int(math.floor(REGION_FRAC[r] * budget)) for r in REGIONS}
    filler["optic_lobe"] += budget - sum(filler.values())
    return {r: (core[r], filler[r], core[r] + filler[r]) for r in REGIONS}


# --------------------------------------------------------------------------- selector resolution


_GROUP_COMPILED: dict[str, re.Pattern[str]] = {k: re.compile(v) for k, v in GROUP_REGEX.items()}


def _type_mask(selector: str, types: Sequence[str]) -> np.ndarray:
    """bool[T]: which unique type labels the selector matches (GROUP_REGEX key first, else fnmatch patterns)."""
    t = list(types)
    if selector in _GROUP_COMPILED:
        pat = _GROUP_COMPILED[selector]
        return np.fromiter((x != "" and pat.fullmatch(x) is not None for x in t), dtype=bool, count=len(t))
    mask = np.zeros(len(t), dtype=bool)
    for pattern in selector.split("|"):
        pattern = pattern.strip()
        if not pattern:
            continue
        for i, x in enumerate(t):
            if x != "" and fnmatch.fnmatchcase(x, pattern):
                mask[i] = True
    return mask


def resolve_selector(selector: str, types: Sequence[str], type_idx: np.ndarray) -> np.ndarray:
    """Sorted int64 neuron indices selected by a ``Proj.src``/``Proj.dst`` string (SPEC g.0 order: (1) exact
    ``GROUP_REGEX`` key -> that group; (2) ``'|'``-separated fnmatch type patterns). Empty result is allowed here;
    ``build_synthetic`` turns it into a build error."""
    mask = _type_mask(selector, types)
    if not mask.any():
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(mask[np.asarray(type_idx, dtype=np.int64)]).astype(np.int64)


def _glomerulus_token(type_name: str) -> str:
    """Glomerulus token: ``ORN_<G>`` -> ``G``; ``<G>_..PN`` -> text before the first ``_``."""
    if type_name.startswith("ORN_"):
        return type_name[4:]
    return type_name.split("_", 1)[0]


# --------------------------------------------------------------------------- neuron layout


@dataclass
class _Layout:
    n: int
    types: list[str]
    type_idx: np.ndarray        # int32[n]
    side: np.ndarray            # int8[n]
    region: np.ndarray          # uint8[n]
    nt: np.ndarray              # object[n]
    sign: np.ndarray            # float32[n]
    is_core: np.ndarray         # bool[n]
    rank_in_side: np.ndarray    # int32[n] position among same-type same-side cells (index order)
    n_in_side: np.ndarray       # int32[n] size of that same-type same-side set
    region_start: np.ndarray    # int64[8] first index of each region (regions are contiguous)
    region_size: np.ndarray     # int64[8]
    pop_rows: list[tuple[Pop, int, int]]   # (scaled Pop, start, stop)
    scale: float = 1.0          # population scale factor s (SPEC g.2)


def _layout(n_neurons: int, seed_pop: np.random.SeedSequence, inhib_frac: float) -> tuple[float, _Layout, dict]:
    """Allocate indices region by region in ``REGIONS`` order (core populations in ``POPULATIONS`` order, then
    the region's filler) and fill ``types/type_idx/side/nt/sign/region`` plus the per-type side ranks."""
    s, pops = scale_populations(n_neurons)
    plan = region_plan(n_neurons)
    rng = np.random.default_rng(seed_pop)

    core_type_names = [p.type for p in pops]
    types = sorted(set(core_type_names) | {""})
    t_index = {t: i for i, t in enumerate(types)}

    type_idx = np.empty(n_neurons, dtype=np.int32)
    side = np.empty(n_neurons, dtype=np.int8)
    region = np.empty(n_neurons, dtype=np.uint8)
    nt = np.empty(n_neurons, dtype=object)
    is_core = np.zeros(n_neurons, dtype=bool)
    rank_in_side = np.zeros(n_neurons, dtype=np.int32)
    n_in_side = np.ones(n_neurons, dtype=np.int32)
    region_start = np.zeros(len(REGIONS), dtype=np.int64)
    region_size = np.zeros(len(REGIONS), dtype=np.int64)
    pop_rows: list[tuple[Pop, int, int]] = []

    # filler NT probabilities: GABA+Glu+His rescaled to inhib_frac, ACh/unknown absorb the rest
    inh = {"gaba", "glutamate", "histamine"}
    inh_share = sum(f for name, f in FILLER_NT if name in inh)
    exc_share = sum(f for name, f in FILLER_NT if name not in inh)
    nt_names = [name for name, _ in FILLER_NT]
    nt_p = np.array(
        [f * (inhib_frac / inh_share) if name in inh else f * ((1.0 - inhib_frac) / exc_share) for name, f in FILLER_NT],
        dtype=np.float64,
    )
    nt_p /= nt_p.sum()

    pos = 0
    by_region: dict[str, list[Pop]] = {r: [] for r in REGIONS}
    for p in pops:
        by_region[p.region].append(p)
    for r_i, r in enumerate(REGIONS):
        region_start[r_i] = pos
        core_r, filler_r, _total = plan[r]
        for p in by_region[r]:
            a, b = pos, pos + p.n
            type_idx[a:b] = t_index[p.type]
            region[a:b] = r_i
            nt[a:b] = p.nt
            is_core[a:b] = True
            if p.n == 1:
                side[a] = 0
                rank_in_side[a] = 0
                n_in_side[a] = 1
            else:
                j = np.arange(p.n, dtype=np.int64)
                side[a:b] = np.where(j % 2 == 0, -1, 1).astype(np.int8)
                rank_in_side[a:b] = (j // 2).astype(np.int32)
                n_l = (p.n + 1) // 2
                n_r = p.n // 2
                n_in_side[a:b] = np.where(j % 2 == 0, n_l, n_r).astype(np.int32)
            pop_rows.append((p, a, b))
            pos = b
        if pos - region_start[r_i] != core_r:  # pragma: no cover - plan and layout share scale_populations
            raise AssertionError(f"core allocation mismatch in {r}: {pos - region_start[r_i]} != {core_r}")
        if filler_r:
            a, b = pos, pos + filler_r
            type_idx[a:b] = t_index[""]
            region[a:b] = r_i
            j = np.arange(filler_r, dtype=np.int64)
            side[a:b] = np.where(j % 2 == 0, -1, 1).astype(np.int8)
            rank_in_side[a:b] = (j // 2).astype(np.int32)
            n_in_side[a:b] = np.where(j % 2 == 0, (filler_r + 1) // 2, filler_r // 2).astype(np.int32)
            choice = rng.choice(len(nt_names), size=filler_r, p=nt_p)
            nt[a:b] = np.asarray(nt_names, dtype=object)[choice]
            pos = b
        region_size[r_i] = pos - region_start[r_i]
    if pos != n_neurons:  # pragma: no cover
        raise AssertionError(f"allocated {pos} neurons, expected {n_neurons}")
    sign = np.asarray([NT_SIGN[nt_normalise(x)] for x in nt.tolist()], dtype=np.float32)
    layout = _Layout(n=n_neurons, types=types, type_idx=type_idx, side=side, region=region, nt=nt, sign=sign,
                     is_core=is_core, rank_in_side=rank_in_side, n_in_side=n_in_side, region_start=region_start,
                     region_size=region_size, pop_rows=pop_rows, scale=float(s))
    return s, layout, plan


# --------------------------------------------------------------------------- projections


def _allowed(lat: str, side_src: np.ndarray, side_dst: np.ndarray) -> np.ndarray:
    """bool[T, S] laterality mask: ipsi = same side, contra = opposite, both = all; side-0 cells count as both."""
    ss = side_src[None, :].astype(np.int16)
    st = side_dst[:, None].astype(np.int16)
    if lat == "ipsi":
        return (ss == st) | (ss == 0) | (st == 0)
    if lat == "contra":
        return (ss == -st) | (ss == 0) | (st == 0)
    if lat == "both":
        return np.ones((side_dst.shape[0], side_src.shape[0]), dtype=bool)
    raise ValueError(f"laterality must be one of {LATERALITIES}, got {lat!r}")


def _take_k(score: np.ndarray, allowed: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row (target) the ``k`` allowed columns (sources) with the smallest score (ties -> lowest column index).
    Returns (row, col) index arrays."""
    T, S = allowed.shape
    if T == 0 or S == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    sc = np.where(allowed, score, np.inf)
    order = np.argsort(sc, axis=1, kind="stable")
    count = np.minimum(allowed.sum(axis=1), k)
    kk = min(int(k), S)
    cols = order[:, :kk]
    mask = np.arange(kk, dtype=np.int64)[None, :] < count[:, None]
    rows = np.repeat(np.arange(T, dtype=np.int64)[:, None], kk, axis=1)
    return rows[mask], cols[mask]


def _project(proj: Proj, lay: _Layout, rng: np.random.Generator, weights_mode: str
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """COO edges (pre int64, post int64, w float32) of one ``Proj`` plus its meta record."""
    if proj.laterality not in LATERALITIES:
        raise ValueError(f"{proj.pid}: laterality must be one of {LATERALITIES}, got {proj.laterality!r}")
    if proj.topology not in TOPOLOGIES:
        raise ValueError(f"{proj.pid}: topology must be one of {TOPOLOGIES}, got {proj.topology!r}")
    S = resolve_selector(proj.src, lay.types, lay.type_idx)
    T = resolve_selector(proj.dst, lay.types, lay.type_idx)
    if S.size == 0 or T.size == 0:
        raise ValueError(f"{proj.pid}: selector resolved to zero neurons (src {proj.src!r} -> {S.size}, "
                         f"dst {proj.dst!r} -> {T.size})")
    allowed = _allowed(proj.laterality, lay.side[S], lay.side[T])
    allowed &= T[:, None] != S[None, :]   # never itself
    topo = proj.topology
    k = int(proj.k_in)
    if topo == "random":
        rows, cols = _take_k(rng.random((T.size, S.size)), allowed, k)
    elif topo == "retinotopic":
        rho_s = lay.rank_in_side[S].astype(np.float64) / lay.n_in_side[S]
        rho_t = lay.rank_in_side[T].astype(np.float64) / lay.n_in_side[T]
        rows, cols = _take_k(np.abs(rho_s[None, :] - rho_t[:, None]), allowed, k)
    else:
        if topo == "glomerular":
            tok_s = np.asarray([_glomerulus_token(lay.types[lay.type_idx[i]]) for i in S.tolist()], dtype=object)
            tok_t = np.asarray([_glomerulus_token(lay.types[lay.type_idx[i]]) for i in T.tolist()], dtype=object)
            allowed &= tok_s[None, :] == tok_t[:, None]
        elif topo in ("wedge", "ring_shift"):
            w_s = (_N_WEDGES * lay.rank_in_side[S].astype(np.int64)) // lay.n_in_side[S]
            w_t = (_N_WEDGES * lay.rank_in_side[T].astype(np.int64)) // lay.n_in_side[T]
            if topo == "ring_shift":
                shift = np.where(lay.side[S] == -1, -1, 1).astype(np.int64)   # +1 on R, -1 on L
                w_s = (w_s + shift) % _N_WEDGES
            allowed &= w_s[None, :] == w_t[:, None]
        elif topo == "all" and proj.src == "Delta7" and proj.dst == "EPG":
            # Delta7 -> EPG additionally skips the source's own wedge (SPEC g.0)
            w_s = (_N_WEDGES * lay.rank_in_side[S].astype(np.int64)) // lay.n_in_side[S]
            w_t = (_N_WEDGES * lay.rank_in_side[T].astype(np.int64)) // lay.n_in_side[T]
            allowed &= w_s[None, :] != w_t[:, None]
        rows, cols = np.nonzero(allowed)
    pre = S[cols]
    post = T[rows]
    target_mv = float(proj.target_mv)
    k_eff = int(proj.k_in)
    if proj.scale_k and rows.size and T.size:
        k_eff = max(1, int(round(float(rows.size) / float(T.size))))
    w_cal = calibrated_weight(k_eff, proj.r_nom_hz, target_mv)
    if weights_mode == "literature" and proj.w_lit > 0:
        w_base = float(proj.w_lit)
    else:
        w_base = float(w_cal)
    jitter = np.exp(rng.normal(0.0, float(proj.w_cv), size=pre.shape[0])) if pre.shape[0] else np.zeros(0)
    w = np.clip(np.maximum(1.0, np.rint(w_base * jitter)), 1.0, float(_W_MAX)).astype(np.float32)
    rec = {
        "edges": int(pre.shape[0]),
        "w_mean": float(w.mean()) if w.size else 0.0,
        "w_base": w_base,
        "w_cal": int(w_cal),
        "target_mv": target_mv,
        "k_eff": k_eff,
        "n_src": int(S.size),
        "n_dst": int(T.size),
        "provenance": proj.provenance,
        "topology": proj.topology,
        "laterality": proj.laterality,
    }
    return pre, post, w, rec


# --------------------------------------------------------------------------- background (SPEC g.4)


def _background(lay: _Layout, rng: np.random.Generator, mean_outdeg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Background out-edges: ``d ~ LogNormal(ln(mean_outdeg) - 0.18, 0.6)`` clipped to [3, 200] (core neurons get
    ``d/2``); target = same region with p 0.85 else a uniformly drawn ``REGION_ADJACENCY`` region, never itself;
    weight ``~ Geometric(0.206)`` capped at 40 (mean 4.85 synapses [D])."""
    n = lay.n
    d = rng.lognormal(math.log(float(mean_outdeg)) - _BG_OUTDEG_SHIFT, _BG_OUTDEG_SIGMA, size=n)
    d = np.clip(np.rint(d), _BG_OUTDEG_MIN, _BG_OUTDEG_MAX).astype(np.int64)
    d = np.where(lay.is_core, np.maximum(1, d // 2), d)
    m = int(d.sum())
    pre = np.repeat(np.arange(n, dtype=np.int64), d)
    reg_pre = lay.region[pre].astype(np.int64)
    same = rng.random(m) < _BG_SAME_REGION_P
    j = np.floor(rng.random(m) * _ADJ_LEN[reg_pre]).astype(np.int64)
    reg_post = np.where(same, reg_pre, _ADJ_TAB[reg_pre, np.minimum(j, 3)])
    post = lay.region_start[reg_post] + np.floor(rng.random(m) * lay.region_size[reg_post]).astype(np.int64)
    post = np.minimum(post, lay.region_start[reg_post] + lay.region_size[reg_post] - 1)
    loops = np.flatnonzero(post == pre)
    guard = 0
    while loops.size and guard < 64:
        r = reg_post[loops]
        post[loops] = lay.region_start[r] + np.floor(rng.random(loops.size) * lay.region_size[r]).astype(np.int64)
        loops = loops[post[loops] == pre[loops]]
        guard += 1
    if loops.size:  # a region of size 1: the edge cannot avoid itself; drop it
        keep = np.ones(m, dtype=bool)
        keep[loops] = False
        pre, post = pre[keep], post[keep]
        m = int(pre.shape[0])
    w = np.minimum(rng.geometric(_BG_WEIGHT_P, size=m), _BG_WEIGHT_MAX).astype(np.float32)
    return pre, post, w


# --------------------------------------------------------------------------- build (SPEC g.5)


def build_synthetic(n_neurons: int = 20_000, seed: int = 1337, mean_outdeg: int = 25,
                    weights: str = "calibrated", inhib_frac: float = 0.33) -> Connectome:
    """Deterministic (``SeedSequence(seed).spawn(7)[0]`` split into three streams: populations/sides, projections,
    background, so that changing ``mean_outdeg`` does not alter the core wiring). Steps: ``scale_populations`` ->
    ``region_plan`` -> assign indices region by region in ``POPULATIONS`` order then filler -> types/sides/nt/sign/
    region -> ``PROJECTIONS`` (topology rules of SPEC g.3, weight per mode) -> background edges (SPEC g.4) ->
    concatenate, ``sum_duplicates`` -> ``apply_patches`` -> ``resolve_groups`` -> validate.

    ``meta``: ``weights_mode``, ``gain_default`` (1.0 calibrated / 0.65 literature), ``engineered_edges`` (pids
    with 'E'), ``projections`` ({pid: {edges, w_mean, ...}}), ``region_counts``, ``license='synthetic (no data)'``,
    ``citation=None``, ``note='synthetic structured stand-in shaped like MaleCNS v1.0; not real connectome data'``,
    plus ``rest_tuning`` ({pid: {field: {spec, used}}}, the SPEC g.3 values that ``REST_TUNING`` overrides) and
    ``rest_brakes`` (the ``REST_BRAKES`` pids).
    """
    n_neurons = int(n_neurons)
    seed = int(seed)
    mean_outdeg = int(mean_outdeg)
    if weights not in _GAIN_DEFAULT:
        raise ValueError(f"weights must be 'calibrated' or 'literature', got {weights!r}")
    if not (0.0 < float(inhib_frac) < 1.0):
        raise ValueError(f"inhib_frac must be in (0, 1), got {inhib_frac}")
    if mean_outdeg < 1:
        raise ValueError(f"mean_outdeg must be >= 1, got {mean_outdeg}")
    root = np.random.SeedSequence(seed).spawn(7)[0]
    ss_pop, ss_proj, ss_bg = root.spawn(3)

    s, lay, plan = _layout(n_neurons, ss_pop, float(inhib_frac))

    rng_proj = np.random.default_rng(ss_proj)
    pre_l: list[np.ndarray] = []
    post_l: list[np.ndarray] = []
    w_l: list[np.ndarray] = []
    proj_meta: dict[str, dict] = {}
    for proj in PROJECTIONS:
        p, q, w, rec = _project(proj, lay, rng_proj, weights)
        pre_l.append(p)
        post_l.append(q)
        w_l.append(w)
        proj_meta[proj.pid] = rec
    e_proj = int(sum(int(x.shape[0]) for x in pre_l))

    rng_bg = np.random.default_rng(ss_bg)
    bp, bq, bw = _background(lay, rng_bg, mean_outdeg)
    pre_l.append(bp)
    post_l.append(bq)
    w_l.append(bw)
    e_bg = int(bp.shape[0])

    pre, post, weight = sum_duplicates(np.concatenate(pre_l), np.concatenate(post_l), np.concatenate(w_l), n_neurons)

    groups = resolve_groups(lay.types, lay.type_idx, lay.side)
    body_id = (SYNTHETIC_BODY_BASE + np.arange(n_neurons, dtype=np.int64)).astype(np.int64)
    mode_tag = "cal" if weights == "calibrated" else "lit"
    meta = {
        "e": int(pre.shape[0]),
        "synapses": float(weight.sum(dtype=np.float64)) if weight.size else 0.0,
        "license": SYNTHETIC_LICENSE,
        "citation": None,
        "seed": seed,
        "build_args": {
            "n_neurons": n_neurons, "seed": seed, "mean_outdeg": mean_outdeg, "weights": weights,
            "inhib_frac": float(inhib_frac), "scale": float(s), "core_total": int(lay.is_core.sum()),
            "filler_total": int(n_neurons - int(lay.is_core.sum())), "e_projections": e_proj, "e_background": e_bg,
            "generator_version": GENERATOR_VERSION,
        },
        "patches_applied": [],
        "gain_default": _GAIN_DEFAULT[weights],
        "weights_mode": weights,
        "engineered_edges": [p.pid for p in PROJECTIONS if p.provenance == "E"],
        "projections": proj_meta,
        "rest_tuning": {
            pid: {f: {"spec": _SPEC_BY_PID[pid].__getattribute__(f), "used": v} for f, v in ov.items()}
            for pid, ov in REST_TUNING.items()
        },
        "rest_brakes": [p.pid for p in REST_BRAKES],
        "region_plan": {r: {"core": c, "filler": f, "total": t} for r, (c, f, t) in plan.items()},
        "region_counts": count_regions(lay.region),
        "group_counts": count_groups(groups),
        "created": utc_now_iso(),
        "note": SYNTHETIC_NOTE,
    }
    conn = Connectome(
        name=f"synthetic-{n_neurons}-s{seed}-{mode_tag}",
        source="synthetic",
        n=n_neurons,
        pre=np.ascontiguousarray(pre, dtype=np.int32),
        post=np.ascontiguousarray(post, dtype=np.int32),
        weight=np.ascontiguousarray(weight, dtype=np.float32),
        sign=lay.sign,
        region=lay.region,
        side=lay.side,
        types=list(lay.types),
        type_idx=lay.type_idx,
        body_id=body_id,
        nt=lay.nt,
        groups=groups,
        meta=meta,
    )
    conn = apply_patches(conn)
    conn.validate()
    validate_groups(conn)
    return conn


_SPEC_BY_PID: dict[str, Proj] = {p.pid: p for p in PROJECTIONS_SPEC}


def mean_signed_in_weight(conn: Connectome, which: str = "filler") -> float:
    """SPEC g.4 subcriticality figure: mean over the selected neurons of the SIGNED synapse count they receive
    (``sum_j weight_ji * sign_j``). ``which``: ``'filler'`` (untyped neurons receive background edges only, so this is
    the background graph's figure; ``test_synthetic::test_background_subcritical`` asserts it is <= 60), ``'core'``
    (typed neurons: projections + background) or ``'all'``. 0.0 when the selection is empty.

    g.4's "<= 60 synapses" bound describes the background layer and matches its own arithmetic there
    (25 partners x 4.85 synapses x 0.34 net sign ~ 41; measured 31.7 at N=4000, 38.2 at N=20000). Read over the whole
    graph it does not hold and cannot: the projection layer on top is what makes the pathways fire (``'all'`` 83.2 at
    N=4000 / 64.8 at N=20000, ``'core'`` 104.4 / 119.5). The test asserts all three so the split is visible."""
    if which not in ("filler", "core", "all"):
        raise ValueError(f"which must be filler|core|all, got {which!r}")
    signed = np.asarray(conn.weight, dtype=np.float64) * np.asarray(conn.sign, dtype=np.float64)[conn.pre]
    in_w = np.bincount(np.asarray(conn.post, dtype=np.int64), weights=signed, minlength=int(conn.n))
    if which == "all":
        sel = np.ones(int(conn.n), dtype=bool)
    else:
        try:
            t_empty = conn.types.index("")
        except ValueError:
            t_empty = -1
        is_filler = np.asarray(conn.type_idx) == t_empty
        sel = is_filler if which == "filler" else ~is_filler
    return float(in_w[sel].mean()) if sel.any() else 0.0


# --------------------------------------------------------------------------- export (SPEC g.7)


_POP_BY_TYPE: dict[str, Pop] = {p.type: p for p in POPULATIONS}
_SIDE_CHAR: dict[int, str] = {-1: "L", 1: "R", 0: "M"}


def export_annotation(conn: Connectome, i: int) -> tuple[str, str]:
    """``(superclass, class)`` written for neuron ``i``: the ``POPULATIONS`` row of its type, or for untyped
    filler the region's ``FILLER_SUPERCLASS`` and ``EXPORT_FILLER_CLASS`` marker [E]."""
    t = conn.types[int(conn.type_idx[i])]
    pop = _POP_BY_TYPE.get(t) if t else None
    if pop is not None:
        return pop.superclass, pop.cls
    region = REGIONS[int(conn.region[i])]
    return FILLER_SUPERCLASS[region], EXPORT_FILLER_CLASS[region]


def export_csv(conn: Connectome, out_dir: Path) -> tuple[Path, Path]:
    """Write ``neurons.csv`` (``bodyId,type,instance,superclass,class,subclass,somaSide,status,consensusNt`` with
    ``instance = f"{type}_{side}"`` and ``status = "Traced"``) and ``connections.csv`` (``bodyId_pre,bodyId_post,
    weight``, unsigned weights, patches included) in the neuPrint-export schema of SPEC c.7 so that
    ``load_csv_dir(out_dir, min_weight=1, subset="all")`` round-trips ``n, e, sign, region, side``, groups and CSR.
    Returns ``(neurons_path, connections_path)``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neurons_path = out_dir / "neurons.csv"
    conns_path = out_dir / "connections.csv"
    types = list(conn.types)
    type_idx = np.asarray(conn.type_idx, dtype=np.int64)
    side = np.asarray(conn.side, dtype=np.int64)
    body = np.asarray(conn.body_id, dtype=np.int64)
    nt = np.asarray(conn.nt, dtype=object)
    with open(neurons_path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "status", "consensusNt"])
        for i in range(int(conn.n)):
            t = types[int(type_idx[i])]
            sc, cls = export_annotation(conn, i)
            sch = _SIDE_CHAR[int(side[i])]
            wr.writerow([int(body[i]), t, f"{t}_{sch}", sc, cls, "", sch, "Traced", str(nt[i])])
    pre_b = body[np.asarray(conn.pre, dtype=np.int64)]
    post_b = body[np.asarray(conn.post, dtype=np.int64)]
    w = np.asarray(conn.weight, dtype=np.float64)
    integral = bool(np.all(w == np.rint(w)))
    with open(conns_path, "w", encoding="utf-8", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["bodyId_pre", "bodyId_post", "weight"])
        if integral:
            for a, b, x in zip(pre_b.tolist(), post_b.tolist(), w.astype(np.int64).tolist()):
                wr.writerow([a, b, x])
        else:
            for a, b, x in zip(pre_b.tolist(), post_b.tolist(), w.tolist()):
                wr.writerow([a, b, f"{x:g}"])
    return neurons_path, conns_path
