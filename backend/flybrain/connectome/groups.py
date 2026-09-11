"""Functional groups: regex table, region/side cascades, readout partition (SPEC section c.4).

``GROUP_REGEX`` is the cell-type table of RESEARCH section 4 (``re.fullmatch`` on the MaleCNS
``type`` string), copied verbatim from SPEC section c.4. Provenance: every type string is
``[V]`` verified to exist in MaleCNS v1.0 unless the RESEARCH table marks it otherwise
(``grn_pher`` identity is ``[?]`` unverified; ``groom_dn`` identity is ``[D]`` derived;
``dn_freeze`` / ``dn_back`` roles are ``[L]`` literature).

Only numpy is imported at module level.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Iterable

import numpy as np

from .schema import REGION_ID

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Connectome

__all__ = [
    "GROUP_REGEX",
    "SIDED",
    "READOUTS",
    "STAR_TYPES",
    "SEZ_TYPE_REGEX",
    "region_of",
    "side_of",
    "resolve_groups",
    "readout_partition",
    "readout_sizes",
    "fold_sided_counts",
    "validate_groups",
]

# --------------------------------------------------------------------------- tables (verbatim, SPEC c.4)

GROUP_REGEX: dict[str, str] = {
  # sensory
  "grn_sugar": r"(LB3b|LB3c|PhG1[abc]|LgLG3|LgLG4|WG2)",
  "grn_sugar_labellar": r"(LB3b|LB3c|PhG1[abc])",
  "grn_water": r"LB3a", "grn_salt": r"LB3d",
  "grn_bitter": r"(LB1[a-e]|LgAG1)",
  "grn_pher": r"(LgLG1a|LgLG1b|LgLG2)",                      # [?] identity unverified
  "jo_aud": r"JO-A.*", "jo_groom": r"JO-(C|E|F).*", "bm": r"BM",
  "photoreceptor": r"(R1-R6|R[78][dpy])", "lamina": r"L[1-5]",
  "motion_in": r"(Mi1|Tm3|Mi4|Mi9|Tm1|Tm2|Tm4|Tm9)", "t4t5": r"T[45][a-d]",
  "lc_loom": r"(LC4|LPLC2)", "lc4": r"LC4", "lplc2": r"LPLC2",
  "lc_loom2": r"(LC6|LPLC1|LPLC4)", "lc_freeze": r"(LC9|LC31a)",
  "orn": r"ORN_.*", "alpn": r"[A-Za-z0-9+]+_(l|ad|il|lv|v)?PN", "alln": r"(lLN|il3LN|v2LN|vLN|lvLN).*",
  # mushroom body
  "kc": r"KC.*", "apl": r"APL", "dpm": r"DPM",
  "mbon_avoid": r"MBON0[1-6]", "mbon_approach": r"MBON(09|11|12|13|14|18)",
  "pam": r"PAM[0-9]{2}", "ppl1": r"PPL10[1-8]",
  # central complex
  "epg": r"EPG", "pen": r"PEN_[ab][(]PEN[12][)]", "peg": r"PEG", "delta7": r"Delta7", "ring": r"ER[1-6].*",
  "pfl3": r"PFL3", "pfl2": r"PFL2", "pfl1": r"PFL1", "hdelta": r"hDelta[A-M]", "pfn": r"PFN.*", "exr": r"ExR[1-8]",
  "dfb_sleep": r"FB[67][A-Z].*",
  # central other
  "p1": r"pC1.*", "mal": r"mAL_.*", "aipg": r"aIPg[0-9]+", "avlp_aud": r"AVLP73[23]m",
  "lal_ps": r"(LAL083|LAL126|LAL179|PS049|PS059|VES051|VES052|AOTU015|AOTU019)",
  # descending
  "gf": r"DNp01", "escape_dn": r"DNp(02|04|11)", "dn_saccade": r"DNp03", "dn_land": r"DNp(07|10)",
  "dn_freeze": r"DNp09", "dn_fwd": r"(DNg100|DNge053|DNg97)", "dng100": r"DNg100", "dn_back": r"MDN",
  "dn_halt": r"(DNg60|DNg74_[ab])",
  "steer_a02": r"DNa02", "steer_a01": r"DNa01", "steer_a03": r"DNa03", "steer_b01": r"DNb01", "steer_g13": r"DNg13",
  "flight_dn": r"DNg02_[a-g]", "groom_dn": r"(DNg62|DNge078)", "feed_dn": r"(DNge062|DNge080|DNg67)",
  "song_dn": r"(pIP10|pMP2)", "pip10": r"pIP10",
  # VNC efferent / motor
  "escape_vnc": r"(TTMn|PSI|GFC2)", "ttmn": r"TTMn", "psi": r"PSI", "gfc2": r"GFC2",
  "wing_power": r"(DLMn a, b|DLMn c-f|DVMn 1a-c|DVMn 2a, b|DVMn 3a, b)",
  "wing_steer": r"((b[123]|i[12]|iii[134]|hg[1-4]|tp[12]|tpn|ps[12]) MN|MNwm3[56])",
  "b1": r"b1 MN", "i1": r"i1 MN", "hg1": r"hg1 MN",
  "song_mn": r"(hg1|hg3|hg4|b1) MN",
  "leg_mn": r"(Ti flexor|Acc[.] ti flexor|Ti extensor|Tr flexor|Acc[.] tr flexor|Tr extensor|Fe reductor|Ta depressor|Ta levator|Sternotrochanter|Sternal anterior rotator|Sternal posterior rotator|Tergotr[.]|Pleural remotor/abductor|ltm|ltm1-tibia|ltm2-femur) MN",
  # SEZ feeding
  "feed_mn": r"MN9", "feed_mn_other": r"(MN1|MN6)",
  "feed_pre_exc": r"(GNG108|GNG120|GNG117|GNG234)", "feed_pre_inh": r"(GNG015|GNG095|GNG130|GNG180|GNG184)",
  "sugar2_exc": r"(GNG215|GNG232|GNG132|GNG089|PRW046|PRW047)",
  "sugar2_inh": r"(GNG042|GNG038|AN13B002|AN05B023d|GNG551)",
  "bitter2": r"(GNG016|GNG087|GNG592)",
  # VNC interneurons / song
  "song_vnc": r"(dPR1|dMS2|vPR6|vPR9_[abc])", "dms2": r"dMS2",
  "an_steer": r"(AN03A008|AN04B003)", "leg_premotor": r"(IN13A001|IN08A002|IN19A016|IN07B010|IN03B015|IN12B003|IN19B043)",
  "sad093": r"SAD093", "gng458": r"GNG458", "an05b102a": r"AN05B102a",
}
SIDED: frozenset[str] = frozenset({"gf","steer_a02","steer_a01","steer_a03","steer_b01","steer_g13","dn_freeze","dng100",
  "dn_fwd","escape_dn","dn_saccade","lc4","lplc2","lc_loom","lc_freeze","epg","pfl3","ttmn","b1","i1","hg1","dms2",
  "feed_mn","p1","pip10","flight_dn","grn_sugar_labellar","grn_bitter"})

# READOUTS: a PARTITION (each neuron in at most one readout) used by SpikeMonitor for per-population rates.
# Order is the wire order of tick.rates.pops. Sided readouts contribute "<name>_L"/"<name>_R" AND "<name>".
READOUTS: tuple[str, ...] = ("gf","escape_dn","dn_saccade","dn_land","dn_freeze","dng100","dn_fwd","dn_back","dn_halt",
  "steer_a02","steer_a01","steer_a03","steer_b01","steer_g13","flight_dn","groom_dn","feed_dn","pip10","song_dn",
  "ttmn","psi","gfc2","wing_power","b1","i1","hg1","wing_steer","leg_mn","feed_mn","feed_pre_exc","feed_pre_inh","sugar2_exc","sugar2_inh",
  "bitter2","grn_sugar","grn_water","grn_bitter","grn_pher","jo_aud","jo_groom","photoreceptor","lamina","motion_in","t4t5",
  "lc4","lplc2","lc_loom2","lc_freeze","orn","alpn","alln","kc","apl","mbon_avoid","mbon_approach","pam","ppl1",
  "epg","pen","delta7","ring","pfl3","dfb_sleep","p1","mal","lal_ps","dms2","song_vnc","an_steer","leg_premotor")
# Overlap resolution for the partition: first match in READOUTS order wins ("grn_sugar" therefore excludes nothing
# because "grn_sugar_labellar" is not a readout; "lc_loom" is derived = lc4 + lplc2 in RateEstimator.derived()).
# Consequences of the order: "pip10" precedes "song_dn" (so song_dn = pMP2 only), "b1"/"i1"/"hg1" precede "wing_steer",
# "dms2" precedes "song_vnc", "dng100" precedes "dn_fwd" (dn_fwd = DNge053 + DNg97). 70 readouts, 26 of them SIDED ->
# 70 + 52 + 9 derived = 131 keys in tick.rates.pops (section d.2).

STAR_TYPES: tuple[str, ...] = ("DNp01","DNa02","DNa01","DNg13","DNp09","DNg100","DNg02_a","MN9","TTMn","PSI","PFL3","EPG",
  "MBON01","MBON11","PAM01","PPL101","LB3b","LC4","LPLC2","LC9","pC1_14a","pIP10","dMS2","hg1 MN","DLMn c-f","R1-R6","L1","Mi1","T4a","ORN_DM1","KCg-m")

# --------------------------------------------------------------------------- region / side cascades (RESEARCH section 6)

#: rule 5 of ``region_of``: SEZ-resident types by name [V] RESEARCH section 6.
SEZ_TYPE_REGEX: str = r"(GNG[0-9]+|PhG.*|LB[0-9].*|PRW[0-9]+|SAD[0-9]+|FLA[0-9]+|MN[0-9]+[A-Za-z]*)"

_OL_SUPERCLASSES = frozenset({"ol_intrinsic", "ol_sensory", "visual_projection", "visual_centrifugal", "visual_projection_tbc"})
_AL_CLASSES = frozenset({"olfactory", "ALPN", "ALLN", "ALIN", "ALON"})
_MB_CLASSES = frozenset({"Kenyon_Cell", "MBON", "DAN"})
_MB_TYPES = frozenset({"APL", "DPM"})
_DN_SUPERCLASSES = frozenset({"descending_neuron", "vnc_motor", "vnc_efferent", "efferent_descending", "sensory_descending"})
_VNC_SUPERCLASSES = frozenset({"ascending_neuron", "sensory_ascending", "efferent_ascending", "sensory_ascending_tbc"})
_ORN_RE = re.compile(r"ORN_.*")
_SEZ_RE = re.compile(SEZ_TYPE_REGEX)


def _clean(s: str | None) -> str:
    return "" if s is None else str(s).strip()


def region_of(superclass: str | None, cls: str | None, type_: str | None) -> int:
    """The 8-rule cascade of RESEARCH section 6 [V]; first rule that matches wins. Returns a ``REGION_ID`` value.

    1. superclass in {ol_intrinsic, ol_sensory, visual_projection, visual_centrifugal, visual_projection_tbc}
       or cls == "visual" -> optic_lobe
    2. cls in {olfactory, ALPN, ALLN, ALIN, ALON} or type matches ``ORN_.*`` -> antennal_lobe
    3. cls in {Kenyon_Cell, MBON, DAN} or type in {APL, DPM} -> mushroom_body
    4. cls == "CX" -> central_complex
    5. superclass == "cb_motor", or (cls == "gustatory" and superclass == "cb_sensory"), or cls == "SEZPN",
       or type matches ``SEZ_TYPE_REGEX`` -> sez
    6. superclass in {descending_neuron, vnc_motor, vnc_efferent, efferent_descending, sensory_descending}
       -> descending_motor
    7. superclass starts with "vnc" or in {ascending_neuron, sensory_ascending, efferent_ascending,
       sensory_ascending_tbc} -> vnc
    8. else -> central_other
    """
    sc, c, t = _clean(superclass), _clean(cls), _clean(type_)
    if sc in _OL_SUPERCLASSES or c == "visual":
        return REGION_ID["optic_lobe"]
    if c in _AL_CLASSES or _ORN_RE.fullmatch(t) is not None:
        return REGION_ID["antennal_lobe"]
    if c in _MB_CLASSES or t in _MB_TYPES:
        return REGION_ID["mushroom_body"]
    if c == "CX":
        return REGION_ID["central_complex"]
    if sc == "cb_motor" or (c == "gustatory" and sc == "cb_sensory") or c == "SEZPN" or _SEZ_RE.fullmatch(t) is not None:
        return REGION_ID["sez"]
    if sc in _DN_SUPERCLASSES:
        return REGION_ID["descending_motor"]
    if sc.startswith("vnc") or sc in _VNC_SUPERCLASSES:
        return REGION_ID["vnc"]
    return REGION_ID["central_other"]


def side_of(soma_side: str | None, instance: str | None = None) -> int:
    """``'L'`` -> -1, ``'R'`` -> +1, else 0; fallback: ``instance`` endswith ``'_L'`` / ``'_R'``.

    Comparison is case-insensitive after stripping. An explicit midline ``'M'`` returns 0 without
    consulting ``instance`` (RESEARCH section 6: "anything else (M, null, '') -> 0").
    """
    s = _clean(soma_side).upper()
    if s == "L":
        return -1
    if s == "R":
        return 1
    if s == "M":
        return 0
    inst = _clean(instance)
    if inst.endswith("_L"):
        return -1
    if inst.endswith("_R"):
        return 1
    return 0


# --------------------------------------------------------------------------- group resolution

_COMPILED: dict[str, re.Pattern[str]] = {name: re.compile(rx) for name, rx in GROUP_REGEX.items()}


def _type_masks(types: list[str], patterns: dict[str, re.Pattern[str]]) -> dict[str, np.ndarray]:
    """bool[T] per group: which UNIQUE type labels fullmatch the group regex."""
    out: dict[str, np.ndarray] = {}
    for name, pat in patterns.items():
        out[name] = np.fromiter((pat.fullmatch(t) is not None for t in types), dtype=bool, count=len(types))
    return out


def resolve_groups(types: list[str], type_idx: np.ndarray, side: np.ndarray) -> dict[str, np.ndarray]:
    """fullmatch every ``GROUP_REGEX`` over the UNIQUE type list (fast), expand to sorted int32 index
    arrays, add ``'<name>_L'`` / ``'<name>_R'`` for ``SIDED`` names. Missing groups are present as
    empty arrays (never ``KeyError`` downstream).

    The unsuffixed key of a sided group is the union of ALL members (including side 0); ``_L`` /
    ``_R`` hold the members with ``side == -1`` / ``+1`` respectively.
    """
    type_idx = np.asarray(type_idx, dtype=np.int64)
    side = np.asarray(side, dtype=np.int8)
    if type_idx.shape != side.shape:
        raise ValueError(f"type_idx and side must have equal shape, got {type_idx.shape} vs {side.shape}")
    if type_idx.size and (type_idx.min() < 0 or type_idx.max() >= len(types)):
        raise ValueError(f"type_idx out of range [0, {len(types)})")
    masks = _type_masks(list(types), _COMPILED)
    groups: dict[str, np.ndarray] = {}
    left = side == -1
    right = side == 1
    for name in GROUP_REGEX:
        tmask = masks[name]
        if not tmask.any():
            members = np.zeros(0, dtype=np.int32)
            groups[name] = members
            if name in SIDED:
                groups[name + "_L"] = members
                groups[name + "_R"] = members.copy()
            continue
        nmask = tmask[type_idx]
        groups[name] = np.flatnonzero(nmask).astype(np.int32)
        if name in SIDED:
            groups[name + "_L"] = np.flatnonzero(nmask & left).astype(np.int32)
            groups[name + "_R"] = np.flatnonzero(nmask & right).astype(np.int32)
    return groups


def readout_partition(groups: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, list[str]]:
    """``pop_id`` int16[n] (-1 = none) and the list of readout names in ``READOUTS`` order (sided ones
    expanded to ``name, name_L, name_R`` as separate ids so that ``name == name_L + name_R``).

    Partition rule: first match in ``READOUTS`` order wins, so a neuron matched by an earlier
    readout is invisible to later ones (e.g. ``DNg100`` cells belong to ``dng100``, and ``dn_fwd``
    keeps only ``DNge053`` / ``DNg97``). For a SIDED readout, members with side -1 / +1 receive the
    ``name_L`` / ``name_R`` id and only members with side 0 (midline / unknown) receive the
    unsuffixed ``name`` id; the full population count is therefore
    ``counts[name] + counts[name_L] + counts[name_R]`` (see ``fold_sided_counts``), and
    ``readout_sizes`` gives the matching per-name population sizes.
    """
    pop_id = np.full(n, -1, dtype=np.int16)
    names: list[str] = []
    empty = np.zeros(0, dtype=np.int32)
    for name in READOUTS:
        members = np.asarray(groups.get(name, empty), dtype=np.int64)
        base = len(names)
        if name in SIDED:
            names.extend((name, name + "_L", name + "_R"))
        else:
            names.append(name)
        if members.size == 0:
            continue
        free = members[pop_id[members] < 0]
        if free.size == 0:
            continue
        pop_id[free] = base
        if name in SIDED:
            for k, suffix in ((1, "_L"), (2, "_R")):
                sided = np.asarray(groups.get(name + suffix, empty), dtype=np.int64)
                if sided.size:
                    hit = np.intersect1d(free, sided, assume_unique=True)
                    pop_id[hit] = base + k
    if len(names) >= 2**15:  # pragma: no cover - table is ~107 ids
        raise ValueError("too many readout ids for int16")
    return pop_id, names


def readout_sizes(pop_id: np.ndarray, names: list[str]) -> np.ndarray:
    """int64[len(names)] number of neurons the partition assigns to every readout id.

    This is the denominator that matches the spike counts of ``pop_id`` (a readout that lost
    members to an earlier readout, e.g. ``dn_fwd`` without ``DNg100``, is sized accordingly).
    For a sided readout the unsuffixed entry is midline + L + R, i.e. exactly what
    ``fold_sided_counts`` produces for the counts.
    """
    pop_id = np.asarray(pop_id)
    direct = np.bincount(pop_id[pop_id >= 0].astype(np.int64), minlength=len(names)).astype(np.int64)
    return fold_sided_counts(direct, names)


def fold_sided_counts(counts: np.ndarray, names: list[str]) -> np.ndarray:
    """Add the ``name_L`` and ``name_R`` counts into the unsuffixed ``name`` slot (copy returned).

    ``counts`` is what ``np.bincount(pop_id[spk], minlength=len(names))`` yields; after folding,
    ``counts[name] == direct(midline) + name_L + name_R`` for every sided readout.
    """
    out = np.array(counts, copy=True)
    pos = {nm: i for i, nm in enumerate(names)}
    for nm, i in pos.items():
        if nm.endswith("_L") or nm.endswith("_R"):
            base = nm[:-2]
            if base in pos:
                out[pos[base]] += out[i]
    return out


def validate_groups(
    conn: "Connectome",
    required: Iterable[str] = ("grn_sugar", "lc_loom", "gf", "feed_mn", "steer_a02", "dng100", "flight_dn",
                               "epg", "pfl3", "p1", "escape_vnc", "wing_power"),
) -> dict[str, int]:
    """Return ``{group: size}``; raise ``ValueError`` listing every required group that resolved to zero neurons."""
    sizes = {name: int(np.asarray(idx).size) for name, idx in conn.groups.items()}
    missing = [name for name in required if sizes.get(name, 0) == 0]
    if missing:
        raise ValueError(
            "required functional groups resolved to zero neurons: " + ", ".join(missing)
            + f" (connectome {conn.name!r}, n={conn.n}, {len(conn.types)} types)"
        )
    return sizes
