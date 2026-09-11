"""Brain summary for the tweet agent (SPEC section c.20 / d.6).

``build_brain_summary`` turns one ``tick`` dict (SPEC d.2) plus a little history and session context into the ONLY
JSON object the LLM ever sees. The shape is fixed (every key always present, numbers rounded as in d.6) and the
serialised size stays <= 1.5 KB so the call costs a fixed, tiny number of input tokens.

Nothing here touches the network or any optional package. All neuron-group labels are stand-ins mapped from the
``RateEstimator`` readout keys of ``tick.rates.pops`` ([E]: the tick carries population rates, not per-type rates,
so ``top_types`` reports the STAR_TYPES representative of the five hottest readouts).
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only (E6 owns server/state.py)
    from flybrain.server.state import TickHistory

__all__ = [
    "SUMMARY_SCHEMA",
    "MAX_SUMMARY_BYTES",
    "RATE_KEYS",
    "DRIVE_KEYS",
    "REGION_KEYS",
    "VOCABULARY_HINT",
    "READOUT_TO_TYPE",
    "build_brain_summary",
    "summary_json",
]

SUMMARY_SCHEMA = "flybrain.summary.v1"
MAX_SUMMARY_BYTES = 1560  # SPEC d.6 / h.2: the LLM payload (compact JSON, UTF-8) never exceeds this.
# Raised from 1500 when market.token_live (the launch disclosure) was added: at 1500 the trim ladder paid for
# the new field by shedding top_types entries, which are the most useful part of the payload for the model.

# The 19 fixed rates_hz keys of SPEC d.6, in wire order; every value comes from tick.rates.pops.
RATE_KEYS: tuple[str, ...] = (
    "grn_sugar", "sugar2_exc", "feed_mn", "pam", "ppl1", "mbon_approach", "mbon_avoid",
    "steer_a02_L", "steer_a02_R", "gf", "lc_loom", "dn_freeze", "dng100", "flight_dn",
    "kc", "epg", "pfl3", "p1", "pip10",
)

DRIVE_KEYS: tuple[str, ...] = ("sugar", "bitter", "looming", "odor", "chop", "courtship", "sleep_pressure", "explore")

REGION_KEYS: tuple[str, ...] = (
    "optic_lobe", "antennal_lobe", "mushroom_body", "central_complex", "sez", "central_other",
    "descending_motor", "vnc",
)

VOCABULARY_HINT: tuple[str, ...] = (
    "sugar GRNs (LB3b/LB3c)", "MN9 proboscis", "PAM dopamine", "mushroom body", "DNa02 steering",
    "giant fiber DNp01", "LC4/LPLC2 looming", "central complex EPG/PFL3",
)

# Readout key -> representative cell type of SPEC c.4 STAR_TYPES ([V] type names from RESEARCH section 4; the mapping
# itself is [E]: the tick carries population rates, so the hottest readouts stand in for the hottest types).
READOUT_TO_TYPE: dict[str, str] = {
    "grn_sugar": "LB3b",
    "feed_mn": "MN9",
    "pam": "PAM01",
    "ppl1": "PPL101",
    "mbon_approach": "MBON01",
    "mbon_avoid": "MBON11",
    "gf": "DNp01",
    "steer_a02": "DNa02",
    "steer_a01": "DNa01",
    "steer_g13": "DNg13",
    "dn_freeze": "DNp09",
    "dng100": "DNg100",
    "flight_dn": "DNg02_a",
    "ttmn": "TTMn",
    "psi": "PSI",
    "pfl3": "PFL3",
    "epg": "EPG",
    "lc4": "LC4",
    "lplc2": "LPLC2",
    "lc_freeze": "LC9",
    "p1": "pC1_14a",
    "pip10": "pIP10",
    "dms2": "dMS2",
    "hg1": "hg1 MN",
    "wing_power": "DLMn c-f",
    "photoreceptor": "R1-R6",
    "lamina": "L1",
    "motion_in": "Mi1",
    "t4t5": "T4a",
    "orn": "ORN_DM1",
    "kc": "KCg-m",
}

# Per-reason ordering of the hints, used when the payload must be trimmed to the byte budget.
_HINTS_BY_REASON: dict[str, tuple[str, ...]] = {
    "euphoria_entry": ("sugar GRNs (LB3b/LB3c)", "MN9 proboscis", "PAM dopamine", "mushroom body",
                       "central complex EPG/PFL3", "DNa02 steering", "giant fiber DNp01", "LC4/LPLC2 looming"),
    "panic_entry": ("giant fiber DNp01", "LC4/LPLC2 looming", "DNa02 steering", "mushroom body",
                    "PAM dopamine", "central complex EPG/PFL3", "sugar GRNs (LB3b/LB3c)", "MN9 proboscis"),
    "escape_burst": ("giant fiber DNp01", "LC4/LPLC2 looming", "DNa02 steering", "central complex EPG/PFL3",
                     "mushroom body", "PAM dopamine", "sugar GRNs (LB3b/LB3c)", "MN9 proboscis"),
    "courtship_entry": ("PAM dopamine", "central complex EPG/PFL3", "mushroom body", "DNa02 steering",
                        "sugar GRNs (LB3b/LB3c)", "MN9 proboscis", "giant fiber DNp01", "LC4/LPLC2 looming"),
}

_MAX_EVENTS = 6
_MAX_NOTE = 96
_MAX_NAME = 48
_MAX_LABEL = 24       # mood / regime / mode / source style enum strings copied from the tick
_MAX_REASON = 32      # reason, session.last_tweet_reason
_MAX_EVENT_LABEL = 40
_SHORT_SYNTH_NOTE = "synthetic stand-in, not real data"

#: ``(container key path, field)`` of every variable-length string the payload copies from its inputs. The hard guard
#: of ``_fit`` shrinks these (longest first) so ``build_brain_summary`` can never exceed ``MAX_SUMMARY_BYTES``.
_STRING_FIELDS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((), "reason"), ((), "lang"),
    (("mood",), "state"), (("mood",), "prev"),
    (("market",), "source"), (("market",), "symbol"), (("market",), "regime"),
    (("market", "last_trade"), "kind"),
    (("fly",), "mode"),
    (("connectome",), "source"), (("connectome",), "name"), (("connectome",), "note"),
    (("session",), "last_tweet_reason"),
)


def _cap(x: Any, limit: int, default: str = "") -> str:
    """``str(x)`` capped at ``limit`` characters (``None``/empty -> ``default``)."""
    s = default if x is None else str(x)
    if not s:
        s = default
    return s[:limit]


def summary_json(summary: dict) -> str:
    """The canonical serialisation the LLM receives (compact separators, UTF-8, no ASCII escaping)."""
    return json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _nbytes(summary: dict) -> int:
    return len(summary_json(summary).encode("utf-8"))


def _intify(obj: Any) -> Any:
    """Integral floats -> int (``131.0`` -> ``131``) recursively; fewer bytes, same meaning for a reader."""
    if isinstance(obj, float):
        return int(obj) if obj.is_integer() and abs(obj) < 1e15 else obj
    if isinstance(obj, dict):
        return {k: _intify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_intify(v) for v in obj]
    return obj


def _intify_inplace(obj: Any) -> None:
    """``_intify`` that mutates nested containers in place, so a held reference stays valid (no stale closures)."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (dict, list)):
                _intify_inplace(v)
            else:
                obj[k] = _intify(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                _intify_inplace(v)
            else:
                obj[i] = _intify(v)


def _round_inplace(block: dict, nd: int) -> None:
    for k, v in list(block.items()):
        if isinstance(v, float):
            block[k] = _intify(round(v, nd))


def _at(summary: dict, path: tuple[str, ...]) -> dict | None:
    node: Any = summary
    for key in path:
        node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict):
            return None
    return node if isinstance(node, dict) else None


def _hard_trim(summary: dict) -> dict:
    """Last resort after the ladder: shrink the longest copied string (halving, floor 4 chars) until the payload fits.
    Every key stays present; only pathological inputs (a 300-char ``market.regime``) ever reach this."""
    for _ in range(256):
        if _nbytes(summary) <= MAX_SUMMARY_BYTES:
            return summary
        best: tuple[int, dict, str] | None = None
        for path, field in _STRING_FIELDS:
            node = _at(summary, path)
            if node is None:
                continue
            val = node.get(field)
            if isinstance(val, str) and len(val) > 4 and (best is None or len(val) > best[0]):
                best = (len(val), node, field)
        if best is None:
            return summary
        _, node, field = best
        node[field] = node[field][: max(4, len(node[field]) // 2)]
    return summary


def _fit(summary: dict) -> dict:
    """Trim ladder: the exact d.6 shape can exceed 1.5 KB (its own example is ~1.7 KB minified), so shave bytes in
    order of least information lost until the compact payload fits ``MAX_SUMMARY_BYTES``. Every key stays present and
    at least one vocabulary hint and one ``top_types`` entry survive; ``connectome.note`` (the synthetic-data label the
    system prompt keys on) is the LAST thing touched, so d.6's "copied from meta['note']" holds for every realistic
    payload. ``_hard_trim`` then guarantees the budget unconditionally."""
    if _nbytes(summary) <= MAX_SUMMARY_BYTES:
        return summary
    reason = str(summary.get("reason") or "manual")
    hints = list(_HINTS_BY_REASON.get(reason, VOCABULARY_HINT))

    def vocab(n: int):
        return lambda s: s.__setitem__("vocabulary_hint", hints[:n])

    def events(n: int):
        return lambda s: s.__setitem__("events_10s", s["events_10s"][-n:])

    def top(n: int):
        return lambda s: s.__setitem__("top_types", s["top_types"][:n])

    def intify(s: dict) -> None:
        _intify_inplace(s)

    def coarse_market(s: dict) -> None:
        m = s["market"]
        for k in ("vol_m5", "liq_usd", "mcap"):
            if isinstance(m.get(k), float):
                m[k] = int(round(m[k]))
        s["session"]["uptime_s"] = int(round(_num(s["session"].get("uptime_s"))))
        _round_inplace(s["drives"], 1)

    def coarse_rates(s: dict) -> None:
        _round_inplace(s["rates_hz"], 0)
        _round_inplace(s["regions_hz"], 0)
        s["top_types"] = [[lab, _intify(round(_num(hz), 0))] for lab, hz in s["top_types"]]

    def short_name(s: dict) -> None:
        conn = s["connectome"]
        conn["name"] = str(conn.get("name") or "")[:24]

    def short_note(s: dict) -> None:
        conn = s["connectome"]
        note = str(conn.get("note") or "")
        conn["note"] = _SHORT_SYNTH_NOTE if "synthetic" in note.lower() else note[:40]

    # Order = least information lost first. The hints are the cheapest thing to drop (SYSTEM_PROMPT already names the
    # same groups), so they go down to 1 before any real measurement (top_types, events, rate precision) is touched.
    steps = (intify, coarse_market, events(4), vocab(4), vocab(2), coarse_rates, vocab(1), top(4), events(2),
             top(3), top(2), short_name, short_note)
    for step in steps:
        step(summary)
        if _nbytes(summary) <= MAX_SUMMARY_BYTES:
            return summary
    return _hard_trim(summary)


def _num(x: Any, default: float = 0.0) -> float:
    """float(x) with None/NaN/inf/garbage -> default."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _r(x: Any, nd: int, default: float = 0.0) -> float:
    return round(_num(x, default), nd)


def _price(x: Any) -> float | None:
    if x is None:
        return None
    v = _num(x, -1.0)
    return None if v < 0 else v


def _d(obj: Any, key: str) -> dict:
    v = obj.get(key) if isinstance(obj, dict) else None
    return v if isinstance(v, dict) else {}


def _event_label(ev: Any) -> str | None:
    if not isinstance(ev, dict):
        return None
    kind = str(ev.get("kind", "") or "")
    if not kind:
        return None
    data = ev.get("data")
    if kind == "mood" and isinstance(data, dict):
        src = _cap(data.get("from"), _MAX_LABEL, "?")
        dst = _cap(data.get("to"), _MAX_LABEL, "?")
        return f"mood:{src}->{dst}"
    return kind[:_MAX_EVENT_LABEL]


def _top_types(pops: dict) -> list[list[Any]]:
    """Five hottest readouts that map to a STAR_TYPES label, as ``[label, hz]`` pairs (desc by Hz)."""
    scored: list[tuple[float, str]] = []
    for key, label in READOUT_TO_TYPE.items():
        if key in pops:
            scored.append((_num(pops.get(key)), label))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [[label, round(hz, 1)] for hz, label in scored[:5]]


def _trail_px(session: dict, history: Any) -> int:
    """Total trail length in px: ``session['trail_px']`` when the orchestrator knows it, else the path length of
    the last 10 s of tick positions (never raises)."""
    v = session.get("trail_px") if isinstance(session, dict) else None
    if v is not None:
        return int(max(0.0, _num(v)))
    try:
        ticks = history.last(10.0) if history is not None else []
    except Exception:
        ticks = []
    total = 0.0
    px = py = None
    for t in ticks:
        fly = _d(t, "fly")
        x, y = _num(fly.get("x")), _num(fly.get("y"))
        if px is not None:
            d = math.hypot(x - px, y - py)
            if d < 100.0:  # ignore wraps / teleports
                total += d
        px, py = x, y
    return int(total)


def build_brain_summary(tick: dict, history: "TickHistory", reason: str, conn_meta: dict, session: dict,
                        market: dict, lang: str) -> dict:
    """The ONLY thing the LLM sees (<= 1.5 KB). Exact shape in SPEC section d.6.

    ``tick`` is the d.2 tick dict, ``history`` a TickHistory (``events``/``gf_spikes``/``jumps``/``last``; duck-typed,
    every call is guarded), ``conn_meta`` the connectome meta (source/name/n/e/note), ``session`` the orchestrator's
    ``{uptime_s, tweets_today, last_tweet_reason[, trail_px]}`` and ``market`` the d.5 market body (``tick.market``
    is used when ``market`` is empty). Missing inputs never raise: every key is present with a neutral value.
    The result is trimmed (see ``_fit``) so that ``summary_json(result)`` is at most ``MAX_SUMMARY_BYTES``.
    """
    tick = tick if isinstance(tick, dict) else {}
    conn_meta = conn_meta if isinstance(conn_meta, dict) else {}
    session = session if isinstance(session, dict) else {}
    mkt = market if isinstance(market, dict) and market else _d(tick, "market")

    mood = _d(tick, "mood")
    drives = _d(tick, "drives")
    rates = _d(tick, "rates")
    pops = _d(rates, "pops")
    fly = _d(tick, "fly")

    # -- history (all guarded: a bare stub or None must not break a tweet)
    try:
        raw_events = list(history.events(10.0)) if history is not None else []
    except Exception:
        raw_events = []
    events_10s: list[str] = []
    for ev in raw_events:
        lab = _event_label(ev)
        if lab is not None and (not events_10s or events_10s[-1] != lab):
            events_10s.append(lab)
    events_10s = events_10s[-_MAX_EVENTS:]
    try:
        gf_10min = int(history.gf_spikes(600.0)) if history is not None else 0
    except Exception:
        gf_10min = 0
    try:
        jumps_60s = int(history.jumps(60.0)) if history is not None else 0
    except Exception:
        jumps_60s = 0

    # -- regions: list in REGIONS order (d.2) or a dict
    regions_raw = rates.get("regions")
    regions_hz: dict[str, float] = {}
    if isinstance(regions_raw, dict):
        for k in REGION_KEYS:
            regions_hz[k] = _r(regions_raw.get(k), 1)
    else:
        seq = list(regions_raw) if isinstance(regions_raw, (list, tuple)) else []
        for i, k in enumerate(REGION_KEYS):
            regions_hz[k] = _r(seq[i], 1) if i < len(seq) else 0.0

    last_trade = mkt.get("last_trade")
    if isinstance(last_trade, dict) and last_trade.get("kind"):
        lt: dict | None = {"kind": _cap(last_trade.get("kind"), _MAX_LABEL), "usd": _r(last_trade.get("usd"), 1)}
    else:
        lt = None

    note = str(conn_meta.get("note") or "")
    if len(note) > _MAX_NOTE:
        note = note[: _MAX_NOTE - 3].rstrip() + "..."
    name = str(conn_meta.get("name") or "")[:_MAX_NAME]

    regime = mkt.get("regime")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "reason": _cap(reason, _MAX_REASON, "manual"),
        "lang": _cap(lang, 8, "en"),
        "mood": {
            "state": _cap(mood.get("state"), _MAX_LABEL, "CRUISING"),
            "prev": _cap(mood.get("prev"), _MAX_LABEL, "CRUISING"),
            "since_s": _r(_num(mood.get("since_ms")) / 1000.0, 1),
            "euphoria": _r(mood.get("euphoria"), 2),
            "anxiety": _r(mood.get("anxiety"), 2),
            "valence": _r(mood.get("valence"), 2),
            "arousal": _r(mood.get("arousal"), 2),
            "hunger": _r(mood.get("hunger"), 2),
        },
        "market": {
            "source": _cap(mkt.get("source"), _MAX_LABEL, "sim"),
            "symbol": _cap(mkt.get("symbol"), 16, "FLY"),
            # FLY_TOKEN_LIVE, false unless the operator explicitly said the tracked pair IS this project's token.
            # While it is false the whole market block belongs to a third-party stand-in pair: the system prompt
            # and the templates must not present these numbers as this project's own price.
            "token_live": bool(mkt.get("token_live", False)),
            "price_usd": _price(mkt.get("price_usd")),
            "chg_m5": _r(mkt.get("chg_m5"), 1),
            "chg_h1": _r(mkt.get("chg_h1"), 1),
            "chg_h24": _r(mkt.get("chg_h24"), 1),
            "buys_m5": int(_num(mkt.get("buys_m5"))),
            "sells_m5": int(_num(mkt.get("sells_m5"))),
            "vol_m5": _r(mkt.get("vol_m5"), 1),
            "liq_usd": _price(mkt.get("liq_usd")),
            "mcap": _price(mkt.get("mcap")),
            "regime": None if regime is None else _cap(regime, _MAX_LABEL),
            "last_trade": lt,
        },
        "drives": {k: _r(drives.get(k), 2) for k in DRIVE_KEYS},
        "rates_hz": {k: _r(pops.get(k), 1) for k in RATE_KEYS},
        "regions_hz": regions_hz,
        "top_types": _top_types(pops),
        "events_10s": events_10s,
        "gf_spikes_10min": gf_10min,
        "jumps_60s": jumps_60s,
        "fly": {
            # d.6 shows 412.3 -> 412 and 233.9 -> 233, i.e. truncation toward zero (not rounding).
            "mode": _cap(fly.get("mode"), _MAX_LABEL, "walk"),
            "x": int(_num(fly.get("x"))),
            "y": int(_num(fly.get("y"))),
            "speed": _r(fly.get("speed"), 1),
            "wing_hz": _r(fly.get("wing_hz"), 1),
            "proboscis": _r(fly.get("proboscis"), 2),
            "trail_px": _trail_px(session, history),
        },
        "connectome": {
            "source": _cap(conn_meta.get("source"), _MAX_LABEL, "synthetic"),
            "name": name,
            "n": int(_num(conn_meta.get("n"))),
            "e": int(_num(conn_meta.get("e"))),
            "note": note,
        },
        "session": {
            "uptime_s": _r(session.get("uptime_s"), 1),
            "tweets_today": int(_num(session.get("tweets_today"))),
            "last_tweet_reason": (None if session.get("last_tweet_reason") is None
                                  else _cap(session.get("last_tweet_reason"), _MAX_REASON)),
        },
        "vocabulary_hint": list(VOCABULARY_HINT),
    }
    return _fit(summary)
