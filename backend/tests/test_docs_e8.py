# Documentation consistency tests owned by E8 (frontend panels + docs).
#
# `docs/PUPPETEERING.md` and `docs/NOTICE.md` promise that the list of engineered ("[E]") projection rows printed for
# reviewers matches the shipped synthetic generator. This test enforces that promise so the two cannot drift silently:
# it parses the E-projection ids out of PUPPETEERING section 3 and compares them with
# `flybrain.connectome.synthetic.PROJECTIONS` (provenance == "E").
#
# The disclosure used to be section 8 of a Turkish engineering README. That README was replaced by the short public
# README (commit 0f0d804), which has no numbered sections at all, so the E-list moved to `docs/PUPPETEERING.md` and
# these markers follow it there; the README and NOTICE section 3 link to it.
#
# NOTE: `synthetic.py` is owned by E1. If it cannot be imported (partial/broken during parallel development) the
# import-dependent assertions skip rather than fail - a broken generator is E1's failure, caught by E1's own tests.
# The pure-text assertions always run.
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
README = REPO_ROOT / "README.md"
NOTICE = REPO_ROOT / "docs" / "NOTICE.md"
PUPPET = REPO_ROOT / "docs" / "PUPPETEERING.md"

#: The E-list lives in PUPPETEERING section 3 ("The 21 engineered projection rows"); section 4 reconciles 112 vs 114.
E_LIST_MARKERS = ("## 3.", "## 4.")

# A projection id token "Pnnn", not preceded by a letter/digit (so the inner "P020" of "PVLP020" is not matched)
# and not followed by another digit.
_PID = r"(?<![A-Za-z0-9])P(\d{3})(?![0-9])"
_RANGE = re.compile(r"(?<![A-Za-z0-9])P(\d{3})\s*[–—-]\s*P(\d{3})(?![0-9])")
_SINGLE = re.compile(_PID)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _section(text: str, start_marker: str, end_marker: str) -> str:
    i = text.find(start_marker)
    assert i >= 0, f"marker not found: {start_marker!r}"
    j = text.find(end_marker, i + len(start_marker))
    return text[i : (j if j >= 0 else len(text))]


def _doc_e_pids(section: str) -> set[str]:
    """Every engineered projection id mentioned in the E-list section, expanding "Pxxx-Pyyy" ranges inclusively."""
    pids: set[str] = set()
    for lo, hi in _RANGE.findall(section):
        for k in range(int(lo), int(hi) + 1):
            pids.add(f"P{k:03d}")
    for m in _SINGLE.findall(section):
        pids.add(f"P{int(m):03d}")
    return pids


def _load_projections():
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
    try:
        from flybrain.connectome.synthetic import PROJECTIONS  # type: ignore
    except Exception as exc:  # ImportError, or a partial file that fails at import time (E1 territory)
        pytest.skip(f"flybrain.connectome.synthetic not importable yet (owned by E1): {exc}")
    return PROJECTIONS


def test_docs_exist_and_cross_reference_the_e_list():
    assert README.is_file(), f"missing {README}"
    assert NOTICE.is_file(), f"missing {NOTICE}"
    assert PUPPET.is_file(), f"missing {PUPPET}"
    # The E-list is only a disclosure if a reader can find it from the two documents people actually open.
    assert "PUPPETEERING.md" in _read(NOTICE), "NOTICE must point at the engineered-row list"
    assert "PUPPETEERING.md" in _read(README), "README must point at the engineered-row list"


def test_puppeteering_lists_engineered_projections():
    section = _section(_read(PUPPET), *E_LIST_MARKERS)
    pids = _doc_e_pids(section)
    # The GF rest-brake rows P113/P114 must be documented (they suppress the resting giant fiber; see NOTICE section 3).
    assert {"P113", "P114"} <= pids, "PUPPETEERING section 3 must document the GF rest-brake rows P113 and P114"
    assert len(pids) >= 19, f"PUPPETEERING section 3 lists too few engineered rows: {sorted(pids)}"


def test_doc_e_projection_list_matches_generator():
    """The doc's E-list == the generator's provenance=='E' ids (anti-drift, promised by NOTICE section 3)."""
    projections = _load_projections()
    code_e = {p.pid for p in projections if getattr(p, "provenance", None) == "E"}
    assert code_e, "the synthetic generator exposes no engineered ('E') projections"
    section = _section(_read(PUPPET), *E_LIST_MARKERS)
    doc_e = _doc_e_pids(section)
    missing = code_e - doc_e               # engineered rows the doc fails to disclose
    extra = doc_e - code_e                 # rows the doc claims but the generator does not produce
    assert not missing, f"PUPPETEERING omits engineered projection rows present in synthetic.py: {sorted(missing)}"
    assert not extra, f"PUPPETEERING lists engineered rows absent from synthetic.py: {sorted(extra)}"


def test_doc_provenance_counts_match_generator():
    """The counts quoted in the E-list section (total / V / D / E) match the generator."""
    projections = _load_projections()
    counts = Counter(getattr(p, "provenance", None) for p in projections)
    total, v, d, e = len(projections), counts.get("V", 0), counts.get("D", 0), counts.get("E", 0)
    section = _section(_read(PUPPET), *E_LIST_MARKERS)
    # The numbers appear as standalone tokens in the sentence "114 projection rules: 79 V, 14 D and 21 E".
    nums = set(re.findall(r"\d+", section))
    for label, value in (("total", total), ("V", v), ("D", d), ("E", e)):
        assert str(value) in nums, f"PUPPETEERING section 3 does not quote the {label} projection count {value}"


def test_docs_reconcile_the_projection_count():
    """SPEC section 0 / section g.3 say 112 rules (19 E); the shipped generator produces 114 (21 E). Regression guard
    for the reviewer-flagged "a reader sees both figures" inconsistency: any doc that quotes 112 in a projection
    context must cross-reference the shipped 114, so no reader is left with a bare, contradictory figure."""
    assert "112" in _read(PUPPET), "PUPPETEERING must name the SPEC design count 112"
    for doc in (PUPPET, NOTICE, README):
        text = _read(doc)
        for m in re.finditer(r"(?<![0-9])112(?![0-9])", text):
            window = text[max(0, m.start() - 300):m.end() + 300]
            if "projection" not in window.lower():
                continue
            assert "114" in window, (
                f"{doc.name} quotes the design count 112 projections without cross-referencing the shipped 114; "
                "a reader must not be left with a bare, contradictory figure"
            )
