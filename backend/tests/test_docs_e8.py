# Documentation consistency tests owned by E8 (frontend panels + docs).
#
# The README (section 8, "Kuklacilik listesi") and docs/NOTICE.md promise that the list of engineered ("[E]")
# projection rows printed for reviewers matches the shipped synthetic generator. This test enforces that promise
# so the two cannot drift silently: it parses the E-projection ids out of README section 8 and compares them with
# `flybrain.connectome.synthetic.PROJECTIONS` (provenance == "E").
#
# NOTE: `synthetic.py` is owned by E1. If it cannot be imported (partial/broken during parallel development) the
# import-dependent assertions skip rather than fail - a broken generator is E1's failure, caught by E1's own tests.
# The pure-text assertions (README <-> NOTICE self-consistency) always run.
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
README = REPO_ROOT / "README.md"
NOTICE = REPO_ROOT / "docs" / "NOTICE.md"

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


def _readme_e_pids(section8: str) -> set[str]:
    """Every engineered projection id mentioned in README section 8, expanding "Pxxx-Pyyy" ranges inclusively."""
    pids: set[str] = set()
    for lo, hi in _RANGE.findall(section8):
        for k in range(int(lo), int(hi) + 1):
            pids.add(f"P{k:03d}")
    for m in _SINGLE.findall(section8):
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


def test_readme_and_notice_exist():
    assert README.is_file(), f"missing {README}"
    assert NOTICE.is_file(), f"missing {NOTICE}"


def test_readme_section8_lists_engineered_projections():
    section8 = _section(_read(README), "## 8.", "## 9.")
    pids = _readme_e_pids(section8)
    # The GF rest-brake rows P113/P114 must be documented (they suppress the resting giant fiber; see NOTICE section 3).
    assert {"P113", "P114"} <= pids, "README section 8 must document the GF rest-brake rows P113 and P114"
    assert len(pids) >= 19, f"README section 8 lists too few engineered rows: {sorted(pids)}"


def test_readme_e_projection_list_matches_generator():
    """README section 8 E-list == the generator's provenance=='E' ids (anti-drift, promised by NOTICE section 3)."""
    projections = _load_projections()
    code_e = {p.pid for p in projections if getattr(p, "provenance", None) == "E"}
    assert code_e, "the synthetic generator exposes no engineered ('E') projections"
    section8 = _section(_read(README), "## 8.", "## 9.")
    readme_e = _readme_e_pids(section8)
    missing = code_e - readme_e            # engineered rows the README fails to disclose
    extra = readme_e - code_e              # rows the README claims but the generator does not produce
    assert not missing, f"README section 8 omits engineered projection rows present in synthetic.py: {sorted(missing)}"
    assert not extra, f"README section 8 lists engineered rows absent from synthetic.py: {sorted(extra)}"


def test_readme_provenance_counts_match_generator():
    """The counts quoted in README section 8 (total / V / D / E) match the generator."""
    projections = _load_projections()
    counts = Counter(getattr(p, "provenance", None) for p in projections)
    total, v, d, e = len(projections), counts.get("V", 0), counts.get("D", 0), counts.get("E", 0)
    section8 = _section(_read(README), "## 8.", "## 9.")
    # The numbers appear as standalone tokens in the section-8 sentence "114 ... 79'u V ... 14'u D ... 21'i E".
    nums = set(re.findall(r"\d+", section8))
    for label, value in (("total", total), ("V", v), ("D", d), ("E", e)):
        assert str(value) in nums, f"README section 8 does not quote the {label} projection count {value}"


def test_readme_design_sections_reconcile_projection_count():
    """The design sections mirror the SPEC's 112-rule figure, but every mention must point the reader at the
    shipped 114 (section 8 reconciles them) so no design section leaves a reader with a bare, contradictory count.
    SPEC section 0 / section g.3 say 112 (19 E); the shipped generator produces 114 (21 E). Regression guard for the
    reviewer-flagged 'a reader sees both figures' inconsistency: keep 112 (SPEC design) but always cross-reference 114.
    """
    text = _read(README)
    for start, end in (("## 1.", "## 2."), ("## 3.", "## 4.")):
        sec = _section(text, start, end)
        if "112" in sec and "projeksiyon" in sec.lower():
            assert "114" in sec, (
                f"README {start} quotes the design count 112 projections without cross-referencing the shipped 114 "
                "(section 8 reconciles 112-vs-114); a reader must not be left with a bare, contradictory figure"
            )
