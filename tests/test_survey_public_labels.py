"""Tests for the public-label screen.

The screen decides whether a GPU retrain gets spent on a foreign label set, and
labels are the only lever that has ever paid on this board (+0.089, +0.077). The
ways it can mislead are the ways that cost:

1. **Reading the exact-cell rate without its floor.** The gold set is 34.5%
   positive, so answering zero to everything already reproduces 65.5% of cells.
   Without that control a coarse but honest labeler looks like an answer key —
   `lixin73` reads 79.9% and is merely ordinary. E080's leaked sets read 100%.
2. **Comparing a candidate against zero rather than the incumbent.** This project
   already banks 0.8927, and `PATH.md` §2.3 warns that crediting a candidate with
   a gain already held is how a null gets read as a win.
3. **Treating "no gold overlap" as a failure.** It is a real category: a set can
   be clean by construction and unscoreable by construction (E088).

No patient data: every value here is written for the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eda.survey_public_labels import (  # noqa: E402
    FINDINGS,
    INCUMBENT_GOLD_AUC,
    screen,
    verdict,
)


def _gold(rows):
    return pd.DataFrame([{"StudyInstanceUID": f"s{i}", **dict(zip(FINDINGS, r, strict=True))}
                         for i, r in enumerate(rows)])


def _cand(ids, rows):
    return pd.DataFrame([{"StudyInstanceUID": s, **dict(zip(FINDINGS, r, strict=True))}
                         for s, r in zip(ids, rows, strict=True)])


ALT = [1, 0] * 6
INV = [0, 1] * 6


def test_the_floor_is_what_all_zeros_scores():
    """The control E080's rule never stated."""
    gold = _gold([ALT, ALT])
    result = screen(gold, _cand(["s0", "s1"], [[0] * 12, [0] * 12]))
    assert result["floor"] == pytest.approx(0.5)
    assert result["exact"] == pytest.approx(result["floor"])


def test_an_answer_key_reproduces_every_cell():
    gold = _gold([ALT, INV])
    result = screen(gold, _cand(["s0", "s1"], [ALT, INV]))
    assert result["exact"] == pytest.approx(1.0)
    assert verdict(result).startswith("ANSWER KEY")


def test_an_ordinary_labeler_above_the_floor_is_not_an_answer_key():
    """79.9% against a 65.5% floor is accuracy, not a leak."""
    assert not verdict({"n_gold": 58, "exact": 0.799, "floor": 0.655,
                        "macro_auc": 0.8352}).startswith("ANSWER KEY")


def test_a_candidate_behind_the_incumbent_is_not_an_upgrade():
    assert verdict({"n_gold": 58, "exact": 0.0, "floor": 0.655,
                    "macro_auc": INCUMBENT_GOLD_AUC - 0.01}).startswith("BEHIND")
    assert verdict({"n_gold": 58, "exact": 0.0, "floor": 0.655,
                    "macro_auc": INCUMBENT_GOLD_AUC + 0.01}).startswith("AHEAD")


def test_no_gold_overlap_is_a_category_not_a_failure(tmp_path):
    gold = _gold([ALT, INV])
    result = screen(gold, _cand(["other0", "other1"], [ALT, INV]))
    assert result["n_gold"] == 0
    assert "floor" in result
    assert verdict(result).startswith("UNSCOREABLE")


def test_a_set_missing_the_finding_columns_is_reported_not_scored():
    gold = _gold([ALT])
    result = screen(gold, pd.DataFrame({"StudyInstanceUID": ["s0"], "ACL": [1]}))
    assert "error" in result


def test_a_missing_auc_is_said_not_dressed_as_a_negative():
    """`nan >= incumbent` is False, so an unguarded nan prints as BEHIND — a
    measurement that could not be made, reported as one that came back bad."""
    gold = _gold([ALT, ALT])                      # every finding single-class
    result = screen(gold, _cand(["s0", "s1"], [[0] * 12, [0] * 12]))
    assert "macro_auc" not in result
    assert verdict(result).startswith("NO AUC")
