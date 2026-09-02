"""Tests for the distillation gate.

The gate decides whether ~7.5 GPU-h gets spent, so the ways it can be wrong are
expensive in one direction and silently misleading in the other:

1. **Overlapping folds.** If two folds predict the same study, the predictions
   are not out-of-fold, the teacher carries leaked information, and the gold
   score that authorises the spend is inflated. This must be refused loudly.
2. **A fitted weight.** An argmax over the blend curve on 58 studies is a free
   parameter fitted to 58 studies. E048 declined exactly that and the script
   must not quietly adopt it.

No patient data: every study ID here is written for the test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("pandas")

from eda.distill_teacher import macro, main, ranked  # noqa: E402
from src.report_schema import FINDINGS  # noqa: E402


def _dump(path: Path, fold: int, studies: list[str], rng) -> None:
    path.write_text(json.dumps({
        "fold": fold, "epoch": 20, "backbone": "resnet34", "source": "t",
        "findings": FINDINGS,
        "studies": studies,
        "predicted": rng.random((len(studies), len(FINDINGS))).round(5).tolist(),
    }))


def test_ranks_are_order_preserving_and_bounded():
    values = np.array([[3.0], [1.0], [2.0]])
    assert ranked(values).ravel().tolist() == [1.0, 0.0, 0.5]


def test_macro_ignores_a_finding_with_one_class():
    """A finding with no positives among the 58 is unscorable, not zero."""
    expert = np.zeros((4, len(FINDINGS)))
    expert[:2, 0] = 1.0                      # only finding 0 has both classes
    score = np.tile(np.array([[0.9], [0.8], [0.2], [0.1]]), (1, len(FINDINGS)))
    assert macro(expert, score) == pytest.approx(1.0)


def test_overlapping_folds_are_refused(tmp_path, capsys):
    """Two folds predicting one study means the split leaked. Refuse, loudly."""
    rng = np.random.default_rng(0)
    _dump(tmp_path / "a.json", 0, ["s1", "s2"], rng)
    _dump(tmp_path / "b.json", 1, ["s2", "s3"], rng)      # s2 twice
    with pytest.raises(SystemExit) as caught:
        main(["--oof", str(tmp_path / "a.json"), str(tmp_path / "b.json"),
              "--train", str(tmp_path / "train.csv")])
    assert "not out-of-fold" in str(caught.value)


def test_finding_order_mismatch_is_refused(tmp_path):
    """Column order is the whole meaning of these arrays."""
    rng = np.random.default_rng(0)
    path = tmp_path / "a.json"
    _dump(path, 0, ["s1"], rng)
    blob = json.loads(path.read_text())
    blob["findings"] = list(reversed(FINDINGS))
    path.write_text(json.dumps(blob))
    with pytest.raises(SystemExit) as caught:
        main(["--oof", str(path), "--train", str(tmp_path / "train.csv")])
    assert "finding order" in str(caught.value)


def test_the_weight_curve_is_recorded_and_not_adopted():
    """The deciding arm must stay the parameter-free 50/50 union."""
    source = (REPO_ROOT / "eda" / "distill_teacher.py").read_text()
    assert "RECORDED AND NOT USED" in source
    assert "union = (model_ranks + teacher_ranks) / 2" in source
    # the word "argmax" appears in the prose explaining why it is refused;
    # what must not appear is a CALL that picks a weight by maximising gold
    assert "np.argmax" not in source
    assert ".argmax(" not in source
    for line in source.splitlines():
        code = line.split("#", 1)[0]
        assert "max(" not in code or "max(len(" in code, line
