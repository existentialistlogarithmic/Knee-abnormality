"""Tests for the public-checkpoint screen.

The screen decides whether hours of cache-building get spent on a foreign
checkpoint family, so the ways it can mislead are the ways that cost:

1. **Comparing a single foreign fold against our five-fold pool.** Our pool is
   0.8980 and one of our folds is 0.8477; judging a foreign single fold against
   the pool would understate it by the entire width of the ensembling effect and
   reject good members. The default comparator is per-fold for that reason.
2. **Treating a self-reported number as a measurement.** It is the author's, on
   the author's split, and may include gold studies their model trained on.

No patient data: every value here is written for the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from eda.survey_public_checkpoints import (  # noqa: E402
    COMPARABLE_WITHIN,
    INCUMBENT_PER_FOLD,
    INCUMBENT_POOLED,
    describe,
    main,
    verdict,
)


def test_the_two_comparators_are_not_interchangeable():
    """0.8980 is a pool of five; 0.8477 is one fold. Using the wrong one moves
    the verdict by more than the band it is being compared against."""
    assert INCUMBENT_POOLED - INCUMBENT_PER_FOLD > COMPARABLE_WITHIN


@pytest.mark.parametrize("gap,expected", [
    (0.00, "COMPARABLE"), (0.02, "COMPARABLE"),
    (0.025, "MARGINAL"), (0.03, "MARGINAL"),
    (0.044, "BEHIND"), (0.10, "BEHIND"),
])
def test_the_band_matches_what_the_experiments_measured(gap, expected):
    assert verdict(gap).startswith(expected)


def test_an_unscored_checkpoint_is_unknown_not_behind():
    """Absence of a self-reported number is not evidence of a bad model — it
    only means the free screen cannot see it."""
    assert verdict(None).startswith("UNKNOWN")


def test_describe_reads_metadata_without_building_a_model(tmp_path):
    path = tmp_path / "m.pt"
    torch.save({"model_state_dict": {"w": torch.zeros(2)},
                "backbone": "convnext_small", "image_size": 224,
                "num_slices": 12, "fold": 3, "auc_gold": 0.7448}, path)
    found = describe(path)
    assert found["backbone"] == "convnext_small"
    assert found["image_size"] == 224
    assert found["auc_gold"] == pytest.approx(0.7448)
    assert found["tensors"] == 1


def test_a_checkpoint_that_says_nothing_is_handled(tmp_path):
    """Plenty of shared checkpoints are a bare state dict."""
    path = tmp_path / "bare.pt"
    torch.save({"w": torch.zeros(2)}, path)
    found = describe(path)
    assert found["file"] == "bare.pt"
    assert "auc_gold" not in found


def test_the_screen_reports_when_it_cannot_screen(tmp_path, capsys):
    path = tmp_path / "bare.pt"
    torch.save({"conv.weight": torch.zeros(2)}, path)
    main(["--checkpoints", str(path)])
    out = capsys.readouterr().out
    assert "cannot be screened for free" in out.replace("\n", " ")


def test_the_screen_labels_every_number_as_self_reported(tmp_path, capsys):
    path = tmp_path / "m.pt"
    torch.save({"model_state_dict": {"w": torch.zeros(2)},
                "backbone": "convnext_small", "auc_gold": 0.80}, path)
    main(["--checkpoints", str(path)])
    out = capsys.readouterr().out
    assert "SELF-REPORTED" in out
    assert "licence is CC0 or Apache" in out
