"""Tests for the gold split of the CoAtNet kernel.

E090 recorded that this project has **no offline gold evaluation** for the CC0
CoAtNet arms — every number about them is their authors' self-report, read out
of the checkpoint's own `gold_auc` tensor. E097 then bought a board submission
to learn that four arms beat one by +0.004, and E098 closed six foreign blends
on an instrument (gold-58) that has never once seen a blend gain and had no way
to be checked.

`knee-gold-raptorcc0x4` is that check. It runs the identical template against
the 58 expert-labelled *training* studies and writes a scored dump instead of a
submission. The failure modes it introduces are new, and these are them:

1. **Writing a submission from training data.** 58 training studies formatted as
   a submission is a well-formed file that scores against a hidden set it does
   not contain. The kernel writes no `submission.csv` at all, so the notebook
   cannot be submitted — asserted here, because a later edit restoring the write
   would not fail anything else.
2. **Scoring the wrong 58.** `train.csv` has 4,407 rows and exactly 58 with all
   twelve findings filled. A schema change that left 57, or that filled the rest
   with zeros, still produces a plausible AUC. `GOLD_EXPECTED` is asserted
   against the competition's own file.
3. **An AUC that is subtly wrong on ties.** A member that returns one value for
   many studies must score 0.5 on that finding, not whatever `argsort` order
   happens to give. That is exactly the degenerate case the dump exists to catch,
   so the tie handling is pinned against hand-computed values.

No patient data: every array here is written for the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import all_kernels  # noqa: E402

GOLD_SLUG = "knee-gold-raptorcc0x4"
TEST_SLUGS = ("knee-infer-raptorcc0", "knee-infer-raptorcc0x4")
GENERATED = REPO_ROOT / "kaggle" / "83_gold_raptorcc0x4" / "run.py"


def _kernel(slug):
    matches = [k for k in all_kernels() if k.slug == slug]
    assert matches, f"{slug} is not in the manifest"
    return matches[0]


def _load(name):
    """Pull one function out of the generated kernel by source slicing.

    Importing the module would execute its top-level `import torch, timm`, which
    is not what is under test and is not installed everywhere this suite runs.
    """
    src = GENERATED.read_text()
    start = src.index(f"def {name}(")
    end = src.index("\ndef ", start + 1)
    ns = {"np": np}
    # `macro_auc` calls `auc`, so the slice has to bring its dependency with it.
    # Loading them into one namespace also means the test exercises the pair the
    # kernel actually uses rather than a reimplementation of either.
    for dep in ("auc", "macro_auc"):
        d0 = src.index(f"def {dep}(")
        exec(compile(src[d0:src.index("\ndef ", d0 + 1)], str(GENERATED), "exec"), ns)  # noqa: S102
    exec(compile(src[start:end], str(GENERATED), "exec"), ns)  # noqa: S102
    return ns[name]


# --------------------------------------------------------------------------- #
# The split is declared, and the other kernels are not silently switched with it
# --------------------------------------------------------------------------- #
def test_the_gold_kernel_declares_the_gold_split():
    assert _kernel(GOLD_SLUG).constants["EVAL_SPLIT"] == "gold"


@pytest.mark.parametrize("slug", TEST_SLUGS)
def test_the_submitting_kernels_still_declare_the_test_split(slug):
    """The hazard runs both ways. A default that flipped to "gold" would make the
    two submission kernels write no submission, which reads on Kaggle as a failed
    run rather than as a wrong one."""
    assert _kernel(slug).constants["EVAL_SPLIT"] == "test"


def test_the_gold_kernel_mounts_no_extra_dataset():
    """The gold labels come from the competition's own `train.csv`. If this ever
    needs a fourth mount, the instrument has stopped being free and something
    about the split has changed."""
    assert set(_kernel(GOLD_SLUG).datasets) == set(_kernel("knee-infer-raptorcc0x4").datasets)


def test_the_gold_kernel_runs_the_same_four_arms_as_the_submitting_blend():
    """It is only an instrument for E097's blend if it is E097's blend."""
    gold = _kernel(GOLD_SLUG).constants["ARMS"]
    board = _kernel("knee-infer-raptorcc0x4").constants["ARMS"]
    assert gold == board


# --------------------------------------------------------------------------- #
# It cannot be submitted
# --------------------------------------------------------------------------- #
def test_the_gold_branch_writes_no_submission():
    """Asserted on the generated source, not on the spec, because this is a
    property of the code path rather than of a constant."""
    src = GENERATED.read_text()
    gold_branch = src[src.index('if EVAL_SPLIT == "gold":', src.index("scores = None")):
                      src.index("    else:\n        sub = pd.DataFrame")]
    # The literal string appears in the branch's own comment explaining why it is
    # absent, so what is asserted is the WRITE, not the mention.
    assert 'to_csv("/kaggle/working/submission.csv"' not in gold_branch, (
        "the gold branch writes a submission built from 58 TRAINING studies")
    for artefact in ("gold_probs.csv", "gold_truth.csv", "gold_scores.json"):
        assert artefact in gold_branch, f"{artefact} is not written"


def test_the_gold_dump_is_raw_per_arm_probabilities_not_the_blend():
    """The whole point is searching blend weights offline. A dump of the blended
    column cannot answer a question about weights, and `rankpct` would hide a
    degenerate member inside it — the exact failure E091 found in
    `prediction_spread`."""
    src = GENERATED.read_text()
    i = src.index("rows = []")
    j = src.index("out.to_csv", i)
    assert "probs[i]" in src[i:j], "the dump is not reading raw per-arm probabilities"
    assert "ranks" not in src[i:j], "the dump is reading the rank blend"


# --------------------------------------------------------------------------- #
# The 58
# --------------------------------------------------------------------------- #
def test_gold_expected_is_the_fifty_eight_this_project_has_always_scored_on():
    assert _kernel(GOLD_SLUG).constants["GOLD_EXPECTED"] == 58


def test_the_gold_count_is_a_hard_failure_not_a_warning():
    src = GENERATED.read_text()
    i = src.index("if len(gold) != GOLD_EXPECTED:")
    assert "raise RuntimeError" in src[i:i + 200]


# --------------------------------------------------------------------------- #
# The AUC
# --------------------------------------------------------------------------- #
def test_auc_on_a_perfect_and_an_inverted_ranking():
    auc = _load("auc")
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert auc(y, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)
    assert auc(y, np.array([0.4, 0.3, 0.2, 0.1])) == pytest.approx(0.0)


def test_a_constant_prediction_scores_one_half_and_not_whatever_argsort_gives():
    """THE TIE CASE, and the reason the ranks are averaged. A member that
    collapsed to one value per finding still writes a valid dump; without tie
    averaging it would score by index order, which on a sorted label column reads
    as 1.0 and looks like the best arm in the ensemble."""
    auc = _load("auc")
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    # sorted labels, constant predictions: the case that reads 1.0 unguarded
    y2 = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert auc(y2, np.ones(6) * 0.3) == pytest.approx(0.5)


def test_auc_handles_a_partial_tie_by_hand_computation():
    """Two positives and two negatives, one positive tied with one negative.
    Pairs: (p=0.9 beats both negatives) 2, (p=0.5 beats 0.1) 1 and ties 0.5 -> 0.5.
    So 3.5 of 4 concordant pairs."""
    auc = _load("auc")
    y = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([0.9, 0.5, 0.5, 0.1])
    assert auc(y, p) == pytest.approx(3.5 / 4.0)


def test_auc_is_undefined_rather_than_zero_for_a_single_class_column():
    auc = _load("auc")
    assert np.isnan(auc(np.zeros(5), np.arange(5.0)))
    assert np.isnan(auc(np.ones(5), np.arange(5.0)))


def test_macro_auc_is_the_mean_of_the_per_finding_aucs():
    macro_auc = _load("macro_auc")
    truth = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    # column 0 ranked perfectly; column 1 ranked exactly backwards, which means
    # every NEGATIVE scoring above every positive -- rows 0 and 2 are the
    # negatives in column 1, not rows 0 and 1.
    pred = np.array([[0.1, 0.9], [0.2, 0.2], [0.3, 0.8], [0.4, 0.1]])
    m, per = macro_auc(truth, pred)
    assert per == pytest.approx([1.0, 0.0])
    assert m == pytest.approx(0.5)


def test_the_gold_split_refuses_a_single_class_finding():
    """A finding with one class makes its AUC nan and the macro nan. Failing loudly
    beats reporting nan as a score."""
    src = GENERATED.read_text()
    i = src.index("degenerate = [f for k, f in enumerate(LAB)")
    assert "raise RuntimeError" in src[i:i + 400]
