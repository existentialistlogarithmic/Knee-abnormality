"""The out-of-fold dump is the instrument E136 called for, so it is pinned.

Two properties matter and both have cost this project a run before. It must
never fire on a full fit — every shipped member is one, and a file named
`oof_fold-1.csv.gz` holding predictions for studies that were in training is
exactly the "a number that is not a score, read as one" failure the training
template already warns about. And it must carry the study id, because a
positional join is the fault class that reads as a weak model rather than a bug
(E088, and the foreign-arm join in the inference template).
"""
import re
from pathlib import Path

import pytest

KERNELS = sorted(Path("kaggle").glob("*/run.py"))
FOLD_CV = [p for p in KERNELS if "v1pubcv" in p.parent.name]


def _src(p):
    return p.read_text()


def test_the_five_cross_validation_folds_exist_and_cover_0_to_4():
    assert len(FOLD_CV) == 5, [p.parent.name for p in FOLD_CV]
    folds = sorted(int(re.search(r"^RUN_FOLD\s*=\s*(-?\d+)", _src(p), re.M).group(1))
                   for p in FOLD_CV)
    assert folds == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("p", FOLD_CV, ids=lambda p: p.parent.name)
def test_each_fold_writes_a_full_validation_dump_keyed_by_study(p):
    s = _src(p)
    assert "oof_fold{args.fold}.csv.gz" in s
    assert '"StudyInstanceUID": val_studies' in s


@pytest.mark.parametrize("p", FOLD_CV, ids=lambda p: p.parent.name)
def test_the_dump_is_guarded_by_best_epoch_and_not_full_fit(p):
    s = _src(p)
    i = s.index("oof_fold{args.fold}.csv.gz")
    guard = s.rindex("if best_epoch == epoch and not full_fit:", 0, i)
    # nothing may re-open the guard between it and the write
    between = s[guard:i]
    assert "\n        if " not in between, "the dump escaped its guard"


def test_full_fit_members_cannot_emit_an_oof_file():
    """A full fit has no holdout, so the guard is the only thing standing
    between us and a memorisation table named like a score."""
    full = [p for p in KERNELS
            if re.search(r"^RUN_FOLD\s*=\s*-1", _src(p), re.M)
            and "oof_fold" in _src(p)]
    for p in full:
        s = _src(p)
        i = s.index("oof_fold{args.fold}.csv.gz")
        assert "not full_fit" in s[s.rindex("if ", 0, i):i]
