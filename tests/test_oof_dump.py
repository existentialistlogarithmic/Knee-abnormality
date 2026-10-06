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


def _const(p, name):
    return re.search(rf"^{name}\s*=\s*(-?\d+)", _src(p), re.M).group(1)


def test_the_five_cross_validation_folds_exist_and_cover_0_to_4():
    # The instrument is the seed-3 set. Reseeds of one fold (E145) are extra
    # draws beside it and must not be pooled into it: pool_oof's partition
    # check would refuse them anyway, but the count here should not move.
    instrument = [p for p in FOLD_CV if _const(p, "RUN_SEED") == "3"]
    assert len(instrument) == 5, [p.parent.name for p in instrument]
    folds = sorted(int(_const(p, "RUN_FOLD")) for p in instrument)
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


def test_the_drift_check_catches_a_kernel_dropped_from_the_manifest(tmp_path):
    """E140: the check walked manifest -> tree only, so deleting a kernel from
    the manifest left its directory on disk and preflight printed 'tree matches
    the manifest'. A live submission kernel was silently dropped that way."""
    import subprocess
    import sys
    src = Path("eda/generate_kernels.py").read_text()
    assert "that no kernel declares" in src, "orphan detection is gone"
    assert "RETIRED" in src, "the allowlist is gone, so it will fail on history"
    # and the allowlist is a literal set, not a pattern that would swallow new ones
    assert "startswith" not in src.split("RETIRED")[1].split("}")[0]
    r = subprocess.run([sys.executable, "eda/generate_kernels.py"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "tree matches the manifest" in r.stdout


def test_every_kernel_defines_the_helpers_it_calls():
    """E140 lost a GPU session to this.

    `find_all_markers` lived inside the template's `@@IF V1_MEMBERS@@` region
    while FOREIGN_ARMS also called it, so a kernel with foreign arms and
    V1_MEMBERS=None generated code that imported fine, ran the entire CoAtNet
    pass, and only then died with NameError. A helper shared by two optional
    regions cannot be gated on one of them, and nothing checked that.
    """
    helpers = ("find_all_markers", "find_marker", "rankpct", "build_model")
    for p in KERNELS:
        s = _src(p)
        for h in helpers:
            called = re.search(rf"[^.\w]{h}\s*\(", s)
            defined = re.search(rf"^\s*def {h}\s*\(", s, re.M)
            if called and not defined:
                raise AssertionError(
                    f"{p.parent.name} calls {h}() but never defines it — the "
                    f"include is probably gated on a region this kernel "
                    f"switched off")


def test_the_or_directive_keeps_a_region_for_either_constant():
    """E140's repair. `@@IF A|B@@` must keep the region when either is set,
    and drop it only when both are off — otherwise the fix either reintroduces
    the NameError or spills the v1 machinery into CoAtNet-only kernels."""
    import sys
    sys.path.insert(0, ".")
    from eda.generate_kernels import strip_regions
    from src.pipeline import Kernel

    tpl = ("keep-always\n@@IF A|B@@\nshared-helper\n@@ENDIF A|B@@\ntail\n")

    def render(**consts):
        k = Kernel(slug="t", directory="t", template="raptor_infer",
                   constants=consts)
        return strip_regions(tpl, k)

    assert "shared-helper" in render(A=1, B=None)
    assert "shared-helper" in render(A=None, B=1)
    assert "shared-helper" in render(A=1, B=1)
    assert "shared-helper" not in render(A=None, B=None)
    for out in (render(A=1, B=None), render(A=None, B=None)):
        assert "keep-always" in out and "tail" in out


def test_build_model_resolves_input_size_before_it_resolves_pretrained():
    """E142 cost two GPU sessions to this ordering.

    `dynamic_img_size` and `img_size` are accepted by different timm families:
    ViTs take the first, the MaxxVit family (CoAtNet, MaxViT) rejects it with
    TypeError and takes the second. The old code tried `dynamic_img_size`,
    caught the TypeError alongside download failures, and fell through to a
    plain build — silently pinning a CoAtNet to its native 224 so that it died
    mid-training on a 196-vs-144 token shape error.
    """
    src = Path("kaggle/_templates/_shared/model.py").read_text()
    assert '"img_size": TARGET_SIZE' in src, "the img_size path is gone"
    assert "except TypeError:" in src, "TypeError must not be caught with downloads"
    # a forward pass at TARGET_SIZE must be attempted before the model is kept
    i = src.index('size_kwargs')
    assert "TARGET_SIZE, TARGET_SIZE" in src[i:], "no size check before returning"
    assert "could not be built at" in src[i:], "silent failure is still possible"
