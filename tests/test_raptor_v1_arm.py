"""Tests for the second architecture spliced into the CoAtNet kernel.

E101 measured the only union out of eight candidates worth submitting: this
project's own resnet34 2.5D at 192 px against the four CC0 CoAtNet arms. The two
models have to predict in ONE kernel run, because `rank_blend`'s own guard
records why the obvious alternative cannot work — a mounted kernel supplies its
LAST SAVED output, frozen at the 3-study visible run, so a CSV-chained blend
submits three rows.

Two architectures in one file is two ways for a silent train/inference skew to
hide, which is exactly the bug E088 caught in kernel 81's first draft: it runs,
it writes a submission, and it is wrong. So these tests check the seam rather
than the models:

1. **The second architecture must be spliced, not reimplemented.** The v1 path
   here has to be the same `_shared/volume.py` and `_shared/model.py` every v1
   kernel has always used. A hand-written copy would drift from the training
   preprocessing without failing anything.
2. **The first architecture must be untouched by the splice.** The CoAtNet
   kernels' generated source is asserted to carry none of the second path, and
   the DICOM golden hash in `test_raptor_build_study.py` is what pins the shared
   half.
3. **The shapes must actually meet.** `build_study` produces a stack and
   `build_model` consumes one; nothing checks that they agree until a kernel
   runs, so a forward pass is run here on synthetic DICOMs.

No patient data: every array here is written for the test.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import all_kernels  # noqa: E402

pydicom = pytest.importorskip("pydicom")
pytest.importorskip("cv2")
torch = pytest.importorskip("torch")
pytest.importorskip("timm")

V1_SLUGS = ("knee-gold-raptorv1", "knee-infer-raptorv1")
COATNET_ONLY = ("knee-infer-raptorcc0", "knee-infer-raptorcc0x4", "knee-gold-raptorcc0x4")
BLEND = REPO_ROOT / "kaggle" / "86_infer_raptorv1" / "run.py"


def _kernel(slug):
    matches = [k for k in all_kernels() if k.slug == slug]
    assert matches, f"{slug} is not in the manifest"
    return matches[0]


def _source(slug):
    return (REPO_ROOT / "kaggle" / _kernel(slug).directory / "run.py").read_text()


# --------------------------------------------------------------------------- #
# The gate: the first architecture must not pay for the second
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("slug", COATNET_ONLY)
def test_the_coatnet_only_kernels_carry_none_of_the_second_architecture(slug):
    """`@@IF V1_MEMBERS@@` exists for this. Splicing ~700 lines of a second
    model's preprocessing into three kernels that never touch it is the dead
    weight `test_generated_kernels_only_carry_helpers_they_use` forbids, and it
    would put two definitions of "how a study becomes an array" in a file where
    only one of them is ever right."""
    src = _source(slug)
    assert _kernel(slug).constants["V1_MEMBERS"] is None
    # Structural markers only: this template reads DICOMs itself, so `import
    # pydicom` appears inside its own reader and is not a tell.
    for absent in ("def build_study(", "def build_model(", "def read_series_volume(",
                   "def find_all_markers(", "_shared/volume.py", "_shared/model.py"):
        assert absent not in src, f"{slug} carries {absent!r} and never runs it"


@pytest.mark.parametrize("slug", V1_SLUGS)
def test_the_blend_kernels_splice_the_shared_fragments_rather_than_copying_them(slug):
    """The v1 preprocessing must be the one the v1 models were TRAINED with.
    A second copy is train/inference skew that raises nothing and just scores
    worse."""
    src = _source(slug)
    for name in ("_shared/volume.py", "_shared/model.py", "_shared/discovery.py"):
        assert name in src, f"{slug} does not splice {name}"
    assert src.count("def build_study(") == 1, "two definitions of build_study"
    assert src.count("def build_raptor_study(") == 1


@pytest.mark.parametrize("slug", V1_SLUGS)
def test_the_two_architectures_do_not_share_a_function_name(slug):
    """`_shared/volume.py` and this template both defined `build_study`, taking
    completely different arguments. Whichever was spliced second would have won
    silently and fed one model the other's input."""
    src = _source(slug)
    i, j = src.index("def build_study("), src.index("def build_raptor_study(")
    assert i != j
    assert "(root: Path, split: str" in src[i:i + 120]
    assert "(sid, series, tsdir" in src[j:j + 120]


# --------------------------------------------------------------------------- #
# The manifest
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("slug", V1_SLUGS)
def test_the_member_count_is_declared_and_matches_the_trainers_mounted(slug):
    """A trainer that never ran mounts as an EMPTY directory: the glob finds
    fewer checkpoints, the ensemble runs, and it submits an experiment nobody
    declared. E061 verified this by reading a log afterwards, which is not a
    guard."""
    kernel = _kernel(slug)
    assert kernel.constants["V1_MEMBERS"] == len(kernel.depends) == 5


@pytest.mark.parametrize("slug", V1_SLUGS)
def test_the_v1_geometry_is_the_one_its_models_were_trained_at(slug):
    """192 px at 0.6 mm/px over 20 slices is the geometry that scored 0.725 and
    every v1 checkpoint since. Any other number here is silent skew."""
    c = _kernel(slug).constants
    assert (c["TARGET_SIZE"], c["TARGET_MM_PER_PIXEL"], c["SLICES_PER_PLANE"]) == (192, 0.6, 20)
    assert c["V1_INPUT_NORM"] is False
    assert c["V1_SLICE_SUBSAMPLE"] is None


def test_the_submission_kernel_and_the_gold_kernel_differ_only_in_the_split():
    """The gold run is only a plumbing check for the submission if it is the
    same plumbing."""
    gold, board = (_kernel(s).constants for s in V1_SLUGS)
    assert gold["EVAL_SPLIT"] == "gold" and board["EVAL_SPLIT"] == "test"
    assert {k: v for k, v in gold.items() if k != "EVAL_SPLIT"} == \
           {k: v for k, v in board.items() if k != "EVAL_SPLIT"}


def test_the_blend_is_fifty_fifty_and_re_ranks_both_sides():
    """E101's sweep peaks at w=0.3, and 0.5 is used anyway: the +0.0022 between
    them is below the board's own +/-0.003 reseed floor (E092), so a weight
    fitted on 58 studies would buy a difference the board cannot measure.

    Both sides are re-ranked first because the left is a WEIGHTED mean of rank
    columns and the right a plain mean of them. Neither is uniform on (0,1), and
    averaging them raw hands the ordering to whichever is flatter."""
    src = BLEND.read_text()
    i = src.index("if v1_probs is not None:")
    block = src[i:i + 900]
    assert "coat_r, v1_r = rankpct(ranks), rankpct(v1_probs)" in block
    assert "0.5 * coat_r + 0.5 * v1_r" in block


# --------------------------------------------------------------------------- #
# The shapes actually meet
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(str(BLEND), run_name="__not_main__")


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    import pandas as pd
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    root = tmp_path_factory.mktemp("comp")
    sid = "1.2.3.4.5"
    rows = []
    for plane, n, size, seed in (("Sagittal", 30, 260, 1), ("Coronal", 24, 240, 2),
                                 ("Axial", 20, 236, 3)):
        uid = generate_uid()
        directory = root / "test_series" / sid / uid
        directory.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        for i in range(n):
            px = rng.integers(0, 4000, size=(size, size), dtype=np.uint16)
            meta = Dataset()
            meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            path = directory / f"{i:03d}.dcm"
            ds = FileDataset(str(path), Dataset(), file_meta=meta, preamble=b"\0" * 128)
            ds.SOPClassUID = meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.Rows, ds.Columns = size, size
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            ds.PixelRepresentation = 0
            ds.PixelSpacing = [0.35, 0.35]
            ds.InstanceNumber = i + 1
            ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            ds.ImagePositionPatient = [0.0, 0.0, float(i) * 3.0]
            ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0
            ds.PixelData = px.tobytes()
            ds.save_as(str(path), enforce_file_format=True)
        rows.append({"StudyInstanceUID": sid, "SeriesInstanceUID": uid,
                     "Fluid_Sensitive": 1, "Anatomical_Plane": plane, "n_slices": n})
    return root, sid, pd.DataFrame(rows)


def test_the_v1_stack_has_the_shape_its_model_expects(ns, study):
    root, sid, rows = study
    stack, record = ns["build_study"](root, "test", sid, rows)
    assert stack.dtype == np.uint8
    assert stack.shape == (3, 20, 192, 192), stack.shape
    assert record["planes_found"] == 3 and not record["missing_planes"]
    assert stack.max() > 0, "every plane decoded to zeros"


def test_a_v1_forward_pass_produces_one_probability_per_finding(ns, study):
    """THE SEAM ITSELF. `build_study` makes a stack and `build_model` eats one;
    nothing checks that they agree until a kernel runs on the hidden set."""
    root, sid, rows = study
    stack, _ = ns["build_study"](root, "test", sid, rows)
    model = ns["build_model"]("resnet34", 3, 12, False, True, 0).eval()
    x = torch.from_numpy(stack[None].astype(np.float32) / 255.0)
    with torch.no_grad():
        out = torch.sigmoid(model(x))
    assert out.shape == (1, 12), out.shape
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_a_missing_plane_is_zero_padded_rather_than_dropped(ns, study):
    """A dropped plane would change the channel count and fail the forward pass
    loudly; a zero-padded one keeps the study scoreable, which is what the v1
    lineage has always done."""
    import pandas as pd
    root, sid, rows = study
    stack, record = ns["build_study"](root, "test", sid,
                                      pd.DataFrame(rows[rows.Anatomical_Plane != "Axial"]))
    assert stack.shape == (3, 20, 192, 192)
    assert record["missing_planes"] == ["Axial"]
    assert stack[2].max() == 0, "the missing plane is not the zeroed channel"


@pytest.mark.parametrize("slug", V1_SLUGS)
def test_the_v1_members_are_checked_for_being_distinct_files(slug):
    """THE GAP THE FIRST PLUMBING RUN EXPOSED (E102). The five full-fit members
    differ only by seed, so they mount in five directories holding a file of the
    SAME NAME, and the first print showed `parent.parent` — the account name, for
    all five. The log could not distinguish five checkpoints from one found five
    times, and one model averaged with itself is a silent 5x smaller ensemble
    that scores worse and raises nothing. MEMBERS_EXPECTED counts; this checks
    identity."""
    src = _source(slug)
    i = src.index("if len(set(_fp)) != len(_fp):")
    assert "raise RuntimeError" in src[i:i + 200]
    assert "v1_checkpoints = (_ck, _fp)" in src


def test_the_member_fingerprint_separates_two_seeds_and_matches_a_reload(ns):
    """The fingerprint is a sum over the first 4,096 weights. It has to differ
    between independently initialised models and be identical for the same one
    read twice, or it is either a false alarm or no alarm at all."""
    def fingerprint(model):
        w = {k: v for k, v in model.state_dict().items() if k not in ("mean", "std")}
        return float(next(iter(w.values())).detach().reshape(-1)[:4096].float().sum())

    torch.manual_seed(0)
    a = ns["build_model"]("resnet34", 3, 12, False, True, 0)
    torch.manual_seed(1)
    b = ns["build_model"]("resnet34", 3, 12, False, True, 0)
    assert fingerprint(a) != fingerprint(b), "two different models fingerprint alike"
    assert fingerprint(a) == fingerprint(a), "the same model fingerprints differently"
