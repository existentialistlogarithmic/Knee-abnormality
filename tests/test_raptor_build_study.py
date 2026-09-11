"""Lock the DICOM path of the CC0 CoAtNet arm to what upstream's code produces.

**How the golden value was established, which is the whole point.** On
2026-09-08 this fixture was pushed through BOTH implementations — upstream's
`build_study` from `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`
and the one in `kaggle/_templates/raptor_infer.py.in` — driven with upstream's
own v4 constants so the comparison was like for like. The volumes were
**identical in all 7,225,344 voxels**, and the masks matched. `EXPECTED_SHA256`
is the hash of that agreed volume.

Upstream's notebook is not vendored here, so this test cannot re-run the
comparison. What it can do is refuse to let our side drift away from a result
that was verified against theirs: series ordering by `ImagePositionPatient`
projected on the slice normal, slot selection and its fluid-sensitive
preference, the span pick, the 2nd/98th percentile window, the millimetre crop,
the resize and the uint8 quantisation are all folded into one number.

**Why a golden hash rather than assertions on each step.** Every one of those
steps is someone else's choice, reproduced. There is no independent ground truth
to assert against — only "does it still do what theirs did". A hash is the
honest shape for that, and E090 is what happens when this path drifts silently:
a wrong-geometry run writes a perfectly well-formed submission.

Reads no patient data: the DICOMs are synthesised here from a fixed seed.
"""

from __future__ import annotations

import hashlib
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

pydicom = pytest.importorskip("pydicom")
pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("timm")

KERNEL = REPO_ROOT / "kaggle" / "81_infer_raptorcc0" / "run.py"

# Upstream's v4 configuration, used because the agreement above was measured
# with it. It is deliberately NOT any of the four arms this project runs —
# those geometries are asserted in test_raptor_arm.py.
V4_IMG = 336
V4_SLOTS = (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
            ("Coronal", 0, 8), ("Axial", -1, 12))
V4_SPAN = (0.06, 0.94)
EXPECTED_SHA256 = "e445d6541239ec252bdde50cdf19e891"
EXPECTED_SHAPE = (64, 336, 336)


def _write_series(root: Path, sid: str, series_uid: str, n_slices: int,
                  rows: int, cols: int, seed: int) -> None:
    from pydicom.dataset import Dataset, FileDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    directory = root / sid / series_uid
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_slices):
        pixels = rng.integers(0, 4000, size=(rows, cols), dtype=np.uint16)
        meta = Dataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        path = directory / f"{i:03d}.dcm"
        ds = FileDataset(str(path), Dataset(), file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns = rows, cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [0.35, 0.35]
        ds.InstanceNumber = i + 1
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i) * 3.0]
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0
        ds.PixelData = pixels.tobytes()
        ds.save_as(str(path), enforce_file_format=True)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """One study with all five slots fillable, from a fixed seed."""
    from pydicom.uid import generate_uid
    root = tmp_path_factory.mktemp("series")
    sid = "1.2.3.4.5"
    plan = [("Sagittal", 1, 40, 260, 260, 1), ("Sagittal", 0, 36, 256, 256, 2),
            ("Coronal", 1, 30, 240, 240, 3), ("Coronal", 0, 28, 244, 244, 4),
            ("Axial", 1, 26, 236, 236, 5)]
    records = []
    for plane, fluid, n, rows, cols, seed in plan:
        uid = generate_uid()
        _write_series(root, sid, uid, n, rows, cols, seed)
        records.append({"StudyInstanceUID": sid, "SeriesInstanceUID": uid,
                        "Fluid_Sensitive": fluid, "Anatomical_Plane": plane})
    return root, sid, {sid: records}


@pytest.fixture(scope="module")
def kernel_ns():
    return runpy.run_path(str(KERNEL), run_name="__not_main__")


def test_the_dicom_path_still_matches_what_upstream_produced(study, kernel_ns):
    root, sid, series = study
    volume, mask = kernel_ns["build_study"](
        sid, series, str(root), kernel_ns["_make_reader"](),
        V4_IMG, V4_SLOTS, V4_SPAN)
    assert volume.shape == EXPECTED_SHAPE
    assert mask.sum() == EXPECTED_SHAPE[0], "every slot should be filled by this fixture"
    digest = hashlib.sha256(volume.tobytes()).hexdigest()[:32]
    assert digest == EXPECTED_SHA256, (
        "the DICOM path changed. This hash was agreed byte for byte with "
        "upstream's own build_study on 2026-09-08; a change here means our "
        "preprocessing no longer matches the pipeline these weights were "
        "trained for, and a run would still write a valid submission.")


def test_a_missing_plane_leaves_its_slot_empty_rather_than_shifting_the_rest(study, kernel_ns):
    """Slot positions are positional. Dropping the axial series must blank those
    12 slices, not slide the coronal ones into them — the model reads the stack
    by position."""
    root, sid, series = study
    without_axial = {sid: [r for r in series[sid] if r["Anatomical_Plane"] != "Axial"]}
    volume, mask = kernel_ns["build_study"](
        sid, without_axial, str(root), kernel_ns["_make_reader"](),
        V4_IMG, V4_SLOTS, V4_SPAN)
    assert volume.shape == EXPECTED_SHAPE
    assert mask[:52].all(), "the first four slots should still be filled"
    assert not mask[52:].any(), "the axial slot should be empty, not back-filled"


def test_a_study_with_no_series_at_all_yields_an_empty_mask(kernel_ns, tmp_path):
    """The kernel treats this as a failure rather than a study with no findings —
    it is how a schema change would present on every study at once."""
    volume, mask = kernel_ns["build_study"](
        "absent", {}, str(tmp_path), kernel_ns["_make_reader"](),
        V4_IMG, V4_SLOTS, V4_SPAN)
    assert volume.shape == EXPECTED_SHAPE
    assert not mask.any()
