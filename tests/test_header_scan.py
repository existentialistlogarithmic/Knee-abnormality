"""Tests for the DICOM header-scan kernel.

No DICOM files and no competition data: the kernel's pure helpers are tested
directly, and the per-series reader is tested against a stubbed `dcmread` so
that slice counting, corrupt-file tolerance, and missing directories are
covered without shipping binaries into the repo.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = REPO_ROOT / "kaggle" / "00_dicom_header_scan" / "run.py"


@pytest.fixture(scope="module")
def kernel():
    pytest.importorskip("pydicom")
    pytest.importorskip("pandas")
    return runpy.run_path(str(KERNEL), run_name="__not_main__")


# --------------------------------------------------------------------------- #
# plane derivation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        ("1|0|0|0|1|0", "Axial"),
        ("1|0|0|0|0|-1", "Coronal"),
        ("0|1|0|0|0|-1", "Sagittal"),
        # a slightly oblique acquisition still resolves to its dominant plane
        ("0.99|0.05|0.01|-0.05|0.99|0.02", "Axial"),
        (None, None),
        ("", None),
        ("1|0|0", None),
        ("not|a|number|at|all|here", None),
    ],
)
def test_plane_from_orientation(kernel, orientation, expected):
    assert kernel["plane_from_orientation"](orientation) == expected


# --------------------------------------------------------------------------- #
# value flattening
# --------------------------------------------------------------------------- #
def test_scalarise_handles_dicom_value_shapes(kernel):
    scalarise = kernel["scalarise"]
    assert scalarise(None) is None
    assert scalarise(b"SIEMENS") == "SIEMENS"
    assert scalarise(["0.33", "0.33"]) == "0.33|0.33"
    assert scalarise(("SK", "SP")) == "SK|SP"
    assert scalarise(3.0) == "3.0"


# --------------------------------------------------------------------------- #
# per-series reading
# --------------------------------------------------------------------------- #
class _FakeDataset:
    """Stands in for a pydicom Dataset: attribute access, missing tags absent."""

    def __init__(self, **values):
        for key, value in values.items():
            setattr(self, key, value)


def _make_series(tmp_path: Path, n_slices: int) -> Path:
    series = tmp_path / "study-1" / "series-1"
    series.mkdir(parents=True)
    for i in range(n_slices):
        (series / f"{i}.dcm").write_bytes(b"not really dicom")
    (series / "notes.txt").write_text("should be ignored")
    return series


def test_read_series_counts_slices_and_extracts_fields(kernel, tmp_path, monkeypatch):
    series = _make_series(tmp_path, 24)
    monkeypatch.setattr(
        kernel["pydicom"],
        "dcmread",
        lambda *_a, **_k: _FakeDataset(
            Manufacturer="SIEMENS",
            ManufacturerModelName="MAGNETOM Vida",
            MagneticFieldStrength=3.0,
            ImagingFrequency=123.255723,
            Laterality="L",
            ImageOrientationPatient=["0", "1", "0", "0", "0", "-1"],
        ),
    )
    record = kernel["read_series"](series, "study-1", "series-1", "train")

    assert record["n_slices"] == 24, "the .txt file must not be counted as a slice"
    assert record["error"] is None
    assert record["manufacturer"] == "SIEMENS"
    assert record["imaging_frequency"] == "123.255723"
    assert record["laterality"] == "L"
    assert record["plane_from_headers"] == "Sagittal"
    # absent tags are present as None rather than missing keys
    assert record["software_versions"] is None


def test_read_series_survives_a_corrupt_first_file(kernel, tmp_path, monkeypatch):
    series = _make_series(tmp_path, 3)
    calls = {"n": 0}

    def flaky(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("truncated file")
        return _FakeDataset(Manufacturer="GE MEDICAL SYSTEMS")

    monkeypatch.setattr(kernel["pydicom"], "dcmread", flaky)
    record = kernel["read_series"](series, "study-1", "series-1", "train")

    assert record["error"] is None, "a corrupt first slice must not fail the series"
    assert record["manufacturer"] == "GE MEDICAL SYSTEMS"


def test_read_series_reports_a_missing_directory(kernel, tmp_path):
    record = kernel["read_series"](tmp_path / "nope", "study-9", "series-9", "test")
    assert record["n_slices"] == 0
    assert "missing" in record["error"]


def test_read_series_reports_an_empty_directory(kernel, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    record = kernel["read_series"](empty, "study-9", "series-9", "test")
    assert record["n_slices"] == 0
    assert record["error"] == "no dicom files"
