"""Tests for the cache-build kernel's pure logic.

These cover the decisions that would corrupt a cache silently rather than
crashing — wrong slice selection, wrong normalisation, missed laterality. A
crash costs a kernel run; silent corruption costs every experiment built on it.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = REPO_ROOT / "kaggle" / "03_cache_build" / "run.py"


@pytest.fixture(scope="module")
def kernel():
    pytest.importorskip("pydicom")
    pytest.importorskip("pandas")
    return runpy.run_path(str(KERNEL), run_name="__not_main__")


# --------------------------------------------------------------------------- #
# slice selection
# --------------------------------------------------------------------------- #
def test_pick_slices_spans_the_whole_stack(kernel):
    """Both ends must survive — meniscal tears sit at the periphery of the
    sagittal stack, which is exactly what a centre crop would discard."""
    picked = kernel["pick_slices"](40, 20)
    assert len(picked) == 20
    assert picked[0] == 0
    assert picked[-1] == 39
    assert picked == sorted(picked)


def test_pick_slices_pads_a_short_stack_without_dropping_any(kernel):
    picked = kernel["pick_slices"](5, 20)
    assert len(picked) == 20
    assert set(range(5)).issubset(set(picked)), "no real slice may be lost"


def test_pick_slices_handles_exact_and_empty(kernel):
    assert kernel["pick_slices"](20, 20) == list(range(20))
    assert kernel["pick_slices"](0, 20) == []


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_normalise_is_per_volume_not_per_slice(kernel):
    """A bright slice and a dark slice in one acquisition must stay different.

    Per-slice normalisation would flatten them to the same range and destroy the
    relative brightness that separates fluid from fat.
    """
    volume = np.stack([np.full((8, 8), 10.0), np.full((8, 8), 200.0)])
    out = kernel["normalise"](volume)
    assert out.dtype == np.uint8
    assert out[0].mean() < out[1].mean(), "relative brightness must survive"


def test_normalise_survives_a_constant_volume(kernel):
    out = kernel["normalise"](np.full((4, 8, 8), 7.0))
    assert out.dtype == np.uint8
    assert np.isfinite(out).all()


def test_normalise_survives_nans(kernel):
    volume = np.full((2, 4, 4), np.nan)
    volume[0, 0, 0] = 5.0
    out = kernel["normalise"](volume)
    assert out.dtype == np.uint8


def test_normalise_clips_outliers(kernel):
    volume = np.concatenate([np.full(998, 100.0), np.array([0.0, 1e6])]).reshape(1, 10, 100)
    out = kernel["normalise"](volume)
    assert out.max() <= 255 and out.min() >= 0


# --------------------------------------------------------------------------- #
# laterality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("LT_t2_tse_fs_cor_obl_ACL", "L"),
        ("RT_pd_tse_sag", "R"),
        ("MR LEFT KNEE", "L"),
        ("MR RIGHT KNEE", "R"),
        ("L_sag_pd", "L"),
        ("R_sag_pd", "R"),
        ("pd_tse_tra_d", None),
        ("", None),
        (None, None),
    ],
)
def test_laterality_from_description(kernel, description, expected):
    assert kernel["_laterality_from_description"](description) == expected


# --------------------------------------------------------------------------- #
# resize
# --------------------------------------------------------------------------- #
def test_resize_produces_the_target_shape(kernel):
    out = kernel["resize"](np.random.default_rng(0).integers(0, 255, (57, 91), dtype=np.uint8), 192)
    assert out.shape == (192, 192)


def test_normalise_maps_nans_to_a_defined_value(kernel):
    """NaN -> uint8 is undefined in numpy and would write arbitrary bytes."""
    volume = np.full((2, 4, 4), np.nan)
    volume[0, 0, 0], volume[0, 0, 1] = 0.0, 100.0
    out = kernel["normalise"](volume)
    assert out.dtype == np.uint8
    assert (out == 0).sum() >= 30, "NaNs should land at the low end, not at random"
