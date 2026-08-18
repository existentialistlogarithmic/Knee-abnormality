"""Contract tests between the cache, the model, and the submission format.

These exist because train/inference skew does not raise an exception — the model
just quietly receives different input than it was trained on and scores worse
for reasons nobody can see. Each test pins one end of that contract.
"""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INFER = REPO_ROOT / "kaggle" / "05_infer" / "run.py"
TRAIN = REPO_ROOT / "kaggle" / "04_train" / "run.py"
CACHE = REPO_ROOT / "kaggle" / "03_cache_build" / "run.py"


def _source(path: Path) -> str:
    return path.read_text()


# --------------------------------------------------------------------------- #
# the geometry the weights were trained on
# --------------------------------------------------------------------------- #
def test_inference_uses_the_same_cache_geometry_as_the_build():
    """If these drift, the model sees a different field of view than it learned."""
    cache, infer = _source(CACHE), _source(INFER)
    for constant in ("TARGET_MM_PER_PIXEL", "TARGET_SIZE", "SLICES_PER_PLANE"):
        pattern = rf"^{constant} = ([0-9.]+)"
        in_cache = re.search(pattern, cache, re.M)
        in_infer = re.search(pattern, infer, re.M)
        assert in_cache and in_infer, f"{constant} missing from one side"
        assert in_cache.group(1) == in_infer.group(1), (
            f"{constant} differs: cache={in_cache.group(1)} infer={in_infer.group(1)}")


def test_plane_order_is_identical_everywhere():
    """Channel order is positional — swapping sagittal and axial is silent."""
    pattern = r'PLANES = \(([^)]*)\)'
    planes = {p.stem: re.search(pattern, _source(p)).group(1).replace(" ", "")
              for p in (CACHE, INFER)}
    assert len(set(planes.values())) == 1, f"plane order differs: {planes}"


# --------------------------------------------------------------------------- #
# the submission contract
# --------------------------------------------------------------------------- #
def test_findings_order_matches_the_sample_submission():
    sample = REPO_ROOT / "data" / "sample_submission.csv"
    if not sample.exists():
        pytest.skip("sample_submission.csv not downloaded")
    import pandas as pd

    expected = list(pd.read_csv(sample).columns)[1:]
    found = re.search(r"FINDINGS = \[(.*?)\]", _source(INFER), re.S).group(1)
    findings = [f.replace("\\'", "'") for f in re.findall(r'"([^"]+)"', found)]
    assert findings == expected, "submission column order would be wrong"


def test_inference_asserts_the_submission_shape_before_writing():
    """The checks must run before to_csv, not after — afterwards is too late."""
    source = _source(INFER)
    write_at = source.index("submission.to_csv(")
    for check in ("columns differ from sample_submission", "row count must match",
                  "no NaNs allowed", "probabilities", "duplicate study ids"):
        position = source.index(check)
        assert position < write_at, f"check '{check}' runs after the file is written"


# --------------------------------------------------------------------------- #
# the model accepts what the cache produces
# --------------------------------------------------------------------------- #
def test_model_consumes_a_cache_shaped_volume():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    namespace = runpy.run_path(str(TRAIN), run_name="__not_main__")

    model = namespace["build_model"]("resnet18", 3, 12)
    model.eval()
    # exactly what one .npy holds: (planes, slices, size, size), batched
    volume = torch.from_numpy(
        np.random.default_rng(0).integers(0, 255, (2, 3, 20, 192, 192))
         .astype(np.float32) / 255.0)
    with torch.no_grad():
        out = model(volume)
    assert out.shape == (2, 12), f"model returned {tuple(out.shape)}, expected (2, 12)"
    assert torch.isfinite(out).all()


def test_inference_falls_back_rather_than_crashing_on_one_bad_study():
    """A single unreadable study must cost a little AUC, never the submission."""
    source = _source(INFER)
    assert "FALLBACK_PRIOR" in source
    assert "failures += 1" in source
    body = source[source.index("for i, study in enumerate(studies)"):]
    assert "except Exception" in body, "per-study loop has no guard"


def test_inference_does_not_hardcode_the_backbone():
    """Training switched resnet18 -> resnet34; a hardcoded name here would break."""
    source = _source(INFER)
    assert 'state.get("backbone"' in source, "backbone must come from the checkpoint"
    assert 'build_model("resnet18"' not in source, "backbone is hardcoded"


def test_inference_refuses_a_partial_weight_load():
    """strict=False silently accepts a mismatched checkpoint; that must be caught."""
    source = _source(INFER)
    assert "refusing to predict" in source
    assert "missing, unexpected" in source


def test_training_saves_unwrapped_weights():
    """DataParallel prefixes every key with 'module.'; inference builds a plain model."""
    source = _source(TRAIN)
    save = source[source.index("torch.save("):]
    assert '"model": ema_state' in save, "checkpoint must save unwrapped EMA weights"
    assert "model.state_dict()" not in save.split("\n")[0], "would save DataParallel keys"


def test_training_does_not_flip_left_right():
    """Right knees are mirrored in the cache; flipping would undo that and make
    the four medial/lateral targets ambiguous."""
    source = _source(TRAIN)
    augment = source[source.index("if self.augment:"):source.index("return (torch.from_numpy")]
    assert "[..., ::-1]" not in augment, "left-right flip would undo laterality correction"
