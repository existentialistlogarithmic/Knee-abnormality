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
# The submission kernel is the fold ensemble; `05_infer` was the single-model
# version it superseded and is no longer part of the generated pipeline.
INFER = REPO_ROOT / "kaggle" / "11_infer_folds" / "run.py"
TRAIN = REPO_ROOT / "kaggle" / "04_train" / "run.py"
CACHE = REPO_ROOT / "kaggle" / "03_cache_build_shard0" / "run.py"


def _source(path: Path) -> str:
    return path.read_text()


# --------------------------------------------------------------------------- #
# the geometry the weights were trained on
# --------------------------------------------------------------------------- #
def constants(path: Path) -> dict[str, str]:
    """Module-level constant assignments in a kernel file.

    Generated kernels align their constants block, so `NAME = 1` and
    `NAME   = 1` both have to parse. Reading them properly rather than with a
    regex is what stops a cosmetic change from silently disabling a test.
    """
    import ast

    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                found[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return found


def test_inference_uses_the_same_cache_geometry_as_the_build():
    """If these drift, the model sees a different field of view than it learned."""
    cache, infer = constants(CACHE), constants(INFER)
    for constant in ("TARGET_MM_PER_PIXEL", "TARGET_SIZE", "SLICES_PER_PLANE"):
        assert constant in cache and constant in infer, f"{constant} missing from one side"
        assert cache[constant] == infer[constant], (
            f"{constant} differs: cache={cache[constant]} infer={infer[constant]}")


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

    model = namespace["build_model"]("resnet18", 3, 12, False)
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
    """A single unreadable study must cost a little AUC, never the submission.

    Asserted against behaviour rather than code shape: the per-study builder
    must catch, the failure must be counted, and predictions must start at the
    fallback prior so an uncaught study still emits a valid row.
    """
    source = _source(INFER)
    assert "FALLBACK_PRIOR" in source
    assert "failures += 1" in source
    builder = source[source.index("def build_one("):source.index("workers =")]
    assert "except Exception" in builder, "the per-study builder has no guard"
    assert "return index, None," in builder, "builder must report failure, not raise"
    # every row starts at the prior, so a study that never gets predicted is valid
    assert "np.full((len(studies), len(FINDINGS)), FALLBACK_PRIOR" in source


def test_inference_predicts_every_study_it_builds():
    """The batching must flush a partial final batch, or the tail is dropped.

    A batched loop that only predicts on full batches silently leaves the last
    1..BATCH_STUDIES-1 studies at the fallback prior — valid output, quietly
    worse score, no error anywhere.
    """
    source = _source(INFER)
    after_loop = source[source.index("for done, (index, stack, error)"):]
    tail = after_loop[after_loop.index("if done % 100"):]
    assert "flush(pending)" in tail, "partial final batch is never flushed"


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
    # The export is the best-scoring EMA snapshot; both it and the live weights
    # for resume are taken from `core`, the unwrapped module.
    assert '"model": export' in save, "checkpoint must save the exported EMA weights"
    assert "core.state_dict()" in save, "weights must come from the unwrapped module"
    assert "model.state_dict()" not in save, "would save DataParallel keys"
    ema = source[source.index("ema_state = "):]
    assert ema.startswith("ema_state = {k: v.detach().cpu().clone() for k, v in core.state_dict()"), \
        "the EMA snapshot must be taken from the unwrapped module"


def test_training_does_not_flip_left_right():
    """Right knees are mirrored in the cache; flipping would undo that and make
    the four medial/lateral targets ambiguous."""
    source = _source(TRAIN)
    augment = source[source.index("if self.augment:"):source.index("return (torch.from_numpy")]
    assert "[..., ::-1]" not in augment, "left-right flip would undo laterality correction"


def test_checkpoint_round_trips_from_training_into_inference():
    """Weights saved by the training kernel must load into the inference model.

    This is train/inference skew in its purest form: a DataParallel prefix, a
    backbone mismatch, or an EMA/live mixup all produce a checkpoint that either
    fails to load or — worse — loads partially and predicts noise.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    train_ns = runpy.run_path(str(TRAIN), run_name="__not_main__")
    infer_ns = runpy.run_path(str(INFER), run_name="__not_main__")

    trained = train_ns["build_model"]("resnet18", 3, 12, False)
    # exactly the shape the training kernel writes
    checkpoint = {"model": {k: v.clone() for k, v in trained.state_dict().items()},
                  "backbone": "resnet18", "epoch": 3, "macro_auc": 0.7}

    assert not any(k.startswith("module.") for k in checkpoint["model"]), \
        "checkpoint carries DataParallel prefixes"

    rebuilt = infer_ns["build_model"](checkpoint["backbone"], 3, 12, False)
    missing, unexpected = rebuilt.load_state_dict(checkpoint["model"], strict=False)
    assert not missing and not unexpected, f"missing={missing[:3]} unexpected={unexpected[:3]}"

    rebuilt.eval()
    volume = torch.zeros(1, 3, 20, 192, 192)
    with torch.no_grad():
        out = torch.sigmoid(rebuilt(volume))
    assert out.shape == (1, 12)
    assert torch.isfinite(out).all()


def test_inference_excludes_setup_from_the_per_study_projection():
    """Setup is paid once; folding it into a per-study average over a 3-study
    public test set and multiplying by 1,300 overstates cost ~20x."""
    source = _source(INFER)
    assert "setup_seconds" in source and "loop_started" in source
    assert "per_study = loop_seconds" in source, "per-study cost must exclude setup"
    assert "setup_seconds + per_study * 1300" in source, "projection must add setup once"


INFER_V2 = REPO_ROOT / "kaggle" / "08_infer_v2" / "run.py"
CACHE_V2 = REPO_ROOT / "kaggle" / "06_cache_v2_shard0" / "run.py"
TRAIN_V2 = REPO_ROOT / "kaggle" / "07_train_v2" / "run.py"


def test_v2_inference_matches_the_v2_cache_geometry():
    cache, infer = constants(CACHE_V2), constants(INFER_V2)
    for constant in ("TARGET_MM_PER_PIXEL", "TARGET_SIZE", "SLICES_PER_PLANE"):
        assert cache[constant] == infer[constant], \
            f"{constant} differs between v2 cache and v2 inference"


def test_v2_inference_subsamples_slices_like_training_did():
    """v2 trained on a random 18 of 24 slices and validated on an evenly spaced
    18; feeding 24 at inference is a silent mismatch — the attention pool sees a
    different sequence length than it was scored with and nothing raises."""
    train, infer = constants(TRAIN_V2), constants(INFER_V2)
    assert train["SLICE_SUBSAMPLE"] == infer["SLICE_SUBSAMPLE_EXPECTED"], \
        "SLICE_SUBSAMPLE differs between v2 training and v2 inference"
    assert "np.linspace(0, stack.shape[1] - 1,\n" in _source(INFER_V2), \
        "inference must take an evenly spaced subset, as validation did"


ENSEMBLE = REPO_ROOT / "kaggle" / "09_infer_ensemble" / "run.py"


def test_ensemble_pairs_models_to_checkpoints_by_name():
    """Pairing by sort order would silently hand the 288px spec the 192px
    weights — wrong input geometry, no exception, just a worse score."""
    source = _source(ENSEMBLE)
    assert '"kernel": "knee-train"' in source and '"kernel": "knee-train-v2"' in source
    assert "found[spec[\"kernel\"]]" in source, "must look up the checkpoint by kernel slug"
    assert "zip(MODELS, weight_dirs)" not in source, "order-based pairing is unsafe"
    assert "missing_kernels" in source, "must fail loudly if a checkpoint is absent"


def test_ensemble_geometries_match_their_caches():
    """Each ensemble member's geometry must match the cache it trained on."""
    source = _source(ENSEMBLE)
    for cache_path, size_key in ((CACHE, 192), (CACHE_V2, 288)):
        cache = constants(cache_path)
        assert cache["TARGET_SIZE"] == size_key
        assert f'"size": {size_key}' in source, f"no ensemble member at {size_key}px"
        assert f'"mm_per_pixel": {cache["TARGET_MM_PER_PIXEL"]:.2f}' in source, \
            f'mm/px {cache["TARGET_MM_PER_PIXEL"]} missing'


def test_training_records_input_geometry_in_the_checkpoint():
    """The checkpoint, not a constant in the inference file, is the authority on
    how the model was fed. Otherwise changing training config silently
    mismatches inference and nothing raises."""
    for source_path in (TRAIN, TRAIN_V2):
        save = _source(source_path)
        save = save[save.index("torch.save("):]
        assert '"backbone"' in save, f"{source_path.parent.name}: backbone not recorded"
        assert '"slice_subsample"' in save, f"{source_path.parent.name}: slice count not recorded"
        assert '"input_norm"' in save, f"{source_path.parent.name}: normalisation not recorded"


def test_ensemble_prefers_the_checkpoint_geometry_over_its_own_spec():
    source = _source(ENSEMBLE)
    assert 'state.get("slice_subsample"' in source
    assert "using the checkpoint" in source, "must prefer the recorded value"
    assert "MODELS[i] = spec" in source, "corrected geometry must reach the builder"


FOLDS = REPO_ROOT / "kaggle" / "11_infer_folds" / "run.py"


def test_fold_ensemble_averages_all_models_not_one():
    """A leftover single-model call would silently ignore every other fold —
    the ensemble would run, cost N times the compute, and score like one model."""
    source = _source(FOLDS)
    assert "torch.sigmoid(model(batch))" not in source, "stale single-model prediction path"
    assert "np.mean([torch.sigmoid(m(batch))" in source, "must average across models"
    assert "for m in models" in source


def test_fold_ensemble_refuses_mismatched_geometries():
    """Averaging models fed different slice counts compares different inputs."""
    source = _source(FOLDS)
    assert "SLICE_SUBSAMPLE_EXPECTED" in source
    assert "INPUT_NORM_EXPECTED" in source
    assert "averaging these models would be" in source


def test_fold_ensemble_decodes_once_for_all_models():
    """Decode dominates cost; rebuilding per model would make an N-fold ensemble
    N times slower for no benefit."""
    source = _source(FOLDS)
    build_calls = source.count("build_study(root,")
    assert build_calls == 1, f"volume built {build_calls} times; should be once per study"
