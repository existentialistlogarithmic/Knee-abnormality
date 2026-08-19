"""The pipeline declaration, and the tree it generates.

`kaggle/` used to be 29 `run.py` files that were 12 distinct programs. The
duplication was not merely ugly: `build_study` was byte-identical in four of
them, `find_marker` in seven, and `build_model` had drifted into two variants
across five — with the training kernel and the inference kernel that scores its
weights on opposite sides of the drift.

These tests pin the properties that replaced it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import pipeline  # noqa: E402

KAGGLE = REPO_ROOT / "kaggle"


def constants(path: Path) -> dict:
    found = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = _value(node.value, found)
    return found


def _value(node: ast.expr, known: dict):
    """A literal, or a simple expression over constants already seen.

    Generated kernels derive some constants from others (`SHIFT_PIXELS =
    TARGET_SIZE // 16`), which is the whole point — but it means a reader that
    only understands literals would silently skip them.
    """
    try:
        return ast.literal_eval(node)
    except ValueError:
        pass
    if isinstance(node, ast.Name):
        return known.get(node.id, _MISSING)
    if isinstance(node, ast.BinOp):
        left, right = _value(node.left, known), _value(node.right, known)
        if left is _MISSING or right is _MISSING:
            return _MISSING
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mult):
            return left * right
    return _MISSING


class _Missing:
    def __repr__(self):
        return "<unevaluated>"


_MISSING = _Missing()


# --------------------------------------------------------------------------- #
# the declaration
# --------------------------------------------------------------------------- #
def test_manifest_has_no_problems():
    assert pipeline.check() == []


def test_every_kernel_has_a_distinct_slug_and_directory():
    kernels = pipeline.all_kernels()
    assert len({k.slug for k in kernels}) == len(kernels)
    assert len({k.directory for k in kernels}) == len(kernels)


def test_submission_kernels_have_internet_off():
    """A kernel with internet enabled cannot be submitted, and finding that out
    from Kaggle costs a session."""
    for kernel in pipeline.all_kernels():
        if kernel.template == "infer":
            assert not kernel.internet, f"{kernel.slug} would be rejected"


def test_a_lineage_cannot_hold_a_geometry_of_its_own():
    """The core invariant. A lineage reaches its geometry through its cache, so
    a trainer physically cannot disagree with the cache it reads — there is no
    second copy of the numbers to disagree with."""
    for lineage in pipeline.LINEAGES:
        assert lineage.geometry is lineage.cache.geometry
    assert not any(f.name == "geometry" for f in
                   pipeline.Lineage.__dataclass_fields__.values())


def test_dinov2_shares_the_v1_cache_object():
    """It is a backbone experiment. Sharing the cache — not a copy of its
    numbers — is what makes its result attributable to the backbone alone."""
    by_name = {lineage.name: lineage for lineage in pipeline.LINEAGES}
    assert by_name["dinov2"].cache is by_name["v1"].cache


# --------------------------------------------------------------------------- #
# the generated tree
# --------------------------------------------------------------------------- #
def test_regenerating_is_a_no_op():
    """The test that the declaration still describes the tree. If someone edits
    a generated kernel by hand, this fails rather than letting the manifest
    quietly become fiction."""
    result = subprocess.run(
        [sys.executable, "eda/generate_kernels.py", "--check"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_generated_kernel_parses():
    for kernel in pipeline.all_kernels():
        path = KAGGLE / kernel.directory / "run.py"
        ast.parse(path.read_text())


def test_generated_kernels_carry_the_geometry_of_their_cache():
    for lineage in pipeline.LINEAGES:
        want = lineage.geometry.constants()
        directories = ([lineage.cache.directory.format(shard=s)
                        for s in range(lineage.cache.shards)]
                       + [t.directory for t in lineage.trainers]
                       + ([lineage.infer_directory] if lineage.infer_slug else []))
        for directory in directories:
            got = constants(KAGGLE / directory / "run.py")
            for name, value in want.items():
                assert got[name] == value, f"{directory}: {name}={got[name]}, want {value}"


def test_shared_code_exists_exactly_once():
    """The point of the exercise. `build_study` lived in four files; fixing it
    meant fixing it four times, and in practice it wasn't."""
    shared = KAGGLE / "_templates" / "_shared"
    names = set()
    for module in shared.glob("*.py"):
        for node in ast.parse(module.read_text()).body:
            if isinstance(node, ast.FunctionDef):
                assert node.name not in names, f"{node.name} defined in two shared modules"
                names.add(node.name)
    assert {"build_study", "build_model", "find_marker", "read_series_volume"} <= names


def test_generated_kernels_only_carry_helpers_they_use():
    """Splicing everything into everything would trade one kind of dead weight
    for another — a CPU cache kernel does not need the GPU check."""
    for kernel in pipeline.all_kernels():
        source = (KAGGLE / kernel.directory / "run.py").read_text()
        tree = ast.parse(source)
        used = {n.id for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name != "main":
                assert node.name in used, \
                    f"{kernel.directory} carries unused {node.name}()"


# --------------------------------------------------------------------------- #
# the properties that make an ensemble valid
# --------------------------------------------------------------------------- #
def test_inference_expectations_match_the_trainers_they_mount():
    """An inference kernel averages its mounted trainers. Two properties are
    invisible in the weights and fatal if they differ."""
    by_slug = {k.slug: k for k in pipeline.all_kernels()}
    for kernel in pipeline.all_kernels():
        if kernel.template != "infer":
            continue
        infer = constants(KAGGLE / kernel.directory / "run.py")
        for slug in kernel.depends:
            train = constants(KAGGLE / by_slug[slug].directory / "run.py")
            assert train["SLICE_SUBSAMPLE"] == infer["SLICE_SUBSAMPLE_EXPECTED"]
            assert train["INPUT_NORM"] == infer["INPUT_NORM_EXPECTED"]


def test_input_norm_is_recorded_not_rewritten():
    """The 0.725 leaderboard model was trained on un-normalised input. Flipping
    the flag here to look tidy would make the checkpoint guard useless, because
    the manifest would no longer describe the weights that exist."""
    by_name = {lineage.name: lineage for lineage in pipeline.LINEAGES}
    assert by_name["v1"].train.input_norm is False
    assert by_name["v2"].train.input_norm is False
    assert by_name["dinov2"].train.input_norm is True


def test_normalisation_buffers_stay_out_of_the_state_dict():
    """`mean` and `std` are constants, not learned state. Persisting them would
    add two keys to every checkpoint and make the strict-load check reject
    every set of weights written before they existed — including the folds
    training right now."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import runpy

    namespace = runpy.run_path(str(KAGGLE / "04_train" / "run.py"),
                               run_name="__not_main__")
    for normalise in (False, True):
        model = namespace["build_model"]("resnet18", 3, 12, normalise)
        keys = set(model.state_dict())
        assert "mean" not in keys and "std" not in keys, sorted(keys & {"mean", "std"})
    del torch


def test_normalisation_flag_actually_changes_the_forward_pass():
    """A flag that is recorded but not applied is worse than no flag."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import runpy

    namespace = runpy.run_path(str(KAGGLE / "04_train" / "run.py"),
                               run_name="__not_main__")
    volume = torch.rand(1, 3, 4, 64, 64)
    outputs = []
    for normalise in (False, True):
        torch.manual_seed(0)
        model = namespace["build_model"]("resnet18", 3, 12, normalise)
        model.eval()
        with torch.no_grad():
            outputs.append(model(volume))
    assert not torch.allclose(outputs[0], outputs[1]), \
        "INPUT_NORM is recorded but has no effect"


# --------------------------------------------------------------------------- #
# the v1 kernels changed shape; they must not have changed behaviour
# --------------------------------------------------------------------------- #
def test_the_unified_template_is_a_no_op_for_the_192px_lineage():
    """v1 folds are training on Kaggle right now against the pre-generator code.
    The generated kernels have to be the same program with the same constants,
    or the ensemble would be averaging two different models."""
    v1 = constants(KAGGLE / "04_train" / "run.py")
    assert v1["ACCUM_STEPS"] == 1, "accumulation must be a no-op for v1"
    assert v1["SLICE_SUBSAMPLE"] is None, "v1 trains on every cached slice"
    assert v1["INPUT_NORM"] is False, "v1 weights were trained un-normalised"
    # the augmentation magnitudes the 0.725 run actually used
    assert v1["TARGET_SIZE"] // 16 == 12
    assert (v1["TARGET_SIZE"] // 12, v1["TARGET_SIZE"] // 4) == (16, 48)


def test_augmentation_magnitudes_follow_the_geometry():
    """Retyping 12 and 18 per lineage is how the 288px run ended up augmenting
    proportionally less than the 192px one it was being compared against."""
    for lineage in pipeline.LINEAGES:
        for trainer in lineage.trainers:
            got = constants(KAGGLE / trainer.directory / "run.py")
            assert got["SHIFT_PIXELS"] == lineage.geometry.size // 16


def test_inference_accepts_a_checkpoint_carrying_the_constant_buffers():
    """`mean`/`std` were briefly persistent, so runs in flight will write them.
    Dropping them is lossless; refusing the checkpoint is not."""
    source = (KAGGLE / "11_infer_folds" / "run.py").read_text()
    assert 'k not in ("mean", "std")' in source, \
        "a checkpoint with the constant buffers would be refused"
    assert "missing, unexpected" in source, "the strict-load check must still run"


def test_training_exports_the_best_epoch_not_the_last():
    """Fold 1 peaked at 0.7334 on epoch 18 and saved epoch 23's 0.7282. Keeping
    the best weights is free, and across an ensemble the loss compounds."""
    source = (KAGGLE / "04_train" / "run.py").read_text()
    save = source[source.index("torch.save("):]
    assert '"model": export' in save, "the checkpoint must export the best weights"
    assert '"macro_auc": best_macro' in save, "the recorded score must be the exported one"
    assert 'best_state if best_state is not None' in source, \
        "an all-NaN fold would write an unloadable checkpoint"


def test_resume_continues_the_trajectory_not_the_export():
    """`model` is now the best EMA export, which is not where training left off.
    Resuming from it would restart a run from an average of its own past."""
    source = (KAGGLE / "04_train" / "run.py").read_text()
    assert 'core.load_state_dict(usable(state.get("live") or state["model"]))' in source
    assert '"live": {k: v.detach().cpu().clone()' in source, \
        "the live weights must be saved for resume"


def test_inference_separates_an_absent_record_from_a_recorded_none():
    """knee-train-v2 trained on 18 of 24 slices but predates the record.
    Reading absent as "no subsampling" refused a correct checkpoint — and would
    have passed a genuinely mismatched one in the other direction."""
    source = (KAGGLE / "08_infer_v2" / "run.py").read_text()
    assert "if key not in state:" in source, \
        "absent and recorded-as-None must be distinguished"
    assert "UNVERIFIED" in source, "an uncheckable property must be announced"
    assert "if state[key] != want:" in source, \
        "a recorded value must still be checked exactly"


def test_a_continuation_run_inherits_the_best_it_was_handed():
    """A warm restart raises the LR and gets worse before it gets better. A
    tracker starting from nothing would export those worse weights, and the log
    would only show the new run's own best, so nothing would look wrong."""
    source = (KAGGLE / "14_train_v2_long" / "run.py").read_text()
    assert 'best_macro = state["macro_auc"]' in source
    assert "inherited best macro AUC" in source, "the inheritance must be visible in the log"


def test_a_resuming_trainer_mounts_what_it_resumes_from():
    by_slug = {k.slug: k for k in pipeline.all_kernels()}
    for lineage in pipeline.LINEAGES:
        for trainer in lineage.trainers:
            if trainer.resume_from:
                assert trainer.resume_from in by_slug[trainer.slug].depends, \
                    f"{trainer.slug} resumes from an output it does not mount"


def _rank_block(directory: str):
    """The ensemble's rank-averaging step, lifted out of a generated kernel."""
    source = (KAGGLE / directory / "run.py").read_text()
    block = source[source.index("    scored = ~np.isnan"):
                   source.index("    submission = pd.DataFrame")]
    return "\n".join(l[4:] if l.startswith("    ") else l for l in block.splitlines())


def test_rank_averaging_is_a_no_op_for_a_single_model():
    """The metric reads order, so a rank transform cannot change a one-model
    submission's AUC. If it does, the transform is wrong."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n, f = 200, 12
    truth = rng.integers(0, 2, (n, f))
    raw = 1 / (1 + np.exp(-(truth * 1.2 + rng.normal(0, 1, (n, f)))))[None].astype(np.float32)
    predictions = np.full((n, f), 0.3, np.float32)
    ns = {"raw": raw, "predictions": predictions, "models": [0], "FINDINGS": list(range(f))}
    exec(_rank_block("11_infer_folds"), {"np": np}, ns)
    before = np.mean([roc_auc_score(truth[:, j], raw[0, :, j]) for j in range(f)])
    after = np.mean([roc_auc_score(truth[:, j], ns["predictions"][:, j]) for j in range(f)])
    assert abs(before - after) < 1e-12, f"{before} != {after}"


def test_rank_averaging_stays_in_range_and_centres_failed_studies():
    """A study that could not be decoded has no rank. With a rank score there is
    no meaningful prevalence to fall back to, so it belongs in the middle."""
    import numpy as np

    rng = np.random.default_rng(1)
    n, f, m = 60, 12, 3
    raw = rng.random((m, n, f)).astype(np.float32)
    raw[:, :, 0] = np.round(raw[:, :, 0], 1)            # ties
    failed = [4, 17, 41]
    raw[:, failed] = np.nan
    predictions = np.full((n, f), 0.3, np.float32)
    ns = {"raw": raw, "predictions": predictions, "models": list(range(m)),
          "FINDINGS": list(range(f))}
    exec(_rank_block("11_infer_folds"), {"np": np}, ns)
    out = ns["predictions"]
    assert out.min() >= 0.0 and out.max() <= 1.0, (out.min(), out.max())
    assert np.allclose(out[failed], 0.5)


def test_the_ensemble_does_not_average_probabilities():
    """Averaging sigmoids lets the most confident member dominate for no reason
    the metric rewards."""
    for directory in ("11_infer_folds", "08_infer_v2", "13_infer_dinov2"):
        source = (KAGGLE / directory / "run.py").read_text()
        assert "np.mean([torch.sigmoid" not in source, f"{directory} still probability-means"
        assert "ranks[order] = np.arange" in source, f"{directory} has no rank transform"


def test_per_finding_pooling_gives_each_finding_its_own_attention():
    """One attention map for twelve findings forces a single compromise about
    which slices matter, and the focal findings pay for it. If the twelve maps
    were identical the change would be cosmetic."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import runpy

    ns = runpy.run_path(str(KAGGLE / "04_train" / "run.py"), run_name="__not_main__")
    torch.manual_seed(0)
    model = ns["build_model"]("resnet18", 3, 12, False, True)
    model.eval()
    out = model(torch.rand(1, 3, 4, 64, 64))
    assert out.shape == (1, 12) and torch.isfinite(out).all()

    embedded = torch.randn(2, 18, model.head_weight.shape[1])
    maps = model.attention(embedded).softmax(dim=1)
    assert maps.shape[-1] == 12, "there must be one map per finding"
    assert maps.std(dim=2).mean() > 0, "the twelve maps are identical"


def test_the_pooling_choice_is_read_from_the_checkpoint_not_the_manifest():
    """It changes the shape of the weights. Building from the manifest instead
    would assemble the wrong architecture whenever the two disagree."""
    for directory in ("11_infer_folds", "08_infer_v2", "13_infer_dinov2"):
        source = (KAGGLE / directory / "run.py").read_text()
        assert 'state.get("per_finding_pool", False)' in source, directory
        assert "does not fit the model this" in source, \
            f"{directory} would report an architecture mismatch as a raw traceback"


def test_a_pre_existing_checkpoint_still_loads():
    """Every checkpoint written so far has a single attention map. A new option
    that orphaned them would throw away the 0.725 model."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import runpy

    ns = runpy.run_path(str(KAGGLE / "04_train" / "run.py"), run_name="__not_main__")
    torch.manual_seed(0)
    legacy = ns["build_model"]("resnet18", 3, 12, False, False)
    torch.manual_seed(1)
    rebuilt = ns["build_model"]("resnet18", 3, 12, False, False)
    missing, unexpected = rebuilt.load_state_dict(legacy.state_dict(), strict=False)
    assert not missing and not unexpected


def test_the_pooling_ab_differs_in_exactly_one_constant():
    """The 288px run changed five things at once and its number meant nothing.
    An A/B that is not one-variable is not an A/B."""
    baseline = constants(KAGGLE / "04_train" / "run.py")
    variant = constants(KAGGLE / "17_train_v1pool" / "run.py")
    differing = {k for k in set(baseline) | set(variant)
                 if baseline.get(k) != variant.get(k)}
    assert differing == {"PER_FINDING_POOL"}, differing


def test_resume_tolerates_the_briefly_persistent_constant_buffers():
    """knee-train-dinov2 is exactly such a checkpoint, and a strict load would
    reject the run meant to continue it."""
    source = (KAGGLE / "19_train_dinov2_long" / "run.py").read_text()
    assert 'k not in ("mean", "std")' in source, "resume would be refused"
    assert "core.load_state_dict(usable(" in source
    assert "usable(state[\"ema\"])" in source, "the EMA copy carries the same keys"


def test_exactly_one_labels_dataset_is_mounted_per_trainer():
    """The training kernel finds soft_labels.parquet by searching the mounted
    inputs. Two datasets carrying that file would make the targets depend on
    directory order — a silent, unreproducible choice of what the model learns.
    """
    known = {pipeline.ARTIFACTS_DATASET, pipeline.FUSED_DATASET}
    for lineage in pipeline.LINEAGES:
        for kernel in lineage.kernels():
            if kernel.template != "train":
                continue
            carrying = [d for d in kernel.datasets if d in known]
            assert len(carrying) == 1, f"{kernel.slug} mounts {carrying}"


def test_the_label_ab_differs_only_in_which_dataset_supplies_the_labels():
    """The union of the two labelers is worth +0.070 on gold. Measuring what
    that buys requires the labels to be the only thing that changed."""
    baseline = constants(KAGGLE / "04_train" / "run.py")
    variant = constants(KAGGLE / "21_train_v1fused" / "run.py")
    differing = {k for k in set(baseline) | set(variant)
                 if baseline.get(k) != variant.get(k)}
    assert not differing, differing

    import json
    a = json.loads((KAGGLE / "04_train" / "kernel-metadata.json").read_text())
    b = json.loads((KAGGLE / "21_train_v1fused" / "kernel-metadata.json").read_text())
    assert a["dataset_sources"] != b["dataset_sources"], "the labels must differ"
    assert b["dataset_sources"] == [pipeline.FUSED_DATASET]
    for field in ("kernel_sources", "enable_gpu", "enable_internet", "machine_shape"):
        assert a[field] == b[field], f"{field} differs; this is no longer an A/B"
