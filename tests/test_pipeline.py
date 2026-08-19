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
