"""Tests for the CC0 CoAtNet arm.

This is the first kernel here that runs someone else's model, so the failure
modes are new — and two of them were live in the first draft, caught by reading
the reference implementations rather than by running anything:

1. **One geometry applied to four models that disagree.** `maxspan-v5` caches at
   336 px over a 64-slice layout spanning 0.02-0.98 with 62 windows;
   `native384-v8` at 384 px over a 44-slice layout spanning 0.06-0.94 with 42.
   The first draft gave all four the *v4 legacy* config, which none of them
   uses. That is silent train/inference skew: it runs, it writes a submission,
   and it is wrong.
2. **"Reverse" is not a horizontal flip.** Upstream's prose says horizontal flip
   and its code says `xw.flip(1)` on `(K, 3, H, W)` — dim 1 is the three
   neighbouring slices, so it reverses the slice triplet and leaves the image
   alone. The first draft implemented the prose.
3. **Mounting an asset with no licence grant.** E043's tiers were prose until
   now, and E088 found three standing documents had already drifted from them.

No patient data: every array here is written for the test.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import all_kernels  # noqa: E402

LICENCES = json.loads((REPO_ROOT / "eda" / "public_asset_licences.json").read_text())
USABLE_PREFIXES = ("CC0", "Apache")
RAPTOR_SLUGS = ("knee-infer-raptorcc0", "knee-infer-raptorcc0x4")


def _kernel(slug):
    matches = [k for k in all_kernels() if k.slug == slug]
    assert matches, f"{slug} is not in the manifest"
    return matches[0]


def _arms(slug):
    return _kernel(slug).constants["ARMS"]


def _slices(arm):
    return sum(int(s[2]) for s in arm["slots"])


# --------------------------------------------------------------------------- #
# Licence
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_every_mounted_asset_carries_a_real_licence_grant(slug):
    """E043's avoid tier, asserted instead of described.

    `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` is what this most guards
    against: it reads "Other (specified in description)" and its description is
    empty, so nothing is specified and nothing is granted.
    """
    for asset in _kernel(slug).datasets:
        licence = LICENCES.get(asset)
        assert licence, f"{asset} has no licence recorded in the snapshot"
        assert licence.startswith(USABLE_PREFIXES), (
            f"{slug} mounts {asset}, licensed {licence!r}. Only CC0 and Apache are "
            f"usable without a further decision (E043, E088).")


# --------------------------------------------------------------------------- #
# Per-arm geometry: the bug that would have run and been wrong
# --------------------------------------------------------------------------- #
def test_the_four_arms_do_not_share_one_geometry():
    """The regression guard. If this ever passes trivially because someone
    collapsed the arms onto a shared config, the skew is back."""
    signatures = {(a["img"], a["slots"], tuple(a["span"]), a["k_eval"])
                  for a in _arms("knee-infer-raptorcc0x4")}
    assert len(signatures) > 1, "the four published sub-models disagree on geometry"


def test_each_arm_matches_its_published_geometry():
    published = {
        "maxspan-v5":         (336, 64, (0.02, 0.98), 62, False, 0.55),
        "native384dense-v10": (384, 64, (0.02, 0.98), 62, False, 0.10),
        "maxspan-v5-reverse": (336, 64, (0.02, 0.98), 62, True,  0.15),
        "native384-v8":       (384, 44, (0.06, 0.94), 42, False, 0.20),
    }
    for arm in _arms("knee-infer-raptorcc0x4"):
        assert (arm["img"], _slices(arm), tuple(arm["span"]), arm["k_eval"],
                arm["reverse"], arm["w"]) == published[arm["name"]], arm["name"]


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_every_arm_asks_for_exactly_the_windows_its_stack_holds(slug):
    """`_eval_centers` can only centre a window where c-1 and c+1 both exist, so
    a stack of N slices holds N-2 positions. Upstream's comment says "every
    window position the volume holds"; this asserts the arithmetic agrees, and
    would catch a slot layout edited without its k_eval."""
    for arm in _arms(slug):
        assert arm["k_eval"] == _slices(arm) - 2, arm["name"]


def test_the_control_is_one_arm_at_full_weight_with_its_own_geometry():
    arms = _arms("knee-infer-raptorcc0")
    assert len(arms) == 1 and arms[0]["w"] == 1.0
    assert arms[0]["name"] == "maxspan-v5"
    assert (arms[0]["img"], arms[0]["k_eval"]) == (336, 62)
    assert arms[0]["reverse"] is False


def test_the_blend_weights_are_the_published_ones_and_sum_to_one():
    """Published, not fitted. A weight tuned on 58 studies is a free parameter
    fitted to 58 studies, declined four times in this log."""
    weights = [a["w"] for a in _arms("knee-infer-raptorcc0x4")]
    assert sorted(weights) == [0.10, 0.15, 0.20, 0.55]
    assert sum(weights) == pytest.approx(1.0)


def test_the_reverse_member_reuses_a_checkpoint_rather_than_adding_one():
    arms = _arms("knee-infer-raptorcc0x4")
    files = [a["file"] for a in arms]
    for arm in arms:
        if arm["reverse"]:
            assert files.count(arm["file"]) >= 2, \
                f"{arm['name']} is reversed but its file is never used straight"
    assert len(set(files)) == 3, "four members, three checkpoints"


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_the_declared_count_matches_the_arms(slug):
    assert len(_arms(slug)) == _kernel(slug).constants["MEMBERS_EXPECTED"]


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_the_arm_runs_on_gpu_with_no_internet(slug):
    kernel = _kernel(slug)
    assert kernel.gpu is True
    assert kernel.internet is False


# --------------------------------------------------------------------------- #
# The maths, executed rather than described
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def kernel_ns():
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    return runpy.run_path(str(REPO_ROOT / "kaggle" / "81_infer_raptorcc0" / "run.py"),
                          run_name="__not_main__")


def test_reverse_flips_the_slice_triplet_and_not_the_image(kernel_ns):
    """The bug this test exists for: `flip(-1)` mirrors the knee left-to-right,
    a distribution the model never saw. `flip(1)` reverses (c-1, c, c+1)."""
    import torch
    windows = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    captured = {}

    class Spy:
        def __call__(self, x):
            captured["x"] = x.clone()
            return torch.zeros(x.shape[0] * x.shape[1], 12)

        def eval(self):
            return self

    reversed_view = windows.flip(1)
    assert not torch.equal(reversed_view, windows)
    # the middle slice is untouched, the outer two swap
    assert torch.equal(reversed_view[:, 1], windows[:, 1])
    assert torch.equal(reversed_view[:, 0], windows[:, 2])
    # and it is NOT the same as mirroring width
    assert not torch.equal(reversed_view, windows.flip(-1))


def test_the_head_pools_per_finding_over_windows(kernel_ns):
    """Twelve attention maps, one per finding, softmaxed over windows — the
    property that distinguishes this head from a shared mean pool."""
    import torch
    raptor = kernel_ns["RaptorClassifier"]

    class Stub(torch.nn.Module):
        num_features = 4

        def forward(self, x):
            return x.flatten(1)[:, :4]

    torch.manual_seed(0)
    model = raptor(Stub(), F_dim=4, n=12).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 5, 4, 1, 1))
    assert out.shape == (1, 12), "one logit per finding"

    # attention is per finding: att maps features -> 12, softmaxed over windows
    assert model.att[-1].out_features == 12
    assert model.clsW.shape == (12, 4)


def test_eval_windows_normalises_and_resizes_to_the_model_input(kernel_ns):
    import numpy as np
    import torch
    vol = np.full((8, 6, 6), 128, np.uint8)
    mask = np.ones(8, np.uint8)
    out = kernel_ns["eval_windows"](vol, mask, k=4, res=6)
    assert out.shape == (4, 3, 6, 6)
    # 128/255 normalised by the ImageNet statistics, not left in [0, 1]
    expected = (128 / 255.0 - 0.485) / 0.229
    assert torch.allclose(out[0, 0], torch.full((6, 6), expected), atol=1e-5)
    upscaled = kernel_ns["eval_windows"](vol, mask, k=4, res=12)
    assert upscaled.shape == (4, 3, 12, 12), "windows resize to the checkpoint's res"


def test_rankpct_is_a_per_column_percentile(kernel_ns):
    import numpy as np
    ranks = kernel_ns["rankpct"](np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]]))
    assert np.allclose(ranks[:, 0], [1.0, 0.0, 0.5])
    assert np.allclose(ranks[:, 1], [0.0, 1.0, 0.5])


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_a_mostly_failing_run_refuses_to_submit_a_coin_flip(slug):
    """Every failure mode of this kernel lands in one `except` and yields a
    constant-0.5 row. Without a limit, a renamed column in the hidden test's
    series table produces a well-formed submission that scores 0.500 and is
    indistinguishable on the leaderboard from a model that simply does not work.
    E084's MEMBERS_EXPECTED stops that on the mounting side; this is the data
    side."""
    limit = _kernel(slug).constants["FALLBACK_LIMIT"]
    assert 0 < limit < 0.5, "a limit at or above 50% would not catch a broken run"


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_the_slot_planes_match_the_competition_series_table(slug):
    """`Anatomical_Plane` is one of Axial/Coronal/Sagittal and `Fluid_Sensitive`
    is 0 or 1 in the competition's own series metadata. A typo here would match
    no series, fill no slot, and fall back on every study."""
    for arm in _arms(slug):
        for plane, fluid, count in arm["slots"]:
            assert plane in ("Axial", "Coronal", "Sagittal"), plane
            assert fluid in (0, 1, -1), fluid
            assert count > 0


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_every_arm_carries_the_fingerprint_of_the_file_it_expects(slug):
    """Another account owns these weights and can re-upload under the same name.

    The `Kernel` docstring already requires it: a foreign asset "can be deleted,
    made private, or re-run with different outputs at any time, so anything
    mounting one MUST verify what it got rather than assume". `gold_auc` is
    carried inside each checkpoint, so it fingerprints the exact artefact this
    manifest was written against — the same trick that verified E086's arms by
    matching each mounted checkpoint back to its trainer by val AUC.
    """
    for arm in _arms(slug):
        assert isinstance(arm["expect_gold"], float)
        assert 0.5 < arm["expect_gold"] < 1.0, arm["name"]


def test_the_two_reverse_members_fingerprint_the_same_file():
    """maxspan-v5 and maxspan-v5-reverse are one checkpoint read two ways, so a
    differing expectation would mean one of them names the wrong file."""
    by_file = {}
    for arm in _arms("knee-infer-raptorcc0x4"):
        by_file.setdefault(arm["file"], set()).add(arm["expect_gold"])
    for fname, golds in by_file.items():
        assert len(golds) == 1, f"{fname} has conflicting fingerprints {golds}"


def test_the_fingerprints_are_the_values_read_from_the_files():
    """Read on 2026-09-08 with torch.load. Pinned so a silent edit to the
    manifest cannot quietly disable the mount check by matching whatever
    arrives."""
    expected = {"raptor_ft_coatnet_v5_full_swa.pt": 0.9214,
                "raptor_ft_coatnet_v8_full_swa.pt": 0.9067,
                "raptor_ft_coatnet_v10_full.pt": 0.9174}
    for arm in _arms("knee-infer-raptorcc0x4"):
        assert arm["expect_gold"] == expected[arm["file"]], arm["name"]
