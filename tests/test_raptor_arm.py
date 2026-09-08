"""Tests for the CC0 CoAtNet arm.

This is the first kernel in the project that mounts someone else's model, so the
ways it can go wrong are new ones:

1. **Mounting an asset with no licence grant.** E043's tiers were prose until
   now, and E088 found three standing documents had already drifted from what
   E043 says. A rule that lives only in a comment gets misquoted; this one is
   asserted against the dated licence snapshot every asset was read from.
2. **Declaring more arms than weights.** `ARM_FILES`, `ARM_WEIGHTS` and
   `ARM_FLIP` are positional and a length mismatch would silently reweight the
   blend or drop a member — the same class of failure `MEMBERS_EXPECTED` was
   added for in E084, where a kernel that never ran mounted as an empty
   directory and a one-member "blend" nearly scored as five.
3. **Blending before the control has scored.** The single-arm kernel exists to
   establish the member is real; if it is ever removed, the blend becomes
   unreadable rather than merely unproven.

No patient data: this reads the manifest and a licence snapshot only.
"""

from __future__ import annotations

import json
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


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_every_mounted_asset_carries_a_real_licence_grant(slug):
    """E043's avoid tier, asserted instead of described.

    `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` is the asset this most
    protects against: it reads "Other (specified in description)" and its
    description is empty, so nothing is specified and nothing is granted.
    """
    for asset in _kernel(slug).datasets:
        licence = LICENCES.get(asset)
        assert licence, f"{asset} has no licence recorded in the snapshot"
        assert licence.startswith(USABLE_PREFIXES), (
            f"{slug} mounts {asset}, licensed {licence!r}. Only CC0 and Apache "
            f"are usable without a further decision (E043, E088).")


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_the_arm_vectors_line_up_and_match_the_declared_count(slug):
    c = _kernel(slug).constants
    assert len(c["ARM_FILES"]) == len(c["ARM_WEIGHTS"]) == len(c["ARM_FLIP"])
    assert len(c["ARM_FILES"]) == c["MEMBERS_EXPECTED"]


def test_the_control_is_exactly_one_arm_and_unweighted():
    c = _kernel("knee-infer-raptorcc0").constants
    assert c["MEMBERS_EXPECTED"] == 1
    assert c["ARM_WEIGHTS"] == (1.0,)
    assert c["ARM_FLIP"] == (False,)


def test_the_blend_uses_the_published_weights_and_they_sum_to_one():
    """Published, not fitted. A weight tuned on 58 studies is a free parameter
    fitted to 58 studies, declined four times in this log."""
    c = _kernel("knee-infer-raptorcc0x4").constants
    assert c["ARM_WEIGHTS"] == (0.55, 0.20, 0.15, 0.10)
    assert sum(c["ARM_WEIGHTS"]) == pytest.approx(1.0)


def test_the_flip_member_reuses_a_checkpoint_rather_than_adding_one():
    """maxspan-v5 appears twice, straight and flipped. If that ever becomes four
    distinct files, the free-diversity claim in the note is no longer true."""
    c = _kernel("knee-infer-raptorcc0x4").constants
    files, flips = c["ARM_FILES"], c["ARM_FLIP"]
    flipped = [f for f, fl in zip(files, flips, strict=True) if fl]
    assert flipped, "the published blend has a flipped member"
    for f in flipped:
        assert files.count(f) >= 2, f"{f} is flipped but never used straight"
    assert len(set(files)) == 3


@pytest.mark.parametrize("slug", RAPTOR_SLUGS)
def test_the_arm_runs_on_gpu_with_no_internet(slug):
    """A submission kernel: internet off is a competition requirement, and the
    weights must come from a mount rather than a download."""
    k = _kernel(slug)
    assert k.gpu is True
    assert k.internet is False


def test_the_geometry_is_shared_so_the_two_kernels_cannot_drift():
    solo, blend = (_kernel(s).constants for s in RAPTOR_SLUGS)
    for key in ("IMG", "CROP_MM", "SLOTS", "SPAN_LO", "SPAN_HI", "K_EVAL", "LAB"):
        assert solo[key] == blend[key], f"{key} differs between the two raptor kernels"
    assert sum(s[2] for s in solo["SLOTS"]) == 64
