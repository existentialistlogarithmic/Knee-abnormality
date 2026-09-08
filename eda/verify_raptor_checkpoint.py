#!/usr/bin/env python3
"""Prove a CC0 CoAtNet checkpoint loads and runs through OUR code, before pushing.

    kaggle datasets download dreaddevelopment/raptor-knee-maxspan \
        -f raptor_ft_coatnet_v5_full_swa.pt -p ckpt
    python eda/verify_raptor_checkpoint.py --checkpoint ckpt/raptor_ft_coatnet_v5_full_swa.pt

**Why this is not covered by the test suite.** The checkpoints are 292 MB each
and belong to someone else, so they are neither committed nor downloadable in
CI. Everything the tests can assert about this arm is static: that the manifest
declares four arms with the published geometry, that the licences are CC0, that
the fingerprints are pinned. **None of that proves the model loads.** A timm
version whose `coatnet_rmlp_2_rw_384` has a different feature width, or a head
key that moved, fails only when `load_state_dict(strict=True)` runs — and on
Kaggle that is an hour into a GPU session.

**What it checks, in the order that matters:**

1. the file's `gold_auc` matches the manifest's `expect_gold` — the mount
   fingerprint, because another account owns this file and can replace it;
2. `strict=True` resolves every one of the 486 tensors, which is what actually
   proves the architecture was reconstructed correctly rather than merely
   plausibly;
3. `eval_windows` returns exactly `(k_eval, 3, res, res)` for that arm's own
   geometry — the bug E090 found was four arms sharing one geometry;
4. a forward pass returns twelve finite probabilities in [0, 1];
5. the reverse member actually differs from the straight one. If it does not,
   the slice-triplet reversal is a no-op and the 0.15-weighted member is a
   duplicate of the 0.55 one — a silent way to reweight the blend.

**What it cannot check: the DICOM path.** `build_study` needs real series, and
this arm has no offline gold evaluation available at its geometry, so the board
control is the only thing that can price it. That is stated here rather than
discovered later.

Reads no patient data: the volume is synthetic.
"""

from __future__ import annotations

import argparse
import runpy
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = REPO_ROOT / "kaggle" / "81_infer_raptorcc0" / "run.py"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kernel", default=str(KERNEL),
                        help="generated run.py to verify against")
    parser.add_argument("--windows", type=int, default=4,
                        help="windows to push through the forward pass on CPU")
    args = parser.parse_args(argv)

    namespace = runpy.run_path(args.kernel, run_name="__not_main__")
    arms = namespace["ARMS"]
    name = Path(args.checkpoint).name
    matching = [a for a in arms if a["file"] == name]
    if not matching:
        print(f"FAIL {name} is not an arm of {Path(args.kernel).parent.name}: "
              f"{[a['file'] for a in arms]}")
        return 1
    arm = matching[0]
    print(f"arm {arm['name']}  img {arm['img']}  span {tuple(arm['span'])}  "
          f"k_eval {arm['k_eval']}  reverse {arm['reverse']}")

    started = time.time()
    model, res, arch, gold = namespace["load_model"](args.checkpoint, "cpu")
    print(f"loaded {arch} in {time.time() - started:.1f}s")

    if gold != arm["expect_gold"]:
        print(f"FAIL fingerprint: file says {gold}, manifest expects "
              f"{arm['expect_gold']}. Upstream changed the file.")
        return 1
    print(f"ok   fingerprint gold_auc {gold}")
    print("ok   strict load resolved every tensor")

    maxs = sum(int(s[2]) for s in arm["slots"])
    volume = np.random.default_rng(0).integers(
        0, 255, size=(maxs, arm["img"], arm["img"]), dtype=np.uint8)
    windows = namespace["eval_windows"](volume, np.ones(maxs, np.uint8),
                                        k=arm["k_eval"], res=res)
    if tuple(windows.shape) != (arm["k_eval"], 3, res, res):
        print(f"FAIL windows {tuple(windows.shape)}, expected "
              f"({arm['k_eval']}, 3, {res}, {res})")
        return 1
    print(f"ok   windows {tuple(windows.shape)}")

    subset = windows[:max(1, args.windows)]
    started = time.time()
    straight = namespace["infer_probs"](model, subset, "cpu", False)
    seconds = time.time() - started
    if straight.shape != (12,) or not np.isfinite(straight).all():
        print(f"FAIL forward returned {straight.shape}")
        return 1
    if not ((straight >= 0) & (straight <= 1)).all():
        print(f"FAIL probabilities outside [0, 1]: "
              f"[{straight.min():.3f}, {straight.max():.3f}]")
        return 1
    print(f"ok   forward {straight.shape} in [{straight.min():.3f}, "
          f"{straight.max():.3f}], {seconds / len(subset):.2f}s/window on CPU")

    reversed_probs = namespace["infer_probs"](model, subset, "cpu", True)
    delta = float(np.abs(straight - reversed_probs).max())
    if np.allclose(straight, reversed_probs):
        print("FAIL the reverse member is a no-op, so it duplicates the straight "
              "one and silently reweights the blend")
        return 1
    print(f"ok   reverse differs, max |delta| {delta:.4f}")

    print("\nPASSED. Real weights load through our loader and produce twelve "
          "calibrated outputs.\nNot checked: the DICOM path. `build_study` needs "
          "real series, and this arm has\nno offline gold evaluation at its "
          "geometry — the board control is the only price.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
