"""Gold-58 predictions from a public 2.5D ConvNeXt reader, run in OUR kernel (E150).

SOURCING. Read this before changing anything below.

The model, its preprocessing and its inference code are NOT this project's
work. They are goodpjw2008's, published as

  weights+code  kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader  (Apache-2.0)
  notebook      kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944

and are imported from the mounted dataset UNCHANGED (`infer.py`, `knee.py`,
`preprocess.py`). The licence was read from the dataset metadata endpoint on
2026-10-07 and is recorded in `eda/public_asset_licences.json`.

WHAT IS OURS: the mount guard, the gold selection, loading every checkpoint
with `weights_only=True` (upstream passes False; the pickles were disassembled
offline and carry only OrderedDict/_rebuild_tensor_v2/FloatStorage, so the
safe loader accepts them), the decode-failure count, and the output contract.

WHY GOLD-58 IS AN HONEST READ OF THIS MODEL. Upstream's `make_labels.py` keeps
every gold study at fold = -1, "never used for training in CV", and
`train.py` saves the final-epoch EMA weights rather than a checkpoint chosen on
gold. So each of the 58 is predicted by models that never trained on it. The
author may still have tuned hyperparameters while watching gold; that bias is
unmeasurable from here and is noted, not corrected.

WHAT IT DECIDES. Nothing on its own. The output is 58 rows of predictions that
`eda/screen_foreign_gold.py` scores against the shipped composite's gold
references — quality and independence, then the paired bootstrap of the
composite with this reader as one more vote. Pre-registered in E150.
"""
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# E150: gold-58 predictions from goodpjw2008's public 2.5D
# ConvNeXt-Tiny reader (Apache-2.0), the one genuinely new public
# asset E149's screen of the top 14 notebooks found. Upstream code
# is imported unchanged; ours is the guard, safe loading, the
# decode count and the output. Screened offline afterwards by
# `eda/screen_foreign_gold.py` against the shipped composite.
#
EXPECTED_FOLDS  = 3
EXPECTED_GOLD   = 58
MAX_DECODE_FAIL = 0.05

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
SKIP_DIRECTORIES = {"train_series", "test_series"}


# --------------------------------------------------------------------------- #
# from kaggle/_templates/_shared/discovery.py
# --------------------------------------------------------------------------- #
def find_marker(marker: str, max_depth: int = 4):
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for entry in entries:
            if entry.is_file() and entry.name == marker:
                return directory
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRECTORIES:
                frontier.append((entry, depth + 1))
    return None


def report_environment() -> bool:
    """Record the accelerator actually granted.

    The Kaggle CLI does not expose the valid `machine_shape` strings, so which
    GPU a kernel receives has been UNVERIFIED for this project. This prints it,
    which matters because the current PyTorch build ships no Pascal kernels and
    a P100 would fail rather than run slowly.

    Returns True if the accelerator can actually run this build.
    """
    import torch

    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  no GPU visible; this will be very slow")
        return True
    usable = True
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name}  compute capability {major}.{minor}")
        if major < 7:
            usable = False
            print("  >>> PRE-VOLTA GPU. The Kaggle PyTorch build ships no Pascal")
            print("  >>> kernels, so every CUDA launch fails with")
            print("  >>> 'no kernel image is available for execution on the device'.")
            print("  >>> Push with --accelerator set to a T4 shape instead.")
    return usable


def install_decoders(reader: Path) -> None:
    """Upstream's pylibjpeg wheels, for this interpreter only, from the mount.

    They exist for cp311 and cp312 only. On any other ABI the install cannot
    succeed, and upstream's loader would then skip every JPEG-compressed
    series without a word -- so say so loudly and let the decode count below
    decide whether the run is usable.
    """
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels = sorted(str(p) for p in reader.glob("pylibjpeg-*-py3-none-any.whl"))
    wheels += sorted(str(p) for p in reader.glob(f"pylibjpeg_*-{tag}-*.whl"))
    if len(wheels) < 3:
        print(f"WARNING: no pylibjpeg wheels for {tag}; found {wheels}")
    if wheels:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                        "--no-deps", "-q", *wheels], check=False)


def main() -> int:
    t0 = time.time()
    report_environment()

    reader = find_marker("cnxt_v0_fold0.pt")
    competition = find_marker("train_series.csv")
    if reader is None or competition is None:
        raise SystemExit(f"mount missing: reader={reader} competition={competition}")
    ckpts = sorted(reader.glob("cnxt_v0_fold*.pt"))
    if len(ckpts) != EXPECTED_FOLDS:
        raise SystemExit(f"expected {EXPECTED_FOLDS} reader folds, found {len(ckpts)}")
    print(f"reader at {reader}: {[p.name for p in ckpts]}")

    install_decoders(reader)
    sys.path.insert(0, str(reader))
    import infer                                   # upstream, unchanged

    original_load = torch.load

    def safe_load(*args, **kwargs):
        kwargs["weights_only"] = True
        return original_load(*args, **kwargs)

    torch.load = safe_load
    infer.torch.load = safe_load

    # Upstream's dataset swallows every decode exception and leaves the slot
    # empty. Count them, in-process (workers=0), so a broken decoder shows up
    # as a number rather than as a quietly weak model.
    counts = {"ok": 0, "failed": 0}
    upstream_load_series = infer.load_series

    def counted_load_series(*args, **kwargs):
        try:
            out = upstream_load_series(*args, **kwargs)
        except Exception:
            counts["failed"] += 1
            raise
        counts["ok"] += 1
        return out

    infer.load_series = counted_load_series

    train = pd.read_csv(competition / "train.csv")
    gold = train[train[FINDINGS].notna().all(axis=1)].reset_index(drop=True)
    if len(gold) != EXPECTED_GOLD:
        raise SystemExit(f"expected {EXPECTED_GOLD} gold studies, found {len(gold)}")
    studies = gold.StudyInstanceUID.tolist()

    per_fold = []
    for ck in ckpts:
        pred = infer.run(str(competition), "train", [str(ck)], studies=studies,
                         workers=0, bs=2)
        per_fold.append(pred.loc[studies, FINDINGS].to_numpy(np.float64))
        print(f"{ck.name}: {time.time() - t0:.0f}s, decodes ok={counts['ok']} "
              f"failed={counts['failed']}")

    total = counts["ok"] + counts["failed"]
    fail_rate = counts["failed"] / max(total, 1)
    print(f"series decoded: {counts['ok']}/{total} (failed {fail_rate:.1%})")
    if fail_rate > MAX_DECODE_FAIL:
        raise SystemExit(f"decode failure rate {fail_rate:.1%} exceeds "
                         f"{MAX_DECODE_FAIL:.0%}: the decoder is broken, and the "
                         f"predictions would read as a weak model")

    from sklearn.metrics import roc_auc_score
    y = gold[FINDINGS].to_numpy(np.float64) > 0.5
    mean = np.mean(per_fold, axis=0)

    def macro(p):
        return float(np.mean([roc_auc_score(y[:, j], p[:, j])
                              for j in range(len(FINDINGS)) if 0 < y[:, j].sum() < len(y)]))

    summary = {"folds": [macro(p) for p in per_fold], "mean": macro(mean),
               "decode_ok": counts["ok"], "decode_failed": counts["failed"],
               "n_gold": len(studies), "seconds": round(time.time() - t0)}
    print(json.dumps(summary, indent=1))

    out = pd.DataFrame(mean, columns=FINDINGS)
    out.insert(0, "StudyInstanceUID", studies)
    for k, p in enumerate(per_fold):
        for j, f in enumerate(FINDINGS):
            out[f"fold{k}__{f}"] = p[:, j]
    out.to_csv("gold_cnxt_reader.csv", index=False)
    Path("gold_cnxt_reader_summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
