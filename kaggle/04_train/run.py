"""Phase 2 training — 2.5D backbone over the cached volumes, 12 heads.

Reads the cache built by `kaggle/03_cache_build`: one uint8 array per study of
shape (3 planes, 20 slices, 192, 192), laterality-corrected and resampled to a
fixed millimetre-per-pixel.

The constraints this design answers
-----------------------------------
**The bar is 0.669 grouped CV, not 0.5.** A metadata-only model already reaches
0.669 against report-derived labels with no pixels at all (`FINDINGS.md` §9).
Anything below that has not shown the images contribute. That is the number to
beat on fold 0 before spending a full cross-validated run.

**Report-label CV overstates the leaderboard by ~0.138** (`FINDINGS.md` §11).
So CV here is for *ranking* configurations, never for predicting the board. The
gap gets re-measured on the first submission from this model.

**Folds are scanner-grouped, always.** Random K-fold inflates macro AUC by 0.087.
`GroupKFold` over the fingerprint from `src/folds.py` is not optional.

**The abstain channel must reach the loss as absent supervision.** A report
silent on the ACL is not a report saying the ACL is intact. Those studies are
masked out of that finding's loss rather than taught as negatives — which is the
whole reason the labeler emits five channels instead of a probability.

**Gold studies are weighted heavily.** 58 studies carry expert labels; they are
the only targets known to match what is scored.

Sessions die, so training checkpoints every epoch and resumes from the last one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# KERNEL RUN CONFIG — Kaggle script kernels take no arguments.
# --------------------------------------------------------------------------- #
RUN_FOLD = 0
RUN_EPOCHS = 6
RUN_BATCH = 8
RUN_LR = 3e-4
RUN_BACKBONE = "resnet18"
RUN_TIME_BUDGET = 7.5 * 3600
GOLD_WEIGHT = 8.0          # how much more an expert label counts than a report one
ABSTAIN_MASKS_LOSS = True  # silence is not supervision
# --------------------------------------------------------------------------- #

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
SKIP_DIRECTORIES = {"train_series", "test_series"}


def find_all_markers(pattern: str, max_depth: int = 4) -> list[Path]:
    """Every mounted directory containing a file matching `pattern`.

    The cache is built as four shard kernels and mounted as four separate
    inputs. Finding only the first would silently train on a quarter of the
    data at full apparent success — the worst kind of bug, because the loss
    curve would look fine.
    """
    found = []
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        if any(e.is_file() and e.match(pattern) for e in entries):
            found.append(directory)
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRECTORIES:
                frontier.append((entry, depth + 1))
    return found


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


def report_environment():
    """Record the accelerator actually granted.

    The Kaggle CLI does not expose the valid `machine_shape` strings, so which
    GPU a kernel receives has been UNVERIFIED for this project. This prints it,
    which matters because the current PyTorch build ships no Pascal kernels and
    a P100 would fail rather than run slowly.
    """
    import torch

    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            major, minor = torch.cuda.get_device_capability(i)
            print(f"  GPU {i}: {name}  compute capability {major}.{minor}")
            if major < 7:
                print("  >>> WARNING: pre-Volta GPU. The current PyTorch build has no")
                print("  >>> Pascal kernels — request a T4 instead.")
    else:
        print("  no GPU visible; this will be very slow")


class StudyDataset:
    """Cached volumes plus soft targets and a supervision mask."""

    def __init__(self, cache_paths: dict, studies, targets, masks, weights, augment: bool):
        self.cache_paths = cache_paths
        self.studies = list(studies)
        self.targets = targets
        self.masks = masks
        self.weights = weights
        self.augment = augment

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        import torch

        study = self.studies[index]
        volume = np.load(self.cache_paths[study])  # (3, S, H, W) uint8
        array = volume.astype(np.float32) / 255.0

        if self.augment:
            if np.random.rand() < 0.5:                      # flip along the slice axis
                array = array[:, ::-1].copy()
            shift = np.random.randint(-8, 9, size=2)        # small translation
            array = np.roll(array, shift, axis=(2, 3))
            array = np.clip(array * np.random.uniform(0.9, 1.1), 0.0, 1.0)

        return (torch.from_numpy(array),
                torch.from_numpy(self.targets[index]),
                torch.from_numpy(self.masks[index]),
                torch.tensor(self.weights[index], dtype=torch.float32))


def build_model(backbone: str, n_planes: int, n_out: int):
    """2.5D: a 2D backbone over slices, attention-pooled to a study.

    Attention pooling rather than mean pooling because a finding is usually
    visible on a handful of slices; averaging over twenty dilutes it.
    """
    import torch.nn as nn
    import torchvision

    weights = "DEFAULT"
    try:
        net = getattr(torchvision.models, backbone)(weights=weights)
    except Exception:  # noqa: BLE001 - no internet, or weights unavailable
        print("pretrained weights unavailable; training from scratch")
        net = getattr(torchvision.models, backbone)(weights=None)

    features = net.fc.in_features
    net.fc = nn.Identity()

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.attention = nn.Sequential(nn.Linear(features, 128), nn.Tanh(), nn.Linear(128, 1))
            self.head = nn.Linear(features, n_out)

        def forward(self, x):                      # x: (B, P, S, H, W)
            b, p, s, h, w = x.shape
            flat = x.reshape(b * p * s, 1, h, w).repeat(1, 3, 1, 1)
            embedded = self.backbone(flat).reshape(b, p * s, -1)
            scores = self.attention(embedded).softmax(dim=1)
            pooled = (embedded * scores).sum(dim=1)
            return self.head(pooled)

    return Model()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=RUN_FOLD)
    parser.add_argument("--epochs", type=int, default=RUN_EPOCHS)
    parser.add_argument("--batch", type=int, default=RUN_BATCH)
    parser.add_argument("--lr", type=float, default=RUN_LR)
    parser.add_argument("--backbone", default=RUN_BACKBONE)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--out", default="/kaggle/working")
    parser.add_argument("--time-budget", type=float, default=RUN_TIME_BUDGET)
    args, _unknown = parser.parse_known_args()

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader

    started = time.time()
    report_environment()

    cache_dirs = ([Path(args.cache)] if args.cache
                  else find_all_markers("cache_index_train_*.parquet"))
    artifacts = find_marker("soft_labels.parquet")
    headers_dir = find_marker("series_headers.parquet")
    if not cache_dirs or artifacts is None or headers_dir is None:
        raise SystemExit(f"missing inputs: cache={cache_dirs} labels={artifacts} "
                         f"headers={headers_dir}")
    print(f"cache shards mounted: {len(cache_dirs)}")
    for directory in cache_dirs:
        print(f"  {directory}")
    print(f"labels: {artifacts}\nheaders: {headers_dir}")

    soft = pd.read_parquet(artifacts / "soft_labels.parquet").set_index("StudyInstanceUID")

    # study -> file, across every shard. A duplicate would mean the shards
    # overlap, which would silently over-weight those studies.
    cache_paths: dict[str, Path] = {}
    duplicates = 0
    for directory in cache_dirs:
        for npy in directory.glob("*.npy"):
            if npy.stem in cache_paths:
                duplicates += 1
            cache_paths[npy.stem] = npy
    print(f"cached studies: {len(cache_paths):,}"
          + (f"   WARNING: {duplicates} duplicate studies across shards" if duplicates else ""))
    available = set(cache_paths)

    headers = pd.read_parquet(headers_dir / "series_headers.parquet")
    frequency = pd.to_numeric(headers.get("imaging_frequency"), errors="coerce")
    headers = headers.assign(imaging_frequency_rounded=frequency.round(2))
    fields = ["manufacturer", "model_name", "software_versions", "field_strength",
              "imaging_frequency_rounded", "transmit_coil"]
    headers["fingerprint"] = headers[fields].astype("string").fillna("?").agg("|".join, axis=1)
    groups = headers.groupby("StudyInstanceUID").fingerprint.agg(
        lambda s: s.value_counts().index[0])

    studies = sorted(available & set(soft.index) & set(groups.index))
    print(f"usable studies: {len(studies):,}")
    if len(studies) < 50:
        raise SystemExit("too few cached studies to train; build the cache first")

    targets = soft.loc[studies, FINDINGS].astype(float).to_numpy()
    channels = soft.loc[studies, [f"{f}__channel" for f in FINDINGS]].to_numpy()
    masks = np.ones_like(targets, dtype=np.float32)
    if ABSTAIN_MASKS_LOSS:
        masks[channels == "absent"] = 0.0
    targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)

    # Gold studies: every finding populated in train.csv. Weighted up because
    # they are the only targets known to match what the leaderboard scores.
    competition = find_marker("train.csv")
    weights = np.ones(len(studies), dtype=np.float32)
    if competition is not None:
        train_csv = pd.read_csv(competition / "train.csv").set_index("StudyInstanceUID")
        gold = train_csv[train_csv[FINDINGS].notna().all(axis=1)].index
        is_gold = np.array([s in set(gold) for s in studies])
        weights[is_gold] = GOLD_WEIGHT
        for position, study in enumerate(studies):
            if is_gold[position]:
                targets[position] = train_csv.loc[study, FINDINGS].to_numpy(dtype=np.float32)
                masks[position] = 1.0
        print(f"gold studies in cache: {int(is_gold.sum())} (weight {GOLD_WEIGHT})")

    group_values = groups.loc[studies].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(studies)), None, group_values))
    train_idx, val_idx = splits[args.fold]
    print(f"fold {args.fold}: train {len(train_idx):,}  val {len(val_idx):,}  "
          f"val groups {pd.Series(group_values[val_idx]).nunique()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.backbone, 3, len(FINDINGS)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    def make_loader(indices, augment, shuffle):
        dataset = StudyDataset(cache_paths, [studies[i] for i in indices],
                               targets[indices], masks[indices], weights[indices], augment)
        return DataLoader(dataset, batch_size=args.batch, shuffle=shuffle,
                          num_workers=2, pin_memory=(device == "cuda"))

    train_loader = make_loader(train_idx, True, True)
    val_loader = make_loader(val_idx, False, False)

    out_dir = Path(args.out)
    checkpoint_path = out_dir / f"checkpoint_fold{args.fold}.pt"
    start_epoch = 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        start_epoch = state["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for volumes, target, mask, weight in train_loader:
            volumes, target = volumes.to(device), target.to(device)
            mask, weight = mask.to(device), weight.to(device)
            optimiser.zero_grad()
            logits = model(volumes)
            loss = criterion(logits, target) * mask * weight.unsqueeze(1)
            denominator = (mask * weight.unsqueeze(1)).sum().clamp(min=1.0)
            loss = loss.sum() / denominator
            loss.backward()
            optimiser.step()
            total += float(loss)

        model.eval()
        predictions, truths, valid = [], [], []
        with torch.no_grad():
            for volumes, target, mask, _weight in val_loader:
                logits = model(volumes.to(device))
                predictions.append(torch.sigmoid(logits).cpu().numpy())
                truths.append(target.numpy())
                valid.append(mask.numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        valid = np.concatenate(valid)

        aucs = {}
        for i, finding in enumerate(FINDINGS):
            keep = valid[:, i] > 0
            y = (truths[keep, i] > 0.5).astype(int)
            if keep.sum() > 20 and 0 < y.sum() < len(y):
                aucs[finding] = roc_auc_score(y, predictions[keep, i])
        macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
        spread = float(np.mean(predictions.std(axis=0)))
        elapsed = time.time() - started
        print(f"epoch {epoch}  loss {total / max(len(train_loader), 1):.4f}  "
              f"val macro AUC {macro:.4f}  spread {spread:.4f}  {elapsed / 60:.1f} min",
              flush=True)
        history.append({"epoch": epoch, "macro_auc": round(macro, 4),
                        "spread": round(spread, 4),
                        "per_finding": {k: round(v, 4) for k, v in aucs.items()}})

        torch.save({"model": model.state_dict(), "optimiser": optimiser.state_dict(),
                    "epoch": epoch, "macro_auc": macro}, checkpoint_path)
        (out_dir / f"history_fold{args.fold}.json").write_text(json.dumps(history, indent=2))

        if time.time() - started > args.time_budget:
            print("time budget reached; checkpoint saved, re-run to resume")
            break

    best = max((h["macro_auc"] for h in history if h["macro_auc"] == h["macro_auc"]),
               default=float("nan"))
    print(f"\nbest val macro AUC (report-derived labels): {best:.4f}")
    print("Bar to clear: 0.669 — metadata alone, no pixels.")
    if best == best and best < 0.669:
        print(">>> Below the metadata baseline. The images are not contributing yet.")
    print("Remember: report-label CV overstates the leaderboard by ~0.138.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
