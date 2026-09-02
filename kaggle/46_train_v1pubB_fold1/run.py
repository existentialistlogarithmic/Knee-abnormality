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
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# GENERATED CONFIG — written by eda/generate_kernels.py from src/pipeline.py.
# Edit the manifest, not this file. Everything outside this block is shared by
# every kernel rendered from this template.
# --------------------------------------------------------------------------- #
# Second seed of the 0.923 configuration. Nothing changes but
# the random seed; this buys ensemble diversity, not a new idea.
#
# seed=1 is an explicit mechanism, not a label. Before the seed
# field existed this lineage would have differed from v1public
# only because two unseeded processes draw different entropy —
# true in practice, but nothing in the source said so and
# nothing would have caught it if it stopped being true.
#
RUN_FOLD            = 1
TARGET_MM_PER_PIXEL = 0.6
TARGET_SIZE         = 192
SLICES_PER_PLANE    = 20
RUN_EPOCHS          = 24
RUN_BATCH           = 16
ACCUM_STEPS         = 1
RUN_LR              = 0.0006
RUN_BACKBONE        = "resnet34"
SLICE_SUBSAMPLE     = None
INPUT_NORM          = False
PER_FINDING_POOL    = False
FOCAL_K             = 0
RUN_SEED            = 1
FULL_FIT_EPOCH      = 20
RUN_TIME_BUDGET     = 7.5 * 3600
GOLD_WEIGHT         = 8.0
ABSTAIN_MASKS_LOSS  = True
WARMUP_EPOCHS       = 2
EMA_DECAY           = 0.999
LABEL_SMOOTH        = 0.02
# --------------------------------------------------------------------------- #

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
SKIP_DIRECTORIES = {"train_series", "test_series"}

# Augmentation magnitudes follow the geometry rather than being retyped per
# lineage: a 12px shift at 192px and an 18px shift at 288px are the same shift.
SHIFT_PIXELS = TARGET_SIZE // 16
BLOB_MIN, BLOB_MAX = TARGET_SIZE // 12, TARGET_SIZE // 4


# --------------------------------------------------------------------------- #
# from kaggle/_templates/_shared/discovery.py
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# from kaggle/_templates/_shared/cohort.py
# --------------------------------------------------------------------------- #
def build_cohort(cache_dirs, artifacts, headers_dir, competition, findings,
                 gold_weight=8.0, abstain_masks_loss=True, min_studies=50,
                 quiet=False):
    """Every study with a cached volume, a target and a scanner fingerprint.

    Returns a dict rather than a tuple: callers want different subsets of this
    and positional unpacking across two kernels is one more thing to get out of
    step.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    def say(*args):
        if not quiet:
            print(*args, flush=True)

    soft = pd.read_parquet(artifacts / "soft_labels.parquet").set_index("StudyInstanceUID")

    # study -> file, across every shard. A duplicate would mean the shards
    # overlap, which would silently over-weight those studies.
    cache_paths = {}
    duplicates = 0
    for directory in cache_dirs:
        for npy in directory.glob("*.npy"):
            if npy.stem in cache_paths:
                duplicates += 1
            cache_paths[npy.stem] = npy
    say(f"cached studies: {len(cache_paths):,}"
        + (f"   WARNING: {duplicates} duplicate studies across shards" if duplicates else ""))
    available = set(cache_paths)

    # The fold-grouping key. No site column exists anywhere in this dataset, so
    # folds group on a scanner fingerprint; random K-fold inflates macro AUC by
    # 0.087 (FINDINGS.md §9). Frequency is rounded to 2 dp because the raw value
    # is near-unique per study, which would make the grouping fake.
    headers = pd.read_parquet(headers_dir / "series_headers.parquet")
    frequency = pd.to_numeric(headers.get("imaging_frequency"), errors="coerce")
    headers = headers.assign(imaging_frequency_rounded=frequency.round(2))
    fields = ["manufacturer", "model_name", "software_versions", "field_strength",
              "imaging_frequency_rounded", "transmit_coil"]
    headers["fingerprint"] = headers[fields].astype("string").fillna("?").agg("|".join, axis=1)
    groups = headers.groupby("StudyInstanceUID").fingerprint.agg(
        lambda s: s.value_counts().index[0])

    # sorted() is load-bearing: it is what makes the ordering, and therefore the
    # fold split, reproducible across kernels and across runs.
    studies = sorted(available & set(soft.index) & set(groups.index))
    say(f"usable studies: {len(studies):,}")
    if len(studies) < min_studies:
        raise SystemExit(f"only {len(studies)} usable studies (min {min_studies}); "
                         "the cache mount is probably wrong — build the cache first")

    targets = soft.loc[studies, findings].astype(float).to_numpy()
    channels = soft.loc[studies, [f"{f}__channel" for f in findings]].to_numpy()
    masks = np.ones_like(targets, dtype=np.float32)
    if abstain_masks_loss:
        masks[channels == "absent"] = 0.0

    # Per-finding confidence, when the labeler recorded it. The loss already
    # multiplies by `mask`, so a continuous mask IS a confidence weight — an
    # explicit "severe" contributes fully, a hedge contributes less, without any
    # change to the loss itself. Older label files have no such column and are
    # unaffected, which is what keeps this comparable to the runs before it.
    confidence_columns = [f"{f}__weight" for f in findings]
    if all(column in soft.columns for column in confidence_columns):
        confidence = soft.loc[studies, confidence_columns].to_numpy(dtype=np.float32)
        masks = masks * np.nan_to_num(confidence, nan=1.0)
        say(f"per-finding confidence weights in use "
            f"(mean {float(masks[masks > 0].mean()):.3f} where supervised)")
    else:
        say("no per-finding confidence column; every supervised target counts equally")

    targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)

    # Gold studies: every finding populated in train.csv. Weighted up because
    # they are the only targets known to match what the leaderboard scores.
    weights = np.ones(len(studies), dtype=np.float32)
    is_gold = np.zeros(len(studies), bool)
    if competition is not None:
        train_csv = pd.read_csv(competition / "train.csv").set_index("StudyInstanceUID")
        gold = train_csv[train_csv[findings].notna().all(axis=1)].index
        is_gold = np.array([s in set(gold) for s in studies])
        weights[is_gold] = gold_weight
        for position, study in enumerate(studies):
            if is_gold[position]:
                targets[position] = train_csv.loc[study, findings].to_numpy(dtype=np.float32)
                masks[position] = 1.0
        say(f"gold studies in cache: {int(is_gold.sum())} (weight {gold_weight})")

    group_values = groups.loc[studies].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(studies)), None, group_values))

    return {"studies": studies, "cache_paths": cache_paths, "targets": targets,
            "masks": masks, "weights": weights, "is_gold": is_gold,
            "group_values": group_values, "splits": splits}


# --------------------------------------------------------------------------- #
# from kaggle/_templates/_shared/model.py
# --------------------------------------------------------------------------- #
# ImageNet statistics. Every pretrained backbone here — torchvision and DINOv2
# alike — was trained on inputs normalised this way. The earliest runs fed raw
# 0..1 values straight in, which shifts the input distribution away from what
# the pretrained filters expect and quietly costs transfer quality. It never
# errors; it just makes the pretrained weights worth less than they should be.
#
# It is therefore a property of a trained model, not a preference: INPUT_NORM
# comes from the manifest at training time and from the checkpoint at inference
# time, and mixing the two in an ensemble is refused rather than averaged.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2 uses 14-pixel patches, so its input side must be a multiple of 14.
# A 192px cache becomes 196 here — a 2% resize — which keeps one cache usable
# by every architecture instead of forcing a rebuild per backbone.
PATCH_MULTIPLE = 14


def build_model(backbone: str, n_planes: int, n_out: int, normalise_input: bool,
                per_finding_pool: bool = False, focal_k: int = 0):
    """2.5D: a 2D backbone over slices, attention-pooled to a study.

    Attention pooling rather than mean pooling because a finding is usually
    visible on a handful of slices; averaging over twenty dilutes it.

    Accepts either a torchvision name or a timm name. DINOv2 is the reason:
    the public baseline for this competition reportedly reaches ~0.809 with
    DINOv2 features while this project's ImageNet resnet34 reached 0.725, and
    self-supervised features transfer to medical imaging far better than
    ImageNet classification features do.
    """
    import torch
    import torch.nn as nn

    net = None
    features = None
    if "." in backbone or backbone.startswith(("vit_", "convnext", "tf_efficientnet")):
        import timm

        try:
            net = timm.create_model(backbone, pretrained=True, num_classes=0,
                                    dynamic_img_size=True)
        except Exception:  # noqa: BLE001 - offline, or no dynamic_img_size support
            try:
                net = timm.create_model(backbone, pretrained=True, num_classes=0)
            except Exception:  # noqa: BLE001 - internet off (inference kernels)
                print("no pretrained download (internet off) — random init; "
                      "at inference the checkpoint replaces all of it")
                net = timm.create_model(backbone, pretrained=False, num_classes=0,
                                        dynamic_img_size=True)
        features = net.num_features
    else:
        import torchvision

        try:
            net = getattr(torchvision.models, backbone)(weights="DEFAULT")
        except Exception:  # noqa: BLE001
            print("no pretrained download (internet off) — random init; "
                  "at inference the checkpoint replaces all of it")
            net = getattr(torchvision.models, backbone)(weights=None)
        features = net.fc.in_features
        net.fc = nn.Identity()

    is_patch_model = backbone.startswith("vit_")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.patch_multiple = PATCH_MULTIPLE if is_patch_model else 0
            self.normalise = normalise_input
            # persistent=False deliberately: these are constants, not learned
            # state. Persisting them would add two keys to every state_dict and
            # the strict-load check at inference would then reject every
            # checkpoint written before they existed — including the folds
            # currently training.
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                                 persistent=False)
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                                 persistent=False)
            # One attention map for twelve findings forces a single compromise
            # over which slices matter. A meniscal tear occupies a handful of
            # sagittal slices and a large effusion occupies most of the study,
            # so the compromise is paid mostly by the focal findings — which are
            # exactly the four weakest here (Medial Meniscus 0.656, PF OA 0.659,
            # Synovitis 0.663, MCL 0.669 on fold 1).
            #
            # With per_finding_pool each finding gets its own attention map over
            # the same slice embeddings and its own weight vector over features.
            # Cost is one 128xN linear and an einsum over ~60 tokens: negligible
            # against the backbone, which is why this is worth trying before
            # anything that buys AUC with runtime.
            self.per_finding = per_finding_pool
            maps = n_out if per_finding_pool else 1
            self.attention = nn.Sequential(nn.Linear(features, 128), nn.Tanh(),
                                           nn.Linear(128, maps))
            if per_finding_pool:
                self.head_weight = nn.Parameter(torch.zeros(n_out, features))
                nn.init.trunc_normal_(self.head_weight, std=0.02)
                self.head_bias = nn.Parameter(torch.zeros(n_out))
            else:
                self.head = nn.Linear(features, n_out)

            # Focal pooling. Measured motivation (E027): against the same 58
            # expert-labelled studies, this model BEATS its own teacher on every
            # diffuse finding — Effusion 0.719 -> 0.924, Lateral OA 0.534 ->
            # 0.723 — and LOSES to it on every focal one: Medial Meniscus
            # 0.744 -> 0.516, MCL 0.820 -> 0.612, PF OA 0.828 -> 0.672,
            # ACL 0.784 -> 0.662. Focal 0.632 against a 0.798 teacher; diffuse
            # 0.783 against a 0.688 teacher.
            #
            # That is what a weighted MEAN over sixty slice embeddings does. A
            # meniscal tear is on three of them, an effusion is on most. So take
            # the top-k slices per finding as well, and let a learned per-finding
            # blend decide which pooling that finding wants. Diffuse findings can
            # keep the mean; focal ones can read off their few slices.
            self.focal_k = focal_k
            if focal_k:
                # 0 -> sigmoid 0.5: both paths start with equal weight and equal
                # gradient, rather than one starting switched off.
                self.mix = nn.Parameter(torch.zeros(n_out))

        def forward(self, x):                      # x: (B, P, S, H, W)
            b, p, s, h, w = x.shape
            flat = x.reshape(b * p * s, 1, h, w).repeat(1, 3, 1, 1)

            # Patch-based backbones need a side length divisible by the patch
            # size. Resizing here rather than in the cache keeps one cache
            # usable by every architecture.
            if self.patch_multiple and (h % self.patch_multiple or w % self.patch_multiple):
                side = int(round(h / self.patch_multiple)) * self.patch_multiple
                flat = torch.nn.functional.interpolate(
                    flat, size=(side, side), mode="bilinear", align_corners=False)

            if self.normalise:
                flat = (flat - self.mean) / self.std
            embedded = self.backbone(flat).reshape(b, p * s, -1)
            scores = self.attention(embedded).softmax(dim=1)   # (B, T, maps)
            if self.per_finding:
                # (B, T, F) x (B, T, C) -> (B, F, C): one pooled vector per
                # finding, each attending wherever that finding actually lives.
                pooled = torch.einsum("btf,btc->bfc", scores, embedded)
                averaged = (pooled * self.head_weight).sum(-1) + self.head_bias
            else:
                averaged = self.head((embedded * scores).sum(dim=1))

            if not self.focal_k:
                return averaged

            # Score every slice on every finding, then keep the best few. This
            # is the path a focal finding can win on: it never averages over the
            # fifty-odd slices the finding is not on.
            if self.per_finding:
                per_slice = (torch.einsum("btc,fc->btf", embedded, self.head_weight)
                             + self.head_bias)
            else:
                per_slice = self.head(embedded)                # (B, T, F)
            k = min(self.focal_k, per_slice.shape[1])
            strongest = per_slice.topk(k, dim=1).values.mean(dim=1)
            weight = torch.sigmoid(self.mix)                   # (F,)
            return weight * averaged + (1 - weight) * strongest

    return Model()


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

        # Slice subsampling: random while training, evenly spaced when scoring,
        # so the model always sees the same slice count either way. None means
        # every cached slice, and the whole block is a no-op.
        if SLICE_SUBSAMPLE and array.shape[1] > SLICE_SUBSAMPLE:
            if self.augment:
                idx = np.sort(np.random.choice(array.shape[1], SLICE_SUBSAMPLE, replace=False))
            else:
                idx = np.linspace(0, array.shape[1] - 1, SLICE_SUBSAMPLE).round().astype(int)
            array = array[:, idx]

        if self.augment:
            # NOTE: no left-right flip. Right knees were already mirrored during
            # the cache build so every volume shows the same anatomy; flipping
            # here would undo that and make medial/lateral findings ambiguous —
            # and four of the twelve targets are explicitly medial or lateral.
            # NO slice-order reversal either, and this one is not a judgement
            # call — it is arithmetic. The model embeds each slice
            # independently and pools with a softmax-weighted sum over the
            # token axis, with no positional encoding anywhere, so it is
            # EXACTLY permutation-invariant over slices: measured max
            # |f(x) - f(reverse(x))| = 2.4e-7, and the same for a random
            # permutation (E050). Every augmentation below is order-independent
            # too, so reversing here produced a volume the model could not
            # distinguish from the one it already had. It was a copy of every
            # second training sample in exchange for nothing.
            #
            # `test_the_model_is_permutation_invariant_over_slices` pins the
            # property this rests on. If that test ever fails, the architecture
            # has gained slice-order sensitivity and this augmentation should
            # come back with it.
            shift = np.random.randint(-SHIFT_PIXELS, SHIFT_PIXELS + 1, size=2)
            array = np.roll(array, shift, axis=(2, 3))
            array = array * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05)
            if np.random.rand() < 0.3:                       # coarse dropout
                size = np.random.randint(BLOB_MIN, BLOB_MAX)
                y = np.random.randint(0, max(1, array.shape[2] - size))
                x = np.random.randint(0, max(1, array.shape[3] - size))
                array[:, :, y:y + size, x:x + size] = 0.0
            if np.random.rand() < 0.3:                       # drop a whole plane
                array[np.random.randint(0, array.shape[0])] = 0.0
            array = np.clip(array, 0.0, 1.0)

        return (torch.from_numpy(array),
                torch.from_numpy(self.targets[index]),
                torch.from_numpy(self.masks[index]),
                torch.tensor(self.weights[index], dtype=torch.float32))


# ImageNet statistics. Every pretrained backbone here — torchvision and DINOv2
# alike — was trained on inputs normalised this way. The earlier runs fed raw
# 0..1 values straight in, which shifts the input distribution away from what
# the pretrained filters expect and quietly costs transfer quality. It never
# errors; it just makes the pretrained weights worth less than they should be.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2 uses 14-pixel patches, so its input side must be a multiple of 14.
# The cache is 192; 196 is the nearest multiple and a 2% resize.
PATCH_MULTIPLE = 14


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=RUN_FOLD)
    parser.add_argument("--epochs", type=int, default=RUN_EPOCHS)
    parser.add_argument("--batch", type=int, default=RUN_BATCH)
    parser.add_argument("--lr", type=float, default=RUN_LR)
    parser.add_argument("--backbone", default=RUN_BACKBONE)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--min-studies", type=int, default=50,
                        help="guard against training on an empty cache mount")
    parser.add_argument("--labels", default=None,
                        help="directory holding soft_labels.parquet (local runs)")
    parser.add_argument("--headers", default=None,
                        help="directory holding series_headers.parquet (local runs)")
    parser.add_argument("--train-csv", default=None,
                        help="competition train.csv. Only needed off Kaggle, "
                             "where there is no /kaggle/input to search.")
    parser.add_argument("--out", default="/kaggle/working")
    parser.add_argument("--time-budget", type=float, default=RUN_TIME_BUDGET)
    args, _unknown = parser.parse_known_args()

    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    started = time.time()

    # RUN_SEED is None for every lineage that ran before this existed, and that
    # is deliberate: seeding them now would not reproduce their checkpoints, it
    # would only claim to. Where it is set, it makes two things true that were
    # previously only assumed — the run is reproducible, and a sibling lineage
    # with a different seed is a *provably* different draw rather than one that
    # happens to differ because two processes each seeded from OS entropy.
    #
    # Seeding torch's global generator is what reaches the augmentation as well
    # as the head init: DataLoader draws a fresh base_seed from that generator
    # for every epoch's worker pool, and each worker reseeds numpy from it. So
    # augmentation stays fresh epoch to epoch (verified — the known
    # numpy-inherited-across-fork bug does not apply on torch >= 1.9) while the
    # whole sequence still descends from this one number.
    if RUN_SEED is not None:
        random.seed(RUN_SEED)
        np.random.seed(RUN_SEED)
        torch.manual_seed(RUN_SEED)
        torch.cuda.manual_seed_all(RUN_SEED)
        print(f"seeded: {RUN_SEED}")
    else:
        print("unseeded (RUN_SEED is None)")

    if not report_environment():
        # Exit immediately rather than burning a session that cannot possibly
        # run. This makes probing accelerator strings cost seconds, not hours.
        print("\nUnusable accelerator — stopping before any work.")
        return 0

    cache_dirs = ([Path(args.cache)] if args.cache
                  else find_all_markers("cache_index_train_*.parquet"))
    artifacts = Path(args.labels) if args.labels else find_marker("soft_labels.parquet")
    headers_dir = Path(args.headers) if args.headers else find_marker("series_headers.parquet")
    if not cache_dirs or artifacts is None or headers_dir is None:
        raise SystemExit(f"missing inputs: cache={cache_dirs} labels={artifacts} "
                         f"headers={headers_dir}")
    print(f"cache shards mounted: {len(cache_dirs)}")
    for directory in cache_dirs:
        print(f"  {directory}")
    print(f"labels: {artifacts}\nheaders: {headers_dir}")

    # Study set, targets, masks and the fold split all come from the shared
    # cohort builder. It is shared rather than copied because an out-of-fold
    # score is only out-of-fold if every kernel cuts the folds identically.
    # Off Kaggle there is no /kaggle/input to search, so the path is passed in.
    # Every other input already had an override; this one did not, and it was
    # the single thing stopping the generated trainer running anywhere else.
    train_csv = Path(args.train_csv) if args.train_csv else find_marker("train.csv")
    cohort = build_cohort(cache_dirs, artifacts, headers_dir, train_csv,
                          FINDINGS, gold_weight=GOLD_WEIGHT,
                          abstain_masks_loss=ABSTAIN_MASKS_LOSS,
                          min_studies=args.min_studies)
    studies = cohort["studies"]
    cache_paths = cohort["cache_paths"]
    targets, masks, weights = cohort["targets"], cohort["masks"], cohort["weights"]
    is_gold, splits = cohort["is_gold"], cohort["splits"]
    group_values = cohort["group_values"]

    # A NEGATIVE fold means full-fit: train on every study, hold nothing out.
    #
    # The fold split exists to produce an out-of-fold score, not because the
    # model needs it. Once a configuration is settled, holding out a fifth of
    # the corpus costs a fifth of the training data for a number that has
    # already been measured — and it costs more than that on the part that
    # matters, because each fold model never sees ~12 of the 58 expert studies,
    # which carry GOLD_WEIGHT and are the only labels known to match what the
    # leaderboard scores. A full-fit model sees all 58.
    #
    # It cannot be validated, and that is the trade rather than an oversight.
    # There is no honest val AUC to early-stop on, so the export epoch is fixed
    # at FULL_FIT_EPOCH — measured, not guessed: across the five v1public folds
    # the mean val AUC peaks at epoch 20 and the mean gold AUC at 21, with both
    # curves flat over 18-21 (E055). A leaked val set is deliberately NOT used
    # to pick the epoch.
    full_fit = args.fold < 0
    if full_fit:
        train_idx = np.arange(len(studies))
        # Monitored and printed, never scored on: these studies are in training.
        val_idx = splits[0][1]
        print(f"FULL FIT: train {len(train_idx):,} (every study)  "
              f"monitor {len(val_idx):,} (IN TRAINING — not a held-out score)")
        print(f"export fixed at epoch {FULL_FIT_EPOCH}; no early stopping")
    else:
        train_idx, val_idx = splits[args.fold]
        print(f"fold {args.fold}: train {len(train_idx):,}  val {len(val_idx):,}  "
              f"val groups {pd.Series(group_values[val_idx]).nunique()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.backbone, 3, len(FINDINGS), INPUT_NORM,
                        PER_FINDING_POOL, FOCAL_K).to(device)

    # Both cards were sitting idle: NvidiaTeslaT4 grants two, and the first run
    # used one. With AMP as well this is roughly a 4x throughput change, which is
    # what pays for the larger backbone and the longer schedule.
    core = model
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"DataParallel across {torch.cuda.device_count()} GPUs")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"mixed precision: {use_amp}")

    # Exponential moving average of the weights. Cheap, and it reliably beats the
    # final step when the targets are noisy — which here they are by construction.
    ema = {k: v.detach().clone().float() for k, v in core.state_dict().items()}

    def make_loader(indices, augment, shuffle):
        dataset = StudyDataset(cache_paths, [studies[i] for i in indices],
                               targets[indices], masks[indices], weights[indices], augment)
        return DataLoader(dataset, batch_size=args.batch, shuffle=shuffle,
                          num_workers=2, pin_memory=(device == "cuda"))

    train_loader = make_loader(train_idx, True, True)
    val_loader = make_loader(val_idx, False, False)

    # The gold studies inside this fold's VALIDATION set are the only expert
    # labels this model never trained on. Report-label CV has been shown to
    # mis-rank models (FINDINGS.md §11), so this is the signal that matters,
    # even though a single fold holds far too few studies to stand alone. The
    # raw predictions are written out so the five folds can be pooled into one
    # n=58 out-of-fold evaluation rather than averaging five tiny AUCs.
    val_studies = [studies[i] for i in val_idx]
    gold_positions = [pos for pos, i in enumerate(val_idx) if is_gold[i]]
    # In full-fit mode NOTHING is held out — val_idx is the monitor set and it
    # is inside training. Printing "held out: 12" next to a per-epoch gold
    # column that climbs to 0.99 invites exactly the misreading this project
    # has made most often: a number that is not a score, read as one. E064's
    # full-fit member had to be explained twice for the same reason.
    if full_fit:
        print(f"gold studies in the monitor set: {len(gold_positions)} "
              "— IN TRAINING, held out from nothing. Any gold figure below is "
              "memorisation, not a score.")
    else:
        print(f"gold studies held out in this fold: {len(gold_positions)}")

    out_dir = Path(args.out)
    # "foldall" rather than "fold-1" on purpose. Inference globs
    # checkpoint_fold*.pt so this still joins the ensemble, while gold_eval
    # parses the fold out of the filename with int() and skips what it cannot
    # read — which is exactly right for a model that trained on every gold
    # study and must never be scored as if it had not.
    tag = "all" if full_fit else str(args.fold)
    checkpoint_path = out_dir / f"checkpoint_fold{tag}.pt"

    # /kaggle/working does not survive a session, so a checkpoint from a previous
    # run only exists if that run's output is mounted. Look there too, otherwise
    # every re-run silently restarts from scratch and the wall-clock guard
    # becomes useless.
    resume_from = checkpoint_path if checkpoint_path.exists() else None
    if resume_from is None:
        mounted = find_marker(f"checkpoint_fold{tag}.pt")
        if mounted is not None:
            resume_from = mounted / f"checkpoint_fold{tag}.pt"
            print(f"found a mounted checkpoint: {resume_from}")

    start_epoch = 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=False)

        # The ImageNet mean/std were briefly persistent buffers, so checkpoints
        # written in that window carry two extra keys that the current model
        # rebuilds for itself. Inference already drops them; resume must too, or
        # a strict load rejects the run it is meant to continue. That is not
        # hypothetical: knee-train-dinov2 is exactly such a checkpoint.
        def usable(weights):
            return {k: v for k, v in weights.items() if k not in ("mean", "std")}

        # "model" is now the best-scoring EMA export, which is NOT where
        # training left off. Older checkpoints have no "live" key and their
        # "model" was the live weights, so that is the correct fallback.
        core.load_state_dict(usable(state.get("live") or state["model"]))
        if state.get("ema"):
            ema = {k: v.to(device).float() for k, v in usable(state["ema"]).items()}
        optimiser.load_state_dict(state["optimiser"])
        start_epoch = state["epoch"] + 1
        print(f"resumed from epoch {start_epoch} "
              f"(previous best macro AUC {state.get('macro_auc', float('nan')):.4f})")

    import math

    steps_per_epoch = max(len(train_loader), 1)

    def learning_rate_at(epoch_index: int, step: int) -> float:
        """Linear warmup then cosine decay, computed per step rather than per epoch."""
        progress = epoch_index + step / steps_per_epoch
        if progress < WARMUP_EPOCHS:
            return args.lr * (progress + 1e-8) / WARMUP_EPOCHS
        span = max(args.epochs - WARMUP_EPOCHS, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * (progress - WARMUP_EPOCHS) / span))
        return args.lr * max(cosine, 0.02)

    history = []
    best_macro, best_epoch, best_state = float("-inf"), -1, None
    if resume_from is not None and state.get("macro_auc") == state.get("macro_auc"):
        # Inherit the best result already achieved. Without this a continuation
        # run starts its own tracker from nothing, and a warm restart that never
        # gets back above where it resumed would export weights WORSE than the
        # checkpoint it was handed — silently, since the log would only show the
        # new run's own best.
        best_macro = state["macro_auc"]
        best_epoch = state.get("best_epoch", state["epoch"])
        best_state = {k: v.detach().cpu().clone() for k, v in state["model"].items()}
        print(f"inherited best macro AUC {best_macro:.4f} from the mounted checkpoint")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total = 0.0
        for step, (volumes, target, mask, weight) in enumerate(train_loader):
            for group in optimiser.param_groups:
                group["lr"] = learning_rate_at(epoch, step)

            volumes, target = volumes.to(device), target.to(device)
            mask, weight = mask.to(device), weight.to(device)
            target = target * (1 - 2 * LABEL_SMOOTH) + LABEL_SMOOTH

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(volumes)
                loss = criterion(logits.float(), target) * mask * weight.unsqueeze(1)
                denominator = (mask * weight.unsqueeze(1)).sum().clamp(min=1.0)
                loss = loss.sum() / denominator

            scaler.scale(loss / ACCUM_STEPS).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)

            with torch.no_grad():
                for key, value in core.state_dict().items():
                    if value.dtype.is_floating_point:
                        ema[key].mul_(EMA_DECAY).add_(value.float(), alpha=1 - EMA_DECAY)
                    else:
                        ema[key] = value.detach().clone().float()

            total += float(loss.detach())

        # Evaluate the averaged weights, not the last step. Swap them in, score,
        # then restore so training continues from the live weights.
        live = {k: v.detach().clone() for k, v in core.state_dict().items()}
        core.load_state_dict({k: v.to(live[k].dtype) for k, v in ema.items()})

        model.eval()
        predictions, truths, valid = [], [], []
        with torch.no_grad():
            for volumes, target, mask, _weight in val_loader:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(volumes.to(device))
                predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
                truths.append(target.numpy())
                valid.append(mask.numpy())
        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        valid = np.concatenate(valid)
        ema_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        core.load_state_dict(live)

        aucs = {}
        for i, finding in enumerate(FINDINGS):
            keep = valid[:, i] > 0
            y = (truths[keep, i] > 0.5).astype(int)
            if keep.sum() > 20 and 0 < y.sum() < len(y):
                aucs[finding] = roc_auc_score(y, predictions[keep, i])
        macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
        spread = float(np.mean(predictions.std(axis=0)))

        # Gold-only macro AUC over this fold's held-out expert labels.
        gold_macro = float("nan")
        if gold_positions:
            g = np.array(gold_positions)
            gold_aucs = []
            for i in range(len(FINDINGS)):
                y = (truths[g, i] > 0.5).astype(int)
                if 0 < y.sum() < len(y):
                    gold_aucs.append(roc_auc_score(y, predictions[g, i]))
            if gold_aucs:
                gold_macro = float(np.mean(gold_aucs))
        elapsed = time.time() - started
        print(f"epoch {epoch}  loss {total / max(len(train_loader), 1):.4f}  "
              f"val macro AUC {macro:.4f}  gold {gold_macro:.4f}  "
              f"spread {spread:.4f}  {elapsed / 60:.1f} min", flush=True)
        history.append({"epoch": epoch, "macro_auc": round(macro, 4),
                        "gold_macro_auc": round(gold_macro, 4),
                        "spread": round(spread, 4),
                        "per_finding": {k: round(v, 4) for k, v in aucs.items()}})

        # Keep the best-scoring weights, not the most recent ones. Fold 1 of the
        # 192px run peaked at 0.7334 on epoch 18 and drifted down to 0.7282 by
        # epoch 23 — and epoch 23 is what got saved, so 0.005 was given away for
        # nothing. Over an ensemble that compounds.
        if full_fit:
            # No honest validation exists, so "best" is the measured epoch and
            # nothing else. Selecting on the monitor set would be selecting on
            # training data.
            if epoch == min(FULL_FIT_EPOCH, args.epochs - 1):
                best_macro, best_epoch, best_state = macro, epoch, ema_state
                print(f"  full-fit export taken at epoch {epoch}")
        elif macro == macro and macro > best_macro:      # NaN-safe
            best_macro, best_epoch = macro, epoch
            best_state = ema_state

        # Save the unwrapped module. A DataParallel state_dict carries a
        # "module." prefix on every key, and the inference kernel builds a plain
        # model — it would fail to load, or worse, load partially.
        # The saved weights are the EMA ones, i.e. exactly what was scored above.
        # Record the input geometry alongside the weights. Inference reads it
        # rather than assuming, which is what stops a config change here from
        # silently feeding the model something it never saw.
        # If every epoch scored NaN — too few positives in a fold to compute an
        # AUC — there is no best, and a None here would write a checkpoint that
        # inference cannot load.
        export = best_state if best_state is not None else ema_state
        torch.save({"model": export, "optimiser": optimiser.state_dict(),
                    "ema": ema_state, "live": {k: v.detach().cpu().clone()
                                               for k, v in core.state_dict().items()},
                    "epoch": epoch, "macro_auc": best_macro,
                    "best_epoch": best_epoch, "last_macro_auc": macro,
                    "backbone": args.backbone,
                    "slice_subsample": SLICE_SUBSAMPLE,
                    "input_norm": INPUT_NORM,
                    "per_finding_pool": PER_FINDING_POOL,
                    "focal_k": FOCAL_K,
                    }, checkpoint_path)
        (out_dir / f"history_fold{tag}.json").write_text(json.dumps(history, indent=2))

        # Raw predictions for the held-out gold studies, so the folds can be
        # pooled. Written from the same weights that are exported: when this
        # epoch set a new best, these are the best model's predictions.
        if gold_positions and best_epoch == epoch and not full_fit:
            g = np.array(gold_positions)
            (out_dir / f"gold_oof_fold{args.fold}.json").write_text(json.dumps({
                "fold": args.fold, "epoch": epoch, "backbone": args.backbone,
                "geometry": {"mm_per_pixel": TARGET_MM_PER_PIXEL,
                             "size": TARGET_SIZE, "slices": SLICES_PER_PLANE,
                             "slice_subsample": SLICE_SUBSAMPLE},
                "findings": FINDINGS,
                "studies": [val_studies[pos] for pos in gold_positions],
                "predicted": predictions[g].round(5).tolist(),
                "expert": truths[g].round(5).tolist(),
            }, indent=2))

        if time.time() - started > args.time_budget:
            print("time budget reached; checkpoint saved, re-run to resume")
            break

    best = best_macro if best_macro > float("-inf") else float("nan")
    print(f"\nbest val macro AUC (report-derived labels): {best:.4f} "
          f"(epoch {best_epoch}, and these are the weights saved)")
    print("Bar to clear: 0.669 — metadata alone, no pixels.")
    if best == best and best < 0.669:
        print(">>> Below the metadata baseline. The images are not contributing yet.")
    print("Remember: report-label CV overstates the leaderboard by ~0.138.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
