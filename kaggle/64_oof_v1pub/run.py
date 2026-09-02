"""Score already-trained checkpoints against the expert labels they never saw.

**This runs on CPU**, which is the point. The weekly GPU allowance is 30 hours
and it is spent; CPU sessions are a separate allowance. The work here is twelve
studies through one backbone — seconds — so there is no reason it should ever
have needed a GPU.

It exists because `knee-train` (fold 0) predates the gold dump that later
training runs emit, which leaves a hole in the middle of the only offline signal
this project trusts. Folds 1–4 pool to n=46; fold 0 holds the other 12. Without
them there is also no like-for-like baseline for the two fold-0 experiments —
per-finding attention pooling and DINOv2 — so both are currently unmeasurable
against the configuration they were meant to improve on.

The fold split comes from the shared cohort builder, not from a copy. An
out-of-fold score is only out-of-fold if this kernel cuts the folds exactly as
the training kernel did, and a second implementation that sorted studies
differently would produce a number that looks held-out and is not.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# GENERATED CONFIG — written by eda/generate_kernels.py from src/pipeline.py.
# Edit the manifest, not this file. Everything outside this block is shared by
# every kernel rendered from this template.
# --------------------------------------------------------------------------- #
# Out-of-fold predictions for every study in the corpus, from the
# five checkpoints behind 0.923. Each study is predicted exactly
# once, by the one model that held it out.
#
# This is the raw material for a distillation teacher, and the
# reason to want one is measured rather than assumed: the model
# scores 0.8980 on the 58 gold studies and the labels that trained
# it score 0.8927. The student has overtaken its teacher, and by
# 0.005 — which puts the two COMPARABLE, the condition E048
# identified as the difference between E023's union paying +0.070
# and the four unions since paying nothing. Every one of those four
# failures added a member 0.03-0.06 behind the incumbent. This is
# the first candidate that is not.
#
# It runs on CPU and therefore costs ZERO GPU quota — the whole
# point of building it from checkpoints that already exist rather
# than re-running five folds with a wider dump, which would have
# cost ~7.5 GPU-h for the same file.
#
# It also self-verifies. The gold macro is still computed from the
# gold subset, so this run must reproduce 0.8980. If it does not,
# the fold split here differs from the trainer's and the OOF is not
# out-of-fold — a teacher built on that would leak, train cleanly,
# and score worse for no visible reason.
#
# NOTHING IS BLENDED HERE. What to mix with the report labels, and
# at what weight, is decided offline on the 58 — not baked into a
# two-hour CPU run that would have to be repeated to revisit it.
#
TARGET_MM_PER_PIXEL = 0.6
TARGET_SIZE         = 192
SLICES_PER_PLANE    = 20
TTA_VIEWS           = ('identity',)
OOF_SCOPE           = "all"
# --------------------------------------------------------------------------- #

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
SKIP_DIRECTORIES = {"train_series", "test_series"}
OUT = Path("/kaggle/working")


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


def apply_view(volume, view):
    """One test-time view of a cached volume, as a fresh array.

    Only symmetries the model was *trained* to be invariant to belong here.
    Training reverses slice order with p=0.5 and rolls the image by up to
    SHIFT_PIXELS, so averaging over those views asks the model the same
    question from an angle it has already been taught not to care about.

    Two augmentations are deliberately excluded. Intensity jitter and coarse
    dropout are noise injected to regularise; averaging over noise adds
    variance without adding a view. And a left-right flip — the obvious TTA on
    any other dataset — is wrong here specifically: right knees were mirrored
    during the cache build so that every volume shows the same anatomy, and
    four of the twelve findings are explicitly medial or lateral. Flipping
    would undo that and make those four ambiguous.
    """
    shift = max(1, volume.shape[-1] // 16)      # SHIFT_PIXELS, as in training
    if view == "identity":
        return volume
    if view == "reverse":
        return volume[:, ::-1].copy()
    if view == "shift_pos":
        return np.roll(volume, (shift, shift), axis=(2, 3))
    if view == "shift_neg":
        return np.roll(volume, (-shift, -shift), axis=(2, 3))
    raise SystemExit(f"unknown TTA view {view!r}")


def main() -> int:
    import torch

    started = time.time()
    cache_dirs = find_all_markers("cache_index_train_*.parquet")
    artifacts = find_marker("soft_labels.parquet")
    headers_dir = find_marker("series_headers.parquet")
    competition = find_marker("train.csv")
    if not cache_dirs or artifacts is None or headers_dir is None:
        raise SystemExit(f"missing inputs: cache={cache_dirs} labels={artifacts} "
                         f"headers={headers_dir}")
    print(f"cache shards mounted: {len(cache_dirs)}")

    cohort = build_cohort(cache_dirs, artifacts, headers_dir, competition, FINDINGS)
    studies = cohort["studies"]
    cache_paths = cohort["cache_paths"]
    targets, is_gold, splits = cohort["targets"], cohort["is_gold"], cohort["splits"]
    if not is_gold.any():
        raise SystemExit("no gold studies found — train.csv did not mount")

    checkpoints = sorted(Path("/kaggle/input").glob("notebooks/*/*/checkpoint_fold*.pt"))
    if not checkpoints:
        raise SystemExit("no checkpoints mounted")
    print(f"checkpoints mounted: {len(checkpoints)}")

    device = "cpu"
    for path in checkpoints:
        # The fold is in the FILENAME, not the checkpoint — training never
        # recorded it. Reading it from anywhere else would risk scoring a model
        # on studies it trained on, which is the one mistake this kernel exists
        # to avoid making.
        stem = path.stem                       # checkpoint_fold{n}
        try:
            fold = int(stem.rsplit("fold", 1)[1])
        except (IndexError, ValueError):
            print(f"  cannot read a fold number from {path.name}; skipping")
            continue

        state = torch.load(path, map_location=device, weights_only=False)
        backbone = state.get("backbone", "resnet18")
        normalise = bool(state.get("input_norm", False))
        pooling = bool(state.get("per_finding_pool", False))
        focal = int(state.get("focal_k", 0) or 0)
        subsample = state.get("slice_subsample")

        model = build_model(backbone, 3, len(FINDINGS), normalise, pooling,
                            focal).to(device)
        weights = {k: v for k, v in state["model"].items() if k not in ("mean", "std")}
        try:
            missing, unexpected = model.load_state_dict(weights, strict=False)
        except RuntimeError as exc:
            print(f"  {path.name}: architecture mismatch, skipping ({exc})")
            continue
        if missing or unexpected:
            print(f"  {path.name}: weight mismatch "
                  f"missing={len(missing)} unexpected={len(unexpected)}; skipping")
            continue
        model.eval()

        _train_idx, val_idx = splits[fold]
        # OOF_SCOPE "gold" predicts only the held-out expert studies, which is
        # what this kernel has always done and what pool_gold_oof.py reads.
        # "all" predicts the WHOLE holdout, which is how a self-distillation
        # teacher gets built: every study in the corpus predicted exactly once,
        # by the model that did not train on it.
        #
        # The gold number is computed from the gold subset either way, so a
        # scope="all" run still reproduces the figure on record. That is the
        # point of computing it: if the pooled macro is not 0.8980, this kernel
        # cut the folds differently from the trainer and its OOF is not
        # out-of-fold. A teacher built from a bad split would leak, train
        # cleanly, and score worse for no visible reason.
        positions = (list(val_idx) if OOF_SCOPE == "all"
                     else [p for p in val_idx if is_gold[p]])
        gold_rows = [i for i, p in enumerate(positions) if is_gold[p]]
        print(f"\n{path.parent.parent.name}/{path.name}: fold {fold}, "
              f"{backbone}, norm={normalise}, pool={pooling}, "
              f"subsample={subsample}, {len(positions)} held-out studies "
              f"({len(gold_rows)} gold)")
        if not positions or not gold_rows:
            continue

        # One row per study per view. TTA_VIEWS = ("identity",) reproduces the
        # single deterministic pass this kernel has always made, so the first
        # view is always directly comparable to every gold number on record.
        by_view = {view: [] for view in TTA_VIEWS}
        for position in positions:
            volume = np.load(cache_paths[studies[position]]).astype(np.float32) / 255.0
            if subsample and volume.shape[1] > subsample:
                # Evenly spaced, exactly as validation did during training.
                idx = np.linspace(0, volume.shape[1] - 1, subsample).round().astype(int)
                volume = volume[:, idx]
            for view in TTA_VIEWS:
                with torch.no_grad():
                    logits = model(torch.from_numpy(apply_view(volume, view))[None])
                by_view[view].append(torch.sigmoid(logits.float())[0].numpy())
        by_view = {view: np.stack(rows) for view, rows in by_view.items()}
        predicted = by_view[TTA_VIEWS[0]]
        # Everything below scores and records GOLD only, so every number this
        # kernel has ever printed keeps its meaning under either scope.
        gold_positions = [positions[i] for i in gold_rows]
        gold_predicted = predicted[gold_rows]
        expert = targets[gold_positions]

        from sklearn.metrics import roc_auc_score

        aucs = {}
        for i, finding in enumerate(FINDINGS):
            y = (expert[:, i] > 0.5).astype(int)
            if 0 < y.sum() < len(y):
                aucs[finding] = roc_auc_score(y, gold_predicted[:, i])
        macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
        print(f"  gold macro AUC {macro:.4f} over {len(aucs)} scorable findings")

        if len(TTA_VIEWS) > 1:
            # A per-fold read only; the fold subsets are far too small to
            # decide anything (E031 put a single fold's interval at ~0.19).
            # The decision is made on the pooled n=58, offline.
            mean_of_views = np.mean(list(by_view.values()), axis=0)[gold_rows]
            tta_aucs = [roc_auc_score((expert[:, i] > 0.5).astype(int),
                                      mean_of_views[:, i])
                        for i, finding in enumerate(FINDINGS)
                        if finding in aucs]
            print(f"  {len(TTA_VIEWS)}-view mean {float(np.mean(tta_aucs)):.4f} "
                  f"(indicative only at n={len(positions)})")

        tag = path.parent.parent.name.replace("/", "_")
        (OUT / f"gold_oof_fold{fold}_{tag}.json").write_text(json.dumps({
            "fold": fold, "epoch": state.get("epoch"), "backbone": backbone,
            "source": tag,
            "geometry": {"mm_per_pixel": TARGET_MM_PER_PIXEL, "size": TARGET_SIZE,
                         "slices": SLICES_PER_PLANE, "slice_subsample": subsample},
            "findings": FINDINGS,
            "studies": [studies[p] for p in gold_positions],
            "predicted": gold_predicted.round(5).tolist(),
            "expert": expert.round(5).tolist(),
            # Every view is written out rather than only their mean, because
            # which subset of views to average is a question to settle offline
            # on the pooled n=58, not one to bake in here and re-run a session
            # to revisit.
            "views": list(TTA_VIEWS),
            "predicted_by_view": {view: rows[gold_rows].round(5).tolist()
                                  for view, rows in by_view.items()},
        }, indent=2))

        # The full holdout, written separately so the gold artifact keeps its
        # exact shape. One row per study, predicted by the one model that held
        # it out — this is the raw material for a distillation teacher, and it
        # is deliberately NOT blended with anything here. What to blend and at
        # what weight is a question for the 58 gold studies offline, not one to
        # bake into a two-hour CPU run.
        if OOF_SCOPE == "all":
            (OUT / f"oof_all_fold{fold}_{tag}.json").write_text(json.dumps({
                "fold": fold, "epoch": state.get("epoch"), "backbone": backbone,
                "source": tag, "findings": FINDINGS,
                "studies": [studies[p] for p in positions],
                "predicted": predicted.round(5).tolist(),
            }, indent=2))
            print(f"  wrote {len(positions)} out-of-fold predictions")

    print(f"\nwall clock {(time.time() - started) / 60:.1f} min on CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
