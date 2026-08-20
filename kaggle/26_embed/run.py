"""Freeze the backbone and write out its embeddings once, on CPU.

Fine-tuning this model on CPU would take **191 hours** — measured, not guessed.
Running the same backbone *frozen* over the whole corpus takes **2.2 hours**,
and everything above it — attention pooling, the focal top-k path, twelve heads,
73,380 parameters in total — then trains over the saved embeddings in **2.6
minutes for a full five-fold run**.

That is the difference between an experiment costing a GPU session and costing a
coffee. The weekly GPU allowance is 30 hours and it is spent; CPU sessions are a
separate allowance. So this kernel exists to turn "wait for quota" into "run the
A/B now".

**What it does and does not buy.** A frozen backbone is not the fine-tuned model
that scored 0.725, so absolute numbers from this rig do not transfer. What does
transfer is *comparisons above the backbone* — does per-finding pooling help,
does focal top-k pooling help, do the fused labels help — because those are
exactly the parts still being trained here. Worth noting that a published system
using a frozen DINOv2 with a trained head reached 0.776 on this leaderboard,
above this project's fine-tuned 0.725, so "frozen" is not automatically "worse".
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
# Writes frozen backbone embeddings for the whole corpus once, so
# that everything above the backbone can be trained in minutes
# instead of hours. Measured: fine-tuning on CPU is 191 hours,
# frozen extraction is 2.2 hours, and a five-fold run of the 73,380
# parameters above the backbone is 2.6 minutes.
#
# input_norm is True here even though the 0.725 model was trained
# without it. A FROZEN backbone has no chance to adapt to the wrong
# input distribution, so feeding it what it was pretrained on
# matters more here than it did there.
#
TARGET_MM_PER_PIXEL = 0.6
TARGET_SIZE         = 192
SLICES_PER_PLANE    = 20
RUN_BACKBONE        = "vit_small_patch14_dinov2.lvd142m"
INPUT_NORM          = True
RUN_MAX_STUDIES     = 0
EMBED_THREADS       = 4
RUN_TIME_BUDGET     = 10.0 * 3600
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

        def embed(self, x):                        # x: (B, P, S, H, W)
            """Per-slice backbone embeddings, (B, P*S, C).

            Everything a slice goes through before pooling lives here, so a
            caller that wants embeddings — the frozen-extraction kernel — gets
            the identical preprocessing rather than a second copy of it. The
            first version of that kernel called `self.backbone` directly and
            died on `Input height (192) should be divisible by patch size (14)`,
            having skipped the resize below and the normalisation with it.
            """
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
            return self.backbone(flat).reshape(b, p * s, -1)

        def forward(self, x):                      # x: (B, P, S, H, W)
            embedded = self.embed(x)
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


def main() -> int:
    import torch

    torch.set_num_threads(EMBED_THREADS)
    started = time.time()

    cache_dirs = find_all_markers("cache_index_train_*.parquet")
    artifacts = find_marker("soft_labels.parquet")
    headers_dir = find_marker("series_headers.parquet")
    if not cache_dirs or artifacts is None or headers_dir is None:
        raise SystemExit(f"missing inputs: cache={cache_dirs} labels={artifacts} "
                         f"headers={headers_dir}")

    cohort = build_cohort(cache_dirs, artifacts, headers_dir,
                          find_marker("train.csv"), FINDINGS)
    studies = cohort["studies"]
    cache_paths = cohort["cache_paths"]

    # The backbone only. Everything above it is what the cheap rig will train,
    # so none of it is used here.
    full = build_model(RUN_BACKBONE, 3, len(FINDINGS), INPUT_NORM, False, 0)
    full.eval()

    limit = RUN_MAX_STUDIES or len(studies)
    studies = studies[:limit]
    print(f"embedding {len(studies):,} studies with {RUN_BACKBONE} "
          f"on {EMBED_THREADS} CPU threads")

    embeddings, kept = [], []
    for index, study in enumerate(studies):
        volume = np.load(cache_paths[study]).astype(np.float32) / 255.0
        with torch.no_grad():
            # model.embed(), not model.backbone(). The backbone alone skips the
            # patch-multiple resize and the input normalisation, and a frozen
            # backbone fed the wrong thing has no chance to adapt to it. The
            # first version of this kernel called the backbone directly and died
            # on "Input height (192) should be divisible by patch size (14)".
            out = full.embed(torch.from_numpy(volume)[None])[0]
        # float16 halves the artefact and costs nothing measurable: these are
        # inputs to a linear layer, not accumulations.
        embeddings.append(out.to(torch.float16).numpy())
        kept.append(study)

        if (index + 1) % 200 == 0 or index + 1 == len(studies):
            elapsed = time.time() - started
            rate = (index + 1) / max(elapsed, 1e-6)
            print(f"  {index + 1:,}/{len(studies):,}  {elapsed / 60:.1f} min  "
                  f"{1 / rate:.2f} s/study  eta "
                  f"{(len(studies) - index - 1) / rate / 60:.1f} min", flush=True)
        if time.time() - started > RUN_TIME_BUDGET:
            print(f"  time budget reached after {index + 1:,} studies")
            break

    stacked = np.stack(embeddings)
    np.save(OUT / "embeddings.npy", stacked)
    (OUT / "embeddings_index.json").write_text(json.dumps({
        "backbone": RUN_BACKBONE, "input_norm": INPUT_NORM,
        "geometry": {"mm_per_pixel": TARGET_MM_PER_PIXEL, "size": TARGET_SIZE,
                     "slices": SLICES_PER_PLANE},
        "shape": list(stacked.shape), "dtype": str(stacked.dtype),
        "studies": kept,
    }, indent=2))
    print(f"\nwrote embeddings.npy {stacked.shape} {stacked.dtype} "
          f"({stacked.nbytes / 1e6:.0f} MB)")
    print(f"wall clock {(time.time() - started) / 60:.1f} min on CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
