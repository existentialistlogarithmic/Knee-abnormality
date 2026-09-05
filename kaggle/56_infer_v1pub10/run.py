"""Fold-ensemble inference — every 192px fold, averaged. Internet off.

All members share one geometry, so each study's volume is decoded and built
ONCE and reused across every model. That is the difference between an ensemble
costing 1x decode and Nx decode, and decode is the dominant cost.

Averaging folds is the highest-confidence gain available here: same config,
different data splits, no new hypothesis that can be wrong. The 288px variant
scored 0.668 against this config's 0.725, so this is the configuration worth
ensembling.

Builds each test study's volume in memory and predicts immediately, rather than
writing a test cache first. At ~1,300 studies that would be ~3 GB of intermediate
files for no benefit, and the measured budget is ~24 s/study against a 9-hour
cap while reaching the data costs 0.059 s/study.

**No report text exists at test time** (`FINDINGS.md` §2.6) — the host states it
outright. So the labeler that produced the training targets is absent here by
design; this kernel sees pixels and DICOM headers only.

The volume-building and model code below is copied verbatim from
`kaggle/03_cache_build/run.py` and `kaggle/04_train/run.py`. A Kaggle script
kernel is one file, and re-typing either would be how train/inference skew gets
introduced — the failure mode where the model is fed subtly different input than
it was trained on and nobody notices because nothing errors.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

# --------------------------------------------------------------------------- #
# GENERATED CONFIG — written by eda/generate_kernels.py from src/pipeline.py.
# Edit the manifest, not this file. Everything outside this block is shared by
# every kernel rendered from this template.
# --------------------------------------------------------------------------- #
# Ten members: v1public plus its second seed.
#
# The only lever with a board-measured coefficient that survived
# the sweep of E050-E058. E036 measured ensembling at +0.032 going
# from one fold to five; log-scaling puts five to ten at roughly
# +0.010. Nothing here is a new hypothesis, which is the point.
#
# E054 measured a ten-member ensemble WORSE than its five when the
# added members were 0.107 behind (v1fused, -0.0296 [-0.046,
# -0.015]). That result is why this one is not assumed: the
# difference is that a reseed is an equal-strength member rather
# than a weak one, and E054's rule decides it on gold before this
# kernel is pushed, not after.
#
# Budget is not a constraint and was measured rather than guessed:
# 0.98 h projected for 1,300 test studies with five members
# against a 9 h cap, and 0.037 h per extra member. Ten members is
# ~1.2 h.
#
TARGET_MM_PER_PIXEL      = 0.6
TARGET_SIZE              = 192
SLICES_PER_PLANE         = 20
BATCH_STUDIES            = 4
SLICE_SUBSAMPLE_EXPECTED = None
INPUT_NORM_EXPECTED      = False
MEMBERS_EXPECTED         = 10
# --------------------------------------------------------------------------- #

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
PLANES = ("Sagittal", "Coronal", "Axial")
SKIP_DIRECTORIES = {"train_series", "test_series"}
OUTPUT = Path("/kaggle/working/submission.csv")
FALLBACK_PRIOR = 0.3     # used only if a study cannot be read at all


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
# from kaggle/_templates/_shared/volume.py
# --------------------------------------------------------------------------- #
def normalise(volume: np.ndarray) -> np.ndarray:
    """Percentile clip then scale to uint8.

    Per-volume rather than per-slice: MRI intensity is arbitrary between studies
    but consistent within one acquisition, and per-slice normalisation would
    destroy the relative brightness that distinguishes fluid from fat.
    """
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    # Non-finite values must be pinned before the cast: NaN -> uint8 is
    # undefined and would write arbitrary bytes into the cache silently.
    filled = np.nan_to_num(volume, nan=low, posinf=high, neginf=low)
    scaled = (np.clip(filled, low, high) - low) / (high - low)
    return (scaled * 255.0).astype(np.uint8)


def resize(image: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def pick_slices(count: int, wanted: int) -> list[int]:
    """Evenly spaced through the stack, always including both ends.

    Centre-cropping would be wrong here: meniscal tears sit at the periphery of
    the sagittal stack, exactly where a centre crop throws data away.
    """
    if count <= 0:
        return []
    if count <= wanted:
        return list(range(count)) + [count - 1] * (wanted - count)
    return list(np.linspace(0, count - 1, wanted).round().astype(int))


def read_series_volume(directory: Path) -> tuple[np.ndarray | None, float | None, str | None]:
    """Return (volume, mm_per_pixel, laterality) with slices in anatomical order."""
    try:
        names = sorted(e.name for e in os.scandir(directory) if e.name.endswith(".dcm"))
    except FileNotFoundError:
        return None, None, None
    if not names:
        return None, None, None

    slices = []
    spacing = None
    laterality = None
    for name in names:
        try:
            ds = pydicom.dcmread(str(directory / name), force=True)
            pixels = ds.pixel_array.astype(np.float32)
        except Exception:  # noqa: BLE001 - one unreadable slice must not lose the series
            continue
        if pixels.ndim != 2:
            continue
        position = getattr(ds, "ImagePositionPatient", None)
        order = float(position[2]) if position is not None and len(position) == 3 else len(slices)
        if spacing is None:
            value = getattr(ds, "PixelSpacing", None)
            if value is not None and len(value) >= 1:
                spacing = float(value[0])
        if laterality is None:
            laterality = (getattr(ds, "Laterality", None)
                          or _laterality_from_description(getattr(ds, "SeriesDescription", "")))
        slices.append((order, pixels))

    if not slices:
        return None, None, None
    slices.sort(key=lambda item: item[0])
    return np.stack([s[1] for s in slices]), spacing, laterality


def _laterality_from_description(description: str) -> str | None:
    text = (description or "").upper()
    if text.startswith("LT") or "_LT_" in text or " LEFT" in text or text.startswith("L_"):
        return "L"
    if text.startswith("RT") or "_RT_" in text or " RIGHT" in text or text.startswith("R_"):
        return "R"
    return None


def build_study(root: Path, split: str, study: str, series_rows: pd.DataFrame) -> tuple:
    """One study to (planes, slices, size, size) uint8, plus a record of what happened."""
    record = {"StudyInstanceUID": study, "split": split, "laterality": None,
              "mirrored": False, "planes_found": 0, "missing_planes": [], "error": None}
    channels = []

    for plane in PLANES:
        candidates = series_rows[(series_rows.Anatomical_Plane == plane)
                                 & (series_rows.Fluid_Sensitive == 1)]
        if candidates.empty:
            candidates = series_rows[series_rows.Anatomical_Plane == plane]
        if candidates.empty:
            record["missing_planes"].append(plane)
            channels.append(np.zeros((SLICES_PER_PLANE, TARGET_SIZE, TARGET_SIZE), np.uint8))
            continue

        # Prefer the series with the most slices — the diagnostic acquisition
        # rather than a localiser.
        chosen = candidates.sort_values("n_slices", ascending=False).iloc[0]
        directory = root / f"{split}_series" / study / chosen.SeriesInstanceUID
        volume, spacing, laterality = read_series_volume(directory)
        if volume is None:
            record["missing_planes"].append(plane)
            channels.append(np.zeros((SLICES_PER_PLANE, TARGET_SIZE, TARGET_SIZE), np.uint8))
            continue

        record["laterality"] = record["laterality"] or laterality
        record["planes_found"] += 1

        indices = pick_slices(len(volume), SLICES_PER_PLANE)
        volume = volume[indices]

        # Physical resampling: crop or pad to the field of view we want, then
        # resize once. Doing it in this order keeps millimetres meaningful.
        if spacing and spacing > 0:
            wanted_pixels = int(round(TARGET_SIZE * TARGET_MM_PER_PIXEL / spacing))
            wanted_pixels = max(8, min(wanted_pixels, max(volume.shape[1], volume.shape[2])))
            centre_y, centre_x = volume.shape[1] // 2, volume.shape[2] // 2
            half = wanted_pixels // 2
            y0, y1 = max(0, centre_y - half), min(volume.shape[1], centre_y + half)
            x0, x1 = max(0, centre_x - half), min(volume.shape[2], centre_x + half)
            volume = volume[:, y0:y1, x0:x1]

        volume = normalise(volume)
        resized = np.stack([resize(frame, TARGET_SIZE) for frame in volume])
        channels.append(resized)

    stack = np.stack(channels)  # (planes, slices, size, size)

    if (record["laterality"] or "").upper().startswith("R"):
        stack = stack[..., ::-1].copy()
        record["mirrored"] = True

    return stack, record


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


def main() -> int:
    import torch

    started = time.time()
    root = find_marker("test.csv")
    if root is None:
        raise SystemExit("competition data not found")
    print(f"competition root: {root}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (compute {major}.x)")
        if major < 7:
            print("pre-Volta GPU; falling back to CPU to avoid a CUDA failure")
            device = "cpu"

    # Every mounted training kernel contributes one member. Discovered rather
    # than listed, so adding a member means mounting it and nothing else.
    #
    # This glob is the ONLY discovery path for weights, and the wildcard in
    # `fold*` is load-bearing. A full-fit model is written as
    # checkpoint_foldall.pt, so a kernel mounting nothing but full-fit members
    # contains no checkpoint_fold0.pt anywhere. An earlier version guarded on
    # that exact filename before reaching this line and would have refused to
    # start with five perfectly good checkpoints mounted.
    # DISCOVERED, not a hardcoded path. This used to glob a fixed
    # notebooks/<user>/<slug>/ path under /kaggle/input, which encoded Kaggle's
    # mount layout as of 2026-09-01. Between then and
    # 2026-09-05 that layout went FLAT — the competition moved from
    # /kaggle/input/competitions/<comp> to /kaggle/input/<comp>, and mounted
    # notebooks with it — so the glob silently matched nothing and two
    # inference runs died with "no checkpoints mounted" while five perfectly
    # good checkpoints sat one directory away.
    #
    # find_marker("test.csv") never broke, because it searches rather than
    # assumes. This now does the same, so a future layout change costs nothing.
    checkpoints = sorted(
        path
        for directory in find_all_markers("checkpoint_fold*.pt")
        for path in sorted(directory.glob("checkpoint_fold*.pt"))
    )
    if not checkpoints:
        raise SystemExit("no checkpoints mounted (checkpoint_fold*.pt)")
    print(f"weights: {checkpoints[0].parent}")
    print(f"checkpoints mounted: {len(checkpoints)}")

    # A member that never trained mounts as an EMPTY notebook, not as an error.
    # The glob then finds fewer checkpoints, the ensemble runs, and it produces
    # a perfectly valid submission for an experiment nobody declared — which is
    # unattributable the moment its board score arrives.
    #
    # This is not hypothetical. On 2026-09-02 the weekly GPU quota ran out with
    # two of five full-fit members unbuilt: one had died on an uncorrectable ECC
    # error and one was never pushed. Every check up to here would have passed.
    #
    # E061 verified "checkpoints mounted: 10" and "6" by reading the log after
    # the fact. Reading a log is not a guard, so the count is declared in the
    # manifest and asserted here instead.
    if MEMBERS_EXPECTED is not None and len(checkpoints) != MEMBERS_EXPECTED:
        raise SystemExit(
            f"expected {MEMBERS_EXPECTED} ensemble members, mounted "
            f"{len(checkpoints)}: {[p.parent.parent.name for p in checkpoints]}. "
            "Refusing to submit an ensemble that is not the one declared."
        )

    # Averaging is only meaningful between models that were fed the same way.
    # Two properties are invisible in the weights and fatal if they differ:
    # how many slices the model saw, and whether its input was ImageNet-
    # normalised. Both are recorded at training time and checked here against
    # the manifest, so a mismatch stops the kernel instead of quietly costing
    # AUC on the leaderboard.
    EXPECTED = {"slice_subsample": SLICE_SUBSAMPLE_EXPECTED,
                "input_norm": INPUT_NORM_EXPECTED}

    models = []
    for path in checkpoints:
        state = torch.load(path, map_location=device, weights_only=False)
        # An ABSENT key is not the same as a key recorded as None, and
        # conflating the two is how this check first went wrong: knee-train-v2
        # trained on 18 of 24 slices but predates the record, so reading absent
        # as "no subsampling" refused a checkpoint that was in fact correct —
        # and would have let a genuinely mismatched one through in the other
        # direction. Present means verified and must match exactly. Absent means
        # UNVERIFIED: it is announced, not silently assumed away.
        for key, want in EXPECTED.items():
            if key not in state:
                print(f"  UNVERIFIED: {path.name} does not record {key}. It was "
                      f"written before this kernel recorded it, so there is no "
                      f"way to check it from the weights. Proceeding on the "
                      f"manifest's declaration of {want!r}.")
                continue
            if state[key] != want:
                raise SystemExit(
                    f"{path.parent.parent.name}/{path.name} was trained with "
                    f"{key}={state[key]!r} but this kernel declares {want!r}. "
                    "The inputs differ, so averaging these models would be "
                    "invalid — refusing to predict.")

        backbone = state.get("backbone", "resnet18")
        # Architecture comes from the checkpoint, not the manifest: this one
        # changes the shape of the weights, so a mismatch is caught by the
        # strict load below rather than silently mis-assembled. Absent means a
        # checkpoint from before the option existed, which had a single map.
        pooling = bool(state.get("per_finding_pool", False))
        focal = int(state.get("focal_k", 0) or 0)
        model = build_model(backbone, 3, len(FINDINGS), INPUT_NORM_EXPECTED,
                            pooling, focal).to(device)

        # The ImageNet mean/std were briefly saved as persistent buffers, so
        # checkpoints written in that window carry two extra keys. They are
        # constants that the model reconstructs for itself, so dropping them is
        # lossless — but leaving them in would trip the strict-load check below
        # and refuse a perfectly good set of weights.
        weights = {k: v for k, v in state["model"].items() if k not in ("mean", "std")}
        try:
            missing, unexpected = model.load_state_dict(weights, strict=False)
        except RuntimeError as exc:
            # strict=False forgives absent and extra keys but NOT a shape
            # mismatch, which is what an architecture change looks like. Good —
            # but the raw traceback in a Kaggle log buries the reason.
            raise SystemExit(
                f"{path.parent.parent.name}/{path.name} does not fit the model this "
                f"kernel built (backbone={backbone}, per_finding_pool={pooling}). "
                f"The checkpoint records a different architecture, so its weights "
                f"mean something else. Original error: {exc}") from exc
        if missing or unexpected:
            raise SystemExit(f"weight mismatch in {path.name} — missing={len(missing)} "
                             f"unexpected={len(unexpected)}; refusing to predict")
        model.eval()
        models.append(model)
        print(f"  {path.parent.parent.name}/{path.name}: {backbone}, "
              f"epoch {state.get('epoch')}, val macro AUC "
              f"{state.get('macro_auc', float('nan')):.4f}")
    print(f"ensembling {len(models)} models")
    model.eval()

    test = pd.read_csv(root / "test.csv")
    series = pd.read_csv(root / "test_series.csv")
    if "n_slices" not in series.columns:
        series["n_slices"] = 1
    studies = test.StudyInstanceUID.astype(str).tolist()
    print(f"test studies: {len(studies):,}   series: {len(series):,}")

    by_study = dict(list(series.groupby("StudyInstanceUID")))
    predictions = np.full((len(studies), len(FINDINGS)), FALLBACK_PRIOR, dtype=np.float32)
    failures = 0

    # Decoding DICOMs is I/O- and CPU-bound and was the whole cost: 19.7 s/study
    # measured single-threaded, which projects to 7.1 h of a 9 h cap on ~1,300
    # studies. Far too little headroom for the run that actually counts. Build
    # volumes on a thread pool and predict in batches so decode overlaps compute.
    def build_one(index_and_study):
        index, study = index_and_study
        try:
            rows = by_study.get(study)
            if rows is None or rows.empty:
                return index, None, "no series rows"
            stack, _record = build_study(root, "test", study, rows)

            # A model trained on a subset of slices was also VALIDATED on an
            # evenly spaced subset of that size. Handing it the full stack here
            # changes the sequence length the attention pool sees and raises
            # nothing at all — it just scores worse. None means the model saw
            # every cached slice, and this is then a no-op.
            if SLICE_SUBSAMPLE_EXPECTED and stack.shape[1] > SLICE_SUBSAMPLE_EXPECTED:
                idx = np.linspace(0, stack.shape[1] - 1,
                                  SLICE_SUBSAMPLE_EXPECTED).round().astype(int)
                stack = stack[:, idx]
            return index, stack, None
        except Exception as exc:  # noqa: BLE001
            return index, None, f"{type(exc).__name__}: {exc}"

    workers = max(2, min(8, (os.cpu_count() or 4)))
    print(f"decoding on {workers} threads, predicting in batches of {BATCH_STUDIES}")

    # Time the study loop separately from kernel startup. Model construction,
    # imports and the first CUDA/BLAS call cost ~50 s and are paid ONCE, but the
    # public test set is 3 studies, so folding them into a per-study average and
    # multiplying by 1,300 overstates the real cost by an order of magnitude.
    # That mistake nearly ruled out CPU inference on a 20x-too-pessimistic number.
    setup_seconds = time.time() - started
    loop_started = time.time()
    print(f"setup took {setup_seconds:.1f}s (paid once, not per study)")

    pending: list = []

    # Each model's raw probabilities are kept separately so they can be combined
    # by RANK at the end. The metric is macro ROC-AUC, which reads order only,
    # so averaging sigmoids is the wrong operation: a member that happens to be
    # confidently wrong drags the mean, while under rank averaging every member
    # gets exactly one vote per study. Ranking has to span the whole test set —
    # ranks inside a batch of four studies mean nothing — which is why this is
    # accumulated here and resolved after the loop.
    raw = np.full((len(models), len(studies), len(FINDINGS)), np.nan, np.float32)

    timing = {"forward_seconds": 0.0}

    def flush(pending_batch):
        if not pending_batch:
            return
        indices = [p[0] for p in pending_batch]
        batch = torch.from_numpy(
            np.stack([p[1] for p in pending_batch]).astype(np.float32) / 255.0).to(device)
        forward_started = time.time()
        with torch.no_grad():
            for m_index, model in enumerate(models):
                out = torch.sigmoid(model(batch)).cpu().numpy()
                for slot, index in enumerate(indices):
                    raw[m_index, index] = out[slot]
        if device == "cuda":
            torch.cuda.synchronize()      # otherwise this times queueing, not work
        timing["forward_seconds"] += time.time() - forward_started

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, (index, stack, error) in enumerate(
                pool.map(build_one, enumerate(studies)), start=1):
            if error is not None or stack is None:
                failures += 1
                if failures <= 5:
                    print(f"  study {index} failed: {error}")
            else:
                pending.append((index, stack))
                if len(pending) >= BATCH_STUDIES:
                    flush(pending)
                    pending = []

            if done % 100 == 0 or done == len(studies):
                elapsed = time.time() - loop_started
                rate = done / max(elapsed, 1e-6)
                print(f"  {done:,}/{len(studies):,}  {elapsed / 60:.1f} min  "
                      f"{1 / rate:.2f} s/study  eta "
                      f"{(len(studies) - done) / rate / 60:.1f} min", flush=True)
        flush(pending)

    # Rank-average across members. Equal weights: the folds differ only by
    # split, so weighting them would be fitting validation noise.
    scored = ~np.isnan(raw[0, :, 0])
    if scored.any():
        ranked = np.zeros((int(scored.sum()), len(FINDINGS)), np.float64)
        for m_index in range(len(models)):
            for j in range(len(FINDINGS)):
                column = raw[m_index, scored, j].astype(np.float64)
                order = np.argsort(column, kind="mergesort")
                ranks = np.empty(len(column), np.float64)
                ranks[order] = np.arange(len(column), dtype=np.float64)
                # average tied ranks, so identical probabilities cannot be
                # separated by whichever study happened to be decoded first
                unique, inverse, counts = np.unique(column, return_inverse=True,
                                                    return_counts=True)
                if len(unique) < len(column):
                    sums = np.zeros(len(unique))
                    np.add.at(sums, inverse, ranks)
                    ranks = (sums / counts)[inverse]
                ranked[:, j] += ranks
        # back to (0, 1). Monotone, so with a single model this is a no-op for
        # AUC and the submission is unchanged.
        denominator = max(len(models) * (int(scored.sum()) - 1), 1)
        predictions[scored] = ranked / denominator
        # Studies that could not be decoded sit in the middle rather than at an
        # arbitrary prior: with a rank score there is no meaningful "prevalence",
        # and the middle is the position that costs the least either way.
        predictions[~scored] = 0.5

    submission = pd.DataFrame({"StudyInstanceUID": studies})
    for j, finding in enumerate(FINDINGS):
        submission[finding] = np.clip(predictions[:, j], 1e-6, 1 - 1e-6)

    sample = pd.read_csv(root / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns), (
        f"columns differ from sample_submission: {list(sample.columns)}")
    assert len(submission) == len(test), "row count must match test.csv"
    assert submission[FINDINGS].notna().all().all(), "no NaNs allowed"
    assert ((submission[FINDINGS] > 0) & (submission[FINDINGS] < 1)).all().all(), "probabilities"
    assert submission.StudyInstanceUID.is_unique, "duplicate study ids"
    submission.to_csv(OUTPUT, index=False)

    elapsed = time.time() - started
    loop_seconds = time.time() - loop_started
    per_study = loop_seconds / max(len(studies), 1)
    projected = (setup_seconds + per_study * 1300) / 3600
    spread = {f: round(float(submission[f].std()), 4) for f in FINDINGS}
    print(f"\nwrote {OUTPUT}  rows={len(submission):,}")
    print(f"failed studies (kept prior): {failures}")
    print(f"wall clock {elapsed / 60:.1f} min  (setup {setup_seconds:.0f}s + "
          f"loop {loop_seconds / 60:.1f} min)")
    print(f"per-study cost EXCLUDING setup: {per_study:.2f} s")
    print(f"projected on 1,300 studies: {projected:.2f} h of the 9 h cap "
          f"(setup once + {per_study:.2f} s x 1300)")

    # Where the runtime actually goes. The efficiency prize charges 0.0502 AUC
    # per extra hour (COMPETITIVE_ANALYSIS.md §4), so the split between decode
    # and forward passes decides whether another fold is nearly free or the most
    # expensive thing available — and until now it has been guessed at. Decode is
    # threaded and overlaps the GPU, so these do not sum to wall clock; the
    # forward total is the part that scales with the number of members.
    scored_count = max(int(scored.sum()), 1)
    forward_per_study = timing["forward_seconds"] / scored_count
    per_member = forward_per_study / max(len(models), 1)
    member_hours = per_member * 1300 / 3600
    print(f"forward passes: {timing['forward_seconds']:.1f}s over {len(models)} "
          f"member(s) = {forward_per_study * 1000:.0f} ms/study, "
          f"{per_member * 1000:.0f} ms/study/member")
    print(f"everything else (decode, threaded and overlapped): "
          f"{max(per_study - forward_per_study, 0) * 1000:.0f} ms/study")
    print(f"one more member costs ~{member_hours:.3f} h on 1,300 studies, so it "
          f"pays for itself above ~{member_hours * 0.0502:.4f} AUC")
    print(f"prediction spread: {spread}")
    Path("/kaggle/working/infer_manifest.json").write_text(json.dumps({
        "n_studies": len(studies), "failures": failures,
        "seconds_per_study_excluding_setup": round(per_study, 3),
        "setup_seconds": round(setup_seconds, 1),
        "wall_clock_seconds": round(elapsed, 1),
        "projected_hours_1300_studies": round(projected, 3),
        "forward_seconds_per_study": round(forward_per_study, 4),
        "forward_seconds_per_study_per_member": round(per_member, 4),
        "hours_per_extra_member_on_1300": round(member_hours, 4),
        "n_models": len(models),
        "prediction_spread": spread}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
