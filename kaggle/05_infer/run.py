"""Phase 3 inference — images only, internet off, writes submission.csv.

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

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker\'s",
            "Contusion", "Fracture"]
PLANES = ("Sagittal", "Coronal", "Axial")
SKIP_DIRECTORIES = {"train_series", "test_series"}
OUTPUT = Path("/kaggle/working/submission.csv")

# These must match the cache the weights were trained on, exactly.
TARGET_MM_PER_PIXEL = 0.6
TARGET_SIZE = 192
SLICES_PER_PLANE = 20
FALLBACK_PRIOR = 0.3     # used only if a study cannot be read at all
BATCH_STUDIES = 4        # studies pushed through the network at once


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


def resize(image: np.ndarray, size: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))

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
        # Expected in any internet-off kernel. Harmless at inference: every
        # parameter is overwritten by the checkpoint moments later, and the
        # strict-load check would reject a partial load. Only a problem if a
        # TRAINING kernel prints it, which would mean no pretrained init.
        print("no pretrained download (internet off) — random init; "
              "at inference the checkpoint replaces all of it")
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
    import torch

    started = time.time()
    root = find_marker("test.csv")
    weights_dir = find_marker("checkpoint_fold0.pt")
    if root is None:
        raise SystemExit("competition data not found")
    if weights_dir is None:
        raise SystemExit("trained weights not mounted (checkpoint_fold0.pt)")
    print(f"competition root: {root}\nweights: {weights_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (compute {major}.x)")
        if major < 7:
            print("pre-Volta GPU; falling back to CPU to avoid a CUDA failure")
            device = "cpu"

    state = torch.load(weights_dir / "checkpoint_fold0.pt", map_location=device,
                       weights_only=False)
    # Read the backbone from the checkpoint rather than hardcoding it. When
    # training switched resnet18 -> resnet34 a hardcoded name here would have
    # thrown a size-mismatch at best, and silently loaded nothing at worst.
    backbone = state.get("backbone", "resnet18")
    print(f"backbone from checkpoint: {backbone}")
    model = build_model(backbone, 3, len(FINDINGS)).to(device)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing or unexpected:
        raise SystemExit(f"weight mismatch — missing={len(missing)} "
                         f"unexpected={len(unexpected)}; refusing to predict")
    model.eval()
    print(f"loaded weights from epoch {state.get('epoch')} "
          f"(val macro AUC {state.get('macro_auc', float('nan')):.4f})")

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

    def flush(pending_batch):
        if not pending_batch:
            return
        indices = [p[0] for p in pending_batch]
        batch = torch.from_numpy(
            np.stack([p[1] for p in pending_batch]).astype(np.float32) / 255.0).to(device)
        with torch.no_grad():
            out = torch.sigmoid(model(batch)).cpu().numpy()
        for slot, index in enumerate(indices):
            predictions[index] = out[slot]

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
    print(f"prediction spread: {spread}")
    Path("/kaggle/working/infer_manifest.json").write_text(json.dumps({
        "n_studies": len(studies), "failures": failures,
        "seconds_per_study_excluding_setup": round(per_study, 3),
        "setup_seconds": round(setup_seconds, 1),
        "wall_clock_seconds": round(elapsed, 1),
        "projected_hours_1300_studies": round(projected, 3),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_val_macro_auc": state.get("macro_auc"),
        "prediction_spread": spread}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
