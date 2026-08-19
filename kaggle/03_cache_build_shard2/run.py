"""Phase 2 cache build — DICOM to normalised, laterality-corrected volumes.

Turns ~820,000 loose DICOM slices into a compact array per study that a training
kernel can read at speed. Runs on CPU, shards by study, and is the prerequisite
for every training run that follows.

The decisions, and why they are what they are
---------------------------------------------
**Series selection** (`FINDINGS.md` §10): one fluid-sensitive series per plane.
An axial fluid-sensitive series exists for **100%** of studies, so it is the
guaranteed fallback; sagittal and coronal are present for 94% and 96%. Planes
come from the host's `Anatomical_Plane`, which agreed with the DICOM headers on
24,371 of 24,371 series, so it is trusted outright.

**Laterality mirroring.** Right knees are flipped so every volume presents the
same anatomy — medial is always the same side of the image. Without this the
model has to learn each finding twice, once per side, from half the data. Source
is the `Laterality` tag (79% of series), falling back to `SeriesDescription`
prefixes, then to the study's other series.

**Physical resampling, not pixel resizing.** Pixel spacing varies from 0.156 to
0.33 mm across this dataset, so a fixed pixel resize would make the same
anatomy a different size on different scanners — handing the model scanner
identity as a feature, which is precisely the leak `FINDINGS.md` §9 measured.
Resampling to a fixed mm-per-pixel removes it.

**uint8 storage.** After percentile normalisation the extra precision is not
information; it is four times the disk and four times the read time.

Output: one `.npy` per study of shape (planes, slices, size, size), plus an
index parquet recording what was selected and what was missing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

# --------------------------------------------------------------------------- #
# GENERATED CONFIG — written by eda/generate_kernels.py from src/pipeline.py.
# Edit the manifest, not this file. Everything outside this block is shared by
# every kernel rendered from this template.
# --------------------------------------------------------------------------- #
# 0.6 mm/px over 192 px covers ~115 mm, which contains the knee joint
# with margin. Chosen against the inference budget: 3 planes x 20 slices
# at 192px is ~2.2 MB per study, so the training cache stays under 10 GB.
# This is the geometry that scored 0.725 on the leaderboard.
#
RUN_SPLIT           = "train"
RUN_SHARD           = 2
RUN_OF              = 4
RUN_LIMIT           = 0
TARGET_MM_PER_PIXEL = 0.6
TARGET_SIZE         = 192
SLICES_PER_PLANE    = 20
# --------------------------------------------------------------------------- #

PLANES = ("Sagittal", "Coronal", "Axial")
SKIP_DIRECTORIES = {"train_series", "test_series"}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=RUN_SPLIT, choices=["train", "test"])
    parser.add_argument("--shard", type=int, default=RUN_SHARD)
    parser.add_argument("--of", type=int, default=RUN_OF)
    parser.add_argument("--out", default="/kaggle/working/cache")
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=RUN_LIMIT)
    parser.add_argument("--time-budget", type=float, default=8.0 * 3600)
    args, _unknown = parser.parse_known_args()

    root = Path(args.root) if args.root else find_marker(f"{args.split}_series.csv")
    if root is None:
        raise SystemExit("competition data not found")
    print(f"competition root: {root}")

    series = pd.read_csv(root / f"{args.split}_series.csv")
    headers_dir = find_marker("series_headers.parquet")
    if headers_dir is not None:
        headers = pd.read_parquet(headers_dir / "series_headers.parquet")
        series = series.merge(headers[["SeriesInstanceUID", "n_slices"]],
                              on="SeriesInstanceUID", how="left")
        print(f"merged slice counts from {headers_dir}")
    if "n_slices" not in series.columns:
        series["n_slices"] = 1
    series["n_slices"] = series["n_slices"].fillna(1)

    studies = sorted(series.StudyInstanceUID.unique())
    if args.of > 1:
        studies = [s for i, s in enumerate(studies) if i % args.of == args.shard]
    if args.limit:
        studies = studies[: args.limit]
    print(f"shard {args.shard}/{args.of}: {len(studies):,} studies")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_study = dict(list(series.groupby("StudyInstanceUID")))
    records = []
    started = time.time()
    truncated = False
    for i, study in enumerate(studies, start=1):
        try:
            stack, record = build_study(root, args.split, study, by_study[study])
            np.save(out_dir / f"{study}.npy", stack)
        except Exception as exc:  # noqa: BLE001 - one bad study must not lose the shard
            record = {"StudyInstanceUID": study, "split": args.split, "laterality": None,
                      "mirrored": False, "planes_found": 0, "missing_planes": [],
                      "error": f"{type(exc).__name__}: {exc}"}
        record["missing_planes"] = "|".join(record["missing_planes"])
        records.append(record)

        if i % 100 == 0 or i == len(studies):
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            print(f"  {i:,}/{len(studies):,}  {elapsed / 60:.1f} min  {rate:.2f}/s  "
                  f"eta {(len(studies) - i) / rate / 60 if rate else 0:.1f} min", flush=True)
        if time.time() - started > args.time_budget:
            print("time budget reached; writing what is built")
            truncated = True
            break

    index = pd.DataFrame.from_records(records)
    index.to_parquet(out_dir / f"cache_index_{args.split}_{args.shard}.parquet", index=False)

    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.npy"))
    elapsed = time.time() - started
    manifest = {
        "split": args.split, "shard": args.shard, "of": args.of,
        "studies_built": len(records), "truncated": truncated,
        "errors": int(index.error.notna().sum()) if "error" in index else 0,
        "mirrored": int(index.mirrored.sum()) if "mirrored" in index else 0,
        "mean_planes_found": round(float(index.planes_found.mean()), 3)
        if "planes_found" in index else None,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_study": round(elapsed / max(len(records), 1), 3),
        "cache_bytes": total_bytes,
        "bytes_per_study": int(total_bytes / max(len(records), 1)),
        "shape": [len(PLANES), SLICES_PER_PLANE, TARGET_SIZE, TARGET_SIZE],
        "mm_per_pixel": TARGET_MM_PER_PIXEL,
    }
    Path(out_dir / f"cache_manifest_{args.split}_{args.shard}.json").write_text(
        json.dumps(manifest, indent=2))
    print("\n" + json.dumps(manifest, indent=2))
    if "planes_found" in index:
        print("\nplanes found per study:")
        print(index.planes_found.value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
