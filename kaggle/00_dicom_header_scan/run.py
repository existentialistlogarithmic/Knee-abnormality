"""Phase 0 step 3 — DICOM header scan (Kaggle CPU kernel, no pixels, no GPU).

Reads headers only: one row per series, never decoding pixel data. That is what
makes a full-dataset pass cheap enough to run on CPU.

Why this kernel is on the critical path
---------------------------------------
No competition CSV carries a site, scanner, or institution column, and the DICOM
headers are de-identified: InstitutionName, StationName, DeviceSerialNumber and
all dates are absent. So there is no site label to group folds by, and a random
K-fold would let the same scanner appear on both sides of the split.

What survives de-identification is a *scanner fingerprint*:

    Manufacturer + ManufacturerModelName + SoftwareVersions
    + MagneticFieldStrength + ImagingFrequency + TransmitCoilName

ImagingFrequency is the Larmor frequency in MHz, quoted to several decimals. It
is set by the magnet's exact field, so it differs between two nominally
identical 3 T scanners and is close to a hardware serial number in practice.
Hashed together with the rest, it is the best available stand-in for "site", and
it is what the fold grouping and the leakage audit will use.

The scan also records PatientID. It is pseudonymised, but if one pseudonym spans
several studies then patient-level leakage is possible on top of scanner-level
leakage, and the folds have to group on patient too. That is worth knowing
before any model is trained, and it costs nothing to check here.

Output
------
`series_headers.parquet`, one row per series. Also `scan_manifest.json` with
runtime and coverage, so the cost of the scan is recorded rather than estimated.

Sharding
--------
Kaggle sessions die. Pass --shard i --of n to split the work by study; each
shard writes its own parquet and they concatenate later. A shard that finishes
early writes its output regardless, and --time-budget stops cleanly before the
session is killed so a partial result is never lost.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import pydicom

# Kaggle mounts competition data under /kaggle/input, but not reliably under a
# directory named after the competition — see find_competition_root below.
DEFAULT_ROOT = Path("/kaggle/input/rsna-knee-abnormality-detection")

def find_competition_root(explicit: str | None = None) -> Path:
    """Locate the mounted competition data.

    Kaggle does not guarantee the mount directory matches the competition slug,
    and a wrong guess costs a whole kernel run to discover — as it did on the
    first attempt here. So the root is found by looking for the files we know
    must exist, and the search is reported in the log either way.
    """
    if explicit:
        return Path(explicit)
    base = Path("/kaggle/input")
    if not base.exists():
        return Path(".")
    candidates = sorted(p for p in base.iterdir() if p.is_dir())
    print(f"/kaggle/input contains: {[p.name for p in candidates]}")
    for marker in ("train_series.csv", "train.csv", "test.csv"):
        for candidate in candidates:
            if (candidate / marker).exists():
                print(f"using competition root: {candidate}  (found {marker})")
                return candidate
        # the data is sometimes nested one level down
        for candidate in candidates:
            for child in sorted(p for p in candidate.iterdir() if p.is_dir()):
                if (child / marker).exists():
                    print(f"using competition root: {child}  (found {marker})")
                    return child
    print("WARNING: no competition root found; falling back to the slug path")
    return base / "rsna-knee-abnormality-detection"

# (attribute name, output column). Kept explicit rather than dumping every tag:
# the point is a small, stable table, not a copy of the headers.
FIELDS = [
    # scanner fingerprint — the reason this kernel exists
    ("Manufacturer", "manufacturer"),
    ("ManufacturerModelName", "model_name"),
    ("SoftwareVersions", "software_versions"),
    ("MagneticFieldStrength", "field_strength"),
    ("ImagingFrequency", "imaging_frequency"),
    ("TransmitCoilName", "transmit_coil"),
    ("ReceiveCoilName", "receive_coil"),
    # geometry — needed for resampling and for series selection
    ("PixelSpacing", "pixel_spacing"),
    ("SliceThickness", "slice_thickness"),
    ("SpacingBetweenSlices", "spacing_between_slices"),
    ("Rows", "rows"),
    ("Columns", "columns"),
    ("ImageOrientationPatient", "image_orientation"),
    # sequence identity — what kind of picture is this
    ("SeriesDescription", "series_description"),
    ("SeriesNumber", "series_number"),
    ("ScanningSequence", "scanning_sequence"),
    ("SequenceVariant", "sequence_variant"),
    ("ScanOptions", "scan_options"),
    ("SequenceName", "sequence_name"),
    ("MRAcquisitionType", "mr_acquisition_type"),
    ("RepetitionTime", "repetition_time"),
    ("EchoTime", "echo_time"),
    ("InversionTime", "inversion_time"),
    ("EchoTrainLength", "echo_train_length"),
    ("FlipAngle", "flip_angle"),
    ("PixelBandwidth", "pixel_bandwidth"),
    ("ImageType", "image_type"),
    # subject and side
    ("PatientID", "patient_id"),
    ("PatientSex", "patient_sex"),
    ("Laterality", "laterality"),
    ("BodyPartExamined", "body_part"),
    ("PatientPosition", "patient_position"),
    ("Modality", "modality"),
]


def scalarise(value):
    """DICOM values are multi-valued, byte strings, or DSFloat. Make them flat."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(v) for v in value)
    return str(value)


def plane_from_orientation(orientation: str | None) -> str | None:
    """Axial / Coronal / Sagittal from ImageOrientationPatient.

    The host already publishes Anatomical_Plane per series, so this is a
    cross-check rather than the primary source — a disagreement would mean one
    of the two is wrong, which is worth catching before it reaches the model.
    """
    if not orientation:
        return None
    try:
        values = [float(x) for x in orientation.split("|")]
    except ValueError:
        return None
    if len(values) != 6:
        return None
    row, col = values[:3], values[3:]
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    axis = max(range(3), key=lambda i: abs(normal[i]))
    return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[axis]


def read_series(series_dir: Path, study_uid: str, series_uid: str, split: str) -> dict:
    """One row per series, from the first readable header plus a file count."""
    try:
        files = sorted(
            (p for p in os.scandir(series_dir) if p.name.endswith(".dcm")),
            key=lambda entry: entry.name,
        )
    except FileNotFoundError:
        return {
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid,
            "split": split,
            "n_slices": 0,
            "error": "series directory missing",
        }

    record = {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "split": split,
        "n_slices": len(files),
        "error": None,
    }
    if not files:
        record["error"] = "no dicom files"
        return record

    dataset = None
    for entry in files[: min(3, len(files))]:
        try:
            dataset = pydicom.dcmread(entry.path, stop_before_pixels=True, force=True)
            break
        except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop the scan
            record["error"] = f"{type(exc).__name__}: {exc}"
    if dataset is None:
        return record

    record["error"] = None
    for attribute, column in FIELDS:
        record[column] = scalarise(getattr(dataset, attribute, None))
    record["plane_from_headers"] = plane_from_orientation(record.get("image_orientation"))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None,
                        help="competition root; auto-discovered when omitted")
    parser.add_argument("--out", default="series_headers.parquet")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--time-budget", type=float, default=8.0 * 3600,
                        help="stop cleanly and write output before the session dies")
    parser.add_argument("--limit", type=int, default=0, help="debug: stop after N series")
    args = parser.parse_args()

    root = find_competition_root(args.root)
    started = time.time()

    frames = []
    for split, csv_name in (("train", "train_series.csv"), ("test", "test_series.csv")):
        path = root / csv_name
        if path.exists():
            frame = pd.read_csv(path)
            frame["split"] = split
            frames.append(frame)
        else:
            print(f"WARNING: {path} not found")
    if not frames:
        print("no series tables found; nothing to scan")
        return 2
    series = pd.concat(frames, ignore_index=True)

    # Shard by study so a study's series never split across shards.
    if args.of > 1:
        studies = sorted(series.StudyInstanceUID.unique())
        mine = {s for i, s in enumerate(studies) if i % args.of == args.shard}
        series = series[series.StudyInstanceUID.isin(mine)]
        print(f"shard {args.shard}/{args.of}: {len(mine):,} studies, {len(series):,} series")

    if args.limit:
        series = series.head(args.limit)

    records = []
    truncated = False
    total = len(series)
    for i, row in enumerate(series.itertuples(index=False), start=1):
        directory = root / f"{row.split}_series" / row.StudyInstanceUID / row.SeriesInstanceUID
        records.append(
            read_series(directory, row.StudyInstanceUID, row.SeriesInstanceUID, row.split)
        )
        if i % 500 == 0 or i == total:
            elapsed = time.time() - started
            rate = i / elapsed if elapsed else 0
            print(f"  {i:,}/{total:,} series  {elapsed / 60:.1f} min  "
                  f"{rate:.1f}/s  eta {(total - i) / rate / 60 if rate else 0:.1f} min",
                  flush=True)
        if time.time() - started > args.time_budget:
            print(f"time budget reached after {i:,} series; writing partial output")
            truncated = True
            break

    out = pd.DataFrame.from_records(records)
    out.to_parquet(args.out, index=False)

    elapsed = time.time() - started
    manifest = {
        "series_scanned": len(out),
        "series_expected": total,
        "truncated": truncated,
        "shard": args.shard,
        "of": args.of,
        "elapsed_seconds": round(elapsed, 1),
        "seconds_per_series": round(elapsed / max(len(out), 1), 4),
        "output_bytes": Path(args.out).stat().st_size,
        "errors": int(out["error"].notna().sum()) if "error" in out else None,
        "total_slices": int(out["n_slices"].sum()) if "n_slices" in out else None,
    }
    Path("scan_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\n" + json.dumps(manifest, indent=2))

    # A first look at the thing this kernel exists to produce.
    fingerprint_cols = [
        c for c in ("manufacturer", "model_name", "software_versions",
                    "field_strength", "imaging_frequency", "transmit_coil")
        if c in out.columns
    ]
    if fingerprint_cols:
        fingerprint = out[fingerprint_cols].fillna("?").agg("|".join, axis=1)
        print(f"\ndistinct scanner fingerprints: {fingerprint.nunique():,}")
        print(f"distinct manufacturers:        {out['manufacturer'].nunique()}")
        print(f"distinct model names:          {out['model_name'].nunique()}")
        studies_per_fp = (
            pd.DataFrame({"fp": fingerprint, "study": out.StudyInstanceUID})
            .groupby("fp").study.nunique().sort_values(ascending=False)
        )
        print("\nstudies per fingerprint (top 15):")
        print(studies_per_fp.head(15).to_string())
        print(f"\nfingerprints with a single study: "
              f"{int((studies_per_fp == 1).sum()):,} of {len(studies_per_fp):,}")
    if "patient_id" in out.columns:
        per_patient = out.groupby("patient_id").StudyInstanceUID.nunique()
        repeat = int((per_patient > 1).sum())
        print(f"\npatients appearing in more than one study: {repeat:,} "
              f"of {len(per_patient):,}")
        if repeat:
            print("  -> folds must group on patient as well as scanner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
