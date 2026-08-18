"""Metadata-only submission — instrumentation, not a contender.

What this measures, and why it is worth a submission slot
--------------------------------------------------------
Scanner metadata cannot see a torn ligament. A model built on it alone reaches
0.664 macro AUC under scanner-grouped CV against **report-derived** labels
(`docs/FINDINGS.md` §9). The leaderboard scores against **expert** labels.

So the distance between this kernel's grouped CV and its leaderboard score is a
direct measurement of the report-versus-expert label gap — on ~1,300 test
studies instead of the 58 gold ones we have locally. That gap is the ceiling on
the entire weak-supervision strategy, and one submission buys a 22x better
estimate of it than anything available offline.

Two secondary purposes: it exercises the Kaggle-to-Kaggle mount path that Phase
2 will need for model weights, and it establishes whether grouped CV tracks the
leaderboard at all before any GPU time is spent trusting it.

It is not expected to be competitive. The top of the leaderboard is 0.951.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom

OUTPUT = Path("/kaggle/working/submission.csv")

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]

# Larmor frequency drifts between sessions on one magnet, so the raw value is
# nearly unique per study and would make grouped folds meaningless. See
# src/folds.py — this is the single most important line in the file.
FREQUENCY_DECIMALS = 2

FINGERPRINT_FIELDS = ["manufacturer", "model_name", "software_versions",
                      "field_strength", "imaging_frequency_rounded", "transmit_coil"]

HEADER_FIELDS = [
    ("Manufacturer", "manufacturer"), ("ManufacturerModelName", "model_name"),
    ("SoftwareVersions", "software_versions"), ("MagneticFieldStrength", "field_strength"),
    ("ImagingFrequency", "imaging_frequency"), ("TransmitCoilName", "transmit_coil"),
    ("PixelSpacing", "pixel_spacing"), ("SliceThickness", "slice_thickness"),
    ("Rows", "rows"), ("Columns", "columns"), ("PatientSex", "patient_sex"),
    ("Laterality", "laterality"),
]


# Mount layout is not guaranteed: the competition lands under
# /kaggle/input/competitions/<slug> and datasets under
# /kaggle/input/datasets/<owner>/<name>. Both were discovered the expensive way,
# by a failed kernel run, so the search is depth-bounded and explicit rather
# than assuming either shape.
SKIP_DIRECTORIES = {"train_series", "test_series"}  # never walk ~1M image files


def find_marker(marker: str, max_depth: int = 4) -> Path | None:
    """Shallow breadth-first search for a directory containing `marker`."""
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


def find_competition_root() -> Path:
    for marker in ("test.csv", "train.csv"):
        found = find_marker(marker)
        if found is not None:
            print(f"competition root: {found}  (found {marker})")
            return found
    raise SystemExit("competition data not found under /kaggle/input")


def find_artifacts() -> Path:
    found = find_marker("series_headers.parquet")
    if found is None:
        raise SystemExit("knee-phase1-artifacts dataset not mounted")
    print(f"artifacts: {found}")
    return found


def scalarise(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "|".join(str(v) for v in value)
    return str(value)


def scan_test_headers(root: Path, series: pd.DataFrame) -> pd.DataFrame:
    """Same fields as the offline scan, so train and test features match exactly."""
    records = []
    started = time.time()
    for i, row in enumerate(series.itertuples(index=False), start=1):
        directory = root / "test_series" / row.StudyInstanceUID / row.SeriesInstanceUID
        record = {"StudyInstanceUID": row.StudyInstanceUID,
                  "SeriesInstanceUID": row.SeriesInstanceUID, "n_slices": 0}
        try:
            names = sorted(e.name for e in os.scandir(directory) if e.name.endswith(".dcm"))
        except FileNotFoundError:
            names = []
        record["n_slices"] = len(names)
        if names:
            try:
                ds = pydicom.dcmread(str(directory / names[0]),
                                     stop_before_pixels=True, force=True)
                for attribute, column in HEADER_FIELDS:
                    record[column] = scalarise(getattr(ds, attribute, None))
            except Exception:  # noqa: BLE001
                pass
        records.append(record)
        if i % 2000 == 0:
            print(f"  scanned {i:,}/{len(series):,} test series "
                  f"({time.time() - started:.0f}s)", flush=True)
    print(f"  test header scan: {len(records):,} series in {time.time() - started:.1f}s")
    return pd.DataFrame.from_records(records)


def build_features(headers: pd.DataFrame) -> pd.DataFrame:
    """Study-level metadata features. Identical code path for train and test."""
    frame = headers.copy()
    for column, _ in [(c, None) for c in ("slice_thickness", "n_slices", "rows", "columns")]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["first_pixel_spacing"] = pd.to_numeric(
        frame.get("pixel_spacing").astype(str).str.split("|").str[0], errors="coerce")

    grouped = frame.groupby("StudyInstanceUID")
    features = pd.DataFrame({
        "n_series": grouped.size(),
        "total_slices": grouped.n_slices.sum(),
        "mean_slices": grouped.n_slices.mean(),
        "max_slices": grouped.n_slices.max(),
        "mean_thickness": grouped.slice_thickness.mean(),
        "mean_spacing": grouped.first_pixel_spacing.mean(),
        "mean_rows": grouped.rows.mean(),
        "mean_cols": grouped.columns.mean(),
    })
    for column in ("manufacturer", "model_name", "field_strength", "transmit_coil",
                   "patient_sex", "laterality"):
        features[column] = grouped[column].agg(
            lambda s: s.value_counts().index[0] if s.notna().any() else "?")
    return features


def align_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    """Encode categories on the union, so an unseen test scanner does not shift codes.

    This is the classic train/test feature-drift bug: fitting codes separately
    silently remaps every category and the model reads garbage at inference.
    """
    categorical = ["manufacturer", "model_name", "field_strength", "transmit_coil",
                   "patient_sex", "laterality"]
    for column in categorical:
        levels = pd.Index(sorted(set(train[column].astype(str)) | set(test[column].astype(str))))
        for frame in (train, test):
            frame[column] = pd.Categorical(frame[column].astype(str), categories=levels).codes
    return train.fillna(-1), test.fillna(-1)


def main() -> int:
    started = time.time()
    root = find_competition_root()
    artifacts = find_artifacts()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    headers = pd.read_parquet(artifacts / "series_headers.parquet")
    soft = pd.read_parquet(artifacts / "soft_labels.parquet")
    headers = headers[headers.split == "train"] if "split" in headers.columns else headers

    train_features = build_features(headers)
    frequency = pd.to_numeric(headers.get("imaging_frequency"), errors="coerce")
    headers = headers.assign(imaging_frequency_rounded=frequency.round(FREQUENCY_DECIMALS))
    headers["scanner_fingerprint"] = (
        headers[FINGERPRINT_FIELDS].astype("string").fillna("?").agg("|".join, axis=1))
    groups = headers.groupby("StudyInstanceUID").scanner_fingerprint.agg(
        lambda s: s.value_counts().index[0])

    test = pd.read_csv(root / "test.csv")
    test_series = pd.read_csv(root / "test_series.csv")
    print(f"\ntest studies: {len(test):,}   test series: {len(test_series):,}")
    test_headers = scan_test_headers(root, test_series)
    test_features = build_features(test_headers)
    test_features = test_features.reindex(test.StudyInstanceUID.astype(str))

    train_features, test_features = align_categoricals(train_features, test_features)
    columns = list(train_features.columns)

    labels = soft.set_index("StudyInstanceUID")
    common = train_features.index.intersection(labels.index).intersection(groups.index)
    X = train_features.loc[common, columns].to_numpy()
    g = groups.loc[common].to_numpy()
    print(f"training studies: {len(common):,}   scanner groups: {pd.Series(g).nunique():,}")

    predictions = {}
    oof_scores = {}
    for finding in FINDINGS:
        y = (labels.loc[common, finding].astype(float).fillna(0.15) > 0.5).astype(int).to_numpy()
        if y.sum() < 30 or (1 - y).sum() < 30:
            predictions[finding] = np.full(len(test_features), float(y.mean()))
            continue
        oof = np.zeros(len(y))
        for train_idx, val_idx in GroupKFold(n_splits=5).split(X, y, g):
            fold = HistGradientBoostingClassifier(max_iter=120, max_depth=4,
                                                  learning_rate=0.1, random_state=0)
            fold.fit(X[train_idx], y[train_idx])
            oof[val_idx] = fold.predict_proba(X[val_idx])[:, 1]
        oof_scores[finding] = roc_auc_score(y, oof)

        final = HistGradientBoostingClassifier(max_iter=120, max_depth=4,
                                               learning_rate=0.1, random_state=0)
        final.fit(X, y)
        predictions[finding] = final.predict_proba(test_features[columns].to_numpy())[:, 1]

    print("\ngrouped OOF AUC vs report-derived labels (NOT expert labels):")
    for finding, score in sorted(oof_scores.items(), key=lambda kv: -kv[1]):
        print(f"  {finding:<18}{score:.4f}")
    macro = float(np.mean(list(oof_scores.values())))
    print(f"  {'MACRO':<18}{macro:.4f}")
    print("\n>>> The leaderboard score for this same model, against EXPERT labels,")
    print(">>> minus this number, is the report-versus-expert label gap.")

    submission = pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID.astype(str)})
    for finding in FINDINGS:
        submission[finding] = np.clip(np.asarray(predictions[finding], dtype=float), 1e-6, 1 - 1e-6)

    sample = pd.read_csv(root / "sample_submission.csv")
    assert list(submission.columns) == list(sample.columns), "column mismatch"
    assert len(submission) == len(test), "row count"
    assert submission[FINDINGS].notna().all().all(), "NaNs"
    assert submission.StudyInstanceUID.is_unique, "duplicate ids"
    submission.to_csv(OUTPUT, index=False)

    spread = {f: round(float(np.std(submission[f])), 4) for f in FINDINGS}
    print(f"\nwrote {OUTPUT}  rows={len(submission):,}")
    print(f"prediction spread (std) per finding: {spread}")
    print("A spread near zero means the model collapsed to predicting priors.")
    print(f"wall clock: {time.time() - started:.1f}s")

    Path("/kaggle/working/run_manifest.json").write_text(json.dumps(
        {"grouped_oof_macro_auc": round(macro, 4),
         "grouped_oof_per_finding": {k: round(v, 4) for k, v in oof_scores.items()},
         "prediction_spread": spread, "n_test": len(submission),
         "wall_clock_seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
