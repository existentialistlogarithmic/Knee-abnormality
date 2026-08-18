"""Phase 0 step 5 — quantify what a random K-fold would have cost us.

The argument in one sentence: a model given **only scanner metadata** — make,
model, field strength, slice counts, pixel spacing — cannot possibly know
whether a knee has a torn ACL. Any AUC it earns above 0.5 is memorisation of
which scanner tends to produce which labels, i.e. site signal.

So the experiment is: train exactly that model, score it under random K-fold and
under scanner-grouped K-fold, and report the gap. Under grouping the model
should collapse toward 0.5, because a scanner it has never seen tells it
nothing. Whatever it retains under random splitting is the inflation any real
model would also enjoy, silently.

Targets are the Phase 1 report-derived labels, because they cover all 4,407
studies; the 58 gold studies are far too few to measure a fold effect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.folds import add_scanner_fingerprint, study_groups  # noqa: E402

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]


def build_features(headers):
    """Study-level metadata only. Nothing here describes the anatomy."""
    import pandas as pd

    marked = add_scanner_fingerprint(headers)
    numeric = marked.copy()
    for column in ("slice_thickness", "n_slices", "rows", "columns"):
        numeric[column] = pd.to_numeric(numeric.get(column), errors="coerce")
    numeric["first_pixel_spacing"] = pd.to_numeric(
        numeric.pixel_spacing.astype(str).str.split("|").str[0], errors="coerce")

    grouped = numeric.groupby("StudyInstanceUID")
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
        mode = grouped[column].agg(lambda s: s.value_counts().index[0]
                                   if s.notna().any() else "?")
        features[column] = mode.astype("category").cat.codes
    return features.fillna(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headers",
                        default=str(REPO_ROOT / "artifacts/00_header_scan/series_headers.parquet"))
    parser.add_argument("--soft-labels",
                        default=str(REPO_ROOT / "artifacts/phase1/soft_labels.parquet"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "phase0"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, KFold

    headers = pd.read_parquet(args.headers)
    soft = pd.read_parquet(args.soft_labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = build_features(headers)
    groups = study_groups(headers)

    data = features.join(groups, how="inner").join(
        soft.set_index("StudyInstanceUID")[FINDINGS], how="inner")
    data = data.dropna(subset=["scanner_fingerprint"])
    print(f"studies: {len(data):,}")
    print(f"distinct scanner fingerprints: {data.scanner_fingerprint.nunique():,}")
    sizes = data.scanner_fingerprint.value_counts()
    print(f"group sizes: median {int(sizes.median())}, mean {sizes.mean():.1f}, "
          f"max {sizes.max()}, singletons {int((sizes == 1).sum())}")
    print(f"largest 5 groups hold {sizes.head(5).sum()} studies "
          f"({sizes.head(5).sum() / len(data):.1%})")

    feature_columns = list(features.columns)
    X = data[feature_columns].to_numpy()
    g = data.scanner_fingerprint.to_numpy()

    rows = []
    for finding in FINDINGS:
        y_soft = data[finding].astype(float)
        y = (y_soft.fillna(0.15) > 0.5).astype(int).to_numpy()
        if y.sum() < 30 or (1 - y).sum() < 30:
            continue

        scores = {}
        for name, splitter in (("random", KFold(args.folds, shuffle=True, random_state=0)),
                               ("grouped", GroupKFold(n_splits=args.folds))):
            oof = np.zeros(len(y))
            split = (splitter.split(X, y, g) if name == "grouped" else splitter.split(X, y))
            for train_idx, val_idx in split:
                model = HistGradientBoostingClassifier(
                    max_iter=120, max_depth=4, learning_rate=0.1, random_state=0)
                model.fit(X[train_idx], y[train_idx])
                oof[val_idx] = model.predict_proba(X[val_idx])[:, 1]
            scores[name] = roc_auc_score(y, oof)

        rows.append({"finding": finding,
                     "random_kfold_auc": round(scores["random"], 4),
                     "grouped_kfold_auc": round(scores["grouped"], 4),
                     "inflation": round(scores["random"] - scores["grouped"], 4),
                     "positives": int(y.sum())})

    report = pd.DataFrame(rows).sort_values("inflation", ascending=False)
    print("\nA model given ONLY scanner metadata — no pixels, no text.")
    print("Anything above 0.5 is site memorisation, by construction.\n")
    print(f"{'finding':<18}{'random':>9}{'grouped':>9}{'inflation':>11}{'pos':>7}")
    print("-" * 54)
    for r in report.itertuples(index=False):
        print(f"{r.finding:<18}{r.random_kfold_auc:>9.3f}{r.grouped_kfold_auc:>9.3f}"
              f"{r.inflation:>11.3f}{r.positives:>7}")
    macro_random = float(report.random_kfold_auc.mean())
    macro_grouped = float(report.grouped_kfold_auc.mean())
    print("-" * 54)
    print(f"{'MACRO':<18}{macro_random:>9.3f}{macro_grouped:>9.3f}"
          f"{macro_random - macro_grouped:>11.3f}")

    print(f"\n>>> Random K-fold inflates macro AUC by {macro_random - macro_grouped:.3f}")
    print(">>> on metadata alone. A real model sees this on top of its own signal.")

    payload = {
        "n_studies": int(len(data)),
        "n_groups": int(data.scanner_fingerprint.nunique()),
        "group_size_median": int(sizes.median()),
        "group_size_max": int(sizes.max()),
        "singleton_groups": int((sizes == 1).sum()),
        "macro_random_kfold_auc": round(macro_random, 4),
        "macro_grouped_kfold_auc": round(macro_grouped, 4),
        "macro_inflation": round(macro_random - macro_grouped, 4),
        "per_finding": rows,
        "features_used": feature_columns,
        "note": "Targets are Phase 1 report-derived labels (all studies); the 58 "
                "gold studies are far too few to measure a fold effect.",
    }
    (out_dir / "leakage_audit.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'leakage_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
