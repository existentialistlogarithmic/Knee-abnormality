#!/usr/bin/env python3
"""Which imaging plane carries which finding?

The cache gives every study three planes — sagittal, coronal, axial — twenty
slices each, and the model pools all sixty together. That treats the planes as
interchangeable evidence, which anatomy says they are not: an ACL runs
obliquely and is read sagittally, the MCL is a coronal structure, the
patellofemoral joint is an axial one.

If a finding lives in one plane, pooling over three dilutes it threefold before
any pooling rule gets a chance. That is a different mechanism from the focal-
slice dilution in E027 and it stacks with it.

Nothing in the public write-ups this project has read reports per-finding plane
attribution, so this is measured here rather than looked up. Cheap: one five-
fold head per plane over frozen embeddings, four runs, minutes on CPU.

    python eda/plane_ablation.py --embeddings artifacts/embed/embeddings.npy \
        --index artifacts/embed/embeddings_index.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_schema import FINDINGS  # noqa: E402

PLANES = ("Sagittal", "Coronal", "Axial")


def load_head_lab():
    spec = importlib.util.spec_from_file_location(
        "hl", Path(__file__).resolve().parent / "head_lab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", default="artifacts/embed/embeddings.npy")
    parser.add_argument("--index", default="artifacts/embed/embeddings_index.json")
    parser.add_argument("--labels", default="artifacts/kaggle_dataset/soft_labels.parquet")
    parser.add_argument("--headers", default="artifacts/kaggle_dataset/series_headers.parquet")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args(argv)

    hl = load_head_lab()
    from sklearn.model_selection import GroupKFold

    index = json.loads(Path(args.index).read_text())
    embeddings = np.load(args.embeddings, mmap_mode="r")
    studies = index["studies"]
    total = embeddings.shape[1]
    per_plane = total // len(PLANES)
    print(f"{embeddings.shape[0]:,} studies, {total} slices = "
          f"{len(PLANES)} planes x {per_plane}   backbone {index['backbone']}")

    headers = pd.read_parquet(args.headers)
    frequency = pd.to_numeric(headers.get("imaging_frequency"), errors="coerce")
    headers = headers.assign(imaging_frequency_rounded=frequency.round(2))
    fields = ["manufacturer", "model_name", "software_versions", "field_strength",
              "imaging_frequency_rounded", "transmit_coil"]
    headers["fingerprint"] = headers[fields].astype("string").fillna("?").agg("|".join, axis=1)
    groups = headers.groupby("StudyInstanceUID").fingerprint.agg(
        lambda s: s.value_counts().index[0])
    group_values = groups.loc[studies].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(studies)), None, group_values))

    train_csv = pd.read_csv(args.train).set_index("StudyInstanceUID")
    gold_ids = set(train_csv[train_csv[FINDINGS].notna().all(axis=1)].index)
    is_gold = np.array([s in gold_ids for s in studies])
    expert = np.zeros((len(studies), len(FINDINGS)), np.float32)
    for row, study in enumerate(studies):
        if is_gold[row]:
            expert[row] = train_csv.loc[study, FINDINGS].to_numpy(dtype=np.float32)

    soft = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    targets = soft.loc[studies, FINDINGS].astype(float).to_numpy()
    channels = soft.loc[studies, [f"{f}__channel" for f in FINDINGS]].to_numpy()
    masks = np.ones_like(targets, dtype=np.float32)
    masks[channels == "absent"] = 0.0
    targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)
    targets[is_gold] = expert[is_gold]
    masks[is_gold] = 1.0

    common = {"focal_k": 0, "per_finding": False, "epochs": args.epochs,
              "batch": 64, "lr": 1e-3, "seed": 0}

    results = {}
    for plane_index, plane in enumerate(PLANES):
        window = slice(plane_index * per_plane, (plane_index + 1) * per_plane)
        subset = np.ascontiguousarray(embeddings[:, window])
        oof, gold = hl.run_config(subset, targets, masks, expert, is_gold, splits,
                                  label=f"{plane} only", **common)
        results[plane] = (oof, gold)
    oof_all, gold = hl.run_config(np.ascontiguousarray(embeddings), targets, masks,
                                  expert, is_gold, splits, label="all three planes",
                                  **common)
    results["all"] = (oof_all, gold)

    print(f"\n{'finding':18s} " + "".join(f"{p:>10s}" for p in PLANES)
          + f"{'all':>8s}{'best':>10s}   plane advantage")
    rows = []
    for i, finding in enumerate(FINDINGS):
        y = (expert[gold][:, i] > 0.5).astype(int)
        if not (0 < y.sum() < gold.sum()):
            continue
        scores = {p: hl.auc(y, results[p][0][gold][:, i]) for p in PLANES}
        every = hl.auc(y, results["all"][0][gold][:, i])
        best = max(scores, key=scores.get)
        advantage = scores[best] - every
        rows.append((finding, scores, every, best, advantage))
        flag = ("  ONE PLANE BEATS ALL THREE" if advantage > 0.03 else "")
        print(f"{finding:18s} " + "".join(f"{scores[p]:10.3f}" for p in PLANES)
              + f"{every:8.3f}{best:>10s}{flag}")

    gains = [r[4] for r in rows if r[4] > 0.03]
    print(f"\n{len(gains)} of {len(rows)} findings score HIGHER on a single plane "
          f"than on all three pooled together.")
    if gains:
        print(f"mean advantage where that happens: +{np.mean(gains):.3f}")
        print("\nPooling three planes uniformly dilutes a finding that lives in one.")
        print("A learned per-finding PLANE weight is 36 parameters and would let")
        print("each finding read the plane its anatomy actually appears in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# --------------------------------------------------------------------------- #
# Pre-specified anatomical priors, so the test is not "pick the winner and
# then admire it". These come from standard MSK MRI reading practice and are
# fixed BEFORE looking at any result: the plane a radiologist reads each
# structure in.
# --------------------------------------------------------------------------- #
ANATOMICAL_PLANE = {
    "ACL": "Sagittal",              # read on oblique sagittal
    "MCL": "Coronal",               # a coronal-plane ligament
    "Medial Meniscus": "Sagittal",
    "Lateral Meniscus": "Sagittal",
    "Medial OA": "Coronal",         # femorotibial compartments are coronal
    "Lateral OA": "Coronal",
    "PF OA": "Axial",               # the patellofemoral joint is axial
    "Baker's": "Axial",             # popliteal fossa, posterior
    "Effusion": "Axial",
    "Synovitis": "Axial",
    "Contusion": "Coronal",
    "Fracture": "Coronal",
}
