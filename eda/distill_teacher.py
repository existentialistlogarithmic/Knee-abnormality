#!/usr/bin/env python3
"""Is the model a better teacher than the labels that trained it? Decide on gold.

    python eda/distill_teacher.py --oof artifacts/oof/oof_all_fold*.json \
        --labels artifacts/kaggle_dataset_public/soft_labels.parquet

**The precondition, measured rather than assumed.** The five `v1public` folds
score **0.8980** on the 58 expert studies; the public report labels that trained
them score **0.8927**. The student has overtaken its teacher — and by 0.005,
which is the part that matters. E048 established the rule this turns on:

    a union pays when its members are COMPARABLE, and imports errors when
    they are not.

E023's union of two readers at 0.7446 and 0.7421 paid **+0.070**. The four
unions since (E033, E039, E046, E048) each added a member 0.03-0.06 behind the
incumbent and paid +0.0046, +0.0022, +0.0036, +0.0027 — none separated. The
model's own out-of-fold predictions are the first candidate member that is not
behind. That is the whole reason this is worth an experiment and it is also the
reason it might still be nothing.

**What this script does not do.** It does not pick a weight and hand it on. A
mixing weight chosen because it maximised a number on 58 studies is a free
parameter fitted to 58 studies, which `dataset-metadata.fused.json` rejects by
name and E048 declined once already. So the curve is printed in full, the
parameter-free 50/50 rank union is the arm that gets tested, and any interior
optimum is recorded as a fact about the curve rather than adopted.

Reads no report text. Prints aggregates and writes a teacher parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_schema import FINDINGS  # noqa: E402

BOOTSTRAP = 2000


def auc(y, score):
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    s = score[order]
    start = 0
    for end in range(1, len(score) + 1):
        if end == len(score) or s[end] != s[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    pos = y > 0
    n = int(pos.sum())
    if n == 0 or n == len(y):
        return float("nan")
    return (ranks[pos].sum() - n * (n + 1) / 2) / (n * (len(y) - n))


def macro(expert, score):
    values = [auc((expert[:, i] > 0.5).astype(int), score[:, i])
              for i in range(len(FINDINGS))]
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")


def paired(name_a, a, name_b, b, expert, seed=0):
    """The delta with the interval it was measured with, or it is not a result."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(BOOTSTRAP):
        rows = rng.integers(0, len(expert), len(expert))
        va, vb = macro(expert[rows], a[rows]), macro(expert[rows], b[rows])
        if va == va and vb == vb:
            deltas.append(va - vb)
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = macro(expert, a) - macro(expert, b)
    verdict = ("A is better" if low > 0 else "B is better" if high < 0
               else "NOT SEPARATED — the interval contains zero")
    print(f"\n  {name_a} − {name_b}: {point:+.4f}  95% CI [{low:+.3f}, {high:+.3f}]")
    print(f"    {verdict}")
    return point, low, high


def ranked(values: np.ndarray) -> np.ndarray:
    """Column-wise ranks in [0, 1]. The metric reads order, so a rank union is
    the parameter-free way to combine two sources on different scales."""
    out = np.empty_like(values, dtype=np.float64)
    for i in range(values.shape[1]):
        column = values[:, i]
        order = np.argsort(column, kind="mergesort")
        r = np.empty(len(column), float)
        r[order] = np.arange(len(column))
        out[:, i] = r / max(len(column) - 1, 1)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", nargs="+", required=True,
                        help="oof_all_fold*.json from knee-oof-v1pub")
    parser.add_argument("--labels",
                        default="artifacts/kaggle_dataset_public/soft_labels.parquet")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--out", default="artifacts/distilled")
    args = parser.parse_args(argv)

    # Files are grouped by their PARENT DIRECTORY, one directory per lineage.
    #
    # Within a lineage a study must appear exactly once — twice means the folds
    # overlap, the predictions are not out-of-fold, and a teacher built on them
    # would leak. Across lineages a study SHOULD appear once per lineage, and
    # those are averaged: a second independent prediction cuts the single-model
    # variance the model arm otherwise carries in full.
    #
    # Grouping on the directory rather than the file is deliberate. Both OOF
    # kernels write `oof_all_fold{n}_{tag}.json` with the same tag, so two
    # lineages downloaded into one directory would silently collide and the
    # second would look like a fold overlap. Keep them in separate directories.
    by_lineage: dict[str, dict[str, list[float]]] = {}
    folds_seen: dict[str, list[int]] = {}
    for path in sorted(args.oof):
        path = Path(path)
        blob = json.loads(path.read_text())
        if blob["findings"] != FINDINGS:
            raise SystemExit(f"{path}: finding order differs from the schema")
        lineage = path.parent.name
        rows = by_lineage.setdefault(lineage, {})
        folds_seen.setdefault(lineage, []).append(blob["fold"])
        for study, row in zip(blob["studies"], blob["predicted"], strict=True):
            if str(study) in rows:
                raise SystemExit(
                    f"{study} predicted twice within {lineage} — the folds "
                    "overlap, so these predictions are not out-of-fold and "
                    "must not be used")
            rows[str(study)] = row

    for lineage, rows in sorted(by_lineage.items()):
        print(f"{lineage}: folds {sorted(folds_seen[lineage])}, "
              f"{len(rows):,} studies predicted exactly once each")

    # Average across lineages, keeping only studies EVERY lineage predicted, so
    # no study is scored from a smaller ensemble than its neighbours.
    shared = set.intersection(*(set(rows) for rows in by_lineage.values()))
    dropped = max(len(rows) for rows in by_lineage.values()) - len(shared)
    if dropped:
        print(f"note: {dropped} studies are missing from at least one lineage "
              "and are excluded, so every study is averaged over the same "
              "number of models")
    predictions = {
        study: np.mean([by_lineage[lineage][study] for lineage in by_lineage],
                       axis=0).tolist()
        for study in shared
    }
    if len(by_lineage) > 1:
        print(f"averaging {len(by_lineage)} lineages over {len(predictions):,} "
              "studies")

    train = pd.read_csv(args.train).set_index("StudyInstanceUID")
    soft = pd.read_parquet(args.labels).set_index("StudyInstanceUID")
    studies = [s for s in train.index.astype(str) if s in predictions]
    missing = len(train) - len(studies)
    if missing:
        print(f"note: {missing} studies have no out-of-fold prediction "
              "(not in the cache, or a fold did not finish)")

    model = np.array([predictions[s] for s in studies], dtype=np.float64)
    teacher = soft.loc[studies, FINDINGS].astype(float).to_numpy()
    teacher = np.nan_to_num(teacher, nan=0.5)

    gold_ids = set(train[train[FINDINGS].notna().all(axis=1)].index.astype(str))
    is_gold = np.array([s in gold_ids for s in studies])
    expert = train.loc[[s for s in studies if s in gold_ids], FINDINGS] \
        .to_numpy(dtype=np.float64)
    print(f"{int(is_gold.sum())} gold studies carry the comparison")

    model_ranks, teacher_ranks = ranked(model), ranked(teacher)
    union = (model_ranks + teacher_ranks) / 2

    print(f"\n{'source':28s} {'gold macro, n=58':>17s}")
    print(f"{'report labels (teacher)':28s} {macro(expert, teacher[is_gold]):17.4f}")
    print(f"{'model out-of-fold (student)':28s} {macro(expert, model[is_gold]):17.4f}")
    print(f"{'50/50 rank union':28s} {macro(expert, union[is_gold]):17.4f}")

    print("\n=== THE DECIDING ARM — parameter-free, nothing fitted ===")
    point, low, high = paired("union   ", union[is_gold],
                              "teacher ", teacher[is_gold], expert)

    # Per finding, because a macro gain carried by ONE finding is fragile and a
    # broad one is real, and 7.5 GPU-h should not be spent without knowing which
    # this is. It also catches the specific way a union can flatter itself: a
    # single easy finding moving a long way while the rest go nowhere.
    print(f"\n{'finding':18s} {'teacher':>8s} {'model':>8s} {'union':>8s} "
          f"{'union-teach':>12s} {'pos':>5s}")
    deltas = []
    for i, finding in enumerate(FINDINGS):
        y = (expert[:, i] > 0.5).astype(int)
        if not 0 < y.sum() < len(y):
            continue
        t = auc(y, teacher[is_gold][:, i])
        m = auc(y, model[is_gold][:, i])
        u = auc(y, union[is_gold][:, i])
        deltas.append((finding, t, m, u, u - t, int(y.sum())))
    for finding, t, m, u, d, n in sorted(deltas, key=lambda r: r[4]):
        mark = "  <-- WORSE" if d < 0 else ""
        print(f"{finding:18s} {t:8.3f} {m:8.3f} {u:8.3f} {d:+12.4f} {n:5d}{mark}")
    if deltas:
        values = np.array([d[4] for d in deltas])
        gained = int((values > 0).sum())
        top = max(deltas, key=lambda r: r[4])
        print(f"\n  union beats the teacher on {gained} of {len(deltas)} findings; "
              f"median {np.median(values):+.4f}, mean {values.mean():+.4f}")
        print(f"  largest single finding: {top[0]} at {top[4]:+.4f}, which is "
              f"{top[4] / len(deltas):+.4f} of the macro — "
              f"{'BROAD, not carried by one finding' if abs(top[4] / len(deltas)) < values.mean() * 0.6 else 'CHECK: one finding carries much of this'}")

        # E059 computed a hard ceiling for Synovitis: 37 of the 58 studies never
        # mention it, so a PERFECT text reader caps at 0.8076. A teacher that
        # exceeds it is not contradicting E059 — it is no longer a text reader.
        # The model separates studies the reports are silent on, which is the
        # whole reason a distilled teacher can go where a better lexicon cannot.
        syn = next((d for d in deltas if d[0] == "Synovitis"), None)
        if syn and syn[3] > 0.8076:
            print(f"\n  SYNOVITIS {syn[3]:.4f} EXCEEDS THE 0.8076 TEXT CEILING (E059).")
            print("  Not a contradiction: E059 bounds what any READER of the reports "
                  "can do,\n  and this teacher is no longer text-only. The pixels "
                  "separate the 37 studies\n  the reports are silent on. Synovitis "
                  "was PATH.md's named blocker for 0.94.")

    # Printed in full and deliberately not adopted: an argmax over this curve is
    # a free parameter fitted to 58 studies. E048 declined exactly this and the
    # reasoning has not changed.
    print("\n  the weight curve, RECORDED AND NOT USED "
          "(an argmax here is a parameter fitted to 58 studies):")
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        blend = w * model_ranks + (1 - w) * teacher_ranks
        print(f"    w_model={w:<5.2f} gold macro {macro(expert, blend[is_gold]):.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table = {"StudyInstanceUID": studies}
    for i, finding in enumerate(FINDINGS):
        table[finding] = union[:, i]
        table[f"{finding}__channel"] = "asserted"
        table[f"{finding}__weight"] = 1.0
    pd.DataFrame(table).to_parquet(out / "soft_labels.parquet", index=False)
    print(f"\nwrote {out / 'soft_labels.parquet'}")

    print("\n" + "=" * 70)
    if low > 0:
        print("PROCEED: the distilled teacher separates. Publish this parquet as a\n"
              "Kaggle dataset, add a v1pubdistil lineage, train five folds\n"
              "(~7.5 GPU-h) and let the board price it.")
    else:
        print("STOP: not separated. This is the fifth union to come back inside\n"
              "its own interval, and the pre-registered rule says no GPU.\n"
              "Record it and leave the parquet in place — it costs nothing to\n"
              "keep and the board may be worth spending on it later.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
