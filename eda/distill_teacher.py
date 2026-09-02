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

    predictions: dict[str, list[float]] = {}
    folds_seen = []
    for path in sorted(args.oof):
        blob = json.loads(Path(path).read_text())
        if blob["findings"] != FINDINGS:
            raise SystemExit(f"{path}: finding order differs from the schema")
        folds_seen.append(blob["fold"])
        for study, row in zip(blob["studies"], blob["predicted"], strict=True):
            if study in predictions:
                raise SystemExit(
                    f"{study} predicted twice — the folds overlap, so these "
                    "predictions are not out-of-fold and must not be used")
            predictions[str(study)] = row
    print(f"folds {sorted(folds_seen)}; {len(predictions):,} studies predicted "
          "exactly once each")

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
