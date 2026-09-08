#!/usr/bin/env python3
"""Screen a public label set against the 58 expert studies, with the control.

    python eda/survey_public_labels.py --gold data/train.csv \
        --labels a.csv b.csv --names "set a" "set b"

**Why the control matters, and why this script exists rather than a one-liner.**
`PATH.md` §2.3 says the screen is *"what fraction of gold cells equal the expert
value exactly"*, because E047 and E080 both found sets built from the answer key,
and those reproduced **100%** of gold cells. Stated without a threshold that rule
is a trap: **the gold set is 34.5% positive, so a labeler that answers zero to
everything already matches 65.5% of cells exactly.** Any hard or coarse label set
therefore scores 65-85% by accuracy alone and looks "suspect" when it is merely
ordinary. E057's lesson, applied to a screen instead of an experiment: **a
negative needs a control arm.** This prints the all-zeros floor beside every
number so the comparison is against the floor, not against zero.

**What separates the two cases.** An answer key sits at or next to 100% and its
AUC is near 1.0. An ordinary labeler sits a little above the floor and has an AUC
worth reading. `lixin73/rsna-knee-llm-report-labels-sol56` reads 79.9% against a
65.5% floor and 0.8352 macro AUC — coarse and mediocre, not leaked.

**The bar is the incumbent, not zero.** This project already trains on
`stevenleehans/rsna-knee-llm-report-labels`, which reads **0.8927** on the 58.
A candidate that scores below that is not an upgrade no matter how clean it is,
and `PATH.md` §2.3 warns against comparing a candidate to a number the project
has already banked.

**A set with no gold overlap cannot be screened at all**, and that is a real
category rather than a failure: `dreaddevelopment/rsna-knee-labels` covers all
4,407 studies *minus exactly the 58 gold*, so it is clean by construction and
unscoreable by construction (E088).

Reads patient-derived label files but prints only aggregates.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]

INCUMBENT_GOLD_AUC = 0.8927      # stevenleehans v4 blend, what we train on today
TEACHER_GOLD_AUC = 0.9188        # E069's fused teacher, the bar PATH 2.3 sets

# An answer key reproduces the expert cell, not merely predicts it well. E047 and
# E080 both saw 100%. Anything near the base rate is accuracy.
ANSWER_KEY_MARGIN = 0.98


def gold_frame(train_csv: str) -> pd.DataFrame:
    """The 58 studies that carry all twelve expert labels."""
    train = pd.read_csv(train_csv)
    complete = train[FINDINGS].notna().all(axis=1)
    return train.loc[complete, ["StudyInstanceUID"] + FINDINGS]


def screen(gold: pd.DataFrame, candidate: pd.DataFrame) -> dict:
    """Exact-cell rate, its all-zeros floor, and macro AUC on the overlap."""
    missing = [f for f in FINDINGS if f not in candidate.columns]
    if "StudyInstanceUID" not in candidate.columns or missing:
        return {"error": f"missing columns: {missing[:3] or 'StudyInstanceUID'}"}
    merged = gold.merge(candidate, on="StudyInstanceUID", suffixes=("_x", "_n"))
    expert = gold[FINDINGS].to_numpy(float)
    floor = float((expert == 0).mean())
    if merged.empty:
        return {"n_gold": 0, "floor": floor,
                "note": "no gold overlap — clean by construction, unscoreable by construction"}
    x = merged[[f + "_x" for f in FINDINGS]].to_numpy(float)
    n = merged[[f + "_n" for f in FINDINGS]].to_numpy(float)
    aucs = [roc_auc_score(x[:, i], n[:, i]) for i in range(len(FINDINGS))
            if len(np.unique(x[:, i])) > 1]
    result = {"n_gold": len(merged), "exact": float((n == x).mean()), "floor": floor,
              "distinct": int(np.unique(n).size)}
    # No finding with both classes present means there is no AUC to report. Saying
    # so beats returning nan: `nan >= incumbent` is False, so a nan would print as
    # "BEHIND the incumbent" — a missing measurement dressed as a negative one.
    if aucs:
        result["macro_auc"] = float(np.mean(aucs))
    return result


def verdict(result: dict) -> str:
    if "error" in result:
        return result["error"]
    if not result.get("n_gold"):
        return "UNSCOREABLE — no gold overlap; only the board can price it"
    if result["exact"] >= ANSWER_KEY_MARGIN:
        return "ANSWER KEY — it reproduces the expert cells; unusable"
    if "macro_auc" not in result:
        return "NO AUC — no finding on the overlap has both classes"
    if result["macro_auc"] >= INCUMBENT_GOLD_AUC:
        return f"AHEAD of the incumbent {INCUMBENT_GOLD_AUC:.4f} — measure it properly"
    return f"BEHIND the incumbent {INCUMBENT_GOLD_AUC:.4f} — not an upgrade"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="competition train.csv")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--names", nargs="*", default=None)
    args = parser.parse_args(argv)

    gold = gold_frame(args.gold)
    names = args.names or args.labels
    print(f"gold studies: {len(gold)}   positive rate: "
          f"{gold[FINDINGS].to_numpy(float).mean() * 100:.1f}%")
    print(f"{'label set':34s} {'gold':>5} {'exact':>7} {'floor':>7} {'macroAUC':>9}  verdict")
    print("-" * 100)
    for name, path in zip(names, args.labels, strict=False):
        result = screen(gold, pd.read_csv(path))
        exact = f"{result['exact'] * 100:.1f}%" if "exact" in result else "—"
        floor = f"{result['floor'] * 100:.1f}%" if "floor" in result else "—"
        auc = f"{result['macro_auc']:.4f}" if "macro_auc" in result else "—"
        print(f"{name[:34]:34s} {result.get('n_gold', 0):>5} {exact:>7} {floor:>7} "
              f"{auc:>9}  {verdict(result)}")
    print(f"\nFLOOR is what an all-zeros labeler scores. Read exact AGAINST THE FLOOR, "
          f"never against zero.\nBars: incumbent {INCUMBENT_GOLD_AUC:.4f} (what we train on), "
          f"E069 teacher {TEACHER_GOLD_AUC:.4f}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
