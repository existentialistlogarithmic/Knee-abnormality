#!/usr/bin/env python3
"""Pool the per-fold gold predictions into one out-of-fold expert evaluation.

Why this exists: report-label CV has been shown to **mis-rank** models
(`docs/FINDINGS.md` §11). It said the 288px model was the best available by
0.028; the leaderboard put it 0.037 behind. The board is trustworthy and allows
two submissions a day, so it cannot be the development signal.

The 58 expert-labelled studies are the only other expert supervision available.
Training assigns them expert labels as targets and splits into folds
afterwards, so each fold's validation set holds the gold studies that fall in
it, scored against expert truth by a model that never saw them. Pooling those
across folds gives one expert-scored prediction per gold study.

    python eda/pool_gold_oof.py a/gold_oof_fold*.json
    python eda/pool_gold_oof.py a/gold_oof_*.json --vs b/gold_oof_*.json

**Read the interval, not the point estimate**, and prefer `--vs`. Both limits
below were measured by simulation at macro AUC ≈ 0.73, not assumed
(`docs/FINDINGS.md` §13):

| comparison | n | 95% interval width |
|---|---:|---:|
| one model, absolute | 12 (one fold) | **0.173** |
| one model, absolute | 58 (five folds) | **0.079** |
| **two models, paired on the same studies** | 58 | **0.044** |

So a single fold's gold subset is worthless on its own, the pooled absolute
number cannot resolve differences under ~0.08 — it could **not** have settled
the 192px-vs-288px question, whose true gap was 0.037 — and the paired
difference is about twice as sharp, detecting a 0.08 gap every time and a 0.04
gap some of the time. Close calls still have to be paid for on the board.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BOOTSTRAP = 2000


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC, identical to sklearn's including tie handling.

    Written out because a bootstrap runs this 2,000 times across 12 findings
    and three passes; `roc_auc_score` in that loop makes the tool take minutes
    instead of seconds, and a slow tool is one nobody runs.
    """
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties, which is what makes this match sklearn
    sorted_scores = score[order]
    start = 0
    for end in range(1, len(score) + 1):
        if end == len(score) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    positives = y > 0
    n_pos = int(positives.sum())
    n_neg = len(y) - n_pos
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def macro_auc(expert: np.ndarray, predicted: np.ndarray, findings: list[str]):
    per_finding = {}
    for i, finding in enumerate(findings):
        y = (expert[:, i] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            per_finding[finding] = auc(y, predicted[:, i])
    macro = float(np.mean(list(per_finding.values()))) if per_finding else float("nan")
    return macro, per_finding


def load(paths: list[str]):
    studies, expert, predicted, folds, findings = [], [], [], [], None
    for path in sorted(paths):
        blob = json.loads(Path(path).read_text())
        findings = findings or blob["findings"]
        if blob["findings"] != findings:
            raise SystemExit(f"{path}: different finding order — cannot pool")
        studies += blob["studies"]
        expert.append(np.array(blob["expert"], float))
        predicted.append(np.array(blob["predicted"], float))
        folds += [blob["fold"]] * len(blob["studies"])

    # A study appearing twice would mean two folds validated on it, which
    # GroupKFold cannot do — so it would mean the dumps are from different runs.
    if len(set(studies)) != len(studies):
        raise SystemExit("a study appears in more than one fold — mixed runs?")

    return studies, np.concatenate(expert), np.concatenate(predicted), folds, findings


def report(label, studies, expert, predicted, folds, findings):
    print(f"{label}: pooled {len(studies)} gold studies from folds {sorted(set(folds))}")
    if len(studies) < 58:
        print(f"  {58 - len(studies)} of the 58 gold studies are NOT covered — "
              "the folds holding them have not been run with a gold dump.")

    macro, per_finding = macro_auc(expert, predicted, findings)

    rng = np.random.default_rng(0)
    samples = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(studies), len(studies))
        value, _ = macro_auc(expert[idx], predicted[idx], findings)
        if value == value:
            samples.append(value)
    low, high = np.percentile(samples, [2.5, 97.5])

    print(f"\nmacro AUC vs expert labels: {macro:.4f}  "
          f"95% CI [{low:.3f}, {high:.3f}]  (n={len(studies)}, {BOOTSTRAP} bootstrap)")
    print(f"width of the interval: {high - low:.3f} — differences smaller than "
          "this are not measurable from the absolute number. Use --vs to compare "
          "two models, which is about twice as sharp.\n")

    for finding, value in sorted(per_finding.items(), key=lambda kv: kv[1]):
        positives = int((expert[:, findings.index(finding)] > 0.5).sum())
        print(f"  {finding:18s} {value:.3f}   ({positives} positive of {len(studies)})")
    missing = [f for f in findings if f not in per_finding]
    if missing:
        print(f"\n  no AUC for {missing} — one class only at this sample size.")
    return macro


def compare(a, b):
    """Paired bootstrap of the difference, resampling STUDIES not models.

    Two models scored on the same studies make correlated errors — a study that
    is hard for one is usually hard for the other — and comparing two
    independent intervals throws that correlation away. Resampling the shared
    studies keeps it, which is where the extra sensitivity comes from.
    """
    studies_a, expert_a, predicted_a, _, findings = a
    studies_b, expert_b, predicted_b, _, findings_b = b
    if findings != findings_b:
        raise SystemExit("different finding order between the two models")

    shared = [s for s in studies_a if s in set(studies_b)]
    if not shared:
        raise SystemExit("the two models share no gold studies — nothing to pair")
    index_a = {s: i for i, s in enumerate(studies_a)}
    index_b = {s: i for i, s in enumerate(studies_b)}
    ia = np.array([index_a[s] for s in shared])
    ib = np.array([index_b[s] for s in shared])
    expert, pa, pb = expert_a[ia], predicted_a[ia], predicted_b[ib]
    if not np.allclose(expert, expert_b[ib]):
        raise SystemExit("expert labels disagree between the two dumps")

    point = macro_auc(expert, pa, findings)[0] - macro_auc(expert, pb, findings)[0]
    rng = np.random.default_rng(0)
    deltas = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(shared), len(shared))
        va = macro_auc(expert[idx], pa[idx], findings)[0]
        vb = macro_auc(expert[idx], pb[idx], findings)[0]
        if va == va and vb == vb:
            deltas.append(va - vb)
    low, high = np.percentile(deltas, [2.5, 97.5])
    verdict = ("A is better" if low > 0 else
               "B is better" if high < 0 else
               "NOT SEPARATED — the interval contains zero")
    print(f"\npaired on {len(shared)} shared gold studies")
    print(f"  A − B = {point:+.4f}  95% CI [{low:+.3f}, {high:+.3f}]")
    print(f"  {verdict}")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    if "--vs" in argv:
        cut = argv.index("--vs")
        a, b = load(argv[:cut]), load(argv[cut + 1:])
        report("A", *a)
        print()
        report("B", *b)
        return compare(a, b)
    report("model", *load(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
