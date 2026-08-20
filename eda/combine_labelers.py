#!/usr/bin/env python3
"""Combine the lexicon labeler and the LLM reader, and score the result on gold.

Neither wins. On the 58 expert-labelled studies the lexicon reaches macro AUC
0.769 and the LLM reader 0.7526 — but they disagree by finding, and by a lot:

    Medial Meniscus  +0.163 to the LLM        Fracture   -0.222
    Medial OA        +0.102                   PF OA      -0.124
    Lateral OA       +0.099                   Contusion  -0.114

**The combination rule has no free parameters**, which is what makes it
legitimate to evaluate on the same 58 studies used to notice the complementarity.
Per study and finding:

- exactly one labeler speaks -> take it
- both speak                 -> average their positions on their own ladders
- neither speaks             -> abstain

Nothing is fitted, nothing is selected per finding. A rule that picked the
better labeler per finding would be fitting 12 choices to 58 studies and would
report a number that means nothing.

Reads report text into this LOCAL process and prints aggregates only
(`docs/STRATEGY.md` rule 4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_labeler import ABSTAIN, ReportLabeler, detect_language  # noqa: E402
from src.report_schema import FINDINGS, STATES  # noqa: E402

BOOTSTRAP = 2000


def auc(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    sorted_scores = score[order]
    start = 0
    for end in range(1, len(score) + 1):
        if end == len(score) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    positives = y > 0
    n_pos = int(positives.sum())
    return (ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * (len(y) - n_pos))


def macro(expert: np.ndarray, score: np.ndarray):
    per = {}
    for i, name in enumerate(FINDINGS):
        y = (expert[:, i] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            per[name] = auc(y, score[:, i])
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def interval(expert, score, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(expert), len(expert))
        value, _ = macro(expert[idx], score[idx])
        if value == value:
            samples.append(value)
    return np.percentile(samples, [2.5, 97.5])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", default="artifacts/llm2/llm_states_gold.json")
    parser.add_argument("--train", default="data/train.csv")
    args = parser.parse_args(argv)

    blob = json.loads(Path(args.states).read_text())
    llm_states = dict(zip(blob["StudyInstanceUID"], blob["states"]))

    train = pd.read_csv(args.train)
    gold = train[train[FINDINGS].notna().all(axis=1)].reset_index(drop=True)
    gold = gold[gold.StudyInstanceUID.astype(str).isin(llm_states)].reset_index(drop=True)
    print(f"{len(gold)} gold studies scored by both labelers")

    labeler = ReportLabeler()
    lexicon = np.full((len(gold), len(FINDINGS)), np.nan)
    machine = np.full((len(gold), len(FINDINGS)), np.nan)
    for row, (study, report) in enumerate(zip(gold.StudyInstanceUID.astype(str),
                                              gold.Report.astype(str))):
        labelled = labeler.label(report, detect_language(report))
        for i, finding in enumerate(FINDINGS):
            score = labelled[finding].score
            if score is not ABSTAIN:
                lexicon[row, i] = score
            state = llm_states[study].get(finding)
            if state in STATES and state != "not_mentioned":
                machine[row, i] = STATES.index(state) / (len(STATES) - 1)

    # Use the KERNEL's implementation rather than a second copy. A local copy
    # is how these two disagreed in the first place: it broke ties by array
    # position where the kernel averages them, and the lexicon emits only five
    # distinct values, so ties are the common case.
    import runpy

    kernel = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "kaggle" / "16_llm_labeler" / "run.py"),
        run_name="__not_main__")
    to_rank = kernel["to_rank"]

    a, b = to_rank(lexicon), to_rank(machine)
    both = ~np.isnan(a) & ~np.isnan(b)
    only_a = ~np.isnan(a) & np.isnan(b)
    only_b = np.isnan(a) & ~np.isnan(b)

    combined = np.full_like(a, np.nan)
    combined[both] = (a[both] + b[both]) / 2
    combined[only_a] = a[only_a]
    combined[only_b] = b[only_b]

    total = a.size
    print(f"\ncoverage of the {total} study x finding slots:")
    print(f"  both labelers speak : {both.sum():4d}  {both.sum()/total:5.1%}")
    print(f"  lexicon only        : {only_a.sum():4d}  {only_a.sum()/total:5.1%}")
    print(f"  LLM only            : {only_b.sum():4d}  {only_b.sum()/total:5.1%}")
    print(f"  neither             : {(~both & ~only_a & ~only_b).sum():4d}  "
          f"{(~both & ~only_a & ~only_b).sum()/total:5.1%}")

    expert = gold[FINDINGS].to_numpy(dtype=float)
    # An abstaining slot still needs a position in the ranking. Silence is weak
    # evidence of absence in this corpus — these reports assert health rather
    # than deny disease — so it sits below anything either labeler asserted.
    def filled(x):
        return np.nan_to_num(x, nan=-1.0)

    print(f"\n{'labeler':12s} {'macro AUC':>10s} {'95% CI':>18s} {'abstain':>9s}")
    rows = [("lexicon", a), ("LLM", b), ("combined", combined)]
    results = {}
    for name, score in rows:
        value, per = macro(expert, filled(score))
        low, high = interval(expert, filled(score))
        results[name] = (value, per)
        print(f"{name:12s} {value:10.4f} [{low:6.3f}, {high:6.3f}] "
              f"{np.isnan(score).mean():8.1%}")

    # Independent intervals at n=58 are ~0.079 wide (FINDINGS.md §13) and would
    # call almost anything a tie. Two labelers scored on the SAME studies make
    # correlated errors, so resampling the shared studies and bootstrapping the
    # DIFFERENCE is about twice as sharp — and it is the difference that is the
    # claim here, not either absolute number.
    def paired(name_a, score_a, name_b, score_b):
        rng = np.random.default_rng(0)
        deltas = []
        for _ in range(BOOTSTRAP):
            idx = rng.integers(0, len(expert), len(expert))
            va, _ = macro(expert[idx], filled(score_a)[idx])
            vb, _ = macro(expert[idx], filled(score_b)[idx])
            if va == va and vb == vb:
                deltas.append(va - vb)
        low, high = np.percentile(deltas, [2.5, 97.5])
        point = macro(expert, filled(score_a))[0] - macro(expert, filled(score_b))[0]
        verdict = ("higher" if low > 0 else "lower" if high < 0
                   else "NOT SEPARATED — the interval contains zero")
        print(f"  {name_a} - {name_b}: {point:+.4f}  95% CI [{low:+.3f}, {high:+.3f}]  {verdict}")

    print("\npaired on the same 58 studies:")
    paired("combined", combined, "lexicon", a)
    paired("combined", combined, "LLM     ", b)
    paired("LLM     ", b, "lexicon", a)

    print(f"\n{'finding':18s} {'lexicon':>8s} {'LLM':>8s} {'combined':>9s} {'best':>9s}")
    for finding in FINDINGS:
        values = {n: results[n][1].get(finding, float("nan")) for n, _ in rows}
        best = max(values, key=lambda k: -1 if values[k] != values[k] else values[k])
        print(f"{finding:18s} {values['lexicon']:8.3f} {values['LLM']:8.3f} "
              f"{values['combined']:9.3f} {best:>9s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
