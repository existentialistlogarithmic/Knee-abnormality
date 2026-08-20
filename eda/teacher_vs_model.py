#!/usr/bin/env python3
"""Per finding: what the teacher knows, versus what the model learned from it.

The imaging model is trained on report-derived labels and scored against expert
ones. So for each finding there are two different failures, and they call for
opposite work:

- **the teacher does not know it** — the reports do not carry the finding, and
  no amount of imaging work recovers it. Fix the labels, or accept the ceiling.
- **the teacher knows it and the model did not learn it** — the signal was
  present in the targets and was lost between the targets and the predictions.
  That is a modelling or pipeline problem, and it is recoverable.

Both are measured on the same 58 expert-labelled studies, so they are directly
comparable. Reads report text into this LOCAL process and prints aggregates
only (`docs/STRATEGY.md` rule 4).
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
    return (ranks[pos].sum() - n * (n + 1) / 2) / (n * (len(y) - n))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", nargs="+", required=True,
                        help="gold_oof_*.json from every fold")
    parser.add_argument("--states", default="artifacts/llm3/llm_states_gold.json")
    parser.add_argument("--train", default="data/train.csv")
    args = parser.parse_args(argv)

    studies, model, expert = [], [], []
    for path in sorted(args.oof):
        b = json.loads(Path(path).read_text())
        studies += b["studies"]
        model.append(np.array(b["predicted"], float))
        expert.append(np.array(b["expert"], float))
    model, expert = np.concatenate(model), np.concatenate(expert)
    print(f"{len(studies)} gold studies with out-of-fold model predictions")

    train = pd.read_csv(args.train).set_index("StudyInstanceUID")
    reports = train.loc[studies, "Report"].astype(str)

    labeler = ReportLabeler()
    lexicon = np.full((len(studies), len(FINDINGS)), np.nan)
    for row, report in enumerate(reports):
        labelled = labeler.label(report, detect_language(report))
        for i, finding in enumerate(FINDINGS):
            score = labelled[finding].score
            if score is not ABSTAIN:
                lexicon[row, i] = score

    machine = np.full_like(lexicon, np.nan)
    states_path = Path(args.states)
    if states_path.exists():
        blob = json.loads(states_path.read_text())
        by_study = dict(zip(blob["StudyInstanceUID"], blob["states"]))
        for row, study in enumerate(studies):
            for i, finding in enumerate(FINDINGS):
                state = by_study.get(study, {}).get(finding)
                if state in STATES and state != "not_mentioned":
                    machine[row, i] = STATES.index(state) / (len(STATES) - 1)

    import runpy

    kernel = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "kaggle" / "16_llm_labeler" / "run.py"),
        run_name="__not_main__")
    fused = kernel["fuse"](lexicon, machine)

    print(f"\n{'finding':18s} {'teacher':>8s} {'model':>7s} {'lost':>7s} "
          f"{'abstain':>8s} {'pos':>4s}  reading")
    rows = []
    for i, finding in enumerate(FINDINGS):
        y = (expert[:, i] > 0.5).astype(int)
        if not (0 < y.sum() < len(y)):
            continue
        # The teacher is the FUSED reader — the same fuse() the labeling kernel
        # uses, not a second implementation. The first version of this script
        # had its own: prefer the lexicon, fall back to the model, never average
        # where both speak. That is a different and worse fusion than the one
        # the training targets are actually built from, and it understated the
        # teacher.
        #
        # It also scored the teacher only on the studies where it spoke, while
        # scoring the model on all 58 — two different denominators in the same
        # comparison. Abstentions now rank at the bottom for the teacher exactly
        # as they do everywhere else, so both sides answer for all 58 studies.
        teacher_score = np.nan_to_num(fused[:, i], nan=-1.0)
        teacher = auc(y, teacher_score)
        got = auc(y, model[:, i])
        lost = teacher - got if teacher == teacher else float("nan")
        abstain = float(np.isnan(fused[:, i]).mean())
        reading = ("teacher is weak — a LABEL problem" if teacher == teacher and teacher < 0.70
                   else "model lost it — RECOVERABLE" if lost == lost and lost > 0.12
                   else "model tracks its teacher")
        rows.append((finding, teacher, got, lost, abstain, int(y.sum()), reading))

    for finding, teacher, got, lost, abstain, pos, reading in sorted(rows, key=lambda r: r[2]):
        print(f"{finding:18s} {teacher:8.3f} {got:7.3f} {lost:+7.3f} "
              f"{abstain:8.1%} {pos:4d}  {reading}")

    recoverable = [r for r in rows if r[3] == r[3] and r[3] > 0.12]
    label_bound = [r for r in rows if r[1] == r[1] and r[1] < 0.70]
    print(f"\n{len(recoverable)} finding(s) where the signal was in the targets and the "
          f"model did not learn it:")
    for r in recoverable:
        print(f"  {r[0]:18s} teacher {r[1]:.3f} -> model {r[2]:.3f}")
    print(f"\n{len(label_bound)} finding(s) the teacher itself does not know:")
    for r in label_bound:
        print(f"  {r[0]:18s} teacher {r[1]:.3f}, {r[4]:.0%} abstain")
    gain = sum(r[3] for r in recoverable) / len(FINDINGS)
    print(f"\nclosing ONLY the recoverable gaps is worth {gain:+.3f} macro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
