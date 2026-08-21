#!/usr/bin/env python3
"""Fuse the lexicon labeler and the LLM reader into one `soft_labels.parquet`.

Measured on the 58 expert-labelled studies (E023): neither labeler beats the
other — the paired interval on their difference contains zero — but their union
beats both by **+0.070**, because they abstain on different findings. The union
leaves 30.2% of study x finding slots unsupervised against 39.7% and 48.7%.

    python eda/build_fused_labels.py \
        --llm artifacts/llm3/llm_states_gold.json artifacts/llm3/llm_states_train.json \
        --out artifacts/phase1_fused

**Probabilities are averaged, not ranks.** Ranks are what the evaluation metric
reads and scored 0.8145 against 0.8060 for probabilities — a 0.0085 difference,
well inside the +-0.044 noise floor at n=58 (`FINDINGS.md` §13). But a rank
position is not a probability, and the training loss consumes magnitudes rather
than order, so the semantically correct operation wins over a difference that
cannot be measured.

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
from src.report_schema import FINDINGS, STATE_SCORE, STATE_WEIGHT, STATES  # noqa: E402


def channel_for(score: float) -> str:
    """A readable channel. Training only reads `"absent"`, which masks the loss;
    the rest exist so the file can be inspected."""
    if score != score:
        return "absent"
    if score < 0.20:
        return "negated"
    if score < 0.50:
        return "low_severity"
    if score < 0.75:
        return "hedged"
    return "asserted"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", nargs="+", required=True)
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--out", default="artifacts/phase1_fused")
    parser.add_argument("--compare", default="artifacts/phase1/soft_labels.parquet")
    args = parser.parse_args(argv)

    states: dict[str, dict] = {}
    for path in args.llm:
        blob = json.loads(Path(path).read_text())
        states.update(dict(zip([str(s) for s in blob["StudyInstanceUID"]],
                               blob["states"], strict=True)))
    train = pd.read_csv(args.train)
    studies = train.StudyInstanceUID.astype(str).tolist()
    covered = sum(s in states for s in studies)
    print(f"{len(studies):,} studies; the reader covers {covered:,}")

    labeler = ReportLabeler()
    n, f = len(studies), len(FINDINGS)
    lexicon = np.full((n, f), np.nan)
    machine = np.full((n, f), np.nan)
    machine_weight = np.zeros((n, f))
    languages = []

    for row, (study, report) in enumerate(zip(studies, train.Report.astype(str), strict=True)):
        language = detect_language(report)
        languages.append(language)
        labelled = labeler.label(report, language)
        for i, finding in enumerate(FINDINGS):
            score = labelled[finding].score
            if score is not ABSTAIN:
                lexicon[row, i] = score
            state = states.get(study, {}).get(finding)
            if state in STATES and STATE_SCORE[state] is not None:
                machine[row, i] = STATE_SCORE[state]
                machine_weight[row, i] = STATE_WEIGHT[state]

    both = ~np.isnan(lexicon) & ~np.isnan(machine)
    only_lexicon = ~np.isnan(lexicon) & np.isnan(machine)
    only_machine = np.isnan(lexicon) & ~np.isnan(machine)
    neither = np.isnan(lexicon) & np.isnan(machine)

    fused = np.where(np.isnan(lexicon), machine, lexicon)
    fused[both] = (lexicon[both] + machine[both]) / 2

    weight = np.zeros_like(fused)
    weight[both] = 1.0
    weight[only_lexicon] = 1.0
    weight[only_machine] = machine_weight[only_machine]

    # Disagreement is genuine uncertainty and is recorded rather than acted on.
    # Down-weighting on it is plausible and UNVALIDATED — 58 studies cannot
    # measure it — so it is a column for a later experiment, not a silent change
    # to the targets of the run that is about to be trained.
    disagreement = np.full_like(fused, np.nan)
    disagreement[both] = np.abs(lexicon[both] - machine[both])

    total = fused.size
    print(f"\ncoverage of {total:,} study x finding slots:")
    for label, mask in (("both labelers", both), ("lexicon only", only_lexicon),
                        ("reader only", only_machine), ("neither", neither)):
        print(f"  {label:14s} {int(mask.sum()):7,d}  {mask.mean():6.1%}")
    print(f"\nwhere both speak, mean |disagreement| = "
          f"{np.nanmean(disagreement):.3f} on a 0-1 scale")

    table = {"StudyInstanceUID": studies, "lang": languages}
    for i, finding in enumerate(FINDINGS):
        table[finding] = fused[:, i]
        table[f"{finding}__channel"] = [channel_for(v) for v in fused[:, i]]
        table[f"{finding}__weight"] = weight[:, i]
        table[f"{finding}__disagreement"] = disagreement[:, i]
    frame = pd.DataFrame(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "soft_labels.parquet", index=False)
    print(f"\nwrote {out / 'soft_labels.parquet'}  shape {frame.shape}")

    compare = Path(args.compare)
    if compare.exists():
        old = pd.read_parquet(compare)
        print(f"\n{'finding':18s} {'lexicon abstain':>16s} {'fused abstain':>14s}")
        for i, finding in enumerate(FINDINGS):
            was = (old[f"{finding}__channel"] == "absent").mean()
            now = float(np.isnan(fused[:, i]).mean())
            print(f"{finding:18s} {was:16.1%} {now:14.1%}")
        was = np.mean([(old[f"{f}__channel"] == "absent").mean() for f in FINDINGS])
        print(f"{'OVERALL':18s} {was:16.1%} {float(np.isnan(fused).mean()):14.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
