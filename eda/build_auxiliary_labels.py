#!/usr/bin/env python3
"""Auxiliary training targets: the radiology the reports write and nobody scores.

    python eda/build_auxiliary_labels.py --out artifacts/auxiliary

The competition scores twelve findings. The reports describe considerably more
than twelve, and every word of it is currently thrown away. This turns thirteen
of those unscored structures into soft targets in exactly the format
`soft_labels.parquet` already uses, so training can hang extra output rows off
the shared trunk and drop them at inference.

**Why this category and not another.** Decomposed on ground truth, of the +0.198
this project has gained since its first imaging model: ensembling +0.032, its own
fused labels +0.089, public CC0 labels +0.077. Labels and targets are +0.166 of
the +0.198; every architecture lever measured zero or negative. This is a target
lever, which is the side that has paid.

**What it is not.** It is not a second labeler and it is not a change to the
twelve. `src/report_labeler.py` reads `auxiliary.csv` through the same matcher,
the same window and the same `cues.csv` as `findings.csv`, so the auxiliary
targets differ from the scored ones in vocabulary and nothing else. Auxiliary
columns are never submitted and never scored against the 58, so no choice made
here can reach the leaderboard through anything but the trunk's weights.

Reads report text into this LOCAL process and prints aggregates only
(`docs/STRATEGY.md` rule 4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_labeler import ABSTAIN, ReportLabeler, detect_language  # noqa: E402
from src.report_schema import AUXILIARY_FINDINGS  # noqa: E402


def channel_for(score: float) -> str:
    """The same four-band channel vocabulary `build_fused_labels.py` writes.

    Training reads only `"absent"`, which masks the loss; the rest exist so the
    file can be inspected without re-deriving anything.
    """
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
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--out", default="artifacts/auxiliary")
    parser.add_argument("--primary", default="artifacts/kaggle_dataset_public/soft_labels.parquet",
                        help="scored-twelve labels, for the overlap table only")
    args = parser.parse_args(argv)

    train = pd.read_csv(args.train)
    studies = train.StudyInstanceUID.astype(str).tolist()
    labeler = ReportLabeler(findings_file="auxiliary.csv")
    if sorted(labeler.findings) != sorted(AUXILIARY_FINDINGS):
        raise SystemExit("auxiliary.csv and AUXILIARY_FINDINGS disagree")

    n, f = len(studies), len(AUXILIARY_FINDINGS)
    score = np.full((n, f), np.nan)
    languages = []
    for row, report in enumerate(train.Report.astype(str)):
        language = detect_language(report)
        languages.append(language)
        labelled = labeler.label(report, language)
        for i, finding in enumerate(AUXILIARY_FINDINGS):
            value = labelled[finding].score
            if value is not ABSTAIN:
                score[row, i] = value

    supervised = ~np.isnan(score)
    print(f"{n:,} studies x {f} auxiliary findings = {score.size:,} slots; "
          f"{int(supervised.sum()):,} supervised ({supervised.mean():.1%})")
    print(f"\n{'auxiliary finding':20s} {'supervised':>11s} {'positive':>9s}"
          f" {'negated':>9s}")
    for i, finding in enumerate(AUXILIARY_FINDINGS):
        column = score[:, i]
        speaks = ~np.isnan(column)
        positive = np.nansum(column > 0.5)
        negated = np.nansum(column < 0.20)
        print(f"{finding:20s} {speaks.mean():10.1%} {positive / n:8.1%}"
              f" {negated / n:8.1%}")

    table = {"StudyInstanceUID": studies, "lang": languages}
    for i, finding in enumerate(AUXILIARY_FINDINGS):
        table[finding] = score[:, i]
        table[f"{finding}__channel"] = [channel_for(v) for v in score[:, i]]
        table[f"{finding}__weight"] = np.where(np.isnan(score[:, i]), 0.0, 1.0)
    frame = pd.DataFrame(table)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "auxiliary_labels.parquet", index=False)
    print(f"\nwrote {out / 'auxiliary_labels.parquet'}  shape {frame.shape}")

    # How much of this is genuinely new supervision rather than the scored
    # twelve under another name. A slot counts as new when the auxiliary
    # labeler speaks about a study the scored labels also cover — the trunk
    # gets a second, different statement about the same images.
    primary = Path(args.primary)
    if primary.exists():
        from src.report_schema import FINDINGS
        soft = pd.read_parquet(primary).set_index("StudyInstanceUID")
        soft = soft.reindex(studies)
        primary_supervised = np.column_stack(
            [(soft[f"{f}__channel"] != "absent").to_numpy() for f in FINDINGS])
        print(f"\nscored twelve:   {primary_supervised.mean():6.1%} of "
              f"{primary_supervised.size:,} slots supervised")
        print(f"auxiliary adds:  {int(supervised.sum()):,} further supervised "
              f"slots, a {supervised.sum() / primary_supervised.sum():+.0%} "
              "change in total supervision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
