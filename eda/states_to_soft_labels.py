#!/usr/bin/env python3
"""Turn the LLM labeler's states into the `soft_labels.parquet` training reads.

    python eda/states_to_soft_labels.py artifacts/llm/llm_states_*.json \
        --out artifacts/phase1_llm

**A naming collision that will bite if it is not stated.** The existing parquet
carries a `__channel` column whose value `"absent"` means *the labeler abstained
— there is no supervision here*, and training masks the loss on it. The state
ladder in `src/report_schema.py` also has a rung called `"absent"`, and it means
the opposite: *the report explicitly says this structure is normal*, which is
real supervision and among the most useful the corpus contains.

So the mapping is deliberately not the identity:

    ladder "not_mentioned"  ->  channel "absent"      (mask the loss)
    ladder "absent"         ->  channel "negated"     (a real negative, score 0.04)
    everything else         ->  channel "asserted"/"hedged"/"low_severity"

Getting this backwards would mask every explicit normal and teach every silence
as a negative — the exact inversion the five-channel design was built to avoid,
and it would not raise anything.

No report text is read or written here; the input is states and study IDs only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_schema import (FINDINGS, STATE_SCORE, STATE_WEIGHT,  # noqa: E402
                               STATES)

# ladder rung -> the channel vocabulary training already understands
CHANNEL = {
    "not_mentioned": "absent",        # abstain: masks the loss
    "absent": "negated",              # an explicit normal, which is supervision
    "equivocal": "hedged",
    "minimal": "low_severity",
    "mild": "asserted",
    "moderate": "asserted",
    "severe": "asserted",
}
assert set(CHANNEL) == set(STATES), "every rung needs a channel"


def load(paths: list[str]) -> pd.DataFrame:
    studies, rows = [], []
    for path in sorted(paths):
        blob = json.loads(Path(path).read_text())
        ids, states = blob["StudyInstanceUID"], blob["states"]
        if len(ids) != len(states):
            raise SystemExit(f"{path}: {len(ids)} ids but {len(states)} state rows")
        studies += [str(s) for s in ids]
        rows += states
    if len(set(studies)) != len(studies):
        raise SystemExit("a study appears twice — gold and corpus dumps overlap?")
    return studies, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="llm_states_*.json from the kernel")
    parser.add_argument("--out", default="artifacts/phase1_llm")
    parser.add_argument("--compare", default="artifacts/phase1/soft_labels.parquet",
                        help="the lexicon labels, for a coverage comparison")
    parser.add_argument("--languages", default="artifacts/phase1/soft_labels.parquet",
                        help="source of the per-study `lang` column, which the "
                             "coverage audit in eda/ groups by and which nothing "
                             "recomputes downstream")
    args = parser.parse_args(argv)

    studies, rows = load(args.paths)
    print(f"{len(studies):,} studies")

    table = {"StudyInstanceUID": studies}
    unknown = 0
    for finding in FINDINGS:
        scores, channels, weights = [], [], []
        for row in rows:
            state = row.get(finding)
            if state not in STATES:
                unknown += state is not None
                state = "not_mentioned"
            scores.append(STATE_SCORE[state] if STATE_SCORE[state] is not None else np.nan)
            channels.append(CHANNEL[state])
            weights.append(STATE_WEIGHT[state])
        table[finding] = scores
        table[f"{finding}__channel"] = channels
        table[f"{finding}__weight"] = weights
    if unknown:
        print(f"  {unknown} unrecognised states abstained")

    frame = pd.DataFrame(table)

    # Carry `lang` across. eda/labeler_coverage_by_language.py groups on it, and
    # a parquet that silently loses a column breaks that audit at read time
    # rather than here, where the reason would be obvious.
    languages = Path(args.languages)
    if languages.exists():
        source = pd.read_parquet(languages, columns=["StudyInstanceUID", "lang"])
        frame = frame.merge(source, on="StudyInstanceUID", how="left")
        missing = int(frame["lang"].isna().sum())
        if missing:
            print(f"  {missing} studies have no recorded language")
    else:
        print(f"  no language source at {languages}; `lang` column omitted")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "soft_labels.parquet", index=False)
    print(f"wrote {out / 'soft_labels.parquet'}  shape {frame.shape}")

    abstain = np.mean([[row.get(f) in (None, "not_mentioned") for f in FINDINGS]
                       for row in rows])
    print(f"\noverall abstain rate: {abstain:.1%}")
    print(f"\n{'finding':18s} " + "  ".join(f"{s[:5]:>5s}" for s in STATES))
    for finding in FINDINGS:
        counts = {s: 0 for s in STATES}
        for row in rows:
            state = row.get(finding)
            counts[state if state in STATES else "not_mentioned"] += 1
        total = max(sum(counts.values()), 1)
        print(f"{finding:18s} " + "  ".join(f"{counts[s] / total:5.0%}" for s in STATES))

    compare = Path(args.compare)
    if compare.exists():
        old = pd.read_parquet(compare)
        old_abstain = np.mean([(old[f"{f}__channel"] == "absent").mean() for f in FINDINGS])
        print(f"\nabstain rate, lexicon labeler: {old_abstain:.1%}")
        print(f"abstain rate, this labeler:   {abstain:.1%}")
        print("\nper finding, abstain rate (lexicon -> LLM):")
        for finding in FINDINGS:
            was = (old[f"{finding}__channel"] == "absent").mean()
            now = np.mean([row.get(finding) in (None, "not_mentioned") for row in rows])
            arrow = "better" if now < was - 0.02 else "worse" if now > was + 0.02 else "same"
            print(f"  {finding:18s} {was:5.1%} -> {now:5.1%}   {arrow}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
