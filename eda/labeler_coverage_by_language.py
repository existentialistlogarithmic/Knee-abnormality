"""Audit labeler coverage per language, without using any expert labels.

The gold subset is 58 studies and 48% English, so it cannot say whether the
Turkish or Greek lexicons work. This does, using a signal that needs no labels
at all: **how often the labeler abstains**.

If a finding abstains on 30% of English reports and 90% of Turkish ones, that
gap is a lexicon hole, not a fact about knees — radiologists in Istanbul are not
declining to mention the ACL. Comparing each language against the English
baseline localises the missing vocabulary to a language-finding cell.

This is the optimisation signal for lexicon work, deliberately: it keeps the 58
gold studies as a test set rather than burning them as a training signal.

Outputs aggregates only — no report text, no identifiers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.report_labeler import ReportLabeler, detect_language  # noqa: E402

MIN_REPORTS = 50  # below this a language's rates are too noisy to act on


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(REPO_ROOT / "data" / "train.csv"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "phase1"))
    parser.add_argument("--baseline", default="en", help="language to compare against")
    parser.add_argument("--gap", type=float, default=0.20,
                        help="flag a cell whose abstain rate exceeds the baseline by this much")
    args = parser.parse_args()

    import pandas as pd

    train = pd.read_csv(args.train)
    findings = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeler = ReportLabeler()
    print(f"labelling {len(train):,} reports …")
    train["lang"] = train.Report.astype(str).map(detect_language)
    labels = [labeler.label(row.Report, row.lang) for row in train.itertuples(index=False)]
    channels = pd.DataFrame({f: [lab[f].channel for lab in labels] for f in findings})
    channels["lang"] = train.lang.to_numpy()

    counts = channels.lang.value_counts()
    langs = [lang for lang in counts.index if counts[lang] >= MIN_REPORTS]
    print(f"languages with >= {MIN_REPORTS} reports: {', '.join(langs)}")

    abstain = pd.DataFrame(
        {lang: [(channels[channels.lang == lang][f] == "absent").mean() for f in findings]
         for lang in langs},
        index=findings,
    )

    print("\nABSTAIN RATE by finding and language (lower is better coverage)")
    header = f"{'finding':<18}" + "".join(f"{lang:>7}" for lang in langs)
    print(header)
    print("-" * len(header))
    for finding in findings:
        print(f"{finding:<18}" + "".join(f"{abstain.loc[finding, lang]:>7.2f}" for lang in langs))
    print(f"{'MEAN':<18}" + "".join(f"{abstain[lang].mean():>7.2f}" for lang in langs))

    # ---- localise the holes ------------------------------------------------- #
    baseline = args.baseline
    gaps = []
    if baseline in abstain.columns:
        for finding in findings:
            base = abstain.loc[finding, baseline]
            for lang in langs:
                if lang == baseline:
                    continue
                delta = abstain.loc[finding, lang] - base
                if delta >= args.gap:
                    gaps.append({"finding": finding, "language": lang,
                                 "abstain": round(float(abstain.loc[finding, lang]), 3),
                                 "baseline_abstain": round(float(base), 3),
                                 "gap": round(float(delta), 3),
                                 "reports": int(counts[lang])})
        gaps.sort(key=lambda g: -g["gap"] * g["reports"])

    print(f"\nLEXICON HOLES — abstain exceeds {baseline} by >= {args.gap:.2f}")
    print("Ranked by gap x reports affected, i.e. by how much fixing it is worth.")
    print(f"\n{'finding':<18}{'lang':>5}{'abstain':>9}{f'{baseline}':>7}{'gap':>7}{'reports':>9}")
    print("-" * 55)
    for g in gaps[:25]:
        print(f"{g['finding']:<18}{g['language']:>5}{g['abstain']:>9.2f}"
              f"{g['baseline_abstain']:>7.2f}{g['gap']:>7.2f}{g['reports']:>9}")
    if not gaps:
        print("  none — coverage is even across languages")

    payload = {
        "reports_per_language": {k: int(v) for k, v in counts.items()},
        "abstain_by_finding_language": {
            f: {lang: round(float(abstain.loc[f, lang]), 4) for lang in langs} for f in findings
        },
        "holes": gaps,
        "note": "Abstain rate needs no labels, so this is the safe optimisation "
                "signal; the 58 gold studies stay a test set.",
    }
    (out_dir / "coverage_by_language.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'coverage_by_language.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
