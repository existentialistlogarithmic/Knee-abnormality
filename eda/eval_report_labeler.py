"""Evaluate the report labeler against the 58 expert-labelled studies.

Reports per-finding AUC with bootstrap intervals, agreement, and — the number
that decides how useful the abstain channel is — how often the report is simply
silent about a finding.

Everything printed is an aggregate. No report text, no identifiers.

Usage:
    python eda/eval_report_labeler.py
    python eda/eval_report_labeler.py --abstain-score 0.10 --emit-soft-labels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.report_labeler import ReportLabeler, detect_language  # noqa: E402


def bootstrap_auc(truth, scores, n: int = 2000, seed: int = 0):
    """Percentile bootstrap. With 58 studies the interval is the headline, not the point."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    truth = np.asarray(truth)
    scores = np.asarray(scores)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(truth), len(truth))
        if len(set(truth[idx])) < 2:
            continue
        values.append(roc_auc_score(truth[idx], scores[idx]))
    if not values:
        return (float("nan"), float("nan"))
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(REPO_ROOT / "data" / "train.csv"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "phase1"))
    parser.add_argument(
        "--abstain-score",
        type=float,
        default=0.15,
        help="score for a finding the report never mentions. Radiology reports "
        "tend to mention what is present, so silence leans negative — but not as "
        "strongly as an explicit denial.",
    )
    parser.add_argument("--emit-soft-labels", action="store_true",
                        help="write soft labels for all 4,407 studies for Phase 2")
    args = parser.parse_args()

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    train = pd.read_csv(args.train)
    findings = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labeler = ReportLabeler()
    print(f"lexicon: {len(labeler.findings)} findings")
    missing = set(findings) - set(labeler.findings)
    if missing:
        print(f"WARNING: no lexicon entries for {sorted(missing)}")

    print(f"labelling {len(train):,} reports …")
    train["lang"] = train.Report.astype(str).map(detect_language)
    labels = [labeler.label(row.Report, row.lang) for row in train.itertuples(index=False)]

    score_frame = pd.DataFrame(
        {f: [lab[f].score for lab in labels] for f in findings}, index=train.index
    )
    channel_frame = pd.DataFrame(
        {f: [lab[f].channel for lab in labels] for f in findings}, index=train.index
    )

    # ---- abstain / channel behaviour over the whole corpus ------------------ #
    print("\nChannel mix across all 4,407 reports (share of studies)")
    print(f"{'finding':<18}{'asserted':>10}{'hedged':>9}{'low sev':>9}"
          f"{'negated':>10}{'absent':>9}")
    channel_stats = {}
    for finding in findings:
        counts = channel_frame[finding].value_counts(normalize=True)
        channel_stats[finding] = {k: round(float(v), 4) for k, v in counts.items()}
        print(f"{finding:<18}"
              f"{counts.get('asserted', 0):>10.3f}{counts.get('hedged', 0):>9.3f}"
              f"{counts.get('low_severity', 0):>9.3f}{counts.get('negated', 0):>10.3f}"
              f"{counts.get('absent', 0):>9.3f}")

    # ---- evaluate on the gold subset ---------------------------------------- #
    gold_mask = train[findings].notna().all(axis=1)
    gold = train[gold_mask]
    gold_scores = score_frame[gold_mask].astype(float).fillna(args.abstain_score)
    print(f"\nEvaluating on {len(gold)} expert-labelled studies "
          f"(abstain scored {args.abstain_score})")

    rows = []
    for finding in findings:
        truth = gold[finding].astype(int).to_numpy()
        scores = gold_scores[finding].to_numpy()
        if len(set(truth)) < 2:
            continue
        auc = roc_auc_score(truth, scores)
        low, high = bootstrap_auc(truth, scores)
        abstain_rate = float((channel_frame[gold_mask][finding] == "absent").mean())
        rows.append({"finding": finding, "auc": round(float(auc), 4),
                     "ci_low": round(low, 4), "ci_high": round(high, 4),
                     "positives": int(truth.sum()),
                     "abstain_rate_gold": round(abstain_rate, 4)})

    report = pd.DataFrame(rows).sort_values("auc", ascending=False)
    print(f"\n{'finding':<18}{'AUC':>7}{'95% CI':>18}{'pos':>5}{'abstain':>9}")
    print("-" * 57)
    for r in report.itertuples(index=False):
        ci = f"[{r.ci_low:.2f}, {r.ci_high:.2f}]"
        print(f"{r.finding:<18}{r.auc:>7.3f}{ci:>18}{r.positives:>5}{r.abstain_rate_gold:>9.2f}")

    macro = float(report.auc.mean())
    print(f"\nMACRO AUC (report labeler vs expert labels): {macro:.4f}")
    print("  crude keyword floor, from the Phase 0 probe:  0.601 balanced accuracy")
    print(f"  findings above 0.70 AUC: {int((report.auc > 0.70).sum())} of {len(report)}")
    print(f"  findings below 0.55 AUC: {int((report.auc < 0.55).sum())} of {len(report)}")

    payload = {
        "n_gold": int(len(gold)),
        "abstain_score": args.abstain_score,
        "macro_auc": round(macro, 4),
        "per_finding": rows,
        "channel_mix_all_studies": channel_stats,
        "caveat": "n=58; intervals are wide and the gold subset is more English "
                  "than the corpus (48% vs 39%), so this overstates real performance.",
    }
    (out_dir / "labeler_eval.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'labeler_eval.json'}")

    if args.emit_soft_labels:
        soft = score_frame.copy()
        soft.insert(0, "StudyInstanceUID", train.StudyInstanceUID)
        for finding in findings:
            soft[f"{finding}__channel"] = channel_frame[finding]
        soft["lang"] = train.lang
        path = out_dir / "soft_labels.parquet"
        soft.to_parquet(path, index=False)
        print(f"wrote {path}  ({len(soft):,} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
