"""Phase 0 diagnostic: how much finding-signal is in the reports at all?

This is deliberately the crudest possible report labeler — case-folded substring
matching against a small multilingual keyword list, **with no negation handling,
no hedging, no severity thresholding and no laterality logic**. Those are exactly
what Phase 1 builds, so this is a *floor*, not an estimate of what a real
labeler achieves.

Why run it before building anything: the whole strategy rests on report-derived
labels being good enough to train on. If even a crude matcher shows real signal
on the 58 gold studies, the premise holds and Phase 1 is worth the effort. If it
shows nothing, something is wrong with the premise and it is much better to find
out now than after two weeks of lexicon work.

Everything printed is an aggregate. No report text and no identifiers.

Usage:
    python eda/label_ceiling_probe.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "artifacts" / "phase0"

# Crude, English-led, with obvious cognates for the larger non-English shares
# (es 16%, tr 12%, de 6%, nl 4%, fr 2%). Greek/Bulgarian/Croatian are left out
# on purpose: their absence is part of what this probe measures.
KEYWORDS: dict[str, list[str]] = {
    "ACL": ["acl", "anterior cruciate", "ligamento cruzado anterior", "kreuzband",
            "ön çapraz", "voorste kruisband", "ligament croisé antérieur", "lca"],
    "MCL": ["mcl", "medial collateral", "ligamento colateral medial", "innenband",
            "iç yan bağ", "mediale collaterale", "ligament collatéral médial"],
    "Medial Meniscus": ["medial meniscus", "menisco medial", "innenmeniskus",
                        "medial menisküs", "mediale meniscus", "ménisque interne",
                        "medial mensicus"],
    "Lateral Meniscus": ["lateral meniscus", "menisco lateral", "aussenmeniskus",
                         "außenmeniskus", "lateral menisküs", "laterale meniscus",
                         "ménisque externe"],
    "Medial OA": ["medial compartment", "medial osteoarthritis", "artrosis medial",
                  "medial gonartroz", "mediale gonarthrose"],
    "Lateral OA": ["lateral compartment", "lateral osteoarthritis", "artrosis lateral",
                   "lateral gonartroz", "laterale gonarthrose"],
    "PF OA": ["patellofemoral", "patello-femoral", "patelofemoral", "retropatellar",
              "femoropatellar", "chondromalacia"],
    "Effusion": ["effusion", "derrame", "erguss", "efüzyon", "effusie", "épanchement",
                 "joint fluid", "hydrops"],
    "Synovitis": ["synovitis", "sinovitis", "synovialitis", "sinovit", "synovite",
                  "synovial thickening"],
    "Baker's": ["baker", "popliteal cyst", "quiste de baker", "bakerzyste",
                "poplietal", "kyste poplité", "bakerzyste"],
    "Contusion": ["contusion", "bone marrow edema", "bone marrow oedema", "bone bruise",
                  "contusión", "knochenmarködem", "kemik iliği ödem", "beenmergoedeem"],
    "Fracture": ["fracture", "fractura", "fraktur", "kırık", "fractuur", "avulsion"],
}


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — behaves sanely at the tiny counts we have here."""
    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=str(DATA / "train.csv"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    import pandas as pd

    train_path = Path(args.train)
    if not train_path.exists():
        print(f"missing {train_path}. Run eda/phase0_02_audit_tabular.py first.")
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_path)
    findings = [c for c in train.columns if c not in ("StudyInstanceUID", "Report")]
    gold = train[train[findings].notna().all(axis=1)].copy()
    print(f"gold studies: {len(gold)}   findings: {len(findings)}")

    lowered = gold.Report.astype(str).str.lower()
    patterns = {
        finding: re.compile("|".join(re.escape(k) for k in keys))
        for finding, keys in KEYWORDS.items()
    }

    rows = []
    for finding in findings:
        truth = gold[finding].astype(int).to_numpy()
        pattern = patterns.get(finding)
        if pattern is None:
            continue
        predicted = lowered.str.contains(pattern, regex=True).astype(int).to_numpy()

        tp = int(((predicted == 1) & (truth == 1)).sum())
        fp = int(((predicted == 1) & (truth == 0)).sum())
        fn = int(((predicted == 0) & (truth == 1)).sum())
        tn = int(((predicted == 0) & (truth == 0)).sum())
        n = tp + fp + fn + tn
        agree = (tp + tn) / n if n else 0.0
        low, high = wilson(tp + tn, n)
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        balanced = (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else float("nan")
        rows.append(
            {
                "finding": finding,
                "n": n,
                "positives": int(truth.sum()),
                "mentioned": int(predicted.sum()),
                "agreement": round(agree, 4),
                "agree_lo": round(low, 4),
                "agree_hi": round(high, 4),
                "sensitivity": round(sens, 4) if not math.isnan(sens) else None,
                "specificity": round(spec, 4) if not math.isnan(spec) else None,
                "balanced_accuracy": round(balanced, 4) if not math.isnan(balanced) else None,
            }
        )

    report = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)

    print("\nCrude keyword match vs expert labels, on the gold subset")
    print("(no negation handling — a report saying 'no ACL tear' counts as a mention)\n")
    header = (f"{'finding':<18}{'pos':>4}{'ment':>6}{'agree':>8}"
              f"{'95% CI':>16}{'sens':>7}{'spec':>7}{'bal.acc':>9}")
    print(header)
    print("-" * len(header))
    for r in report.itertuples(index=False):
        ci = f"[{r.agree_lo:.2f},{r.agree_hi:.2f}]"
        print(f"{r.finding:<18}{r.positives:>4}{r.mentioned:>6}{r.agreement:>8.3f}"
              f"{ci:>16}{(r.sensitivity or 0):>7.2f}{(r.specificity or 0):>7.2f}"
              f"{(r.balanced_accuracy or 0):>9.3f}")

    macro = report.balanced_accuracy.dropna().mean()
    print(f"\nmacro balanced accuracy: {macro:.3f}")
    print("Read this as a floor. Negation alone should move several of these a lot:")
    print("a report reading 'no meniscal tear' is currently scored as a positive mention.")

    # Which languages do the gold studies speak? If they are English-heavy, the
    # gold evaluation flatters an English-led labeler and hides the real gap.
    try:
        import py3langid

        langs = (
            gold.Report.astype(str)
            .map(lambda t: py3langid.classify(t[:2000])[0])
            .value_counts()
        )
        all_langs = (
            train.Report.astype(str)
            .sample(min(4000, len(train)), random_state=0)
            .map(lambda t: py3langid.classify(t[:2000])[0])
            .value_counts(normalize=True)
        )
        print("\nlanguage mix, gold subset vs whole training set")
        print(f"{'lang':<6}{'gold n':>8}{'gold %':>9}{'train %':>10}")
        for lang, count in langs.items():
            print(f"{lang:<6}{count:>8}{count / len(gold):>9.1%}"
                  f"{all_langs.get(lang, 0):>10.1%}")
        print("\nIf the gold subset is more English than the training set, every")
        print("evaluation on it overstates how well an English-led labeler will do.")
    except ImportError:
        print("\npy3langid not installed; skipping the language comparison")
        langs = None

    payload = {
        "n_gold": int(len(gold)),
        "macro_balanced_accuracy": round(float(macro), 4),
        "per_finding": rows,
        "gold_language_counts": {str(k): int(v) for k, v in langs.items()} if langs is not None else None,
        "caveat": (
            "Crude substring matching with no negation, hedging, severity or "
            "laterality handling. A floor, not an estimate of Phase 1 performance. "
            "n=58, so every interval is wide."
        ),
    }
    (out_dir / "label_ceiling_probe.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_dir / 'label_ceiling_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
