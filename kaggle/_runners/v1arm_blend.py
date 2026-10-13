"""Rank-blend our v1 arm into another pipeline's submission (E155).

    python v1arm_blend.py <base.csv> <v1arm.csv> <out.csv> <weight>

`base` is the submission another notebook already wrote (e.g. a public 0.957
pipeline); `v1arm` is ours, written by `run_v1arm.py`. For each of the twelve
findings: out = (1 - w) * rankpct(base) + w * rankpct(v1arm), the same
rank-space voting every blend in this project uses, because AUC reads order
only and the two pipelines' probabilities are not on a common scale.

Joined on StudyInstanceUID, never on position; refuses on missing studies,
non-finite values, or a weight outside (0, 1). Rows keep `base`'s order.
"""
import sys

import numpy as np
import pandas as pd

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]


def rankpct(a: np.ndarray) -> np.ndarray:
    """Tie-averaged percentile rank of a 1-D array, in [0, 1]."""
    s = pd.Series(a)
    return ((s.rank(method="average") - 1) / max(len(s) - 1, 1)).to_numpy()


def blend(base_csv: str, ours_csv: str, out_csv: str, w: float) -> pd.DataFrame:
    if not 0.0 < w < 1.0:
        raise SystemExit(f"weight must be in (0, 1), got {w}")
    base = pd.read_csv(base_csv)
    ours = pd.read_csv(ours_csv).set_index("StudyInstanceUID")
    missing = [s for s in base.StudyInstanceUID if s not in ours.index]
    if missing:
        raise SystemExit(f"v1 arm has no row for {len(missing)} studies, first {missing[:3]}")
    ours = ours.loc[base.StudyInstanceUID]
    out = base.copy()
    for f in FINDINGS:
        b, o = base[f].to_numpy(float), ours[f].to_numpy(float)
        if not (np.isfinite(b).all() and np.isfinite(o).all()):
            raise SystemExit(f"non-finite values in {f}")
        out[f] = (1.0 - w) * rankpct(b) + w * rankpct(o)
    out.to_csv(out_csv, index=False)
    agree = np.mean([np.corrcoef(rankpct(base[f].to_numpy(float)),
                                 rankpct(ours[f].to_numpy(float)))[0, 1]
                     if len(base) > 2 else np.nan for f in FINDINGS])
    print(f"[v1arm-blend] {len(out)} studies, weight {w:.2f} on our v1 arm, "
          f"mean rank correlation base/v1 {agree:.3f} -> {out_csv}", flush=True)
    return out


if __name__ == "__main__":
    blend(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
