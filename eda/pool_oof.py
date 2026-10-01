#!/usr/bin/env python3
"""Pool the five full validation folds into one out-of-fold matrix, then score it.

**Why this exists, and what it is not.** `pool_gold_oof.py` pools the *gold*
rows of each fold into an n=58 expert evaluation. That instrument has 9
positives in its thinnest column and a per-finding standard error of ±0.05 to
±0.08 (E136), so it cannot rank two variants that differ by less than that —
which is every variant this project now tries. This pools the *whole* fold
instead: one honestly out-of-fold prediction for every study in the cohort,
roughly 75× the rows, scored against the report labels the models trained on.

    python eda/pool_oof.py runs/oof_fold*.csv.gz
    python eda/pool_oof.py a/oof_fold*.csv.gz --vs b/oof_fold*.csv.gz

**READ THIS BEFORE TRUSTING A NUMBER FROM IT.** The targets are LLM-extracted
report labels, not radiologist truth, and they are measurably worse than the
models trained on them (E133: our models beat their own supervision by up to
+0.155). So:

* **Absolute values mean little.** A macro of 0.86 here is not comparable to
  0.93 on gold-58 or 0.945 on the board. Do not put them in the same table.
* **Rankings are sound, with one exception.** Under class-conditional noise,
  AUC against noisy labels is an affine, rank-preserving function of clean AUC,
  so "A beats B" transfers even though "A scores x" does not.
* **The exception is the lateral pair.** E136's literature review found lateral
  posterior-horn tears are *unreported* rather than reported wrongly, which is
  instance-dependent noise, and the affine argument does not hold there. For
  `Lateral Meniscus` and `Lateral OA`, gold stays the judge. This script prints
  those two rows marked, and refuses to fold them into a headline macro unless
  `--include-lateral` is passed.

**The integrity check is not optional.** Five folds must partition the cohort:
every study exactly once. A study appearing twice means the fold assignment
moved between runs — a different cache, a changed grouping key, a re-run after
an edit — and pooling then averages a study's prediction with one from a model
that *trained* on it. That reads as a strong model and is a silent fabrication,
so it raises rather than warns.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LATERAL = ("Lateral Meniscus", "Lateral OA")
N_BOOT = 10000


def auc(y, s):
    """Tie-averaged Mann-Whitney AUC; nan for a single-class column."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    s = np.asarray(s, dtype=np.float64)
    r = s.argsort().argsort().astype(np.float64) + 1.0
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def load(paths):
    """Read the fold dumps and prove they partition the cohort."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        sys.exit("no fold files given")
    df = pd.concat(frames, ignore_index=True)
    dup = df["StudyInstanceUID"].duplicated(keep=False)
    if dup.any():
        offenders = df.loc[dup, ["StudyInstanceUID", "fold"]]
        raise SystemExit(
            f"{int(dup.sum())} rows cover {offenders['StudyInstanceUID'].nunique()} "
            f"studies more than once. The folds do not partition the cohort, so "
            f"pooling would average a held-out prediction with an in-sample one "
            f"and read as a strong model. First few:\n{offenders.head(10)}"
        )
    folds = sorted(df["fold"].unique())
    print(f"{len(paths)} files | folds {folds} | {len(df):,} studies, each exactly once")
    return df


def columns(df):
    findings = [c for c in df.columns
                if c not in ("StudyInstanceUID", "fold") and not c.startswith("y_")]
    missing = [f for f in findings if f"y_{f}" not in df.columns]
    if missing:
        raise SystemExit(f"no target column for {missing}")
    return findings


def score(df, findings):
    out = {}
    for f in findings:
        y = (df[f"y_{f}"].to_numpy(float) > 0.5).astype(int)
        out[f] = (auc(y, df[f].to_numpy(float)), int(y.sum()))
    return out


def paired_bootstrap(df_a, df_b, findings, n_boot=N_BOOT, seed=0):
    """Resample STUDIES, scoring both sides on the same draw."""
    common = sorted(set(df_a["StudyInstanceUID"]) & set(df_b["StudyInstanceUID"]))
    a = df_a.set_index("StudyInstanceUID").loc[common]
    b = df_b.set_index("StudyInstanceUID").loc[common]
    Y = np.column_stack([(a[f"y_{f}"].to_numpy(float) > 0.5).astype(int) for f in findings])
    A = np.column_stack([a[f].to_numpy(float) for f in findings])
    B = np.column_stack([b[f].to_numpy(float) for f in findings])
    rng = np.random.default_rng(seed)
    n = len(common)
    d = []
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        yy = Y[k]
        ma = np.nanmean([auc(yy[:, j], A[k, j]) for j in range(len(findings))])
        mb = np.nanmean([auc(yy[:, j], B[k, j]) for j in range(len(findings))])
        d.append(ma - mb)
    d = np.asarray(d)
    return (float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)),
            float(np.nanpercentile(d, 97.5)), float(np.nanmean(d > 0)), len(common))


def report(df, findings, include_lateral):
    s = score(df, findings)
    scored = [f for f in findings if include_lateral or f not in LATERAL]
    print(f"\n{'finding':20s} {'AUC':>8s} {'n_pos':>7s}")
    for f in sorted(findings, key=lambda x: s[x][0]):
        mark = "" if f in scored else "   <- report-label noise is instance-dependent here;"
        print(f"{f:20s} {s[f][0]:8.4f} {s[f][1]:7d}{mark}")
        if mark:
            print(f"{'':20s} {'':8s} {'':7s}    score it on gold-58, not here (E136)")
    macro = float(np.nanmean([s[f][0] for f in scored]))
    print(f"\nmacro over {len(scored)} findings: {macro:.4f}"
          + ("" if include_lateral else "   (the lateral pair excluded; --include-lateral to override)"))
    return macro


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--vs", nargs="+", type=Path, default=None,
                    help="a second set of fold dumps; prints the paired bootstrap")
    ap.add_argument("--include-lateral", action="store_true",
                    help="fold Lateral Meniscus and Lateral OA into the macro anyway")
    a = ap.parse_args()

    df = load(a.files)
    findings = columns(df)
    report(df, findings, a.include_lateral)

    if a.vs:
        print("\n--- vs ---")
        df_b = load(a.vs)
        if columns(df_b) != findings:
            raise SystemExit("the two sides declare different findings")
        report(df_b, findings, a.include_lateral)
        scored = [f for f in findings if a.include_lateral or f not in LATERAL]
        d, lo, hi, p, n = paired_bootstrap(df, df_b, scored)
        print(f"\nfirst minus second, paired on {n:,} studies:")
        print(f"   {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(first better) {p:.3f}")
        print("\nRead the interval. A difference whose CI spans zero has not been "
              "shown,\nand on this many studies the interval is the point of the "
              "exercise.")


if __name__ == "__main__":
    main()
