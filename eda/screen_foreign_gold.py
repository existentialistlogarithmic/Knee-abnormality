"""Screen a foreign asset's gold-58 predictions against our composite (E125's rule).

A candidate member has to clear TWO bars, not either one:

  * **comparable in quality** -- E121 found same-author checkpoints inside our
    correlation band do not pay even when comparable, and E125 found an asset at
    rho 0.317, the most independent ever measured here, cost -0.0333 because it
    was 0.29 behind.
  * **independent in error** -- rho against the composite, not against truth.

So this prints both axes and then the only number that decides it: the paired
bootstrap of the composite WITH the candidate as one more vote against the
composite alone, on the same 58 studies and the same resamples.

Everything here is arithmetic on 58 rows. It spends no GPU and no submission.
"""
from __future__ import annotations

import numpy as np

N_BOOT = 10000
BOOT_SEED = 0


def rankpct(a):
    """Column-wise rank in [0, 1], the blend space every kernel here votes in."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(axis=0).argsort(axis=0).astype(np.float64)
    return order / max(len(a) - 1, 1)


def auc(y, s):
    """Mann-Whitney AUC. Returns nan for a column with one class, as macro must skip it."""
    y = np.asarray(y).astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = np.asarray(s, dtype=np.float64).argsort().argsort().astype(np.float64) + 1.0
    # average ties, or a saturated head scores by its ordering of equal scores
    s = np.asarray(s, dtype=np.float64)
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    return (r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auc(y, s):
    per = [auc(y[:, j], s[:, j]) for j in range(y.shape[1])]
    return float(np.nanmean(per)), per


def spearman(a, b):
    """Mean over findings of the rank correlation between two members' columns."""
    ra, rb = rankpct(a), rankpct(b)
    out = []
    for j in range(ra.shape[1]):
        x, y = ra[:, j], rb[:, j]
        sx, sy = x.std(), y.std()
        out.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan)
    return float(np.nanmean(out)), out


def paired_bootstrap(y, base, cand, n_boot=N_BOOT, seed=BOOT_SEED):
    """Resample STUDIES, not cells: the 58 are the unit the board would draw."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if (yy.sum(0) == 0).all() or (yy.sum(0) == len(yy)).all():
            continue
        d.append(macro_auc(yy, base[idx])[0] - macro_auc(yy, cand[idx])[0])
    d = np.asarray(d)
    return {
        "delta": float(np.nanmean(d)),
        "ci": (float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))),
        "p_better": float(np.nanmean(d > 0)),
        "n_boot": int(np.isfinite(d).sum()),
    }


def add_as_vote(composite_rank, candidate, n_prior):
    """One vote each, which is what the inference template does.

    `composite_rank` already carries `n_prior` pipelines. Two wrong forms were
    shipped before this one, and both are worth naming because the screen has
    to mirror the kernel or it scores a blend nobody runs:

    * `(rankpct(prior) + c) / 2` gives the newcomer HALF the ensemble --
      1/sqrt(2) influence on independent uniforms. E124 caught it.
    * `(n * rankpct(prior) + c) / (n + 1)` re-uniformises the prior, undoing
      the averaging that made n pipelines count as n, and lands the newcomer at
      1/sqrt(n + 3) instead. E124's own fix; E127 measured the overshoot.

    Leaving the prior un-re-ranked is the one that gives 1/sqrt(n + 1), which
    is one vote each.
    """
    return (n_prior * composite_rank + rankpct(candidate)) / (n_prior + 1)
