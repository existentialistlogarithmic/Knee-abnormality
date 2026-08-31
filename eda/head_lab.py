#!/usr/bin/env python3
"""Train everything above the backbone, five-fold, in minutes on CPU.

Measured costs, on four CPU threads:

| | cost |
|---|---|
| fine-tuning the whole model, 4,407 studies x 24 epochs | **191 hours** |
| one frozen pass over the corpus, saved as embeddings | **2.2 hours, once** |
| five folds x 24 epochs of the 73,380 parameters above it | **2.6 minutes** |

So the question "does focal top-k pooling help" costs a coffee rather than a GPU
session, and the weekly GPU allowance is spent.

**What transfers and what does not.** A frozen backbone is not the fine-tuned
model that scored 0.725, so absolute numbers here do not predict the board. What
transfers is *comparisons above the backbone* — pooling and labels — because
those are exactly what is being trained. Every comparison below is therefore run
as a one-variable A/B and reported with a paired interval, not as a score.

    python eda/head_lab.py --embeddings artifacts/embed/embeddings.npy \
        --index artifacts/embed/embeddings_index.json --compare focal
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.report_schema import FINDINGS  # noqa: E402

BOOTSTRAP = 2000


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
    if n == 0 or n == len(y):
        return float("nan")
    return (ranks[pos].sum() - n * (n + 1) / 2) / (n * (len(y) - n))


def macro(expert, score):
    values = [auc((expert[:, i] > 0.5).astype(int), score[:, i])
              for i in range(len(FINDINGS))]
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")


def build_head(channels, focal_k, per_finding, seed, positional=0):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    n_out = len(FINDINGS)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            maps = n_out if per_finding else 1
            self.per_finding = per_finding
            self.attention = nn.Sequential(nn.Linear(channels, 128), nn.Tanh(),
                                           nn.Linear(128, maps))
            self.weight = nn.Parameter(torch.zeros(n_out, channels))
            nn.init.trunc_normal_(self.weight, std=0.02)
            self.bias = nn.Parameter(torch.zeros(n_out))
            self.focal_k = focal_k
            if focal_k:
                self.mix = nn.Parameter(torch.zeros(n_out))
            # A learned vector per TOKEN POSITION, which is the one thing this
            # architecture currently cannot see. Attention scores are computed
            # from embedding content alone and pooled with a sum, so the study
            # is an unordered bag of 60 slices: E050 measured the model exactly
            # invariant to reversing them, and to any permutation at all. That
            # discards two facts the tokens carry positionally — which of the
            # three planes a slice came from (0-19 sagittal, 20-39 coronal,
            # 40-59 axial) and where in its stack it sat, so that a finding on
            # three ADJACENT slices looks different from one on three scattered
            # ones.
            #
            # Initialised to exact zeros, so at step 0 this head IS the
            # baseline and every difference it later shows is something it
            # learned rather than something it started with.
            self.position = (nn.Parameter(torch.zeros(positional, channels))
                             if positional else None)

        def forward(self, embedded):
            if self.position is not None:
                embedded = embedded + self.position[:embedded.shape[1]]
            scores = self.attention(embedded).softmax(dim=1)
            if self.per_finding:
                pooled = torch.einsum("btf,btc->bfc", scores, embedded)
                averaged = (pooled * self.weight).sum(-1) + self.bias
            else:
                pooled = (embedded * scores).sum(dim=1)
                averaged = pooled @ self.weight.T + self.bias
            if not self.focal_k:
                return averaged
            per_slice = torch.einsum("btc,fc->btf", embedded, self.weight) + self.bias
            k = min(self.focal_k, per_slice.shape[1])
            strongest = per_slice.topk(k, dim=1).values.mean(dim=1)
            blend = torch.sigmoid(self.mix)
            return blend * averaged + (1 - blend) * strongest

    return Head()


def run_config(embeddings, targets, masks, expert, is_gold, splits, *,
               focal_k, per_finding, epochs, batch, lr, seed, label,
               positional=0, seeds=1):
    """Five folds, averaged over `seeds` restarts. Out-of-fold for every study.

    Averaging restarts is not a nicety. Measured on the focal top-k A/B against
    the public labels, one seed each: the focal arm landed on 0.7361, 0.7428,
    0.7433, 0.7419 — a range of 0.007 — while the baseline arm landed on 0.7106,
    0.7292, 0.6990, 0.7347, a range of 0.036. The DIFFERENCE therefore swung
    from +0.007 to +0.044 depending almost entirely on which baseline was drawn.

    A single-seed A/B on 58 gold studies is reading that swing as if it were the
    architecture. Averaging the out-of-fold predictions across restarts before
    scoring removes it from both arms, which is the difference between measuring
    a head and measuring an initialisation.
    """
    import torch
    import torch.nn as nn

    torch.set_num_threads(4)
    channels = embeddings.shape[2]
    accumulated = np.zeros((len(targets), len(FINDINGS)), np.float64)
    counts = np.zeros((len(targets), 1), np.float64)
    started = time.time()

    for restart in range(seeds):
      run_seed = seed + restart * 100
      out_of_fold = np.full((len(targets), len(FINDINGS)), np.nan, np.float32)
      for fold, (train_idx, val_idx) in enumerate(splits):
        head = build_head(channels, focal_k, per_finding, run_seed + fold, positional)
        optimiser = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        order = np.array(train_idx)
        for epoch in range(epochs):
            rng = np.random.default_rng(run_seed * 1000 + epoch)
            rng.shuffle(order)
            head.train()
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                e = torch.from_numpy(embeddings[idx].astype(np.float32))
                y = torch.from_numpy(targets[idx])
                m = torch.from_numpy(masks[idx])
                loss = (criterion(head(e), y) * m).sum() / m.sum().clamp(min=1.0)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
        head.eval()
        with torch.no_grad():
            for start in range(0, len(val_idx), 256):
                idx = np.array(val_idx[start:start + 256])
                e = torch.from_numpy(embeddings[idx].astype(np.float32))
                out_of_fold[idx] = torch.sigmoid(head(e)).numpy()
      scored = ~np.isnan(out_of_fold[:, 0])
      accumulated[scored] += out_of_fold[scored]
      counts[scored] += 1

    out_of_fold = np.divide(accumulated, counts, out=np.full_like(accumulated, np.nan),
                            where=counts > 0).astype(np.float32)
    gold = is_gold & ~np.isnan(out_of_fold[:, 0])
    score = macro(expert[gold], out_of_fold[gold])
    print(f"  {label:28s} gold n={int(gold.sum())}  macro {score:.4f}  "
          f"({seeds} seed{'s' if seeds > 1 else ''}, {time.time() - started:.0f}s)")
    return out_of_fold, gold


def paired(name_a, a, name_b, b, expert, gold, seed=0):
    rng = np.random.default_rng(seed)
    idx_all = np.flatnonzero(gold)
    deltas = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(idx_all), len(idx_all))
        rows = idx_all[pick]
        va, vb = macro(expert[rows], a[rows]), macro(expert[rows], b[rows])
        if va == va and vb == vb:
            deltas.append(va - vb)
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = macro(expert[idx_all], a[idx_all]) - macro(expert[idx_all], b[idx_all])
    verdict = ("A is better" if low > 0 else "B is better" if high < 0
               else "NOT SEPARATED — the interval contains zero")
    print(f"\n  {name_a} − {name_b}: {point:+.4f}  95% CI [{low:+.3f}, {high:+.3f}]")
    print(f"    {verdict}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--labels", default="artifacts/kaggle_dataset/soft_labels.parquet")
    parser.add_argument("--fused", default="artifacts/kaggle_dataset_fused/soft_labels.parquet")
    parser.add_argument("--headers", default="artifacts/kaggle_dataset/series_headers.parquet")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--compare",
                        choices=["focal", "pool", "labels", "position", "all"],
                        default="all")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1,
                        help="restarts per arm, averaged before scoring. One "
                             "seed measures an initialisation as much as a head")
    args = parser.parse_args(argv)

    index = json.loads(Path(args.index).read_text())
    embeddings = np.load(args.embeddings, mmap_mode="r")
    studies = index["studies"]
    print(f"{embeddings.shape[0]:,} studies x {embeddings.shape[1]} slices x "
          f"{embeddings.shape[2]} channels   backbone {index['backbone']}")

    from sklearn.model_selection import GroupKFold

    headers = pd.read_parquet(args.headers)
    frequency = pd.to_numeric(headers.get("imaging_frequency"), errors="coerce")
    headers = headers.assign(imaging_frequency_rounded=frequency.round(2))
    fields = ["manufacturer", "model_name", "software_versions", "field_strength",
              "imaging_frequency_rounded", "transmit_coil"]
    headers["fingerprint"] = headers[fields].astype("string").fillna("?").agg("|".join, axis=1)
    groups = headers.groupby("StudyInstanceUID").fingerprint.agg(
        lambda s: s.value_counts().index[0])
    group_values = groups.loc[studies].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(np.zeros(len(studies)), None, group_values))
    print(f"{len(set(group_values))} scanner groups")

    train_csv = pd.read_csv(args.train).set_index("StudyInstanceUID")
    gold_ids = set(train_csv[train_csv[FINDINGS].notna().all(axis=1)].index)
    is_gold = np.array([s in gold_ids for s in studies])
    expert = np.zeros((len(studies), len(FINDINGS)), np.float32)
    for row, study in enumerate(studies):
        if is_gold[row]:
            expert[row] = train_csv.loc[study, FINDINGS].to_numpy(dtype=np.float32)
    print(f"{int(is_gold.sum())} gold studies")

    def targets_from(path):
        soft = pd.read_parquet(path).set_index("StudyInstanceUID")
        t = soft.loc[studies, FINDINGS].astype(float).to_numpy()
        channels = soft.loc[studies, [f"{f}__channel" for f in FINDINGS]].to_numpy()
        m = np.ones_like(t, dtype=np.float32)
        m[channels == "absent"] = 0.0
        columns = [f"{f}__weight" for f in FINDINGS]
        if all(c in soft.columns for c in columns):
            m = m * np.nan_to_num(soft.loc[studies, columns].to_numpy(np.float32), nan=1.0)
        t = np.nan_to_num(t, nan=0.0).astype(np.float32)
        # gold studies are supervised by expert labels, exactly as training does
        t[is_gold] = expert[is_gold]
        m[is_gold] = 1.0
        return t, m

    common = {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "seed": args.seed, "seeds": args.seeds}
    base_t, base_m = targets_from(args.labels)

    if args.compare in ("focal", "all"):
        print("\n=== focal top-k pooling, one variable ===")
        a, gold = run_config(embeddings, base_t, base_m, expert, is_gold, splits,
                             focal_k=3, per_finding=False, label="FOCAL_K=3", **common)
        b, _ = run_config(embeddings, base_t, base_m, expert, is_gold, splits,
                          focal_k=0, per_finding=False, label="baseline", **common)
        paired("FOCAL_K=3", a, "baseline ", b, expert, gold)

    if args.compare in ("pool", "all"):
        print("\n=== per-finding attention maps, one variable ===")
        a, gold = run_config(embeddings, base_t, base_m, expert, is_gold, splits,
                             focal_k=0, per_finding=True, label="per-finding maps", **common)
        b, _ = run_config(embeddings, base_t, base_m, expert, is_gold, splits,
                          focal_k=0, per_finding=False, label="baseline", **common)
        paired("per-finding", a, "baseline   ", b, expert, gold)

    if args.compare in ("position", "all"):
        print("\n=== slice position, one variable ===")
        tokens = embeddings.shape[1]
        a, gold = run_config(embeddings, base_t, base_m, expert, is_gold, splits,
                             focal_k=0, per_finding=False, positional=tokens,
                             label=f"positional ({tokens} tokens)", **common)
        b, _ = run_config(embeddings, base_t, base_m, expert, is_gold, splits,
                          focal_k=0, per_finding=False, label="baseline", **common)
        paired("positional", a, "baseline  ", b, expert, gold)

    if args.compare in ("labels", "all") and Path(args.fused).exists():
        print("\n=== fused labels versus lexicon labels, one variable ===")
        fused_t, fused_m = targets_from(args.fused)
        a, gold = run_config(embeddings, fused_t, fused_m, expert, is_gold, splits,
                             focal_k=0, per_finding=False, label="fused labels", **common)
        b, _ = run_config(embeddings, base_t, base_m, expert, is_gold, splits,
                          focal_k=0, per_finding=False, label="lexicon labels", **common)
        paired("fused  ", a, "lexicon", b, expert, gold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
