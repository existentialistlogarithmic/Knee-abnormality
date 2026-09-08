#!/usr/bin/env python3
"""Screen publicly shared checkpoints before spending anything on them.

    python eda/survey_public_checkpoints.py --checkpoints artifacts/public/*.pt

**Why a screen exists at all.** The top of this leaderboard is not built from
one team's compute. Read the public notebooks and they say so themselves: the
strongest ones are rank blends of twenty-plus checkpoints pooled from a dozen
competitors, each crediting the others by name. The rules permit it — *"It's
okay to share code if made available to all Participants on the forums"* — so
publicly shared weights are legitimate inputs, with attribution.

**But four consecutive unions have paid nothing here**, and they failed for one
reason. E048 states it:

    a union pays when its members are COMPARABLE, and imports errors when
    they are not.

E023 united two readers 0.002 apart and gained **+0.070**. E033, E039, E046 and
E048 each added a member 0.03-0.06 behind the incumbent and gained +0.0046,
+0.0022, +0.0036, +0.0027 — none separated. So the question to ask of any new
public asset is not "is it good" but "**is it close to ours**", and that is a
question the file itself can often answer for free.

**Many checkpoints are self-describing.** `shingo257/rsna-knee-trained-
checkpoints-v1` stores `backbone`, `image_size`, `num_slices`, `fold`, `epoch`
and the author's own `auc_gold` alongside the weights. Reading those costs one
download and no inference, which matters because actually *running* a foreign
checkpoint means first rebuilding the cache at its geometry — hours of work that
should follow a positive screen rather than precede it. E039's rule: a probe
must not be a job that costs something if it succeeds.

**A self-reported number is not a measurement.** It is the author's, computed on
the author's split, and it may include gold studies their model trained on. It
is used here only to decide whether OUR measurement is worth building, and the
output says so on every line.

Reads no patient data. Prints aggregates only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The incumbent, from E044/E046. Per-fold is the honest comparator for a
# per-fold number: our five-fold POOL is 0.8980, but one fold of ours scores
# 0.8477 on its own gold holdout, and comparing a foreign single fold against
# our pooled five would understate it by the width of the ensembling effect.
INCUMBENT_POOLED = 0.8980
INCUMBENT_PER_FOLD = 0.8477

# E048's band. Inside it, a member is comparable and a union is worth measuring;
# outside it, four experiments say the union imports errors.
COMPARABLE_WITHIN = 0.02
KNOWN_DEAD_BEYOND = 0.03

METADATA_KEYS = ("backbone", "image_size", "num_slices", "fold", "epoch",
                 "auc_gold", "auc_all", "auc", "cv", "score")

# The same six facts get written under different names by different authors, and
# reading only one spelling makes the screen report "cannot be priced for free"
# about a file that states everything. That is a FALSE NEGATIVE on exactly the
# assets this screen exists to find, and it happened: `raptor_ft_coatnet_v4_full.pt`
# carries arch, res, gold_auc and a per-finding AUC table, and the screen said it
# carried nothing because it spells them differently from shingo257's family.
# Any new spelling met in the wild belongs here, not in a special case downstream.
ALIASES = {
    "arch": "backbone",
    "model_name": "backbone",
    "encoder": "backbone",
    "res": "image_size",
    "img_size": "image_size",
    "size": "image_size",
    "n_slices": "num_slices",
    "slices": "num_slices",
    "gold_auc": "auc_gold",
    "auc_gold_oof": "auc_gold",
    "val_auc": "auc",
}

# Not a scalar fact, so it is collected separately: a dict of per-finding AUCs.
PER_FINDING_KEYS = ("aucs", "auc_per_class", "per_finding")


def describe(path: Path) -> dict:
    """Everything a checkpoint says about itself, without building the model."""
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        return {"file": path.name, "note": "not a dict; no self-description"}
    found = {"file": path.name}
    for key in (*METADATA_KEYS, *ALIASES):
        if key in blob and not hasattr(blob[key], "shape"):
            found[ALIASES.get(key, key)] = blob[key]
    for key in PER_FINDING_KEYS:
        if isinstance(blob.get(key), dict):
            found["aucs"] = blob[key]
            break
    state = next((blob[k] for k in ("model_state_dict", "model", "state_dict")
                  if isinstance(blob.get(k), dict)), None)
    found["tensors"] = len(state) if state else 0
    return found


def verdict(gap: float | None) -> str:
    if gap is None:
        return "UNKNOWN — no self-reported score; only our own run can price it"
    if gap <= COMPARABLE_WITHIN:
        return "COMPARABLE — worth measuring properly (E048's band)"
    if gap <= KNOWN_DEAD_BEYOND:
        return "MARGINAL — at the edge of the band four unions have died in"
    return "BEHIND — four unions with a member this far back all paid nothing"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--per-fold", action="store_true", default=True,
                        help="compare against our per-fold gold (0.8477), which "
                             "is the like-for-like comparator for a single "
                             "foreign fold")
    parser.add_argument("--pooled", dest="per_fold", action="store_false",
                        help="compare against our five-fold pool (0.8980) "
                             "instead; only honest for a pooled foreign score")
    args = parser.parse_args(argv)

    ours = INCUMBENT_PER_FOLD if args.per_fold else INCUMBENT_POOLED
    scope = "per-fold" if args.per_fold else "pooled"
    print(f"incumbent ({scope}): {ours:.4f}\n")

    rows = [describe(Path(p)) for p in sorted(args.checkpoints)]
    print(f"{'file':32s} {'backbone':16s} {'px':>5s} {'sl':>4s} {'fold':>5s} "
          f"{'their auc_gold':>15s}")
    scores = []
    for row in rows:
        score = row.get("auc_gold", row.get("auc", row.get("score")))
        # A per-finding table is a stronger description than the macro alone:
        # it says WHERE a foreign member would help, which is the only thing
        # E048's comparability band cannot tell you.
        row["per_finding"] = row.get("aucs")
        if isinstance(score, (int, float)):
            scores.append(float(score))
        print(f"{str(row.get('file'))[:32]:32s} {str(row.get('backbone', '?'))[:16]:16s} "
              f"{str(row.get('image_size', '?')):>5s} "
              f"{str(row.get('num_slices', '?')):>4s} "
              f"{str(row.get('fold', '?')):>5s} "
              f"{(f'{score:.4f}' if isinstance(score, (int, float)) else '—'):>15s}")

    if not scores:
        print("\nno self-reported scores found — this family cannot be screened "
              "for free, and pricing it means rebuilding the cache at its "
              "geometry first")
        return 0

    mean = sum(scores) / len(scores)
    gap = ours - mean
    print(f"\nmean self-reported gold over {len(scores)} checkpoint(s): {mean:.4f}")
    print(f"gap to ours ({scope} {ours:.4f}): {gap:+.4f}")
    print(f"\n  {verdict(abs(gap) if gap > 0 else 0.0)}")
    if gap < 0:
        print("  NOTE: it reports AHEAD of ours. That is the first time a public "
              "family has,\n  and it is still the author's number on the author's "
              "split — measure it before believing it.")

    print("\nEVERY NUMBER ABOVE IS SELF-REPORTED BY THE CHECKPOINT'S AUTHOR.")
    print("It decides whether to build our own measurement, and nothing else.")
    print("Before mounting anything: confirm the licence is CC0 or Apache "
          "(E043),\nand never mount a consolidation with `not-declared` files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
