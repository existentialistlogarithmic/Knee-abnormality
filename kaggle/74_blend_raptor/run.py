"""Rank-blend two finished submissions. No model, no GPU, no training.

**This runs on CPU** and reads nothing but two `submission.csv` files. Every
member it blends has already been priced by the board, which is the whole point:
there is no offline instrument in the loop and therefore nothing for a leak to
get into. E083 is the reason that matters — the gold rig separated the distilled
teacher at +0.0221 with an interval excluding zero and the board came back
−0.013, because both lineages cut the same folds and the evaluation leaked
(E082). A blend of two submitted CSVs cannot repeat that mistake.

WHY A BLEND, AND WHY THIS PAIR. E048's rule, refined by E075: a union pays when
its members are COMPARABLE and of DIFFERENT KINDS, and imports errors otherwise.

| union | gap | different kind? | result |
|---|---|---|---|
| E023 lexicon ∪ LLM | 0.0025 | yes | **+0.070** |
| E064 five folds ∪ five reseeds | ~0 | **no** | 0.923, exactly the five |
| E069 labels ∪ own OOF | 0.005 | yes | separated offline, **−0.013 on the board** |
| this pair | **0.002** | **yes** | unpriced |

The reseed union failed because a second draw of one configuration makes
correlated errors — the average inherits the weaker member's bias and cancels
nothing. These two members share no architecture, no geometry, no slice
selection and no label source, so they fail on different studies, which is the
only mechanism by which averaging has ever helped here.

FIFTY-FIFTY, AND NOTHING FITTED. The weight is not tuned and there is no
per-finding weighting. A weight chosen on 58 studies is a free parameter fitted
to 58 studies — `dataset-metadata.fused.json` rejects that by name, E048
declined it, E069 declined it with the curve printed in full, and E081 declined
it again when the fitted optimum would have turned a null into a headline. The
midpoint is the only weight that requires no justification.

RANKS, NOT PROBABILITIES. The metric is per-finding AUC, which depends only on
the ordering within a column. The two members are calibrated differently — one
emits sigmoid outputs from a 2.5D resnet, the other from a wide-dense stack
model — so averaging raw scores would let whichever member happens to use more
of [0,1] dominate the ordering. Ranking each column first makes the average
depend on what the metric actually reads and nothing else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# A 50/50 rank blend of the public CC0 Raptor model and this
# project's full-fit ensemble. No model, no GPU, no training —
# two finished submission.csv files and a rank average.
#
# ATTRIBUTION, required by the competition rules and by E043:
# the Raptor arm is Dread Development's, reproduced from the
# public notebook `knee-mri-twelve-findings-from-a-single-model`
# with weights from `dreaddevelopment/raptor-knee-widedense`,
# licensed CC0-1.0. The rules permit publicly shared code and
# data as inputs for every participant, with credit.
#
# Every member here has already been priced by the board, so no
# offline instrument is in the loop and there is nothing for a
# leak to get into. That is deliberate: E083 is three days old and
# cost a board point to the gold rig separating something at
# +0.0221 that the board scored at −0.013.
#
# The weight is 0.5 and nothing is fitted. E048, E069 and E081
# each declined a weight chosen on 58 studies; this declines it a
# fourth time. The blend ranks each finding before averaging
# because the metric reads ordering only, and the two members are
# calibrated differently enough that raw averaging would let the
# wider-spread member decide the order.
#
MEMBERS_EXPECTED = 2

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]


# --------------------------------------------------------------------------- #
# from kaggle/_templates/_shared/discovery.py
# --------------------------------------------------------------------------- #
def find_all_markers(pattern: str, max_depth: int = 4) -> list[Path]:
    """Every mounted directory containing a file matching `pattern`.

    The cache is built as four shard kernels and mounted as four separate
    inputs. Finding only the first would silently train on a quarter of the
    data at full apparent success — the worst kind of bug, because the loss
    curve would look fine.
    """
    found = []
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        if any(e.is_file() and e.match(pattern) for e in entries):
            found.append(directory)
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRECTORIES:
                frontier.append((entry, depth + 1))
    return found


def main() -> int:
    # Every mounted kernel output that contains a submission. Discovered rather
    # than hardcoded: E078 cost two runs because a glob assumed where Kaggle
    # mounts inputs, and then a test pinned the assumption in place.
    directories = find_all_markers("submission.csv")
    paths = sorted({d / "submission.csv" for d in directories})
    print(f"submissions mounted: {len(paths)}")
    for path in paths:
        print(f"  {path}")

    # A kernel that errored or never ran mounts as an EMPTY directory rather
    # than failing, so a short blend is silent unless something counts. E078
    # introduced this guard for the ensemble members; the same failure mode
    # applies here and is worse, because a "blend" of one member is just its
    # first member wearing the blend's name and would score exactly what that
    # member already scored.
    if len(paths) != MEMBERS_EXPECTED:
        raise SystemExit(
            f"expected {MEMBERS_EXPECTED} submissions to blend, found "
            f"{len(paths)} — refusing to write a submission that silently "
            "blends the wrong number of members")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = [c for c in ["StudyInstanceUID", *FINDINGS]
                   if c not in frame.columns]
        if missing:
            raise SystemExit(f"{path} is missing columns: {missing}")
        frames.append(frame.set_index("StudyInstanceUID"))

    # Every member must cover exactly the same studies. A member scored on a
    # subset would be averaged against nothing on the rest, which silently
    # reweights those rows to a single model without changing the row count.
    index = frames[0].index
    for path, frame in zip(paths, frames, strict=True):
        if set(frame.index) != set(index):
            raise SystemExit(
                f"{path} covers {len(frame.index):,} studies against "
                f"{len(index):,} in the first member — refusing to blend "
                "members that disagree about which studies exist")
        if frame.index.duplicated().any():
            raise SystemExit(f"{path} lists a study more than once")

    print(f"{len(index):,} studies, {len(frames)} members, {len(FINDINGS)} findings")

    # Rank within each finding, per member, then average the ranks. Normalised
    # to [0,1] so the output is readable, which changes no ordering.
    blended = pd.DataFrame(index=index, columns=FINDINGS, dtype=float)
    for finding in FINDINGS:
        ranks = [frame.loc[index, finding].rank(method="average").to_numpy()
                 for frame in frames]
        mean_rank = np.mean(ranks, axis=0)
        blended[finding] = (mean_rank - 1) / max(len(index) - 1, 1)

        # How far apart the members were on this finding. Spearman near 1.0
        # means they agree and the blend cannot help; the whole premise is that
        # they disagree in a structured way, so this is the number that says
        # whether the premise held for each finding.
        if len(frames) == 2:
            a, b = ranks
            rho = np.corrcoef(a, b)[0, 1]
            print(f"  {finding:18s} member agreement (Spearman) {rho:.3f}")

    out = Path("/kaggle/working/submission.csv")
    blended.reset_index().to_csv(out, index=False)
    print(f"wrote {out} — {len(blended):,} rows")

    # Read it back. A submission that cannot be parsed, carries a NaN or drifts
    # out of [0,1] fails at scoring time rather than here, and this competition
    # accepts submissions only from notebooks, so a failed run costs a human
    # round trip rather than a retry.
    check = pd.read_csv(out)
    values = check[FINDINGS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise SystemExit("submission carries a non-finite value")
    if values.min() < 0.0 or values.max() > 1.0:
        raise SystemExit(f"submission out of range: [{values.min()}, {values.max()}]")
    print(f"verified: {len(check):,} rows, range "
          f"[{values.min():.4f}, {values.max():.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
