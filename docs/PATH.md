# What is left, what it is worth, and what 0.95 actually requires

Standing: **0.725 on the leaderboard**, 0.7201 gold out-of-fold at n=58.
Leaderboard top **0.952**. 1,866 teams. Final submission **2026-10-22**.

Budget from 2026-08-20: **63 days, ~270 GPU-hours** at the 30 h weekly quota,
plus unlimited CPU and 2 submissions a day.

| affordable in that budget | cost each | count |
|---|---:|---:|
| 5-fold resnet34 @192px | 6.7 h | 40 |
| 5-fold DINOv2 ViT-S, 40 epochs | 23 h | 11 |
| one submission run | 0.8 h | plenty |

---

## 1. Ready now, costs nothing but CPU

| | status |
|---|---|
| fused labels (lexicon ∪ LLM), **+0.070 teacher on gold** | built, tested, not yet trained on |
| focal top-k pooling, **+0.060 of measured headroom** | built, tested, A/B queued |
| per-finding attention maps | built, +0.014 at n=12 — not separated |
| rank-mean ensembling | shipped |
| gold OOF at n=58, tracks the board to **+0.005** | working |
| frozen-embedding rig: 5-fold A/B in **2.6 min on CPU** | `knee-embed` running now |

**The immediate move is not a GPU run.** Once `knee-embed` finishes, every open
architecture and label question is answerable in minutes on CPU. Do that first,
then spend GPU only on configurations that already won on the rig.

## 2. The ranked plan

### Phase A — labels (CPU only, no quota)
The teacher bounds everything: a 0.725 model came from a 0.769 teacher. The
union of two readers is 0.8145. Leading public systems use **three independent
report readers**, not two. A second open-weights model reading into the same
closed ladder is ~4 h of CPU and the union rule already exists.

### Phase B — decide the architecture on the rig (CPU, minutes)
`eda/head_lab.py` answers focal-k, per-finding pooling and fused-vs-lexicon as
one-variable paired A/Bs. Nothing here needs quota. Only what wins goes to GPU.

### Phase C — the backbone (≈50 GPU-h)
DINOv2 reached 0.7041 at epoch 34 **and had not converged** — the curve was
still climbing when the clock stopped. Published evidence puts adaptation at
**+0.09**, roughly five times what resolution is worth. Run it to convergence,
five folds. This is the single largest published lever and it has not had a fair
run here.

### Phase D — ensemble (≈100 GPU-h)
Three independent families — a CNN, a ViT, and one more — five folds each, rank
blended. Published gain **+0.02 to +0.05**. Decode once, score N times, so it
costs almost nothing at inference.

### Phase E — spend submissions as confirmation
Gold OOF estimates the board to within 0.005. Submissions confirm; they do not
explore.

### Where that lands
Compounding the measured and published values from 0.725 gives roughly
**0.82–0.88**. That is an honest range, not a promise, and it assumes each lever
delivers near the top of its published band.

---

## 3. About 0.95, plainly

**0.952 is rank 1 of 1,866 teams.** Asking how the code reaches 0.95 is asking
how it wins the competition outright, and no plan should pretend otherwise.

Two things are true about the systems up there, both from reading their own
published notebooks:

1. **They are aggregations, not single models.** The strongest public notebook
   is a rank blend of **twenty DINOv2 checkpoints plus a DINOv3 ViT plus two
   RadImageNet stages**, and its author credits ten other competitors by name
   for the checkpoints, the label sets and the aggregation ideas. That is
   hundreds of GPU-hours of *pooled community* compute, not one team's 270.
2. **Their advantage starts at the labels.** Three independent report readers,
   cross-checked against each other. This project has two, and the second one
   only exists as of today.

The rules permit building on that: *"It's okay to share code if made available
to all Participants on the forums."* Publicly shared weights and label sets are
legitimate inputs, with attribution. That is the actual mechanism by which the
top of this leaderboard is at 0.95, and a solo run from 0.725 to 0.95 in nine
weeks on 30 GPU-hours a week is not a realistic alternative to it.

**So the honest split:**

- **0.82–0.88 is a plan.** Every step is measured or published, and the budget
  covers it.
- **0.90 is a stretch** that needs every lever near the top of its band.
- **0.95 needs the community-aggregation route** — publicly shared checkpoints
  and label sets, properly credited — or a genuinely novel result this project
  has not found yet.

The one thing that will not get there is optimism about the numbers. This
project has already overturned six of its own confident claims, and every one
was caught by a measurement rather than by reasoning.
