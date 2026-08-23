# What is left, what it is worth, and what 0.95 actually requires

Standing: **0.846 on the leaderboard** (2026-08-22, fused-label 5-fold, E034),
up from 0.725. Gold OOF for that model is 0.7918, so **gold understates the
board by 0.054 here and by 0.005 on the previous model — it ranks, it does not
forecast.**
Leaderboard top **0.952**. 1,866 teams. Final submission **2026-10-22**.

Budget from 2026-08-20: **63 days, ~270 GPU-hours** at the 30 h weekly quota,
plus unlimited CPU and 5 submissions a day.

| affordable in that budget | cost each | count |
|---|---:|---:|
| 5-fold resnet34 @192px | 6.7 h | 40 |
| 5-fold DINOv2 ViT-S, 40 epochs | 23 h | 11 |
| one submission run | 0.8 h | plenty |

---

## 1. Ready now, costs nothing but CPU

| | status |
|---|---|
| fused labels (lexicon ∪ LLM) | **DONE, 5 folds (E032): +0.0717 gold at n=58, CI [+0.042, +0.103] — separated.** Awaiting a submission |
| focal top-k pooling | **measured, no effect**: +0.006 [−0.041, +0.051] (E030) |
| per-finding attention maps | **still not separated at n=58**: +0.039 [−0.009, +0.090] (E030) |
| rank-mean ensembling | shipped |
| gold OOF at n=58, tracks the board to **+0.005** | working |
| frozen-embedding rig: 5-fold A/B in **~8 min on CPU** | `knee-embed` **finished**; three A/Bs run (E030) |

**The immediate move is not a GPU run.** `knee-embed` has finished and the three
open questions above the backbone are answered (E030): the fused labels win, and
neither focal top-k nor per-finding maps separates from zero. GPU goes only to
configurations that already won on the rig — which now means the fused labels,
and not the two pooling changes.

## 2. The ranked plan

### Phase A — labels (CPU only, no quota) — **narrowed by E035**
"The teacher bounds everything" was true at 0.769 teacher / 0.725 model. It is
no longer: the fused model scores **+0.0126 above its own teacher** at the macro
(not separated, CI [−0.042, +0.066]) and beats it outright on **Fracture
+0.239** and **Baker's +0.165**, where the reports barely carry the finding and
the pixels do.

So a third reader is no longer a general lever — it is a **targeted** one, and
E035 says exactly where:

| finding | teacher | coverage | why a reader helps |
|---|---:|---:|---|
| **Synovitis** | **0.520 — chance** | 36% | reports do not carry it at all |
| Lateral OA | 0.708 | 47% | thin coverage |
| PF OA | 0.765 | 66% | thin coverage |

Synovitis is the model's second-weakest finding (0.616) and its teacher is at
chance. No amount of imaging work fixes that one. ~4 h of CPU, no quota.

### Phase B — decide the architecture on the rig (CPU, minutes) — **done**
`eda/head_lab.py` answered focal-k, per-finding pooling and fused-vs-lexicon as
one-variable paired A/Bs, in 8.5 min of CPU (E030). **Only the labels won**;
both pooling changes came back unseparated, and the focal-k "+0.060" this file
previously carried was a *headroom* figure, never a measured gain. The rig stays
the gate: nothing goes to GPU that has not won here first.

### Phase C — the backbone — **RUN, AND IT FAILED (E036)**
This phase claimed DINOv2 "had not converged — the curve was still climbing when
the clock stopped". Run properly to 40 epochs on fused labels, it **peaks at
epoch 23 and decays**: fold 0 loses 0.019 of val and 0.072 of gold by epoch 39.
The "still climbing" reading was noise on a truncated run.

DINOv2 is **0.074 behind resnet34** on gold (0.7151 vs 0.7888), and blending the
two families is worth **+0.0022, CI [−0.012, +0.016] — not separated**. Cost:
~20 GPU-hours, the most expensive error in this project's log.

**Do not restart this without a new reason.** A different ViT, or RadImageNet
pretraining, is a different claim and would need its own evidence.

### Phase D — ensemble (≈100 GPU-h) — **blocked by Phase C's failure**
Three independent families — a CNN, a ViT, and one more — five folds each, rank
blended. Published gain **+0.02 to +0.05**. Decode once, score N times, so it
costs almost nothing at inference.

**The cheap version of this does not work and has been tested (E033).** Blending
the fused and lexicon 5-folds — two label sets on one backbone — is *worse than
the fused model alone at every weight*, and the curve rises monotonically toward
"use none of the lexicon model". Blending needs members that are comparably
strong and decorrelated; a strictly weaker model trained on a subset of the same
information is neither. **Phase D requires a second architecture, not a second
label set**, which means Phase C has to come first.

### Phase E — spend submissions as confirmation
Gold OOF estimates the board to within 0.005. Submissions confirm; they do not
explore.

### Where that lands
Compounding the measured and published values from 0.725 gave roughly
**0.82–0.88**. **The board reached 0.846 on 2026-08-22 from the labels and
ensembling alone**, i.e. inside that range on the first two levers, with Phase C
and Phase D untouched. The range was not wrong, but it was not conservative
either — it is now the *floor* of what the remaining levers start from.

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

---

## 4. What 0.94 actually requires, as arithmetic

The metric is a **macro average over 12 findings**, so a target score is not a
single goal — it is a **floor every finding has to clear**. Standing system:
board 0.846, gold macro 0.7918 (E032), gold-to-board offset +0.054 (E038).

**The decisive fact: the weak findings cap everything.**

| if the strongest findings were PERFECT (1.000) and the weakest never moved | macro caps at | board |
|---|---:|---:|
| top 8 perfect, weakest 4 unchanged | 0.885 | **~0.939** |
| top 7 perfect, weakest 5 unchanged | 0.865 | ~0.919 |
| top 6 perfect, weakest 6 unchanged | 0.847 | ~0.901 |

**Board 0.94 is almost exactly the "eight findings at a literal 1.000" ceiling.**
No amount of work on the strong half reaches it. There is no subset of easy
wins — E026 said this for 0.90 and it is far more binding at 0.94.

**So the goal, stated correctly: no finding below ~0.87.**

| finding | now | needs | gap | teacher | kind of problem |
|---|---:|---:|---:|---:|---|
| Synovitis | 0.616 | 0.870 | **+0.254** | 0.520 | teacher below target too |
| MCL | 0.628 | 0.870 | **+0.242** | **0.884** | **modelling — the teacher already knows it** |
| PF OA | 0.658 | 0.870 | +0.212 | 0.765 | teacher below target too |
| Lateral OA | 0.721 | 0.870 | +0.149 | 0.708 | teacher below target too |
| Lateral Meniscus | 0.753 | 0.870 | +0.117 | 0.817 | teacher below target too |
| Medial Meniscus | 0.786 | 0.870 | +0.084 | **0.889** | **modelling — the teacher already knows it** |
| ACL | 0.812 | 0.870 | +0.058 | 0.863 | teacher below target too |

**Only two of the seven gaps are recoverable from supervision that already
exists** — MCL and Medial Meniscus, where the teacher scores above the target
and the model is throwing it away. Everything else needs the *teacher itself*
lifted past 0.87, and E035/E037 measured why that is hard: Synovitis's teacher
is at chance because the corpus barely reports it, and Lateral OA's speaks for
47% of studies.

### What this means for the plan

- **0.90** needs every finding at ~0.80. Reachable: it is mostly MCL, Medial
  Meniscus and PF OA, and two of those three are modelling losses.
- **0.94** needs every finding at ~0.87, including Synovitis, whose *labels* top
  out near 0.52. That is not a training-budget problem and no number of Colab
  folds solves it.
- **The only documented route to that band is §3** — pooled community
  checkpoints and label sets, properly credited. The strongest public notebook
  is a rank blend of twenty DINOv2 checkpoints plus a DINOv3 and two RadImageNet
  stages, crediting ten competitors by name. That is hundreds of GPU-hours of
  *other people's* compute and, more importantly, *other people's label sets*,
  which is exactly the constraint above.
