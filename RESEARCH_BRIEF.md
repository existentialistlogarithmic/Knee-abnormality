# RSNA Knee Abnormality Detection — research brief

**Written 2026-09-21. Everything here is measured in this project unless marked
as someone else's self-report. Numbers have dates because several decay.**

We are stuck at **0.940** and our last attempt scored **0.921 / 0.923**. We want
ideas that survive the constraints in §4 — not a list of standard Kaggle advice,
most of which we have already measured and closed (§3).

---

## 1. The task and the numbers

- RSNA knee MRI, **12 binary findings**, metric **macro ROC-AUC** (threshold-free).
- **Code competition**: notebook-only submission, hidden test ~1,300 studies,
  9 h runtime cap, no internet at inference, 5 submissions/day.
- **Our board: 0.940.** Top of board **0.957**. ~3,800 teams.
- Public forks of the public baseline cluster at **0.936–0.939**, so our
  independently built system is roughly at the level of the free public notebook.
- **The wall**: ~837 teams at ≥0.938, ~706 at ≥0.940, but only **81 at ≥0.945**.
  The cliff is around 0.943.

## 2. What we ship

A rank-mean blend of two arms, 50/50:

| arm | what it is | board alone |
|---|---|---:|
| CoAtNet | 4 arms from **3 distinct public CC0 checkpoints** (`coatnet_rmlp_2_rw_384`), at the authors' published weights 0.55/0.20/0.15/0.10, 336–384 px, 44–64 slices | **0.932** |
| v1 (ours) | full-fit 2.5D resnet34s, 3 planes × 20 slices at 192 px / 0.6 mm, per-finding attention pooling | **0.926** |

Blended: **0.940**. The blend gain (+0.008) comes from **disagreement**: the two
arms correlate **0.542**, while the four CoAtNet arms correlate 0.905–0.986 with
each other.

## 3. THE ANOMALY WE MOST WANT EXPLAINED

We added two new v1 members from new backbone families (`resnext50_32x4d`,
`regnet_y_3_2gf`) to the five resnet34s and one resnet50, giving **8 v1 members**,
changing nothing else. The board returned **0.921 and 0.923**.

**That is below BOTH constituent arms** (v1 alone 0.926, CoAtNet alone 0.932).
A rank-mean blend of two arms scoring below the worse of them is not "weak new
members" — it is the signature of something broken. **−0.017 against a ±0.003
reseed floor.**

Facts around it, all verified in the stub log:
- `v1 members 8 | distinct weight fingerprints 8` — eight distinct checkpoints
  actually loaded, not one file counted eight times.
- `fallbacks 0/3` on every arm; `submission.csv rows=3`; runtime 2.62 h of 9 h.
- The v1 arm rank-averages its members **uniformly**, then the two arms are
  rank-meaned 50/50.
- The two new members trained to completion, export at epoch 20, loss decreasing
  monotonically. Their training AUCs are **memorisation** (full fit on all 4,407
  studies including our 58 expert-labelled ones), so they carry no information
  about generalisation.
- **Open question we cannot resolve from the API**: the two low submissions carry
  **no version description**, while our 0.940 reads `Notebook
  knee-infer-raptorv1 | Version 6`. We are not certain the 8-member version is
  what was scored.

**Questions we want answered**: what mechanism makes a uniform rank-average of 8
members score below the same arm with 6? Is uniform member averaging the wrong
aggregation once members differ in quality? Should a member be weighted by
anything measurable when we have no honest validation set for full-fit models?

## 4. THE CONSTRAINT THAT KILLS MOST ADVICE: we have no trustworthy validation

This is the crux. Three instruments, each defective:

| instrument | size | defect |
|---|---:|---|
| gold-58 (expert labels) | 58 | ±0.0153 absolute, ±0.006 paired. Cannot fit 12 parameters. Fitting anything on it and reporting on it is circular — this destroyed one of our experiments. |
| report labels | 4,349 | **The CoAtNet checkpoints were trained on these studies**, so their predictions here are in-sample. Anything fitted against this arbiter reads memorisation. |
| the board | ~1,300 | ±0.003 reseed floor, 5/day. The only honest judge, and low-resolution. |

**Our full-fit members train on all 58 gold studies, so they cannot be scored
offline at all.** Any proposal requiring a validation signal must say which of
these three it uses and why that one is not fatal.

## 5. CLOSED ON MEASUREMENT — please do not propose these

| route | result |
|---|---|
| Higher resolution (288 px/0.40 mm vs 192 px/0.60) | **−0.088** on 881 held-out studies, same fold/teacher. Worse with a *better* teacher, which kills the "label noise punished capacity" explanation. |
| Per-finding blend weights | Derived leakage-free, +0.0023 on held-out gold, P(better) 0.897 → board returned **exactly the same 0.940**. |
| Blend mixing weight (coat vs v1) | 0.40 vs 0.50 → **0.938 both times**. Inert. |
| Re-mixing the 4 CoAtNet arm weights | closed |
| 6 more public CC0 checkpoints from the same author | All scored **below** every incumbent on gold-58 (0.906→0.884 vs 0.912–0.920). Both parameter-free weightings **lost**. |
| TTA (slice reversal, span jitter) | closed |
| Public label sets | Three of four newer sets were literal answer keys (gold macro 1.0000). Ours is the best admissible. |
| Synovitis specifically | **Label ceiling**: our labels score 0.790 on it, our model 0.779. Control: fracture labels 0.793, model 0.885. |
| A second seed of an existing member | **+0.000** on the board |
| AutoML / hyperparameter search | Rejected on §4: no trustworthy signal to search against. |

**What HAS moved the board**: adding *members* — resnet34→resnet50 (a depth
change) gave **+0.002**, reproduced at two different blend weights. And mounting
the public CoAtNet checkpoints at all (0.926 → 0.932 → 0.938).

## 6. Operational facts that bound any proposal

- **~30 GPU-h/week** on 2×T4 (16 GB). One resnet34 full fit ≈ 1.5 h; resnet50 ≈
  3.5 h. **20.3 h left this week.**
- **Inference budget is NOT the constraint**: we use 2.62 h of a 9 h cap. ~6.4 h
  idle. Anything that spends inference rather than training is cheap for us.
- **Three depthwise-separable architectures have host-OOM-killed** at our
  geometry (`convnext_tiny`, `tf_efficientnet_b0`, `shufflenet_v2_x1_0`), while
  four dense/grouped ones trained fine (resnet34/50, resnext50, regnet_y_3_2gf).
  Correlation over seven runs, **no mechanism established**. Host RAM, not CUDA.
- Everything mounted must be **CC0 or otherwise licence-clear**; we audit this.
- Final submission **2026-10-22**.

## 7. What we are asking for

1. **A mechanism for §3.** Why would 8 uniformly-averaged members underperform 6?
2. **Aggregation alternatives** to uniform rank-mean that need no fitted
   parameters and no validation set — we can only afford rules that are arguments
   rather than knobs.
3. **Anything that converts our ~6.4 idle inference hours into score**, given
   that more public checkpoints are closed (§5).
4. **How teams reach 0.957** on this task — specifically what the top of a
   12-label MRI leaderboard typically does that a 2-arm blend does not.
5. Honest assessment of whether **0.945 is reachable at all** from 0.940 with
   ~20 GPU-h/week and one month, or whether the remaining gap is compute.

Please distinguish clearly between (a) things that need a validation set we do
not have, (b) things testable directly on the board at 5 clicks/day, and (c)
things testable offline on 58 studies without circularity. Only (b) and (c) are
actionable for us.
