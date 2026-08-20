# Project status — everything, categorised by how well it is known

This file exists because the project accumulated 28 experiments, 31 generated
kernels and 62 tagged claims, and the single most expensive mistakes in it came
from **treating a weakly-known number as a well-known one**. So the organising
axis here is not topic. It is *evidence strength*.

Last updated 2026-08-19.

---

## 1. Results, by what actually established them

### A — measured on the leaderboard (the only ground truth)

| submission | score | note |
|---|---:|---|
| constant priors | 0.500 | the benchmark the efficiency metric uses |
| scanner metadata, no pixels | 0.531 | the bar the images must clear |
| **imaging, resnet34 2.5D, 192px, 1 fold** | **0.725** | the standing result |
| imaging, 288px, effective batch 4 | 0.668 | confounded run |
| imaging, 288px, effective batch 16 | 0.688 | the confound corrected; still behind 192px |

**Leaderboard position: 0.725.** Field top 0.952; top-200 cut 0.917.

### B — measured against expert labels out-of-fold (n = 46, CI ±0.05)

| system | gold macro AUC | 95% CI |
|---|---:|---|
| imaging, 192px, folds 1–4 pooled | **0.7264** | [0.672, 0.776] |

This is the only offline signal currently trusted, and §3 explains why the
obvious alternative is not.

### C — measured against expert labels on the 58, same convention

| labeler | macro AUC | abstain |
|---|---:|---:|
| lexicon | 0.7446 | 39.7% |
| LLM reader (Qwen2.5-7B) | 0.7421 | 48.7% |
| **union of the two** | **0.8145** | **30.2%** |

Paired bootstrap: union − lexicon **+0.0698**, CI [+0.041, +0.097]. LLM −
lexicon −0.0025, CI [−0.060, +0.049] — **not separated**.

### D — report-label CV only, and therefore weak evidence

| run | CV | gold (own fold, n≈12) |
|---|---:|---:|
| fold 0 (192px) | 0.7001 | — |
| fold 1 | 0.7434 | 0.754 |
| fold 2 | 0.7410 | 0.783 |
| fold 3 | 0.7260 | 0.687 |
| fold 4 | 0.7639 | 0.831 |
| 288px, effective batch 16 | 0.7282 | — |
| 288px + 30 more epochs | 0.7282 | — (no gain) |
| DINOv2 ViT-S/14, 34 epochs | 0.7041 | 0.644 |
| per-finding attention pooling | 0.7088 | 0.742 |

**Fold spread is 0.033** on one configuration, which is wider than most
differences this project has acted on. A single fold's gold subset (n≈12) has a
0.173 interval and settles nothing on its own.

### E — built and tested, never measured on anything

Rank-mean ensembling · fused labels wired into training (65.4% of slots
supervised against 50.6%) · per-finding confidence weights · inference timing
instrumentation. All are blocked behind the GPU budget, not behind doubt.

---

## 2. What is settled

- **4,407 studies, 58 with expert labels**, 12 binary findings, 819,635 slices,
  ten languages, no report text at test time.
- **No site column exists anywhere.** Folds group on a scanner fingerprint;
  random K-fold inflates macro AUC by **0.087**.
- **Axial fluid-sensitive series exist for 100% of studies**; all three planes
  for 90.6%.
- **A P100 fails outright**, not slowly. `machine_shape: "NvidiaTeslaT4"`.
- **Two separate GPU limits**: concurrency (2 sessions) and a **30 h weekly
  quota**. Both messages begin with "Maximum" and confusing them wasted real
  time. CPU is a separate allowance.
- **Efficiency prize**: `AUC/(Benchmark − maxAUC) + Runtime/32400`, minimised,
  so an extra hour costs **0.0502 AUC** at today's top.

---

## 3. What was believed and turned out to be wrong

Each of these was acted on before it was corrected. They are listed because the
pattern matters more than any individual entry.

| claim | status | what it cost |
|---|---|---|
| "imaging inverts the CV-to-LB gap" | **contradicted** — generalised from n=1 | a wrong mental model for a day |
| "the 288px model is genuinely worse" | **contradicted** — the run was confounded across five variables | nearly ended the highest-value line of work |
| "higher resolution is the answer" | **contradicted** — 288px is 0.037 behind on the board | two of the most expensive experiments |
| "report-label CV ranks models correctly" | **contradicted** — CV put 288px ahead by 0.028; the board put it 0.037 behind | the loss of the only cheap selection signal |
| "the 288px curve is still climbing" | **contradicted** — 30 more epochs peaked at 0.7280 and decayed to 0.706 | 3.59 GPU-hours |
| "gold-58 sits 0.044 below the board" (published) | **does not reproduce** — measured offset −0.001 | nothing; caught before use |

The common shape: **a small number of observations read as a trend.** The
countermeasure now in place is that every comparison is one-variable by
construction and a test asserts it.

---

## 4. Code, by what it is for

| area | files | lines | what it is |
|---|---:|---:|---|
| `src/` | 5 | 1,037 | labeler, fold grouping, label schema, **the pipeline manifest** |
| `kaggle/_templates/` | 7 | 1,800 | 3 kernel templates + 3 shared modules — **the only hand-edited kernel code** |
| `kaggle/*/run.py` | 31 | 17,087 | **generated**, never hand-edited; `--check` fails if they drift |
| `eda/` | 11 | 2,296 | verification, evaluation, label fusion, push queueing |
| `tests/` | 9 | 1,804 | **151 tests** |

Six kernels remain hand-written one-offs: the header scan, two baseline
submissions, single-model inference, CPU inference, and the cross-geometry
ensemble.

---

## 5. Claim ledger

`docs/FINDINGS.md` tags every claim: **49 `VERIFIED`, 6 `UNVERIFIED`,
6 `CONTRADICTED`, 1 `CORRECTED`.** `docs/EXPERIMENTS.md` holds 28 numbered runs
with what changed, the runtime, the result and what it meant.

---

## 6. Blocked, and on what

The **30-hour weekly GPU quota is exhausted**. Kaggle's API exposes neither the
remaining balance nor the reset time — only the account page does — so the reset
moment is `UNVERIFIED`.

Waiting on it, in the order they are worth doing:

1. **Train on the fused labels.** The teacher improved by +0.070 on gold; a
   0.725 imaging model came from a 0.769 teacher. Most likely to move the board.
2. **Submit the 4-fold rank-mean ensemble.** Same configuration, different
   splits, no hypothesis that can be wrong.
3. **Complete the gold pool** to n=58 by running fold 0 with a gold dump.
4. **Re-examine DINOv2** — 0.7041 at epoch 34, still not clearly converged.

CPU work is unaffected and is where anything further will happen until the
quota returns.
