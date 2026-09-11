# Project status — everything, categorised by how well it is known

This file exists because the project accumulated 88 experiments, 90 generated
kernels and 58 tagged claims, and the single most expensive mistakes in it came
from **treating a weakly-known number as a well-known one**. So the organising
axis here is not topic. It is *evidence strength*.

Last updated 2026-09-07 (E087, and E086 still open).

---

## 1. Results, by what actually established them

### A — measured on the leaderboard (the only ground truth)

| submission | score | note |
|---|---:|---|
| constant priors | 0.500 | the benchmark the efficiency metric uses |
| scanner metadata, no pixels | 0.531 | the bar the images must clear |
| imaging, resnet34 2.5D, 192px, 1 fold | 0.725 | the first imaging model |
| imaging, 288px, effective batch 4 | 0.668 | confounded run |
| imaging, 288px, effective batch 16 | 0.688 | the confound corrected; still behind 192px |
| imaging, 192px, 5-fold, lexicon labels | 0.757 | the control (E036) |
| imaging, 192px, 5-fold, FUSED labels | 0.846 | E034 |
| imaging, 192px, 5-fold, PUBLIC CC0 labels | 0.923 | E045 |
| 5-fold PUBLIC + 5 reseeded folds (10 members) | 0.923 | **a second seed buys nothing** (E064) |
| 5-fold PUBLIC + one FULL-FIT member | 0.924 | E064 |
| 5-fold PUBLIC, DISTILLED teacher | 0.910 | **the offline rig had the sign wrong** (E083) |
| 5 folds + 5 full-fit (10 members) | 0.926 | equals pure full fit exactly (E087) |
| five FULL-FIT members, nothing else | 0.926 | E083 |
| CC0 CoAtNet, one borrowed model | 0.928 | E091/E092 |
| **CC0 CoAtNet, four arms at published weights** | **0.932** | **the standing result** (E097) |

**The whole +0.201 decomposes on ground truth:** own fused labels **+0.089**,
public CC0 labels **+0.077**, ensembling one fold → five **+0.032**, full fit at
full weight **+0.003**, a distilled teacher **−0.013**, **architecture 0.000
every time it has been measured**. Labels and data are +0.166 of +0.201.

**Leaderboard position: 0.926**, up from 0.924 on 2026-09-02, 0.923, 0.846 on
2026-08-29 and 0.725 on 2026-08-22. **Rank 1,104 of 3,263, measured 2026-09-07
(E085).** Field top **0.954**.

**A rank is a measurement with a date on it.** E070 read #866 of 1,866 on
2026-09-02 and this file carried "clears the top-200 cut of 0.917" from it. The
field then grew 75% in five days and both statements went stale without anything
about this project changing. 492 teams now sit at exactly 0.936 and 163 at
0.937, forks of two public notebooks — **655 teams are 0.010+ ahead by clicking
Copy & Edit**. **E043's rule excludes less of that than this file used to claim
— see E088**: E043 treats CC-BY-NC-SA as needing a ShareAlike read, not as
excluded, and reserves "avoid" for undeclared licences.

**Three board results closed levers rather than opening them.**

- **Ensembling is not a series.** Ten members scored exactly what five did, so
  E036's +0.032 for one fold → five was the end of that lever (E064).
- **Mixing families is linear.** Five folds plus one full-fit member scored
  0.924, which is what interpolation predicts at a sixth weight; all five
  full-fit members mixed in scored 0.926, exactly what pure full fit already
  scored (E087). **E023 remains the only union in this log that beat both of
  its members.**
- **Full fit pays once, and +0.003 is a measurement rather than a rate.** The
  same lever read +0.001 diluted to a sixth and +0.003 at full weight, same
  direction. Nothing says a second helping pays again.

**E086 is open and unresolved.** `FULL_FIT_EPOCH = 20` was read off the *fold*
models, which train on 3,526 studies against full fit's 4,407, so the same epoch
buys 25% more optimisation steps. Two arms are built and verified from the same
five trajectories — epoch 20 and epoch 16 — and **neither has been submitted.**
It also carries the board's first like-for-like reseed of a full-weight
ensemble, which bounds how much of the +0.003 is draw rather than lever.

**Gold OOF is not a calibrated predictor of the board.** E026 measured the
offset at +0.005 on one model and concluded no correction was needed; the second
point came in at **+0.054**. It ranked the two correctly, which is what it is
for. It does not forecast a score — see §3.


### B — measured against expert labels out-of-fold (n = 46, CI ±0.05)

| system | gold macro AUC | 95% CI |
|---|---:|---|
| **fused labels, five folds pooled** | **0.7918** | [0.754, 0.829] |
| lexicon labels, five folds pooled | 0.7201 | [0.672, 0.767] |
| imaging, 192px, folds 1–4 pooled | 0.7264 | [0.672, 0.776] |

Paired on all 58 shared studies: **+0.0717, CI [+0.042, +0.103] — A is better**
(E032). **The first change in this project to separate from zero offline.** The
lexicon baseline re-pooled from freshly fetched outputs reproduces E026's
0.7201 exactly, so the comparison does not rest on a remembered number.

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

### E — measured on the frozen-embedding rig, out-of-fold on the 58

`knee-embed` finished on 2026-08-20, so the questions *above* the backbone no
longer wait on the GPU quota. Paired one-variable A/Bs, 5-fold grouped on the
scanner fingerprint, 8.5 min of CPU for all six configurations (E030):

| comparison | delta | 95% CI | verdict |
|---|---:|---|---|
| **fused labels − lexicon** | **+0.0508** | **[+0.001, +0.102]** | **A is better** |
| per-finding maps − baseline | +0.0389 | [−0.009, +0.090] | not separated |
| focal top-k (k=3) − baseline | +0.0060 | [−0.041, +0.051] | not separated |

The labels comparison replicates across seeds — +0.0508, +0.0838, +0.0501,
**mean +0.062**, direction consistent 3/3 — so it is the one configuration with
offline support for spending GPU on it. **It has since been spent** (E031, E032): a
fine-tuned resnet34 gives **+0.0721** on fold 0 and **+0.0717 across all five
folds**, where the interval finally excludes zero. Four instruments sharing no
code path — teacher +0.070, rig +0.062, fold-0 +0.0721, five-fold +0.0717 —
span 0.010.

A frozen backbone is not the fine-tuned model, so **absolute numbers here do not
predict the board**; comparisons above the backbone do transfer, because those
are exactly what is being trained.

### F — built and tested, never measured on anything

Rank-mean ensembling · fused labels wired into *GPU* training (65.4% of slots
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
| "gold OOF estimates the board directly, no correction" | **contradicted** — offset was +0.005 on one model and +0.054 on the next, same architecture and cache. Generalised from n=1. It still *ranks* correctly | nothing; the error was favourable, and it was flagged as a risk before the submission |
| "the teacher bounds everything; raise it and the model follows" | **no longer supported** — the fused model is +0.0126 above its own teacher at the macro (not separated) and beats it outright on Fracture +0.239 and Baker's +0.165. True at 0.769/0.725; not true now | nothing yet; it would have mis-aimed the next phase at labels |
| "Synovitis cannot be fixed from the reports" (E037) | **contradicted** — true of *this project's* reader, false of the reports. A public label set reads Synovitis at **0.790** where ours sat at 0.520, chance | nothing; caught before the imaging-only conclusion was acted on |
| "Synovitis is the best remaining label lever" | **half wrong** — right that it is the weakest finding, wrong that reports can fix it. Radiologists in this corpus mostly do not report synovitis (tr 4.0%, hr 1.2%, bg 0.5%), and the model already beats its teacher there, 0.616 vs 0.520 | nothing; a day of CPU that also found the cue bug |
| "DINOv2 had not flattened at epoch 34; run it to convergence" | **contradicted** — run to 40 epochs it peaks at **epoch 23** and decays, and is 0.074 behind resnet34 on gold. The plan's largest lever was chosen on noise from a truncated run | **~20 GPU-hours**, the most expensive error in the log |
| "focal top-k pooling is worth +0.060" | **contradicted** — +0.060 was the model-to-teacher *headroom* on focal findings, not a gain; measured, top-k gives +0.006 with an interval eight times its width | nothing; the rig caught it for 8 min of CPU |
| "the distilled student is worth +0.0221, interval excluding zero" | **contradicted** — the board priced the finished lineage at **0.910 against 0.923, −0.013**, with the teacher the only variable. E082 found the mechanism first: both lineages cut the same folds, so fold 0's expert labels reach fold 0's training targets through the other folds' models. The rig was scoring a leaked arm against a clean one | a board point, ~9 GPU-h, and the whole distillation route |
| "Synovitis clears the report-text ceiling under a distilled teacher" (E076) | **withdrawn** — it rested on the leaked evaluation above. Synovitis stands where E059 left it, unwritten in the reports rather than badly read | nothing further; withdrawn with the lineage |
| "a union that separates offline is a lever" (E048's comparability rule) | **not a predictor** — it has now failed twice at forecasting what pays: E081 offline (+0.0062, CI touching zero) and E087 on the board (0.926, exactly the better member). It survives as a filter on what is *worth an afternoon*, not on what pays | two afternoons and 1.2 GPU-h |
| "all five full-fit runs are complete and both arms are verified" (a session handoff, 16:30 UTC) | **contradicted** — `s15` was 20 minutes into a 97-minute run and neither arm had started. `lastRunTime` in Kaggle's kernel listing is when a run **starts**, not when it ends | nothing; caught before a click, and the arms have since run for real |
| "E043's licence rule excludes every asset the 0.936 notebooks use, CC-BY-NC-SA included" | **corrected (E088)** — E043 says the opposite: NC matches the competition's own CC-BY-NC 4.0 winner licence, and its avoid-tier is `not-declared`. A "read ShareAlike before shipping" hardened into an "excluded" and sat on the largest measured gap in the project for a week | not yet counted; the ShareAlike question is still unanswered, and answering it is free |
| "we now beat what we were borrowing" (E046) | **stale (E088)** — true on 2026-08-29 against a public system at 0.917. The public systems now score 0.936-0.939 against our 0.926. A closure resting on a moving external number needs a date and a re-read, exactly like a rank | a month of treating the borrowing route as closed while it reopened |
| "the raptor family cannot be screened for free" | **contradicted (E088)** — the tool said it, and the tool was reading the wrong metadata spellings. The checkpoint states `arch`, `res`, a per-finding AUC table and `gold_auc` = 0.9167, which is **ahead of our pooled 0.8980 and inside E048's band**. A screen that silently under-reports converts "we have not looked" into "we looked and there was nothing" | the one CC0 asset that reports ahead of us, left unexamined; fixed with two regression tests |
| "full fit at full weight is worth +0.003, the only positive board lever" (E083) | **contradicted (E092)** — a pure reseed of the same ensemble, seeds 11-15 against 3-7 with nothing else changed, scored 0.923 and 0.921 against 0.926. The board's own like-for-like reseed spread is ~0.003, so the lever is the size of a draw. E086 pre-registered this arm precisely to bound it, and it bounds it at zero | ~9.5 GPU-h across the full-fit lineage, and a lever the plan was built on |

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
| `eda/` | 12 | 2,349 | verification, evaluation, label fusion, push queueing, **the preflight gate** |
| `tests/` | 9 | 1,804 | **159 tests** — 158 pass with the artifacts present, 6 skip without them |

Six kernels remain hand-written one-offs: the header scan, two baseline
submissions, single-model inference, CPU inference, and the cross-geometry
ensemble.

---

## 5. Claim ledger

`docs/FINDINGS.md` tags every claim: **52 `VERIFIED`, 6 `UNVERIFIED`,
9 `CONTRADICTED`, 1 `CORRECTED`.** `docs/EXPERIMENTS.md` holds 45 numbered runs
with what changed, the runtime, the result and what it meant.

---

## 6. What to spend the quota on

**The quota has reset.** It refused at 2026-08-21 18:17 UTC and accepted at
2026-08-22 00:17 UTC, so the weekly window turns over somewhere in that
six-hour band — still not pinned exactly, because only the account page reports
it. **~1.5 h of the new 30 is spent** on E031.

In the order they are worth doing:

1. ~~**Train on the fused labels.**~~ **Done — E031, E032.** All five folds
   run; **+0.0717 paired at n=58, CI [+0.042, +0.103], separated.** Gold OOF
   0.7918 against the lexicon model's 0.7201.
   ~~**The open action is now to submit it.**~~ **Submitted — 0.846** (E034).
   **The open action is now to submit the LEXICON 5-fold**, which already exists
   and costs 0.8 h. Gold OOF scores one model per study; the submission
   rank-averages five. So E032's +0.0717 and the board's +0.121 are not
   measuring the same system, and until the lexicon 5-fold is on the board,
   "+0.121 from better labels" is an attribution nobody has earned.
2. **Submit the 4-fold rank-mean ensemble.** Same configuration, different
   splits, no hypothesis that can be wrong. *(Superseded by the fused 5-fold,
   which is the same idea on better labels.)*
   **Not worth doing: blending the fused and lexicon families.** Measured in
   E033 at zero quota — worse than fused alone at every weight, monotonically.
3. **Complete the gold pool** to n=58 by running fold 0 with a gold dump.
4. **Re-examine DINOv2** — 0.7041 at epoch 34, still not clearly converged.

CPU work is unaffected and is where anything further will happen until the
quota returns. As of E030 the rig has answered every question that was waiting
on it, so what remains on CPU is a **third report reader** (`PATH.md` Phase A,
~4 h of CPU, the union rule already exists) rather than more head-level A/Bs.
