# HANDOFF — read this first

The session entry point. Everything here is either a live instruction or a
pointer to the file that holds the detail. Last updated **2026-09-07**.

Read order: **this file**, then `docs/STATUS.md` (what is known, by evidence
strength), then `docs/PATH.md` (what is left and what it is worth).

---

## 1. Before any push

```bash
bash eda/preflight.sh
```

Four gates, the same four `.github/workflows/tests.yml` runs, in the same
order: **lint**, **218 tests**, **kernel drift**, **no patient-derived file
tracked**. Green here is green there. The drift gate is the one that matters
most — `kaggle/*/run.py` is generated from `src/pipeline.py` and a hand-edited
kernel still pushes and still runs, it just is not the pipeline any more.

**It covers 90 of the 96 kernel directories.** Six predate the generator and are
outside it: `00_dicom_header_scan`, `01_submission_baseline`,
`02_metadata_submission`, `05_infer`, `05_infer_cpu`, `09_infer_ensemble`. All
six are superseded and nothing in the current lineage mounts them, but the gate
is silent about them, so do not read a green drift check as covering everything
under `kaggle/`.

`ruff` is pinned exactly in `requirements.txt`, not floated. Minor ruff
releases add lint rules, and a float means CI can fail on a commit that changed
no code — which is exactly what it did before 2026-08-21.

## 2. Credentials

Kaggle account **achelijndiamantidis**. The CLI is not in the base image:

```bash
pip install kaggle                       # 2.2.4
export KAGGLE_API_TOKEN=<token>          # or ~/.kaggle/access_token, chmod 600
```

Note `import kaggle` resolves to this repo's own `kaggle/` **directory** when
the working directory is the repo root, so it is not a test of whether the
package is installed. Check `which kaggle` instead.

## 3. Where the work stands

**0.926 on the leaderboard**, up from 0.924, 0.923, 0.846 and 0.725. **Rank
1,104 of 3,263, measured 2026-09-07 (E085).** Field top **0.954**. Final
submission 2026-10-22. The field grew 75% in five days, so E070's "#866 of
1,866" went stale almost immediately — **a rank is a measurement with a date on
it.** 492 teams sit at exactly 0.936 and 163 at 0.937, forks of two public
notebooks; see §4d.

**The standing system is `knee-infer-v1pubfull5` — five FULL-FIT members and
nothing else**, every member trained on all 4,407 studies: resnet34 2.5D at
192px on **publicly shared CC0 report labels**
(`stevenleehans/rsna-knee-llm-report-labels`, repackaged with attribution as
`knee-phase1-public`), rank-mean. E083 priced it against the five-fold
ensemble's 0.923 at **+0.003**, with data exposure the only variable. **A
full-fit model cannot be scored offline by construction** — it trains on all 58
gold studies. The five-fold ensemble it replaced reads 0.8980 gold OOF.

**Two board results closed mixing rather than opening it.** E064: five folds
plus one full-fit member scored 0.924, which is what linear interpolation
predicts at a sixth weight. E087: five folds plus all five full-fit members
scored **0.926 — exactly what pure full fit already scored.** Nothing here is
super-additive, and E023 remains the only union in this log that beat both of
its members.

**Self-distillation is closed and the way it closed is the thing to read first.**
It separated offline twice — teacher +0.0261 (E069), student +0.0221 with a 95%
interval excluding zero (E076) — and the board scored the finished lineage at
**0.910 against 0.923, −0.013**. The offline rig had the sign wrong. E082 found
why before the score landed: both lineages cut the same folds, so fold 0's
expert labels reach fold 0's training targets through the other folds' models,
and `v1public`'s report-text targets carry no such path. **Gold-58 is retired
for teacher comparisons**, and E076's Synovitis-ceiling claim is withdrawn.

**Labels and data are the whole story.** Of the **+0.201** from the first
imaging model at 0.725 to 0.926: own fused labels **+0.089**, public CC0 labels
**+0.077**, ensembling one fold → five **+0.032**, full fit at full weight
**+0.003**, a distilled teacher **−0.013**, **architecture 0.000, every time.**
Labels and data are +0.166 of the +0.201.

**Two numbers govern every decision here. Read them before planning anything.**

1. **The offline instrument has a ±0.03 noise floor.** E060 proved it the hard
   way: a pure reseed — changing *only* the RNG seed — scored −0.0284
   [−0.063, +0.002] against the incumbent, the same magnitude that had just
   been reported as an "architecture effect" (E057), which was therefore wrong.
   **Any fine-tuned A/B under ±0.03 is noise.** Only DINOv2's −0.148 ever
   cleared it.
2. **The frozen rig does not predict fine-tuned architecture.**
   `eda/head_lab.py` called per-finding pooling +0.034 and +0.053; fine-tuned
   came back −0.034, which was itself noise. **Trust it on labels and targets** —
   it called the fused labels +0.0508 and the board paid +0.089. **Never spend
   GPU on its architecture verdicts.** Pass `--seeds 4` or it measures an
   initialisation rather than a hypothesis.

Consequence: **the board is the only trustworthy instrument.** Five submissions
a day. Inference is ~1.0 h of a 9 h cap plus 0.037 h per extra member, so
ensemble size is limited by *training* quota alone, never by the submission.


## 4. Quota state

**The reset time is `UNVERIFIED` and the band below is two observations, not a
rule** (E077). Measured: refused 2026-08-21 18:17 and accepted 2026-08-22 00:17;
then refused 2026-09-04 at 18:20, 20:30 and 22:37 and accepted **2026-09-05
00:36** — past the edge the first observation implied. All that is known is that
on two occasions it fell somewhere between 18:17 and 00:36 UTC, six hours wide
from two points. This file previously stated the band as fact from a single
observation, which is the error the standing rule exists to prevent.

Kaggle's API reports neither the balance nor the reset moment. **Only the account
page does** — `https://www.kaggle.com/settings` shows hours remaining and when
they reset. **Do not predict the reset: push the thing you want and read the
refusal** (E039). A refused push costs nothing.

Two limits exist and both refusals begin with "Maximum"; confusing them has cost
real time before:

| message | meaning | does waiting help |
|---|---|---|
| `Maximum batch GPU session count` | concurrency, 2 slots | **yes** — `eda/push_queue.sh` polls |
| `Maximum weekly GPU quota` | the 30 h allowance | **no** — nothing runs until reset |

CPU is a separate allowance with **5 slots**, and it is a bigger lever than it
looks: E050, E053, E056–E063 were all CPU-only and cost zero GPU quota.

A resnet34 fold is **~1.5 GPU-h, not 7**. The 7-hour figure came from DINOv2's
40-epoch ViT runs. Five resnet34 folds is ~7.5 h.

**E039's rule: a probe must not be a job that costs something if it succeeds.**
Pushing the thing you actually want is fine — it runs if quota exists and errors
for free if not.

**One failure mode is hardware, not code.** `knee-train-v1pubfull-s5` died 54
seconds in with `CUDA error: uncorrectable ECC error encountered` (E064) — a
fault in the assigned card's memory, before any batch trained. That is the one
case where re-pushing is the right response rather than an excuse. Anything else
that fails is this project's bug until proven otherwise.

## 4c. Free GPU elsewhere — allowed, and already wired

`colab/train_fold_on_colab.ipynb` runs **the same generated `run.py`**,
byte-identical, on Colab free's T4. Only the paths and the machine change, so a
fold trained there pools with one trained on Kaggle. **It is currently aimed at
the `v1pubdistil` folds, which E083 closed at −0.013 — retarget it before use.**
The full-fit lineage (`knee-train-v1pubfull-s*`) is what the board pays for.

**The rules split cleanly and both halves matter** (`FINDINGS.md` 2.7 and 2.15):

| | |
|---|---|
| free GPU elsewhere for **training** | **allowed** — only inference must be a Kaggle notebook, internet off |
| a **second Kaggle account** for more quota | **forbidden**, verbatim: *"you cannot enter or submit from multiple accounts"* |
| teaming | the sanctioned route — max 5, merger deadline 2026-10-15, share datasets not credentials |

**What it actually costs**, so the setup is not a surprise: ~2.5–3 h per fold on
one T4 against ~1.4 h on Kaggle's two, one fold per session, and a **~10 GB
cache against free Drive's 15 GB total**. The Drive quota binds before the GPU
does. Weights return as a private Kaggle Dataset and inference mounts them.

This is worth doing for throughput, not for the calendar: it is a second GPU
stream for the remaining weeks, not a way to save the ~3 days until a reset.

**Paid GPU is wired but BLOCKED**: `gcp/` runs the same generated trainer on a
GCE VM, unattended, and stops the machine on exit — but this project's GCP
quota was measured at **`GPUS_ALL_REGIONS = 0` on 2026-09-03**, so no GPU VM
can start until an increase is granted. Check before assuming it is an
option. It beats Colab on session cap
and disk but is not free, and **a new GCP project has zero GPU quota** — an
increase can take days, so check it before assuming it is faster than
waiting for the weekly reset. `gcp/README.md` has the costs and the trap.

## 4b. SUBMITTING — the API cannot do it

**`kaggle competitions submit` fails with HTTP 400:**

> `{"error":{"code":400,"message":"Submission not allowed:  This competition
> only accepts Submissions from Notebooks.","status":"FAILED_PRECONDITION"}}`

Verified 2026-09-01 against two finished inference kernels holding valid
`submission.csv` output. Older rows in `kaggle competitions submissions` show
`fileName submission.csv` and score fine, so this reads as a competition-side
change rather than something wrong with the file — **do not waste time debugging
the CSV.**

**The cause is now named rather than inferred.** The competition's own metadata
carries `is_kernels_submissions_only = True`, read from the API on 2026-09-07:

```python
from kaggle.api.kaggle_api_extended import KaggleApi
a = KaggleApi(); a.authenticate()
c = a.competitions_list(search='rsna-knee-abnormality').competitions[0]
c.is_kernels_submissions_only   # True
c.max_daily_submissions         # 5
c.deadline                      # 2026-10-22 23:59
```

  So the 400 is **by design and permanent**, not a transient change that might
  revert. Do not retry it periodically hoping it was a glitch. The same call
  confirms `max_daily_submissions = 5`, `max_team_size = 5`, the 2026-10-22
  deadline and the 2026-10-15 merger deadline, all first-hand rather than from
  the rules page.

## 4e. FINAL SUBMISSION SELECTION — `UNVERIFIED`, and it decides the result

Nothing in this repo has ever checked how the private leaderboard picks which
submissions count. **The API does not expose it** — there is no field for the
selection cap in the competition metadata, and `privateScore` is empty on every
row of `kaggle competitions submissions`, as expected before the reveal.

What is known: `FINDINGS.md` §2.13 quotes the efficiency prize as requiring a
**selected** submission, so selection exists and is not automatic for that
prize. What is not known is how many may be selected, and what happens on
2026-10-22 if none are.

**Check the competition's My Submissions page before the deadline and record the
answer as a finding.** This is cheap now and unrecoverable afterwards: every
board result in `STATUS.md` §1A is a *public* score, and the private score is
the one that is paid.



**The only route is the browser**, once per submission:

1. open `https://www.kaggle.com/code/achelijndiamantidis/<kernel-slug>`
2. **Submit to Competition**

Everything up to that point is scriptable — push the kernel, wait for COMPLETE,
verify `checkpoints mounted: N` in its log — and the click is not. Budget for
it: a run that nobody clicks is a run that scored nothing. Submitting the same
notebook twice returns the same score and spends two of the five daily slots.

## 4d. The public-notebook route, and the one decision it needs

**Measured 2026-09-07 (E085).** The field is **3,263 teams**, top **0.954**, and
0.926 is **rank 1,104**. 492 teams sit at exactly 0.936 and 163 at 0.937 — forks
of two public notebooks. **655 teams are 0.010+ ahead of this project by
clicking Copy & Edit.** The rules allow it: *"It's okay to share code if made
available to all Participants on the forums."*

**E043's licence rule is what stands between here and there.** Every notebook at
0.936+ mounts `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` or
`prvsiyan/rsna-knee-v52-radimagenet-heads` (both `other`), or the `marwanmath` /
`antoinegg1` RadImageNet heads (CC-BY-NC-SA). **0.936 is not reachable under
that rule as written.**

The strongest admissible arm is **`dreaddevelopment/raptor-knee-widedense`** —
CC0-1.0, upstream 0.924 from one model with no ensembling or TTA, 0.9167 on the
58 gold with gold held out. Blended with our 0.926 full-fit ensemble it is the
best-shaped union in the log: 0.002 apart, maximally different in kind
(CoAtNet / 64 slices / five plane slots / 336px / 140 mm crop against resnet34
2.5D / 192px / our labels). `knee-blend-raptor` and `_templates/rank_blend.py.in`
are written, tested and ready.

**What is blocked, and why it is not an engineering problem.** Upstream's
notebook persists no output files (`files: []`), so its submission cannot be
mounted — E085 ran the blend and it correctly refused with one member. The arm
has to be RUN, which means a fork of someone else's notebook under this account.

**E088 narrows this objection by half, and the half it leaves standing is the
preprocessing.** The paragraph above used to say reimplementing means
reproducing `RaptorClassifier`, `build_backbone`, `eval_windows` and the
five-slot selection. **The checkpoint specifies the model itself**, so
`RaptorClassifier` and `build_backbone` are not reverse-engineered from anyone:

```
arch = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'   res = 384   src = timm-pretrained
backbone.*  478 tensors    norm.*  2 (LayerNorm 1024)
att.0/att.3  4 tensors     Linear(1024->256) -> Linear(256->12)
clsW/clsb    2 tensors     Linear(1024->12)
```

  A named timm backbone plus an eight-tensor head is a specification, not a
  copy. **What the file does not carry is the input pipeline** — slice choice,
  crop, windowing — and there `eval_windows` and the five-slot selection are
  still someone else's work, with real silent-skew risk and no way to check it.
  **So the decision below is still the account owner's**; it is just a narrower
  decision than this section claimed.

**And the CC0 family is fifteen datasets, not one (E088).** `dreaddevelopment`
publishes 15 datasets, every one CC0-1.0, carrying **14 checkpoints** — CoAtNet
in seven geometry variants, plus ConvNeXtV2-B336, EfficientNetV2-L480 and a
CNN336. This project has used none of them. `eda/survey_public_checkpoints.py`
was reporting "cannot be screened for free" about them because it read the wrong
metadata spellings; that is fixed and has two regression tests.

**E088 REMOVED THIS BLOCKER FOR THE COATNET ARM, AND IT IS NOW BUILT.**
`kaggle/81_infer_raptorcc0` and `kaggle/82_infer_raptorcc0x4` run the CC0
CoAtNet weights **in our own kernel**, mounting `dreaddevelopment`'s datasets
directly. Nothing is forked and nothing restricted is mounted: the `tonylica`
asset that gates the 0.937 system's DINOv3 arm stays out, and
`tests/test_raptor_arm.py` asserts every mounted asset is CC0 or Apache against
the dated licence snapshot, so E043's rule is now enforced rather than described.

The fork question below still stands for the DINOv2, DINOv3 and RadImageNet
arms. It no longer stands for the strongest one.

**So the remaining fork decision is the account owner's call.** If the answer is
yes, for the other three arms:

1. Open `kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model`
   and **Copy & Edit**. Keep the attribution cell.
2. Confirm it mounts `dreaddevelopment/raptor-knee-widedense` and the
   competition, GPU on, internet **off**, then **Save & Run All**.
3. Submit that run on its own first. **It is the control**: if it does not come
   back near 0.924, the blend has an untrustworthy member and no reading of the
   blend means anything.
4. Set `external_kernels` on `knee-blend-raptor` to the new fork's `owner/slug`,
   regenerate, push, submit. The pre-registered reading is in the kernel note.

If the answer is no, Raptor is out and **0.926 is close to the ceiling of what
this project can reach on its own work** — §2.1's full-fit lever paid +0.003 and
E064 says a second helping of the same kind does not compound.

## 5. The next action

**TWO SUBMISSIONS ARE OWED A CLICK, AND NOTHING ELSE IS BLOCKING.** Both
inference arms of E086 finished on 2026-09-07 at ~17:52 UTC and were verified
end to end at 23:20 UTC — five checkpoints each, matched back to their five
trainers by val macro AUC, no cross-contamination in either direction. **Neither
has reached the board.** The API cannot submit (§4b), so:

1. open `https://www.kaggle.com/code/achelijndiamantidis/knee-infer-v1pubfe`
   → **Submit to Competition** — five full-fit members at **epoch 20**
2. open `https://www.kaggle.com/code/achelijndiamantidis/knee-infer-v1pubfe-early`
   → **Submit to Competition** — the same five models at **epoch 16**

Four of five daily slots were spent on 2026-09-07 and the allowance resets at
00:00 UTC, so from then both fit in one day and neither needs to go first.
**Record the result as the next free E-number** — reserving one in advance just means renumbering it every time something else lands first, which has now happened twice. The three readings are pre-registered in E086 and
must not be re-derived after the scores land:

1. `v1pubfe-early` vs `v1pubfe` — the export epoch, one variable, same five
   trajectories. *early > late* → epoch 20 was overtraining every member and
   0.926 was left short. *early < late* → the step-count argument is wrong.
   *within 0.001* → the curve is flat here, as E055 found it flat over 18–21.
2. **`v1pubfe` vs the standing 0.926** — seeds 11–15 against seeds 3–7, nothing
   else changed. The board's first like-for-like reseed of a full-weight
   ensemble. **It bounds how much of any full-fit result is draw rather than
   lever, including the +0.003 E083 credited to full fit.** Do not skip it
   because it is the uncomfortable arm.
3. Whichever arm wins is submittable on its own merits.

**The monitor-set figures are not evidence for either arm.** Epoch 16 reads
0.918–0.926 and epoch 20 reads 0.943–0.951, but that set is *in training*, so
those are memorisation. They establish only that the model is still actively
fitting between 16 and 20, which is what makes the question live rather than
what answers it.

**A caution the addendum to E086 records in full.** The handoff that announced
these arms declared them run and verified at 16:30 UTC, when `s15` was 20
minutes into a 97-minute run and neither arm had started. `lastRunTime` in
Kaggle's kernel listing is the moment a run **starts**, not the moment it ends —
confirmed at five consecutive queue transitions. Read it that way before
concluding anything is finished.

**THEN THE CC0 COATNET ARM, IN THIS ORDER AND NOT ANOTHER.**

```bash
bash eda/preflight.sh && kaggle kernels push -p kaggle/81_infer_raptorcc0
# wait for COMPLETE, verify the log, submit it, and READ THE SCORE FIRST
kaggle kernels push -p kaggle/82_infer_raptorcc0x4      # only after that
```

`81` is one CC0 checkpoint, no blending, no TTA — **the control**. Upstream
self-reports this family at 0.924 on the board and 0.9167 on the 58 gold with
gold held out. **If it does not come back near 0.92, our mounting or
preprocessing is wrong and no later blend can be read at all.** `82` is the same
model family at the published four-sub-model weights (0.55 / 0.20 / 0.15 / 0.10,
the third being a horizontal flip that costs no extra checkpoint). Pushing `82`
first would give a mounting bug four places to hide.

`74_blend_raptor` is retargeted onto `81` and must not be pushed until `81` has
**scored**, not merely run.

**Verify the weights before pushing, and it costs nothing:**

```bash
kaggle datasets download dreaddevelopment/raptor-knee-maxspan \
    -f raptor_ft_coatnet_v5_full_swa.pt -p ckpt
python eda/verify_raptor_checkpoint.py --checkpoint ckpt/raptor_ft_coatnet_v5_full_swa.pt
```

All three mounted checkpoints have passed this: fingerprint matched, all 486
tensors loaded `strict=True`, windows built at each arm's own geometry, twelve
calibrated outputs, reverse member not a no-op (E090). Re-run it if upstream
ever changes a file — the kernel will refuse to start on a fingerprint
mismatch, but finding that out here costs a second rather than a GPU session.

**Runtime is the remaining risk, and it is sized.** Forward is ~1.0-1.1 s per
window on CPU; `81` pushes 62 windows per study and `82` pushes **228 across
three preprocessing passes, 3.7x the control**. Against a 9 h cap on ~1,300
studies that is comfortable for `81` and genuinely tight for `82`. Each group
now prints its projected hours from study 100, so a run that will not fit is
visible early. If `82` does not fit, drop the 0.10 member (`native384dense-v10`)
first — never change a `k_eval` or a span to save time, because those are the
geometry the weights were trained at and E090 is what happens when they drift.

**After E088 and E089, the log is out of measured levers on its own work.** §4d's
public-notebook route is the only remaining item with a large measured gap and
it is blocked on an account-owner decision, not on engineering. **Do not spend
GPU on architecture of any kind** — the instrument that made those ideas look
promising is retired, and every board-level architecture test has returned zero
or negative.


## 6. What the CPU rig has already settled

`knee-embed` finished on 2026-08-20 — a frozen DINOv2 ViT-S/14 pass over all
4,407 studies, 344.6 min of CPU, saved as `(4407, 60, 384)` float16. With it,
every question above the backbone costs **~8 minutes on CPU** instead of a GPU
session:

```bash
kaggle kernels output achelijndiamantidis/knee-embed -p artifacts/embed
kaggle datasets download achelijndiamantidis/knee-phase1-artifacts \
    -p artifacts/kaggle_dataset --unzip
kaggle datasets download achelijndiamantidis/knee-phase1-fused \
    -p artifacts/kaggle_dataset_fused --unzip
kaggle competitions download rsna-knee-abnormality-detection -f train.csv -p data
python eda/head_lab.py --embeddings artifacts/embed/embeddings.npy \
    --index artifacts/embed/embeddings_index.json --compare all
```

**Pass `--seeds 4`.** Without it the rig measures an initialisation as much as
a hypothesis: on the focal A/B the treated arm moved 0.007 across restarts
while the *baseline* moved 0.036, so the difference swung from +0.007 to +0.044
depending on the draw (E052). `--seeds N` averages out-of-fold predictions over
restarts before scoring, which removes that from both arms and roughly halves
the interval.

Answers as of E053, all paired one-variable A/Bs scored out-of-fold on the 58:

| question | answer |
|---|---|
| do the fused labels help | **yes**, +0.0508 [+0.001, +0.102], replicated — **and confirmed on the board** at +0.089 |
| do per-finding attention maps help | rig says **+0.035**, replicated three times. Fine-tuned gave −0.0338 — but a **pure reseed gives −0.0284** (E060), so that was seed noise. **Unmeasured, not negative.** |
| does focal top-k pooling help | +0.0163 [−0.001, +0.035] alone, −0.0075 on top of pooling (E056) — **rig-only, and the rig is not trusted on heads** |
| does slice position help | +0.0035 [−0.017, +0.025] — rig-only. Separately: the model is *exactly* permutation-invariant over slices (E050), which is arithmetic and does hold. |

| do auxiliary report targets help | **no**, against their own shuffled control: −0.0050 then +0.0060 across two lexicons, sign flipping, neither separated (E063). Note the rig freezes the trunk, which is what auxiliary supervision is supposed to shape — so it is a weak instrument for this one, and E062 should have said so first. |

**READ E060 FIRST — IT CORRECTS E057 AND E058.**

| | per-finding − baseline |
|---|---:|
| rig, frozen DINOv2 | +0.0338 [+0.009, +0.061] |
| rig, frozen resnet34 | +0.0528 [+0.013, +0.097] |
| fine-tuned resnet34 | −0.0338 [−0.067, −0.007] |
| **fine-tuned, SEED CHANGE ONLY** | **−0.0284** [−0.063, +0.002] |

That last row is the control arm E057 lacked. Changing *nothing but the seed*
reproduces almost exactly the loss E057 attributed to per-finding pooling, so
the fine-tuned number is seed noise and the rig was never contradicted. E058's
headline ("the rig inverted") is withdrawn; only its hedge survives — the rig is
**unvalidated** as a predictor of fine-tuned architecture behaviour, not proven
wrong either way.

**The load-bearing number for anyone planning work here: fine-tuned single-seed
gold comparisons have a noise floor of about ±0.03**, which is larger than any
architecture effect this project has ever hypothesised. Only differences well
outside that mean anything — DINOv2's −0.148 (E051) clears it and stands;
per-finding pooling's −0.034 does not. There is no cheap fix, either: four seeds
of five folds is 30 GPU-h, a whole weekly quota for one A/B.

- **Labels: trust the rig.** A label comparison changes the target rather than
  the head, and the board confirmed it — +0.0508 on the rig, +0.089 paid.
- **Architecture: the rig is unvalidated, and the fine-tuned instrument cannot
  resolve ±0.03 either.** Treat both a rig positive and a single-seed
  fine-tuned negative as "not measured". This is why ensembling, not
  architecture, is the lever with a board-confirmed coefficient.

Absolute numbers from the rig do not predict the board — a frozen backbone is
not the fine-tuned model. **Comparisons above the backbone do transfer**,
because those are the part being trained.

**Every single-seed negative in this log predating 2026-08-31 is suspect**, for
the reason E053 gives: the rig could not resolve a ~0.035 effect at one seed,
and recorded the failure to resolve as an absence. Re-run before trusting one.

**Two embedding banks now exist and they are not interchangeable.** Every rig
number above was measured on frozen **DINOv2 ViT-S/14** features
(`artifacts/embed_dinov2/`), while the lineage they are being used to justify
is fine-tuned **resnet34**. That transfer is an assumption, not a measurement —
`knee-embed`'s manifest declares `RUN_BACKBONE = "resnet34"` and its stored
output was stale DINOv2 from an earlier version, so it was re-run on CPU to
produce the matching bank. `head_lab` prints `index['backbone']` on every run;
read it before comparing two numbers.

## 7. The standing rule

This project has overturned **eight** of its own confident claims, and every one
was caught by a measurement rather than by reasoning. The recurring shape is a
small number of observations read as a trend, or a number measured as a
*ceiling* and later used as a *gain*.

So: every comparison is one-variable by construction, a test asserts it, and a
claim carries the interval it was measured with. **A negative carries its
interval width too, plus a control arm.** E057 carried its interval and was
still wrong, because it had no control arm — a pure reseed reproduced its entire
"effect" (E060).

**Pre-register the acceptance rule before the data arrives.** E054 did it and it
worked; E062 did it and the answer came back null, which is what a
pre-registration is for. E062 also shows the failure mode to avoid next time: it
picked an instrument (the frozen rig) that is structurally blind to the
mechanism it was testing, and only noticed afterwards. **Say what the instrument
cannot see, before running it, not after.**

`docs/FINDINGS.md` tags every claim `VERIFIED` / `UNVERIFIED` / `CONTRADICTED` /
`CORRECTED`; `EXPERIMENTS.md` is append-only, E001 through E064, and a run with
no entry did not happen.
