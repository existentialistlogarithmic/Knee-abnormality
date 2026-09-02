# HANDOFF — read this first

The session entry point. Everything here is either a live instruction or a
pointer to the file that holds the detail. Last updated **2026-09-02**.

Read order: **this file**, then `docs/STATUS.md` (what is known, by evidence
strength), then `docs/PATH.md` (what is left and what it is worth).

---

## 1. Before any push

```bash
bash eda/preflight.sh
```

Four gates, the same four `.github/workflows/tests.yml` runs, in the same
order: **lint**, **183 tests**, **kernel drift**, **no patient-derived file
tracked**. Green here is green there. The drift gate is the one that matters
most — `kaggle/*/run.py` is generated from `src/pipeline.py` and a hand-edited
kernel still pushes and still runs, it just is not the pipeline any more.

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

**0.924 on the leaderboard** as of 2026-09-02 (E064), up from 0.923, 0.846 and
0.725. **This clears the top-200 cut of 0.917.** Field top 0.952, 1,866 teams,
final submission 2026-10-22.

The standing system is `v1public` + one full-fit member: resnet34 2.5D at 192px,
five folds trained on **publicly shared CC0 report labels**
(`stevenleehans/rsna-knee-llm-report-labels`, repackaged with attribution as
`knee-phase1-public`), rank-mean, plus one model trained on every study. Gold
OOF for the five folds alone is 0.8980; the full-fit member cannot be scored
offline by construction.

**Labels and data are the whole story.** Of the +0.199 from the first imaging
model: own fused labels +0.089, public CC0 labels +0.077, ensembling +0.032,
full fit (as one member in six) +0.001, **architecture 0.000, every time**.

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

The weekly window turns over in an 18:17–00:17 UTC band, seven days apart from
2026-08-22, so the next reset is **~2026-09-05**. Kaggle's API reports neither
the balance nor the reset moment — only the account page does — so the only test
is to attempt the push and read the refusal.

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

## 4b. SUBMITTING — the API cannot do it

**`kaggle competitions submit` fails with HTTP 400:**

> `{"error":{"code":400,"message":"Submission not allowed:  This competition
> only accepts Submissions from Notebooks.","status":"FAILED_PRECONDITION"}}`

Verified 2026-09-01 against two finished inference kernels holding valid
`submission.csv` output. Older rows in `kaggle competitions submissions` show
`fileName submission.csv` and score fine, so this reads as a competition-side
change rather than something wrong with the file — **do not waste time debugging
the CSV.**

**The only route is the browser**, once per submission:

1. open `https://www.kaggle.com/code/achelijndiamantidis/<kernel-slug>`
2. **Submit to Competition**

Everything up to that point is scriptable — push the kernel, wait for COMPLETE,
verify `checkpoints mounted: N` in its log — and the click is not. Budget for
it: a run that nobody clicks is a run that scored nothing. Submitting the same
notebook twice returns the same score and spends two of the five daily slots.

## 5. The next action

**THE WEEKLY GPU QUOTA IS SPENT** (2026-09-02, E067). It resets ~2026-09-05.
`s4` is COMPLETE, `s6` is running, **`s5` errored on an ECC fault and `s7` was
never pushed** — so the full-fit lineage is 3 of 5. At the reset, push both:

```bash
kaggle kernels push -p kaggle/60_train_v1pubfull_s5
kaggle kernels push -p kaggle/62_train_v1pubfull_s7
```

**Then push `kaggle/63_infer_v1pubfull5` and click submit.** Five full-fit members and
nothing else — the data lever at full weight instead of the sixth-weight
mixture E064 priced at +0.001. Check with:

```bash
for k in s4 s5 s6 s7; do kaggle kernels status achelijndiamantidis/knee-train-v1pubfull-$k; done
bash eda/preflight.sh && kaggle kernels push -p kaggle/63_infer_v1pubfull5
```

The kernel now **refuses to run** at anything other than five mounted members
(`MEMBERS_EXPECTED`), so a missing checkpoint stops it instead of quietly
submitting a three-member ensemble against a five-member claim.

Each trainer must log `FULL FIT: train 4,407 (every study)` and emit
`checkpoint_foldall.pt` with **no gold dump** — a model trained on all 58 gold
studies must never produce a file `pool_gold_oof.py` can glob.

If the five-member full fit does not beat 0.924, **the data lever is spent** and
`PATH.md` §2.2 (the weekly label-set survey) is all that remains with a
board-measured coefficient behind it.

**Do not spend GPU on architecture of any kind.** The instrument that made those
ideas look promising is retired, and every board-level architecture test
returned zero or negative.

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
