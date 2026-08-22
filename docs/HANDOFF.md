# HANDOFF — read this first

The session entry point. Everything here is either a live instruction or a
pointer to the file that holds the detail. Last updated **2026-08-22**.

Read order: **this file**, then `docs/STATUS.md` (what is known, by evidence
strength), then `docs/PATH.md` (what is left and what it is worth).

---

## 1. Before any push

```bash
bash eda/preflight.sh
```

Four gates, the same four `.github/workflows/tests.yml` runs, in the same
order: **lint**, **159 tests**, **kernel drift**, **no patient-derived file
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

**0.846 on the leaderboard** as of 2026-08-22, up from 0.725 — the fused-label
5-fold rank-mean (E034). Field top 0.952, top-200 cut 0.917, 1,866 teams, final
submission 2026-10-22. Full picture in `STATUS.md`; the costed plan in
`PATH.md`.

**The board allows 5 submissions a day, not 2.** Four files in this repo said 2
and were wrong; corrected 2026-08-22 after a live submit reported "4 remaining".

**Gold OOF ranks models; it does not forecast a score.** The offset to the board
was +0.005 on one model and +0.054 on the next, same architecture and cache. Use
it to choose between models, never to predict where a submission will land.

## 4. Quota state

**The quota reset on 2026-08-22.** It refused at 2026-08-21 18:17 UTC and
accepted at 00:17 UTC, so the weekly window turns over in that six-hour band.
Kaggle's API reports neither the balance nor the reset moment — only the account
page does — so the only test is to attempt the push and read the refusal.

**~1.5 h of the new 30 is spent** (E031). Two limits exist and both refusals
begin with "Maximum"; confusing them has cost real time before:

| message | meaning | does waiting help |
|---|---|---|
| `Maximum batch CPU sessions` | concurrency, 5 slots | **yes** — `eda/push_queue.sh` polls |
| `Maximum weekly GPU quota` | the 30 h allowance | **no** — nothing runs until reset |

CPU is a separate allowance and is unaffected.

## 5. The next action

**Submit the lexicon 5-fold** — `kaggle/11_infer_folds`, models already trained,
~0.8 h and one of five daily submissions.

This is not a formality, it is the missing control. Gold OOF scores each study
with the *single* fold that held it out, so E032's **+0.0717** measured a
one-model system. The submission that scored 0.846 rank-averages **five**. So
the +0.121 board jump is *labels plus ensembling*, and nothing yet separates
them. If the lexicon 5-fold also jumps, most of the credit belongs to
ensembling, and every future single-fold gold comparison is understating what
its ensemble will do — which changes how the remaining GPU budget should be
spent.

After that, the ranked queue is `PATH.md` Phase C (DINOv2 to convergence, ~23
GPU-h for five folds — the largest published lever, still never given a fair
run) and Phase A (a third report reader, CPU only).

**Synovitis is the weakest finding at 0.616** and has inherited Medial
Meniscus's old role as the biggest single drag on the macro. E029 flagged it as
the one finding the fused teacher genuinely does not know, so it is a *label*
problem before it is a modelling one.

Do **not** re-push any `21/27/28/29/30_train_v1fused*` kernel — all five folds
are done. ~6.9 h of the weekly 30 is spent, ~23 h remain.

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

Answers as of E030, all paired one-variable A/Bs scored out-of-fold on the 58:

| question | answer |
|---|---|
| do the fused labels help | **yes**, +0.0508 [+0.001, +0.102], replicated |
| does focal top-k pooling help | **no measurable effect**, +0.006 [−0.041, +0.051] |
| do per-finding attention maps help | **not separated**, +0.039 [−0.009, +0.090] |

Absolute numbers from the rig do not predict the board — a frozen backbone is
not the fine-tuned model. **Comparisons above the backbone do transfer**,
because those are the part being trained.

## 7. The standing rule

This project has overturned **six** of its own confident claims, and every one
was caught by a measurement rather than by reasoning. The recurring shape is a
small number of observations read as a trend, or a number measured as a
*ceiling* and later used as a *gain* — E030 is the most recent instance.

So: every comparison is one-variable by construction, a test asserts it, and a
claim carries the interval it was measured with. `docs/FINDINGS.md` tags every
claim `VERIFIED` / `UNVERIFIED` / `CONTRADICTED` / `CORRECTED`; `EXPERIMENTS.md`
is append-only and a run with no entry did not happen.
