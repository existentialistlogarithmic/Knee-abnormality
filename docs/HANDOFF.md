# HANDOFF — read this first

The session entry point. Everything here is either a live instruction or a
pointer to the file that holds the detail. Last updated **2026-08-21**.

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

**0.725 on the leaderboard.** Field top 0.952, 1,866 teams, final submission
2026-10-22. Full picture in `STATUS.md`; the costed plan in `PATH.md`.

## 4. The live blocker

**The 30-hour weekly GPU quota is exhausted.** Confirmed again on 2026-08-21 —
pushing `kaggle/21_train_v1fused` was refused with:

```
Kernel push error: Maximum weekly GPU quota of 30.00 hours reached.
```

Kaggle's API exposes neither the remaining balance nor the reset moment; only
the account page does. So the reset time is `UNVERIFIED` and the only way to
test it is to attempt the push and read the refusal.

Two separate limits exist and both refusals begin with "Maximum". Confusing
them has cost real time before:

| message | meaning | does waiting help |
|---|---|---|
| `Maximum batch CPU sessions` | concurrency, 5 slots | **yes** — `eda/push_queue.sh` polls |
| `Maximum weekly GPU quota` | the 30 h allowance | **no** — nothing runs until reset |

CPU is a separate allowance and is unaffected.

## 5. The next action

**Push `kaggle/21_train_v1fused` the moment the quota resets.**

```bash
bash eda/preflight.sh && kaggle kernels push -p kaggle/21_train_v1fused
```

It needs nothing local: the fused labels live in the Kaggle dataset
`achelijndiamantidis/knee-phase1-fused` and the caches are kernel sources.

Two independent lines of evidence now point at it, which is why it is first:

- the fused teacher scores **+0.070** above the lexicon on the 58 expert
  studies (`STATUS.md` §1C);
- a trained head on frozen embeddings gains **+0.062 mean across three seeds**
  from the same swap, direction consistent 3/3 (`EXPERIMENTS.md` E030).

Then, in order: submit the 4-fold rank-mean ensemble, complete the gold pool at
fold 0, re-examine DINOv2 to convergence.

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
