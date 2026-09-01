# HANDOFF — read this first

The session entry point. Everything here is either a live instruction or a
pointer to the file that holds the detail. Last updated **2026-08-31**.

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

**0.923 on the leaderboard** as of 2026-08-29 (E045), up from 0.846 and from
0.725. **This clears the top-200 cut of 0.917.** Field top 0.952, 1,866 teams,
final submission 2026-10-22.

The standing system is `v1public`: resnet34 2.5D at 192px, five folds,
rank-mean, trained on **publicly shared CC0 report labels**
(`stevenleehans/rsna-knee-llm-report-labels`, repackaged with attribution as
`knee-phase1-public`). Gold OOF 0.8980.

**Labels are the whole story.** Decomposed on ground truth, of the +0.198 from
this project's first imaging model: ensembling +0.032, this project's fused
labels +0.089, public CC0 labels +0.077 — **+0.166 of +0.198 is labels.** Every
architectural lever tried (288px, DINOv2, focal top-k, per-finding pooling, two
blend families) measured zero or negative.

**Gold OOF ranks models; it does not forecast a score.** Offsets to the board
have been +0.005, +0.054, +0.025. E038's correction (`board ≈ gold_OOF + 0.032 +
0.005`) got its first close call here — predicted 0.935, actual 0.923.

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

**Standing system is unchanged: `v1public`, five folds, 0.923.** Nothing
measured since has beaten it. `v1pubpool` (per-finding pooling) was queued on
the rig's recommendation and **dropped at folds 0-1** by E054's pre-registered
rule — see E057. `v1publicB` (a second seed) is generated but never pushed;
its lineage is still in the manifest.

**0.94 needs +0.017.** `PATH.md` §4 says that means no finding below ~0.870.
After E044 only **Synovitis (0.779)** is below 0.80; eight of twelve clear 0.870.

Two untried routes, both cheap:

1. **CC0 public weights, ~0.9 GPU-h.** E042 measured that system's OOF at
   **0.8576** standalone on our 58; E043 cleared the licensing — the fold
   checkpoints are **CC0-1.0** in `mattiaangeli/knee-mri-fold-weights` and
   `pilkwang/rsna-knee-weights`. Pull from those public datasets, **never** from
   `tonylica/…repro-assets` (a private consolidation with `not-declared` files).
   Needs a new inference kernel wired in `src/pipeline.py`.
2. **Blend that system with this one.** E042 measured +0.0046 against the
   *0.846* system, not separated — but this system is 0.107 stronger, so the
   blend is worth re-measuring on gold before spending a submission.

**Not worth quota**: the stale `v1fused` retrain (folds 2–4 on old labels),
superseded twice over.

**Still unseparated (E044)**: better labels and ~53% more supervised slots
arrived together. A masked-coverage run (~7 GPU-h) would tell them apart. It is
a question about *why*, not a route to a higher score.

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
| do per-finding attention maps help | rig says **+0.035**, replicated three times. **Fine-tuned it is −0.0338 [−0.067, −0.007]** (E057). The rig was wrong by a sign. |
| does focal top-k pooling help | +0.0163 [−0.001, +0.035] alone, −0.0075 on top of pooling (E056) — **rig-only, and the rig is not trusted on heads** |
| does slice position help | +0.0035 [−0.017, +0.025] — rig-only. Separately: the model is *exactly* permutation-invariant over slices (E050), which is arithmetic and does hold. |

**READ E057 AND E058 BEFORE TRUSTING ANY ROW ABOVE EXCEPT THE FIRST.** The
rig's standing claim — absolute numbers do not transfer but *comparisons above
the backbone* do — was checked against a fine-tuned run for the first time and
**inverted**:

| | per-finding − baseline |
|---|---:|
| rig, frozen DINOv2 | +0.0338 [+0.009, +0.061] |
| rig, frozen resnet34 | +0.0528 [+0.013, +0.097] |
| **fine-tuned resnet34** | **−0.0338** [−0.067, −0.007] |

E057 blamed the backbone mismatch. E058 tested that by rebuilding the bank on
resnet34 — and matching it made the disagreement **wider**, not narrower. So the
split is **freezing**, not architecture. On frozen features the encoder cannot
adapt, so a richer head is the only way to get more out of fixed inputs and head
capacity is rewarded on its own merits. Fine-tuned, the encoder adapts to the
head it has, and extra head capacity buys parameters and overfitting instead.
**The rig systematically over-values head capacity, because head capacity is the
only capacity it has.**

- **Labels: trust it.** A label comparison changes the target, not the head's
  capacity, so freezing does not bite. Called the fused labels at +0.0508; the
  board paid +0.089.
- **Architecture: do not act on it**, and matching the backbone does not fix
  this. Every architecture verdict here — E030 and E052's positives, E029's
  negatives — is a statement about frozen heads. A positive means "not ruled
  out", never "worth GPU".

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

This project has overturned **six** of its own confident claims, and every one
was caught by a measurement rather than by reasoning. The recurring shape is a
small number of observations read as a trend, or a number measured as a
*ceiling* and later used as a *gain* — E030 is the most recent instance.

So: every comparison is one-variable by construction, a test asserts it, and a
claim carries the interval it was measured with. **A negative carries its
interval width too, or it is not a negative** — E053 is the seventh overturned
claim and the second where an effect the instrument could not resolve was
written down as an effect that was not there. `docs/FINDINGS.md` tags every
claim `VERIFIED` / `UNVERIFIED` / `CONTRADICTED` / `CORRECTED`; `EXPERIMENTS.md`
is append-only and a run with no entry did not happen.
