# Handoff — everything a fresh session needs

Written 2026-08-20. Read this first, then `docs/STATUS.md` and `docs/PATH.md`.

## Identity and locations

| | |
|---|---|
| repo | `existentialistlogarithmic/Knee-abnormality`, branch **`rsna-knee-abnormality-pipeline`** (77 commits) |
| repo visibility | **private** |
| Kaggle account | **`achelijndiamantidis`** (not the GitHub name) |
| competition | `rsna-knee-abnormality-detection` |
| working dir | `/home/user/Knee-abnormality` |
| Kaggle token | **NOT in this repo, deliberately.** Regenerate at <https://www.kaggle.com/settings/api>, export as `KAGGLE_API_TOKEN`. A token committed here would be a credential in a repo that may go public. |

Links:
- competition — <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>
- efficiency LB — <https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb>
- kernels — <https://www.kaggle.com/achelijndiamantidis/code>
- datasets — `achelijndiamantidis/knee-phase1-artifacts`, `achelijndiamantidis/knee-phase1-fused`

## Where it stands

**Leaderboard 0.725.** Field top 0.952, 1,866 teams. Final submission **2026-10-22**.
Gold out-of-fold **0.7201** at n=58, which tracks the board to **+0.005** — so it
is a usable leaderboard estimator and submissions are confirmation, not search.

| submitted | LB |
|---|---:|
| constant priors | 0.500 |
| scanner metadata, no pixels | 0.531 |
| **imaging resnet34 2.5D 192px, 1 fold** | **0.725** |
| imaging 288px, effective batch 4 | 0.668 |
| imaging 288px, effective batch 16 | 0.688 |

## Hard constraints

- **GPU: 30 h/week, currently EXHAUSTED.** Error is `Maximum weekly GPU quota of
  30.00 hours reached` — different from `Maximum batch GPU session count of 2`,
  which is concurrency and clears by waiting. The API exposes neither the
  remaining balance nor the reset time; only the account page does.
- **CPU sessions are a separate allowance and still work.**
- Submissions: **2 per day**.
- It is a code competition: inference must be a Kaggle notebook, internet off,
  ≤9 h. Training may happen anywhere — the top public notebooks train off-Kaggle.
- **T4 only.** `machine_shape: "NvidiaTeslaT4"`. A P100 fails outright.
- **No report text to any hosted LLM API** (Rule 4.b). This binds an AI assistant
  working on the repo too — print aggregates, never a report string.

## The one command to run before pushing

```bash
bash eda/preflight.sh     # ruff, pytest, generated-tree check, tracked-data check
```

CI runs ruff, not pyflakes. That gap caused a red build once.

## Architecture in one paragraph

2.5D: a 2D backbone over slice stacks, attention-pooled to a study, 12 heads.
resnet34, 21.4M params, input `(3 planes, 20 slices, 192, 192)` — 60 images per
study. Only 58 of 4,407 studies have expert labels, so a **report labeler** reads
the other 4,349 into training targets ("the teacher"). No report text exists at
test time.

`kaggle/` is **generated**: `src/pipeline.py` declares the pipeline,
`eda/generate_kernels.py` renders 35 kernels from 3 templates + 4 shared modules
in `kaggle/_templates/`. Never hand-edit a generated `run.py` — `--check` fails.

## What is measured

| | value |
|---|---|
| teacher (lexicon ∪ LLM), gold-58 | **0.8145** — lexicon alone 0.7446, LLM alone 0.7421 |
| model vs teacher | model **beats** teacher on diffuse findings, **loses** on focal |
| weakest findings | Medial Meniscus **0.516** (teacher 0.903), MCL 0.612, ACL 0.662 |
| strongest | Effusion 0.924, Baker's 0.830, Medial OA 0.817 |
| recoverable gap | **+0.103** macro if the model matched its teacher |
| efficiency prize | `AUC/(Bench−maxAUC)+sec/32400`, minimised → an extra hour costs **0.0502 AUC**. At 0.8 h this project already beats a public 0.883/4 h model. |

## The CPU rig — use it, it is why GPU quota stopped being a blocker

Fine-tuning on CPU is 191 h. **Frozen** extraction is 2.2–2.6 h once, and then a
five-fold A/B of the 73k parameters above the backbone is **2.6 minutes**.

```bash
python eda/head_lab.py --embeddings artifacts/embed_dino/embeddings.npy \
    --index artifacts/embed_dino/embeddings_index.json --compare all
```

Results so far, DINOv2 ViT-S/14 frozen features, paired on 58 gold studies:

| comparison | delta | 95% CI | verdict |
|---|---:|---|---|
| **fused labels − lexicon labels** | **+0.0508** | **[+0.001, +0.102]** | **SEPARATED — act on this** |
| per-finding attention maps − baseline | +0.0389 | [−0.009, +0.090] | not separated |
| FOCAL_K=3 − baseline | +0.0056 | [−0.039, +0.049] | not separated |

## Next actions, in order

1. **Train on the fused labels.** The only separated result the rig has produced.
   `kaggle/21_train_v1fused` is built and waiting on quota.
2. **Submit the 4-fold rank-mean ensemble** (`kaggle/11_infer_folds`) — same
   config, different splits, no hypothesis that can be wrong. Needs one GPU run.
3. **DINOv2 to convergence, 5 folds.** It reached 0.7041 at epoch 34 and was
   still climbing. Largest published lever (+0.09) and has never had a fair run.
4. **A third report reader.** Leaders use three; this has two. CPU only.
5. If an HPC cluster becomes available, move training there — the code already
   takes `--cache/--labels/--headers/--out` and only three `/kaggle/` paths are
   hardcoded, all in the input-discovery helper.

## Things that were believed and turned out false

Read `docs/STATUS.md` §3 for the full ledger. The pattern that matters: **a
second implementation of a shared idea is where the bug lives** — this session
had three (`to_rank` broke on ties, `fuse` understated the teacher by 0.08,
preprocessing crashed the DINOv2 run). And **a small number of observations read
as a trend** cost 3.6 GPU-hours on a run that gained nothing.
