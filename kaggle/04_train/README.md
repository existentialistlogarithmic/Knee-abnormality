# 04 — training (T4 GPU)

2.5D backbone over the cached volumes, attention-pooled to study level, 12 heads.

## Not yet runnable

It needs the cache from `kaggle/03_cache_build` published as a Dataset and added
to `dataset_sources`. The pilot is what decides how many shards that takes.

## Why `enable_internet: true` here

Only *submission* kernels must have internet disabled. Training may download
pretrained backbone weights; those weights are then saved into the kernel output
and mounted by the inference kernel, which runs with internet off. That is the
standard Kaggle-to-Kaggle pattern and the reason the inference kernel never
needs a download.

## What it prints on the first run, and why it matters

`report_environment()` logs the accelerator and its compute capability. Which GPU
a kernel actually receives has been `UNVERIFIED` for this project, because the
Kaggle CLI does not expose the valid `machine_shape` strings. A pre-Volta card
(compute capability < 7, i.e. P100) would **fail** rather than run slowly, since
the current PyTorch build ships no Pascal kernels. The first run settles it.

## The three rules baked in

1. **Folds are scanner-grouped.** Random K-fold inflates macro AUC by 0.087
   (`FINDINGS.md` §9).
2. **Abstain masks the loss.** A report silent on a finding contributes no
   gradient for it, rather than teaching a negative. This is the whole reason
   the labeler emits five channels instead of a probability.
3. **Gold studies carry 8x weight** and override report-derived targets, being
   the only labels known to match what is scored.

## Reading the result

The bar is **0.669** — what scanner metadata alone achieves with no pixels. Below
that, the images are contributing nothing. And report-label CV overstates the
leaderboard by about **0.138**, so CV ranks configurations; it does not predict
the board.
