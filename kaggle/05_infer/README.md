# 05 — inference (submission kernel)

**`enable_internet: false`.** This is the kernel that gets submitted, so it must
run offline. Weights arrive by mounting the training kernel's output through
`kernel_sources`; nothing is downloaded.

## Why it builds volumes in memory

No test cache is written. At ~1,300 studies that would be ~3 GB of intermediate
files for no benefit. Each study is built, predicted, and discarded. The measured
budget is ~24 s/study against the 9-hour cap, and reaching the data costs
0.059 s/study, so the headroom is large.

## Why the code is copied rather than imported

A Kaggle script kernel is one file. The volume-building functions are lifted
verbatim from `03_cache_build/run.py` and the model from `04_train/run.py`,
because re-typing either is how train/inference skew appears — the model quietly
receives different input than it learned from, nothing raises, and the score is
just worse for invisible reasons.

`tests/test_infer_contract.py` pins that contract: the geometry constants must
match the cache build, the plane order must match, the finding order must match
`sample_submission.csv`, the model must accept exactly the array shape the cache
produces, and the submission assertions must run *before* the file is written.

## Failure policy

A study that cannot be read keeps the fallback prior and increments a counter,
rather than raising. One bad study should cost a little AUC, never the whole
submission.
