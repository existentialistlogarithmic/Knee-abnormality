# Kaggle kernels

**Most of this directory is generated.** `src/pipeline.py` declares the
pipeline and `eda/generate_kernels.py` renders it:

```bash
python eda/generate_kernels.py            # what would change
python eda/generate_kernels.py --diff     # the diffs
python eda/generate_kernels.py --write    # apply
python eda/generate_kernels.py --check    # exit 1 if the tree has drifted
```

`--check` runs in the test suite, so editing a generated `run.py` by hand fails
CI rather than quietly making the manifest fiction.

Kaggle script kernels are single files, so sharing code between them means
splicing it at generation time. `kaggle/_templates/` holds three templates
(`cache_build`, `train`, `infer`) and three shared modules under `_shared/`.
A template is ordinary Python with two placeholders:

```
@@CONFIG@@                        the constants for this kernel, from the manifest
@@INCLUDE volume@@                the whole of _shared/volume.py
@@INCLUDE discovery:find_marker@@ just that one definition
```

Why it is worth the machinery: `build_study` used to be byte-identical in four
kernels, `find_marker` in seven, and `build_model` had **drifted into two
variants across five files** — the training kernel normalising its input and
the inference kernel that scores those weights not. That does not raise; it
just scores worse. There is now one of each.

**Six folders are not generated** and are edited by hand: `00_dicom_header_scan`,
`01_submission_baseline`, `02_metadata_submission`, `05_infer`, `05_infer_cpu`
and `09_infer_ensemble`. They are one-offs rather than members of a lineage.

Push with:

```bash
kaggle kernels push -p kaggle/<folder>
kaggle kernels status <username>/<slug>
kaggle kernels output <username>/<slug> -p artifacts/<folder>
```

## Chain of custody

The bulk data never touches the local machine. Each kernel mounts the previous
one's output:

```
competition data ──► 00_dicom_header_scan (CPU)  ──► header metadata parquet
                                                          │
                                      (as a Dataset or kernel_sources mount)
                                                          ▼
                                            01_cache_build (CPU) ──► volumes
                                                          │
                                                          ▼
                                            02_train (T4 GPU) ──► weights
                                                          │
                                                          ▼
                                     03_infer (T4, internet off, < 9 h)
```

## Metadata fields that matter here

Field names below are taken from the `kaggle` CLI 2.2.4 source
(`kernels_initialize` / `kernels_push`), not from memory:

| field | note |
|---|---|
| `id` | `<username>/<slug>` — the **Kaggle** username (`achelijndiamantidis`), which is not the same as the GitHub one |
| `kernel_type` | `script` or `notebook` |
| `language` | `python`, `r`, or `rmarkdown` |
| `enable_internet` | **`false`** for anything that will become a submission |
| `enable_gpu` / `enable_tpu` | booleans |
| `machine_shape` | the accelerator string; also settable per-push via `kaggle kernels push --accelerator <name>`, which overrides this field |
| `competition_sources` | e.g. `["rsna-knee-abnormality-detection"]` |
| `dataset_sources`, `kernel_sources`, `model_sources` | how a kernel mounts previous outputs |
| `docker_image`, `docker_image_pinning_type` | pin the image once training results need to be reproducible |

### Requesting a T4 — the exact strings

```
NvidiaTeslaT4     <- use this
NvidiaTeslaP100   <- unusable, see below
Tpu1VmV38
```

Set `machine_shape` in `kernel-metadata.json`. These are documented in the
`kagglesdk` docstring for `machine_shape`
(`kagglesdk/kernels/types/kernels_api_service.py`), **despite the CLI source
claiming the enum "is not currently included in kagglesdk"**. That comment cost
this project several probe runs.

**A P100 does not run slowly — it fails.** `enable_gpu: true` with no shape
gives a P100 (compute capability 6.0), and the Kaggle PyTorch build ships no
Pascal kernels, so the first CUDA launch dies with `no kernel image is available
for execution on the device`.

Two things that do **not** work, recorded so nobody repeats them:

- `--accelerator INVALID_PROBE` is accepted silently. The CLI does not validate
  the value, so you cannot discover the vocabulary from an error message.
- `--accelerator GpuT4x2` is silently ignored and you get a P100 anyway. A wrong
  shape name falls back rather than failing.

Put a device check at the top of any GPU kernel and exit early if the compute
capability is below 7 — see `report_environment()` in `kaggle/04_train/run.py`.
It turns a wasted session into a few seconds.

### Non-negotiable for the submission kernel

`enable_internet: false`, all weights and code mounted from Datasets/Models, and
total runtime comfortably under the 9-hour limit with headroom for a hidden test
set larger than the public one.

## A slug gotcha worth knowing

Kaggle derives the kernel slug from the **title**, not from the `id` you set. A
title of "Knee — DICOM header scan (CPU, headers only)" produces the slug
`knee-dicom-header-scan-cpu-headers-only`, and the push warns but proceeds,
creating the kernel at the title-derived slug. If `id` then disagrees, the next
push creates a *second* kernel rather than updating the first.

Rule: after the first push of any kernel, set `id` to the slug in the URL the
push prints, and leave it alone. Also note the Kaggle account here is
`achelijndiamantidis`, which is not the GitHub username.

## Kaggle API rate limits — pace the polling

The API returns **HTTP 429** readily, and this project has tripped it three
times: paging the competition file listing, polling submission status, and
downloading kernel output.

Practical rules learned the hard way:

- **Poll status no faster than every 30 s**, and prefer a single check after a
  realistic delay over a tight loop. Submission scoring re-runs the whole kernel
  against the hidden test set, so it takes minutes, not seconds.
- **Never download a whole kernel output when you only want the manifest.** Use
  `kaggle kernels output --file-pattern '.*\.(json|parquet)$'`. A cache-build
  kernel emits gigabytes of `.npy` that belong in a Dataset, not on a laptop.
- **Long paginated reads need checkpointing**, not just retries — see
  `eda/phase0_01_auth_and_files.py`, which writes its page token every 25 pages
  so a 429 costs a resume rather than the whole listing.
- Back off exponentially and honour `Retry-After` when present.

## Mounting a kernel's output instead of downloading it

The cache is ~10 GB across four shards. It never touches this machine: the
training kernel lists the cache kernels in `kernel_sources`, and Kaggle mounts
their outputs directly. That is the Kaggle-to-Kaggle chain the project depends
on, and it also sidesteps the output-download rate limit entirely.

```
03_cache_build_shard{0..3}  ──kernel_sources──►  04_train  ──►  weights
                                                                  │
                                                       kernel_sources
                                                                  ▼
                                                            inference kernel
```

Note `kernel_sources` mounts a kernel's **latest** output, which is why the
cache is four separate kernels rather than one kernel run four times.

## Two GPU limits, and they are not the same thing

The accelerator is **2× NVIDIA Tesla T4** per session (compute capability 7.5),
requested with `machine_shape: "NvidiaTeslaT4"`. Both cards are granted to a
single kernel; `torch.cuda.device_count()` returns 2 and `DataParallel` uses
both.

There are **two separate ceilings**, with different error messages, and
confusing them wastes a lot of thinking:

| limit | message | what it means |
|---|---|---|
| concurrency | `Maximum batch GPU session count of 2 reached.` | two kernels are running now; wait and retry |
| **weekly quota** | `Maximum weekly GPU quota of 30.00 hours reached.` | **no more GPU runs this week at any concurrency** |

The first is a queueing problem and `eda/push_queue.sh` handles it. The second is
a budget problem and no amount of waiting inside a session fixes it. This
project spent its 30 hours and hit the second one, having assumed for some time
that it was still hitting the first.

`eda/push_queue.sh` distinguishes them: it retries on `Maximum batch` and stops
on anything else, so a quota exhaustion surfaces as `FAILED` rather than an
endless "waiting for a slot".

**Kaggle's API does not expose remaining quota or the reset time** — the only
place either appears is the account page in the browser. So the reset moment is
`UNVERIFIED` here; the allowance is stated by the error message itself.

**CPU sessions are a separate allowance and keep working** when GPU is
exhausted. Verified by pushing a CPU kernel immediately after a GPU push was
refused for quota. That matters for planning: cache builds, header scans and
CPU inference remain available, and CPU inference was measured at 19.7 s/study
for one model — 7.1 h of the 9 h cap on ~1,300 studies, so it fits for a single
model and does not fit for an ensemble.

## Concurrency limit: 5 CPU sessions

`Kernel push error: Maximum batch CPU session count of 5 reached.` Kaggle runs at
most **5 batch CPU sessions** per account at once. Pushing a sixth is rejected
outright — the kernel is not queued, so a fan-out wider than 5 needs its own
queue. `eda/push_queue.sh` waits for a slot and pushes the rest.

This is why the v1 cache used 4 shards and the v2 cache, at 8, has to drip-feed.
