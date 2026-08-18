# Kaggle kernels

One folder per kernel. Each folder holds a `kernel-metadata.json` and the code
file it points at. Push with:

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

### Requesting a T4 — how to get the exact string

Kaggle's PyTorch build ships no Pascal kernels, so P100 is unusable and the
accelerator must be a T4. The CLI does **not** expose the list of valid
`machine_shape` values — the source comments that the enum "is not currently
included in kagglesdk" — so the correct string is `UNVERIFIED` and must not be
guessed. Two ways to obtain it:

1. Set the accelerator in the Kaggle notebook UI, then
   `kaggle kernels pull <username>/<slug> -p /tmp/probe -m`. The pull writes the
   server's own `machine_shape` value into the metadata file. Copy it verbatim.
2. Push a throwaway kernel with a deliberately invalid accelerator and read the
   allowed values back out of the error.

Whichever you use, record the result in `docs/FINDINGS.md` before any GPU kernel
is pushed.

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
