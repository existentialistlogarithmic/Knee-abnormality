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
| `id` | `<username>/<slug>` — must be your own username |
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
