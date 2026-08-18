# 00 — DICOM header scan (CPU)

Headers only, no pixel data, no GPU. One output row per series.

## Why this is on the critical path

There is no site label in this competition. Not in the CSVs, and not in the
DICOM headers either — `InstitutionName`, `StationName`, `DeviceSerialNumber`
and every date are stripped (`docs/FINDINGS.md` §5.1). So a site-grouped fold,
which the strategy treats as non-negotiable, has nothing to group on until this
kernel runs.

What survives is a **scanner fingerprint**:

```
Manufacturer | ManufacturerModelName | SoftwareVersions
             | MagneticFieldStrength | ImagingFrequency | TransmitCoilName
```

`ImagingFrequency` is the Larmor frequency in MHz. It is set by the magnet's
exact field, so two nominally identical 1.5 T scanners differ at the fourth
decimal — in a 7-file sample the values were 63.881601, 63.870660, 63.685261 and
63.648174, i.e. four distinct magnets. That makes the concatenated key a good
stand-in for a hardware serial number, and it is what the fold grouping and the
leakage audit use.

The scan also records `PatientID`. It is pseudonymised, but if one pseudonym
spans several studies then folds must group on patient as well, or the same knee
lands on both sides of a split. Cheap to check here, expensive to discover later.

## Running it

```bash
kaggle kernels push -p kaggle/00_dicom_header_scan
kaggle kernels status existentialistlogarithmic/knee-dicom-header-scan
kaggle kernels output existentialistlogarithmic/knee-dicom-header-scan -p artifacts/00_header_scan
```

Locally, against a directory laid out like the Kaggle mount:

```bash
python kaggle/00_dicom_header_scan/run.py --root data --out artifacts/series_headers.parquet
```

Sharding, for when a session dies:

```bash
python run.py --shard 0 --of 4      # shards split by study, never within one
```

`--time-budget` (default 8 h) stops the scan cleanly and writes what it has, so
an over-running session costs a re-run rather than the whole result.

## Outputs

- `series_headers.parquet` — one row per series: the fingerprint fields,
  geometry (`PixelSpacing`, `SliceThickness`, `SpacingBetweenSlices`, `Rows`,
  `Columns`), sequence identity, `Laterality`, `PatientID`, slice count, and a
  `plane_from_headers` column derived from `ImageOrientationPatient`.
- `scan_manifest.json` — series scanned, error count, total slices, wall clock,
  seconds per series, output size. The cost of the scan is recorded, not
  estimated.

`plane_from_headers` is deliberately redundant: the host already publishes
`Anatomical_Plane` in `train_series.csv`. Where the two disagree, one of them is
wrong, and it is much cheaper to find that out here than after training on it.

## Status

Written and tested (`tests/test_header_scan.py`, plus a real run over 7 sampled
DICOM files covering 5 manufacturers). **Not yet pushed to Kaggle.**
