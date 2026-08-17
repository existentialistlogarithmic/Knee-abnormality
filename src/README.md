# src/

Empty by design. Nothing here gets written until Phase 0 is reviewed — the whole
point of the verification gate is that model and label code is not built on
second-hand assumptions.

Planned contents, in order of priority:

| file | phase | purpose |
|---|---|---|
| `report_labeler.py` | 1 | multilingual report → 12-finding soft labels + abstain channel |
| `lexicons/*.csv` | 1 | per-language term tables, human-reviewable (the one data class this repo *does* commit — no patient text, only vocabulary) |
| `folds.py` | 1 | site/scanner-grouped fold assignment, the single source of truth for every split |
| `calibration.py` | 1 | in-fold calibration of report scores; never fitted on the gold subset |
| `dicom_cache.py` | 2 | DICOM → normalised, laterality-corrected, resampled volumes |
| `model.py` | 2 | 2.5D backbone + attention pooling → 12 heads |
| `train.py` | 2 | resumable, checkpointed, wall-clock guarded |
| `infer.py` | 3 | submission kernel entry point + pre-submit sanity checks |
