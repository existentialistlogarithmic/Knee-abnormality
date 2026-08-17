# 00 — DICOM header scan (CPU)

Placeholder. `kernel-metadata.json` is in place; `run.py` is written in Phase 0
step 0.3, once step 0.2 has established the actual DICOM path layout. Do not
push this folder until `run.py` exists — `code_file` points at a file that is
not here yet.

Intended output: one parquet row per **series**, with StudyInstanceUID,
SeriesInstanceUID, series description, modality, plane/orientation (derived from
`ImageOrientationPatient`), slice count, pixel spacing, slice thickness,
manufacturer and model, laterality, and magnetic field strength. Headers only —
no pixel data is read, which is what makes a full-dataset scan feasible on CPU.
