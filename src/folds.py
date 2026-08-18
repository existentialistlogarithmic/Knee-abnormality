"""Fold grouping. The single source of truth for every split in this project.

There is no site label in this competition — not in the CSVs, and not in the
DICOM headers, which are de-identified (`docs/FINDINGS.md` §5.1). The grouping
key is therefore a *scanner fingerprint* assembled from what survives.

One detail decides whether this works at all
--------------------------------------------
`ImagingFrequency` is the Larmor frequency, and at full precision it drifts
between sessions on the same magnet: the Philips Ingenia 3T scanners in this
dataset produce **739 distinct raw values across 2,480 series**. Grouping on the
raw value would create a fingerprint that is nearly unique per study, and a
"grouped" K-fold built on it would be a random K-fold wearing a disguise — the
exact failure this whole scheme exists to prevent, and an invisible one.

Rounding to 2 decimal places collapses that Ingenia cluster to 4 values while
still separating different magnets of the same model. Erring coarse is the safe
direction: over-merging makes the validation task harder than reality, while
under-merging leaks.
"""

from __future__ import annotations

import pandas as pd

FREQUENCY_DECIMALS = 2

FINGERPRINT_FIELDS = [
    "manufacturer",
    "model_name",
    "software_versions",
    "field_strength",
    "imaging_frequency_rounded",
    "transmit_coil",
]


def add_scanner_fingerprint(headers: pd.DataFrame,
                            decimals: int = FREQUENCY_DECIMALS) -> pd.DataFrame:
    """Add `imaging_frequency_rounded` and `scanner_fingerprint` to a header table."""
    out = headers.copy()
    frequency = pd.to_numeric(out.get("imaging_frequency"), errors="coerce")
    out["imaging_frequency_rounded"] = frequency.round(decimals)
    out["scanner_fingerprint"] = (
        out[FINGERPRINT_FIELDS].astype("string").fillna("?").agg("|".join, axis=1)
    )
    return out


def study_groups(headers: pd.DataFrame, decimals: int = FREQUENCY_DECIMALS) -> pd.Series:
    """One grouping key per study.

    A study can in principle span scanners; in practice it does not, so the
    modal fingerprint across the study's series is used and any study that
    disagrees with itself is still assigned a single group.
    """
    marked = add_scanner_fingerprint(headers, decimals)
    modal = (
        marked.groupby("StudyInstanceUID")["scanner_fingerprint"]
        .agg(lambda s: s.value_counts().index[0])
    )
    modal.name = "scanner_fingerprint"
    return modal
