"""Baseline submission kernel — de-risks the pipeline and measures the budget.

This kernel is not trying to score well. Constant predictions give an AUC of
exactly 0.5 whatever constants you choose, and that is fine, because the two
things it *is* for are worth more right now than a few points of AUC:

1. **Prove the submission path works end to end** — data mounts, `test.csv`
   parses, `submission.csv` lands with the right shape, internet off. In a code
   competition this is the component most likely to fail, and it fails at the
   deadline when there is no time to fix it. Getting a scored submission early
   converts that risk into a known-good baseline.

2. **Measure the real per-study cost of touching DICOM data.** The hidden test
   set is ~1,300 studies and the cap is 9 hours, so the budget is ~24 s/study.
   Nobody knows what fraction of that a cold Kaggle mount eats until it is
   measured on the actual filesystem. This kernel times it and extrapolates.

Everything it prints about timing is measured here, not estimated elsewhere.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

OUTPUT = Path("/kaggle/working/submission.csv")

def find_competition_root(explicit: str | None = None) -> Path:
    """Locate the mounted competition data.

    Kaggle does not guarantee the mount directory matches the competition slug,
    and a wrong guess costs a whole kernel run to discover — as it did on the
    first attempt here. So the root is found by looking for the files we know
    must exist, and the search is reported in the log either way.
    """
    if explicit:
        return Path(explicit)
    base = Path("/kaggle/input")
    if not base.exists():
        return Path(".")
    candidates = sorted(p for p in base.iterdir() if p.is_dir())
    print(f"/kaggle/input contains: {[p.name for p in candidates]}")
    for marker in ("train_series.csv", "train.csv", "test.csv"):
        for candidate in candidates:
            if (candidate / marker).exists():
                print(f"using competition root: {candidate}  (found {marker})")
                return candidate
        # the data is sometimes nested one level down
        for candidate in candidates:
            for child in sorted(p for p in candidate.iterdir() if p.is_dir()):
                if (child / marker).exists():
                    print(f"using competition root: {child}  (found {marker})")
                    return child
    print("WARNING: no competition root found; falling back to the slug path")
    return base / "rsna-knee-abnormality-detection"

INPUT = find_competition_root()

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]

HIDDEN_TEST_STUDIES = 1300   # host-stated, data-description page
RUNTIME_CAP_SECONDS = 32400  # 9 hours


def priors(train_path: Path) -> dict[str, float]:
    """Per-finding rate from the expert-labelled studies.

    AUC does not care about the constants, so this is presentation only — but a
    plausible prior beats 0.5 if the metric is ever changed to something
    threshold-based, and it costs nothing.
    """
    fallback = dict.fromkeys(FINDINGS, 0.5)
    if not train_path.exists():
        return fallback
    train = pd.read_csv(train_path, usecols=["StudyInstanceUID", *FINDINGS])
    gold = train[train[FINDINGS].notna().all(axis=1)]
    if gold.empty:
        return fallback
    return {f: round(float(gold[f].mean()), 4) for f in FINDINGS}


def time_dicom_access(series: pd.DataFrame, studies: list[str], sample: int = 40) -> dict:
    """How long does it take to reach the pixels, per study?

    Reads headers only, which is the floor on any real pipeline's cost: an
    imaging model must do at least this much before it decodes anything.
    """
    import pydicom

    subset = list(studies[:sample])
    started = time.time()
    files_seen = 0
    series_seen = 0
    header_reads = 0
    for study in subset:
        for row in series[series.StudyInstanceUID == study].itertuples(index=False):
            directory = INPUT / "test_series" / study / row.SeriesInstanceUID
            try:
                names = [e.name for e in os.scandir(directory) if e.name.endswith(".dcm")]
            except FileNotFoundError:
                continue
            series_seen += 1
            files_seen += len(names)
            if names:
                try:
                    pydicom.dcmread(str(directory / sorted(names)[0]),
                                    stop_before_pixels=True, force=True)
                    header_reads += 1
                except Exception:  # noqa: BLE001 - timing must not die on one bad file
                    pass
    elapsed = time.time() - started
    n = max(len(subset), 1)
    per_study = elapsed / n
    return {
        "studies_timed": len(subset),
        "series_seen": series_seen,
        "dicom_files_seen": files_seen,
        "header_reads": header_reads,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_study": round(per_study, 4),
        "projected_seconds_for_hidden_test": round(per_study * HIDDEN_TEST_STUDIES, 1),
        "projected_share_of_9h_cap": round(per_study * HIDDEN_TEST_STUDIES
                                           / RUNTIME_CAP_SECONDS, 4),
    }


def main() -> int:
    started = time.time()
    test = pd.read_csv(INPUT / "test.csv")
    studies = test.StudyInstanceUID.astype(str).tolist()
    print(f"test studies: {len(studies):,}")

    series_path = INPUT / "test_series.csv"
    series = pd.read_csv(series_path) if series_path.exists() else pd.DataFrame()
    print(f"test series : {len(series):,}")

    values = priors(INPUT / "train.csv")
    print("\npriors from the expert-labelled studies:")
    for finding, value in values.items():
        print(f"  {finding:<18}{value:.4f}")

    timing = {}
    if not series.empty:
        print("\ntiming DICOM access …")
        timing = time_dicom_access(series, studies)
        print(json.dumps(timing, indent=2))
        share = timing.get("projected_share_of_9h_cap")
        if share is not None:
            print(f"\n>>> reaching the data costs {timing['seconds_per_study']:.2f} s/study,")
            print(f">>> i.e. {share:.1%} of the 9-hour cap on a {HIDDEN_TEST_STUDIES}-study")
            print(">>> test set, BEFORE any decoding or inference.")

    submission = pd.DataFrame({"StudyInstanceUID": studies})
    for finding in FINDINGS:
        submission[finding] = values[finding]

    # ---- sanity checks, before writing, not after ---------------------------- #
    assert list(submission.columns) == ["StudyInstanceUID", *FINDINGS], "column order"
    assert len(submission) == len(test), "row count must match test.csv"
    assert submission[FINDINGS].notna().all().all(), "no NaNs allowed"
    assert ((submission[FINDINGS] >= 0) & (submission[FINDINGS] <= 1)).all().all(), "probabilities"
    assert submission.StudyInstanceUID.is_unique, "duplicate study ids"

    sample_path = INPUT / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        assert list(submission.columns) == list(sample.columns), (
            f"columns differ from sample_submission: {list(sample.columns)}")
        print("\ncolumn names and order match sample_submission.csv")

    submission.to_csv(OUTPUT, index=False)
    print(f"\nwrote {OUTPUT}  rows={len(submission):,}  cols={submission.shape[1]}")
    print(f"total kernel wall clock: {time.time() - started:.1f} s")

    Path("/kaggle/working/run_manifest.json").write_text(
        json.dumps({"n_test_studies": len(studies), "priors": values, "timing": timing,
                    "wall_clock_seconds": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
