"""The fused label file, checked through the training kernel's own loading path.

Built from the union of the lexicon labeler and the LLM reader, which beat
either alone by +0.070 on the 58 expert-labelled studies (E023). A schema
problem here would not raise on Kaggle — it would train a model on quietly wrong
targets and cost a GPU session to notice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FUSED = REPO_ROOT / "artifacts" / "kaggle_dataset_fused" / "soft_labels.parquet"
LEXICON = REPO_ROOT / "artifacts" / "kaggle_dataset" / "soft_labels.parquet"

pytestmark = pytest.mark.skipif(
    not FUSED.exists(), reason="fused labels not built in this checkout")


def load_as_the_kernel_does(path: Path):
    """The exact sequence from kaggle/04_train/run.py."""
    import pandas as pd

    from src.report_schema import FINDINGS

    soft = pd.read_parquet(path).set_index("StudyInstanceUID")
    studies = list(soft.index)
    targets = soft.loc[studies, FINDINGS].to_numpy(dtype=np.float32)
    channels = soft.loc[studies, [f"{f}__channel" for f in FINDINGS]].to_numpy()
    masks = np.ones_like(targets, dtype=np.float32)
    masks[channels == "absent"] = 0.0
    columns = [f"{f}__weight" for f in FINDINGS]
    if all(column in soft.columns for column in columns):
        confidence = soft.loc[studies, columns].to_numpy(dtype=np.float32)
        masks = masks * np.nan_to_num(confidence, nan=1.0)
    targets = np.nan_to_num(targets, nan=0.0).astype(np.float32)
    return targets, masks


def test_the_fused_file_loads_through_the_kernel_path_without_nans():
    targets, masks = load_as_the_kernel_does(FUSED)
    assert not np.isnan(targets).any()
    assert not np.isnan(masks).any()


def test_targets_are_probabilities_and_masks_are_non_negative():
    targets, masks = load_as_the_kernel_does(FUSED)
    supervised = masks > 0
    assert 0.0 <= targets[supervised].min() and targets[supervised].max() <= 1.0
    assert masks.min() >= 0.0


def test_an_abstaining_slot_carries_no_target_the_loss_could_see():
    """The whole point of the abstain channel: silence must not be taught as a
    zero. A masked slot with a non-zero target would be harmless today because
    the mask zeroes it, and a landmine the moment anyone reads targets directly.
    """
    targets, masks = load_as_the_kernel_does(FUSED)
    assert (targets[masks == 0] == 0).all()


def test_the_fusion_supervises_more_than_the_lexicon_alone():
    """The reason it exists. If this ever regresses, the fusion is not doing its
    job whatever its gold AUC says."""
    if not LEXICON.exists():
        pytest.skip("lexicon labels not present in this checkout")
    _, fused = load_as_the_kernel_does(FUSED)
    _, lexicon = load_as_the_kernel_does(LEXICON)
    assert (fused > 0).mean() > (lexicon > 0).mean() + 0.10, (
        f"fused supervises {(fused > 0).mean():.1%} of slots against the "
        f"lexicon's {(lexicon > 0).mean():.1%}")


def test_no_column_could_hold_report_text():
    """This file ships as a Kaggle Dataset. A report string in it is
    patient-derived text leaving the machine (STRATEGY.md rule 4)."""
    import pandas as pd

    frame = pd.read_parquet(FUSED)
    for column in frame.columns:
        if frame[column].dtype == object:
            longest = frame[column].astype(str).str.len().max()
            assert longest < 40, f"{column} holds strings up to {longest} chars"
