"""The Colab notebook must drive the real generated trainer, not a copy.

Its whole value is that a fold trained on Colab pools with one trained on
Kaggle. Three ways that breaks, each silent:

1. **It points at kernel directories that no longer exist**, so it fails at the
   last step of a three-hour setup rather than the first.
2. **It fetches the wrong teacher.** A fold trained against the wrong labels
   produces a checkpoint that loads fine, pools without complaint, and confounds
   the one variable its lineage exists to isolate — E040's confound.
3. **It reimplements the training loop.** Then every number in EXPERIMENTS.md
   becomes incomparable and nothing in this repo means what it says.

No patient data: this reads the notebook and the generated tree only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "colab" / "train_fold_on_colab.ipynb"


def source() -> str:
    nb = json.loads(NOTEBOOK.read_text())
    return "".join("".join(c["source"]) for c in nb["cells"])


def test_every_kernel_directory_it_names_exists():
    named = sorted(set(re.findall(r"kaggle/\d+_train_[a-z0-9_]+", source())))
    assert named, "the notebook names no trainer at all"
    for directory in named:
        assert (REPO_ROOT / directory / "run.py").exists(), (
            f"{directory} does not exist; the notebook would fail after the "
            "cache download, not before it"
        )


def test_it_runs_the_generated_script_rather_than_its_own_loop():
    text = source()
    assert "/run.py'," in text or "/run.py\"," in text
    for reimplementation in ("nn.Module", "optim.AdamW", "for epoch in range"):
        assert reimplementation not in text, (
            f"the notebook defines its own training ({reimplementation}); a "
            "Colab-specific loop makes every past number incomparable"
        )


def test_it_does_not_override_the_model_constants():
    """Epochs, LR, batch and geometry come from src/pipeline.py or nothing means
    anything. --time-budget and --fold are the only legitimate overrides."""
    text = source()
    for flag in ("--epochs", "--lr", "--batch", "--backbone"):
        assert flag not in text, f"{flag} must not be settable from the notebook"


def test_it_states_the_one_account_rule():
    """Free compute elsewhere is allowed; a second Kaggle account is not, and
    the notebook is exactly where someone would think of that shortcut."""
    text = source()
    assert "multiple accounts" in text


def test_the_labels_it_downloads_match_the_lineage_it_trains():
    text = source()
    trains_distil = "train_v1distil" in text
    fetches_distil = "knee-phase1-distilled" in text
    assert trains_distil == fetches_distil, (
        "the notebook trains one lineage and downloads another teacher's labels"
    )
