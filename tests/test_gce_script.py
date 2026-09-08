"""The GCE trainer must be the generated one, and must not leave a VM running.

Three failure modes, in cost order:

1. **A VM left running.** A two-hour fold on a machine nobody stops bills until
   someone notices. The shutdown must fire on failure too, not only on success.
2. **A reimplemented training loop.** Then a GCE fold cannot be pooled with a
   Kaggle one and every number in EXPERIMENTS.md becomes incomparable.
3. **A committed credential.** The Kaggle token arrives as instance metadata
   supplied at create time; a literal token in the repo is a leak that survives
   in git history.

No patient data and no secrets: this reads the script text only.
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "gcp" / "train_fold_on_gce.sh"
README = REPO_ROOT / "gcp" / "README.md"


def text() -> str:
    return SCRIPT.read_text()


def test_the_vm_stops_even_when_the_run_fails():
    body = text()
    assert "trap " in body and "shutdown" in body, (
        "the shutdown must be a trap, not a final line: a failing run that "
        "skips it bills until someone notices"
    )
    trap_line = next(line for line in body.splitlines() if line.startswith("trap "))
    assert "EXIT" in trap_line, trap_line


def test_it_runs_the_generated_script():
    body = text()
    assert re.search(r'python3 "\$DIR/run\.py"', body)
    for reimplementation in ("nn.Module", "optim.AdamW", "for epoch in"):
        assert reimplementation not in body, reimplementation


def test_it_does_not_override_the_model_constants():
    body = text()
    for flag in ("--epochs", "--lr ", "--batch", "--backbone"):
        assert flag not in body, f"{flag} must come from src/pipeline.py"


def test_every_kernel_directory_it_names_exists():
    named = sorted(set(re.findall(r"kaggle/\d+_train_[a-z0-9_]+", text())))
    assert named, "the script names no trainer"
    for directory in named:
        assert (REPO_ROOT / directory / "run.py").exists(), directory


def test_it_verifies_the_gpu_and_the_pipeline_before_training():
    body = text()
    assert "nvidia-smi" in body, "a CPU VM would run for days before anyone noticed"
    assert "generate_kernels.py --check" in body, "a drifted run.py is not the pipeline"


def test_it_waits_for_the_driver_before_declaring_no_gpu():
    """DLVM images install the NVIDIA driver on first boot, after the startup
    script has already begun. A bare `nvidia-smi ||` exits in the first minute on
    a machine that has a perfectly good GPU, and the error reads like a wrong
    accelerator flag."""
    body = text()
    assert "for attempt in $(seq 1 60)" in body
    assert "sleep 10" in body
    # the fatal must be inside the retry, not before it
    before_loop = body.split("for attempt in")[0]
    assert "FATAL: no GPU" not in before_loop


def test_no_credential_is_committed():
    for path in (SCRIPT, README):
        body = path.read_text()
        assert "KGAT_" not in body, f"{path.name} contains a literal Kaggle token"
        assert re.search(r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9]{16,}", body) is None


def test_the_script_is_executable():
    assert SCRIPT.stat().st_mode & stat.S_IXUSR


def test_the_readme_states_the_one_account_rule_and_the_quota_trap():
    body = README.read_text()
    assert "multiple accounts" in body
    assert "zero GPU quota" in body, (
        "a new project cannot start a GPU VM at all; that belongs above the fold"
    )
    assert "not free" in body.lower()
