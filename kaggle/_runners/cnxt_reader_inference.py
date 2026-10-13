"""goodpjw2008's public 2.5D ConvNeXt reader as one FOREIGN_ARMS vote (E154).

SOURCING. The model, preprocessing and inference code are goodpjw2008's:

  weights+code  kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader  (Apache-2.0)
  notebook      kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944

This file is OURS and contains none of their code. It only gives their unchanged
`infer.run` the `find_artifact_root` / `run_submission` interface that
`raptor_infer`'s FOREIGN_ARMS block drives, and adds three guards:

* ISOLATION. Their modules are called `infer`, `knee` and `preprocess`, generic
  enough to collide, so they run in a child process with their directory first
  on sys.path and nothing of theirs is ever imported into the parent.
* SAFE LOADING. Upstream calls torch.load(weights_only=False). All three pickles
  were disassembled offline (E150) and carry only OrderedDict/_rebuild_tensor_v2/
  FloatStorage, so the child forces weights_only=True.
* DECODE ACCOUNTING. Upstream's dataset skips any series it cannot decode and
  leaves the slot empty, which reads as a weak model, not as a bug. Every failure
  is appended to a file (the DataLoader forks workers, so an in-memory counter
  would see nothing) and the run aborts above MAX_DECODE_FAIL. E150 measured
  0 / 855 on the gold studies under this image's Python 3.13.
"""
import subprocess
import sys
from pathlib import Path

MARKER = "cnxt_v0_fold0.pt"
EXPECTED_FOLDS = 3
MAX_DECODE_FAIL = 0.05
SKIP = {"train_series", "test_series"}

_CHILD = r'''
import os, sys
import numpy as np
import pandas as pd
artifact, competition, out, ok_log, fail_log = sys.argv[1:6]
sys.path.insert(0, artifact)
import torch
_load = torch.load
def _safe(*a, **k):
    k["weights_only"] = True
    return _load(*a, **k)
torch.load = _safe
import infer
infer.torch.load = _safe
_up = infer.load_series
def _counted(*a, **k):
    try:
        r = _up(*a, **k)
    except Exception:
        with open(fail_log, "a") as f:
            f.write("x\n")
        raise
    with open(ok_log, "a") as f:
        f.write("x\n")
    return r
infer.load_series = _counted
ckpts = sorted(str(p) for p in __import__("pathlib").Path(artifact).glob("cnxt_v0_fold*.pt"))
studies = pd.read_csv(os.path.join(competition, "test.csv")).StudyInstanceUID.tolist()
pred = infer.run(competition, "test", ckpts, studies=studies)
pred = pred.loc[studies]
assert np.isfinite(pred.to_numpy()).all(), "non-finite reader output"
pred.reset_index().to_csv(out, index=False)
print(f"[cnxt-child] {len(studies)} studies, {len(ckpts)} folds", flush=True)
'''


def find_artifact_root(max_depth: int = 4) -> Path:
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        if any(e.is_file() and e.name == MARKER for e in entries):
            return directory
        frontier += [(e, depth + 1) for e in entries
                     if e.is_dir() and e.name not in SKIP]
    raise RuntimeError(f"no {MARKER} under /kaggle/input: is the reader dataset mounted?")


def run_submission(competition_root, artifact_root, output_path) -> dict:
    artifact_root, output_path = Path(artifact_root), Path(output_path)
    folds = sorted(artifact_root.glob("cnxt_v0_fold*.pt"))
    if len(folds) != EXPECTED_FOLDS:
        raise RuntimeError(f"expected {EXPECTED_FOLDS} reader folds, found {len(folds)}")
    work = Path("/kaggle/working")
    ok_log, fail_log = work / "_cnxt_decode_ok", work / "_cnxt_decode_fail"
    for p in (ok_log, fail_log):
        p.unlink(missing_ok=True)
        p.touch()
    r = subprocess.run(
        [sys.executable, "-c", _CHILD, str(artifact_root), str(competition_root),
         str(output_path), str(ok_log), str(fail_log)],
        cwd=str(work), capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"cnxt reader child failed rc={r.returncode}: {r.stderr[-2000:]}")
    with open(ok_log) as f:
        ok = sum(1 for _ in f)
    with open(fail_log) as f:
        failed = sum(1 for _ in f)
    rate = failed / max(ok + failed, 1)
    print(f"[cnxt] series decoded {ok}/{ok + failed} (failed {rate:.1%})", flush=True)
    if rate > MAX_DECODE_FAIL:
        raise RuntimeError(f"cnxt decode failure rate {rate:.1%} > {MAX_DECODE_FAIL:.0%}; "
                           "the reader would read as weak, not broken")
    return {"status": "ok", "decoded": ok, "decode_failed": failed}
