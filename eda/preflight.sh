#!/usr/bin/env bash
# Everything CI will check, run before a push so the failure arrives here
# rather than on the Actions tab ten minutes later.
#
#   bash eda/preflight.sh
#
# The four gates below are exactly what .github/workflows/tests.yml enforces,
# in the same order, so a green run here means a green run there. The kernel
# drift check is the one that matters most: kaggle/*/run.py is generated from
# src/pipeline.py and must never be hand-edited, and a drifted kernel is a
# silent failure — it pushes and runs, it just is not the pipeline any more.
#
# Exit codes: 0 all gates pass, 1 a gate failed.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

fail=0
run() {
  local name=$1; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then
    printf 'ok    %s\n' "$name"
  else
    printf 'FAIL  %s\n' "$name"
    fail=1
  fi
}

run "lint"           python3 -m ruff check eda src tests
run "tests"          python3 -m pytest tests -q
run "kernel drift"   python3 eda/generate_kernels.py --check

# Patient-derived files must never enter git history: file names and tabular
# columns carry StudyInstanceUIDs and report text. src/lexicons/ holds
# vocabulary only and docs/ holds aggregates, so both are whitelisted.
printf '\n=== no patient-derived files tracked ===\n'
bad=$(git ls-files | grep -E '\.(dcm|dicom|nii|npy|npz|parquet|pt|pth|ckpt)$' || true)
csv=$(git ls-files '*.csv' | grep -v '^src/lexicons/' | grep -v '^docs/' || true)
if [ -n "$bad$csv" ]; then
  echo "FAIL  patient-derived files are tracked in git:"
  printf '%s\n' "$bad" "$csv"
  fail=1
else
  echo "ok    no patient-derived files tracked"
fi

printf '\n'
if [ "$fail" -ne 0 ]; then
  echo "preflight FAILED — do not push"
  exit 1
fi
echo "preflight passed"
