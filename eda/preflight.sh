#!/usr/bin/env bash
# Everything CI runs, in the order CI runs it. One command, so there is no gap
# between "it passes locally" and "it passes on push".
#
# The gap this closes cost a red build: local checking used pyflakes, CI uses
# ruff, and ruff caught twelve things pyflakes does not look for — including
# five `zip()` calls without `strict=`, which silently truncate to the shorter
# iterable rather than complaining that two lists disagree.
set -euo pipefail

echo "== ruff =="
python3 -m ruff check eda src tests

echo "== pytest =="
python3 -m pytest tests -q

echo "== generated kernels match the manifest =="
python3 eda/generate_kernels.py --check

echo "== no patient-derived file tracked in git =="
bad=$(git ls-files | grep -E '\.(dcm|dicom|nii|npy|npz|parquet|pt|pth|ckpt)$' || true)
csv=$(git ls-files '*.csv' | grep -v '^src/lexicons/' | grep -v '^docs/' || true)
if [ -n "$bad$csv" ]; then
  echo "patient-derived files are tracked in git:"
  printf '%s\n' "$bad" "$csv"
  exit 1
fi
echo "clean"

echo
echo "preflight passed"
