"""The pooling script's job is to refuse bad input, so that is what is tested.

The integrity check matters more than the arithmetic. Five folds must partition
the cohort; if a study appears twice, pooling averages a held-out prediction
with an in-sample one and the result reads as a strong model. That is a silent
fabrication, and a warning would not be enough.
"""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]


def _fold(tmp, fold, studies, rng, signal=1.0):
    y = (rng.random((len(studies), 12)) < 0.3).astype(int)
    p = y * signal + rng.normal(0, 1.0, y.shape)
    d = pd.DataFrame({"StudyInstanceUID": studies, "fold": fold})
    for j, f in enumerate(FINDINGS):
        d[f] = p[:, j]
    for j, f in enumerate(FINDINGS):
        d[f"y_{f}"] = y[:, j]
    path = tmp / f"oof_fold{fold}.csv.gz"
    d.to_csv(path, index=False, compression="gzip")
    return path


def _run(*args):
    return subprocess.run([sys.executable, "eda/pool_oof.py", *map(str, args)],
                          capture_output=True, text=True)


@pytest.fixture
def folds(tmp_path):
    rng = np.random.default_rng(0)
    ids = [f"s{i:04d}" for i in range(500)]
    return [_fold(tmp_path, f, ids[f * 100:(f + 1) * 100], rng) for f in range(5)]


def test_a_clean_partition_is_accepted_and_reports_every_study_once(folds):
    r = _run(*folds)
    assert r.returncode == 0, r.stderr
    assert "500 studies, each exactly once" in r.stdout


def test_a_study_in_two_folds_is_refused_not_warned(tmp_path):
    rng = np.random.default_rng(1)
    ids = [f"s{i:04d}" for i in range(200)]
    a = _fold(tmp_path, 0, ids[:100], rng)
    b = _fold(tmp_path, 1, ids[50:150], rng)   # 50 studies overlap
    r = _run(a, b)
    assert r.returncode != 0
    assert "do not partition the cohort" in (r.stdout + r.stderr)


def test_the_lateral_pair_is_held_out_of_the_macro_by_default(folds):
    r = _run(*folds)
    assert "macro over 10 findings" in r.stdout
    assert "instance-dependent" in r.stdout
    r2 = _run(*folds, "--include-lateral")
    assert "macro over 12 findings" in r2.stdout


def test_a_missing_target_column_is_refused(tmp_path):
    rng = np.random.default_rng(2)
    p = _fold(tmp_path, 0, [f"s{i}" for i in range(50)], rng)
    d = pd.read_csv(p)
    d = d.drop(columns=["y_ACL"])
    d.to_csv(p, index=False, compression="gzip")
    r = _run(p)
    assert r.returncode != 0
    assert "no target column" in (r.stdout + r.stderr)


def test_a_stronger_model_wins_the_paired_bootstrap(tmp_path):
    """Same studies and same labels on both sides, different score quality."""
    ids = [f"s{i:04d}" for i in range(400)]
    rng = np.random.default_rng(3)
    y = (rng.random((400, 12)) < 0.3).astype(int)
    noise = rng.normal(0, 1.0, y.shape)

    def write(dirname, signal):
        d = tmp_path / dirname
        d.mkdir()
        out = []
        for f in range(5):
            lo, hi = f * 80, (f + 1) * 80
            frame = pd.DataFrame({"StudyInstanceUID": ids[lo:hi], "fold": f})
            for j, name in enumerate(FINDINGS):
                frame[name] = y[lo:hi, j] * signal + noise[lo:hi, j]
            for j, name in enumerate(FINDINGS):
                frame[f"y_{name}"] = y[lo:hi, j]
            path = d / f"oof_fold{f}.csv.gz"
            frame.to_csv(path, index=False, compression="gzip")
            out.append(path)
        return out

    strong, weak = write("strong", 2.5), write("weak", 0.4)
    r = _run(*strong, "--vs", *weak)
    assert r.returncode == 0, r.stderr
    line = next(ln for ln in r.stdout.splitlines() if "P(first better)" in ln)
    assert "P(first better) 1.000" in line, line
    assert line.strip().startswith("+"), line
