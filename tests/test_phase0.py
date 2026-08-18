"""Tests for the Phase 0 scripts. No credentials, no network, no patient data.

The important ones are the leak tests: they assert that the outputs meant to
travel (CI job summaries, anything pasted into docs) never contain a study
identifier or a line of report text.
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

import fixture  # noqa: E402


def _stub_kaggle(monkeypatch, files=None, meta=None):
    """Install a fake kaggle package so the scripts run offline."""
    import datetime as dt

    class FakeFile:
        def __init__(self, name, size):
            self.name = name
            self.total_bytes = size
            self.creation_date = dt.datetime(2026, 6, 1)

    class FakePage:
        def __init__(self, files):
            self.files = files
            self.next_page_token = None

    class FakeComp:
        ref = "rsna-knee-abnormality-detection"
        evaluation_metric = "UNKNOWN-IN-TEST"
        user_has_entered = True

    class FakeResp:
        competitions = [FakeComp()]

    class FakeApi:
        def authenticate(self):
            return None

        def competitions_list(self, **_):
            return FakeResp()

        def competition_list_files(self, _comp, page_token=None, page_size=200):
            return FakePage([FakeFile(n, s) for n, s in (files or [])])

        def competition_download_file(self, *_a, **_k):
            raise AssertionError("test fixture files are already local")

    module = types.ModuleType("kaggle.api.kaggle_api_extended")
    module.KaggleApi = FakeApi
    monkeypatch.setitem(sys.modules, "kaggle", types.ModuleType("kaggle"))
    monkeypatch.setitem(sys.modules, "kaggle.api", types.ModuleType("kaggle.api"))
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", module)


def _run(script: str, argv: list[str]) -> None:
    sys.argv = [script, *argv]
    path = REPO_ROOT / "eda" / script
    namespace = runpy.run_path(str(path), run_name="__not_main__")
    assert namespace["main"]() == 0


# --------------------------------------------------------------------------- #
# step 1
# --------------------------------------------------------------------------- #
def test_step1_redacts_nested_paths(tmp_path, monkeypatch):
    """The public summary must not leak a StudyInstanceUID; the full one may."""
    uid = "1.2.826.0.1.3680043.4242"
    _stub_kaggle(
        monkeypatch,
        files=[
            ("train.csv", 120_000),
            (f"train_images/{uid}/series1/1.dcm", 512_000),
            (f"reports/{uid}.txt", 4_000),
        ],
    )
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-token-for-tests")
    out = tmp_path / "phase0"
    _run("phase0_01_auth_and_files.py", ["--out-dir", str(out)])

    public = (out / "step1_summary_public.md").read_text()
    full = (out / "step1_summary.md").read_text()

    assert uid not in public, "public summary leaked a StudyInstanceUID"
    assert "redacted" in public
    assert uid in full, "full summary should keep paths for local use"
    # aggregates survive redaction
    assert "train.csv" in public
    assert "train_images" in public


def test_step1_reports_missing_credentials(tmp_path, monkeypatch):
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    sys.argv = ["phase0_01_auth_and_files.py", "--out-dir", str(tmp_path / "o")]
    ns = runpy.run_path(
        str(REPO_ROOT / "eda" / "phase0_01_auth_and_files.py"), run_name="__not_main__"
    )
    assert ns["main"]() == 2, "missing credentials must be a clean failure, not a crash"


# --------------------------------------------------------------------------- #
# step 2
# --------------------------------------------------------------------------- #
@pytest.fixture
def audited(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    data = tmp_path / "data"
    out = tmp_path / "phase0"
    facts = fixture.build(tmp_path, data, out)
    _stub_kaggle(monkeypatch)
    _run(
        "phase0_02_audit_tabular.py",
        ["--data-dir", str(data), "--out-dir", str(out)],
    )
    return facts, (out / "step2_audit.md").read_text()


def test_step2_discovers_the_targets_blind(audited):
    _facts, report = audited
    for finding in fixture.FINDINGS:
        assert f"`{finding}`" in report, f"{finding} not discovered as a binary column"
    assert "Binary columns — 12 found" in report


def test_step2_measures_the_gold_subset(audited):
    _facts, report = audited
    assert f"Only {fixture.N_GOLD} of {fixture.N_STUDIES:,} rows carry these labels" in report


def test_step2_finds_sites_and_languages(audited):
    _facts, report = audited
    assert "site_id" in report and "17" in report
    for language in fixture.REPORTS:
        assert f"`{language}`" in report, f"language {language} not detected"


def test_step2_quantifies_report_coverage(audited):
    _facts, report = audited
    # 4,380 of 4,407 studies have a report -> the join table must say so
    assert f"{fixture.N_REPORTS:,}" in report
    assert "Joins between files" in report


def test_step2_leaks_neither_identifiers_nor_report_text(audited):
    facts, report = audited
    for uid in facts["uids"][:50]:
        assert uid not in report, "audit leaked a StudyInstanceUID"
    for text in fixture.REPORTS.values():
        assert text[:40] not in report, "audit leaked report text"
