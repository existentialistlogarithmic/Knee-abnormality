"""Tests for the public-notebook field survey.

The survey decides whether a whole plateau is reachable, so the ways it can
mislead are the ways that cost:

1. **Reading CC-BY-NC-SA as excluded.** E043 says NC matches this competition's
   own winner licence and that ShareAlike is a read before shipping. Three
   standing documents had hardened that into an exclusion, sitting on the
   largest measured gap in the project (E088). A test pins the tier.
2. **Reading `other` as a licence.** `tonylica/rsna-knee-bend-dinov3-0917-repro-
   assets` is "Other (specified in description)" with an **empty description**,
   so nothing is specified and no grant exists. It belongs in the avoid tier on
   the merits, not on the label.
3. **Counting a failed pull as a notebook with no dependencies**, which would
   silently understate every asset's mount count.

No patient data: every value here is written for the test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eda.survey_public_notebooks import (  # noqa: E402
    main,
    read_notebook,
    tier,
)


def test_cc0_and_apache_are_usable():
    assert tier("CC0: Public Domain") == "usable"
    assert tier("Apache 2.0") == "usable"


def test_sharealike_is_a_read_not_an_exclusion():
    """E043, verbatim: NC matches the competition's own CC-BY-NC 4.0 winner
    licence, so NC is not the obstacle; ShareAlike is the part to read."""
    assert tier("CC BY-NC-SA 4.0") == "sharealike"
    assert tier("CC BY-NC-SA 4.0") != "no-grant"


def test_other_with_nothing_specified_is_no_grant():
    """The label says the licence is in the description; the description is
    empty. Nothing is specified, so nothing is granted."""
    assert tier("Other (specified in description)") == "no-grant"


def test_an_unsupplied_licence_is_unknown_not_usable():
    """Absence of a lookup is not a permission."""
    assert tier(None) == "unknown"
    assert tier("") == "unknown"


def _write(tmp_path, name, meta, source=""):
    d = tmp_path / name
    d.mkdir()
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    if source:
        (d / "run.py").write_text(source)
    return d


def test_a_failed_pull_is_dropped_not_counted_as_empty(tmp_path):
    d = tmp_path / "owner~slug"
    d.mkdir()
    assert read_notebook(d) is None


def test_metadata_and_source_are_both_read(tmp_path):
    d = _write(tmp_path, "o~s",
               {"id": "o/s", "dataset_sources": ["a/b", "c/d"], "enable_gpu": True},
               "import pydicom\nm = timm.create_model('coatnet_rmlp_2_rw_384')\n")
    found = read_notebook(d)
    assert found["ref"] == "o/s"
    assert found["datasets"] == ["a/b", "c/d"]
    assert found["reads_dicom"] is True
    assert "coatnet" in found["backbones"]


def test_a_notebook_that_mounts_a_preprocessed_corpus_reads_no_dicom(tmp_path):
    d = _write(tmp_path, "o~s", {"id": "o/s"}, "vols = np.load('all_vols.npy')\n")
    assert read_notebook(d)["reads_dicom"] is False


def test_assets_rank_by_how_many_notebooks_mount_them(tmp_path, capsys):
    _write(tmp_path, "a~1", {"id": "a/1", "dataset_sources": ["shared/asset", "rare/asset"]})
    _write(tmp_path, "b~2", {"id": "b/2", "dataset_sources": ["shared/asset"]})
    licences = tmp_path / "lic.json"
    licences.write_text(json.dumps({"shared/asset": "Other (specified in description)",
                                   "rare/asset": "CC0: Public Domain"}))
    main(["--corpus", str(tmp_path), "--licences", str(licences)])
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "asset" in ln and "/" in ln]
    assert "shared/asset" in lines[0], "the most-mounted asset must rank first"
    assert "no-grant" in lines[0]
    assert "avoid tier" in out
