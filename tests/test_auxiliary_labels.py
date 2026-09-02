"""Tests for the auxiliary report targets.

Auxiliary findings are supervision the reports already contain for structures
the competition does not score. Three properties have to hold, and each of them
is a way this could go quietly wrong rather than loudly:

1. **The auxiliary targets never reach a submission.** They are extra output
   rows on a shared trunk and nothing else. If an auxiliary name ever appeared
   in the scored twelve, a column nobody validated would be graded.
2. **They are built by the same machinery as the scored twelve.** The whole
   claim "auxiliary targets added" is one variable rests on the matcher, the
   window and the cues being shared, not re-implemented.
3. **The cue engine applies to them.** A lexicon without negation reads
   "PCL intact" as a torn PCL — the exact failure the Phase 0 probe found on
   ACL, at specificity 0.44.

No patient data: every string here is written for the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.report_labeler import ReportLabeler  # noqa: E402
from src.report_schema import AUXILIARY_FINDINGS, FINDINGS  # noqa: E402


@pytest.fixture(scope="module")
def aux():
    return ReportLabeler(findings_file="auxiliary.csv")


def channel(labeler, text, language, finding):
    return labeler.label(text, language)[finding].channel


# --------------------------------------------------------------------------- #
# 1. auxiliary targets are training-only and must never be scored
# --------------------------------------------------------------------------- #
def test_auxiliary_and_scored_findings_are_disjoint():
    assert not set(AUXILIARY_FINDINGS) & set(FINDINGS)


def test_schema_and_lexicon_agree(aux):
    """A name in one and not the other silently drops or invents a target."""
    assert sorted(aux.findings) == sorted(AUXILIARY_FINDINGS)


def test_auxiliary_findings_are_unique():
    assert len(AUXILIARY_FINDINGS) == len(set(AUXILIARY_FINDINGS))


# --------------------------------------------------------------------------- #
# 2. the same machinery, not a second labeler
# --------------------------------------------------------------------------- #
def test_auxiliary_labeler_shares_the_cue_file(aux):
    """Both labelers read the same cues, so polarity is decided identically."""
    scored = ReportLabeler()
    assert aux._cues.keys() == scored._cues.keys()
    for language in scored._cues:
        assert aux._cues[language].keys() == scored._cues[language].keys()


def test_default_labeler_is_unchanged_by_the_new_parameter():
    """The scored twelve must not move because an auxiliary file now exists."""
    assert ReportLabeler().findings == ReportLabeler(findings_file="findings.csv").findings
    assert sorted(ReportLabeler().findings) == sorted(FINDINGS)


# --------------------------------------------------------------------------- #
# 3. the cue engine reaches the auxiliary vocabulary
# --------------------------------------------------------------------------- #
def test_normality_assertion_on_an_auxiliary_finding(aux):
    assert channel(aux, "The posterior cruciate ligament is intact.", "en", "PCL") == "negated"


def test_explicit_negation_on_an_auxiliary_finding(aux):
    assert channel(aux, "No evidence of a ganglion cyst.", "en", "Ganglion") == "negated"


def test_asserted_auxiliary_finding(aux):
    assert channel(aux, "Full-thickness chondral defect of the trochlea.", "en",
                   "Chondral") == "asserted"


def test_hedged_auxiliary_finding(aux):
    assert channel(aux, "Possible partial tear of the patellar tendon.", "en",
                   "PatellarTendon") == "hedged"


def test_silence_abstains(aux):
    """Silence is not a negative — the distinction the whole design protects."""
    label = aux.label("Small joint effusion. Menisci intact.", "en")["Plica"]
    assert label.abstained and label.score is None


@pytest.mark.parametrize("language,text,finding", [
    ("es", "Ligamento cruzado posterior íntegro.", "PCL"),
    ("de", "Das hintere Kreuzband ist intakt.", "PCL"),
    ("tr", "Arka çapraz bağ normaldir.", "PCL"),
])
def test_negation_reaches_the_other_languages(aux, language, text, finding):
    assert channel(aux, text, language, finding) == "negated"


# --------------------------------------------------------------------------- #
# the lexicon file itself
# --------------------------------------------------------------------------- #
def test_auxiliary_lexicon_has_no_blank_terms():
    import csv
    path = REPO_ROOT / "src" / "lexicons" / "auxiliary.csv"
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    assert rows, "auxiliary.csv is empty"
    for row in rows:
        assert row["term"].strip(), row
        assert row["finding"] in AUXILIARY_FINDINGS, row
        assert row["corpus"] in {"0", "1"}, row
