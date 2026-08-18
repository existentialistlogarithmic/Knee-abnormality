"""Tests for the report labeler.

The negation and normality logic is the core of the whole strategy — the Phase 0
probe showed that a matcher without it scores specificity 0.44 on ACL. These
tests pin the behaviour that fixes that, in each of the larger languages.

No patient data: every string here is written for the test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.report_labeler import ReportLabeler, detect_language  # noqa: E402


@pytest.fixture(scope="module")
def labeler():
    return ReportLabeler()


def channel(labeler, text, language, finding):
    return labeler.label(text, language)[finding].channel


# --------------------------------------------------------------------------- #
# the failure the probe found: mentions that assert health
# --------------------------------------------------------------------------- #
def test_explicit_negation_is_not_a_positive(labeler):
    assert channel(labeler, "No evidence of medial meniscus tear.", "en",
                   "Medial Meniscus") == "negated"


def test_normality_assertion_is_not_a_positive(labeler):
    """'The ACL is intact' — no negation word anywhere, still a negative."""
    assert channel(labeler, "The anterior cruciate ligament is intact.", "en", "ACL") == "negated"


def test_plain_mention_is_a_positive(labeler):
    assert channel(labeler, "Full thickness tear of the anterior cruciate ligament.",
                   "en", "ACL") == "asserted"


# --------------------------------------------------------------------------- #
# cues must bind to the nearest mention, not the whole sentence
# --------------------------------------------------------------------------- #
def test_two_findings_opposite_polarity_in_one_sentence(labeler):
    text = "The anterior cruciate ligament is torn but the medial meniscus is intact."
    result = labeler.label(text, "en")
    assert result["ACL"].channel == "asserted"
    assert result["Medial Meniscus"].channel == "negated"


def test_negation_does_not_leak_across_a_sentence_boundary(labeler):
    text = "No joint effusion. Anterior cruciate ligament tear is present."
    result = labeler.label(text, "en")
    assert result["Effusion"].channel == "negated"
    assert result["ACL"].channel == "asserted"


# --------------------------------------------------------------------------- #
# abstain: silence is not a negative
# --------------------------------------------------------------------------- #
def test_unmentioned_finding_abstains(labeler):
    result = labeler.label("The anterior cruciate ligament is intact.", "en")
    assert result["Fracture"].channel == "absent"
    assert result["Fracture"].score is None
    assert result["Fracture"].abstained


def test_abstain_is_distinct_from_negation(labeler):
    """The distinction the gold set punishes hardest, so it gets its own test."""
    silent = labeler.label("The anterior cruciate ligament is intact.", "en")["Effusion"]
    denied = labeler.label("No joint effusion is seen.", "en")["Effusion"]
    assert silent.channel == "absent" and silent.score is None
    assert denied.channel == "negated" and denied.score is not None
    assert silent.channel != denied.channel


# --------------------------------------------------------------------------- #
# hedging and severity get their own bands
# --------------------------------------------------------------------------- #
def test_hedged_mention_scores_below_an_asserted_one(labeler):
    hedged = labeler.label("Possible tear of the lateral meniscus.", "en")["Lateral Meniscus"]
    asserted = labeler.label("Tear of the lateral meniscus.", "en")["Lateral Meniscus"]
    assert hedged.channel == "hedged"
    assert hedged.score < asserted.score


def test_low_severity_scores_below_hedged(labeler):
    """Gold labels are severity-thresholded, so 'grade 1' probably means 0."""
    mild = labeler.label("Grade 1 signal within the medial meniscus.", "en")["Medial Meniscus"]
    hedged = labeler.label("Possible medial meniscus tear.", "en")["Medial Meniscus"]
    assert mild.channel == "low_severity"
    assert mild.score < hedged.score


def test_negation_outranks_severity(labeler):
    assert channel(labeler, "No grade 1 change of the medial meniscus.", "en",
                   "Medial Meniscus") == "negated"


# --------------------------------------------------------------------------- #
# the same logic in the other large languages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "language", "finding", "expected"),
    [
        # Spanish — 16% of the corpus
        ("Rotura del menisco medial.", "es", "Medial Meniscus", "asserted"),
        ("Sin rotura del menisco medial.", "es", "Medial Meniscus", "negated"),
        ("Derrame articular leve.", "es", "Effusion", "low_severity"),
        # Turkish — 12%. 'normaldir' asserts health with no negation word.
        ("Ön çapraz bağ normaldir.", "tr", "ACL", "negated"),
        ("Medial menisküs yırtığı izlenmedi.", "tr", "Medial Meniscus", "negated"),
        # German — 6%. 'intakter' likewise.
        ("Intakter vorderes Kreuzband.", "de", "ACL", "negated"),
        ("Kein Gelenkerguss.", "de", "Effusion", "negated"),
        ("Innenmeniskus Riss.", "de", "Medial Meniscus", "asserted"),
        # Dutch — 4%
        ("Geen hydrops.", "nl", "Effusion", "negated"),
        ("Mediale meniscus scheur.", "nl", "Medial Meniscus", "asserted"),
        # Bulgarian — 5%
        ("Без данни за ставен излив.", "bg", "Effusion", "negated"),
    ],
)
def test_other_languages(labeler, text, language, finding, expected):
    assert channel(labeler, text, language, finding) == expected


# --------------------------------------------------------------------------- #
# English abbreviations appear inside non-English reports
# --------------------------------------------------------------------------- #
def test_english_abbreviation_inside_a_german_report(labeler):
    assert channel(labeler, "MCL intakt.", "de", "MCL") == "negated"


# --------------------------------------------------------------------------- #
# shape of the output
# --------------------------------------------------------------------------- #
def test_every_finding_gets_a_label(labeler):
    result = labeler.label("Normal knee.", "en")
    assert len(result) == 12
    assert all(v.channel in {"asserted", "hedged", "low_severity", "negated", "absent"}
               for v in result.values())


def test_empty_report_abstains_everywhere(labeler):
    result = labeler.label("", "en")
    assert all(v.abstained for v in result.values())


def test_language_detection_falls_back_on_short_text(labeler):
    assert detect_language("", default="en") == "en"
    assert detect_language("ok", default="en") == "en"
