"""The closed vocabulary, its parser, and the rule that governs both.

The parser matters more than it looks. Every report it fails to read becomes a
study with no supervision, and a parser that guesses instead of abstaining
teaches the imaging model a confident negative it invented.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL = REPO_ROOT / "kaggle" / "16_llm_labeler" / "run.py"


@pytest.fixture(scope="module")
def kernel():
    import runpy

    return runpy.run_path(str(KERNEL), run_name="__not_main__")


def test_the_ladder_is_ordered_and_shared_by_both_copies():
    from src import report_schema

    source = KERNEL.read_text()
    in_kernel = re.search(r"^STATES = (\[.*?\])", source, re.M | re.S).group(1)
    assert eval(in_kernel) == report_schema.STATES, \
        "the kernel and the module disagree about the ladder"
    assert report_schema.STATES[0] == "absent"
    assert report_schema.STATES[-1] == "severe"


def test_scores_are_monotone_in_the_ladder():
    """ROC-AUC reads order only, so a non-monotone map would change the training
    targets while claiming to preserve the evaluation."""
    from src import report_schema

    scored = [(s, report_schema.STATE_SCORE[s]) for s in report_schema.STATES
              if report_schema.STATE_SCORE[s] is not None]
    values = [v for _, v in scored]
    assert values == sorted(values), scored


def test_silence_abstains_rather_than_scoring_zero():
    """A report that never mentions the ACL is not a report saying it is intact.
    This is the single distinction the whole labeler is built around."""
    from src import report_schema

    assert report_schema.STATE_SCORE["not_mentioned"] is None
    assert report_schema.STATE_WEIGHT["not_mentioned"] == 0.0
    assert report_schema.STATE_SCORE["absent"] is not None


@pytest.mark.parametrize("payload,expect", [
    ('{"ACL": "severe"}', "severe"),
    ('```json\n{"ACL": "moderate"}\n```', "moderate"),
    ('Here is the answer:\n{"ACL": "mild"}\nHope that helps.', "mild"),
    ('{"ACL": "SEVERE"}', "severe"),
    ('{"ACL": "  absent  "}', "absent"),
    ('{"ACL": "torn"}', "not_mentioned"),        # invented state -> abstain
    ('{"ACL": 3}', "not_mentioned"),             # wrong type -> abstain
    ('{"ACL": null}', "not_mentioned"),
    ('not json at all', "not_mentioned"),
    ('', "not_mentioned"),
    ('{"unclosed": ', "not_mentioned"),
    ('["a", "list"]', "not_mentioned"),
])
def test_parse_states_abstains_rather_than_guessing(kernel, payload, expect):
    assert kernel["parse_states"](payload)["ACL"] == expect


def test_parse_states_always_returns_every_finding(kernel):
    for payload in ('{"ACL": "severe"}', "", "garbage", '{"Fracture": "mild"}'):
        states = kernel["parse_states"](payload)
        assert set(states) == set(kernel["FINDINGS"]), payload


def test_a_missing_finding_abstains_without_disturbing_the_others(kernel):
    states = kernel["parse_states"](json.dumps({"ACL": "severe", "Fracture": "absent"}))
    assert states["ACL"] == "severe"
    assert states["Fracture"] == "absent"
    assert states["Synovitis"] == "not_mentioned"


def test_rank_value_is_monotone_and_bounded(kernel):
    values = [kernel["rank_value"](s) for s in kernel["STATES"]]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[-1] == 1.0
    # an unrecognised state ranks as silence, not as a positive
    assert kernel["rank_value"]("nonsense") == kernel["rank_value"]("not_mentioned")


# --------------------------------------------------------------------------- #
# competition Rule 4.b
# --------------------------------------------------------------------------- #
HOSTED = ["api.openai.com", "openai", "api.anthropic.com", "anthropic",
          "generativelanguage.googleapis", "gemini", "cohere", "mistral.ai",
          "together.ai", "replicate.com", "huggingface.co/api/inference"]


def test_nothing_in_the_label_path_can_reach_a_hosted_model():
    """STRATEGY.md: report text must never go to a hosted LLM API. The models
    used here are open weights, downloaded and run inside the kernel.

    Asserted against the source rather than trusted, because this is the one
    mistake in the project that could not be undone by a later commit — the text
    would already have left.
    """
    for path in (KERNEL, REPO_ROOT / "src" / "report_schema.py",
                 REPO_ROOT / "src" / "report_labeler.py"):
        lowered = path.read_text().lower()
        for provider in HOSTED:
            # a prose mention inside a comment naming the rule is fine; a URL,
            # an import or a client construction is not
            for pattern in (f"https://{provider}", f"import {provider}",
                            f"from {provider}", f"{provider}("):
                assert pattern not in lowered, f"{path.name} contains {pattern!r}"


def test_the_kernel_writes_states_and_metrics_but_never_report_text():
    """Kernel output becomes a Kaggle Dataset. A report string written there is
    patient-derived text leaving the kernel, which .gitignore cannot help with
    after the fact."""
    source = KERNEL.read_text()
    writes = re.findall(r"write_text\(json\.dumps\((.*?)\, indent=2\)", source, re.S)
    assert writes, "no outputs found — has the kernel changed shape?"
    for block in writes:
        assert ".Report" not in block, "a report column is being written out"
        assert "build_prompt" not in block, "a prompt embeds the report text"
