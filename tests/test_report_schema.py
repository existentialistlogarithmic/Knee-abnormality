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


# --------------------------------------------------------------------------- #
# the collision between two vocabularies that both use the word "absent"
# --------------------------------------------------------------------------- #
def test_silence_and_explicit_normal_map_to_different_channels():
    """The training parquet's channel `"absent"` means *no supervision here* and
    masks the loss. The state ladder's rung `"absent"` means *the report says
    this is normal*, which is supervision and among the best the corpus has.

    Mapping one onto the other would mask every explicit normal and teach every
    silence as a negative — the precise inversion the five-channel design exists
    to prevent, and nothing would raise.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "s2s", REPO_ROOT / "eda" / "states_to_soft_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.CHANNEL["not_mentioned"] == "absent", \
        "silence must map to the channel that masks the loss"
    assert module.CHANNEL["absent"] == "negated", \
        "an explicit normal must stay supervision, not become an abstention"


def test_every_rung_has_a_channel_and_channels_stay_in_the_known_vocabulary():
    import importlib.util

    from src import report_schema

    spec = importlib.util.spec_from_file_location(
        "s2s", REPO_ROOT / "eda" / "states_to_soft_labels.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert set(module.CHANNEL) == set(report_schema.STATES)
    known = {"absent", "negated", "hedged", "low_severity", "asserted"}
    assert set(module.CHANNEL.values()) <= known, \
        f"unknown channel value: {set(module.CHANNEL.values()) - known}"


def test_training_uses_a_confidence_column_only_when_it_exists():
    """Older label files have no per-finding weight. Silently defaulting one in
    would make every run before this incomparable to every run after it."""
    source = (REPO_ROOT / "kaggle" / "04_train" / "run.py").read_text()
    assert 'if all(column in soft.columns for column in confidence_columns):' in source
    assert "no per-finding confidence column" in source, \
        "the absence must be visible in the log, not silent"


def test_the_prompt_carries_a_worked_example_in_the_exact_output_shape():
    """A chat model without an example reaches for markdown fences and prose,
    and every unparseable answer is a study left with no supervision. The
    example is the cheapest substitute for a decoding grammar."""
    from src import report_schema

    prompt = report_schema.build_prompt("REPORT BODY")
    start, end = prompt.find("{"), prompt.find("}")
    assert start >= 0 and end > start, "no example object in the prompt"
    example = json.loads(prompt[start:end + 1])
    assert set(example) == set(report_schema.FINDINGS), \
        "the example must show every finding, or the model will omit some"
    assert set(example.values()) <= set(report_schema.STATES)


def test_the_kernel_retries_a_batch_instead_of_abstaining_on_memory():
    """An out-of-memory error is a statement about batch size, not about the
    report. The first run of this kernel abstained on 100% of studies for that
    reason and printed a 0.500 that read like a verdict on the method."""
    source = KERNEL.read_text()
    assert "except torch.cuda.OutOfMemoryError:" in source
    assert "size //= 2" in source, "an OOM must shrink the batch and retry"
    assert "RUN FAILED, not a verdict" in source, \
        "a broken run and a bad method must not report the same way"


def test_the_reader_is_spread_across_every_visible_gpu():
    """device_map="auto" filled one T4 and left the other idle, which is what
    caused the out-of-memory in the first place."""
    source = KERNEL.read_text()
    assert "max_memory=budget" in source
    assert "torch.cuda.device_count()" in source
    assert "RUN_GPU_BUDGET" in source


def test_to_rank_averages_ties_rather_than_breaking_them_by_position(kernel):
    """The lexicon emits only 2-4 distinct values per finding, so ties are the
    common case. `argsort().argsort()` would invent an order between equal
    scores based on array position — pure noise, baked into 4,407 studies'
    training targets. It also made the local analysis and the kernel disagree.
    """
    import numpy as np

    values = np.array([[0.9], [0.9], [0.9], [0.05]], float)
    ranked = kernel["to_rank"](values)
    assert ranked[0, 0] == ranked[1, 0] == ranked[2, 0], \
        "equal scores must receive equal ranks"
    assert ranked[3, 0] < ranked[0, 0]

    # a permutation of the input must give the same multiset of ranks
    shuffled = kernel["to_rank"](values[[3, 1, 0, 2]])
    assert sorted(ranked[:, 0]) == sorted(shuffled[:, 0])


def test_fuse_is_a_coverage_union_with_no_free_parameters(kernel):
    """A rule that picked the better labeler per finding would fit twelve
    choices to 58 studies and report a number that means nothing."""
    import numpy as np

    lexicon = np.array([[0.9], [np.nan], [0.1], [np.nan]], float)
    machine = np.array([[1.0], [0.5], [np.nan], [np.nan]], float)
    fused = kernel["fuse"](lexicon, machine)
    assert not np.isnan(fused[0, 0]), "both spoke; must combine"
    assert not np.isnan(fused[1, 0]), "only the model spoke; must take it"
    assert not np.isnan(fused[2, 0]), "only the lexicon spoke; must take it"
    assert np.isnan(fused[3, 0]), "neither spoke; must abstain"


def test_the_kernel_gates_on_the_union_not_the_standalone_score():
    """The first run asked 'does this beat the lexicon', answered no at 0.7526
    against 0.769, and stopped — while the union of the two scored well above
    either. That is a good idea very nearly discarded for asking the wrong
    question."""
    source = KERNEL.read_text()
    assert "if combined_macro <= lex_macro:" in source
    assert "The union adds" in source
    assert 'find_marker("soft_labels.parquet")' in source, \
        "the kernel must load the lexicon labels to compare against"
