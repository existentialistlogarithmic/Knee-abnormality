"""The closed vocabulary a language model reads reports into, and the map out.

`src/report_labeler.py` reaches macro AUC 0.769 against the 58 expert-labelled
studies. Published public systems reach **0.881** with a different method
(`docs/COMPETITIVE_ANALYSIS.md`), and the difference is not vocabulary coverage
that a bigger lexicon would close: it is paraphrase, implication and negation
scope across ten languages. The abstain rates say so plainly — Synovitis is
72% abstain and scores 0.580, Fracture 62% and 0.759.

**Two layers, deliberately.** The model picks one state from a fixed ladder;
Python turns states into numbers. A model is good at choosing from a closed set
and bad at emitting calibrated probabilities, and asking it for the second thing
is how you get confident nonsense that no downstream check can catch.

**Compliance.** `docs/STRATEGY.md` forbids sending report text to a hosted LLM
API (competition Rule 4.b). This is designed for an **open-weights model running
inside a Kaggle kernel**, which that rule explicitly permits. Nothing here
should ever be pointed at an external endpoint, and there is a test asserting
this module names no hosted provider.
"""

from __future__ import annotations

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]

# One ordered ladder for all twelve findings rather than a bespoke vocabulary
# each. A shared ladder keeps the prompt short enough to leave room for a
# 4,700-character report, and — more importantly — the metric is ROC-AUC, which
# reads ORDER ONLY. So the ladder's ordering is the entire signal, and the
# numbers attached to it below cannot change a single AUC.
STATES = ["absent", "not_mentioned", "equivocal", "minimal", "mild", "moderate", "severe"]

STATE_HELP = {
    "absent": "the report states this structure is normal, intact, or the "
              "finding is explicitly denied",
    "not_mentioned": "the report does not address this structure or finding "
                     "at all — silence, not a statement of health",
    "equivocal": "hedged or uncertain: possible, suspected, cannot be excluded, "
                 "apparent but not confirmed",
    "minimal": "present but at the lowest grade: degeneration without a tear, "
               "trace fluid, minimal or early change",
    "mild": "explicitly mild, small, or grade 1",
    "moderate": "explicitly moderate, grade 2, or a partial tear",
    "severe": "explicitly severe, large, advanced, grade 3, complete or "
              "full-thickness tear, displaced fragment",
}

# Monotone in the ladder, so every AUC computed from these is identical to the
# AUC of the ladder itself. They exist to be TRAINING targets, where magnitude
# does matter, and they are deliberately not 0 and 1: an expert reading the
# images disagrees with the report often enough that a hard target is a lie.
STATE_SCORE = {
    "absent": 0.04,
    "not_mentioned": None,      # abstain — the loss is masked, not taught a zero
    "equivocal": 0.45,
    "minimal": 0.55,
    "mild": 0.68,
    "moderate": 0.85,
    "severe": 0.95,
}

# Confidence weights scale each study's contribution to the loss for that
# finding. An explicit severe finding is worth more supervision than a hedge.
STATE_WEIGHT = {
    "absent": 1.0,
    "not_mentioned": 0.0,
    "equivocal": 0.4,
    "minimal": 0.8,
    "mild": 1.0,
    "moderate": 1.0,
    "severe": 1.0,
}

SYSTEM_PROMPT = (
    "You are a musculoskeletal radiologist reading knee MRI reports. Reports "
    "may be in English, Spanish, Turkish, Croatian, Bosnian, Greek, German, "
    "Bulgarian, Dutch or French. Read the report in its original language; do "
    "not translate it first.\n\n"
    "For each of twelve findings, choose exactly one state from the fixed list. "
    "Never invent a state. Never explain. Output JSON only."
)


def build_prompt(report: str) -> str:
    """The user turn. Kept deterministic so a rerun reproduces the run."""
    ladder = "\n".join(f'  "{state}": {STATE_HELP[state]}' for state in STATES)
    findings = "\n".join(f"  {name}" for name in FINDINGS)
    return (
        f"States, in order of increasing severity:\n{ladder}\n\n"
        f"Findings to report on:\n{findings}\n\n"
        "Rules:\n"
        "- 'absent' means the report says it is normal or denies it. "
        "'not_mentioned' means the report is silent. These are different and "
        "the difference matters more than any other judgement you will make.\n"
        "- A finding described only in another compartment does not transfer. "
        "Medial and lateral are separate findings; so are the three "
        "osteoarthritis compartments.\n"
        "- Degeneration, mucoid change or signal change WITHOUT a tear is "
        "'minimal', not a tear.\n"
        "- Report what the text says, not what you would expect to see.\n\n"
        # A worked example is the cheapest available substitute for a decoding
        # grammar. Without it a chat model reaches for markdown fences and
        # commentary, and every unparseable answer is a study dropped to no
        # supervision at all.
        "Format, shown on a fragment that is not a real report — copy this "
        "shape exactly:\n"
        '{"ACL": "absent", "MCL": "not_mentioned", "Medial Meniscus": "severe", '
        '"Lateral Meniscus": "minimal", "Medial OA": "mild", "Lateral OA": '
        '"not_mentioned", "PF OA": "moderate", "Effusion": "mild", "Synovitis": '
        '"not_mentioned", "Baker\'s": "absent", "Contusion": "not_mentioned", '
        '"Fracture": "not_mentioned"}\n\n'
        f"Report:\n<<<\n{report}\n>>>\n\n"
        "Return a JSON object mapping each of the twelve finding names to one "
        "state string. No other keys, no commentary."
    )


def json_schema() -> dict:
    """A grammar for constrained decoding, so the output cannot be malformed.

    Worth the extra setup: a free-running model emits prose, markdown fences and
    invented states, and every one of those is a study silently dropped to the
    prior. Constrained decoding makes the failure mode impossible rather than
    caught.
    """
    return {
        "type": "object",
        "properties": {name: {"type": "string", "enum": STATES} for name in FINDINGS},
        "required": list(FINDINGS),
        "additionalProperties": False,
    }


def to_scores(states: dict[str, str]) -> tuple[dict[str, float | None], dict[str, float]]:
    """States to (soft score, confidence weight). Unknown states abstain.

    Abstaining on an unrecognised state rather than guessing keeps a decoding
    failure from being taught to the model as a negative.
    """
    scores, weights = {}, {}
    for name in FINDINGS:
        state = states.get(name)
        if state not in STATE_SCORE:
            state = "not_mentioned"
        scores[name] = STATE_SCORE[state]
        weights[name] = STATE_WEIGHT[state]
    return scores, weights


def rank_value(state: str) -> float:
    """Position on the ladder, for ranking when a score would abstain.

    ROC-AUC needs an order over every study, including the silent ones. Silence
    is genuinely weak evidence of absence — these reports assert health rather
    than deny disease — so it sits just above an explicit denial and below any
    positive statement.
    """
    if state not in STATES:
        state = "not_mentioned"
    return STATES.index(state) / (len(STATES) - 1)
