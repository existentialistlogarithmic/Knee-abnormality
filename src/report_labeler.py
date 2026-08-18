"""Turn multilingual radiology reports into calibrated soft labels for 12 findings.

The competition gives expert labels for 58 of 4,407 training studies and a
free-text report for all of them, and the host says outright that the reports
are there "from which you may wish to derive the labels for the remaining
studies". This module is that derivation.

Design, in the order the decisions matter
-----------------------------------------
1. **Soft labels with an abstain channel, not hard 0/1.** A report that never
   mentions the ACL is not a report saying the ACL is intact. Those two cases
   get different outputs: `absent` (abstain) versus `negated`. Collapsing them
   throws away the distinction the gold set punishes hardest.

2. **Normality assertions matter as much as negation.** Mining the corpus showed
   these reports mostly *assert* health rather than deny disease: "intact",
   "normaldir", "regelrecht", "održanog kontinuiteta", "запазена". A negation-only
   system reads "ACL intact" as a positive ACL mention. That single error is why
   the crude baseline scored specificity 0.44 on ACL.

3. **Cues bind to the nearest mention, inside sentence bounds.** "The ACL is
   torn but the medial meniscus is intact" has to resolve to ACL positive and
   meniscus negative. A clause-level flag cannot do that; nearest-cue-within-
   window can.

4. **Severity is a separate channel.** The gold labels are severity-thresholded
   with "on the fence" graded negative, so "grade 1 signal change" and "mild
   chondropathy" are mentions that probably map to a *negative* label. They get
   their own score band rather than being treated as clean positives.

Everything language-specific lives in `src/lexicons/*.csv` so it can be reviewed
and corrected by hand without touching this file.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

LEXICON_DIR = Path(__file__).resolve().parent / "lexicons"

# Score bands. These are deliberate priors, not fitted values — calibration
# against the gold subset happens downstream and inside folds only.
SCORE = {
    "asserted": 0.90,      # mentioned, no negation, no hedge
    "hedged": 0.62,        # "possible", "cannot exclude"
    "low_severity": 0.32,  # "grade 1", "mild" — likely below the labelling threshold
    "negated": 0.05,       # explicitly denied, or the structure asserted normal
}
ABSTAIN = None  # the report is silent; the caller substitutes a prior

# How far from a mention a cue may sit and still bind to it. Negation usually
# precedes the structure ("no evidence of a tear"); normality usually follows it
# ("the ACL is intact"), hence the asymmetry.
WINDOW_BEFORE = 90
WINDOW_AFTER = 70

SENTENCE_BREAK = re.compile(r"[.;:\n\r]|\bbut\b|\bhowever\b|\bancak\b|\baber\b|\bpero\b")


@dataclass
class Mention:
    finding: str
    term: str
    start: int
    end: int
    verdict: str  # asserted | hedged | low_severity | negated
    cue: str | None


@dataclass
class FindingLabel:
    score: float | None
    channel: str  # asserted | hedged | low_severity | negated | absent
    n_mentions: int

    @property
    def abstained(self) -> bool:
        return self.channel == "absent"


def _read_lexicon(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(line for line in fh if not line.startswith("#")))


class ReportLabeler:
    """Lexicon-driven, auditable, and deliberately not a model.

    A model would need the 58 gold studies to fit, and spending them on fitting
    leaves nothing to evaluate on. This needs no gold data, so the gold subset
    stays a clean test set.
    """

    def __init__(self, lexicon_dir: Path = LEXICON_DIR, fallback_language: str = "en"):
        self.fallback_language = fallback_language

        terms = _read_lexicon(lexicon_dir / "findings.csv")
        self.findings = sorted({row["finding"] for row in terms})
        # finding -> language -> compiled alternation
        self._terms: dict[str, dict[str, re.Pattern]] = defaultdict(dict)
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in terms:
            grouped[(row["finding"], row["language"])].append(row["term"].strip().lower())
        for (finding, language), words in grouped.items():
            self._terms[finding][language] = self._compile(words)

        cues = _read_lexicon(lexicon_dir / "cues.csv")
        self._cues: dict[str, dict[str, re.Pattern]] = defaultdict(dict)
        cue_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for row in cues:
            cue_groups[(row["language"], row["cue_type"])].append(row["cue"].strip().lower())
        for (language, cue_type), words in cue_groups.items():
            self._cues[language][cue_type] = self._compile(words)

    @staticmethod
    def _compile(words: list[str]) -> re.Pattern:
        # Longest first so "no evidence of" wins over "no"; \b is unreliable for
        # non-Latin scripts, so anchor on a non-word char or string edge.
        parts = sorted((re.escape(w) for w in words), key=len, reverse=True)
        return re.compile(r"(?<![^\W\d_])(?:" + "|".join(parts) + r")", re.IGNORECASE | re.UNICODE)

    def _languages_for(self, language: str) -> list[str]:
        """The report's language, plus English.

        English is always included because these reports borrow English
        abbreviations regardless of their own language — "ACL", "MCL", "grade 1"
        all appear inside Turkish and German reports in this corpus.
        """
        if language == self.fallback_language:
            return [language]
        return [language, self.fallback_language]

    def _window(self, text: str, start: int, end: int) -> str:
        """Text around a mention, clipped at the nearest sentence break.

        Clipping is what stops a negation in one sentence leaking into the next.
        """
        left = max(0, start - WINDOW_BEFORE)
        right = min(len(text), end + WINDOW_AFTER)
        before = text[left:start]
        after = text[end:right]
        breaks = list(SENTENCE_BREAK.finditer(before))
        if breaks:
            before = before[breaks[-1].end():]
        forward = SENTENCE_BREAK.search(after)
        if forward:
            after = after[: forward.start()]
        return before + text[start:end] + after

    def mentions(self, text: str, language: str) -> list[Mention]:
        lowered = text.lower()
        found: list[Mention] = []
        for finding in self.findings:
            for lang in self._languages_for(language):
                pattern = self._terms[finding].get(lang)
                if pattern is None:
                    continue
                for match in pattern.finditer(lowered):
                    window = self._window(lowered, match.start(), match.end())
                    verdict, cue = self._judge(window, language)
                    found.append(
                        Mention(finding, match.group(0), match.start(), match.end(), verdict, cue)
                    )
        return found

    def _judge(self, window: str, language: str) -> tuple[str, str | None]:
        """Decide polarity for one mention from the cues around it.

        Order matters: an explicit denial outranks a hedge, and a hedge outranks
        a severity qualifier. "Possible grade 1 tear" is hedged, not low-severity;
        "no grade 1 tear" is negated outright.
        """
        for cue_type, verdict in (
            ("negation", "negated"),
            ("normality", "negated"),
            ("hedge", "hedged"),
            ("severity_low", "low_severity"),
        ):
            for lang in self._languages_for(language):
                pattern = self._cues.get(lang, {}).get(cue_type)
                if pattern is not None:
                    hit = pattern.search(window)
                    if hit:
                        return verdict, hit.group(0)
        return "asserted", None

    def label(self, text: str, language: str) -> dict[str, FindingLabel]:
        """One report to twelve soft labels.

        Aggregation across several mentions of the same finding takes the
        strongest positive evidence. A report that says "no acute ACL tear" in
        one sentence and "ACL rupture" in another describes a torn ACL; taking
        the maximum is the reading that matches how radiologists write.
        """
        by_finding: dict[str, list[Mention]] = defaultdict(list)
        for mention in self.mentions(text or "", language):
            by_finding[mention.finding].append(mention)

        rank = ["negated", "low_severity", "hedged", "asserted"]
        out: dict[str, FindingLabel] = {}
        for finding in self.findings:
            hits = by_finding.get(finding, [])
            if not hits:
                out[finding] = FindingLabel(ABSTAIN, "absent", 0)
                continue
            best = max(hits, key=lambda m: rank.index(m.verdict))
            out[finding] = FindingLabel(SCORE[best.verdict], best.verdict, len(hits))
        return out


def detect_language(text: str, default: str = "en") -> str:
    try:
        import py3langid
    except ImportError:
        return default
    snippet = (text or "").strip()[:2000]
    if len(snippet) < 15:
        return default
    return py3langid.classify(snippet)[0]
