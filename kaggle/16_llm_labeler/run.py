"""Read every radiology report into a closed state ladder with open weights.

`src/report_labeler.py` — a hand-built multilingual lexicon — reaches macro AUC
**0.769** against the 58 expert-labelled studies. Its failures are legible in
its own abstain rates: Synovitis abstains on 72% of reports and scores 0.580,
Fracture abstains on 62% and scores 0.759. Those are not gaps a longer word list
closes. They are paraphrase, implication and negation scope, across ten
languages, which is what a language model is actually for.

**Two layers.** The model picks one state per finding from a fixed ladder under
a JSON grammar; `src/report_schema.py` turns states into numbers in ordinary
Python. Asking a model for calibrated probabilities directly is asking it to do
the one thing it is bad at, and no downstream check can catch a confident wrong
number.

**The 58 gold studies are scored first, before the other 4,349.** They cost
about a minute and they answer the only question that matters — is this better
than 0.769 — before an hour is spent on the rest. If it is not, the log says so
in the first two minutes rather than at the end.

**COMPLIANCE.** `docs/STRATEGY.md` forbids sending report text to a hosted LLM
API under competition Rule 4.b. This runs open weights inside a Kaggle kernel,
which that rule explicitly permits. Report text never leaves this kernel: the
outputs written are states, scores and aggregate metrics, never report strings.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# GENERATED CONFIG — written by eda/generate_kernels.py from src/pipeline.py.
# Edit the manifest, not this file. Everything outside this block is shared by
# every kernel rendered from this template.
# --------------------------------------------------------------------------- #
# Reads every report into a closed 7-state ladder with an
# open-weights model, then maps states to soft targets in Python.
# The lexicon labeler reaches 0.769 against the 58 expert-labelled
# studies; published systems using this method reach 0.881, and the
# gap is paraphrase and negation scope across ten languages rather
# than missing vocabulary.
#
# COMPLIANCE: open weights, inside a Kaggle kernel. No report text
# leaves this kernel and no hosted API is contacted — see
# docs/STRATEGY.md on competition Rule 4.b.
#
RUN_MODEL          = "Qwen/Qwen2.5-7B-Instruct"
RUN_GPU_BUDGET     = "9GiB"
RUN_MAX_REPORTS    = 0
RUN_BATCH          = 8
RUN_MAX_NEW_TOKENS = 220
RUN_TIME_BUDGET    = 8.0 * 3600
# --------------------------------------------------------------------------- #

FINDINGS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
            "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
            "Contusion", "Fracture"]
SKIP_DIRECTORIES = {"train_series", "test_series"}
OUT = Path("/kaggle/working")

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
SYSTEM_PROMPT = (
    "You are a musculoskeletal radiologist reading knee MRI reports. Reports "
    "may be in English, Spanish, Turkish, Croatian, Bosnian, Greek, German, "
    "Bulgarian, Dutch or French. Read the report in its original language; do "
    "not translate it first.\n\n"
    "For each of twelve findings, choose exactly one state from the fixed list. "
    "Never invent a state. Never explain. Output JSON only."
)


# --------------------------------------------------------------------------- #
# from kaggle/_templates/_shared/discovery.py
# --------------------------------------------------------------------------- #
def find_marker(marker: str, max_depth: int = 4):
    frontier = [(Path("/kaggle/input"), 0)]
    while frontier:
        directory, depth = frontier.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(directory.iterdir())
        except (FileNotFoundError, PermissionError):
            continue
        for entry in entries:
            if entry.is_file() and entry.name == marker:
                return directory
        for entry in entries:
            if entry.is_dir() and entry.name not in SKIP_DIRECTORIES:
                frontier.append((entry, depth + 1))
    return None


def report_environment() -> bool:
    """Record the accelerator actually granted.

    The Kaggle CLI does not expose the valid `machine_shape` strings, so which
    GPU a kernel receives has been UNVERIFIED for this project. This prints it,
    which matters because the current PyTorch build ships no Pascal kernels and
    a P100 would fail rather than run slowly.

    Returns True if the accelerator can actually run this build.
    """
    import torch

    print(f"torch {torch.__version__}  cuda available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("  no GPU visible; this will be very slow")
        return True
    usable = True
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"  GPU {i}: {name}  compute capability {major}.{minor}")
        if major < 7:
            usable = False
            print("  >>> PRE-VOLTA GPU. The Kaggle PyTorch build ships no Pascal")
            print("  >>> kernels, so every CUDA launch fails with")
            print("  >>> 'no kernel image is available for execution on the device'.")
            print("  >>> Push with --accelerator set to a T4 shape instead.")
    return usable


def build_prompt(report: str) -> str:
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


def parse_states(text: str) -> dict:
    """Model output to a state per finding. Anything unreadable abstains.

    Abstaining on a parse failure rather than guessing keeps a decoding problem
    from being taught to the imaging model as a confident negative.
    """
    states = {name: "not_mentioned" for name in FINDINGS}
    if not text:
        return states
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return states
    try:
        blob = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return states
    if not isinstance(blob, dict):
        return states
    for name in FINDINGS:
        value = blob.get(name)
        if isinstance(value, str) and value.strip().lower() in STATES:
            states[name] = value.strip().lower()
    return states


def rank_value(state: str) -> float:
    return STATES.index(state) / (len(STATES) - 1) if state in STATES else 1 / 6


def macro_auc(expert: np.ndarray, score: np.ndarray) -> tuple:
    from sklearn.metrics import roc_auc_score

    per_finding = {}
    for i, name in enumerate(FINDINGS):
        y = (expert[:, i] > 0.5).astype(int)
        if 0 < y.sum() < len(y):
            per_finding[name] = roc_auc_score(y, score[:, i])
    macro = float(np.mean(list(per_finding.values()))) if per_finding else float("nan")
    return macro, per_finding


def load_engine():
    """Load the reader, SPREAD ACROSS BOTH T4s.

    The first attempt at this run died on `device_map="auto"`, which put all of
    Qwen2.5-7B's fp16 weights on GPU 0 — 13.59 GiB of a 14.56 GiB card — leaving
    under a gigabyte for activations, and every batch then raised
    OutOfMemoryError. `NvidiaTeslaT4` grants two cards and the second sat idle.
    Capping per-device memory forces the shard.

    vLLM is not attempted. It is not installed on the Kaggle image, and pulling
    it in would drag its own torch build onto a Turing card mid-session. The
    grammar-constrained decoding it offers is worth having, but not at the price
    of the session; the prompt below carries a worked example instead, and the
    parser abstains on anything it cannot read.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = RUN_MODEL
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Leave headroom on every visible card rather than filling the first one.
    visible = torch.cuda.device_count() or 1
    budget = {i: RUN_GPU_BUDGET for i in range(visible)}
    print(f"loading {name} across {visible} device(s), {RUN_GPU_BUDGET} each")
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, device_map="auto", max_memory=budget)
    model.eval()
    placement = sorted({str(p.device) for p in model.parameters()})
    print(f"engine: transformers on {name}; weights on {placement}")

    def generate(prompts):
        chats = [tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        batch = tokenizer(chats, return_tensors="pt", padding=True,
                          truncation=True, max_length=3000)
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=RUN_MAX_NEW_TOKENS,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        return tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:],
                                      skip_special_tokens=True)

    return generate, name


def run_batch(generate, chunk: list[str]) -> list[str]:
    """Generate for a chunk, halving on out-of-memory rather than giving up.

    An OOM is a statement about batch size, not about the report — abstaining on
    it throws away real supervision for an infrastructure reason. The first run
    of this kernel abstained on 100% of studies for exactly that, and reported
    the resulting 0.500 as though it were a verdict on the method.
    """
    import torch

    size = len(chunk)
    while size >= 1:
        try:
            out = []
            for start in range(0, len(chunk), size):
                out.extend(generate([build_prompt(r) for r in chunk[start:start + size]]))
            return out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            size //= 2
            if size >= 1:
                print(f"  out of memory; retrying at batch {size}", flush=True)
        except Exception as exc:  # noqa: BLE001 - anything else abstains
            print(f"  batch failed ({type(exc).__name__}: {exc}); abstaining")
            return [""] * len(chunk)
    print("  out of memory even at batch 1; abstaining")
    return [""] * len(chunk)


def score_reports(generate, reports: list[str], label: str) -> list[dict]:
    """Reports to states, in batches, with progress and a wall-clock guard."""
    results, started = [], time.time()
    for start in range(0, len(reports), RUN_BATCH):
        chunk = reports[start:start + RUN_BATCH]
        raw = run_batch(generate, chunk)
        results.extend(parse_states(text) for text in raw)
        done = len(results)
        if start == 0 or done % (RUN_BATCH * 10) == 0 or done == len(reports):
            elapsed = time.time() - started
            rate = done / max(elapsed, 1e-6)
            print(f"  {label} {done:,}/{len(reports):,}  {elapsed / 60:.1f} min  "
                  f"eta {(len(reports) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)
        if time.time() - started > RUN_TIME_BUDGET:
            print(f"  time budget reached after {done:,}; the rest abstain")
            results.extend({name: "not_mentioned" for name in FINDINGS}
                           for _ in range(len(reports) - done))
            break
    return results


def main() -> int:
    started = time.time()
    report_environment()

    root = find_marker("train.csv")
    if root is None:
        raise SystemExit("competition data not mounted (train.csv)")
    train = pd.read_csv(root / "train.csv")
    print(f"reports: {len(train):,}")

    is_gold = train[FINDINGS].notna().all(axis=1)
    gold = train[is_gold].reset_index(drop=True)
    rest = train[~is_gold].reset_index(drop=True)
    print(f"gold studies: {len(gold)}   unlabelled: {len(rest):,}")

    generate, engine_name = load_engine()

    # The 58 first. They cost a minute and they answer the only question that
    # matters before the other 4,349 are paid for.
    gold_states = score_reports(generate, gold.Report.astype(str).tolist(), "gold")
    expert = gold[FINDINGS].to_numpy(dtype=float)
    ranks = np.array([[rank_value(s[name]) for name in FINDINGS] for s in gold_states])
    macro, per_finding = macro_auc(expert, ranks)

    abstain = float(np.mean([[s[name] == "not_mentioned" for name in FINDINGS]
                             for s in gold_states]))
    print(f"\n{'=' * 62}")
    print(f"GOLD-58 macro AUC: {macro:.4f}   (lexicon labeler: 0.769)")
    print(f"abstain rate: {abstain:.1%}   (lexicon labeler: 40.6%)")
    print(f"{'=' * 62}")
    for name, value in sorted(per_finding.items(), key=lambda kv: kv[1]):
        rate = np.mean([s[name] == "not_mentioned" for s in gold_states])
        print(f"  {name:18s} {value:.3f}   abstain {rate:5.1%}")
    print(flush=True)

    (OUT / "llm_gold_eval.json").write_text(json.dumps({
        "engine": "transformers", "model": engine_name,
        "n_gold": len(gold), "macro_auc": round(macro, 4),
        "abstain_rate": round(abstain, 4),
        "per_finding": {k: round(v, 4) for k, v in per_finding.items()},
        "lexicon_baseline": 0.769,
    }, indent=2))
    # States only — no report text is ever written out.
    (OUT / "llm_states_gold.json").write_text(json.dumps(
        {"StudyInstanceUID": gold.StudyInstanceUID.astype(str).tolist(),
         "states": gold_states}, indent=2))

    # A near-total abstain rate is an infrastructure failure, not a result. The
    # first run of this kernel abstained on every study after an out-of-memory
    # error and printed a 0.500 that read like a judgement on the method. Say
    # which of the two it is, because they call for opposite responses.
    if abstain > 0.5:
        print(f"RUN FAILED, not a verdict: {abstain:.0%} of findings abstained, "
              "which means the model produced nothing readable rather than "
              "reading the reports and finding nothing. Fix the run and repeat; "
              "do not record this as a measurement of the method.")
        return 1
    if macro < 0.769:
        print("Genuinely NOT better than the lexicon labeler on the 58 — the "
              "model read the reports and the states it chose rank worse. "
              "Stopping before the other 4,349; scoring gold first is what makes "
              "this decision cost a minute.")
        return 0

    limit = RUN_MAX_REPORTS or len(rest)
    rest = rest.head(limit)
    print(f"clears the baseline; reading the remaining {len(rest):,} reports\n", flush=True)
    rest_states = score_reports(generate, rest.Report.astype(str).tolist(), "corpus")

    (OUT / "llm_states_train.json").write_text(json.dumps(
        {"StudyInstanceUID": rest.StudyInstanceUID.astype(str).tolist(),
         "states": rest_states}, indent=2))

    counts = {name: {} for name in FINDINGS}
    for row in rest_states:
        for name in FINDINGS:
            counts[name][row[name]] = counts[name].get(row[name], 0) + 1
    (OUT / "llm_state_counts.json").write_text(json.dumps(counts, indent=2))
    print("\nstate distribution over the corpus:")
    for name in FINDINGS:
        total = sum(counts[name].values()) or 1
        share = {s: f"{counts[name].get(s, 0) / total:.0%}" for s in STATES}
        print(f"  {name:18s} " + "  ".join(f"{s}:{share[s]}" for s in STATES))

    print(f"\nwall clock {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
