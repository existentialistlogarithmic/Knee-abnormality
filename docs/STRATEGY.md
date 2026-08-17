# STRATEGY

## The bet

If the second-hand picture in `docs/FINDINGS.md` §3 holds — roughly 4,400 training
studies of which only ~58 carry expert image-derived labels — then this is not
primarily a computer vision competition. It is a **weak supervision** problem:
the scarce resource is not GPU time, it is trustworthy targets.

Everyone's imaging model is trained on labels derived from the same free-text
reports. The quality of those derived labels is an upper bound on everyone's
score, and it is upstream of every architectural choice anyone makes. That is
where a small-compute entrant can win, and it happens to be the part that looks
most like ordinary NLP engineering.

**Corollary:** effort goes to the labeler first, the imaging model second. A
mediocre backbone on well-calibrated targets beats a strong backbone on noisy
ones, and we can't out-GPU anyone anyway.

*(This bet is conditional. If Phase 0 shows the gold-label set is much larger
than believed, the bet is wrong and the plan changes — that is precisely why
Phase 0 comes first.)*

## Non-negotiables

1. **Grouped folds.** Any validation that lets the same site or scanner appear
   in both train and validation is measuring memorisation. Whatever the leakage
   audit (Phase 0 step 5) returns, folds are grouped — the number only tells us
   how badly we would have fooled ourselves.
2. **Soft labels with an abstain channel.** A report that does not mention the
   ACL is not a report that says the ACL is intact. Hard 0/1 targets destroy
   that distinction and it is exactly the distinction the gold set will punish.
3. **Calibration fits inside folds.** Any mapping from report-derived score to
   probability is fitted on training folds only. Fitting it on the gold set and
   then evaluating on the gold set produces a number that means nothing.
4. **No report text leaves the machine.** Competition Rule 4.b (Data Security).
   No hosted LLM API — not OpenAI, Anthropic, Gemini, or any other — sees a
   single report string. Multilingual work uses open-weights models running
   locally or inside a Kaggle kernel. If a shortcut ever seems to require it,
   the shortcut is wrong.
5. **Kaggle-to-Kaggle.** The bulk data is never downloaded locally. Each kernel
   mounts the previous kernel's output as a Dataset. Local machine handles CSVs,
   metadata, and report text analysis only.
6. **T4, never P100.** The current Kaggle PyTorch build ships no Pascal kernels.

## Compute budget

~30 GPU-hours per week on Kaggle, free tier. That is the entire budget. It
buys roughly one full cross-validated training run per week plus small
experiments — so a run that is not worth logging in `EXPERIMENTS.md` is not
worth launching. Anything projected over ~2 GPU-hours gets discussed first.

## Architecture sketch (subject to Phase 0)

```
DICOM headers ──► series selection rule ──► CPU cache kernel ──► volume cache
                                                                      │
report text ──► rule/lexicon labeler ──► soft labels + abstain ──►  training
                     │                          ▲                   kernel
                     └── open-weights encoder ──┘                      │
                                                                       ▼
gold ~58 studies ──► calibration + evaluation ◄────────────────── OOF preds
                                                                       │
                                                                       ▼
                                                            inference kernel
                                                          (internet off, <9 h)
```

## Why a rule/lexicon layer before a model

Not nostalgia — three concrete reasons:

1. **It is auditable in twelve languages.** A bilingual term table is something
   a human can review and correct. A multilingual encoder's mistakes on Turkish
   negation are invisible until they show up as a lost 0.02 AUC.
2. **The gold set is tiny.** With ~58 studies there is no honest way to
   fine-tune and validate a text model on gold labels. The rule layer needs no
   gold data to build, so the gold set stays a pure test set.
3. **It is a floor, not a ceiling.** The encoder is compared *against* the rule
   layer on the same gold subset. If it wins, it wins measurably; if it does not,
   we kept the interpretable thing.

The linguistic phenomena that actually decide the score are the ordinary ones:
negation ("no evidence of meniscal tear"), hedging ("possible", "cannot
exclude"), severity thresholds (grade 1 signal change vs a tear), laterality,
and prior-surgery mentions that read like findings. Each needs handling per
language, and each is a place where a lexicon can be inspected and fixed.

## Efficiency track

A second, lighter configuration is maintained from Phase 2 onward, not
retrofitted at the end: smaller input resolution, fewer slices, single fold.
Retrofitting efficiency after the fact usually means rebuilding the inference
kernel under deadline pressure.

## What would falsify this strategy

- Gold labels turn out to be plentiful → drop the weak-supervision emphasis,
  train directly, compete on imaging.
- Reports are available for test studies at inference time → the labeler becomes
  a first-class inference component, not just a training-time device, and the
  balance of effort shifts further toward text.
- Site metadata is absent from test → site-conditioned features are out;
  grouped CV stays.
