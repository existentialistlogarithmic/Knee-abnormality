# STRATEGY

## The bet — now confirmed

`docs/FINDINGS.md` settles it: **4,407 training studies, exactly 58 with expert
labels** (1.3%), every one of the other 4,349 carrying a free-text report and
nothing else. This is not primarily a computer vision competition. It is a
**weak supervision** problem: the scarce resource is not GPU time, it is
trustworthy targets.

Everyone's imaging model is trained on labels derived from the same free-text
reports. The quality of those derived labels is an upper bound on everyone's
score, and it is upstream of every architectural choice anyone makes. That is
where a small-compute entrant can win, and it happens to be the part that looks
most like ordinary NLP engineering.

**Corollary:** effort goes to the labeler first, the imaging model second. A
mediocre backbone on well-calibrated targets beats a strong backbone on noisy
ones, and we can't out-GPU anyone anyway.

### One verified fact reshapes the pipeline

**Reports do not exist at inference time.** `test.csv` has a single column,
`StudyInstanceUID` — no `Report`. So the labeler never runs on test data. Its
entire job is to manufacture training targets for 4,349 studies that would
otherwise be unusable, and then it is thrown away before submission.

That makes the labeler *more* valuable, not less. It is pure upstream leverage:
every point of label quality propagates into the imaging model, which is the
only thing that scores. It also means a labeler mistake cannot be caught at
inference — there is no text there to sanity-check against.

Concretely, the pipeline is:

```
4,349 reports ──► labeler ──► soft targets ─┐
                                            ├──► imaging model ──► submission
58 gold studies ────────────────────────────┘         (images only)
```

## The 0.90 target

The stated goal is **above 0.90**. Two things need saying before it becomes a
plan, and neither is a reason not to aim there.

### It should be macro AUC, not accuracy

If the metric is macro ROC-AUC (still `UNVERIFIED` — claim 2.4), then accuracy
is not the scored quantity, and at these prevalences it is actively misleading.
A finding present in 5% of studies gets **95% accuracy** from a model that
predicts "negative" every single time and has learned nothing. Chasing accuracy
above 0.90 on twelve imbalanced findings is a target that a useless model meets.

So the target is read as **macro ROC-AUC ≥ 0.90 on the leaderboard**. The
evaluation harness reports macro AUC, per-finding AUC, balanced accuracy, and
accuracy-at-threshold side by side, so if the metric turns out to be something
else the numbers are already there and nothing has to be rebuilt.

### 0.90 is a stretch, and the honest version of that is a number

The public baseline is reported at ~0.809 (`UNVERIFIED` — claim 5.4). Going from
0.809 to 0.90 is not an incremental gain; it is roughly halving the remaining
distance to a perfect score. That can happen — the competition opened
2026-08-05 and already has 1,866 teams, so public baselines are early and weak,
and the label pipeline is genuinely under-exploited — but it is not the default
outcome of doing competent work.

I am not going to promise a leaderboard number, and I would distrust anyone who
did. What can be promised is this: the thing that most plausibly *caps* the
score gets measured first, early, and reported plainly, so effort is spent
against a known ceiling instead of against hope.

### The ceiling worth measuring first

If report-derived labels agree with image-derived truth only ~82% of the time
(`UNVERIFIED` — claim 5.2), then **98.7% of training targets are noisy**, and
that noise sets a ceiling on what any architecture can reach. Two consequences:

1. **Measure the ceiling before scaling the model.** Train the cheapest possible
   model on report-derived labels, evaluate on the gold subset, and compare that
   against the same model trained on gold labels alone. The gap is the price of
   weak supervision, and it tells us whether 0.90 is reachable through better
   labels, better images, or not at all.
2. **Every point of label quality is worth more than a point of backbone.**
   Which is the whole reason the label pipeline outranks the imaging model in
   this repo.

There is a hard caveat, and it is now measured rather than feared: the gold
subset is **exactly 58 studies**, and the rarest finding (`MCL`) has **9
positives**. A per-finding AUC estimated there carries a 95% interval roughly
±0.13 wide, and nearer ±0.20 for `MCL` — wider than the entire 0.809-to-0.90 gap
we are chasing.

**So a "0.90" from the gold subset means nothing.** Any claim of hitting the
target must come from the leaderboard, or from a properly grouped OOF estimate
over thousands of studies. The 58 can tell us a labeler is broken. They cannot
tell us it is good. This is the single easiest way to fool ourselves here.

### Gates

| when | check | if it fails |
|---|---|---|
| end of Phase 0 | Is the ROC-AUC macro-averaged over the 12 findings? (The API says only "Roc Auc Score".) | retarget against the real averaging scheme |
| end of Phase 1 | Report-label AUC against gold, per finding | if the labeler cannot clear ~0.85 on the findings it should find easily, the ceiling is the labeler, not the model |
| first Phase 2 run | Grouped OOF macro AUC, plus prediction spread | a collapsed spread means the model is predicting priors; the AUC is not real progress |
| every run after | OOF vs LB gap | a widening gap means the fold scheme is leaking, and the CV number is fiction |

## Non-negotiables

1. **Grouped folds.** Any validation that lets the same site or scanner appear
   in both train and validation is measuring memorisation. Whatever the leakage
   audit returns, folds are grouped — the number only tells us how badly we
   would have fooled ourselves. **Note the new blocker:** there is no site
   column in any competition CSV (`FINDINGS.md` §3.6), so the grouping key has
   to be recovered from DICOM headers before any fold scheme can be trusted.
   Until then, treat every CV number as provisional.
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
   the shortcut is wrong. Language identification already runs offline
   (`py3langid`), so the ten-language split cost nothing in this regard.
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
train_series.csv (plane, fluid-sensitive)  ─┐
                                            ├─► series selection ─► CPU cache
DICOM headers (site, scanner, laterality) ──┘         │              kernel
        │                                             ▼
        └──► fold grouping key ─────────────────► volume cache
                                                      │
4,349 reports ─► lexicon labeler ─► soft labels ──►  training kernel (T4)
                      │                  ▲                │
                      └─ open-weights ───┘                ▼
                         encoder                     OOF preds
                                                          │
58 gold studies ──► evaluation (with intervals) ◄─────────┘
                                                          │
                                                          ▼
                                        inference kernel — IMAGES ONLY
                                       (no reports at test; internet off)
```

Two arrows that do *not* exist are the important ones: report text never
reaches the inference kernel, and no site column reaches the fold splitter
until the header scan provides it.

## Why a rule/lexicon layer before a model

Not nostalgia — three concrete reasons:

1. **It is auditable in ten languages.** The measured mix is en 39%, es 16%,
   tr 12%, el 7%, hr 7%, de 6%, bg 5%, nl 4%, fr 2%, bs 2%. A bilingual term
   table is something a human can review and correct; a multilingual encoder's
   mistakes on Turkish negation are invisible until they show up as a lost
   0.02 AUC. Note that English covers only 39% — an English-only labeler
   forfeits three fifths of the training set.
2. **The gold set is tiny.** With 58 studies there is no honest way to
   fine-tune *and* validate a text model on gold labels — one use exhausts it.
   The rule layer needs no gold data to build, so the gold set stays a pure
   test set, used once, late.
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
