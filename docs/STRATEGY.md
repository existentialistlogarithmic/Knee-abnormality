# STRATEGY

## The bet — now confirmed

`docs/FINDINGS.md` settles it: **4,407 training studies, exactly 58 with expert
labels** (1.3%), every one of the other 4,349 carrying a free-text report and
nothing else. This is not primarily a computer vision competition. It is a
**weak supervision** problem: the scarce resource is not GPU time, it is
trustworthy targets.

Everyone's imaging model is trained on labels derived from the same free-text
reports. The quality of those derived labels is an upper bound on everyone's
score, and it is upstream of every architectural choice anyone makes.

**Caveat added after reading the leaderboard:** 200 teams are already above
0.917, which nobody reaches without turning those reports into usable labels. So
this is not an untapped edge — it is table stakes, and the edge is in doing it
better than a field that has already done it once. Plan accordingly: the label
pipeline is still where the effort goes, but expecting it to be a secret is
wrong.

**Corollary:** effort goes to the labeler first, the imaging model second. A
mediocre backbone on well-calibrated targets beats a strong backbone on noisy
ones, and we can't out-GPU anyone anyway.

**The host agrees.** The data-description says the reports are provided "from
which you may wish to derive the labels for the remaining studies". This is the
intended solution path, not a loophole — which also means every serious
competitor is doing it, and the edge comes from doing it better rather than from
thinking of it.

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

### It is macro AUC, not accuracy — now confirmed

The Evaluation page states the metric exactly: `Final Score = (1/12) Σ AUC_i`,
"the macro-averaged AUC ROC". So accuracy is not the scored quantity, and at
these prevalences it is actively misleading.
A finding present in 5% of studies gets **95% accuracy** from a model that
predicts "negative" every single time and has learned nothing. Chasing accuracy
above 0.90 on twelve imbalanced findings is a target that a useless model meets.

So the target is **macro ROC-AUC ≥ 0.90 on the leaderboard**. The evaluation
harness still reports per-finding AUC, balanced accuracy and
accuracy-at-threshold alongside it, because macro AUC hides which of the twelve
findings is dragging, and that is the thing you act on.

One consequence of the averaging worth internalising: every finding counts for
exactly 1/12 of the score regardless of how rare it is. `MCL`, the rarest in the
gold subset, is worth precisely as much as `Effusion`. A model that is excellent
on the eight easy findings and random on four caps out at about 0.83. Chasing
0.90 means the *worst* findings decide it, so effort belongs there rather than
on polishing the ones already working.

### 0.90 is below the bar, not above it — corrected 2026-08-18

The ~0.809 "public baseline" in the brief is **wrong by a wide margin**. Read
from the actual leaderboard: the top score is **0.9510**, and the **top 200
teams all sit at 0.9170 or better**, median 0.9200.

So 0.90 does not reach the top 200 of 1,866 teams. The real bars are ~0.941 for
a top-ten prize, ~0.930 for the top 20%, ~0.917 to make the top 200.

**Hitting 0.90 would therefore not be success; it would be the floor.** Anyone
planning against 0.90 as a stretch goal — which is what this document said
before the leaderboard was read — is planning to finish near the bottom of the
ranked field. The target for real work is **0.94+**.

I am still not going to promise a leaderboard number. What can be promised is
that the thing which most plausibly caps the score gets measured first and
reported plainly, so effort is spent against a known ceiling rather than hope.

### The ceiling, now measured on 1,300 studies instead of 58

A metadata-only model scored **0.669** in scanner-grouped CV against
report-derived labels and **0.531** on the leaderboard against expert labels
(`FINDINGS.md` §11). That 0.138 gap is the clearest measurement available of how
far report-derived targets sit from the truth being scored.

It does not mean the labels are worthless — that model had no anatomical
information at all, so everything it learned was site reporting convention. But
it does establish a hard working rule: **report-label CV ranks models; it never
estimates the leaderboard.**

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
   No hosted LLM API of any provider sees a single report string. Multilingual
   work uses open-weights models running locally or inside a Kaggle kernel. If a
   shortcut ever seems to require it, the shortcut is wrong. Language
   identification already runs offline (`py3langid`), so the ten-language split
   cost nothing in this regard.
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

## The runtime budget is now a number

The hidden test set is **~1,300 studies**, about 5.5 series each at a median 30
slices — roughly **215,000 slices**. The cap is 9 hours (32,400 s). That is
**~24 seconds per study**, including DICOM reading, for everything the kernel
does.

This is a design constraint, not a warning. It rules out reading every slice of
every series at full resolution, and it means series selection (which of the ~5.5
series to actually use) is a performance decision as much as an accuracy one.
Measure per-study inference cost in Phase 2, early, on real data.

## Efficiency track — $18,000, and cheaper to reach

The efficiency track pays $7,000 / $6,000 / $5,000. Its scoring, quoted from the
Efficiency Prize Evaluation page:

```
Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400   (minimised)
```

Eligibility is low: the submission must be one the team selected, and must beat
the `sample_submission.csv` benchmark on the private leaderboard. Since runtime
enters divided by the 9-hour cap, **a fast kernel scores well even without a
top-ten AUC** — which is exactly the shape of a small-compute entry.

*(As written that first term is negative, because `Benchmark` is below `maxAUC`.
Recorded verbatim rather than silently "fixed"; watch the forum for an erratum
before optimising against it too literally.)*

So the light configuration is maintained from Phase 2 onward, not retrofitted:
smaller input resolution, fewer slices, single fold. Retrofitting efficiency
after the fact means rebuilding the inference kernel under deadline pressure,
and here it would mean forfeiting the more reachable half of the prize pool.

## What would falsify this strategy

- Gold labels turn out to be plentiful → drop the weak-supervision emphasis,
  train directly, compete on imaging.
- Reports are available for test studies at inference time → the labeler becomes
  a first-class inference component, not just a training-time device, and the
  balance of effort shifts further toward text.
- Site metadata is absent from test → site-conditioned features are out;
  grouped CV stays.
