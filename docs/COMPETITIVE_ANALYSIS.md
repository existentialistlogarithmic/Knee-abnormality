# Why the leaders are at 0.88–0.95 and this project is at 0.725

Read from **public Kaggle notebooks published by their authors for others to
learn from**. Nothing here is copied code — what is recorded is each author's
stated reasoning and their measured numbers, credited, so that the techniques
can be implemented independently. Where a number is theirs it is labelled as
theirs.

Sources, all public notebooks on this competition:

| author | notebook | what it contributes here |
|---|---|---|
| Sadam Torres | *Domain adaptation beats resolution: DINOv2 on knee* (LB 0.883) | the most rigorous writeup; nearly every measurement below |
| Mattia Angeli | *Bend the Knee to DinoV3 (ensembled)* | cross-attention over all series; rank blending; community credits |
| AADIGUPTA | *[0.899] Let me Cook* | per-finding pooling rules; evidence-gated promotion |
| Ryan Holbrook (host) | *Efficiency LB* | the efficiency track exists and is scored daily |

---

## 1. The gap, itemised

| axis | leading public systems | this project | gap |
|---|---|---|---|
| **report-label quality** vs the 58 gold studies | **0.881** — local LLM (Qwen3-14B-AWQ, vLLM, fixed JSON schema) reading into a closed vocabulary, then a deterministic map to probabilities *(Torres)* | **0.769** — hand-built multilingual lexicon | **−0.112** |
| backbone | DINOv2 ViT-B/14, **fully fine-tuned** | ImageNet resnet34 | large |
| input resolution | 224 → 336 px | 192 px | small (**+0.017** for them, 224→336) |
| per-slice features | CLS token **and** patch-token mean | pooled backbone output only | some |
| study pooling | mean **within plane**, then concatenate planes; laterality and plane embeddings | attention over all planes×slices flattened together | some |
| ensembling | **rank**-mean over 5 folds | probability-mean, 1–2 folds | +0.006 … +0.03 |

**The label gap is the headline.** Their report reader scores 0.881 against
expert truth; this project's lexicon scores 0.769. Everything the imaging model
learns is bounded by what its teacher knows.

## 2. The measurement that reframes everything here

*Torres* reports that **gold-58 macro AUC predicts the leaderboard with a
constant offset of about +0.044**, across three independent systems:

| system | gold-58 | public LB | offset |
|---|---:|---:|---:|
| frozen DINOv2 + head | 0.771 | 0.776 | +0.005 |
| fine-tuned @224, 5 fold | 0.824 | 0.866 | +0.042 |
| a different architecture entirely | 0.857 | 0.903 | +0.046 |

(The first row's +0.005 is theirs as published; the +0.044 claim rests on the
second and third. Treated here as **UNVERIFIED against this project's own
models** until `eda/pool_gold_oof.py` produces a number to check it with.)

This project reached the same conclusion independently and from the opposite
direction — §11 of `FINDINGS.md` records report-label CV mis-ranking models, and
§13 measures what 58 studies can resolve. The published offset, if it holds
here, turns the gold-58 pool from a filter into a **leaderboard estimator**, and
that is worth more than any single modelling change on this list.

Working backwards from it: this project's 0.725 leaderboard score implies a
gold-58 of roughly **0.68**. The folds now training will say whether that is
right, and that check is the first thing to do when they land.

## 3. The finding that contradicts a conclusion recorded here

`FINDINGS.md` §11 concluded from a clean like-for-like run that 288 px is 0.037
*worse* than 192 px on the board, and therefore that the resolution hypothesis
is not supported. *Torres* measures the opposite sign — 224 → 336 px is worth
**+0.017** — but on a **fine-tuned self-supervised backbone**, and the same
writeup argues resolution is second-order:

> Fine-tuning at the same 224px moved exactly the focal findings the most:
> Medial Meniscus 0.679 → 0.850, MCL 0.708 → 0.825, ACL 0.727 → 0.840.
> Resolution was worth +0.017 on top, not the +0.09 that adaptation bought.

Both can be true: resolution may pay on a backbone that already knows what to
look for and not on one that does not. What is not in dispute is the **ordering
of effort** — this project spent its two most expensive experiments on the axis
worth +0.017 and none on the axis worth +0.09.

That diagnosis fits this project's own per-finding numbers exactly. The weakest
findings on fold 1 are **Medial Meniscus 0.656, PF OA 0.659, Synovitis 0.663,
MCL 0.669** — the focal ones, the same set their frozen backbone collapsed on
and their fine-tuned one recovered.

## 4. The efficiency track is the winnable one — `VERIFIED`

Quoted verbatim from the competition's *Efficiency Prize Evaluation* page:

> Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400

minimised, where `Benchmark` is `sample_submission.csv` (**0.500**, confirmed by
this project's own first submission) and `maxAUC` is the best private score
(public top today: **0.952**).

The denominator is **negative**, so the AUC term is negative and more AUC lowers
the score. Break-even for one extra hour is `(maxAUC − Benchmark) × 3600/32400`
= **0.0502 AUC per hour** at today's top (0.0444 if the top were 0.90).

Scored with published runtimes — lower is better:

| system | AUC | hours | efficiency |
|---|---:|---:|---:|
| hypothetical: this project at 0.85 in 2.0 h | 0.850 | 2.0 | **−1.658** |
| hypothetical: this project at 0.80 in 1.5 h | 0.800 | 1.5 | −1.603 |
| public DINOv2 @224, 5 fold *(Torres)* | 0.866 | 3.0 | −1.583 |
| **this project today, 192px, 1 fold** | **0.725** | **0.8** | **−1.515** |
| public DINOv2 @336, 5 fold *(Torres)* | 0.883 | 4.0 | −1.509 |
| public top, assuming ~8 h | 0.952 | 8.0 | −1.217 |

**This project's 0.725 model already outranks the public 0.883 model on
efficiency**, purely because it runs in 0.8 h instead of 4. That runtime has
been treated throughout as slack against the 9-hour cap. It is not slack, it is
the one axis on which this project is currently ahead of the public field.

Two consequences:

1. **Spend the runtime, but spend it on AUC.** There are ~8 hours of cap and
   ~2 hours of efficiency-competitive budget. Anything that buys more than
   0.05 AUC per hour is worth it on *both* boards.
2. **Cheap AUC beats expensive AUC twice over.** Rank-mean ensembling, better
   labels and a better backbone cost no inference time at all; more folds and
   more resolution cost it linearly.

## 5. What to do, in order of measured value per hour spent

1. **Replace the lexicon labeler with an open-weights LLM reader.** Their +0.112
   of label quality is the largest single number on this page. `STRATEGY.md`
   forbids sending report text to a *hosted* API and explicitly permits
   open-weights models run locally or in a Kaggle kernel, which is exactly what
   *Torres* describes — so this is available without touching Rule 4.b. Two
   layers, as they argue: ask the model to pick from a **closed vocabulary**,
   then map vocabulary to probabilities in deterministic Python, because a model
   is good at the first job and bad at the second.
2. **Fine-tune a self-supervised backbone properly** — DINOv2 is already
   training here. Their evidence says this is where the focal findings come from
   and it is worth ~5× what resolution is.
3. **Rank-mean, not probability-mean, when combining models.** AUC reads order
   only, so averaging sigmoids lets the most confident member dominate for no
   reason. Free.
4. **Soft targets calibrated from data, not chosen by hand.** This project's
   0.90/0.62/0.32/0.05 are stipulated; theirs come from measured positive rates
   ("an effusion described as mild is positive about 45% of the time").
5. **Per-slice CLS + patch-mean, plane and laterality embeddings, mean-within-
   plane pooling.** Small, cheap, and each is argued from a specific failure.
6. **More folds.** Worth it on the main board, and at ~0.006–0.03 for an extra
   hour it is roughly break-even on the efficiency board — so a main-board
   config and an efficiency config should diverge here, not earlier.
