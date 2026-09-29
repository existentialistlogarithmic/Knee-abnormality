# RSNA Knee Abnormality Detection — research brief

**Rewritten 2026-09-29.** Every number below is measured in this project unless
marked as someone else's self-report. Paste this whole file into Claude Research
and ask the questions in §6.

---

## 1. The task

Kaggle **code competition**: submissions run as notebooks, 9 h cap, 2×T4, five
submissions a day. Input is knee MRI **DICOM studies** (multiple series per
study — sagittal / coronal / axial, fluid-sensitive and T1, with and without fat
suppression). Output is **12 binary findings** per study:

ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
Effusion, Synovitis, Baker's cyst, Contusion, Fracture.

Metric: **macro ROC-AUC** — the plain mean of twelve per-finding AUCs, so the
weakest findings dominate the headroom.

Hidden test ≈1,300 studies. The **public leaderboard is only ~390 of them
(~30%)**; the rest decide the final standing. Deadline 22 October.

## 2. Where we are

**Public LB 0.945.** The best public *notebook* in the field self-reports
**0.943**, so we lead the shared recipes. Ladder, all measured:

| config | board |
|---|---:|
| 5 own members + 4 CoAtNet arms | 0.938 |
| 6 members | 0.940 |
| 8 members | 0.942 |
| 8 members + 1 foreign pipeline | 0.944 |
| 8 members + 3 foreign pipelines | **0.945** |
| same + corrected vote arithmetic | 0.945 (+0.000) |
| 10 members, no foreign pipelines | 0.944 |

**Four +0.002 steps, then +0.001, then +0.000.** The curve has flattened.

**What we actually run:** four CoAtNet "Raptor" arms (someone else's CC0
checkpoints) + three foreign CoAtNet packages from a second author + **ten of
our own 2.5D CNN members** (resnet34 ×5 seeds, resnet50, resnext50_32x4d,
regnet_y_3_2gf, wide_resnet50_2, resnet101 — a 2D backbone over slices with
attention pooling to a study). Everything combined by **equal-weight rank-mean**.

## 3. The instruments, and why this is the core problem

| instrument | n | usable resolution |
|---|---:|---|
| public leaderboard | ~390 studies | **±0.003** |
| "gold" radiologist labels | **58 studies** | ±0.0153 absolute, ±0.006 paired |
| LLM-extracted report labels | 4,349 studies | noisy; disjoint from the gold 58 |

**We have no honest out-of-fold predictions from our own models** — every member
is a full fit over all 4,407 studies, so it memorises the gold 58 (each reads
0.99+ there, which we use only as a load-integrity check). Gold-58 has as few as
**9–11 positives** in its weakest columns.

So: we cannot measure anything smaller than roughly ±0.015 offline, and the
board's floor is ±0.003. Several routes below were closed by measurements whose
intervals span zero, and that is the honest limit of what we know.

## 4. Per-finding position — this is where the remaining headroom is

Our composite on the 58 gold studies, worst first:

| finding | ours | n_pos |
|---|---:|---:|
| **Lateral OA** | **0.8375** | 11 |
| **Synovitis** | **0.8483** | 27 |
| **PF OA** | **0.8610** | 21 |
| **Lateral Meniscus** | **0.8845** | 23 |
| Fracture | 0.9431 | 18 |
| Contusion | 0.9453 | 19 |
| Baker's | 0.9710 | 12 |
| Medial Meniscus | 0.9712 | 26 |
| ACL | 0.9755 | 24 |
| Medial OA | 0.9829 | 15 |
| MCL | 0.9864 | 9 |
| Effusion | 0.9938 | 35 |

**Four findings hold 0.370 of the 0.380 total gap to 0.95.** Because the metric
is a plain mean of twelve, **+0.005 of macro = +0.060 summed** — one finding
moved 0.060, or four moved 0.015 each.

**A strong structural signal: medial ≫ lateral on both paired findings.**

| | medial | lateral | gap |
|---|---:|---:|---:|
| OA | 0.9829 | 0.8375 | **0.145** |
| Meniscus | 0.9712 | 0.8845 | **0.087** |

**We tested and killed the obvious explanation (field of view).** All ten
independently-trained arms we have access to show the same gap, across four
different crops, and the *widest* field of view has the *largest* gap
(+0.191 on OA). It is not coverage.

**Partial explanation — the labels carry the same asymmetry.** Two public
LLM report-labellers scored against radiologist truth on the 58:

| finding | labeller A | labeller B | our model |
|---|---:|---:|---:|
| Synovitis | 0.6938 | 0.6762 | **0.8483** |
| Lateral OA | 0.7885 | 0.8240 | **0.8375** |
| Lateral Meniscus | 0.8414 | 0.8665 | **0.8845** |
| Medial OA | 0.9079 | 0.8915 | **0.9829** |
| Effusion | 0.8299 | 0.6957 | **0.9938** |

Spearman(label quality, model quality) over the twelve = **+0.566** (n=12, not
separated from zero). **But the models beat their own supervision nearly
everywhere** — Synovitis by +0.155, Effusion by +0.164 — so label noise is being
averaged out, not inherited. **The exceptions are exactly the lateral pair:**
Lateral OA beats its best label by only +0.014 and Lateral Meniscus by +0.018.
Those two look label-limited; Synovitis and PF OA do not.

## 5. Routes already closed, with numbers — please do not re-suggest these

- **More resolution** — −0.088. Closed.
- **Per-finding blend weights** (fitted, leakage-free) — board returned the
  identical number to a flat 0.5. Closed.
- **Blend weight tuning** generally — 0.000.
- **More own members / more families** — 10 members vs 8 gave +0.002, inside the
  floor; two new families at once should have moved more if the lever were live.
- **Reseeding** — +0.000 measured explicitly.
- **AutoML** — no trustworthy validation signal to select on.
- **Gradient-boosted stacker over member scores** (n=4,349, honest OOF): plain
  rank-mean **0.8635**, GBM on scores **0.8598**, GBM + scanner/metadata
  **0.8636**. Doing nothing wins.
- **Per-scanner score normalisation** — **−0.0209**, against a shuffled-group
  control of −0.0028. Scanner identity is *signal* here (prevalence varies by
  site); removing it destroys information.
- **Eight borrowed model families screened on gold-58.** Clean split at ~0.90:
  everything ≥0.909 paid, everything ≤0.892 cost. Rejected: DINOv2×20 slot
  attention (0.8381, costs −0.0171), RadImageNet heads (0.8543 and 0.8256),
  another team's member set (0.8918), an EfficientNet-B3 MIL (0.6278).
  Correlation with our composite ran 0.317–0.890 across **both** groups and did
  not separate them — **quality is the binding axis, not independence.**
- **Notably: no foreign asset beats us on a single finding**, weak ones included.
  Borrowing is exhausted.

**Also closed for us:** synovitis-via-effusion surrogate (effusion head 0.7975 vs
synovitis 0.8094 on the target); left-right flip augmentation (knees are
pre-mirrored and four targets are medial/lateral-specific); slice reversal
(model is permutation-invariant to 2.4e-7).

## 6. What we want researched

Please prioritise **clinical/imaging-literature answers** over generic ML advice.
Anything in §5 is already measured and closed.

1. **The medial/lateral asymmetry.** Is a large, consistent medial ≫ lateral
   performance gap a known phenomenon in knee-MRI machine learning or in
   radiology reporting itself? Candidate causes we have not ruled out:
   lateral-compartment pathology being rarer and subtler; reporting conventions
   that mention medial findings more consistently; the lateral meniscus's
   popliteal hiatus being a known false-positive source. **Which is best
   supported, and does anything in the literature actually close the gap?**

2. **Synovitis specifically.** Our model reads 0.848 and beats its report label
   by +0.155, so this is signal-limited rather than label-limited. What MRI
   sequences and findings do radiologists actually use to call synovitis
   (effusion-synovitis complex on fluid-sensitive sequences? Hoffa-synovitis?
   post-contrast only?), and **is non-contrast synovitis grading known to have a
   performance ceiling?**

3. **Sequence routing per finding.** We feed 3-plane triplets at 336–384 px,
   12–20 slices. Is there evidence that specific findings are decisively better
   read from specific sequences — e.g. PD-FS sagittal for meniscal tears, axial
   for patellofemoral OA, coronal for MCL — strongly enough to justify
   **per-finding specialist models** rather than one 12-output head?

4. **Training with noisy LLM-extracted labels.** Our labels come from radiology
   reports via an LLM and are measurably worse than the models trained on them.
   Beyond plain BCE, what actually works: noise-robust losses (GCE, symmetric
   CE), co-teaching, per-finding label smoothing calibrated to per-finding label
   quality, or loss re-weighting by estimated label reliability? **Is there
   evidence any of these beats plain BCE at this noise level (label-vs-truth AUC
   0.69–0.99 depending on finding)?**

5. **Submission selection under a small public leaderboard.** ~390 public
   studies with a ±0.003 floor, 1,300 total, two final submissions to choose.
   What is the defensible selection strategy to protect *private* rank — and
   what is the evidence on how often public-LB-chasing at this sample size loses
   places on the private split?

6. **Is +0.005 even distinguishable?** Given macro-AUC over 12 findings on ~390
   studies with the positive counts above, what is the actual standard error of
   the public score, and is a 0.945 → 0.950 difference measurable or noise?

## 7. What we can act on

We have GPU quota and ~3 weeks. We can train new models, change architectures,
change losses, change sequence handling, and rebuild the preprocessing cache. We
cannot get more labelled data, and we cannot use non-CC0/non-permissive assets
without checking the licence first. **Answers that imply "train something
different, here is why it should help the four weak findings" are the most
useful thing you can give us.**
