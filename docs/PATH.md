# What is left, what it is worth, and what 0.95 actually requires

Standing: **0.932 on the leaderboard** (2026-09-11, the CC0 CoAtNet arm at four
published weights, E097), previously 0.928 and 0.926 (2026-09-07, E083 — five full-fit
members), up from 0.924, 0.923, 0.846 and 0.725.
Leaderboard top **0.954**. **3,332 teams.** Final submission **2026-10-22**.

**Rank ~1,166 of 3,332, measured 2026-09-08 11:05 UTC (E088)** — 1,104 of 3,263
a day earlier (E085), #866 of 1,866 on 2026-09-02 (E070). The field grew 75% in
five days and 69 teams overnight. **A rank is a measurement with a date on it**,
and this one decays whether or not the score moves.

**461 teams sit at exactly 0.936 and 165 at exactly 0.939**, forks of public
notebooks, so the free public baseline is 0.010+ ahead of this project's
independently built system. **E043's rule excludes fewer of those assets than
this file claimed for a week — corrected in §2.0.** And §4 now answers the 0.95
question with the field's own shape: **the best available fork lands at 0.939**,
so 0.95 is not a gap that acquiring anything closes.

Last rewritten 2026-09-08 (E088). If this header ever reads more than a week
old, distrust the priorities below before distrusting the numbers.

---

## 1. The one number that governs the plan

Decomposed on ground truth, of the **+0.201** from this project's first imaging
model (0.725) to now (0.926):

| lever | board-measured contribution |
|---|---:|
| this project's own fused labels | **+0.089** |
| public CC0 report labels | **+0.077** |
| ensembling, one fold → five | **+0.032** |
| ~~full fit, at full weight (E083)~~ | **0.000 — inside the board's own seed floor (E092)** |
| **a distilled teacher (E083)** | **−0.013** |
| **architecture, every attempt** | **0.000** |

**CORRECTED 2026-09-10 (E092): full fit's +0.003 does not survive.** A pure
reseed of the same five-member full-weight ensemble — seeds 11-15 against 3-7,
nothing else changed — scored 0.923/0.921 against 0.926, so **the board's
like-for-like reseed spread is about 0.003 and the full-fit "lever" is the same
size as a draw.** Every board difference in this log under 0.003 is noise,
E064's +0.001 included.

**Labels and data are +0.166 of the +0.201.** Architecture has contributed zero
every time it has been measured, across 288px, DINOv2 twice, focal top-k,
per-finding pooling, slice positional embeddings, TTA and two blend families.

That is not a claim that architecture cannot matter. It is the observation that
this project cannot *measure* it: E060 established a **±0.03 noise floor** on
fine-tuned single-seed gold comparisons by changing nothing but the RNG seed and
watching −0.0284 come out. Every architecture effect ever hypothesised here is
smaller than that. Only DINOv2's −0.148 ever cleared it.

## 2. What is left, in priority order

### NOW. The export epoch — **built, verified, and owed two clicks**
**Zero further cost. Both arms already exist.** E086 found that
`FULL_FIT_EPOCH = 20` was read off the *fold* models, which train on 3,526
studies where full fit trains on 4,407 — so the same epoch buys a full-fit
member **25% more optimisation steps**, and E055 measured the fold curves
decaying past 21. Either every member behind 0.926 is a quarter past its peak,
or more data per pass supports more passes and 16 is undertrained. **A full-fit
model cannot be validated offline by construction**, so only the board answers
it.

`knee-infer-v1pubfe` (epoch 20) and `knee-infer-v1pubfe-early` (epoch 16) ran on
2026-09-07 at ~17:52 UTC and were verified end to end at 23:20 — five
checkpoints each, matched back to their five trainers by val macro AUC, no
cross-contamination in either direction. **Neither has been submitted.** The
three readings are pre-registered in E086 and the second is the one that matters
beyond this test: `v1pubfe` against the standing 0.926 is **seeds 11–15 against
seeds 3–7 with nothing else changed**, the board's first like-for-like reseed of
a full-weight ensemble, and it bounds how much of §2.1's +0.003 is draw rather
than lever.

### 0. The public-notebook route — **the largest gap, blocked on a decision**
**933 teams** are 0.010+ ahead of us by forking a public notebook, which the
rules permit (E088, 2026-09-08).

**CORRECTED BY E088: this paragraph used to say E043 excludes every asset those
notebooks use, `other` and CC-BY-NC-SA alike. E043 does not say that.** Verbatim,
its middle tier reads *"non-commercial matches this competition's own CC-BY-NC
4.0 winner licence, so NC is not the obstacle it first looked like; ShareAlike on
derivatives is the part to read before shipping."* Its avoid-tier is
**`not-declared`**. A "read this before shipping" had hardened into an
"excluded", on the largest measured gap in the project. **The ShareAlike question
is unanswered, not settled** — and answering it is free.

What *is* cleanly excluded, and on the merits rather than the label:
`tonylica/rsna-knee-bend-dinov3-0917-repro-assets`, which gates the 0.937
notebook, is licensed *"Other (specified in description)"* **with an empty
description**. Nothing is specified, so no grant exists (E088).

The strongest admissible arm is `dreaddevelopment/raptor-knee-widedense` — CC0,
0.924 upstream from one model, 0.9167 on the 58 gold with gold held out. Against
our 0.926 full-fit ensemble that is **0.002 apart and maximally different in
kind** — the best-shaped union in this log, better than E023's 0.0025 pair that
paid +0.070. The blend kernel is written and tested.

It is blocked because upstream persists no outputs, so the arm must be *run*,
which means forking someone else's notebook under this account. **That is the
account owner's decision** — `HANDOFF.md` §4d has the steps and the control arm.

**E088's census changes the shape of this, and it is the most actionable thing
in this file.** All 626 public notebooks were pulled and every asset's licence
read. Of 94 distinct assets, **48 are CC0 or Apache**, and the three most-mounted
assets in the entire competition are CC0. More usefully, the 0.937 system's own
public write-up names **CoAtNet as its strongest single arm at base weight
0.60**, and that arm is **entirely CC0**:

| sub-model | weight | file | dataset (all CC0) |
|---|---:|---|---|
| maxspan-v5 | 0.55 | `raptor_ft_coatnet_v5_full_swa.pt` | `raptor-knee-maxspan` |
| native384-v8 | 0.20 | `raptor_ft_coatnet_v8_full_swa.pt` | `raptor-knee-native384` |
| maxspan-v5-reverse | 0.15 | *none* — horizontal flip of v5 at inference | — |
| native384dense-v10 | 0.10 | `raptor_ft_coatnet_v10_full.pt` | `raptor-knee-native384dense` |

**The 0.937 notebook's only `no-grant` dependency gates the DINOv3 arm, not this
one.** So the strongest arm needs no restricted asset and no fork of one — which
is a materially smaller decision than the one this section was written around.
It is still one arm of four and is not 0.937 by itself, and every figure in it is
its authors' self-report. **Worth measuring; not worth believing.**

**Two CC0 label sets were missed by the weekly survey** and are mounted by 182
and 89 public notebooks respectively: `pilkwang/rsna-knee-llm-labels` and
`lixin73/rsna-knee-llm-report-labels-sol56`. §2.3's survey searched by author
name rather than by what the field mounts. **Screen both with E080's exact-cell
test first** — and note the label lever is the only one that has ever paid here.

### 1. The full-fit lineage — **PAID, and it is the standing score**
Each fold model trains on 80% of the corpus and never sees ~12 of the 58 expert
studies. A full-fit model sees all 4,407 and all 58.

E064 priced one full-fit member inside a six-model ensemble at **+0.001**,
diluted to a sixth and indistinguishable from noise. **E083 measured the same
lever at full weight: five full-fit members and nothing else scored 0.926
against the five-fold ensemble's 0.923 — +0.003, with data exposure the only
variable.** Three times the diluted reading, same direction.

**This is the only lever with a positive board reading, and it is the same
family as the +0.089 and +0.077 that built the project: more and better data,
never architecture.**

### 2. Self-distillation — **CLOSED, and it cost a board point to learn**
The teacher separated offline by +0.0261 (E069), the student by +0.0221 (E076),
the gain looked broad across findings (E071), and the board priced the finished
lineage at **0.910 against 0.923 — −0.013, with the teacher the only variable
(E083).**

**The offline rig got the sign wrong on an interval that excluded zero.** E082
found the mechanism before the score landed: both lineages cut the same folds,
so fold 0's expert labels reach fold 0's training targets through the other
folds' models, while `v1public`'s report-text targets carry no such path. E076
scored a leaked arm against a clean one.

Three things follow and they are binding:

1. **Gold-58 is retired for teacher comparisons**, as E060 retired single-seed
   gold for architecture.
2. **E076's Synovitis-above-the-text-ceiling claim is withdrawn.** It rested on
   a leaked evaluation.
3. **A union that separates offline is not a lever.** E048's comparability rule
   says which unions are worth an afternoon, not which ones pay.

### 3. Survey public label sets — **CLOSED BY MEASUREMENT (E089)**, not by assumption
**Three passes have now looked and none found a public set that beats the
incumbent.** E089 screened the two most-mounted CC0 label sets in the field —
182 and 89 public notebooks respectively, both missed by earlier surveys because
those searched by author name rather than by what the field mounts:

| label set | gold macro AUC | against our 0.8927 |
|---|---:|---|
| `pilkwang/rsna-knee-llm-labels` | 0.8700 | **behind by 0.023** |
| `lixin73/…-report-labels-sol56` | 0.8352 | **behind by 0.058** |

**One candidate survives and it is a board question, not an offline one.**
`dreaddevelopment/rsna-knee-labels` is CC0 and covers all 4,407 studies *minus
exactly the 58 gold*, so gold-58 cannot price it at all. Its authors report
**+0.013** from adopting it. It is the label set behind the CC0 CoAtNet weights
in §2.0.

**And the screen itself needed the control it demanded of everything else.** The
rule below said "what fraction of gold cells equal the expert value exactly"
with no threshold. **The gold set is 34.5% positive, so all-zeros already
reproduces 65.5% of cells** — `lixin73` reads 79.9% and is ordinary, not leaked,
while E047's and E080's real answer keys read 100%. `eda/survey_public_labels.py`
now prints the floor beside every rate and compares only against the incumbent.

The original guidance, still worth following if a genuinely new set appears:

### 3. Survey public label sets weekly — free, CPU only
Both of the largest board jumps came from label sets appearing. E047 found four
new sets in five days; E062's survey on 2026-09-02 found none since. Check
`stevenleehans`, `pilkwang`, `tasmeemreza`, `shingo257`, `mattiaangeli`, and
search Kaggle datasets for "rsna knee", "rsna knee labels", "knee report
labels", sorted by update time.

**The screen is now a script**, and it runs before anything is mounted:

```bash
python eda/survey_public_checkpoints.py --checkpoints artifacts/public/*.pt
```

Many shared checkpoints are self-describing — shingo257's ConvNeXt family stores
its backbone, geometry and the author's own gold AUC beside the weights — so a
family can often be screened for the price of a download, with no inference, no
cache and no GPU (E066). Compare a foreign **single fold** against our
**per-fold 0.8477**, never against the pooled 0.8980: the pooled comparison is
wrong by the entire width of the ensembling effect. And read every fold — fold 0
of that family reads 0.8677 and the mean is 0.8034.

**Score any candidate against the 58 gold first, and run the exact-cell test
before the score.** Contamination is the norm, not the exception: E047 saw two
answer-key sets in five, and **E081's survey saw three in four**. A set can be
built from the answer key without reproducing it perfectly, so the screen is
"what fraction of gold cells equal the expert value exactly", not the AUC alone.

**The bar is the E069 teacher at 0.9188, not the report labels at 0.8927.**
Comparing a candidate against 0.8927 credits it with a gain already banked.
E081's `tsuyu122` set — clean, four cohorts, 0.9124, Spearman 0.665 against ours
— is the first outside member since E023 to land *inside* the comparability
band, and the three-way union still came back **+0.0062, CI [−0.000, +0.014],
not separated**. E048's rule says which unions are worth an afternoon; it does
not say they pay.

### 4. Board-test a blend once — the offline nulls do not settle it
**One submission, ~1 GPU-h, no training.**

E046 concluded "blending is not a lever at this sample size." That is
**instrument-limited, not measured-dead.** The four blend results were +0.002 to
+0.005 against a gold-58 interval of ±0.03 — a fifth of the width. The board
just resolved **+0.001** (0.923 → 0.924, E064). Five submissions a day are
available and inference is ~1.0 h of a 9 h cap.

So the cheapest honest test of every blend this project has declined is to spend
one submission on the best of them. It cannot make the standing score worse —
submissions are scored independently.

### 5. Nothing else has a measured coefficient
Everything below is either closed or unmeasurable at this budget. Listed so that
a future session does not re-derive them as ideas:

| route | status |
|---|---|
| **new public label sets, 2026-09-07 survey** | **three of four are answer keys** (E080); the clean fourth does not separate in union, +0.0062 CI [−0.000, +0.014] (E081) |
| auxiliary report targets | **null against its own control**, twice, sign flipping (E063) |
| self-distillation, the whole route | **closed by E083** — separated offline at +0.0261 and +0.0221, scored −0.013 on the board; the offline rig was leaking (E082) |
| more seeds of the 5-fold config | **closed by E064** — ten members scored 0.923, exactly the five |
| **mixing fold and full-fit members** | **closed by E087** — ten mixed members scored 0.926, exactly what five full-fit already scored. Purely linear, nothing super-additive |
| borrowing public weights | **E046's closure is STALE (E088).** It closed the route with "we now beat what we were borrowing", measured 2026-08-29 against a public system at 0.917. The public systems now score **0.936-0.939 against our 0.926**. True when written, false now — a closure resting on a moving external number needs a date and a re-read, exactly like a rank. shingo257's CC0 ConvNeXt family stays closed at 0.8034 vs our 0.8477 (E066) |
| `mattiaangeli/rsna-knee-cnx-m448-f0-public` | licensed **`other`**, so excluded by E043's rule despite shipping complete geometry and model code |
| blending anything with anything | three attempts: +0.0046, +0.0022, +0.0036, none separated |
| a Synovitis reader | **closed by E059** — only 13 of 27 true cases are written about at all; a perfect reader caps at 0.8076 and the model already scores 0.790 |
| DINOv2 as a second family | −0.035 and −0.148 on two folds (E051) |
| more epochs | converged; folds peak 18–21 then decay (E046, E055) |
| TTA | −0.0006 [−0.006, +0.005], the sharpest null in the log (E050) |
| 288px geometry | 0.668 on the board against 0.725 |
| architecture A/Bs generally | **the instrument cannot resolve them.** Four seeds of five folds is 30 GPU-h — a whole weekly quota for one A/B |
| **full resolution (288px) against a good teacher** | **closed by E079** — val macro 0.8094 against 0.9067 on 881 held-out studies, same fold and teacher, geometry the only variable. E046's "one large untested region" is tested and negative |

## 3. What 0.94 would require

`PATH.md` carried a per-finding reading of this from E041: +0.017 on the board
means no finding below roughly **0.870**, and after E044 only **Synovitis
(0.779)** sat under 0.80. E059 then closed Synovitis as *unwritten in the
reports* rather than badly read, with a text ceiling of 0.8076.

**That reasoning was superseded by E071 and the supersession is now itself
withdrawn (E083).** The claim that a distilled teacher broke the Synovitis
ceiling rested on the leaked evaluation E082 identified, and the board priced
the lineage that produced it at −0.013. Synovitis stands where E059 left it.

So 0.94 has **one** candidate mechanism, not two: **more data per model**, which
is §2.1 and which the board has actually paid for. There is no measured second.

## 4. About 0.95, plainly — answered with the field's own shape

**Measured 2026-09-08 11:05 UTC (E088). 3,332 teams, top 0.954.**

| score | teams | share |
|---|---:|---:|
| ≥ 0.954 | 2 | 0.06% |
| **≥ 0.950** | **12** | **0.36%** |
| ≥ 0.945 | 48 | 1.4% |
| ≥ 0.940 | 141 | 4.2% |
| ≥ 0.939 | 306 | 9.2% |
| ≥ 0.936 | 933 | 28.0% |
| **≥ 0.926 (ours)** | **1,166** | **35.0%** |

**The plateaus say which scores are bought and which are earned.** 461 teams sit
at *exactly* 0.936, 165 at *exactly* 0.939, 129 at 0.937 — forks. Above 0.939 the
clustering stops: 26, 26, 12, 15, 14, then single digits. **The best fork lands
at 0.939. 141 teams are above it on unpublished work.**

**So the answer is structural, not tactical. There is no route to 0.95 that
consists of acquiring something.** Copying every public asset in this
competition — every notebook, checkpoint and label set, licence questions set
aside entirely — reaches **0.939**. The remaining **0.011** is unpublished by
construction: it is precisely what the top twelve have and have not shared. And
129 of the 141 teams above the fork line have not closed it either.

**0.95 is the prize band.** Top 10 of 3,332, against a top of 0.954. Asking this
repo how to reach it is asking how it wins the competition outright, and no
lever in this log has a coefficient that gets there. **Anyone proposing one
should be asked for the coefficient and the interval it was measured with.**

### What the honest targets actually are

- **0.926 is banked** and is rank ~1,166 of 3,332.
- **0.929 or so is the one measurement this project still owns.** E086's two
  arms are built, verified and unsubmitted; the export-epoch question is a
  genuine coin flip and the reseed arm bounds how much of full fit's +0.003 is
  draw. Cost: two clicks. See §2 NOW.
- **0.936–0.939 is commodity, and it is where the licence question actually
  bites.** It is 28% of the field. Reaching it means using public assets, and
  §2.0 now records that E043 excludes fewer of them than this file claimed for a
  week. **The unanswered question is ShareAlike, and answering it costs a read.**
- **0.940+ needs something nobody has published.** There is no measured
  mechanism in this log, and §1's decomposition says why: labels and data
  contributed +0.166 of the +0.201 and are spent, ensembling is closed by E064,
  mixing by E087, resolution by E079, distillation by E083, and architecture has
  measured **0.000 every single time** against a ±0.03 noise floor it was never
  able to see through.
- **0.95 has no path this file can draw**, and E088 is the measurement that says
  so rather than the opinion that says so.

### The one genuinely unexploited asset, and what it is worth

E088 found that `dreaddevelopment` publishes **15 datasets, all CC0**, carrying
**14 checkpoints** across CoAtNet (seven geometry variants), ConvNeXtV2-B336,
EfficientNetV2-L480 and a CNN336 — and that this project has used **none** of
them. The screen that should have surfaced them was reporting a false negative
because it read the wrong metadata spellings; that is fixed and tested.

Screened correctly, `raptor_ft_coatnet_v4_full` self-reports **0.9167 gold
against our pooled 0.8980** — inside E048's ±0.02 band and **ahead of ours, the
first public family in this log to be both**. Its per-finding profile is
complementary where it counts least and most: Baker's 0.9873 and Medial OA
0.9860, but **Synovitis 0.7575, below our 0.779**.

**This is worth measuring. It is not worth believing yet.** Every number in the
paragraph above is the author's, on the author's split. E048's comparability
rule has now failed twice as a predictor of what *pays* (E081 offline, E087 on
the board), and four unions in this log have returned +0.0046, +0.0022, +0.0036
and +0.0027, none separated. **A different-in-kind member that reports ahead of
us is the best-shaped union the log has ever had, and the log's own history says
the most likely outcome is still nothing.**

What it plausibly buys, if it behaves: something in the 0.93s. **It does not
reach 0.95, and nothing here does.**
