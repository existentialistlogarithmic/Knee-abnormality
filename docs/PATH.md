# What is left, what it is worth, and what 0.95 actually requires

Standing: **0.926 on the leaderboard** (2026-09-07, E083 — five full-fit
members), up from 0.924, 0.923, 0.846 and 0.725.
Leaderboard top **0.954**. 3,263 teams. Final submission **2026-10-22**.

**Rank 1,104 of 3,263, measured 2026-09-07 (E085).** The field grew 75% in five
days, which is why E070's "#866 of 1,866" went stale almost immediately — a rank
is a measurement with a date on it. **492 teams sit at exactly 0.936 and 163 at
0.937**, forks of two public notebooks, so the free public baseline is 0.010+
ahead of this project's independently built system. E043's licence rule excludes
every asset those notebooks depend on; see §2.0.

Last rewritten 2026-09-07, revised 23:20 UTC the same day to add §2 NOW (E086's
two arms, built and unsubmitted) and E087. If this header ever reads more than a
week old, distrust the priorities below before distrusting the numbers.

---

## 1. The one number that governs the plan

Decomposed on ground truth, of the **+0.201** from this project's first imaging
model (0.725) to now (0.926):

| lever | board-measured contribution |
|---|---:|
| this project's own fused labels | **+0.089** |
| public CC0 report labels | **+0.077** |
| ensembling, one fold → five | **+0.032** |
| **full fit, at full weight (E083)** | **+0.003** |
| **a distilled teacher (E083)** | **−0.013** |
| **architecture, every attempt** | **0.000** |

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
655 teams are 0.010+ ahead of us by forking a public notebook, which the rules
permit. **E043's licence rule excludes every asset they use** (`other`, or
CC-BY-NC-SA), so 0.936 is not reachable under that rule as written.

The strongest admissible arm is `dreaddevelopment/raptor-knee-widedense` — CC0,
0.924 upstream from one model, 0.9167 on the 58 gold with gold held out. Against
our 0.926 full-fit ensemble that is **0.002 apart and maximally different in
kind** — the best-shaped union in this log, better than E023's 0.0025 pair that
paid +0.070. The blend kernel is written and tested.

It is blocked because upstream persists no outputs, so the arm must be *run*,
which means forking someone else's notebook under this account. **That is the
account owner's decision** — `HANDOFF.md` §4d has the steps and the control arm.

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
| borrowing public weights | **closed by E046** for the family it priced; re-opened as a question by E066 and closed again for shingo257's CC0 ConvNeXt family at 0.8034 vs our 0.8477 |
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

## 4. About 0.95, plainly

**0.952 is rank 1 of 1,866 teams.** Asking how this code reaches 0.95 is asking
how it wins the competition outright, and no plan should pretend otherwise.

Two things are true of the systems up there, both from their own published
notebooks:

1. **They are aggregations, not single models.** The strongest public notebook
   is a rank blend of twenty DINOv2 checkpoints plus a DINOv3 ViT plus two
   RadImageNet stages, and its author credits ten other competitors by name for
   the checkpoints, the label sets and the aggregation ideas. That is hundreds
   of GPU-hours of *pooled community* compute, not one team's 30 a week.
2. **Their advantage starts at the labels.** This project's own biggest single
   jump came from adopting one publicly shared CC0 label set.

The rules permit this: *"It's okay to share code if made available to all
Participants on the forums."* Publicly shared weights and label sets are
legitimate inputs, with attribution.

**The honest split, updated for where the board actually is:**

- **0.926 is banked.** It was #866-territory at 0.924 and has not been re-ranked.
- **0.93–0.94 has one live mechanism and no measured coefficient for it.**
  Full fit paid +0.003 at full weight. Nothing in the log says a second helping
  of the same lever pays again — E064 showed ten same-kind members scoring
  exactly what five scored — so +0.003 is a measurement, not a rate.
- **0.945+ has no route this file can draw.** The teacher route was the
  candidate and the board closed it at −0.013. The label-set route was surveyed
  on 2026-09-07: three of four new sets were answer keys, and the clean one did
  not separate (E080, E081). What remains is waiting for a genuinely new public
  asset, which is not a plan so much as a subscription.
- **0.95+ has no measured path from here.** If anyone proposes one, ask for the
  coefficient and the interval it was measured with. Every route this project
  has tried and closed is listed in §2.5, with its number.

**The lesson of 2026-09-07 is worth more than the +0.002.** An offline
instrument said +0.0221 with a 95% interval excluding zero, and the board said
−0.013. The interval was honest and the instrument was compromised, and no
amount of care with the statistics would have caught it — only E082's reading of
*how the folds touch each other*, and then the board. **This project has now
overturned nine of its own confident claims, and every single one was caught by
a measurement rather than by reasoning.**
