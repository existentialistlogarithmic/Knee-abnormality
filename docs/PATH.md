# What is left, what it is worth, and what 0.95 actually requires

Standing: **0.924 on the leaderboard** (2026-09-02, E064 — five `v1public`
folds plus one full-fit member), up from 0.923, 0.846 and 0.725.
Leaderboard top **0.952**. 1,866 teams. Final submission **2026-10-22**.

Last rewritten 2026-09-02. The version before this one described the 0.846
world and had gone two board jumps stale; if this header ever reads more than a
week old, distrust the priorities below before distrusting the numbers.

---

## 1. The one number that governs the plan

Decomposed on ground truth, of the **+0.199** from this project's first imaging
model (0.725) to now (0.924):

| lever | board-measured contribution |
|---|---:|
| this project's own fused labels | **+0.089** |
| public CC0 report labels | **+0.077** |
| ensembling, one fold → five | **+0.032** |
| full fit, as one member in six | +0.001 |
| **architecture, every attempt** | **0.000** |

**Labels and data are +0.166 of the +0.199.** Architecture has contributed zero
every time it has been measured, across 288px, DINOv2 twice, focal top-k,
per-finding pooling, slice positional embeddings, TTA and two blend families.

That is not a claim that architecture cannot matter. It is the observation that
this project cannot *measure* it: E060 established a **±0.03 noise floor** on
fine-tuned single-seed gold comparisons by changing nothing but the RNG seed and
watching −0.0284 come out. Every architecture effect ever hypothesised here is
smaller than that. Only DINOv2's −0.148 ever cleared it.

## 2. What is left, in priority order

### 1. The full-fit lineage — **running now**, ~6 GPU-h
Each fold model trains on 80% of the corpus and never sees ~12 of the 58 expert
studies, the ones carrying `GOLD_WEIGHT=8.0` and the only labels known to match
what the leaderboard scores. A full-fit model sees all 4,407 studies and all 58.

E064 priced one full-fit member inside a six-model ensemble at **+0.001**, which
is the smallest move the board can show and is diluted to a sixth. Four more
seeds (`knee-train-v1pubfull-s4/5/6/7`) then `knee-infer-v1pubfull5` — five
full-fit members and nothing else — measure the same lever at full weight.
Against the 0.923 five-fold ensemble it varies exactly one thing: how much data
each member saw.

**This is the last lever with a positive board reading.** If it returns 0.924 or
below, the data lever is spent too.

### 2. Survey public label sets weekly — free, CPU only
Both of the largest board jumps came from label sets appearing. E047 found four
new sets in five days; E062's survey on 2026-09-02 found none since. Check
`stevenleehans`, `pilkwang`, `tasmeemreza`, `shingo257`, `mattiaangeli`, and
search Kaggle datasets for "rsna knee", "rsna knee labels", "knee report
labels", sorted by update time.

**Score any candidate against the 58 gold first.** Two sets score 1.0000 because
they *contain the answer key* (E047), which makes them unevaluable rather than
good. The incumbent scores 0.8927, which is why its measurement means what it
appears to mean.

### 3. Nothing else has a measured coefficient
Everything below is either closed or unmeasurable at this budget. Listed so that
a future session does not re-derive them as ideas:

| route | status |
|---|---|
| more seeds of the 5-fold config | **closed by E064** — ten members scored 0.923, exactly the five |
| auxiliary report targets | **null against its own control**, twice, sign flipping (E063) |
| borrowing public weights | **closed by E046** — this system overtook the one it was borrowing from |
| blending anything with anything | three attempts: +0.0046, +0.0022, +0.0036, none separated |
| a Synovitis reader | **closed by E059** — only 13 of 27 true cases are written about at all; a perfect reader caps at 0.8076 and the model already scores 0.790 |
| DINOv2 as a second family | −0.035 and −0.148 on two folds (E051) |
| more epochs | converged; folds peak 18–21 then decay (E046, E055) |
| TTA | −0.0006 [−0.006, +0.005], the sharpest null in the log (E050) |
| 288px geometry | 0.668 on the board against 0.725 |
| architecture A/Bs generally | **the instrument cannot resolve them.** Four seeds of five folds is 30 GPU-h — a whole weekly quota for one A/B |

## 3. What 0.94 would require

`PATH.md` has carried a per-finding reading of this since E041 and it still
holds: +0.017 on the board means no finding below roughly **0.870**. After E044
only **Synovitis (0.779)** sits below 0.80, and E059 closed it — the finding is
unwritten in the reports rather than badly read, and the text ceiling is 0.8076
against a model already at 0.790.

So 0.94 does not come from fixing the weakest finding. It comes from more data
per model, which is exactly what §2.1 is testing, or from a label set that has
not appeared yet.

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

- **0.924 is banked**, and it clears the top-200 cut of 0.917.
- **0.93–0.94 is plausible** if the full-fit lineage pays at full weight, and it
  is the only route with a positive reading behind it.
- **0.95+ has no measured path from here.** If anyone proposes one, ask for the
  coefficient and the interval it was measured with. Every route this project
  has tried and closed is listed in §2.3, with its number.

The one thing that will not get there is optimism about the numbers. This
project has overturned **eight** of its own confident claims, and every single
one was caught by a measurement rather than by reasoning.
