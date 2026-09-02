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

### 2. Self-distillation — untested, right category, precondition measured
**~2 h of CPU for the go/no-go, then 0 or 7.5 GPU-h depending on the answer.**

The model scores **0.8980** on the 58 gold studies. The public report labels
that trained it score **0.8927**. The student has overtaken its teacher.

That 0.005 gap is the whole reason this is worth trying, and the reason is
E048's rule: **a union pays when its members are comparable and imports errors
when they are not.** E023's union of two readers at 0.7446 and 0.7421 paid
**+0.070**. The four unions since — E033, E039, E046, E048 — each added a member
0.03–0.06 behind the incumbent and paid +0.0046, +0.0022, +0.0036, +0.0027, none
separated. **The model's own out-of-fold predictions are the first candidate
member that is not behind.**

The shape of the experiment:

1. `kaggle/64_oof_v1pub` — every study predicted once, by the model that held it
   out. Runs on **CPU from checkpoints that already exist**, so it costs zero
   GPU quota; re-running five folds with a wider dump would have cost ~7.5 h for
   the same file. It self-verifies: the gold macro is still computed from the
   gold subset, so the run must reproduce **0.8980**. If it does not, this
   kernel cut the folds differently from the trainer and its "out-of-fold"
   predictions are not out-of-fold — a teacher built on that would leak, train
   cleanly, and score worse for no visible reason.
2. `eda/distill_teacher.py` — score the 50/50 rank union of model and labels
   against the 58. **Free, minutes of CPU.**
3. **PRE-REGISTERED: spend GPU only if the union beats the report labels with a
   95% interval excluding zero.** The weight curve is printed in full and
   deliberately not adopted — an argmax over it is a free parameter fitted to 58
   studies, which `dataset-metadata.fused.json` rejects by name and E048
   declined once already. A test asserts the script cannot adopt one.
4. If it separates: publish the teacher, add a `v1pubdistil` lineage, five
   folds (~7.5 GPU-h), and let the board price it.

Honest coefficient: **unmeasured.** What is measured is the precondition, and
this is the first time in five attempts that it has been met.

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

**Score any candidate against the 58 gold first.** Two sets score 1.0000 because
they *contain the answer key* (E047), which makes them unevaluable rather than
good. The incumbent scores 0.8927, which is why its measurement means what it
appears to mean.

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
| auxiliary report targets | **null against its own control**, twice, sign flipping (E063) |
| more seeds of the 5-fold config | **closed by E064** — ten members scored 0.923, exactly the five |
| borrowing public weights | **closed by E046** for the family it priced; re-opened as a question by E066 and closed again for shingo257's CC0 ConvNeXt family at 0.8034 vs our 0.8477 |
| `mattiaangeli/rsna-knee-cnx-m448-f0-public` | licensed **`other`**, so excluded by E043's rule despite shipping complete geometry and model code |
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
- **0.93–0.94 is plausible** if the full-fit lineage pays at full weight, or if
  the distilled teacher separates. Those are the two routes with a mechanism.
- **0.945 needs both to pay near the top of their band AND a new public asset to
  appear.** It sits 0.007 under the field top, which is top-handful territory in
  a 1,866-team field. There is no measured path to it and this file will not
  draw one.
- **0.95+ has no measured path from here.** If anyone proposes one, ask for the
  coefficient and the interval it was measured with. Every route this project
  has tried and closed is listed in §2.3, with its number.

The one thing that will not get there is optimism about the numbers. This
project has overturned **eight** of its own confident claims, and every single
one was caught by a measurement rather than by reasoning.
