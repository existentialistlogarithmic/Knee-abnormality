# ROADMAP

Final submission: **2026-10-22 23:59 UTC** (`VERIFIED`).
Entry and team-merge deadline: **2026-10-15 23:59 UTC** (`VERIFIED`).
Winners' requirement deadline: **2026-11-05** — training code, video and method
description. Worth knowing now, because it means the winning solution has to be
reproducible, not just scored.
From 2026-08-18 that is **nine weeks** to final submission, eight to the entry
deadline. The competition opened 2026-08-05 and has 1,866 teams.

**Target: macro ROC-AUC ≥ 0.90** (`VERIFIED` metric: `(1/12) Σ AUC_i`).
**Budget: ~24 s per study** — ~1,300 test studies inside the 9-hour cap.

See `STRATEGY.md` → "The 0.90 target" for why it is read as AUC rather than
accuracy. Note the target is **below the field**, not above it: the leaderboard
top is 0.951 and the top 200 teams all exceed 0.917 (`FINDINGS.md` §8).

Phase 0 is complete, so Phase 1 and Phase 2 below are planned from measurements
rather than assumptions.

---

## Phase 0 — verification and data audit  (target: days, not weeks)

| step | deliverable | state |
|---|---|---|
| 0.1 | Auth verified; competition metadata and file inventory | **DONE for metadata** (metric, deadlines, prize, code-comp flag, rules accepted). Full file listing is paginating with checkpoints — Kaggle rate-limits after ~25k files, so it resumes rather than restarting. Total size/count still open (`FINDINGS.md` 3.1). |
| 0.2 | Metric, submission schema, study counts, gold-label count, target names and positive rates, site metadata presence, report language/length | **DONE.** See `FINDINGS.md` §2–§4. Headline: 4,407 studies, 58 gold, 12 named targets, 10 languages, **no reports at test time**, **no site column anywhere**. |
| 0.3 | DICOM header-only scan | **DONE.** 24,386 series, **819,635 slices**, 0 errors, **372 s** on a Kaggle CPU kernel (0.015 s/series). Output 2.6 MB parquet. |
| 0.4 | Series selection rule | **DONE** (`FINDINGS.md` §10). Host plane column agrees with headers on 24,371/24,371 series. **Axial fluid-sensitive exists for 100% of studies**; all three fluid-sensitive planes for 90.6%. Rule: one fluid-sensitive series per plane, axial as the guaranteed fallback. |
| 0.5 | **Leakage audit** | **DONE. Random K-fold inflates macro AUC by 0.087** (`FINDINGS.md` §9). Grouping key is a scanner fingerprint with frequency rounded to 2 dp; 178 groups. |

**PHASE 0 COMPLETE — 2026-08-18.** All five steps done, every headline claim
verified. `docs/FINDINGS.md` is the record.

---

## Phase 1 — the label problem  (the priority)

`src/report_labeler.py` + evaluation against the gold studies.

Target vocabulary is fixed and known: `ACL`, `MCL`, `Medial Meniscus`,
`Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`,
`Synovitis`, `Baker's`, `Contusion`, `Fracture`.

Language priority follows the measured mix — English alone covers only 39% of
reports, so Spanish (16%) and Turkish (12%) are first-class, not afterthoughts:

| tier | languages | share of reports |
|---|---|---:|
| 1 | en, es, tr | 67% |
| 2 | el, hr/bs, de, bg | 25% |
| 3 | nl, fr | 5% |

1. ~~Bilingual term table per language, human-reviewable, checked in.~~ **DONE** —
   `src/lexicons/findings.csv` (198 terms) and `cues.csv` (154 cues, 10 languages).
2. ~~Rule/lexicon layer: negation, hedging, severity thresholds.~~ **DONE** —
   `src/report_labeler.py`, macro AUC **0.745** on gold. Laterality still to do.
3. ~~Soft labels with a confidence/abstain channel.~~ **DONE** — five channels:
   asserted / hedged / low_severity / negated / absent.
3a. **NEXT: close the three coverage gaps.** Synovitis (0.561 AUC, 89% abstain),
   Lateral OA (0.691, 77%), Medial OA (0.695, 76%). AUC tracks abstain rate
   almost monotonically, so these are the cheapest gains on the board.
4. Evaluation on the 58 gold studies: per-finding AUC, agreement rate, and a
   confusion analysis of *where* report and image labels diverge — **always with
   intervals**, because 58 studies cannot separate 0.86 from 0.90.
5. Only then: comparison against an open-weights multilingual model.
6. Calibration fitted inside folds only.

---

## Phase 2 — imaging baseline

Everything below is now determined by Phase 0 measurements rather than guessed.

### Inputs, settled

- **Series selection:** one fluid-sensitive series per plane, preferring
  sagittal + coronal + axial. Axial fluid-sensitive exists for **100%** of
  studies and is the guaranteed fallback; all three planes are present for 90.6%
  (`FINDINGS.md` §10). The host's `Anatomical_Plane` is trustworthy — it agreed
  with the DICOM headers on 24,371 of 24,371 series.
- **Laterality:** `Laterality` is populated on 79% of series; `SeriesDescription`
  carries it for some of the rest (e.g. `LT_...`). Mirror right knees so the
  model sees one anatomy.
- **Targets:** `artifacts/phase1/soft_labels.parquet` — 12 soft scores plus a
  five-way channel per study. Gold studies weighted heavily; the abstain channel
  must reach the loss as "no supervision here", not as a zero.
- **Folds:** `src/folds.py`, scanner fingerprint with frequency rounded to 2 dp.
  178 groups. **Non-negotiable**: random K-fold inflates macro AUC by 0.087.

### The budget, measured

~1,300 test studies inside the 9-hour cap is **~24 s/study**. Reaching the data
costs 0.059 s/study, so essentially all of it is available for decode and
inference. At a median 99 fluid-sensitive slices per study, one series per plane
is roughly 90 slices — comfortably affordable.

### Baselines that must be beaten

| baseline | grouped CV | note |
|---|---:|---|
| constant priors | 0.500 | confirmed on the leaderboard |
| **scanner metadata only** | **0.669** | no pixels at all — see below |
| report labeler vs expert (58 gold) | 0.761 | the label ceiling, roughly |

**An imaging model that does not clear ~0.67 has not demonstrated the pixels
contribute anything.** That is the bar, and it is higher than it looks because
the metadata model is free.

### Findings that need special handling

- **Synovitis is text-limited** — report vocabulary carries almost no
  information about it (sensitivity 0.59 at precision 0.57 against a 0.47 base
  rate). Its report-derived labels should be down-weighted; if the model is to
  learn it, it must come from the images and the 58 gold studies.
- **Lateral OA** has the weakest labeler AUC after Synovitis and the widest
  interval. Expect it to be noisy throughout.

### Architecture

2.5D: a pretrained backbone over slice stacks, attention pooling to study level,
12 output heads. Resumable and checkpointed with a wall-clock guard, because
Kaggle sessions die. T4, never P100.

### Order of work

1. ~~CPU cache-build kernel~~ **DONE** — `kaggle/03_cache_build_shard{0..3}`,
   four shards, (3, 20, 192, 192) uint8 per study, ~2.2 MB each. Mounted by the
   training kernel via `kernel_sources` so the ~10 GB never leaves Kaggle.
2. **Train one fold, confirm it beats 0.669 grouped, check prediction spread.**
   In progress — `kaggle/04_train`.
3. Only then the full cross-validated run.

## Phase 3 — inference kernel

**Images only** — there is no report text at test time (`FINDINGS.md` §2.6).
Self-contained, internet off, comfortably under 9 h with headroom for a hidden
test set far larger than the 3-study public stub. Pre-submit sanity checks: row count, column names and order,
no NaNs, probabilities in [0,1], IDs matching the sample submission exactly.
A second lightweight config is kept current for the efficiency track.

---

## Standing risks

| risk | early warning | response |
|---|---|---|
| **0.90 is out of reach because report labels cap it** | Phase 1 gold evaluation | say so plainly and early rather than at the deadline; redirect effort to whichever of labels / images is actually binding |
| ~~Report text unavailable at test~~ | **confirmed in 0.2** | labeler is training-time only; inference kernel is image-only. Already reflected in the plan. |
| Gold set is site-concentrated | needs 0.3 | calibration claims get much weaker; report intervals, not point estimates |
| No fold-grouping key is recoverable from headers either | 0.3 output | fall back to grouping on whatever proxy exists (manufacturer, field strength) and say plainly that CV is weaker than we want |
| Header scan exceeds kernel time limits | 0.3 runtime | shard the scan across several kernels by study prefix |
| GPU budget exhausted mid-week | weekly GPU-hours in `EXPERIMENTS.md` | CPU-only work (labeler, cache) is always available as fallback |
| Submission kernel times out on the hidden test set | 0.1 file counts vs test size | measure per-study inference cost early; keep the efficiency config as the escape hatch |
