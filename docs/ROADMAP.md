# ROADMAP

Final submission: **2026-10-22 23:59 UTC** (`VERIFIED`).
Entry and team-merge deadline: **2026-10-15 23:59 UTC** (`VERIFIED`).
Winners' requirement deadline: **2026-11-05** — training code, video and method
description. Worth knowing now, because it means the winning solution has to be
reproducible, not just scored.
From 2026-08-18 that is **nine weeks** to final submission, eight to the entry
deadline. The competition opened 2026-08-05 and has 1,866 teams.

**Target: macro ROC-AUC ≥ 0.90** (`VERIFIED` metric: `(1/12) Σ AUC_i`).
**Budget: ~24 s per study** — ~1,300 test studies inside the 9-hour cap. See `STRATEGY.md` → "The 0.90 target" for why
it is read as AUC rather than accuracy, why it is a stretch against a ~0.809
public baseline, and which gate catches it if the label ceiling makes it
unreachable.

Only Phase 0 is planned in detail. Planning Phase 2 now would be planning
against unverified facts.

---

## Phase 0 — verification and data audit  (target: days, not weeks)

| step | deliverable | state |
|---|---|---|
| 0.1 | Auth verified; competition metadata and file inventory | **DONE for metadata** (metric, deadlines, prize, code-comp flag, rules accepted). Full file listing is paginating with checkpoints — Kaggle rate-limits after ~25k files, so it resumes rather than restarting. Total size/count still open (`FINDINGS.md` 3.1). |
| 0.2 | Metric, submission schema, study counts, gold-label count, target names and positive rates, site metadata presence, report language/length | **DONE.** See `FINDINGS.md` §2–§4. Headline: 4,407 studies, 58 gold, 12 named targets, 10 languages, **no reports at test time**, **no site column anywhere**. |
| 0.3 | Kaggle CPU kernel: DICOM **header-only** scan → metadata parquet (site, manufacturer, model, field strength, laterality, pixel spacing, slice thickness, slice count) | **NEXT, and now on the critical path.** It is the only source of a fold-grouping key (§3.6) and of laterality. Series *plane* no longer needs deriving — the host provides it. |
| 0.4 | Series-per-study distribution; consistent sequences/planes; cheapest series-selection rule | **PARTLY DONE from the CSVs**: 24,371 series over 4,407 studies, mean 5.53, median 5, range 3–14; planes Sagittal 9,864 / Coronal 8,609 / Axial 5,898; one fluid-sensitive flag (the two columns are identical). Remaining: which combination is reliably present per study. |
| 0.5 | **Leakage audit**: AUC gap between random and site-grouped K-fold on a trivial baseline | **BLOCKED on 0.3** — there is no site column to group by until the headers are scanned. |

**Gate: full stop after 0.5 for review before any modelling.**

Sequencing note: 0.3 is now the bottleneck for both 0.4 and 0.5, so it launches
first. It needs no pixels and no GPU.

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

CPU cache-build kernel → 2.5D backbone + attention pooling → 12 heads trained on
Phase 1 soft labels with gold studies heavily weighted. Site-grouped CV.
Resumable, checkpointed, wall-clock guarded. Report OOF macro AUC, per-finding
AUC, and prediction spread.

---

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
