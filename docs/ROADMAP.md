# ROADMAP

Final submission: **2026-10-22** (UNVERIFIED — see `FINDINGS.md` 2.8).
Entry and team-merge deadline: **2026-10-15** (UNVERIFIED).
From 2026-08-17 that is roughly **nine weeks**, of which the entry deadline
consumes eight. Both dates are confirmed by the same API call that unblocks
Phase 0, so they stop being guesses on day one.

**Target: macro ROC-AUC ≥ 0.90.** See `STRATEGY.md` → "The 0.90 target" for why
it is read as AUC rather than accuracy, why it is a stretch against a ~0.809
public baseline, and which gate catches it if the label ceiling makes it
unreachable.

Only Phase 0 is planned in detail. Planning Phase 2 now would be planning
against unverified facts.

---

## Phase 0 — verification and data audit  (target: days, not weeks)

| step | deliverable | state |
|---|---|---|
| 0.1 | Kaggle auth verified; competition file inventory with sizes, nothing downloaded | **written, tested, blocked on the `KAGGLE_API_TOKEN` secret** (`eda/phase0_01_auth_and_files.py`, runs in CI) |
| 0.2 | Metric, submission schema, study counts, gold-label count, target names and positive rates, site metadata presence, report language/length distribution | **written and tested against a synthetic fixture** (`eda/phase0_02_audit_tabular.py`); discovers the schema blind rather than assuming it. Same blocker as 0.1. |
| 0.3 | Kaggle CPU kernel: DICOM **header-only** scan across the full training set → metadata parquet (study UID, series description, modality, plane, slice count, pixel spacing, slice thickness, manufacturer, laterality, field strength). Report runtime and output size. | not started |
| 0.4 | Series-per-study distribution; which sequences/planes are consistently present; the cheapest reliable series-selection rule | not started |
| 0.5 | **Leakage audit**: measured AUC gap between random K-fold and site/scanner-grouped K-fold on a trivial baseline. One number, with its uncertainty. | not started |

**Gate: full stop after 0.5 for review before any modelling.**

Sequencing note: 0.3 is a long-running kernel, so it launches as soon as 0.2
identifies the DICOM path layout, and 0.4/0.5 wait on its output. 0.5 needs
only the header scan and the labels — no pixels, no GPU.

---

## Phase 1 — the label problem  (the priority)

`src/report_labeler.py` + evaluation against the gold studies.

1. Bilingual term table per language, human-reviewable, checked in.
2. Rule/lexicon layer: negation, hedging, severity thresholds, laterality.
3. Soft labels with a confidence/abstain channel.
4. Evaluation on the gold subset: per-finding AUC, agreement rate, and a
   confusion analysis of *where* report and image labels diverge — with
   interval estimates, because the gold set is tiny.
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

Self-contained, internet off, comfortably under 9 h with headroom for a larger
hidden test set. Pre-submit sanity checks: row count, column names and order,
no NaNs, probabilities in [0,1], IDs matching the sample submission exactly.
A second lightweight config is kept current for the efficiency track.

---

## Standing risks

| risk | early warning | response |
|---|---|---|
| **0.90 is out of reach because report labels cap it** | Phase 1 gold evaluation | say so plainly and early rather than at the deadline; redirect effort to whichever of labels / images is actually binding |
| Report text is unavailable for test studies | answered in 0.2 | labeler becomes training-time only; inference is image-only |
| Gold set is site-concentrated | answered in 0.2 | calibration claims get much weaker; report intervals, not point estimates |
| Header scan exceeds kernel time limits | 0.3 runtime | shard the scan across several kernels by study prefix |
| GPU budget exhausted mid-week | weekly GPU-hours in `EXPERIMENTS.md` | CPU-only work (labeler, cache) is always available as fallback |
| Submission kernel times out on the hidden test set | 0.1 file counts vs test size | measure per-study inference cost early; keep the efficiency config as the escape hatch |
