# FINDINGS — Phase 0 verification ledger

**Rule for this file: no claim appears here without a tag.**

| tag | meaning |
|---|---|
| `VERIFIED` | read directly out of a competition file, the Kaggle API, or the competition's own pages, with the source named |
| `UNVERIFIED` | believed, second-hand, or inferred — **not usable as a basis for design decisions** |
| `CONTRADICTED` | checked and found false |

Last updated: 2026-08-18, after the first authenticated Phase 0 run.

---

## 0. Headline

The kickoff brief was **substantially accurate**. 4,407 training studies, exactly
58 with expert labels, 12 findings, multilingual reports — all confirmed to the
digit. Three things it did not say, and each one changes the build:

1. **Reports do not exist at inference time.** `test.csv` has exactly one column,
   `StudyInstanceUID`. The report labeler is a training-time device only; the
   submission kernel is image-only. (§2.6)
2. **There is no site, scanner, or institution column anywhere in the CSVs.**
   Site-grouped folds and the leakage audit are now blocked on the DICOM header
   scan rather than being a CSV `groupby`. (§3.6)
3. **The host already provides series-level plane and sequence metadata**, so the
   header scan does not have to derive orientation from
   `ImageOrientationPatient` for series selection. (§3.8)

---

## 1. Environment and access

| # | Claim | Tag | Evidence |
|---|---|---|---|
| 1.1 | The competition exists | `VERIFIED` | API `ref` = `https://www.kaggle.com/competitions/rsna-knee-abnormality-detection` |
| 1.2 | Kaggle reachable | `VERIFIED` | authenticated API calls succeed |
| 1.3 | `kaggle` CLI installed | `VERIFIED` | `Kaggle CLI 2.2.4` |
| 1.4 | **Kaggle API auth works** | `VERIFIED` | `KAGGLE_API_TOKEN` authenticates; all calls below made with it |
| 1.5 | **Rules accepted by this account** | `VERIFIED` | API `user_has_entered` = `True` |
| 1.6 | Credential file locations | `VERIFIED` | CLI 2.2.4 accepts `kaggle.json`, `~/.kaggle/access_token`, `KAGGLE_API_TOKEN`, or `kaggle auth login` |
| 1.7 | GitHub repo is private | `VERIFIED` | GitHub API `"private": true` |
| 1.8 | Offline language ID works | `VERIFIED` | `py3langid`, no network; used for §3.5 |
| 1.9 | Phase 0 runs in GitHub Actions | `VERIFIED (mechanism)` | `.github/workflows/phase0.yml`; the run reported here was executed directly, secret still to be added for CI runs |
| 1.10 | Host / organisation | `VERIFIED` | API `organization_name` = `Radiological Society of North America`, `category` = `Research` |

---

## 2. Task, metric, submission format

| # | Claim | Tag | Evidence |
|---|---|---|---|
| 2.1 | **12 binary findings per study** | `VERIFIED` | `sample_submission.csv` has 13 columns: `StudyInstanceUID` + 12 findings |
| 2.2 | **The 12 target names** | `VERIFIED` | `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture` — column order identical in `train.csv` and `sample_submission.csv` |
| 2.3 | Positive rates | `VERIFIED (on the 58 gold studies)` | see §4 |
| 2.4 | Metric is ROC-AUC | `VERIFIED (partially)` | API `evaluation_metric` = `Roc Auc Score`. **The averaging scheme is not stated by the API** — whether it is macro over the 12 findings, or something else, still needs the Evaluation page. |
| 2.5 | **Submission format** | `VERIFIED` | wide: one row per study, `StudyInstanceUID` + the 12 findings in the fixed order above; sample values are `0.5`, so probabilities |
| 2.6 | **Reports available at inference?** | `VERIFIED — NO` | `test.csv` has exactly one column, `StudyInstanceUID`. `train.csv` carries `Report`; the test table does not. Caveat: the public test stub is 3 rows, so this is the *schema* rather than the rescored file, but a hidden test that added a column would break every published notebook. |
| 2.7 | Code competition | `VERIFIED` | API `is_kernels_submissions_only` = `True`. The 9-hour limit and internet-off rule are `UNVERIFIED` — they come from the Code Requirements page, not the API. |
| 2.8 | **Deadlines** | `VERIFIED` | entry `new_entrant_deadline` and `merger_deadline` both **2026-10-15 23:59**; final `deadline` **2026-10-22 23:59**. Competition opened 2026-08-05. |
| 2.9 | **Prize** | `VERIFIED` | API `reward` = `77,000 Usd`. The separate efficiency track is `UNVERIFIED` — not an API field. |
| 2.10 | Submission limits | `VERIFIED` | `max_daily_submissions` = 5, `max_team_size` = 5 |
| 2.11 | Field size | `VERIFIED` | `team_count` = 1,866 as of this run |

---

## 3. Data scale and structure

| # | Claim | Tag | Evidence |
|---|---|---|---|
| 3.1 | ~570 GB, ~819k DICOM files | `UNVERIFIED` | full listing still running; the first 600 files were 601.5 MB, and the listing rate-limits (HTTP 429) after ~25k files, so it is checkpointed and resumed |
| 3.2 | **4,407 training studies** | `VERIFIED` | `train.csv` has 4,407 rows, 4,407 distinct `StudyInstanceUID` — the brief's figure exactly |
| 3.3 | **Exactly 58 studies carry expert labels** | `VERIFIED` | 58 rows have all 12 findings populated; 58 have at least one. There is no partially-labelled middle ground — a study is fully labelled or not at all. |
| 3.4 | **Every training study has a report** | `VERIFIED` | `Report` is non-null for all 4,407 rows. The brief implied some studies might lack one; none do. |
| 3.5 | **Reports are multilingual** | `VERIFIED` | detected over 4,000 sampled reports: en 39.3%, es 15.6%, tr 12.4%, el 7.4%, hr 7.3%, de 5.9%, bg 5.0%, nl 3.5%, fr 1.9%, bs 1.8%, la ~0%. That is 10 languages with real mass. Croatian/Bosnian are near-identical and the detector will confuse them; "la" (Latin) is near-certainly a misdetection of short anatomical text. So "~12 languages" is the right order of magnitude, and the exact count depends on how one splits Croatian/Bosnian/Serbian. |
| 3.6 | **Site / scanner / institution metadata in the CSVs** | `CONTRADICTED` | `train.csv` has 14 columns (UID, Report, 12 findings) and `train_series.csv` has 5 (UID, SeriesUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane). **No site column exists.** Grouped folds must come from DICOM headers. |
| 3.7 | Site metadata in both train and test | `UNVERIFIED` | needs the header scan on both splits |
| 3.8 | **Series-level metadata is provided by the host** | `VERIFIED` | `train_series.csv`: 24,371 series over 4,407 studies. `Anatomical_Plane` ∈ {Sagittal 9,864, Coronal 8,609, Axial 5,898}. |
| 3.9 | **Series per study** | `VERIFIED` | mean 5.53, median 5, min 3, max 14; p25 = 5, p75 = 6, p99 = 10. Every training study has at least 3 series. |
| 3.10 | **`Fluid_Sensitive` and `Fat_Suppression` are the same column** | `VERIFIED` | identical on all 24,371 rows (14,010 ones, 10,361 zeros; off-diagonal counts are zero). Two column names, one bit of information. Treat as a single flag and do not build a feature that assumes they differ. |
| 3.11 | Report length | `VERIFIED` | mean 1,098 chars, median 977, p5 205, p95 2,452, max 4,743 |
| 3.12 | Public test set size | `VERIFIED` | `test.csv` has 3 rows, `test_series.csv` 15 rows (5 series per study). This is a placeholder; the rescored hidden test is larger and its size is `UNVERIFIED` — which is exactly why the inference kernel needs runtime headroom. |
| 3.13 | Train and test studies are disjoint | `VERIFIED` | zero `StudyInstanceUID` overlap between `test.csv` and `train.csv` |
| 3.14 | Data layout | `VERIFIED` | top-level `train_series/` and `test_series/` directories of `.dcm`, plus 5 root CSVs. (Not `train_images/`, as the brief assumed.) |

---

## 4. The gold subset (n = 58)

Positive counts out of 58, from `train.csv`. **These are counts, not prevalences
of the disease** — 58 studies is a small, probably deliberately balanced sample,
so these rates should not be read as the base rates in the hidden test set.

| finding | positives / 58 | rate |
|---|---:|---:|
| Effusion | 35 | 0.603 |
| Synovitis | 27 | 0.466 |
| Medial Meniscus | 26 | 0.448 |
| ACL | 24 | 0.414 |
| Lateral Meniscus | 23 | 0.397 |
| PF OA | 21 | 0.362 |
| Contusion | 19 | 0.328 |
| Fracture | 18 | 0.310 |
| Medial OA | 15 | 0.259 |
| Baker's | 12 | 0.207 |
| Lateral OA | 11 | 0.190 |
| MCL | 9 | 0.155 |

**The statistical reality of n = 58.** A per-finding AUC estimated here has a
95% interval roughly ±0.13 wide at best, and for `MCL` (9 positives) closer to
±0.20. This subset can rank a labeler as clearly-good or clearly-broken. It
cannot distinguish 0.86 from 0.90, and no amount of careful methodology changes
that. Every number computed on it gets an interval printed next to it.

---

## 5. Modelling-relevant beliefs — still unverified

| # | Claim | Tag | Note |
|---|---|---|---|
| 5.1 | Ground truth is image-derived (2 MSK radiologists + adjudicator) | `UNVERIFIED` | needs the Data page description |
| 5.2 | Report-derived labels agree ~82% with image labels | `UNVERIFIED` | measurable in Phase 1 against the 58, with the interval from §4 |
| 5.3 | Random K-fold inflates AUC by 0.05–0.14 | `UNVERIFIED` | **now blocked on the header scan**, since no site column exists (§3.6) |
| 5.4 | Public baseline ~0.809 | `UNVERIFIED` | needs the leaderboard |

---

## 6. Open questions, updated

1. ~~Are reports available at test time?~~ **Answered: no** (§2.6). The labeler
   is training-time only.
2. ~~How many studies lack a report?~~ **Answered: none** (§3.4).
3. ~~Unit of prediction?~~ **Answered:** one row per study, 12 columns, wide
   (§2.5).
4. **Is the ROC-AUC macro-averaged over the 12 findings?** The API says only
   "Roc Auc Score". Needs the Evaluation page.
5. **Where does site/scanner grouping come from?** Only the DICOM headers now.
   Until the scan runs, every fold scheme is unvalidated.
6. **Is laterality recoverable?** No column in any CSV; must come from headers.
7. **Are the 58 gold studies site-diverse?** Unanswerable until sites are known.
   If they cluster in one or two sites, calibration claims weaken sharply.
8. **How large is the hidden test set?** Drives the inference-kernel budget.

---

## 7. Sources

| source | status |
|---|---|
| Kaggle API, authenticated | primary source for §1–§2 |
| `train.csv`, `train_series.csv`, `sample_submission.csv`, `test.csv`, `test_series.csv` | primary source for §2.5, §3, §4 |
| Competition overview page | client-rendered; only title/meta readable without a browser session |
| Kickoff brief | second-hand; scored above — mostly right, wrong on §3.6 and silent on §2.6 |
