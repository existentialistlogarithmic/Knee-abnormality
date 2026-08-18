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
2. **There is no site column anywhere — and the DICOM headers are
   de-identified too**, so `InstitutionName`, `StationName`, `DeviceSerialNumber`
   and all dates are gone. A true site label does not exist in this dataset. What
   does exist is a strong *scanner fingerprint*, and that becomes the grouping
   key. (§3.6, §5)
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
| 3.6 | **Site / scanner / institution metadata in the CSVs** | `CONTRADICTED` | `train.csv` has 14 columns (UID, Report, 12 findings) and `train_series.csv` has 5 (UID, SeriesUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane). **No site column exists.** |
| 3.7 | Scanner metadata present in both train and test headers | `VERIFIED (on a 7-file sample: 4 train, 3 test)` | the fingerprint fields below were populated in both splits |
| 3.8 | **Series-level metadata is provided by the host** | `VERIFIED` | `train_series.csv`: 24,371 series over 4,407 studies. `Anatomical_Plane` ∈ {Sagittal 9,864, Coronal 8,609, Axial 5,898}. |
| 3.9 | **Series per study** | `VERIFIED` | mean 5.53, median 5, min 3, max 14; p25 = 5, p75 = 6, p99 = 10. Every training study has at least 3 series. |
| 3.10 | **`Fluid_Sensitive` and `Fat_Suppression` are the same column** | `VERIFIED` | identical on all 24,371 rows (14,010 ones, 10,361 zeros; off-diagonal counts are zero). Two column names, one bit of information. Treat as a single flag and do not build a feature that assumes they differ. |
| 3.11 | Report length | `VERIFIED` | mean 1,098 chars, median 977, p5 205, p95 2,452, max 4,743 |
| 3.12 | Public test set size | `VERIFIED` | `test.csv` has 3 rows, `test_series.csv` 15 rows (5 series per study). This is a placeholder; the rescored hidden test is larger and its size is `UNVERIFIED` — which is exactly why the inference kernel needs runtime headroom. |
| 3.13 | Train and test studies are disjoint | `VERIFIED` | zero `StudyInstanceUID` overlap between `test.csv` and `train.csv` |
| 3.14 | Data layout | `VERIFIED` | top-level `train_series/` and `test_series/` directories of `.dcm`, plus 5 root CSVs. (Not `train_images/`, as the brief assumed.) |

---

## 5. DICOM headers — sampled, 7 files from 7 studies (4 train, 3 test)

69 distinct tags across the sample. Read with `stop_before_pixels=True`; no pixel
data was decoded.

### 5.1 What de-identification removed — `VERIFIED`

Absent from **all** sampled files:

`InstitutionName`, `InstitutionAddress`, `InstitutionalDepartmentName`,
`StationName`, `PerformedStationName`, `DeviceSerialNumber`, `ProtocolName`,
`StudyDate`, `SeriesDate`, `PatientAge`, `PatientSize`, `PatientWeight`.

**There is no site identifier in this dataset, in any file, in any form.** Any
plan that assumed a site label — including a straightforward site-grouped
K-fold — has to be rebuilt around a proxy.

### 5.2 The scanner fingerprint — `VERIFIED`

These survive and are populated:

| tag | present | example values from the sample |
|---|---:|---|
| `Manufacturer` | 7/7 | SIEMENS, Siemens Healthineers, GE MEDICAL SYSTEMS, Philips Healthcare, TOSHIBA |
| `ManufacturerModelName` | 7/7 | MAGNETOM Vida, MAGNETOM Lumina, MAGNETOM Avanto fit, Aera, SIGNA Artist, Ingenia, Vantage |
| `SoftwareVersions` | 6/7 | syngo MR XA60, syngo MR E11, 5.6.1 |
| `MagneticFieldStrength` | 6/7 | 1.5, 3 |
| `ImagingFrequency` | 6/7 | 63.881601, 63.870660, 63.685261, 63.648174, 123.255723, 123.238133 |
| `TransmitCoilName` | 5/7 | TxRx_Knee_18, TxRx_15Ch_Knee, 16Knee |

**`ImagingFrequency` is the load-bearing field.** It is the Larmor frequency in
MHz, set by the magnet's exact field, and it separates two nominally identical
1.5 T scanners at the fourth decimal — 63.881601 vs 63.870660 vs 63.685261 vs
63.648174 are four different magnets. Concatenated with manufacturer, model,
software version and coil, it behaves like a hardware serial number.

Seven random files produced **seven distinct fingerprints and five
manufacturers**, which independently corroborates the "many sites, many
scanners" claim.

The fingerprint degrades gracefully but not uniformly: the TOSHIBA Vantage file
had no software version, field strength, frequency or coil name, so for that
vendor the key collapses to manufacturer + model. How much of the dataset sits
in that degraded state is `UNVERIFIED` until the full scan runs.

### 5.3 Other useful survivors — `VERIFIED`

| tag | present | note |
|---|---:|---|
| `Laterality` | 6/7 | values `L`, `R`, and one empty. Needed for mirroring right knees; the empty case is real and must be handled. `SeriesDescription` sometimes encodes it too (e.g. `LT_t2_tse_fs_cor_obl_ACL`). |
| `SeriesDescription` | 7/7 | free text, informative, vendor-specific — e.g. `LT_t2_tse_fs_cor_obl_ACL`, `pd_tse_tra_d`, `SG PD FatSat` |
| `PatientID` | 7/7 | pseudonymised (e.g. `fd7379f0-da7`). **Whether one pseudonym spans several studies is `UNVERIFIED`** — if it does, folds must group on patient as well as scanner, or the same knee appears on both sides of a split. The scan checks this. |
| `PixelSpacing`, `SliceThickness`, `SpacingBetweenSlices` | 7/7, 7/7, 6/7 | needed for resampling to fixed mm-per-pixel |
| `ImageOrientationPatient` | 7/7 | lets the scan cross-check the host's `Anatomical_Plane` |
| `PatientSex` | 6/7 | |
| `BodyPartExamined` | 7/7 | |
| `EchoTime`, `RepetitionTime`, `EchoTrainLength`, `ScanningSequence`, `SequenceVariant` | 6/7 | sequence identity beyond the host's single fluid-sensitive bit |

### 5.4 Scan cost — `VERIFIED (extrapolated)`

The kernel read 7 series in 0.04 s (0.006 s/series) on warm local files. At that
rate 24,371 training series is a few minutes of CPU, though a cold Kaggle mount
over ~800k files will be slower. The kernel shards by study and stops cleanly on
a wall-clock budget, so a slow mount costs a re-run, not a lost session.

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

## 6. Label-ceiling probe — is there signal in the reports at all?

`eda/label_ceiling_probe.py`, run on the 58 gold studies. The matcher is
deliberately crude: case-folded substring matching against a small multilingual
keyword list, **with no negation, hedging, severity or laterality handling**.
Those are what Phase 1 builds, so this is a **floor**, not a forecast.

**Macro balanced accuracy: 0.601** (`VERIFIED`, n = 58).

| finding | positives | mentions | agreement | 95% CI | sens | spec | bal. acc |
|---|---:|---:|---:|---|---:|---:|---:|
| Baker's | 12 | 18 | 0.759 | [0.63, 0.85] | 0.67 | 0.78 | 0.725 |
| MCL | 9 | 27 | 0.621 | [0.49, 0.73] | 0.78 | 0.59 | 0.685 |
| Contusion | 19 | 25 | 0.655 | [0.53, 0.76] | 0.63 | 0.67 | 0.649 |
| Lateral Meniscus | 23 | 35 | 0.621 | [0.49, 0.73] | 0.78 | 0.51 | 0.648 |
| Synovitis | 27 | 15 | 0.655 | [0.53, 0.76] | 0.41 | 0.87 | 0.639 |
| Medial OA | 15 | 13 | 0.724 | [0.60, 0.82] | 0.40 | 0.84 | 0.619 |
| Fracture | 18 | 17 | 0.672 | [0.54, 0.78] | 0.44 | 0.78 | 0.610 |
| Medial Meniscus | 26 | 34 | 0.586 | [0.46, 0.70] | 0.69 | 0.50 | 0.596 |
| ACL | 24 | 37 | 0.569 | [0.44, 0.69] | 0.75 | 0.44 | 0.596 |
| Lateral OA | 11 | 11 | 0.724 | [0.60, 0.82] | 0.27 | 0.83 | 0.551 |
| Effusion | 35 | 40 | 0.500 | [0.38, 0.62] | 0.66 | 0.26 | 0.459 |
| PF OA | 21 | 21 | 0.483 | [0.36, 0.61] | 0.29 | 0.59 | 0.440 |

### What the shape of these numbers says

The failure mode is **specificity, not sensitivity**, and it is concentrated
exactly where you would expect. `ACL` produces 37 mentions for 24 true
positives: sensitivity 0.75, specificity 0.44. `Effusion` produces 40 mentions
for 35 positives at specificity 0.26. Radiologists mention the ACL in almost
every knee report — *"ACL intact"*, *"no ACL tear"* — and a matcher with no
negation counts every one of those as a positive.

So the dominant error is a solved problem in NLP terms, and Phase 1's ordering
(negation and hedging first, severity thresholds second) is aimed at the right
target. The findings that already score best are the ones whose keywords are
rare and therefore rarely negated: `Baker's` 0.725, `MCL` 0.685.

**This confirms the premise the whole strategy rests on**: the reports carry
real finding-level signal, and the gap between 0.601 and something useful is
mostly linguistic work rather than missing information.

### The gold subset is more English than the corpus — `VERIFIED`

| language | gold n | gold % | train % |
|---|---:|---:|---:|
| en | 28 | 48.3% | 39.3% |
| es | 10 | 17.2% | 15.6% |
| tr | 6 | 10.3% | 12.4% |
| hr | 4 | 6.9% | 7.3% |
| el | 3 | 5.2% | 7.4% |
| bg | 3 | 5.2% | 5.0% |
| nl | 2 | 3.4% | 3.5% |
| de | 2 | 3.4% | 5.9% |

English is 48% of the gold subset against 39% of the corpus, and French, Bosnian
and Latin do not appear in the gold subset at all. **Every evaluation on these
58 studies therefore flatters an English-led labeler**, and the true performance
on the 4,349 training reports will be somewhat worse than the gold number
suggests. Worth stating in any write-up of Phase 1 results rather than
discovering later.

---

## 5. Modelling-relevant beliefs — still unverified

| # | Claim | Tag | Note |
|---|---|---|---|
| 5.1 | Ground truth is image-derived (2 MSK radiologists + adjudicator) | `UNVERIFIED` | needs the Data page description |
| 5.2 | Report-derived labels agree ~82% with image labels | `UNVERIFIED` | measurable in Phase 1 against the 58, with the interval from §4 |
| 5.3 | Random K-fold inflates AUC by 0.05–0.14 | `UNVERIFIED` | blocked on the header scan. Note the audit now compares random K-fold against **scanner-fingerprint**-grouped K-fold, since no true site label exists (§5.1). |
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
5. ~~Where does site/scanner grouping come from?~~ **Answered:** a scanner
   fingerprint built from the headers (§5.2). A true site label does not exist.
6. ~~Is laterality recoverable?~~ **Answered:** `Laterality` is in the headers,
   populated in 6 of 7 sampled files, with `SeriesDescription` as a fallback.
7. **Does one `PatientID` span several studies?** If so, folds must group on
   patient too. The header scan answers this.
8. **What fraction of series have a degraded fingerprint** (missing frequency
   or software version, as the TOSHIBA file did)?
9. **Are the 58 gold studies scanner-diverse?** If they cluster on one or two
   scanners, calibration claims weaken sharply.
10. **How large is the hidden test set?** Drives the inference-kernel budget.

---

## 7. Sources

| source | status |
|---|---|
| Kaggle API, authenticated | primary source for §1–§2 |
| `train.csv`, `train_series.csv`, `sample_submission.csv`, `test.csv`, `test_series.csv` | primary source for §2.5, §3, §4 |
| 7 sampled DICOM files (4 train, 3 test), headers only | primary source for §5 |
| Competition overview page | client-rendered; only title/meta readable without a browser session |
| Kickoff brief | second-hand; scored above — mostly right, wrong on §3.6 and silent on §2.6 |
