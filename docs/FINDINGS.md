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
58 with expert labels, 12 findings, macro-averaged ROC-AUC, the $77,000 prize
split, both deadlines — all confirmed. Four things it did not say, and each one
changes the build:

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
4. **The hidden test set is ~1,300 studies**, roughly 215,000 slices. Against the
   9-hour cap that is about **24 seconds per study**, end to end, including
   DICOM reading. That is the real design constraint on Phase 2, and it is a
   number rather than a worry. (§2.12)

And one that reframes the whole thing: **the host explicitly intends
report-derived labels.** The data-description says the reports are provided
"from which you may wish to derive the labels for the remaining studies"
(§3.17). The weak-supervision approach is the designed solution path, not a
clever workaround.

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
| 2.4 | **Metric is macro-averaged ROC-AUC over the 12 findings** | `VERIFIED` | Evaluation page, quoted: "Final Score = (1/12) Σ AUC_i … the macro-averaged AUC ROC". Retrieved with `kaggle competitions pages list --page-name evaluation --content`. |
| 2.5 | **Submission format** | `VERIFIED` | wide: one row per study, `StudyInstanceUID` + the 12 findings in the fixed order above; sample values are `0.5`, so probabilities |
| 2.6 | **Reports available at inference?** | `VERIFIED — NO` | data-description page, verbatim: "The `Report` field will not be provided at the testing stage." This confirms the inference from `test.csv` having a single column. |
| 2.7 | **Code competition limits** | `VERIFIED` | Code Requirements page: submissions via Notebooks, **CPU ≤ 9 h, GPU ≤ 9 h**, **internet access disabled**, output must be named **`submission.csv`**. Freely and publicly available external data and pre-trained models are allowed. |
| 2.8 | **Deadlines** | `VERIFIED` | entry `new_entrant_deadline` and `merger_deadline` both **2026-10-15 23:59**; final `deadline` **2026-10-22 23:59**. Competition opened 2026-08-05. |
| 2.9 | **Prize, and the efficiency track** | `VERIFIED` | Prizes page: main leaderboard $9,000 / $7,000 / $6,500 / $6,000 / $5,500 / then $5,000 for places 6–10 = $59,000; **Efficiency Track $7,000 / $6,000 / $5,000 = $18,000**. Total $77,000, matching the API `reward`. Winners must open-source, publish weights, and produce a short video. |
| 2.12 | **Hidden test set size** | `VERIFIED` | data-description page: "There are about 1300 studies in the test set." The 3-row `test.csv` is a stub replaced at scoring time. |
| 2.13 | **Efficiency score formula** | `VERIFIED (quoted)` | `Efficiency = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400`, minimised, where `Benchmark` is `sample_submission.csv`'s score and `maxAUC` the best private-LB score. 32,400 s is the 9-hour cap. Eligibility: must be a selected submission and must beat the `sample_submission.csv` benchmark on the private LB. *(As written the first term is negative, since Benchmark < maxAUC; recorded verbatim rather than "corrected", but worth watching the forum for an erratum.)* |
| 2.14 | **Timeline** | `VERIFIED` | Start 2026-07-30; entry and team-merger deadline 2026-10-15; final submission 2026-10-22; winners' requirement deadline 2026-11-05. All 23:59 UTC. *(The API's `enabled_date` says 2026-08-05, which disagrees with the Timeline page's start date; immaterial, but noted.)* |
| 2.10 | Submission limits | `VERIFIED` | `max_daily_submissions` = 5, `max_team_size` = 5 |
| 2.11 | Field size | `VERIFIED` | `team_count` = 1,866 as of this run |

---

## 3. Data scale and structure

| # | Claim | Tag | Evidence |
|---|---|---|---|
| 3.1 | ~570 GB, ~819k DICOM files | `UNVERIFIED — extrapolation disagrees` | Full listing still paginating (165,000 files / 107.5 GiB measured so far, all DICOM, mean 683 KiB each). Those 165k files cover 4,770 of the 24,371 training series at a mean 34.6 slices per series — and a **median of exactly 30, matching the host's stated median**, which is a good sign the partial sample is representative. Extrapolating over 24,371 train series plus ~7,150 test series gives **~1.09M DICOM files and ~0.69 TiB (~710 GB)**, noticeably above the brief's 570 GB / 819k. Treat as an extrapolation until the listing completes. Either way the conclusion is unchanged: far too large to pull locally. |
| 3.2 | **4,407 training studies** | `VERIFIED` | `train.csv` has 4,407 rows, 4,407 distinct `StudyInstanceUID` — the brief's figure exactly |
| 3.3 | **Exactly 58 studies carry expert labels** | `VERIFIED` | 58 rows have all 12 findings populated; 58 have at least one. There is no partially-labelled middle ground — a study is fully labelled or not at all. |
| 3.4 | **Every training study has a report** | `VERIFIED` | `Report` is non-null for all 4,407 rows. The brief implied some studies might lack one; none do. |
| 3.5 | **Reports are multilingual** | `VERIFIED` | detected over 4,000 sampled reports: en 39.3%, es 15.6%, tr 12.4%, el 7.4%, hr 7.3%, de 5.9%, bg 5.0%, nl 3.5%, fr 1.9%, bs 1.8%, la ~0%. That is 10 languages with real mass. Croatian/Bosnian are near-identical and the detector will confuse them; "la" (Latin) is near-certainly a misdetection of short anatomical text. So "~12 languages" is the right order of magnitude, and the exact count depends on how one splits Croatian/Bosnian/Serbian. |
| 3.6 | **Site / scanner / institution metadata in the CSVs** | `CONTRADICTED` | `train.csv` has 14 columns (UID, Report, 12 findings) and `train_series.csv` has 5 (UID, SeriesUID, Fluid_Sensitive, Fat_Suppression, Anatomical_Plane). **No site column exists.** |
| 3.7 | Scanner metadata present in both train and test headers | `VERIFIED (on a 7-file sample: 4 train, 3 test)` | the fingerprint fields below were populated in both splits |
| 3.8 | **Series-level metadata is provided by the host** | `VERIFIED` | `train_series.csv`: 24,371 series over 4,407 studies. `Anatomical_Plane` ∈ {Sagittal 9,864, Coronal 8,609, Axial 5,898}. |
| 3.9 | **Series per study** | `VERIFIED` | mean 5.53, median 5, min 3, max 14; p25 = 5, p75 = 6, p99 = 10. Every training study has at least 3 series. |
| 3.10 | **`Fluid_Sensitive` and `Fat_Suppression` are identical in train — but not guaranteed to be in test** | `VERIFIED, with a host warning` | Measured: identical on all 24,371 training rows (14,010 ones, 10,361 zeros; both off-diagonal cells are zero). The host's own data-description says they are "often correlated, as observed in the training set, **[but] not necessarily equivalent for every case**." So the training set offers no signal to learn any difference between them, while the test set may contain rows where they diverge. Keep both columns as separate inputs; do not collapse them into one flag, and do not assume the training correlation transfers. |
| 3.11 | Report length | `VERIFIED` | mean 1,098 chars, median 977, p5 205, p95 2,452, max 4,743 |
| 3.12 | **Test set size** | `VERIFIED` | Public stub: `test.csv` 3 rows, `test_series.csv` 15 rows. Real hidden test: **~1,300 studies** (data-description). At ~5.5 series and ~30 slices per series that is roughly 215,000 slices to process inside 9 hours — about 24 s per study, which is the budget the inference kernel must hit. |
| 3.15 | **Slices per series** | `VERIFIED (host-stated)` | data-description: "Series typically contain 20–45 slices (median 30), with a long tail out to a few hundred." |
| 3.16 | **Ground truth is per-study, twelve binary conditions, with definitions** | `VERIFIED` | data-description names each: `ACL` anterior cruciate ligament injury; `MCL` medial collateral ligament injury; `Medial`/`Lateral Meniscus` tear; `Medial`/`Lateral OA` tibiofemoral compartment osteoarthritis; `PF OA` patellofemoral osteoarthritis; `Effusion` joint effusion; `Synovitis` inflammation of the joint lining; `Baker's` cyst; `Contusion` bone bruise; `Fracture`. |
| 3.17 | **The host expects report-derived labels** | `VERIFIED` | data-description: "Only a small subset of training studies carry per-condition labels. We also provide the original text of the radiology report **from which you may wish to derive the labels for the remaining studies**." The weak-supervision framing is the intended solution path, not a workaround. |
| 3.13 | Train and test studies are disjoint | `VERIFIED` | zero `StudyInstanceUID` overlap between `test.csv` and `train.csv` |
| 3.14 | Data layout | `VERIFIED` | top-level `train_series/` and `test_series/` directories of `.dcm`, plus 5 root CSVs. (Not `train_images/`, as the brief assumed.) |
| 3.18 | **Kaggle mount path** | `VERIFIED` | The competition data mounts at **`/kaggle/input/competitions/rsna-knee-abnormality-detection`**, nested under `competitions/` — *not* at `/kaggle/input/<slug>`. The first header-scan kernel run failed on this and scanned zero series. Both kernels now discover the root by searching for files that must exist. |
| 3.19 | **Cost of reaching the DICOM data** | `VERIFIED (measured on Kaggle)` | Directory traversal plus one header read per series costs **0.059 s per study** (3 studies, 15 series, 557 files). Extrapolated to the ~1,300-study hidden test that is **77 seconds, or 0.2% of the 9-hour cap**. So file access is effectively free and the entire ~24 s/study budget is available for pixel decoding and inference. Note this did **not** decode pixels — the real constraint remains the ~215,000 slices. |

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

## 9. Leakage audit — Phase 0 step 5, `VERIFIED`

`eda/leakage_audit.py`. A gradient-boosted model is given **only scanner
metadata** — manufacturer, model, field strength, coil, slice counts, pixel
spacing, image dimensions, sex, laterality. No pixels, no text. Such a model
cannot know whether a knee has a torn ACL, so anything above 0.5 is site
memorisation by construction. Targets are the Phase 1 report-derived labels,
because they cover all 4,407 studies; 58 gold studies cannot measure a fold
effect.

| finding | random K-fold | scanner-grouped | inflation |
|---|---:|---:|---:|
| MCL | 0.802 | 0.695 | **0.107** |
| Fracture | 0.692 | 0.592 | 0.100 |
| Effusion | 0.755 | 0.660 | 0.095 |
| Baker's | 0.694 | 0.600 | 0.094 |
| Medial OA | 0.788 | 0.698 | 0.091 |
| PF OA | 0.749 | 0.659 | 0.090 |
| ACL | 0.771 | 0.682 | 0.089 |
| Contusion | 0.681 | 0.593 | 0.088 |
| Lateral OA | 0.842 | 0.759 | 0.083 |
| Synovitis | 0.802 | 0.728 | 0.074 |
| Lateral Meniscus | 0.740 | 0.666 | 0.074 |
| Medial Meniscus | 0.702 | 0.643 | 0.059 |
| **MACRO** | **0.752** | **0.664** | **0.087** |

**Random K-fold inflates macro AUC by 0.087 on metadata alone**, and a real
model would enjoy that on top of its own signal. The brief's 0.05–0.14 estimate
was right. Grouped folds are settled, not a preference.

### The more interesting number is 0.664

Scanner metadata still scores **0.664 macro AUC under grouped K-fold**, where
memorisation is impossible because every validation scanner is unseen. Two
things are mixed in there, and they are worth separating in Phase 2:

- **Genuine clinical signal.** Protocol choice reflects suspicion — a
  radiologist who orders extra sequences is looking for something. Slice counts
  and sequence mix legitimately carry information.
- **Population effects that generalise.** A 3 T academic centre and a 1.5 T
  community clinic see different patients, and that difference transfers to
  unseen scanners of the same class.

**Caveat, and it matters:** these targets are report-derived, not expert labels.
Some of the 0.664 is shared site convention — a site whose radiologists write
verbosely produces both more positive report labels *and* a distinctive scanner
signature. Against true expert labels the figure would likely be lower. It is a
baseline to beat, not a result to celebrate.

### Grouping structure

178 scanner fingerprints over 4,407 studies: median group 8 studies, mean 24.8,
largest 246, 42 singletons, and the five largest groups hold 21% of the data.
That is workable for 5-fold `GroupKFold`.

### The precision trap that would have silenced this whole audit

The first fingerprint design used `ImagingFrequency` at full precision and
produced **8,618 fingerprints for 4,410 studies, 5,633 of them singletons** — a
key nearly unique per study, on which "grouped" K-fold would have been a random
K-fold in disguise, reporting no inflation and hiding the leak completely.

The cause: Larmor frequency drifts between sessions on the same magnet. The
Philips Ingenia 3 T scanners here produce **739 distinct raw values across 2,480
series**. Rounding to 2 decimals collapses that cluster to 4 values and yields
the 178 usable groups above. `src/folds.py` documents this, because it is the
kind of detail that silently invalidates everything downstream.

---

## 8. Where the leaderboard actually sits — `VERIFIED`

Read via `competition_leaderboard_view` on 2026-08-18, top 200 of 1,866 teams.

| score | approximate rank | percentile |
|---:|---:|---|
| 0.951 | 1 | top of the board |
| 0.940 | ~11 | top 6% of the read sample |
| 0.930 | ~40 | top 20% |
| 0.920 | ~94 | top 47% |
| ≤ 0.917 | outside the top 200 | — |

**This inverts the framing of the 0.90 target.** 0.90 is not an ambitious score
in this competition — it does not reach the top 200 teams. The bar is roughly:

- **~0.941** for a top-ten finish, which is where the main-leaderboard prizes are
- **~0.930** for a top-20% finish
- **~0.917** merely to enter the top 200

Two consequences that matter more than the number itself:

1. **The weak-supervision problem is not an untapped edge.** Two hundred teams
   are already above 0.917, which is not reachable without deriving usable labels
   from the reports. Everyone has done it. The strategy's bet — that the label
   pipeline is where a small-compute entrant wins — has to be downgraded from
   "under-explored" to "table stakes, and the edge is in doing it better".
2. **The efficiency track is the more reachable prize.** It pays $18,000, and
   eligibility requires only beating the `sample_submission.csv` benchmark on the
   private leaderboard (§2.13). Runtime enters the score divided by the 9-hour
   cap, so a fast, decent kernel can place there without a top-ten AUC. For an
   entrant on a free Kaggle account this is a better expected return per GPU-hour
   than chasing 0.941.

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
| 5.1 | Ground truth is image-derived (2 MSK radiologists + adjudicator, severity-thresholded) | `UNVERIFIED` | the data-description page does **not** describe the annotation process. Not stated anywhere I can reach through the API; would need the forum or the RSNA challenge page. |
| 5.2 | Report-derived labels agree ~82% with image labels | `UNVERIFIED` | measurable in Phase 1 against the 58, with the interval from §4 |
| 5.3 | Random K-fold inflates AUC by 0.05–0.14 | **`VERIFIED` — 0.087** | Measured (§9). Squarely inside the claimed range. |
| 5.4 | Public baseline ~0.809 | `CONTRADICTED` | Read from the public leaderboard: **top score 0.9510**, and the **top 200 teams all score ≥ 0.9170** (median of that group 0.9200). The 0.809 figure is stale by a wide margin — it may have been an early baseline notebook. See §8. |

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
10. ~~How large is the hidden test set?~~ **Answered: ~1,300 studies** (§2.12),
    which sets the inference budget at roughly 24 s per study.
11. **How were the 58 gold labels produced?** The host does not say. The
    "two radiologists plus adjudicator, severity-thresholded" story is still
    second-hand, and it matters because it predicts *which way* report-derived
    labels are biased.

---

## 7. Sources

| source | status |
|---|---|
| Kaggle API, authenticated | primary source for §1–§2 |
| `train.csv`, `train_series.csv`, `sample_submission.csv`, `test.csv`, `test_series.csv` | primary source for §2.5, §3, §4 |
| 7 sampled DICOM files (4 train, 3 test), headers only | primary source for §5 |
| Competition pages via `competition_list_pages` | primary source for the metric, code requirements, prizes, timeline, efficiency score and data description. **This is the API route to the pages the web UI renders client-side** — `kaggle competitions pages list --page-name <name> --content`. |
| Kickoff brief | second-hand; scored above — mostly right, wrong on §3.6 and silent on §2.6 |
