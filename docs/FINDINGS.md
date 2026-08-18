# FINDINGS — Phase 0 verification ledger

**Rule for this file: no claim appears here without a tag.**

| tag | meaning |
|---|---|
| `VERIFIED` | read directly out of a competition file, the Kaggle API, or the competition's own pages, with the source named |
| `UNVERIFIED` | believed, second-hand, or inferred — **not usable as a basis for design decisions** |
| `CONTRADICTED` | checked and found false |

Last updated: 2026-08-18.

---

## 0. Status of this ledger

**Nothing about the competition data is verified yet.** The verification run has
not been executed against a credentialled Kaggle account. See §1 for exactly
what blocks it.

---

## 1. Environment and access

| # | Claim | Tag | Evidence |
|---|---|---|---|
| 1.1 | The competition `rsna-knee-abnormality-detection` exists on Kaggle | `VERIFIED` | `GET https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview` → HTTP 200, `<title>RSNA Knee Abnormality Detection \| Kaggle</title>`, meta description "Create a model that can detect knee abnormalities based on multimodal imaging data" |
| 1.2 | Kaggle is reachable from this working environment | `VERIFIED` | `GET /api/v1/competitions/list` → HTTP 401 (reached the API, rejected for lack of credentials — not a network failure) |
| 1.3 | The `kaggle` CLI is installed and runnable | `VERIFIED` | `kaggle --version` → `Kaggle CLI 2.2.4` |
| 1.4 | **Kaggle API auth works for this account** | `UNVERIFIED` | **BLOCKED**: no credentials exist in this environment (`~/.kaggle/` absent, no `KAGGLE_API_TOKEN`). See §1a. |
| 1.5 | The competition rules have been accepted by this account | `UNVERIFIED` | Requires 1.4. The API field `user_has_entered` answers this exactly; `eda/phase0_01_auth_and_files.py` prints it. |
| 1.6 | Credentials go in `~/.kaggle/kaggle.json` | `VERIFIED (with a caveat)` | CLI 2.2.4 still reads `kaggle.json` (`kaggle_api_extended.py:924`), but its own auth help now advertises `kaggle auth login` (OAuth), `KAGGLE_API_TOKEN`, or `~/.kaggle/access_token`. Any of the four works. |
| 1.7 | The GitHub repository is **private** | `VERIFIED` | GitHub API: `"private": true`, `"visibility": "private"`. This is what makes it acceptable for CI to retain the identifier-bearing file inventory as a build artifact; the workflow refuses that upload if the repo ever becomes public. |
| 1.8 | Language identification works offline, with no hosted API | `VERIFIED` | `py3langid` correctly labelled seven test phrases (en, de, fr, ru, ja, es, tr) in 0.24 s including import. Matters twice: Rule 4.b forbids sending report text to a hosted LLM, and the submission kernel has internet off. |
| 1.9 | Phase 0 can run on GitHub Actions using a repository secret | `VERIFIED (mechanism)` `UNVERIFIED (run)` | `.github/workflows/phase0.yml` is in place and its YAML parses; it has not yet executed against Kaggle because the `KAGGLE_API_TOKEN` secret is not set. |

### 1a. What is blocking Phase 0

One thing: the `KAGGLE_API_TOKEN` repository secret is not set.

Everything else is built and tested. Add the secret at
**Settings → Secrets and variables → Actions → New repository secret**, using a
token from <https://www.kaggle.com/settings/api>, then run the **phase0-verify**
workflow from the Actions tab. Steps 1 and 2 both run there; nothing needs to
touch a local machine.

The run reports `user_has_entered`, so whether the rules have been accepted is
answered by the run rather than assumed.

---

## 2. Task, metric, submission format — ALL UNVERIFIED

| # | Claim (second-hand, from the kickoff brief) | Tag | How it gets verified |
|---|---|---|---|
| 2.1 | 12 binary findings per knee MRI study | `UNVERIFIED` | column count of `sample_submission` / the train label file |
| 2.2 | The 12 target column names | `UNVERIFIED` | header of the train label file |
| 2.3 | Positive rate per finding | `UNVERIFIED` | computed from the train label file |
| 2.4 | Metric is macro ROC-AUC over the 12 findings | `UNVERIFIED` | API field `evaluation_metric`, cross-checked against the competition's Evaluation page |
| 2.5 | Submission format (row count, ID column, wide vs long) | `UNVERIFIED` | `sample_submission.csv` itself — the only authority |
| 2.6 | Multimodal: DICOM images + radiology report text | `UNVERIFIED` | file inventory from step 1 |
| 2.7 | Code competition, internet off, ≤ 9 h runtime | `UNVERIFIED` | API field `is_kernels_submissions_only`, plus the competition's Rules/Code Requirements page |
| 2.8 | Entry/merge deadline 2026-10-15, final 2026-10-22 | `UNVERIFIED` | API fields `new_entrant_deadline`, `merger_deadline`, `deadline` |
| 2.9 | Prize pool $77,000 with a separate efficiency track | `UNVERIFIED` | API field `reward`, plus the Overview/Prizes page |

---

## 3. Data scale and structure — ALL UNVERIFIED

| # | Claim | Tag | How it gets verified |
|---|---|---|---|
| 3.1 | ~570 GB total, ~819k DICOM files | `UNVERIFIED` | sum of `total_bytes` and the file count from step 1 |
| 3.2 | ~4,407 training studies | `UNVERIFIED` | distinct study IDs in the train label file |
| 3.3 | Only ~58 studies carry expert image-derived labels | `UNVERIFIED` | count of fully-populated label rows vs report-only rows |
| 3.4 | The rest have only a free-text radiology report | `UNVERIFIED` | join between the label file and the report file |
| 3.5 | Reports span ~12 languages, 16–19 sites, five continents | `UNVERIFIED` | language ID over the report text; distinct site/institution values |
| 3.6 | Site / scanner / institution metadata exists | `UNVERIFIED` | DICOM header scan (Phase 0 step 3) — `InstitutionName`, `Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength`, `DeviceSerialNumber` |
| 3.7 | That metadata appears in **both** train and test | `UNVERIFIED` | header scan over both splits. **This one is load-bearing**: if site metadata is stripped from test, site-grouped CV is still right for honest validation, but any site-conditioned *feature* is unusable at inference. |

---

## 4. Modelling-relevant beliefs — ALL UNVERIFIED

| # | Claim | Tag | How it gets verified |
|---|---|---|---|
| 4.1 | Ground truth is image-derived (2 MSK radiologists + adjudicator, severity-thresholded, "on the fence" → negative) | `UNVERIFIED` | the competition's Data page description |
| 4.2 | Report-derived labels agree with image labels only ~82%, systematically | `UNVERIFIED` | measurable only against the gold subset, in Phase 1 — and with ~58 studies, any per-finding agreement estimate carries a wide interval that must be reported alongside it |
| 4.3 | Random K-fold inflates AUC by ~0.05–0.14 vs site-grouped K-fold | `UNVERIFIED` | Phase 0 step 5 measures this directly on a trivial baseline |
| 4.4 | Public baseline (DINOv2) scores ~0.809 | `UNVERIFIED` | the public leaderboard |

---

## 5. Open questions to settle during Phase 0

1. **What is the unit of prediction** — one row per study, or one row per study × finding? Decides the whole output head layout.
2. **Is laterality (left/right knee) in the labels, the DICOM headers, or both?** Mirroring right knees is only correct if laterality is reliable.
3. **Are report texts available for the test set at inference time?** If not, the report labeler is purely a training-time device, and the inference kernel is image-only. This changes the architecture, not just the code.
4. **Does the gold subset (~58) overlap the report set**, and is it site-diverse or concentrated? A gold set from two sites cannot calibrate twelve.
5. **How many studies have no report at all?**

---

## 6. Sources consulted

| source | status |
|---|---|
| Competition overview page | reachable, but the page body is client-rendered — only the title/meta tags are readable without a browser session. Field descriptions must come from the API or an authenticated fetch. |
| Kaggle API (`/api/v1/competitions/list`) | reachable, 401 without credentials |
| Kaggle internal JSON API (`/api/i/competitions.CompetitionService/*`) | 400 without a session token — not a usable unauthenticated path |
| Kickoff brief | second-hand throughout; treated as hypotheses, not facts |
