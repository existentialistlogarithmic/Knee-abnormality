# RSNA Knee Abnormality Detection

Competition pipeline for the Kaggle challenge
[`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Maintainer: **existentialistlogarithmic**

## Where this stands

**Phase 0 is complete.** All five verification steps ran against the live
competition; `docs/FINDINGS.md` is the record and is almost entirely `VERIFIED`.
Phase 1 (the report labeler) is built. Phase 2 (imaging) is in progress.

| stage | state |
|---|---|
| Phase 0 — verification | complete, 5/5 steps |
| Phase 1 — report labeler | macro AUC **0.769** vs the 58 expert-labelled studies |
| Phase 2 — cache | v1 192px (4 shards) and v2 288px (8 shards), both verified complete |
| Phase 2 — imaging model | **0.725 on the leaderboard** |
| Phase 3 — inference | working, 2.2 s/study → 0.8 h of the 9 h cap |

**Leaderboard: 0.725.** Constant priors score 0.500 and scanner metadata alone
scores 0.531, so the pixels are contributing 0.194 of real signal.

What the data actually is: **4,407 training studies, exactly 58 with expert
labels**, 12 binary findings, one free-text report per study across ten
languages (English is only 39% of them), and **819,635 DICOM slices**.

Four measurements govern everything downstream:

- **No report text at inference.** `test.csv` has a single column, so the report
  labeler manufactures training targets and never ships in the submission.
- **No site label exists anywhere** — the DICOM headers are de-identified too.
  Folds group on a *scanner fingerprint* instead (`src/folds.py`). Random
  K-fold inflates macro AUC by **0.087**.
- **The CV-to-leaderboard gap has opposite signs for the two model families**,
  which is the most useful thing measured so far:

  | model | report-label CV | leaderboard | gap |
  |---|---:|---:|---:|
  | scanner metadata (no pixels) | 0.669 | 0.531 | **+0.138** |
  | imaging (resnet34 2.5D) | 0.700 | **0.725** | **−0.025** |

  CV is scored against noisy report-derived labels; the leaderboard against
  expert labels. A model that learns anatomy disagrees with its own targets
  exactly where they are wrong, so label noise depresses its CV but not its
  score. A model with no anatomical information cannot do that — all it can
  learn is the site-convention part, which does not transfer.
  **Consequence: report-label CV understates imaging progress and overstates
  everything else.**
- **The label ceiling is not hard.** A model trained on 0.769-quality labels
  scored 0.725 against expert truth, and was not converged.

Target: **macro ROC-AUC ≥ 0.90**. Current standing: **0.725**. The leaderboard
top is 0.951 and the top 200 teams are all above 0.917 (`docs/FINDINGS.md` §8),
so 0.90 is below the field, not above it — there is a long way to go.

## Layout

```
data/          competition CSVs + sample DICOMs        (gitignored)
eda/           local CPU analysis scripts
src/           report labeler, preprocessing, model code
kaggle/        one folder per kernel, each with kernel-metadata.json
docs/          FINDINGS.md, STRATEGY.md, EXPERIMENTS.md, ROADMAP.md
artifacts/     derived data                            (gitignored)
```

`data/` and `artifacts/` are gitignored, as are `*.csv`, `*.dcm`, `*.parquet`
and friends wherever they appear — anything that can carry a StudyInstanceUID,
a report string, or a pixel stays out of git history. The only whitelisted
tabular files are the language lexicons under `src/lexicons/`, which contain
vocabulary and no patient data.

## Running Phase 0 on GitHub (no local setup needed)

1. Create a Kaggle API token at <https://www.kaggle.com/settings/api>.
2. Add it here as a repository secret named **`KAGGLE_API_TOKEN`**:
   *Settings → Secrets and variables → Actions → New repository secret*.
3. Click **Join Competition** on the competition page and accept the rules with
   that same account.
4. Actions tab → **phase0-verify** → *Run workflow*. Choose `1` for the file
   inventory alone, or `1+2` to also download and audit the small tables.

The run posts a redacted summary to the run page and, because this repository is
private, keeps the full outputs as a build artifact for 14 days. If the
repository is ever made public the workflow refuses that upload — file paths
embed StudyInstanceUIDs, and competition data may not be redistributed.

Whether the rules have been accepted is answered by the run itself: it prints
the API's `user_has_entered` field.

### Running the same thing locally

```bash
pip install -r requirements.txt
export KAGGLE_API_TOKEN=<token>          # or: kaggle auth login
python eda/phase0_01_auth_and_files.py   # inventory; downloads nothing
python eda/phase0_02_audit_tabular.py    # downloads only the small tables
```

`.devcontainer/` is set up too, so a Codespace comes with the dependencies
installed and forwards a `KAGGLE_API_TOKEN` Codespace secret.

## What the two Phase 0 scripts do

**`phase0_01_auth_and_files.py`** — prints competition metadata verbatim from
the API (metric, deadlines, reward, code-competition flag, rules acceptance) and
inventories every data file by name and size. Downloads nothing.

**`phase0_02_audit_tabular.py`** — downloads only tabular files under a size
limit, then audits them **blind**: it assumes nothing about column names or how
many targets there are. It discovers binary target columns and their positive
rates, spots partially-labelled columns (the gold subset), profiles free-text
columns by length and detected language, finds low-cardinality columns that look
like site or scanner metadata, and measures how much the files overlap on shared
keys. Its report contains aggregates only — no identifiers, no report text.

## Tests

```bash
pytest tests -q
```

Runs on every push, with no secrets and no competition data: the scripts are
exercised against a synthetic fixture. Two of the tests exist specifically to
assert that outputs meant to travel — CI job summaries, anything pasted into
docs — never contain a study identifier or a line of report text.

## Ground rules

Documented in `docs/STRATEGY.md`. The short version: verify before building,
never fabricate a number, group the folds, and no report text ever goes to a
hosted LLM API (competition Rule 4.b).
