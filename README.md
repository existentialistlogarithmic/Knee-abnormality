# RSNA Knee Abnormality Detection

Competition pipeline for the Kaggle challenge
[`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Maintainer: **existentialistlogarithmic**

## Where this stands

**Phase 0 (verification), step 1 — blocked on Kaggle credentials.**
No modelling has started, and none will until the facts in `docs/FINDINGS.md`
are verified. Every claim in that file currently carries an `UNVERIFIED` tag
except the handful about the environment itself.

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

## Setup

```bash
pip install -r requirements.txt

# any one of these authenticates the CLI (2.x):
kaggle auth login                        # OAuth, nothing to store
export KAGGLE_API_TOKEN=<token>          # from kaggle.com/settings/api
# or ~/.kaggle/access_token, or legacy ~/.kaggle/kaggle.json (chmod 600)
```

You must also click **Join Competition** and accept the rules before any data
call works. The verification script reports `user_has_entered` so this is not
left to guesswork.

## Phase 0, step 1

```bash
python eda/phase0_01_auth_and_files.py
```

Lists competition metadata and every data file with its size. **Downloads
nothing** — safe to point at a several-hundred-GB competition. Writes
`artifacts/phase0/{competition_meta.json,competition_files.csv,step1_summary.md}`.

## Ground rules

Documented in `docs/STRATEGY.md`. The short version: verify before building,
never fabricate a number, group the folds, and no report text ever goes to a
hosted LLM API (competition Rule 4.b).
