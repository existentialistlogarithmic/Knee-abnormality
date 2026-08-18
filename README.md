# RSNA Knee Abnormality Detection

Competition pipeline for the Kaggle challenge
[`rsna-knee-abnormality-detection`](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Maintainer: **existentialistlogarithmic**

## Where this stands

**Phase 0 (verification) — built, tested, and waiting on one repository secret.**
No modelling has started, and none will until the facts in `docs/FINDINGS.md`
are verified. Every claim about the competition data currently carries an
`UNVERIFIED` tag.

Target: **macro ROC-AUC ≥ 0.90** — see `docs/STRATEGY.md` for why that is read
as AUC rather than accuracy, and what has to be true for it to be reachable.

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
