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
| Phase 2 — imaging model | **0.725 on the leaderboard** (192px); 288px scored 0.688 |
| Phase 2 — kernel tree | generated from `src/pipeline.py`; 22 of 28 kernels |
| Phase 3 — inference | working, 2.2 s/study → 0.8 h of the 9 h cap |

**Leaderboard: 0.725.** Constant priors score 0.500 and scanner metadata alone
scores 0.531, so the pixels are contributing 0.194 of real signal.

What the data actually is: **4,407 training studies, exactly 58 with expert
labels**, 12 binary findings, one free-text report per study across ten
languages (English is only 39% of them), and **819,635 DICOM slices**.

Six measurements govern everything downstream:

- **No report text at inference.** `test.csv` has a single column, so the report
  labeler manufactures training targets and never ships in the submission.
- **No site label exists anywhere** — the DICOM headers are de-identified too.
  Folds group on a *scanner fingerprint* instead (`src/folds.py`). Random
  K-fold inflates macro AUC by **0.087**.
- **Report-label CV neither predicts the score nor reliably ranks models.**
  This is the most important thing measured, and it is bad news:

  | model | report-label CV | leaderboard | gap | CV rank | LB rank |
  |---|---:|---:|---:|:--:|:--:|
  | scanner metadata (no pixels) | 0.669 | 0.531 | +0.138 | 4 | 4 |
  | imaging, 192px, eff. batch 16 | 0.700 | **0.725** | −0.025 | 2 | **1** |
  | imaging, 288px, eff. batch 4 | 0.690 | 0.668 | +0.022 | 3 | 3 |
  | imaging, 288px, eff. batch 16 | **0.728** | 0.688 | **+0.040** | **1** | 2 |

  CV called the last row the best model by **+0.028** — a wider margin than any
  difference this project had acted on. The board says it is **0.037 worse**.
  CV is scored against noisy report labels and the board against expert ones, so
  a model can raise its CV by **fitting label noise more precisely**, and every
  point bought that way is worth nothing on the board. See `FINDINGS.md` §11.

  **Consequence: there is no trustworthy offline selection signal**, and the
  board allows 2 submissions a day. The one scalable alternative is out-of-fold
  scoring against the 58 expert-labelled studies — training already uses expert
  labels for gold studies and splits afterwards, so a complete 5-fold run yields
  one expert-scored prediction per gold study from a model that never saw it.
- **A confounded experiment is worse than no experiment — and correcting it does
  not always vindicate the idea.** The first 288px run changed five things at
  once. Re-running it with only the effective batch corrected moved the board
  from 0.668 to 0.688, so the diagnosis was right and worth +0.020. But 288px is
  **still 0.037 behind 192px**, in both runs. The resolution hypothesis is not
  supported; the confounded run simply never tested it.
- **Fold spread is 0.033.** The same configuration reached 0.7001 on fold 0 and
  0.7334 on fold 1. That is wider than most differences this project has treated
  as signal, so single-fold comparisons carry it as a caveat.
- **The label ceiling is not hard.** A model trained on 0.769-quality labels
  scored 0.725 against expert truth, and was not converged.

Target: **macro ROC-AUC ≥ 0.90**. Current standing: **0.725**. The leaderboard
top is 0.952 and the top 200 teams are all above 0.917 (`docs/FINDINGS.md` §8),
so 0.90 is below the field, not above it — there is a long way to go.

**But there are two boards, and this project is much better placed on the other
one.** The efficiency prize is scored `AUC/(Benchmark − maxAUC) + Runtime/32400`,
minimised, which makes an extra hour cost **0.0502 AUC**. Inference here takes
**0.8 h** where published systems take 3–4, and that alone puts the 0.725 model
ahead of a public **0.883** model on efficiency. See
`docs/COMPETITIVE_ANALYSIS.md` — which also itemises the main-board gap, most of
which is **label quality**: the leading public report reader scores 0.881
against the 58 expert-labelled studies where this project's lexicon scores
0.769.

## Layout

```
data/          competition CSVs + sample DICOMs        (gitignored)
eda/           local CPU analysis scripts
src/           report labeler, preprocessing, model code, pipeline manifest
kaggle/        one folder per kernel; 22 of 28 generated from src/pipeline.py
docs/          HANDOFF.md first; then STATUS.md, PATH.md, FINDINGS.md, ...
artifacts/     derived data                            (gitignored)
```

`kaggle/` is mostly generated output: `src/pipeline.py` declares the pipeline
and `eda/generate_kernels.py` renders every cache, training and inference
kernel from the templates in `kaggle/_templates/`. `python
eda/generate_kernels.py --check` fails if a generated kernel has been edited by
hand, and runs as a test. See `kaggle/README.md`.

**Start here: `docs/HANDOFF.md`** — the session entry point, with the live
blocker and the next action. **Before any push, run `bash eda/preflight.sh`**;
it runs the same four gates as CI (lint, tests, kernel drift, and the check
that no patient-derived file is tracked) so a failure arrives locally in a
minute rather than on the Actions tab in ten.

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
