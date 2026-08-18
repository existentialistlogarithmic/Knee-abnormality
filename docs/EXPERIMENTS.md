# EXPERIMENTS

One entry per run. Append only — never rewrite history, and never delete a run
because it was bad. A run with no entry did not happen.

## Entry template

```markdown
### E000 — one-line title
- **date**:
- **commit**:
- **what changed**: (exactly one thing, if possible)
- **config**: fold scheme, backbone, input size, slices, epochs, LR
- **runtime**: wall clock, accelerator (T4 / CPU), GPU-hours spent
- **CV**: macro AUC = , per-finding AUC in artifacts/…
- **LB**: public = , private = (blank until the end)
- **prediction spread**: std of predicted probabilities per finding
- **what it means**: one honest sentence. "Unclear" is a valid answer.
- **next**:
```

Notes on the fields that people fudge:

- **CV** and **LB** are separate columns because their *gap* is the signal. A
  CV that improves while LB does not means the fold scheme is leaking.
- **prediction spread** catches the classic failure where a model quietly
  collapses to predicting base rates and still posts a respectable AUC on the
  easy findings.
- **GPU-hours spent** is tracked because the weekly budget is ~30 and it is
  gone before it feels gone.

---

## Log

### E000 — repo skeleton and Phase 0 tooling
- **date**: 2026-08-17
- **commit**: (this commit)
- **what changed**: repo layout, `.gitignore`, `requirements.txt`, docs stubs,
  and `eda/phase0_01_auth_and_files.py` (competition file inventory, no download).
- **config**: n/a
- **runtime**: n/a (no compute used)
- **CV**: n/a
- **LB**: n/a
- **what it means**: nothing is verified yet. The Kaggle API is reachable from
  the working environment but no credentials are present, so the step 1 run is
  blocked on auth. The inventory script was smoke-tested against a stubbed API
  so that the reporting path is known-good before it is pointed at the real one;
  the stub's numbers were discarded and never entered `FINDINGS.md`.
- **next**: run step 1 with real credentials; fill `FINDINGS.md` §1–§3.

### E001 — Phase 0 moved to GitHub Actions; blind tabular audit added
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: `.github/workflows/phase0.yml` runs Phase 0 against Kaggle
  from a `KAGGLE_API_TOKEN` repository secret; `.github/workflows/tests.yml`
  runs lint + tests on every push with no secrets. Added
  `eda/phase0_02_audit_tabular.py` (schema-discovering audit of the small
  tables), `tests/` with a synthetic fixture, `.devcontainer/`, and `ruff.toml`.
- **config**: n/a
- **runtime**: n/a (no compute used; CI only)
- **CV**: n/a
- **LB**: n/a
- **what it means**: Phase 0 no longer needs a local machine. Verified along the
  way: the repo is private (so CI may retain identifier-bearing artifacts), and
  `py3langid` identifies en/de/fr/ru/ja/es/tr offline in 0.24 s — which matters
  because Rule 4.b forbids sending report text to a hosted API and the
  submission kernel has internet off. Still nothing verified about the
  competition data itself; the token secret is the only remaining blocker.
- **next**: set the secret, run **phase0-verify** with steps `1+2`, fill
  `FINDINGS.md` §2–§3 from the output, then Phase 0 steps 3–5.

### E002 — first authenticated Phase 0 run; §2–§4 of FINDINGS filled from real data
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: ran `phase0_01` and `phase0_02` against the live competition.
  Fixed a real bug found by doing so: the file listing had no checkpointing and
  no rate-limit handling, so it lost 25,200 listed files to a single HTTP 429.
  It now backs off, retries, and checkpoints every 25 pages, and resumes from
  the saved page token.
- **config**: n/a
- **runtime**: metadata + 5 CSVs in ~2 min; full file listing still paginating
- **CV**: n/a
- **LB**: n/a
- **what it means**: the kickoff brief was accurate on the numbers — 4,407
  studies and exactly 58 gold labels, both to the digit — and wrong or silent on
  two things that change the build. There is **no site/scanner column in any
  CSV**, so grouped folds and the leakage audit are blocked on the DICOM header
  scan rather than being a `groupby`. And **`test.csv` has one column**, so no
  report text exists at inference: the labeler is a training-time device that
  never ships. On the other side, the host hands us `Anatomical_Plane` per
  series, so series selection does not need orientation maths. One data quirk:
  `Fluid_Sensitive` and `Fat_Suppression` are byte-identical across all 24,371
  series — one bit, two names.
- **next**: 0.3, the header scan kernel — now on the critical path for both the
  fold scheme and laterality.

### E003 — label-ceiling probe: reports carry real signal, negation is the bottleneck
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: added `eda/label_ceiling_probe.py` and the DICOM header-scan
  kernel `kaggle/00_dicom_header_scan/run.py`.
- **config**: crude multilingual substring matching, no negation/hedging/severity
- **runtime**: seconds, CPU
- **CV**: macro balanced accuracy **0.601** on the 58 gold studies
- **LB**: n/a
- **prediction spread**: n/a
- **what it means**: the premise holds — the reports do carry finding-level
  signal — and the failure mode is specificity, not sensitivity. `ACL` gives 37
  mentions for 24 true positives (sens 0.75, spec 0.44) and `Effusion` 40 for 35
  (spec 0.26), because radiologists write "ACL intact" and "no effusion" and a
  matcher without negation scores those as positives. That is the single
  highest-value fix in Phase 1, and it is ordinary NLP work rather than missing
  information. Also measured: the gold subset is 48% English against 39% in the
  corpus and contains no French or Bosnian, so gold-set numbers will overstate a
  labeler's real performance.
- **next**: Phase 1 negation/hedging layer, and push the header scan kernel so
  fold grouping stops being provisional.

### E004 — competition pages retrieved through the API; metric and budget now exact
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: no code. Pulled the competition's own pages via
  `competition_list_pages`, which is the API route to the content the web UI
  renders client-side and which the overview fetch could not reach.
- **runtime**: seconds
- **what it means**: the remaining guesses are now facts. Metric is
  `(1/12) Σ AUC_i`, macro-averaged ROC-AUC. Limits are 9 h CPU or GPU, internet
  off, output named `submission.csv`. The hidden test is **~1,300 studies**,
  which at ~5.5 series and a median 30 slices is ~215,000 slices, so the
  inference budget is **~24 s per study** — the real constraint on Phase 2, and
  now a number rather than a worry. Prize split confirmed at $59,000 main plus
  **$18,000 efficiency**, and the efficiency score divides runtime by the 9-hour
  cap, so a fast kernel places well without a top-ten AUC. The host also states
  outright that reports are provided "from which you may wish to derive the
  labels for the remaining studies" — the weak-supervision path is the intended
  one. One correction to an earlier reading: `Fluid_Sensitive` and
  `Fat_Suppression` are byte-identical across all 24,371 training rows, but the
  host warns they are "not necessarily equivalent for every case", so both stay
  as separate model inputs rather than being collapsed.
- **next**: push the header-scan kernel; begin the Phase 1 negation layer.

### E005 — Phase 1 labeler v1: macro AUC 0.745 against expert labels
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: `src/report_labeler.py` plus reviewable lexicons
  (`src/lexicons/findings.csv`, 198 terms; `src/lexicons/cues.csv`, 154 cues
  across 10 languages). Cues bind to the nearest mention within sentence bounds,
  so opposite polarities in one sentence resolve correctly.
- **config**: lexicon only, no fitted parameters; abstain scored 0.15
- **runtime**: ~40 s for all 4,407 reports, CPU
- **CV**: **macro AUC 0.7448** on the 58 gold studies (crude Phase 0 floor was
  0.601 balanced accuracy). 9 of 12 findings above 0.70; none below 0.55.
- **LB**: n/a
- **what it means**: the decisive addition was **normality cues**, not negation.
  These reports mostly assert health — "intact", "normaldir", "regelrecht",
  "запазена" — rather than deny disease, so a negation-only system reads "ACL
  intact" as a positive. Adding that channel moved ACL from specificity 0.44 to
  AUC 0.822.
  The abstain rates now say exactly where the remaining work is, because AUC
  tracks them almost monotonically: **Synovitis 0.561 AUC at 89% abstain** is the
  worst by a distance — it is positive in 27 of 58 gold studies but the lexicon
  finds it in barely a tenth of reports, so radiologists must be describing it in
  words the term list does not have. **Lateral OA (77% abstain) and Medial OA
  (76%)** are next; compartment osteoarthritis is being written as cartilage or
  chondropathy language that the current terms miss.
- **next**: lexicon coverage for Synovitis and the two compartment-OA findings;
  they are the three cheapest AUC gains available and need no GPU.
