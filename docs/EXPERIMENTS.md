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

### E006 — composite (proximity) terms: macro AUC 0.745 → 0.759
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: added proximity matching to the labeler — a term written
  `anchor~degeneration` fires only when both halves appear within 45 characters.
  Plus 490 composite rows for the three osteoarthritis findings and 20
  synovium-specific terms.
- **config**: lexicon only, no fitted parameters; abstain scored 0.15
- **runtime**: 1 min 44 s for 4,407 reports (was ~40 s; composites cost more)
- **CV**: macro AUC **0.7589**, up from 0.7448. 10 of 12 findings now above 0.70.
- **gold-set evaluations used so far: 2.** Tracked deliberately — 58 studies is
  the only test set there is, and every extra look at it converts a test into a
  fit. Changes were chosen from corpus frequency evidence, not from gold-set
  feedback.
- **what it means**: the compartment hypothesis was right. Osteoarthritis is
  almost never written as one phrase; corpus mining found degeneration
  vocabulary in 84% of reports but a single-string match scored sensitivity 0.87
  at precision 0.27, because it could not tell which compartment the words
  belonged to. Proximity fixes the attribution: **PF OA 0.709 → 0.782** (abstain
  57% → 38%) and **Medial OA 0.695 → 0.778**. Lateral OA barely moved
  (0.691 → 0.684) and has the widest interval of any finding, [0.53, 0.84] on 11
  positives — that difference is noise, not a regression.
- **Synovitis is text-limited, and this is now measured rather than suspected.**
  Before touching the lexicon I checked the ceiling: across *all* synovial
  vocabulary, gold-positive studies mention it 16 times out of 27 and
  gold-negative studies 12 times out of 31 — sensitivity 0.59 at precision 0.57
  against a 0.47 base rate, which is close to no information at all. The
  labeler landed at 0.580 AUC, exactly where that ceiling predicted. Further
  lexicon work on Synovitis cannot pay off; it needs image supervision, and its
  report-derived labels should be down-weighted in Phase 2.
  Hoffa's fat pad (19% of reports) and plica (7%) were deliberately excluded as
  synovitis terms — they are structures usually mentioned as normal, and adding
  them would have cost precision.
- **next**: laterality handling; then per-language error review, since English
  is over-represented in the gold set and the other 61% is unaudited.

### E007 — per-language coverage audit: the non-English lexicons were broken
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: added `eda/labeler_coverage_by_language.py`, which measures
  abstain rate per finding per language. It needs **no labels**, so it is a safe
  optimisation signal — the 58 gold studies stay a test set instead of becoming
  a training signal. Then 153 lexicon rows driven entirely by what it found.
- **runtime**: ~2 min per pass, CPU
- **CV**: macro AUC 0.7589 → **0.7608** (third gold evaluation)
- **what it means**: the audit exposed that several non-English lexicons were
  barely functioning, which no amount of gold-set evaluation would have revealed
  — the gold subset holds 2 German and 3 Greek studies. Abstain rates against
  the English baseline, before → after:

  | hole | before | after | reports |
  |---|---:|---:|---:|
  | German ACL | 0.71 | **0.09** | 262 |
  | Greek ACL | 0.74 | **0.09** | 321 |
  | Bulgarian MCL | 0.97 | **0.03** | 220 |
  | Turkish Lateral OA | 0.99 | **0.80** | 546 |
  | Turkish Medial OA | 0.98 | **0.79** | 546 |
  | Croatian Contusion | 1.00 | **0.61** | 330 |
  | Spanish ACL | 0.87 | **0.44** | 682 |
  | Spanish MCL | 0.94 | **0.46** | 682 |

  Every fix came from a frequency scan rather than from intuition, and the
  causes were ordinary morphology that a non-speaker would never guess:
  Spanish reports name the cruciates in the **plural** ("ligamentos cruzados
  íntegros" — 44% of Spanish reports, against 1% for "cruzado anterior"); German
  prefers the plural **"Kreuzbänder"** (51% vs 26%); Greek writes **stems**
  ("χιαστ" in 91%, but "έσω πλάγι" in only 7%); Bulgarian uses bare
  **"колатерал"** (97%). Turkish compound anchors were simply wrong —
  "medial femorotibial" appears in **one** report of 546 — so the anchor became
  the bare compartment word, paired only with degeneration terms that cannot
  describe a meniscus, which keeps meniscal degeneration from firing
  compartment OA.

- **Why the gold number moved so little, and why that is expected.** The gold
  subset is 48% English while the corpus is 39%, and it contains 2 German and
  3 Greek studies. Almost everything fixed in this round is invisible to it. The
  +0.002 on gold is therefore a **floor on the real improvement**, not a measure
  of it — the training labels for the 61% of studies that are not English got
  materially better, and there is no labelled data anywhere that can prove it.
  This is the clearest example so far of why abstain rate, not gold AUC, is the
  right instrument for lexicon work.
- **next**: the remaining holes are compartment OA in Greek, Croatian and
  Spanish (all still ≥0.77 abstain). Then laterality.

### E008 — first kernels on Kaggle; submission pipeline proven, budget measured
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: pushed two kernels to the Kaggle account
  (`achelijndiamantidis`, not the GitHub name): the DICOM header scan and
  `kaggle/01_submission_baseline`.
- **runtime**: submission kernel **1.2 s** wall clock
- **CV**: n/a
- **LB**: **0.500 public** — exactly as predicted. Constant predictions give an
  AUC of 0.5 whatever constants are chosen, so this is confirmation the scoring
  path works, not a result.
- **what it means**: three things bought cheaply.
  1. **The submission pipeline is proven end to end** while there is still time
     to fix it, which is the failure mode that kills code-competition entries at
     the deadline. There is now a known-good baseline to diff against.
  2. **The mount path was wrong and cost a kernel run to discover.** Competition
     data lives at `/kaggle/input/competitions/<slug>`, nested under
     `competitions/`, not at `/kaggle/input/<slug>`. The first header-scan run
     found nothing and exited having scanned zero series. Both kernels now
     search for the files that must exist and log what they found.
  3. **The runtime budget is measured, not guessed.** Directory traversal plus
     one header read per series costs **0.059 s/study**, which extrapolates to
     77 s over the ~1,300-study hidden test — **0.2% of the 9-hour cap**. File
     access is effectively free, so essentially the whole ~24 s/study is
     available for pixel decoding and inference. Caveat: this did not decode
     pixels, and ~215,000 slices is still the real constraint.
- **also learned**: Kaggle derives the kernel slug from the **title**, not the
  `id` field; a mismatch silently creates a second kernel on the next push.
  Set `id` to the slug in the URL the first push prints.
- **next**: the header scan is running over ~31,500 series; its output gives the
  scanner fingerprint, and with it grouped folds and the leakage audit.

### E009 — header scan complete; leakage audit closes Phase 0
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: header scan ran to completion on Kaggle; added
  `src/folds.py` (scanner fingerprint, the single source of truth for splits)
  and `eda/leakage_audit.py`.
- **runtime**: header scan **372 s** for 24,386 series / **819,635 slices**,
  0 errors, 2.6 MB parquet — 0.015 s per series on a CPU kernel.
- **CV**: **random K-fold inflates macro AUC by 0.087** on metadata alone
  (0.752 random vs 0.664 scanner-grouped).
- **what it means**: the brief's 0.05–0.14 estimate was right, and grouped folds
  are now settled rather than a preference.
  Two findings matter more than the headline number:

  **1. The fingerprint nearly failed silently.** The first design used
  `ImagingFrequency` at full precision and produced 8,618 groups for 4,410
  studies — 5,633 of them singletons. A "grouped" K-fold on that key is a random
  K-fold in disguise: it would have reported near-zero inflation and hidden the
  leak completely, while every downstream CV number stayed quietly wrong.
  The cause is Larmor drift between sessions on one magnet — the Philips Ingenia
  3 T units here emit **739 distinct raw values across 2,480 series**. Rounding
  to 2 decimals gives 178 usable groups (median 8 studies, max 246, 42
  singletons). This is recorded prominently in `src/folds.py`.

  **2. Metadata alone scores 0.664 grouped**, where memorisation is impossible.
  Part is genuine — protocol choice reflects clinical suspicion — and part is
  population effects that transfer across scanners of a class. It is a baseline
  any imaging model must beat before we can claim the pixels are contributing.
  Caveat: targets are report-derived, so shared site convention inflates this;
  against expert labels it would likely be lower.

  Also confirmed: **no patient appears in more than one study** (0 of 4,410), so
  folds do not additionally need patient grouping.
- **next**: Phase 0 gate. Then a metadata-only submission — it should land near
  0.6 rather than 0.5, and would tell us whether LB agrees with grouped CV.

### E010 — metadata-only submission: CV 0.669 → LB 0.531, a gap of 0.138
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: `kaggle/02_metadata_submission` — scanner and geometry
  features only, trained on report-derived labels with 5-fold `GroupKFold` over
  178 scanner fingerprints. Supporting artifacts uploaded as a private Kaggle
  dataset, which also exercised the mount path Phase 2 needs for weights.
- **runtime**: 19 s wall clock
- **CV**: grouped OOF macro AUC **0.6687** (vs report-derived labels)
- **LB**: **0.531 public** (vs expert labels)
- **prediction spread**: healthy, 0.08–0.35 per finding — no collapse to priors
- **what it means**: this was the point of the submission, and the answer is
  blunt. Of 0.169 apparent skill above chance, **~0.031 survives** contact with
  expert labels.
  The careful reading matters. This model has **no anatomical information** — it
  cannot see a ligament. Everything it learned was a correlation between scanner
  identity and *reporting convention*: verbose sites produce more positive report
  labels and have distinctive scanners. That is real for report-derived targets
  and nearly worthless for expert ones. Grouped CV removed the *memorisation*
  but could not remove the convention baked into the targets themselves.
  So the rule going forward: **report-label CV ranks candidate models; it never
  estimates the leaderboard.** Absolute claims come from the leaderboard or from
  the 58 gold studies with their intervals.
  An imaging model should transfer better, since pathology in pixels is the same
  pathology the experts graded — but it will inherit some of this, and its own
  CV-to-LB gap must be measured on the very first run rather than assumed.
- **also**: a third mount shape cost a run. Datasets land at
  `/kaggle/input/datasets/<owner>/<name>`; all kernels now share a depth-bounded
  search that refuses to walk the ~1M-file image directories.
- **next**: Phase 2 cache-build kernel. The bar an imaging model must clear is
  **0.669 grouped CV / 0.531 LB** — below that, the pixels are contributing
  nothing.
