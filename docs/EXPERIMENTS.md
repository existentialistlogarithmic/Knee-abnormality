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

### E011 — cache built; the P100 trap; accelerator names found
- **date**: 2026-08-18
- **commit**: (this commit)
- **what changed**: four cache shard kernels built the full training cache;
  `kaggle/04_train` written and pushed.
- **runtime**: four cache shards ran in parallel and all completed
- **what it means**: three results, one of them a trap that would have cost days.

  **1. The cache is complete and correct.** The training kernel mounted all four
  shards and reported **4,407 cached / 4,407 usable / 58 gold present**, with
  fold 0 splitting 3,525 train against 882 validation across **35 scanner
  groups**. That verification came from the training log because the
  output-download endpoint was rate-limited — worth noting as a technique.

  **2. Kaggle's default GPU is a P100 and it cannot run this build.** Not slowly
  — at all. `enable_gpu: true` with no shape gives compute capability 6.0, and
  the first CUDA launch dies with `no kernel image is available for execution on
  the device`. The brief's warning was right and understated.

  **3. The accelerator names were in the SDK the whole time**: `NvidiaTeslaT4`,
  `NvidiaTeslaP100`, `Tpu1VmV38`, documented in the `kagglesdk` docstring for
  `machine_shape` — while the CLI source comments that the enum "is not
  currently included in kagglesdk". That comment cost three probe runs.
  Two dead ends recorded: `--accelerator` is **not validated** (an invalid value
  pushes silently), and a wrong-but-plausible name (`GpuT4x2`) is **silently
  ignored**, returning a P100 rather than erroring.

- **the cheap thing that paid off**: putting a device check at the top of the
  kernel that exits before touching CUDA. Each wrong shape name cost ~8 seconds
  instead of a session. Against a ~30 GPU-hour weekly budget, three dead
  sessions is real money.
- **caught before it mattered**: the training kernel originally found a single
  cache directory. With four mounted shards it would have trained on a quarter
  of the data with a perfectly healthy-looking loss curve. It now collects
  across all shards and warns on duplicates.
- **next**: fold 0 result against the 0.669 bar.

### E012 — first imaging model: fold 0 macro AUC 0.6928 vs a 0.669 bar
- **date**: 2026-08-18
- **commit**: (this commit)
- **config**: resnet18, 2.5D over (3 planes × 20 slices) at 192px, attention
  pooling, 12 heads, 6 epochs, batch 8, LR 3e-4, gold weight 8, abstain masked,
  scanner-grouped fold 0 of 5
- **runtime**: **9.2 min/epoch, 54.9 min total** on **2× Tesla T4** (compute 7.5
  — `NvidiaTeslaT4` grants two cards). ~1.8 GPU-hours of the ~30/week budget.
- **CV**: best val macro AUC **0.6928** (report-derived labels, fold 0)
- **LB**: not submitted
- **prediction spread**: 0.159 — healthy, no collapse to priors

| epoch | loss | val macro AUC |
|---:|---:|---:|
| 0 | 0.6482 | 0.6164 |
| 1 | 0.6258 | 0.6520 |
| 2 | 0.6208 | 0.6648 |
| 3 | 0.6087 | 0.6300 |
| 4 | 0.6016 | 0.6669 |
| 5 | 0.5931 | **0.6928** |

- **what it means — read this carefully.** It clears the metadata bar, but by
  **0.024**, and that is weak evidence rather than a result.

  The metadata model reached 0.669 using **no pixels at all**, purely by
  correlating scanner identity with reporting convention. An imaging model
  sitting 0.024 above that has not yet shown it is reading *anatomy* — image
  appearance is itself scanner-correlated (contrast, noise, resolution), so a
  CNN can rediscover the same shortcut through the pixels. **The margin is too
  small to distinguish "sees pathology" from "sees the scanner".**

  Two things do point the right way: prediction spread is healthy at 0.159, so
  it is not predicting priors; and it was **still improving when it stopped** —
  AUC rose 0.667 → 0.693 on the last epoch and loss was still falling. This run
  is undertrained, not converged.

- **what would actually settle it**: a leaderboard submission. The metadata
  model's CV-to-LB gap was 0.138; if this model's gap is materially smaller,
  the pixels are contributing real anatomical signal. If it is similar, the CNN
  has found the same shortcut. That comparison is worth more than another
  epoch.
- **next**: (a) more epochs, since it had not converged; (b) an inference kernel
  so the gap can be measured. (b) is more informative.

### E013 — second corpus round: macro AUC 0.7608 → 0.7690
- **date**: 2026-08-19
- **commit**: (this commit)
- **what changed**: 70 lexicon rows, again driven entirely by the label-free
  coverage audit rather than by gold feedback.
- **CV**: macro AUC **0.7690** on the 58 gold studies (4th evaluation). 11 of 12
  findings now above 0.70; none below 0.55.
- **what it means**: the non-English OA anchors were near-dead for a third time,
  and for the same reason — vocabulary guessed rather than measured.

  | hole | abstain before | after | cause |
  |---|---:|---:|---|
  | Croatian Medial OA | 0.85 | **0.29** | `hrskavic` in 94% of Croatian reports, `medijaln` in 99%; the old anchor `medijalnom odjeljku` was rare |
  | Croatian Lateral OA | 0.92 | **0.36** | same |
  | Greek Medial OA | 0.91 | **0.66** | `χόνδρ` in 92%, `έσω` in 83%; the old anchor `έσω μηριαίου` in 5% |
  | Spanish Contusion | 0.84 | **0.49** | written as `médula ósea` (42%) + `edema` (42%), not `contusión` (7%) |
  | Spanish Baker's | 0.79 | **0.49** | named by site — `quiste` 52%, `poplíte` 44% — not by eponym (`baker` 10%) |
  | Spanish Medial OA | 0.77 | **0.62** | Spanish uses `interno`/`externo` for medial/lateral |

  Per-finding gold AUC moved where the coverage moved: **Medial OA 0.778 →
  0.836**, **Lateral OA 0.689 → 0.732**, Baker's 0.826 → 0.834. That is the
  coverage audit validating itself — it predicted which findings would improve
  before the gold set was touched.

- **one finding is not a lexicon hole**: Turkish `Fracture` abstains at 0.96, but
  `kırık` appears in 1% of Turkish reports, `fraktür` 2%, `fissür` 3%. Fractures
  are genuinely rare in this corpus. No vocabulary work will fix that, and
  chasing it would only cost precision.
- **next**: the labels are now the training targets for the v2 (288px) run.

### E014 — resnet34 / 24 epochs / 2×T4+AMP: 0.7001, and the curve says capacity is not the problem
- **date**: 2026-08-19
- **commit**: (this commit)
- **config**: resnet34, 2.5D over 3 planes × 20 slices at 192px, batch 16,
  cosine + 2-epoch warmup, EMA 0.999, label smoothing 0.02, AMP, DataParallel
- **runtime**: **3.3 min/epoch**, 79.7 min for 24 epochs. The v1 run was
  9.2 min/epoch on one card in fp32, so AMP + both T4s delivered the predicted
  ~2.8× — the throughput work paid for itself in one run.
- **CV**: best val macro AUC **0.7001** (report-derived labels, grouped fold 0)
- **prediction spread**: 0.196, rising monotonically — no collapse

| epoch | 0 | 5 | 11 | 17 | 21 | 23 |
|---|---:|---:|---:|---:|---:|---:|
| val macro AUC | 0.543 | 0.627 | 0.663 | 0.693 | 0.700 | **0.7001** |

- **what it means**: this is the important negative result of the night.
  Against the v1 run (resnet18, 6 epochs, same 192px cache) at **0.6928**, a
  **1.9× larger backbone trained 4× longer bought +0.007**. The curve is also
  clearly flattening — the last six epochs moved it 0.6946 → 0.7001 while
  training loss kept falling from 0.539 to 0.502, which is the signature of a
  model running out of *input* information rather than capacity.

  That is direct support for the resolution hypothesis: the v1 cache used
  0.60 mm/px against a native median of 0.312 mm, downsampling 96% of series
  roughly 2×. Adding parameters cannot recover detail the cache already threw
  away. The 288px v2 cache and its training run are the test of that.
- **inference measured on CPU**: 19.7 s/study, projecting **7.1 h of the 9 h
  cap** for ~1,300 studies. That works but leaves little headroom — the real
  submission kernel should be GPU. CPU was used here only because Kaggle caps
  concurrent GPU sessions at 2 and the v2 training run held both.
- **also learned**: with internet off, torchvision cannot fetch pretrained
  weights and `build_model` logs "training from scratch". That is harmless in
  the inference kernel — the checkpoint overwrites every parameter and the
  strict-load check confirms 0 missing and 0 unexpected keys — but the message
  is alarming and should be reworded.

### E015 — v2 cache built at 288px: 4,407/4,407 studies, zero errors
- **date**: 2026-08-19
- **commit**: (this commit)
- **what changed**: `kaggle/06_cache_v2_shard{0..7}` — 288px @ 0.40 mm/px,
  24 slices, same 115 mm field of view as v1.
- **runtime**: 14–23 min per shard, 8 shards, 5 concurrent (Kaggle's CPU cap)
  with the rest drip-fed by `eda/push_queue.sh`.
- **result**: **4,407 studies built across 8/8 shards, 0 errors**, 6.0 MB/study,
  shape `(3, 24, 288, 288)` — 26 GB total.
- **why it exists**: v1 used 0.60 mm/px against a native median of **0.312 mm**,
  so **96% of series were downsampled roughly 2×**, and it kept 20 of a median
  30 slices. Meniscal tears and cartilage lesions are thin structures; that is
  the detail they live in. The resnet34 run supports this reading — 1.9× the
  backbone and 4× the epochs bought +0.007 over resnet18 while the curve
  flattened and training loss kept falling, which is a model starved of input
  rather than capacity.
- **verification technique worth reusing**: Kaggle publishes a kernel's log only
  when it completes, so the "did every shard land" check could not be read from
  the training log mid-run. Fetching each shard's few-KB
  `cache_manifest_*.json` with `--file-pattern` verifies the same fact
  independently without touching the 26 GB of volumes — and without the
  output-download rate limit that has bitten this project repeatedly.

### E016 — the kernel tree became generated, and two live bugs fell out of it
- **date**: 2026-08-19
- **commit**: (this commit)
- **what changed**: `src/pipeline.py` declares the pipeline; `eda/generate_kernels.py`
  renders every cache, training and inference kernel from three templates and
  three shared modules in `kaggle/_templates/`. 22 of the 28 kernel folders are
  now generated output.
- **runtime**: no GPU time. Local generation is instant; `--check` runs in the
  test suite.
- **result**: measured, not asserted —

  | | before | after |
  |---|---:|---:|
  | hand-edited lines under `kaggle/` | 10,972 across 29 files | 1,148 across 6 templates |
  | plus the manifest | — | 381 |
  | functions with more than one variant across generated kernels | `build_model` ×2 | **none** |

- **why it exists**: the duplication was measurable and it had already caused a
  bug. `build_study` was byte-identical in 4 kernels, `find_marker` in 7,
  `read_series_volume` in 4 — and `build_model` had **drifted into two variants
  across 5 files**, the newer one applying ImageNet normalisation and the older
  one not. The training kernel and the inference kernel that scores its weights
  were on opposite sides of that split. Nothing would have raised.
- **two real bugs found while unifying**, both of which would have cost a
  submission rather than crashing:
  1. **The v2 inference kernel would have refused every existing checkpoint.**
     Registering the ImageNet `mean`/`std` as ordinary buffers puts two keys
     into every `state_dict`; the strict-load check at inference would then have
     rejected all five 192px folds — including the ones training right now.
     Fixed with `persistent=False`: they are constants, not learned state.
  2. **The fold ensemble's geometry guard did not fire.** It read
     `if recorded not in (None, SLICE_SUBSAMPLE_EXPECTED)`, so a checkpoint
     recording `None` passed a kernel expecting 18 — exactly the mismatch it was
     written to catch. Now an equality check over both `slice_subsample` and
     `input_norm`, with the pre-existing defaults spelled out rather than
     inferred from a missing key.
- **what is deliberately NOT tidied**: `input_norm` is `False` for v1 and v2 and
  `True` only for dinov2. That is a record of how those weights were trained,
  not a preference. Setting it True everywhere would read better and would make
  the manifest describe models that do not exist, which is the one thing that
  would make the ensemble guard worthless.
- **the check that keeps it honest**: `python eda/generate_kernels.py --check`
  fails if any generated kernel has been edited by hand, and runs as a test. A
  manifest nobody regenerates from is documentation, and documentation drifts.

### E017 — the 288px result reverses once the confound is removed: 0.7001 → 0.7282
- **date**: 2026-08-19
- **commit**: (this commit)
- **what changed**: nothing about the cache. The 288px run was relaunched with
  batch 4 × 4-step gradient accumulation, reproducing the **effective batch 16**
  that the 192px run used, instead of the effective batch 4 the first attempt
  actually trained at.
- **runtime**: 3.6 h on 2×T4 (192px was 1.3 h).
- **result, same fold, same split, same labels** —

  | run | geometry | effective batch | epochs | val macro AUC |
  |---|---|---:|---:|---:|
  | v1 fold 0 | 192px @ 0.60 mm/px | 16 | 24 | 0.7001 |
  | v2 fold 0, first attempt | 288px @ 0.40 mm/px | **4** | 30 | 0.690 |
  | **v2 fold 0, corrected** | 288px @ 0.40 mm/px | **16** | 30 | **0.7282** |

  Verified like-for-like from the logs before drawing any conclusion: both runs
  report `fold 0: train 3,525  val 882  val groups 35`. Same studies, same
  scanner groups, same targets.
- **what this overturns**: E015 read the first 288px result as evidence against
  the resolution hypothesis, and the README recorded 288px as **worse** (LB
  0.668 vs 0.725). That reading was wrong, and it was wrong for a reason already
  written down at the time — the run **changed five things at once**. Batch size
  was one of them, and it was the one that mattered. Correcting only the batch
  turns a 0.011 deficit into a **0.028 surplus**.
- **the honest limit on this**: it is one fold and it is CV, not the board.
  Report-label CV has ranked models correctly three times (0.700 > 0.690
  predicted 0.725 > 0.668) but has never predicted an absolute score — the gap
  has been +0.138, −0.025 and +0.022. `knee-infer-v2` is running against the
  corrected weights; that submission is what settles it.
- **the curve had not stopped**: the last three epochs went 0.725 → 0.727 →
  0.7282, still climbing at epoch 29 of 30. The 192px run flattened by epoch 18.
  So 0.7282 is a floor for this configuration, not its ceiling.
- **the error profiles are complementary**, which matters for ensembling. v2 is
  far stronger on Synovitis (0.781) and Medial Meniscus (0.753) and far weaker on
  Medial OA (0.519) and Lateral OA (0.599). The 192px fold-1 model is the mirror
  image — 0.724 Medial OA, 0.663 Synovitis. Two models that fail on different
  findings average better than two that fail on the same ones.

### E018 — fold 1 at 192px: 0.7334
- **date**: 2026-08-19
- **result**: best val macro AUC **0.7334** at epoch 18 of 24, then a slow
  decline to 0.7282 — the EMA weights are saved per epoch, so the checkpoint is
  the last epoch's, not the best. Worth fixing: keeping the best-epoch weights
  is free and this run gives up 0.005 by not doing it.
- **what it says about fold variance**: fold 0 reached 0.7001 and fold 1 reached
  0.7334 on the same configuration. **0.033 of spread between folds** is larger
  than most of the differences this project has been treating as signal. Any
  single-fold comparison from here needs that number attached to it.

### E019 — the corrected 288px model scores 0.688, and report-label CV is no longer a ranking device
- **date**: 2026-08-19
- **submission**: `knee-infer-v2` v3, public score **0.688**.
- **what it settles**: two things, and they point in opposite directions.
  - The batch-size diagnosis in E017 was **right**: 288px went **0.668 → 0.688**
    on the board with only the effective batch corrected. +0.020 of real gain.
  - The resolution hypothesis is **not supported**. 288px is 0.037 behind 192px
    on the board in both runs. The confounded run never tested the idea; the
    corrected run tested it and it lost.
- **what it breaks**: E017 leaned on "CV ranks models correctly", which had held
  across three submissions. It does not hold. CV said the corrected 288px model
  was the best available by **+0.028**; the board put it **0.037 behind**. The
  ranking inverted on exactly the comparison that was being used to steer.
- **the mechanism, stated as a hypothesis**: CV is scored against report-derived
  labels (0.769 macro AUC vs expert truth) and the board against expert labels.
  A model can raise CV by fitting the label noise more precisely. The 288px
  model's biggest CV gain is **Synovitis 0.781** — the one finding measured as
  text-limited, where report vocabulary carries almost no information (sens 0.59
  at prec 0.57, base rate 0.47). Scoring 0.781 against near-noise labels is not
  learning synovitis. Cross-fold comparison, so suggestive rather than proven.
- **what I got wrong, plainly**: E017 was written as though the CV reversal
  settled the resolution question, with the submission described as
  confirmation. It was not confirmation, it was the test, and the test failed.
  The 14 GPU-hours of 288px fold training that E017 implied were worth
  considering would have been spent on the losing geometry.
- **what this changes about the plan**: the 192px configuration is the one that
  scores. The highest-confidence gain available is a **fold ensemble of that
  configuration** — same config, different splits, no new hypothesis that can be
  wrong — and completing 5 folds also produces the only offline selection signal
  left: one expert-scored out-of-fold prediction per gold study, n = 58.

### E020 — DINOv2 at 16 epochs: 0.6878, and the curve never flattened
- **date**: 2026-08-19
- **what changed**: backbone only. `vit_small_patch14_dinov2.lvd142m` on the
  **same v1 cache**, same fold, batch 6 × 3-step accumulation for an effective
  18 against the resnet34 run's 16. ImageNet normalisation on, which the
  resnet34 runs did not have.
- **result**: best val macro AUC **0.6878** at epoch 15 of 16, against
  resnet34's 0.7001 at epoch 23 of 24 on the same fold.
- **why that comparison is not a comparison**: the curve went 0.588 → 0.688 and
  **the last three epochs still added 0.002 each**. It never plateaued. The
  resnet34 run flattened by epoch 18 of 24 and spent its last six epochs moving
  0.0055. So 0.6878 is not a measurement of this backbone; it is where the clock
  stopped, and the clock was set to 16 because a ViT costs more per epoch.
  `knee-train-dinov2-long` continues it to 40.
- **what the per-finding profile suggests**, held loosely because it is
  cross-fold: DINOv2 reaches **Medial Meniscus 0.740** and **MCL 0.731** where
  the resnet34 fold-1 model reaches 0.656 and 0.669 — the focal findings, and
  the direction published work predicts for a self-supervised backbone
  (`COMPETITIVE_ANALYSIS.md` §3). It is much weaker on **Medial OA 0.561** and
  **Lateral OA 0.579**. Different folds, so this ranks nothing; it is a reason
  to finish the run rather than a result.
- **a bug this surfaced before it cost anything**: `knee-train-dinov2` was
  written when the ImageNet mean/std were *persistent* buffers, so its
  checkpoint carries two keys the current model rebuilds for itself. The resume
  path used a strict load and would have **refused the very run meant to
  continue it**. Inference already dropped those keys; resume now does too.
- **what it does not have**: a gold dump. It predates that output, so this
  backbone currently cannot be compared to the baseline on anything but CV,
  which `FINDINGS.md` §11 shows mis-ranks. The continuation run fixes that,
  which is a second reason to run it.

### E021 — the LLM labeler's first run failed on memory, and cost three minutes
- **date**: 2026-08-19
- **result**: **no result.** Every batch raised `OutOfMemoryError`, every study
  abstained, and the reported macro AUC was 0.500 — the value you get from
  ranking nothing. Not a measurement of the method.
- **the cause**: `device_map="auto"` put all of Qwen2.5-7B's fp16 weights on
  GPU 0 — **13.59 GiB of a 14.56 GiB card** — leaving under a gigabyte for
  activations, while the second T4 that `NvidiaTeslaT4` grants sat completely
  idle. Batch 32 at 3,000-token prompts then needed several GiB of KV cache.
- **what worked**: scoring the 58 gold studies **first** cost three minutes to
  learn the run was broken, against roughly an hour if the 4,349 had gone first.
  That ordering was the single best decision in the kernel's design and it paid
  off on its first use.
- **what did not work — a defect in the design, not the run**: the kernel
  treated "abstained on everything because the GPU ran out of memory" and
  "read the reports and chose states that rank badly" as the same outcome, and
  printed the same message for both. Those call for opposite responses — fix the
  run versus abandon the method — and the log said abandon. Fixed: an abstain
  rate above 50% now reports **RUN FAILED, not a verdict** and exits non-zero.
- **three fixes, in order of importance**:
  1. **Shard across both cards.** `max_memory` per device, 9 GiB each, so a 15 GiB
     model occupies both T4s and leaves real headroom. The idle second GPU was
     free capacity the whole time.
  2. **Halve the batch and retry on OOM** instead of abstaining. An
     out-of-memory error is a statement about batch size, not about the report,
     and abstaining discards real supervision for an infrastructure reason.
  3. **A worked example in the prompt**, showing all twelve findings in the exact
     output shape. vLLM's grammar-constrained decoding is not available — it is
     not on the Kaggle image and installing it would drag its own torch build
     onto a Turing card mid-session — so an example is the cheapest substitute,
     and the parser still abstains on anything it cannot read.
- **also corrected**: the results file recorded `RUN_MODEL` as the model, which
  named a 14B AWQ checkpoint the run never loaded. It now records what actually
  ran.

### E022 — local CPU dry run: the schema holds, the token budget was over-provisioned
- **date**: 2026-08-19
- **what it is**: the labeling kernel's whole path — chat template, generation,
  parse, states, ranks, macro AUC — run locally on **CPU** with
  `Qwen2.5-0.5B-Instruct` over 8 gold reports. Deliberately a plumbing test: a
  0.5B model's answers say nothing about what a 7B will score, but the last
  Kaggle run died on infrastructure and this catches that class of bug for a
  local minute instead of a GPU session.
- **the format holds completely**, which was the open question a worked example
  was added to address:

  | | count | share |
  |---|---:|---:|
  | unparseable or truncated outputs | 0 | 0% |
  | key absent from the object | 0 | 0% |
  | key present, value invalid | 0 | 0% |
  | key present, value `not_mentioned` | 60 | 62% |

  All twelve findings were emitted every time. So the 62% abstention is the
  **model's judgement**, not a schema failure — which is the distinction that
  decides whether to fix the prompt or change the model, and they are not the
  same problem.
- **the token budget was wrong in the safe direction**: prompts run 716–1,138
  tokens against a 3,000-token truncation limit, and completions 109–128 against
  a 220 cap. So the first run's out-of-memory came from **batch 32 on a single
  card**, not from long sequences. Batch raised 8 → 16 and the completion cap
  lowered 220 → 160, roughly halving the wall clock, with `run_batch`'s halving
  as the safety net if 16 is still too many.
- **the number to watch when the real run lands**: the lexicon labeler abstains
  on **40.6%** of gold findings. If the 7B abstains more than that, it is not
  obviously better regardless of what its AUC says, because abstention and AUC
  have tracked each other closely across every version of this labeler.
- **compliance note**: this reads report text into a *local* process. Nothing
  was printed but aggregates. `STRATEGY.md` rule 4 covers why that distinction
  is the whole difference.

### E023 — the LLM reader is a wash alone and worth +0.070 combined
- **date**: 2026-08-19
- **run**: `knee-llm-labeler`, Qwen2.5-7B-Instruct sharded across both T4s,
  3.5 min for the 58 gold studies (one automatic batch halving, 16 → 8).
- **standalone result**: macro AUC **0.7526** against the lexicon's 0.769, with
  a **48.7%** abstain rate against 40.6%. Worse on both headline numbers, and
  the kernel stopped there by its own gate.
- **that gate asked the wrong question.** "Does this beat the lexicon" is not
  the decision; "does adding this to the lexicon beat the lexicon" is. Scored on
  the same 58 studies with an identical convention, and with a combination rule
  that has **no free parameters** — exactly one labeler speaks, take it; both
  speak, average their ranks; neither, abstain:

  | labeler | macro AUC | 95% CI | abstain |
  |---|---:|---|---:|
  | lexicon | 0.7446 | [0.701, 0.787] | 39.7% |
  | LLM | 0.7421 | [0.689, 0.791] | 48.7% |
  | **union** | **0.8145** | [0.772, 0.853] | **30.2%** |

  Paired bootstrap on the same studies, which is about twice as sharp as
  comparing independent intervals (`FINDINGS.md` §13):

  | comparison | delta | 95% CI | verdict |
  |---|---:|---|---|
  | union − lexicon | **+0.0698** | [+0.041, +0.097] | higher |
  | union − LLM | **+0.0724** | [+0.033, +0.116] | higher |
  | LLM − lexicon | −0.0025 | [−0.060, +0.049] | **not separated** |

  Note the absolute numbers differ from the 0.769 recorded elsewhere because
  this scores abstentions at the bottom of the ranking rather than at a prior.
  All three rows share that convention, which is what makes the comparison
  valid; none of them is comparable to the earlier 0.769.
- **why they combine so well**: they fail on different findings. The LLM is far
  better on Medial Meniscus (+0.163), Medial OA (+0.102) and Lateral OA
  (+0.099); far worse on Fracture (−0.222), PF OA (−0.124), Contusion (−0.114)
  and Baker's (−0.095). Coverage explains most of it — the union leaves only
  30.2% of study×finding slots unsupervised against 39.7% and 48.7% — and
  coverage has tracked AUC across every version of this labeler.
- **a defect found while checking the two implementations against each other**:
  the first version ranked with `argsort().argsort()`, which breaks ties by
  array position. The lexicon emits only **2–4 distinct values per finding**, so
  ties are the common case, and that arbitrary order would have been baked into
  4,407 studies' training targets as noise. Averaging tied ranks *raised* the
  union to 0.8145 from 0.7994 and the gain to +0.070 from +0.049. The local
  analysis now calls the kernel's implementation rather than keeping a second
  copy, since a second copy is how they disagreed.
- **next**: re-run so the corpus is read, then rebuild `soft_labels.parquet`
  from the union and retrain. On this project's history — the imaging model
  scored 0.725 against a 0.769 teacher — a better teacher is the change most
  likely to move the board.

### E024 — 30 more epochs of 288px bought nothing, and the guards held
- **date**: 2026-08-19
- **runtime**: 3.6 GPU-hours.
- **result**: **no improvement.** Resumed at 0.7282 and never beat it. The best
  of the 30 new epochs was 0.7280 at epoch 34; from there it declined
  monotonically to **0.706** by epoch 59.
- **what this corrects**: E017 read the last three epochs (0.725, 0.727, 0.7282)
  as "still climbing" and called 0.7282 a floor. It was not a floor, it was the
  top. Thirty epochs of evidence say that curve had converged and the small
  rises at the end were noise. **A rising tail of three points is not a trend**,
  and this cost 3.6 GPU-hours to learn — on a geometry the board had already
  placed 0.037 behind.
- **both guards fired, and one of them mattered a lot**:
  - The run **inherited** the mounted checkpoint's 0.7282 as its best, so a
    continuation that never recovered exported the old weights rather than its
    own. Log: `inherited best macro AUC 0.7282 from the mounted checkpoint`.
  - The **best-epoch export** kept epoch 29's weights instead of the last
    epoch's. Under the previous behaviour this run would have shipped epoch 59
    at **0.706** — a 0.022 loss, silently, with a healthy-looking log.

  Final line: `best val macro AUC 0.7282 (epoch 29, and these are the weights
  saved)`. Between them the two guards turned a wasted run into a no-op instead
  of a regression, which is what they were built for.

### E025 — four runs land; the gold instrument starts working, and disagrees with CV
- **date**: 2026-08-19

| run | epochs | report-label CV | gold-subset AUC | n |
|---|---:|---:|---:|---:|
| `knee-train` (baseline, fold 0) | 24 | 0.7001 | — | — |
| **`v1pool`** — per-finding attention, fold 0 | 24 | **0.7088** | **0.7355** | 12 |
| `dinov2-long` — ViT-S/14, fold 0 | 40 | 0.7041 | **0.5890** | 12 |
| `fold2` (baseline) | 24 | 0.7410 | 0.7589 | 12 |
| `fold3` (baseline) | 24 | 0.7260 | 0.6793 | 11 |

- **DINOv2 has now converged, and it is not the answer here.** Continuing 16 → 40
  epochs moved it 0.6878 → **0.7041** and the last eight epochs are flat
  (0.700–0.704), so E020's "the clock stopped early" reading was right about the
  cause and wrong about the size: the extra 24 epochs bought 0.016, not the gap
  to the baseline. Against `v1pool` on **the same 12 gold studies** it scores
  0.5890 against 0.7355.
- **but that comparison is not conclusive, and the tooling says so**: paired
  bootstrap gives **+0.1465, 95% CI [−0.054, +0.295] — NOT SEPARATED**. Twelve
  studies cannot resolve a 0.15 gap. This is exactly what `FINDINGS.md` §13
  predicted from simulation (a single fold's gold subset has a ~0.173 interval),
  and it is the first time that prediction has been checked against real data.
- **CV and gold disagree about these two models, sharply.** CV puts them at
  0.7088 and 0.7041 — a 0.005 gap, effectively tied. Gold puts them 0.147 apart.
  Given §11 records CV mis-ranking on the one comparison that was checked
  against the board, gold is the one to believe; but at n=12 neither is
  actionable yet. Folds 1 and 4 are running and will take the pool toward 58.
- **per-finding attention is ahead on both metrics** — +0.0087 CV over the same
  fold with **one constant changed**, and the higher gold number — and its curve
  was still rising at epoch 23 of 24 (0.706, 0.708, 0.708, 0.709). Cheap to
  extend, and unlike the 288px case the rise is over eight monotone epochs
  rather than three.
- **the baseline's pooled gold OOF, folds 2+3**: **0.7078**, 95% CI
  [0.636, 0.779] over 23 studies. Published work reports gold-58 + 0.044 ≈ LB
  (`COMPETITIVE_ANALYSIS.md` §2); that would predict 0.752 against an actual
  0.725. Encouraging for the estimator, but 23 studies and a 0.143-wide interval
  cannot confirm a 0.027 discrepancy. Revisit at n = 58.

### E025 — the GPU budget ran out, and the gold instrument came online
- **date**: 2026-08-19
- **the block, correctly identified**: `Maximum weekly GPU quota of 30.00 hours
  reached.` Not the concurrency limit this project had been assuming — that one
  says `Maximum batch GPU session count of 2 reached` and is fixed by waiting.
  A quota exhaustion is not. `eda/push_queue.sh` now distinguishes them and
  exits instead of waiting forever, and `kaggle/README.md` records both.
  **CPU sessions are a separate allowance and still work**, verified by pushing
  a CPU kernel immediately after a GPU push was refused.
- **what the 30 hours bought**, from the fetched logs: 1.33 h fold 0, 1.33 h
  fold 1, 3.63 h the 288px run, **3.59 h its continuation which gained nothing**
  (E024), 1.84 h DINOv2, 3.74 h the labeler, plus folds 2–4, the pooling A/B,
  the DINOv2 continuation and two failed labeler attempts. The single largest
  avoidable item is the 3.59 h continuation of a geometry the board had already
  placed 0.037 behind.
- **six runs completed before the quota closed**:

  | run | best report-label CV | gold studies held out | gold AUC |
  |---|---:|---:|---:|
  | fold 1 | 0.7434 | 14 | 0.754 |
  | fold 2 | 0.7410 | 11 | 0.783 |
  | fold 3 | 0.7260 | 12 | 0.687 |
  | fold 4 | **0.7639** | 9 | 0.831 |
  | v1pool (per-finding attention) | 0.7088 | 12 | 0.742 |
  | dinov2-long (34 epochs) | 0.7041 | 12 | 0.644 |

- **the gold out-of-fold pool works** — folds 1–4 verified to share one
  configuration and one label file (all four logged `no per-finding confidence
  column`, so none saw the fused labels):

  **n = 46, macro AUC 0.7264, 95% CI [0.672, 0.776].**

- **and it lands on the leaderboard, not 0.044 above it.** The public writeup
  reports gold-58 predicting the board with a constant **+0.044** offset across
  three systems (`COMPETITIVE_ANALYSIS.md` §2). This configuration scored
  **0.725** on the board and **0.7264** out-of-fold on gold — an offset of
  **−0.001**. The published correction does not reproduce here.

  Held loosely for two reasons: the board score came from a fold-0 model while
  this pool is folds 1–4, and at n=46 the interval is 0.103 wide, so the offset
  is only pinned to about ±0.05. But the practical consequence is real and
  useful — **gold OOF appears to estimate this project's leaderboard score
  directly, with no correction**, and it costs nothing but the folds already
  being trained.

### E026 — the gold pool is complete at n=58, and it tracks the leaderboard
- **date**: 2026-08-20
- **how**: `knee-gold-eval` on **CPU**, 1.8 minutes, no GPU quota consumed. It
  filled the fold-0 hole that `knee-train` left by predating the gold dump.
- **result**: **n = 58, macro AUC 0.7201, 95% CI [0.672, 0.767].**
- **and it lands on the board.** This configuration scored **0.725** on the
  leaderboard. Gold out-of-fold says **0.7201** — an offset of **+0.005**. The
  published +0.044 correction (`COMPETITIVE_ANALYSIS.md` §2) does not reproduce
  here, and the practical consequence is better than it would have been: **gold
  OOF estimates this project's leaderboard score directly, with no correction**,
  for the cost of folds that were being trained anyway.
- **where the score actually is**, which the pooled number hides:

  | finding | gold AUC | positives |
  |---|---:|---:|
  | **Medial Meniscus** | **0.516** | 26 of 58 |
  | MCL | 0.612 | 9 |
  | Synovitis | 0.654 | 27 |
  | ACL | 0.662 | 24 |
  | PF OA | 0.672 | 21 |
  | Lateral Meniscus | 0.696 | 23 |
  | Lateral OA | 0.723 | 11 |
  | Contusion | 0.758 | 19 |
  | Fracture | 0.778 | 18 |
  | Medial OA | 0.817 | 15 |
  | Baker's | 0.830 | 12 |
  | Effusion | 0.924 | 35 |

  **Medial Meniscus is at chance** on the most common finding in the set. Every
  finding is 1/12 of the metric regardless of difficulty, so the weakest one
  costs exactly as much as the strongest earns.
- **the arithmetic that should drive everything from here**: lifting the worst
  four to 0.80 is worth **+0.063** and takes the macro to 0.783. Reaching the
  0.90 target requires **essentially every finding at 0.90** — there is no
  subset of easy wins that gets there.
- **both fold-0 experiments are NOT SEPARATED from the baseline**, paired on the
  same 12 studies:

  | comparison | delta | 95% CI |
  |---|---:|---|
  | per-finding attention pooling − baseline | +0.0142 | [−0.078, +0.114] |
  | DINOv2 − baseline | −0.1322 | [−0.303, +0.083] |

  Neither is established. DINOv2 looks materially worse and still cannot be
  ruled out at n=12, which is exactly the resolution limit measured in
  `FINDINGS.md` §13. Both need a full five-fold run to be settled, and that
  needs GPU quota.

### E027 — CORRECTED BELOW by E029. The teacher figures in this entry came from
a fusion that is not the one the training targets use, and they understate it.
The direction of the finding survives; the magnitudes do not.

### E027 — the model beats its teacher on diffuse findings and loses on focal ones
- **date**: 2026-08-20
- **how**: `eda/teacher_vs_model.py`, on CPU. Scores the report labelers and the
  imaging model's out-of-fold predictions against the **same 58 expert-labelled
  studies**, per finding, so the two are directly comparable.
- **the split is anatomical and it is clean**:

  | | model | teacher | |
  |---|---:|---:|---|
  | **focal** (ACL, MCL, both menisci, PF OA) | 0.632 | 0.798 | model is **worse** |
  | **diffuse** (effusion, OA, synovitis, Baker's, contusion, fracture) | 0.783 | 0.688 | model is **better** |

  Per finding, where the signal was in the targets and did not survive:

  | finding | teacher | model | lost |
  |---|---:|---:|---:|
  | Medial Meniscus | 0.744 | 0.516 | **−0.228** |
  | MCL | 0.820 | 0.612 | −0.208 |
  | PF OA | 0.828 | 0.672 | −0.156 |
  | ACL | 0.784 | 0.662 | −0.123 |

  And where the model **exceeds** its own supervision: Effusion 0.719 → 0.924,
  Lateral OA 0.534 → 0.723, Medial OA 0.719 → 0.817, Fracture 0.695 → 0.778.
- **what that rules out**: Medial Meniscus at 0.516 is not a hard problem being
  lost to. Its teacher scores 0.744 on the same studies, and the finding has 26
  positives of 58 — the most common in the set. The signal was **present in the
  targets and thrown away between the targets and the predictions.**
- **the mechanism it points at**: attention pooling produces a weighted MEAN
  over sixty slice embeddings. A meniscal tear is on about three of them; an
  effusion is on most. Averaging is the right operation for the second kind and
  the wrong one for the first, and the measurement splits exactly that way.
- **the change**: `FOCAL_K` keeps the **top-k slices per finding** alongside the
  weighted mean, and learns a per-finding blend between them, so a diffuse
  finding can keep the mean while a focal one reads off its few slices. Twelve
  extra parameters — one blend weight per finding.

  Verified: `FOCAL_K=0` reproduces the 0.725 configuration bit-for-bit, so every
  earlier number stays comparable. A single strong slice out of sixty moves the
  top-k path **8×** further than it moves the mean path. The blend starts at
  0.5, so neither path begins switched off.
- **what it is worth if it works**: closing only the four recoverable gaps is
  **+0.060 macro**, taking 0.720 to 0.780.
- **`knee-train-v1focal`** is the A/B, differing from the 0.725 baseline in
  exactly one constant, asserted by a test. **Blocked on GPU quota.**

### E028 — training does not need a GPU, if the backbone is frozen
- **date**: 2026-08-20
- **the measurement that changes the plan**, on four CPU threads:

  | | cost |
  |---|---|
  | fine-tune the whole model, 4,407 studies × 24 epochs | **191 hours** |
  | one frozen pass over the corpus, saved as embeddings | **2.2 hours, once** |
  | five folds × 24 epochs of the 73,380 parameters above it | **2.6 minutes** |

  The corpus of embeddings is 60 × 512 float16 per study — **0.27 GB total**.
- **why this matters more than it looks**: "we are blocked on GPU quota" was
  true of *fine-tuning* and quietly assumed of everything else. The questions
  actually open — does focal top-k pooling help, does per-finding attention
  help, do the fused labels help — are all about the part **above** the backbone,
  which is the part that costs 2.6 minutes. The blocked thing was never the
  thing that needed answering first.
- **what transfers and what does not.** A frozen backbone is not the fine-tuned
  model that scored 0.725, so absolute numbers from this rig do not predict the
  board. **Comparisons above the backbone do transfer**, because those are
  exactly what is being trained. A published system using a frozen DINOv2 with a
  trained head reached **0.776** on this leaderboard — above this project's
  fine-tuned 0.725 — so "frozen" is not automatically "worse" either.
- **positive control before trusting it**: run on synthetic embeddings with a
  focal signal planted on 2 slices of 20 for six findings and a diffuse signal
  spread across all 20 for the other six, the rig recovers
  **FOCAL_K=3 − baseline = +0.0445, 95% CI [+0.025, +0.064]**. A rig that could
  not find an effect deliberately put there would be worth nothing, and this one
  finds it.
- **`knee-embed`** runs the frozen pass on CPU; `eda/head_lab.py` runs the A/Bs.
  Every comparison is one-variable and reported as a paired interval rather than
  a score.


### E029 — the teacher was understated: two fusions in my own code
- **date**: 2026-08-20
- **what was wrong**: `eda/teacher_vs_model.py` implemented its own fusion of the
  two report readers — *prefer the lexicon, fall back to the model, never average
  where both speak* — while `eda/combine_labelers.py` and the labeling kernel use
  the real one: rank-normalise both, average where both speak. It also scored the
  teacher **only on the studies where it spoke** while scoring the model on all
  58, which is two different denominators inside one comparison.
- **corrected, with both sides on all 58 studies and the kernel's `fuse()`**:

  | finding | teacher (as reported) | teacher (correct) | model | lost |
  |---|---:|---:|---:|---:|
  | Medial Meniscus | 0.744 | **0.903** | 0.516 | −0.388 |
  | MCL | 0.820 | **0.918** | 0.612 | −0.306 |
  | ACL | 0.784 | **0.907** | 0.662 | −0.245 |
  | Lateral Meniscus | 0.813 | **0.858** | 0.696 | −0.163 |
  | Medial OA | 0.719 | **0.947** | 0.817 | −0.130 |

  | | as reported | correct |
  |---|---:|---:|
  | teacher macro | 0.734 | **0.814** |
  | recoverable gap | +0.060 | **+0.103** |
  | findings the teacher genuinely does not know | 4 | **1** (Synovitis) |
  | macro if the model merely matched its teacher | 0.792 | **0.835** |

- **what this changes, and it is not small**: advice given on the wrong numbers
  said *"to reach 0.90 the model must beat its teacher on all eight weakest
  findings, and the teacher is below 0.897 on every one."* Correctly fused,
  **four teachers are already at or above 0.897** — ACL 0.907, MCL 0.918, Medial
  Meniscus 0.903, Medial OA 0.947. On those the model does not need to exceed
  its supervision at all; it needs to stop throwing it away.
- **what survives**: the direction. The model still loses on focal findings and
  wins on diffuse ones, and Medial Meniscus at 0.516 against a teacher now
  measured at **0.903** is a starker version of the same story, not a weaker one.
- **the lesson, which is the same one as three entries ago**: a second
  implementation of a shared idea is where the error lives. `to_rank` was
  duplicated once and broke on ties; `fuse` was duplicated once and understated
  the teacher by 0.08. Both scripts now call the kernel's implementation.


### E030 — the rig answers all three open questions, and only the labels survive
- **date**: 2026-08-21
- **what changed**: nothing in the model. `knee-embed` finished — one frozen
  DINOv2 ViT-S/14 pass over the corpus, **344.6 min of CPU, 4.69 s/study**,
  written as `embeddings.npy (4407, 60, 384) float16`, 203 MB — so the three
  questions that had been waiting on the GPU quota were run as paired,
  one-variable A/Bs above that frozen backbone.
- **config**: `eda/head_lab.py`, 5-fold grouped on the scanner fingerprint (178
  groups), 24 epochs, batch 64, LR 1e-3, scored out-of-fold against the 58
  expert-labelled studies. 2,000-sample paired bootstrap.
- **runtime**: **8.5 min on CPU for all six configurations.** No GPU quota spent.
- **results**, each against the same baseline (lexicon labels, mean pooling,
  single attention map):

  | comparison | delta | 95% CI | verdict |
  |---|---:|---|---|
  | focal top-k (k=3) − baseline | +0.0060 | [−0.041, +0.051] | not separated |
  | per-finding maps − baseline | +0.0389 | [−0.009, +0.090] | not separated |
  | **fused labels − lexicon** | **+0.0508** | **[+0.001, +0.102]** | **A is better** |

- **seed replication of the one that separated**, because its interval only
  barely excluded zero:

  | seed | fused | lexicon | delta | 95% CI |
  |---|---:|---:|---:|---|
  | 0 | 0.6542 | 0.6034 | +0.0508 | [+0.001, +0.102] |
  | 1 | 0.6738 | 0.5900 | +0.0838 | [+0.035, +0.134] |
  | 2 | 0.6543 | 0.6043 | +0.0501 | [−0.011, +0.111] |
  | **mean** | | | **+0.0616** | direction consistent 3/3 |

  Two of three intervals exclude zero and all three point the same way at a
  stable magnitude. That is *supporting* evidence, not proof — n=58 with a
  ±0.05 interval cannot deliver proof — but it is the first offline signal for
  the fused labels that does not come from the teacher's own scoring.
- **what it means**:
  - **The fused labels are the right next GPU run.** They were already the top
    of the queue on the strength of the teacher improving +0.070 on gold. That
    argument was about label quality in isolation; this one measures what a
    trained head actually does with them, and it agrees.
  - **Focal top-k does not survive contact with a measurement.** `docs/PATH.md`
    carried it as "+0.060 of measured headroom". The +0.060 was a *ceiling* —
    the gap between the model and its teacher on focal findings — and not a
    prediction of what top-k pooling recovers. Measured, top-k recovers
    **+0.006 with an interval eight times its width.** The rig's positive
    control (E028) plants a focal signal and recovers +0.0445, so the rig can
    see this kind of effect when it is there. It is not there.
  - **Per-finding maps remain unseparated**, now at n=58 rather than n=12.
    +0.0389 with the interval crossing zero is the same verdict as before with
    a wider base to say it on.
- **the pattern this repeats**: a quantity measured as *headroom* was carried
  forward as though it were a *gain*. That is the sixth entry in this log where
  a number changed meaning between where it was measured and where it was used.
- **next**: `kaggle/21_train_v1fused` on a T4 the moment the weekly quota
  resets. The push was attempted on 2026-08-21 and refused with "Maximum weekly
  GPU quota of 30.00 hours reached", so the reset had not happened by then.


### E031 — the fused labels train, and the direction holds a third time
- **date**: 2026-08-22
- **what changed**: the label file, and nothing else. `RUN_FOLD=0`,
  `TARGET_SIZE=192`, `resnet34`, 24 epochs, batch 16, LR 6e-4, `FOCAL_K=0`,
  `PER_FINDING_POOL=False` — byte-identical to the configuration that scored
  **0.725** on the board, with `knee-phase1-fused` mounted in place of
  `knee-phase1-artifacts`.
- **runtime**: **88.8 min wall clock** on 2×T4 (DataParallel), ~1.5 h of the
  weekly 30. The quota reset between 2026-08-21 18:17 UTC (refused) and
  2026-08-22 00:17 UTC (accepted).
- **CV**: best val macro AUC **0.7350** at epoch 19, against **fused** report
  labels.
- **gold (12 held-out expert studies in this fold)**: **0.7933**, peaking at
  0.7996 at epoch 18.
- **prediction spread**: 0.1904 at the saved epoch, rising monotonically from
  0.0273 — no collapse to base rates.

- **the CV number is not a valid A/B and must not be quoted as one.** The
  lexicon baseline's 0.7001 was scored against *lexicon* report labels; this
  0.7350 is scored against *fused* ones. The targets changed, so the two
  numbers have **different denominators** — the identical mistake E029 caught in
  `teacher_vs_model.py`. `0.7350 − 0.7001 = +0.035` measures nothing.
- **the gold comparison is the valid one**, paired on the same 12 studies
  against the fold-0 lexicon baseline from `knee-gold-eval` (E026):

  | | macro AUC vs expert | 95% CI |
  |---|---:|---|
  | fused labels | **0.7933** | [0.638, 0.888] |
  | lexicon labels | 0.7213 | [0.548, 0.840] |
  | **paired difference** | **+0.0721** | **[−0.009, +0.183]** |

  **NOT SEPARATED — the interval contains zero.** At n=12 the paired interval is
  ~0.19 wide, and `FINDINGS.md` §13 measured that limit by simulation before
  this run existed. A +0.072 point estimate cannot be called established here no
  matter how much the direction is liked.

- **where it moved**, and the shape is the interesting part:

  | finding | lexicon | fused | delta | pos |
  |---|---:|---:|---:|---:|
  | **Medial Meniscus** | 0.281 | **0.656** | **+0.375** | 8 |
  | Contusion | 0.800 | 1.000 | +0.200 | 5 |
  | Fracture | 0.714 | 0.886 | +0.171 | 7 |
  | Baker's | 0.850 | 1.000 | +0.150 | 2 |
  | MCL | 0.550 | 0.650 | +0.100 | 2 |
  | PF OA | 0.704 | 0.778 | +0.074 | 3 |
  | Effusion | 0.950 | 1.000 | +0.050 | 10 |
  | ACL / Medial OA / Lateral OA | — | — | 0.000 | 7/1/3 |
  | Lateral Meniscus | 0.833 | 0.778 | −0.056 | 6 |
  | **Synovitis** | 0.600 | **0.400** | **−0.200** | 7 |

  **Medial Meniscus is the headline.** E026 measured it at **0.516 pooled over
  all 58 — at chance on the most common finding in the set** — and it is the
  single largest drag on the macro. Here it moves +0.375 on 8 positives.
  Synovitis moving −0.200 is the one that argues against reading too much into
  this: it is also the one finding E029 flagged as genuinely unknown to the
  fused teacher, so a loss there is consistent rather than anomalous, but a
  12-study fold produces swings of this size from resampling alone.

- **what it means**: three independent lines now point the same way at the same
  magnitude — teacher **+0.070** on the 58 (E029), frozen-embedding rig
  **+0.062** across three seeds (E030), fold-0 gold **+0.072** paired (here).
  None is individually conclusive and the first two are not measurements of this
  model. But they are three different instruments, and they agree to within
  0.010. That is the strongest case this project has assembled for any change,
  and it is still not proof.
- **what would settle it**: folds 1–4 on fused labels, ~6 GPU-hours, taking the
  paired gold comparison from n=12 to **n=58 and the interval from ~0.19 to
  ~0.044** (`FINDINGS.md` §13). That resolution *would* separate a +0.072 gap.
  A board submission (0.8 h) is the other decisive option and is ground truth,
  directly comparable to the standing 0.725 since that was also one fold.
- **next**: spend the folds. This is the one configuration that has earned GPU
  hours on evidence rather than on hope.


### E032 — the fused labels separate at n=58: +0.0717, CI [+0.042, +0.103]
- **date**: 2026-08-22
- **what changed**: nothing since E031 except the number of folds. Folds 1–4 of
  the `v1fused` lineage, declared in `src/pipeline.py` so they inherit the
  lineage's config and label dataset by construction. `RUN_FOLD` is verifiably
  the only difference between these `run.py` files and the fold-0 one.
- **runtime**: 82.4 / 79.7 / 76.3 / 84.6 min, two at a time against the GPU
  concurrency cap of 2. **~5.4 GPU-hours**, ~6.9 including fold 0.

- **the measurement this was run to make**, paired on all 58 gold studies
  against the lexicon-label folds:

  | | macro AUC vs expert | 95% CI |
  |---|---:|---|
  | **fused labels, 5 folds** | **0.7918** | [0.754, 0.829] |
  | lexicon labels, 5 folds | 0.7201 | [0.672, 0.767] |
  | **paired difference** | **+0.0717** | **[+0.042, +0.103]** |

  **A is better — the interval excludes zero.** This is the first change in the
  project to clear that bar offline.

- **the baseline reproduces exactly.** Pooled lexicon gold comes back at
  **0.7201**, matching E026 to four decimals from independently re-fetched
  outputs. The comparison is not resting on a remembered number.

- **four instruments, one answer.** Every estimate of this effect, from
  measurements that share no code path:

  | instrument | delta | n |
  |---|---:|---|
  | teacher on the 58 (E029) | +0.070 | labels only |
  | frozen-embedding rig, 3 seeds (E030) | +0.062 | 4,407 |
  | fold-0 gold, paired (E031) | +0.0721 | 12 |
  | **five-fold gold, paired (here)** | **+0.0717** | **58** |

  The spread across all four is 0.010. E031's point estimate survived a
  five-fold increase in sample size essentially unchanged, which is the
  behaviour of a real effect rather than a favourable draw.

- **where it moved**:

  | finding | lexicon | fused | delta | pos |
  |---|---:|---:|---:|---:|
  | **Medial Meniscus** | 0.516 | **0.786** | **+0.270** | 26 |
  | ACL | 0.662 | 0.812 | +0.151 | 24 |
  | Baker's | 0.830 | 0.964 | +0.134 | 12 |
  | Medial OA | 0.817 | 0.929 | +0.112 | 15 |
  | Fracture | 0.778 | 0.883 | +0.106 | 18 |
  | Contusion | 0.758 | 0.846 | +0.088 | 19 |
  | Lateral Meniscus | 0.696 | 0.753 | +0.057 | 23 |
  | MCL | 0.612 | 0.628 | +0.016 | 9 |
  | Lateral OA | 0.723 | 0.721 | −0.003 | 11 |
  | PF OA | 0.672 | 0.658 | −0.014 | 21 |
  | Effusion | 0.924 | 0.906 | −0.019 | 35 |
  | **Synovitis** | 0.654 | **0.616** | **−0.037** | 27 |

  **Medial Meniscus was the whole problem and is now not.** E026 measured it at
  **0.516 — chance — on the most common finding in the set**, and named it the
  single largest drag on the macro. It is now 0.786. Findings at or above 0.80
  go from **3 to 6**; findings below 0.70 go from **6 to 3**.

  **Synovitis moves the wrong way**, as it did on fold 0, and at n=58 that is
  no longer dismissible as fold noise. E029 identified Synovitis as the one
  finding the fused teacher genuinely does not know, so a label change that
  helps everywhere else and hurts here is the predicted shape, not an anomaly.
  It is now the second-weakest finding and inherits Medial Meniscus's old role
  as the thing most worth fixing.

- **what this predicts for the board, and the caveat on it.** E026 established
  that gold OOF tracks this project's leaderboard to **+0.005** with no
  correction. Taken at face value that puts this configuration near **0.787**
  against a standing **0.725**. That relationship was calibrated on **one**
  submission, so it is a single-point calibration being asked to extrapolate
  0.07 beyond where it was fitted. Treat 0.787 as an expectation to be tested,
  not a result — the board is the only ground truth and it is two clicks away.
- **next**: submit. `knee-infer-v1fused` now depends on all five checkpoints, so
  it is a 5-fold rank-mean and costs ~0.8 h. That converts the strongest offline
  result this project has produced into a number on the board that cannot be
  argued with.


### E033 — the cross-label ensemble is worse than its better half, at every weight
- **date**: 2026-08-22
- **what changed**: nothing was trained. This blends two model families that
  already exist — the fused 5-fold (E032) and the lexicon 5-fold — by
  rank-averaging their gold out-of-fold predictions, exactly as
  `22_infer_v1fused` rank-averages folds. **Cost: seconds of CPU, zero quota,
  zero submissions.**
- **why it looked promising**: `PATH.md` Phase D carries a published +0.02 to
  +0.05 for blending independent families, and these two share a backbone and a
  cache but differ in supervision, which is the axis the fusion was built on.

| blend | gold macro AUC at n=58 |
|---|---:|
| lexicon alone | 0.7201 |
| rank blend, w_fused = 0.5 | 0.7740 |
| rank blend, w_fused = 0.7 | 0.7853 |
| rank blend, w_fused = 0.8 | 0.7902 |
| rank blend, w_fused = 0.9 | 0.7911 |
| **fused alone** | **0.7918** |

- **result**: **every blend is below fused alone, and the curve rises
  monotonically toward w=1.** The best blend tested (0.7) is −0.0065 against
  fused alone, CI [−0.017, +0.004] — not separated, but pointing down, and the
  monotonicity is the real evidence: there is no interior optimum. The
  optimiser's answer is "use none of the lexicon model".
- **what it means**: blending helps when members are *comparably strong and
  decorrelated*. These are neither — the lexicon model is 0.072 worse and was
  trained on labels that are a strict subset of the fused ones' information, so
  it contributes noise where it disagrees rather than an independent view.
  Phase D's +0.02–0.05 assumed independent *families*; two label sets on one
  backbone is not that.
- **what it saved**: one of two daily submissions, and the temptation to read a
  published ensemble gain as applying here. The check cost seconds because the
  gold OOF predictions already existed.
- **next**: a genuine ensemble needs a second *architecture*, not a second label
  set. That is `PATH.md` Phase C — DINOv2 to convergence — and it costs ~23
  GPU-hours for five folds, which is the entire remaining weekly allowance.


### E034 — 0.846 on the board, and the gold-to-board calibration does not hold
- **date**: 2026-08-22
- **what**: `knee-infer-v1fused` v1, the 5-fold rank-mean of the fused-label
  models from E031/E032, submitted to the competition. Kaggle re-ran the
  notebook against the ~1,300 hidden studies.
- **result**: **public leaderboard 0.846**, from a standing **0.725**.
  **+0.121 in one day**, the largest move this project has made by a wide
  margin, and the second-largest single lever after the pixels themselves.

| submission | date | board |
|---|---|---:|
| constant priors | 08-18 | 0.500 |
| scanner metadata, no pixels | 08-18 | 0.531 |
| imaging, 192px, 1 fold, lexicon labels | 08-19 | 0.725 |
| imaging, 288px, effective batch 16 | 08-19 | 0.688 |
| **imaging, 192px, 5-fold, fused labels** | **08-22** | **0.846** |

- **THE CALIBRATION IS CONTRADICTED, and this is the more important half of the
  entry.** E026 measured gold OOF against the board on one model and concluded
  "**gold OOF estimates this project's leaderboard score directly, with no
  correction**". Two points now exist:

  | model | gold OOF (n=58) | board | offset |
  |---|---:|---:|---:|
  | lexicon, 192px | 0.7201 | 0.725 | **+0.005** |
  | fused, 192px 5-fold | 0.7918 | **0.846** | **+0.054** |

  The offset moved by **an order of magnitude** between two models of the same
  architecture on the same cache. Gold OOF **understated** the board both times,
  so it is not biased in a dangerous direction — but it is **not a calibrated
  predictor** and the "no correction needed" claim was, once again, a
  relationship measured at n=1 and generalised. It ranked the two models
  correctly, which is what it is actually good for.
- **the prediction made before submitting was 0.787, and it was wrong by
  −0.059.** That was stated in advance with the reason it might fail — a
  single-point calibration extrapolating 0.07 past where it was fitted — and the
  failure arrived in exactly that form, in the favourable direction. Recorded so
  the next forecast is not made with more confidence than this one earned.
- **two plausible reasons the five-fold gains more on the board than on gold**,
  neither established:
  1. **The gold 58 are a harder, non-representative slice.** Every gold study is
     one a model never saw in its own fold, and they are only 1.3% of the
     corpus. The hidden 1,300 may simply be easier.
  2. **Rank-mean ensembling helps the board more than the gold measurement can
     see.** Gold OOF scores each study with the *single* fold that held it out,
     so it measures a **1-model** system. The submission rank-averages **5**.
     The 4-fold ensemble gain was never measured, and this is the first
     submission that carries it — so E032's +0.0717 and the board's +0.121 are
     not measuring the same system, and the difference is the ensemble.

  **Reason 2 is testable and matters**: if most of the +0.121 is ensembling
  rather than labels, the lexicon 5-fold would also have jumped, and every
  future single-fold gold comparison is understating what its ensemble will do.
- **next**: the honest follow-up is to submit the **lexicon 5-fold** — the
  models already exist, it costs 0.8 h and one of five daily submissions, and it
  separates the label effect from the ensemble effect on ground truth. Without
  it, "+0.121 from better labels" is an attribution nobody has earned.


### E035 — the teacher is no longer the ceiling, and the gold set has run out of resolution
- **date**: 2026-08-22
- **what**: a diagnostic, not a run. Zero quota. Scores the **fused teacher**
  and the **fused 5-fold model** on the same 58 expert studies, per finding,
  with a 2,000-sample paired bootstrap, to split the remaining gap into work
  that is recoverable by modelling and work that is not.

| finding | teacher | model | model − teacher | 95% CI | pos | teacher spoke |
|---|---:|---:|---:|---|---:|---:|
| MCL | 0.884 | 0.628 | **−0.256** | **[−0.456, −0.069]** | 9 | 86% |
| PF OA | 0.765 | 0.658 | −0.107 | [−0.292, +0.074] | 21 | 66% |
| Medial Meniscus | 0.889 | 0.786 | −0.103 | [−0.264, +0.042] | 26 | 88% |
| Lateral Meniscus | 0.817 | 0.753 | −0.064 | [−0.239, +0.106] | 23 | 81% |
| ACL | 0.863 | 0.812 | −0.050 | [−0.196, +0.090] | 24 | 91% |
| Medial OA | 0.935 | 0.929 | −0.006 | [−0.081, +0.067] | 15 | 53% |
| Lateral OA | 0.708 | 0.721 | +0.013 | [−0.189, +0.221] | 11 | 47% |
| Contusion | 0.773 | 0.846 | +0.074 | [−0.066, +0.218] | 19 | 72% |
| Synovitis | **0.520** | 0.616 | +0.097 | [−0.060, +0.261] | 27 | 36% |
| Effusion | 0.750 | 0.906 | +0.156 | [−0.004, +0.312] | 35 | 97% |
| Baker's | 0.799 | 0.964 | **+0.165** | **[+0.026, +0.316]** | 12 | 78% |
| Fracture | 0.644 | 0.883 | **+0.239** | **[+0.037, +0.434]** | 18 | 43% |
| **MACRO** | **0.7788** | **0.7918** | **+0.0126** | **[−0.042, +0.066]** | | |

- **"The teacher bounds everything" is no longer supported.** That claim runs
  through `PATH.md` Phase A and `STATUS.md`, and it was true when a 0.769
  teacher produced a 0.725 model. The fused model now scores **+0.0126 above its
  own teacher at the macro** — **not separated**, interval spanning zero, so the
  honest statement is *the model is no longer measurably behind its teacher*,
  not that it has beaten it. Either way, "raise the teacher and the model
  follows" has stopped being the obviously correct next move.
- **The model beats the teacher outright on two findings**: Fracture **+0.239**
  and Baker's **+0.165**, both intervals excluding zero. On Fracture the teacher
  is at 0.644 and speaks for only 43% of studies — the reports barely carry it
  and **the pixels do**. This is direct evidence that imaging supervision from
  noisy labels can exceed those labels, which the label-ceiling framing did not
  anticipate.
- **Exactly one recoverable modelling loss separates: MCL, −0.256.** Recovering
  it in full is worth **+0.021 macro** — and it rests on **9 positives**, so
  even it is fragile. The menisci, ACL and PF OA all point down, sum to another
  +0.027 if real, and **not one of them separates**.
- **and that is the actual finding here: the gold set is out of resolution.**
  Per-finding intervals at n=58 are **±0.2**. Nine of twelve findings cannot be
  told apart from their teacher at all. `FINDINGS.md` §13 measured the *macro*
  limits by simulation; this is the per-finding version, and it is far worse
  because each finding sees only its own 9–35 positives.

  **The consequence is strategic.** The board now carries ~1,300 studies and
  allows **5 submissions a day** (§2.10, corrected 2026-08-22). Gold-58 has been
  the development instrument because submissions were believed to be scarce.
  They are not, and for per-finding questions the board has ~20x the sample. The
  gold set should be demoted to what it is still good at — **ranking whole
  models cheaply before spending a submission** — and per-finding diagnosis
  should move to the board.
- **where the remaining headroom actually is**, given the above:
  - **Synovitis**: teacher at **0.520 — chance** — speaking for 36% of studies.
    Not a modelling problem; the reports do not carry it. A third reader is the
    only lever, and it is CPU-only.
  - **Lateral OA**: teacher 0.708 at 47% coverage. Same shape, smaller.
  - **MCL**: the one confirmed modelling loss.
  - **Everything else**: below the resolution of the instrument being used.
- **next**: not more gold-set optimisation. `PATH.md` Phase C — DINOv2 to
  convergence, five folds, ~23 GPU-h — is the largest untried lever and costs
  exactly the remaining weekly quota. Phase D (rank-blending it with the
  resnet34 family) then costs nothing at training time. E033 showed blending two
  *label sets* fails; blending two *architectures* is the version with published
  support behind it.


### E036 — the control attributes the jump, and Phase C's premise is contradicted
- **date**: 2026-08-23

#### Part 1 — the control, and it is good news

| system | board |
|---|---:|
| lexicon labels, 192px, **1 fold** | 0.725 |
| lexicon labels, 192px, **5-fold rank-mean** | **0.757** |
| fused labels, 192px, **5-fold rank-mean** | **0.846** |

The +0.121 from E034 decomposes cleanly, on ground truth, with architecture and
cache held fixed:

| component | delta | share |
|---|---:|---:|
| **ensembling** (1 fold → 5 folds) | **+0.032** | 27% |
| **labels** (lexicon → fused) | **+0.089** | 73% |

**Labels are the dominant lever in this competition, by 3:1**, and E032's
offline +0.0717 was measuring the smaller of the two components while the board
saw both. The attribution that E034 flagged as unearned is now earned.

#### Part 2 — DINOv2 does not work, and the reason it was tried was wrong

`PATH.md` Phase C rested on: *"DINOv2 reached 0.7041 at epoch 34 and had NOT
flattened — that number is where the clock stopped, not where the backbone
converges."* Run properly to 40 epochs on fused labels, fresh from pretrained:

| | fold 0 | fold 1 |
|---|---:|---:|
| best val (report labels) | 0.7099 @ **epoch 23** | 0.7130 @ **epoch 32** |
| val at epoch 39 | 0.6913 | 0.7071 |
| gold at epoch 30 → 39 | 0.6664 → **0.5948** | 0.7329 → 0.7171 |
| runtime | 298 min | 272 min |

**It peaks and then decays.** Fold 0 tops out at epoch 23 and loses 0.019 of val
and 0.072 of gold by epoch 39. The "still climbing at 34" reading was noise on a
truncated run, and **the single largest published lever in the plan was chosen
on it.** That is the eighth claim in this project overturned by a measurement,
and the most expensive: ~20 GPU-hours.

Against the resnet34 fused models on the same studies: **0.7151 vs 0.7888
gold** — DINOv2 is **0.074 worse alone**.

#### Part 3 — and the blend does not rescue it

Rank-blending the two architectures on gold OOF, n=26 (folds 0+1 of each):

| blend | gold macro |
|---|---:|
| dinov2 alone | 0.7151 |
| w_resnet = 0.5 | 0.7635 |
| w_resnet = 0.7 | 0.7893 |
| **w_resnet = 0.85** | **0.7910** |
| **resnet34 alone** | **0.7888** |

**+0.0022 at the best weight, 95% CI [−0.012, +0.016] — not separated.**

Unlike E033 there *is* a weak interior optimum, which is what a genuinely
decorrelated second family should produce — so the mechanism is real and the
magnitude is not. **Even the optimistic end of that interval, +0.016, does not
approach the +0.054 needed for 0.90.** Phase D's published +0.02–0.05 assumed
members of comparable strength; a member 0.074 behind contributes almost
nothing however uncorrelated it is.

- **folds 2 and 3 could not be stopped.** Kaggle exposes no cancel through the
  API or CLI — only `delete`, which destroys the kernel and its history. They
  ran to completion, taking the week's spend to **~28.7 h of 30**.
- **what this costs and what it buys**: ~20 GPU-hours for a negative result on
  the plan's largest lever, and a 4-fold DINOv2 family that can be blend-tested
  at better power than n=26 before being abandoned or kept as a small
  contributor.
- **next**: the lever that actually moved the board is **labels, +0.089**, and
  E035 already located the remaining label headroom precisely — Synovitis with a
  teacher at **0.520, chance, covering 36% of studies**, then Lateral OA and PF
  OA. That work is **CPU-only and costs no quota**, which is what is left of
  this week anyway.


### E037 — a cue was negating the term it judged; Synovitis is still not fixable
- **date**: 2026-08-23. CPU only, no quota.
- **the task**: E035 named Synovitis the best remaining label lever — teacher at
  **0.520, chance**, covering 36% of gold studies, model's second-weakest
  finding. This is what came of trying to fix it.

#### The bug, which is not a Synovitis bug at all
`ReportLabeler._compile` anchored the **start** of a cue and not the end:

```python
re.compile(r"(?<![^\W\d_])(?:" + "|".join(parts) + r")", ...)   # no trailing anchor
```

So a cue matched the **prefix of the very term it was judging**. Spanish
*sinovitis* begins with *sin* — "without" — and the cue negated the mention:
**8 of 8 Spanish Synovitis mentions negated, 6 of them expert-positive.** That
is the anti-correlation behind Synovitis scoring below chance: where the lexicon
said *negative*, 5 of 7 studies were expert-**positive**.

It was never language-specific. English *not* matched *noted*; *no* matched
*nodular*. **166 mention-decisions across all 12 findings and 5 languages** were
decided this way on the 58 gold studies alone.

**The fix is not simply "anchor both ends".** `_compile` is shared with the
finding *terms*, which need prefix matching for plurals and Turkish
agglutination — anchoring those too drops mean coverage from **60.3% to 54.0%**
and the macro to 0.7388. The anchor is now opt-in and set only for cues.

| | teacher macro on the 58 | mean coverage |
|---|---:|---:|
| before | 0.7523 | 60.3% |
| both ends anchored everywhere (**wrong**) | 0.7388 | 54.0% |
| **cues only (shipped)** | **0.7578** | **60.3%** |

Per finding: MCL **+0.040**, Synovitis **+0.036**, Effusion +0.022, Medial
Meniscus +0.022, ACL **−0.031**.

**Macro +0.0055, 95% CI [−0.0094, +0.0223] — not separated**, better in 73.4% of
resamples. **The fix ships because a cue negating its own term is indefensible,
not because the gain is established.** Four regression tests assert both halves:
that "sinovitis" is no longer self-negated, and that "sin"/"not" still negate as
whole words, and that terms still match inflected forms.

#### Synovitis vocabulary: one honest negative and one small win
- **Domain-guessed terms moved nothing.** Eleven MOAKS-vocabulary terms —
  *effusion-synovitis*, *hoffitis*, *fat pad edema*, *synovial enhancement* —
  committed before measuring, then measured once: **zero change**, on the 58 and
  on coverage. The corpus does not use those words. Guessing vocabulary from
  domain knowledge failed; this is recorded rather than quietly deleted.
- **Corpus mining found what guessing missed.** Enrichment analysis over all
  4,407 reports (unsupervised — no expert labels touched) surfaced **`synovium`**:
  120 reports, exactly 1 without another synovitis stem. The prefix term
  *synovial* cannot reach it, so it was invisible. Corpus-wide Synovitis
  supervision goes **603 → 621 studies (13.7% → 14.1%)**. Zero on the 58, which
  are a 1.3% sample and contain none of them.

#### Why Synovitis stays broken, and it is not the lexicon's fault
Mention rates for a synovitis stem across the corpus:

| language | reports | mention synovitis |
|---|---:|---:|
| en | 1,735 | 20.6% |
| es | 682 | 16.4% |
| el | 321 | 9.3% |
| de | 262 | 9.5% |
| tr | 546 | **4.0%** |
| nl | 153 | 3.9% |
| hr | 330 | **1.2%** |
| bg | 220 | **0.5%** |

**Radiologists in this corpus mostly do not report synovitis.** 14.1% coverage
is not a vocabulary gap that more terms will close.

**And the imaging already knows better than the text.** E035 measured the model
at **0.616** on Synovitis against a teacher at 0.520 — the model *beats* its
teacher here, as it does on Fracture and Baker's, exactly where the reports are
thinnest. Raising a 0.569 teacher has little room to help a 0.616 student.

- **conclusion**: Synovitis is **not a label problem that can be fixed from the
  reports.** The remaining lever for it is imaging, not text. E035's ranking of
  Synovitis as the top label opportunity was right about *where* the weakness is
  and wrong about *what fixes it*.
- **next**: the cue fix touches every finding and every language, so the fused
  labels should be rebuilt and the 5-fold retrained on them before anything else
  is concluded from it. That is ~7 GPU-hours and this week's quota is spent.


### E038 — why gold OOF mis-forecasts, and a laterality theory that died
- **date**: 2026-08-23. CPU only, no quota.

#### The forecaster is repairable, and the control is what repairs it
E034 recorded the gold-to-board offset as **contradicted**: +0.005 on one model,
+0.054 on the next. E036's control supplies the missing point and the mechanism.

**Gold OOF always scores ONE model per study** — the single fold that held it
out. A submission **rank-averages five**. Gold OOF is structurally blind to
ensembling, which is why its offset grew:

| system | gold OOF | board | offset |
|---|---:|---:|---:|
| lexicon, 1 fold | 0.7201 | 0.725 | +0.005 |
| lexicon, **5-fold** | 0.7201 | 0.757 | **+0.037** |
| fused, 5-fold | 0.7918 | 0.846 | +0.054 |

The lexicon gold OOF is **the same number** for the 1-fold and 5-fold systems —
0.7201 — while the board moved +0.032. The offset grew by exactly the ensembling
gain. That is not a coincidence, it is the definition of what gold OOF cannot
see.

**Corrected: `board ≈ gold_OOF + ensembling_gain + 0.005`**, with
`ensembling_gain = +0.032` measured directly on the board.

| system | uncorrected error | corrected error |
|---|---:|---:|
| lexicon, 1 fold | +0.000 | +0.000 |
| lexicon, 5-fold | **−0.032** | +0.000 |
| fused, 5-fold | **−0.049** | **−0.017** |

Two of three points are fitted by construction, so only the fused row is a real
test: **worst-case error 0.049 → 0.017**. Still not a precision instrument, and
it now has a *stated mechanism* rather than an empirical constant, which is why
it is worth more than the old one-point fit. The honest use remains: rank
models, and forecast only with the interval attached.

#### A laterality theory, tested and refused
The cache build mirrors right knees so medial is always the same side, and the
header table looked alarming — **more than half of all series carry no
laterality tag** (7,272 blank, 5,105 NaN, against 11,914 R/L), where the
docstring claims 79%. If that propagated to studies it would scramble
medial-versus-lateral, which is exactly where the model loses most to its
teacher (MCL −0.256, the menisci, PF OA).

**It does not propagate.** Resolved per *study*, using the sibling-series
fallback the builder already implements:

| | studies | share |
|---|---:|---:|
| laterality resolved | 4,314 | **97.8%** |
| no series in the study carries it | **0** | 0.0% |
| **series within a study disagree** | **96** | **2.2%** |

The theory is dead: MCL's deficit is not a mirroring artefact. **The 2.2% that
disagree are a real if small defect** — a study whose sagittal and coronal
series report opposite sides gets mirrored inconsistently across planes, so its
volume is internally incoherent. 96 studies is too few to explain anything at
n=58 gold, and worth fixing when the cache is next rebuilt.

- **the point of recording a dead theory**: it was checked in ten minutes of CPU
  before anything was rebuilt on it. The 79% figure in the docstring is also now
  known to be wrong at the series level, which is how the theory got started.


### E039 — Phase D closed, and the corrected labels are live
- **date**: 2026-08-23

**The DINOv2 blend, at 4-fold power (n=49) rather than E036's n=26:**

| blend | gold macro |
|---|---:|
| dinov2 alone | 0.7139 |
| w_resnet = 0.5 | 0.7675 |
| w_resnet = 0.85 | 0.7846 |
| w_resnet = 0.95 | 0.7873 |
| **resnet34 alone** | **0.7883** |

**Monotonic to w = 1. No interior optimum.** E036 saw a weak one at 0.85 and
read it as "the decorrelation mechanism is real, the magnitude is not"; at
n=49 it is gone, so it was noise. Best blend is **−0.0010, CI [−0.0038,
+0.0015]** — the same shape E033 found for the two label sets.

**Phase D is closed with this family.** Two independent blend attempts, both
monotonic toward using none of the second member. A blend needs members of
comparable strength; 0.074 behind is too far, whatever the correlation.

**The corrected labels are published.** `knee-phase1-fused` has a new version
carrying the E037 cue fix: **710 polarity flips across 34,606 supervised slots
(2.1%)** — Synovitis 7.4%, Effusion 3.9%, Lateral Meniscus 2.7%. Coverage is
unchanged at 34.5% abstain, because the fix changes *polarity*, not reach.
Every version before this one produced the 0.846 board result; Kaggle retains
them, so that run stays reproducible.

- **a method error worth recording**: the weekly GPU quota was probed by
  *pushing a real kernel* (`35_train_dinov2f_fold4`) rather than by reasoning
  from the reset date. It was accepted, so ~5 GPU-hours went to a fifth fold of
  a family already measured dead. Kaggle has no cancel. A probe must not be a
  job that costs something if it succeeds.
- **next**: retrain the resnet34 5-fold on the corrected labels and submit.
  ~7 GPU-h. That is the only outstanding claim: whether 710 flipped labels move
  a board score of 0.846.


### E040 — the corrected labels move the model +0.0096, not separated
- **date**: 2026-08-23
- **what happened**: the retrain on the E037-corrected labels got **2 of 5
  folds** through before the weekly GPU quota ran out. Folds 0 and 1 ran on the
  corrected labels (2026-08-23); folds 2, 3 and 4 still hold their 2026-08-22
  runs on the old ones.
- **so the 5-fold cannot be pooled or submitted.** Mixing two label sets inside
  one ensemble is a confound, and the resulting board number would answer no
  question. This is recorded because pooling them was the obvious next command
  and it would have been wrong.
- **what is valid**: folds 0 and 1 against *the same folds* on the old labels —
  one variable, same splits, same architecture, same cache.

| | gold macro, n=26 |
|---|---:|
| corrected labels (E037 cue fix) | — |
| old labels | — |
| **paired difference** | **+0.0096, 95% CI [−0.032, +0.057]** |

  **NOT SEPARATED.** 710 flipped labels — 2.1% of supervised slots — move the
  model by about +0.01 on 26 studies, and the instrument cannot resolve that.
- **this was the predicted outcome**, stated before the run: a 2% label change
  was expected to land under the noise floor. It did. The fix ships on
  correctness grounds — a cue must not negate the term it judges — and its
  effect on the board remains unmeasured and probably small.
- **budget**: the week's 30 h is spent. ~5 h of it went to the DINOv2 fold-4
  quota probe (E039), which is the difference between finishing this retrain and
  not.
- **next**: folds 2–4 on the corrected labels when the quota resets, then pool
  all five, then submit. Until then the standing board result remains **0.846**
  from the pre-fix labels.


### E041 — a publicly shared label set beats this project's own by +0.114
- **date**: 2026-08-24. CPU only, no quota.
- **why this was run**: `PATH.md` §4 established that board 0.94 requires **no
  finding below ~0.870**, and that only two of the seven gaps were recoverable
  from supervision this project already had. The other five needed the *teacher*
  lifted past 0.87, and E037 measured why that was hard from our own reports.
  §3 named the alternative — publicly shared label sets — so it was surveyed.

**What is public.** Kaggle carries several openly shared LLM report-label
datasets for this competition, with four-figure download counts. Scored on the
same 58 expert studies with this project's own convention:

| label set | macro on gold 58 | coverage |
|---|---:|---:|
| **`stevenleehans/…llm-report-labels` → `llm_labels_v4_blend.csv`** | **0.8927** | **100%** |
| same author, `llm_labels_v2.csv` | 0.8873 | 100% |
| same author, `llm_labels_full.csv` | 0.8780 | 100% |
| `pilkwang/rsna-knee-llm-labels` | 0.8658 | 98.3% |
| `lixin73/…-sol56` (GPT-5.6) | 0.8352 | 100% |
| **this project's fused lexicon + Qwen** | **0.7827** | 65.5% |

**+0.114 over the labels this project spent weeks building**, at 100% slot
coverage against 65.5%.

**Per finding, against the 0.870 floor that 0.94 requires:**

| finding | ours | public | delta |
|---|---:|---:|---:|
| **Synovitis** | 0.520 | **0.790** | **+0.270** |
| Fracture | 0.644 | 0.793 | +0.149 |
| Baker's | 0.799 | 0.944 | +0.145 |
| PF OA | 0.765 | 0.902 | +0.137 |
| Effusion | 0.750 | 0.877 | +0.127 |
| Lateral OA | 0.708 | 0.833 | +0.125 |
| ACL | 0.863 | 0.987 | +0.124 |
| Contusion | 0.773 | 0.860 | +0.087 |
| MCL | 0.884 | 0.968 | +0.084 |
| Lateral Meniscus | 0.817 | 0.879 | +0.062 |
| Medial Meniscus | 0.889 | 0.948 | +0.059 |
| Medial OA | 0.935 | 0.932 | −0.003 |

**Findings clearing the 0.870 floor: 8 of 12, against 3 of 12 for ours.**
Synovitis — which E037 concluded was unfixable from these reports, and which
that entry called "not a label problem that can be fixed" — goes from **chance
to 0.790**. That conclusion was right about *this project's* reader and wrong as
a statement about the reports.

**Blending ours in makes it worse**, parameter-free rank union:

| combination | macro |
|---|---:|
| public v4 + `pilkwang` | 0.8939 |
| **public v4 alone** | **0.8927** |
| public v4 + ours | **0.8717** |

The same shape as E033 and E039: a member 0.11 behind drags the union down. The
+0.0012 from adding `pilkwang` is not worth a second dependency. **Use the
source alone.**

- **published as** `achelijndiamantidis/knee-phase1-public`, a straight
  repackaging into this pipeline's column schema — values copied unchanged, no
  rescaling, no thresholding, no per-finding selection. **The labels are not
  mine and the dataset description credits the author.** The rules permit it:
  *"Freely and publicly available external data and pre-trained models are
  allowed."*
- **wired as the `v1public` lineage**, byte-identical to the 0.846 trainer in
  every constant, so the labels stay the single variable. One consequence worth
  naming: at 100% coverage `ABSTAIN_MASKS_LOSS` now masks nothing, so the model
  sees **~53% more supervised targets** as well as better ones. Those two
  changes arrive together and this run cannot separate them.
- **what this costs the project, honestly**: the two-reader label pipeline —
  lexicon plus Qwen, the fusion rule, E023 through E032, the work that delivered
  +0.089 of the +0.121 board move — is superseded by a file someone published in
  early August. That is the mechanism `PATH.md` §3 described for how the top of
  this leaderboard is at 0.95, now measured rather than cited.
- **next**: train the five `v1public` folds when the quota resets (~2026-08-29,
  ~7 GPU-h), pool against the 0.846 system's gold OOF, and submit. Forecast with
  E038: `board ≈ gold_OOF + 0.032 + 0.005`.


### E042 — 0.90 does not need training. It needs one hour of inference.
- **date**: 2026-08-24. CPU only, no quota.
- **the question**: reach 0.90+ with the weekly GPU quota exhausted.

**A publicly shared 0.917 system publishes its out-of-fold predictions.**
`tonylica/rsna-knee-bend-dinov3-0917-repro-assets` — a pinned reproduction of
the public notebook `mattiaangeli/bend-the-knee-to-dinov3-the-original` —
carries `v52_e11_oof.csv`: one out-of-fold prediction per study for all 4,407,
with fold and gold flags. Scored on **this project's own 58 expert studies**:

| finding | ours (board 0.846) | public 0.917 | delta |
|---|---:|---:|---:|
| MCL | 0.615 | **0.871** | **+0.256** |
| PF OA | 0.664 | **0.874** | **+0.210** |
| Synovitis | 0.634 | 0.766 | +0.150 |
| Lateral OA | 0.716 | 0.818 | +0.102 |
| ACL | 0.826 | 0.909 | +0.083 |
| Contusion | 0.835 | 0.899 | +0.064 |
| Effusion | 0.866 | 0.911 | +0.045 |
| Medial OA | 0.924 | 0.953 | +0.029 |
| Medial Meniscus | 0.826 | 0.833 | +0.007 |
| **Baker's** | **0.976** | 0.920 | −0.056 |
| **Fracture** | **0.874** | 0.838 | −0.036 |
| **Lateral Meniscus** | **0.740** | 0.699 | −0.041 |
| **MACRO** | **0.7913** | **0.8576** | **+0.0663** |

**Findings at or above 0.80: 10 of 12, against 6 of 12 for ours.** `PATH.md` §4
established that board 0.90 needs every finding near 0.80; this is the first
system measured here that is close.

**MCL is the headline.** E035 identified it as the single separated recoverable
modelling loss — teacher 0.884, our model 0.628. The public system reaches
**0.871**, which is the teacher's level. The signal was always in the pixels;
this project's model was the thing failing to extract it.

**The blend has a real interior optimum, and it is not separated.**

| w_ours | gold macro |
|---:|---:|
| 0.0 (public alone) | 0.8576 |
| 0.2 | 0.8614 |
| **0.3** | **0.8622** |
| 0.5 | 0.8529 |
| 1.0 (ours alone) | 0.7913 |

**+0.0046 over public alone, CI [−0.0065, +0.0152] — not separated.** Unlike
E033 and E039, which slid monotonically to "use none of the second member",
this one has a genuine optimum at w=0.3, and ours wins outright on three
findings. The decorrelation is real; the margin is inside the instrument's
resolution. **Public alone is the honest choice.**

E038 forecast for the public system: **board ≈ 0.895**. Their reported board is
**0.917**, so the forecaster still understates — consistent with E034.

#### The finding that actually matters
**Reaching ~0.90 requires no training at all.** The weights exist
(`m_f0..f4.pt`, five folds, 94 MB each). The remaining cost is **one inference
run, ~0.9 GPU-h**, not the ~7 GPU-h of a 5-fold retrain. Every hour spent
training this project's own architecture was buying less than an hour of someone
else's inference.

#### Two constraints, both real
1. **Quota.** Even a 0.9 h inference push is refused until the weekly reset
   (~2026-08-29). CPU inference is a separate allowance and is measured at
   **19.7 s/study → 7.1 h of the 9 h cap** — viable for *this project's* single
   resnet34, too tight for a five-fold DINOv3 + RadImageNet ensemble.
2. **Licensing, and it is not pedantry.** The bundle's own README says it *"must
   remain private"*. Its manifest declares **30 files CC-BY-NC-SA-4.0**, **18
   `not-declared`**, and 3 `other`, alongside 53 Apache-2.0 and 38 CC0. The
   competition permits *freely and publicly available* pre-trained models and
   requires winners to license under CC-BY-NC 4.0. Undeclared redistribution
   terms and ShareAlike assets are a question to resolve **before** shipping
   these weights, not after. Measuring against their published OOF, as done
   here, redistributes nothing and is unaffected.
- **next**: at the reset, run inference from the *original public sources* —
  which are public and carry their authors' own declared terms — rather than
  from this private consolidation, and credit the authors. That is a ~0.9 h
  spend for a measured +0.066 on gold.


### E043 — the licensing clears: the weights that matter are CC0
- **date**: 2026-08-26. CPU only, no quota.
- **why**: E042 parked the public-weights route behind a licensing question — the
  bundle's README says it *"must remain private"* and its manifest showed 30
  files CC-BY-NC-SA-4.0, 18 `not-declared`. That was the stated blocker, so the
  manifest's per-source records were read properly rather than left as a worry.

**The bundle is a consolidation of twelve separately-licensed sources**, not one
artifact:

| licence | source | files | MB |
|---|---|---:|---:|
| **CC0-1.0** | `pilkwang/rsna-knee-weights` | 23 | **1,787** |
| **CC0-1.0** | `mattiaangeli/knee-mri-fold-weights` | 6 | **473** |
| **CC0-1.0** | `stevenleehans/rsna-knee-llm-report-labels` | 3 | 1.7 |
| **CC0-1.0** | `pilkwang/rsna-knee-llm-labels` | 2 | 1.0 |
| apache-2.0 | `pilkwang/…notebooks-figures` | 52 | 88 |
| CC-BY-NC-SA-4.0 | `marwanmath/resnet-50-radimagenet-marwan` | 1 | 94 |
| CC-BY-NC-SA-4.0 | `antoinegg1/…e11-diverse-heads-v20` | 9 | 65 |
| CC-BY-NC-SA-4.0 | `antoinegg1/…e9-radimagenet-heads-v15` | 9 | 65 |
| CC-BY-NC-SA-4.0 | `mattiaangeli/…radimagenet-foldsv1-heads` | 7 | 64 |
| **not-declared** | `sofiaanjenje/rsna-knee-e11-train` | 8 | 65 |
| **not-declared** | `sofiaanjenje/rsna-knee-e13-train` | 8 | 65 |
| other | `prvsiyan/…v52-radimagenet-heads` | 2 | 64 |

**CC0 or Apache: 2,350 MB. Restricted or undeclared: 479 MB.**

**The model weights are almost entirely CC0** — public domain dedication, no
attribution required, no restrictions:

- `knee-mri-fold-weights/m_f0..f4.pt` — **the five fold checkpoints**, 94 MB each
- `rsna-knee-weights/m_*.pt` — **twenty more checkpoints**, 89 MB each

Only the **RadImageNet heads** (CC-BY-NC-SA-4.0) and the two `sofiaanjenje`
notebook outputs (`not-declared`) are constrained. So a **CC0-only ensemble of
25 checkpoints** is available with no licensing question at all.

**And E041 is retroactively clean.** The label set that beat this project's own
by +0.114 — `stevenleehans/rsna-knee-llm-report-labels` — is **CC0-1.0**. It was
repackaged with attribution anyway, which CC0 does not require but which remains
the right thing to do.

**Reading the three tiers:**
1. **CC0** — use freely. Attribution is still given here as a courtesy.
2. **CC-BY-NC-SA-4.0** — non-commercial matches this competition's own
   **CC-BY-NC 4.0** winner licence, so NC is not the obstacle it first looked
   like; ShareAlike on derivatives is the part to read before shipping.
3. **`not-declared`** — no grant of any kind. **Avoid.** An undeclared licence
   is not a permissive one, and this is the tier the bundle's "keep it private"
   warning is really about.

- **what this changes**: E042's ~0.9 GPU-h inference plan is unblocked for the
  CC0 subset, which is where the fold weights live. Reproducing *exactly* 0.917
  needs the RadImageNet heads and therefore a ShareAlike decision; a CC0-only
  ensemble does not.
- **note on the OOF measured in E042**: `v52_e11_oof.csv` comes from
  `sofiaanjenje/rsna-knee-e11-train`, the `not-declared` tier. Reading published
  predictions to *measure* against this project's own gold set redistributes
  nothing and ships nothing. Using that file's weights in a submission is a
  different act and is not cleared.
- **next**: at the quota reset, run inference from the CC0 sources directly —
  `mattiaangeli/knee-mri-fold-weights` and `pilkwang/rsna-knee-weights`, both
  public Kaggle datasets — rather than from this private consolidation.


### E044 — the public labels take gold OOF from 0.7913 to 0.8980
- **date**: 2026-08-29. Quota reset between 2026-08-28 12:42 UTC (refused) and
  2026-08-29 00:39 UTC (accepted).
- **what changed**: the label dataset, and nothing else. The `v1public` lineage
  is byte-identical to the 0.846 trainer in every constant — same cache, same
  192px geometry, same resnet34, 24 epochs, batch 16, LR 6e-4. Five folds,
  ~7 GPU-h.

**Paired on all 58 gold studies against the 0.846 system:**

| | gold macro | 95% CI |
|---|---:|---|
| **v1public (public CC0 labels)** | **0.8980** | [0.869, 0.925] |
| v1fused (this project's labels) | 0.7913 | [0.755, 0.828] |
| **paired difference** | **+0.1067** | **[+0.077, +0.140]** |

**Separated by a wide margin** — the widest this project has measured, and about
15× the interval's distance from zero compared with E032's +0.0717.

**Per finding, against the 0.870 floor `PATH.md` §4 says board 0.94 needs:**

| finding | v1fused | v1public | delta | ≥0.870 |
|---|---:|---:|---:|:--:|
| MCL | 0.615 | **0.882** | **+0.267** | ✓ |
| Synovitis | 0.634 | 0.779 | +0.145 | |
| PF OA | 0.664 | 0.858 | +0.194 | |
| Lateral OA | 0.716 | 0.822 | +0.106 | |
| Lateral Meniscus | 0.740 | 0.851 | +0.111 | |
| ACL | 0.826 | **0.962** | +0.136 | ✓ |
| Medial Meniscus | 0.826 | **0.929** | +0.103 | ✓ |
| Contusion | 0.835 | **0.915** | +0.080 | ✓ |
| Effusion | 0.866 | **0.929** | +0.063 | ✓ |
| Fracture | 0.874 | **0.885** | +0.011 | ✓ |
| Medial OA | 0.924 | **0.980** | +0.056 | ✓ |
| Baker's | 0.976 | **0.984** | +0.008 | ✓ |

**Eleven of twelve findings now clear 0.80; eight clear 0.870.** The 0.846
system cleared 0.80 on six and 0.870 on four.

**MCL, +0.267.** E035 named it the single separated recoverable modelling loss —
teacher 0.884, model 0.628 — and concluded the signal was in the pixels and the
model was failing to extract it. With better supervision the same architecture
on the same cache reaches **0.882**, which is its teacher's level. The
diagnosis was right and the fix was labels, not modelling.

**Synovitis is now the only finding below 0.80**, at 0.779 — up from 0.634, and
up from a teacher E037 measured at **chance**. E037 concluded Synovitis "cannot
be fixed from the reports"; that was true of *this project's* reader and false
of the reports, and this is the second measurement to say so.

- **two changes arrive together and this run cannot separate them.** The public
  labels are both *better* (teacher 0.8927 vs 0.7827) and *more complete* (100%
  slot coverage vs 65.5%, so `ABSTAIN_MASKS_LOSS` now masks nothing and the
  model sees ~53% more supervised targets). Attributing +0.1067 to label
  quality alone would be unearned. Separating them needs a run with the public
  labels artificially masked to 65.5% coverage, which is ~7 more GPU-h and has
  not been done.
- **forecast**: E038's corrected rule — `board ≈ gold_OOF + 0.032 + 0.005` —
  puts this near **0.935** against a standing 0.846. That rule has one genuine
  test behind it (E038) and understated the board on the run before it, so treat
  it as a direction, not a number. The submission is the measurement.
- **credit**: the labels are `stevenleehans/rsna-knee-llm-report-labels`
  (`llm_labels_v4_blend.csv`), shared publicly under **CC0-1.0** and repackaged
  with attribution as `knee-phase1-public` (E041, E043). None of this +0.1067 is
  this project's own label work.


### E045 — 0.923 on the board, and the forecaster held to +0.012
- **date**: 2026-08-29
- **what**: `knee-infer-v1pub` v1 — the five `v1public` folds, rank-mean, ~1 h.

| submission | date | board |
|---|---|---:|
| constant priors | 08-18 | 0.500 |
| scanner metadata, no pixels | 08-18 | 0.531 |
| imaging 192px, 1 fold, lexicon labels | 08-19 | 0.725 |
| imaging 288px, effective batch 16 | 08-19 | 0.688 |
| lexicon labels, 5-fold (the control) | 08-23 | 0.757 |
| fused labels, 5-fold | 08-22 | 0.846 |
| **public CC0 labels, 5-fold** | **08-29** | **0.923** |

**+0.077 today. +0.198 from this project's first imaging model.** Field top
0.952, top-200 cut 0.917 — **0.923 clears the top-200 cut**.

**The forecaster held.** E038's corrected rule — `board ≈ gold_OOF + 0.032
(ensembling) + 0.005` — predicted **0.935** against an actual **0.923**, an
error of **+0.012**. Its previous test (E034) missed by −0.059 in the other
direction, so this is the first forecast it has made that was close, and the
first time it has *overstated* rather than understated.

**The raw gold-to-board offset is still unstable and still positive:**

| model | gold OOF | board | offset |
|---|---:|---:|---:|
| lexicon, 192px, 1 fold | 0.7201 | 0.725 | +0.005 |
| fused, 5-fold | 0.7918 | 0.846 | +0.054 |
| **public, 5-fold** | **0.8980** | **0.923** | **+0.025** |

Three points, offsets +0.005 / +0.054 / +0.025. The correction narrows the
spread but does not remove it, and `FINDINGS.md` §14.5 stands: **gold OOF ranks
models; it does not forecast a score.** It ranked all three correctly.

- **what produced the +0.198 overall**, decomposed on ground truth where it
  could be:

  | lever | delta | how measured |
  |---|---:|---|
  | ensembling 1 fold → 5 | +0.032 | E036 control, board |
  | this project's fused labels | +0.089 | E036, board |
  | **public CC0 labels** | **+0.077** | here, board |

  **Labels account for +0.166 of +0.198. Everything else — architecture,
  resolution, pooling, backbone — accounts for the rest.** Every architectural
  lever this project tried (288px, DINOv2, focal top-k, per-finding pooling,
  cross-family blending) measured zero or negative. That is the whole story of
  this competition as this project has experienced it.
- **the honest accounting of whose work this is**: the +0.077 came from a label
  file published by **stevenleehans** under CC0-1.0, found by surveying what the
  community had shared, not built here. `PATH.md` §3 predicted this mechanism
  before it was used and is now confirmed by measurement.
- **still unseparated**: better labels and 53% more supervised slots arrived
  together (E044) and this submission cannot tell them apart either.
- **next**: 0.94 needs **+0.017**. `PATH.md` §4 says that means no finding below
  ~0.870, and **Synovitis at 0.779** is the one still short — every other finding
  now clears 0.80 and eight clear 0.870. Two routes remain, both untried: the
  CC0 public *weights* (E042/E043, ~0.9 GPU-h, measured 0.8576 gold as a
  standalone system) and blending them with this one.


### E046 — the borrowing route is exhausted: we now beat what we were borrowing
- **date**: 2026-08-29. CPU only, no quota, no submission.
- **the question**: E042 measured the public 0.917 system at **0.8576** gold
  against the *0.846* system's 0.7913 and found a blend worth +0.0046, not
  separated. The 0.923 system is 0.107 stronger, so the blend was worth
  re-measuring before spending ~0.9 GPU-h building a CC0-weights inference
  kernel.

| | gold macro, n=58 |
|---|---:|
| **ours (board 0.923)** | **0.8980** |
| public 0.917 system | 0.8576 |

**The relationship has inverted.** When E042 ran, the public system was **+0.066
ahead** of this project's. It is now **0.040 behind**. The system this project
was preparing to borrow from is weaker than the one it has.

**Blend, parameter-free rank union:**

| w_ours | gold macro |
|---:|---:|
| 0.5 | 0.8971 |
| 0.65 | 0.9015 |
| **0.70** | **0.9016** |
| 0.8 | 0.9013 |
| 1.0 (ours alone) | 0.8980 |

**+0.0036 at the optimum, 95% CI [−0.0078, +0.0160] — not separated.**

There is a **broad** interior optimum from 0.65 to 0.8, all within 0.0003 — so
the decorrelation is real and stable, not the noise E039 found. But the margin
is a third of the instrument's resolution. Three blend attempts now
(E033 labels, E039 architectures, E042 and this) have produced +0.0046, +0.0022
and +0.0036, none separated. **Blending is not a lever in this competition at
this sample size.**

Per finding, ours wins on 10 of 12 — the exceptions are PF OA (0.858 vs 0.874)
and Synovitis (0.779 vs 0.766, effectively tied).

- **decision: do not build the CC0-weights inference kernel.** It would cost
  ~0.9 GPU-h and a submission for +0.0036 that cannot be distinguished from
  zero, on top of a system that already beats it. E042's "cheapest route to
  ~0.90" was correct when written and is now obsolete — the route arrived first
  by another road.
- **what this closes**: every borrowing lever identified in `PATH.md` §3 has now
  been tried. The **labels** were worth +0.077 on the board (E045). The
  **weights** are worth nothing, because this project's own model overtook them
  in the same week.
- **what remains for 0.94** (+0.017 on the board): **Synovitis at 0.779** is the
  only finding below 0.80 and the only one clearly short of the ~0.870 floor
  `PATH.md` §4 requires. Every architecture lever in this project was tested
  against the *old* weak labels — 288px, DINOv2, focal top-k, per-finding
  pooling all measured zero or negative when the teacher was 0.78. **None has
  been retested against a 0.89 teacher**, and that is the one large untested
  region left.

- **and more epochs will not help.** The five `v1public` runs are converged:

  | fold | peak val | at epoch | final (23) |
  |---|---:|---:|---:|
  | 0 | 0.8551 | 19 | 0.8515 |
  | 1 | 0.8383 | **23** | 0.8383 |
  | 2 | 0.8725 | 19 | 0.8701 |
  | 3 | 0.8233 | 18 | 0.8200 |
  | 4 | 0.8448 | 19 | 0.8419 |

  Four of five peak at epoch 18–19 and decay after; only fold 1 is still
  rising at 23. This is the same shape E036 found for DINOv2 and read
  correctly — a curve that has turned over, not one still climbing. **Saves
  ~7 GPU-h** that a "train it longer" instinct would have spent.
- **so a one-fold probe is queued instead**: `43_train_dinov2pub_fold0`,
  DINOv2 on the public labels, ~5 GPU-h, constants otherwise identical to the
  `dinov2fused` lineage. If its gold lands near the resnet34's **0.8477** on
  fold 0, a second family is competitive against a good teacher and worth four
  more folds; if it is 0.07 behind as in E036, the probe saved ~20 GPU-h.


### E047 — two new public label sets are unevaluable: they contain the answer key
- **date**: 2026-08-31. CPU only, no quota.
- **why**: labels are the dominant lever (+0.166 of the +0.198 board gain), and
  the last survey was 2026-08-24. Five days is a long time in an active
  competition, so it was repeated.

**Four new label sets have appeared.** Two carry all twelve findings in a usable
schema, and both score **macro 1.0000** on the 58 gold studies.

**A perfect score is a leak signal, not a result.** Checked directly:

| dataset | slots on the 58 equal to the expert label | binary on gold | binary on 2,000 non-gold |
|---|---:|---:|---:|
| `tasmeemreza/rsna-knee-refined-llm-labels` | **100.0%** | 100% | 76.5% |
| `shingo257/rsna-knee-calibrated-labels-v1` | **100.0%** | 100% | 0.0% |

Both **copy the expert labels verbatim** for the 58 gold studies — which is
public information, sitting in `train.csv`, and entirely legitimate for their
authors to use. It is not cheating and it is not necessarily bad labelling. Away
from the gold set they behave quite differently: one is 76.5% hard 0/1, the
other fully continuous.

**The consequence is that this project's only offline instrument is blind to
them.** Gold-58 measures a label set by asking how well it agrees with the 58
expert answers; a file that contains those answers scores 1.0000 regardless of
its quality on the other 4,349 studies — which is the part that actually trains
the model.

**The incumbent is clean.** `stevenleehans/llm_labels_v4_blend`, the label set
behind the 0.923 board result, scores **0.8927** — not 1.0 — so it does not carry
the gold labels and its measurement means what it appears to mean. That is why
E041's +0.114 was a real comparison and this one cannot be.

- **what this rules out**: swapping in either new set on the strength of
  "1.0000 > 0.8927". That number compares one file's copy of an answer key
  against another file's honest attempt, and acting on it would have been the
  most expensive kind of mistake this log records — a weakly-known number read
  as a well-known one, for the eighth time.
- **what it does not rule out**: that they are better. They may well be. The
  only instrument that can tell is **the board**, at ~7 GPU-h to train plus one
  of five daily submissions, per label set, blind.
- **note on training**: this project already assigns expert labels to gold
  studies at `GOLD_WEIGHT=8.0`, so a label file that also contains them
  introduces no leakage into training that is not already there. The problem is
  purely one of *evaluation*.
- **next**: with ~7 GPU-h left this week and two DINOv2 folds running, the
  honest options are (a) spend a blind 7 GPU-h on one new label set and let the
  board judge, or (b) wait for next week's quota. Neither is a measurement this
  project can make offline.


### E048 — a fourth independent reader adds nothing; the label lever is spent
- **date**: 2026-08-31. CPU only, no quota.
- **why**: E047 established the two newest label sets are unevaluable (they carry
  the answer key). A third, `laymond/…qwen3-8b-weak-labels`, is a genuinely
  independent reader — Qwen3-8B, with `__label`, `__confidence`, `__mentioned`
  and `__negated` per finding — and is **evaluable**: it matches the expert
  label on 83.2% of gold slots, not 100%, so it is a real attempt rather than a
  copy.

**Each reader alone, on the 58:**

| reader | macro | Synovitis |
|---|---:|---:|
| **incumbent (`llm_labels_v4_blend`)** | **0.8927** | **0.790** |
| pilkwang | 0.8658 | 0.687 |
| gpt-5.6-sol | 0.8352 | 0.676 |
| Qwen3-8B (label × confidence) | 0.8293 | 0.657 |

**Parameter-free rank unions:**

| combination | macro |
|---|---:|
| incumbent alone | 0.8927 |
| incumbent + qwen3 | 0.8843 |
| incumbent + pilkwang | 0.8939 |
| incumbent + qwen3 + pilkwang | 0.8943 |
| **all four** | **0.8954** |

**+0.0027, 95% CI [−0.0086, +0.0132] — not separated.**

- **the tempting number was +0.0044**, from weighting the incumbent ×2 in the
  union. That weight is **a free parameter fitted to 58 studies**, which is the
  exact practice `dataset-metadata.fused.json` rejects: *"a rule that picked the
  better reader per finding would be fitting twelve choices to 58 studies and
  would report a number that means nothing."* It is recorded and not used.
- **why this differs from E023**, where the union of two readers beat both by
  **+0.070**: there, the two readers were **comparable** (0.7446 and 0.7421) and
  abstained on different findings. Here the incumbent is 0.03–0.06 ahead of
  every other reader, so a union mostly imports their errors. This is the same
  weak-member shape as E033, E039 and E046 — the fourth time it has appeared.
- **the conclusion, and it is the important one**: **the label lever is spent
  for this project.** Four public readers exist; the best is 0.8927; combining
  them adds nothing measurable. Labels delivered +0.089 and +0.077 on the board
  and there is no fifth reader on the shelf to deliver a third instalment.
  **Synovitis stays at 0.790 in the best available teacher** — no public reader
  is above 0.79 on it — which is why the model sits at 0.779 there.
- **what remains for a higher score**, with the label route closed:
  1. **more models** — the only lever with a measured coefficient (+0.032 for
     1→5 folds), extrapolating to ~+0.010 for 5→10;
  2. **a second architecture**, being probed now against the good teacher;
  3. **a Synovitis reader better than 0.790**, which nobody in this competition
     appears to have published, and which would have to be built rather than
     borrowed.


### E049 — Synovitis is not recoverable from Effusion either
- **date**: 2026-08-31. CPU only, no quota.
- **hypothesis, declared before measuring**: "effusion-synovitis" is a single
  named construct in the MOAKS scoring system, so the model's **strong Effusion
  channel (0.929)** might carry Synovitis signal that its **weak Synovitis
  channel (0.779)** misses. Synovitis is worth **+0.010 board** alone (E046
  arithmetic), the largest single prize left.
- **the clinical coupling is real**: on the 58 expert studies, Synovitis is
  positive in **62.9%** of Effusion-positive knees against **21.7%** of
  Effusion-negative ones, φ = **+0.403**.
- **and the model has already extracted it**:

  | predicting Synovitis with | AUC |
  |---|---:|
  | **its own Synovitis channel** | **0.779** |
  | the Effusion channel | 0.705 |
  | blend, w_synovitis = 0.9 | 0.778 |
  | blend, w_synovitis = 0.8 | 0.774 |
  | blend, w_synovitis = 0.5 | 0.743 |

  **Monotonic to w = 1.0.** Best is the Synovitis channel alone, delta +0.000.
  The correlation exists in the labels but contributes nothing the model has not
  already used.
- **what this closes**: the cheap route to Synovitis. Three attempts now — a
  better report reader (E037), a better *public* report reader (E048, none above
  0.790), and a correlated-finding prior (here). Synovitis at 0.779 is where it
  stays without a genuinely new source of supervision.
- **the standing tally of what is measured dead**: this project's own labels
  (superseded), blending in four forms (E033, E039, E042, E046, E048), borrowed
  weights (E046 — we overtook them), more epochs (E046 — converged), 288px,
  DINOv2 against weak labels, focal top-k, per-finding pooling, Synovitis via
  reports and via correlation. **One lever retains a measured coefficient: more
  models, +0.032 for 1→5 folds.**


### E050 — TTA is worth nothing, and the reason is worth more than the measurement
- **date**: 2026-08-31. **CPU only — zero GPU hours**, 16.4 min wall clock.
- **why**: inference has never used test-time augmentation and no experiment in
  this log had ever tested it, which made it the last untried lever that was
  not "train more models". Unlike every other remaining lever it could be
  measured without a GPU: `gold_eval` already scores checkpoints on CPU, and
  58 studies through resnet34 is minutes. `knee-tta-eval` (kernel 50) mounts
  the five `v1public` folds behind the 0.923 board result and runs each held-out
  gold study through four views.
- **views**: only symmetries training already teaches — `identity`, `reverse`
  (slice order, trained at p=0.5) and a pixel roll of ±`TARGET_SIZE//16` in
  both directions. Intensity jitter and coarse dropout were excluded on
  purpose: they inject noise to regularise, and averaging over noise adds
  variance without adding a view. A left-right flip was excluded for a reason
  specific to this dataset — right knees are mirrored during the cache build so
  every volume shows the same anatomy, and four of the twelve findings are
  explicitly medial or lateral, so a flip would move the answer rather than ask
  for it twice.

- **the baseline reproduces the record exactly**, which is what makes the rest
  of this readable: pooled identity over n=58 is **0.8980**, the same figure
  E044 recorded for the public-label lineage, to four decimals. Per-fold the
  recomputation differs from the training kernel's own gold dump by at most
  6.5e-4 — T4 versus CPU float arithmetic, enough to swap two near-tied ranks
  inside a 12-study fold and nothing more.

| view(s) | gold macro, n=58 |
|---|---:|
| `identity` | **0.8980** |
| `reverse` | **0.8980** |
| `shift_pos` | 0.8883 |
| `shift_neg` | 0.8962 |
| `identity,reverse` | 0.8980 |
| all four (unweighted) | 0.8986 |

- **paired: identity − 4-view = −0.0006, 95% CI [−0.006, +0.005]. Not
  separated.** This is the tightest interval in the whole log — 0.011 wide
  against the usual 0.044 — because both sides are the *same model* on the
  *same studies*, so almost all the variance the paired bootstrap normally has
  to carry cancels. It is therefore a sharp null rather than an underpowered
  one: TTA is not unmeasured here, it is measured at zero.
- **no weighted variant was tried**, for the reason E048 gives. Weights over
  four views fitted to 58 studies are four free parameters bought with 58
  studies.

- **the reason `reverse` ties identity to four decimals is that it is exactly
  the same number.** The architecture embeds each slice independently and pools
  with a softmax-weighted sum over the token axis — no positional encoding, no
  operation anywhere that mixes neighbouring slices. So it is *exactly*
  permutation-invariant over slices. Measured: max |f(x) − f(reverse(x))| =
  **2.4e-7**, and the same for a random permutation, on both the plain and the
  focal-top-k head.
- **so the slice-reversal augmentation in training was dead code**, and has
  been in every run this project has ever made. Every augmentation applied
  after it is order-independent too (the roll, the intensity scale, the
  dropout box and the plane drop are all identical across slices), so
  reversing produced a volume the model could not distinguish from the one it
  already had. It bought a full array copy on half of all training samples in
  exchange for nothing. **Removed**, with
  `test_the_model_is_permutation_invariant_over_slices` pinning the property it
  rests on — if that test ever fails, the architecture has gained slice-order
  sensitivity and the augmentation should come back with it.
- the two shift views *do* change the output and both land at or below
  identity. `np.roll` wraps content across the border, which is not a symmetry
  of a knee; the model tolerates it because it was trained on it, and gains
  nothing from being asked.

- **the finding underneath is bigger than TTA and is not being chased today.**
  The model does not see a study as a stack. It sees an unordered bag of 60
  slices. That a meniscal tear appears on three *adjacent* sagittal slices —
  the continuity that makes a radiologist scroll rather than shuffle — is
  information this architecture is structurally incapable of using. That is the
  strongest architectural lead in this log, and it is a lead precisely because
  nothing in the 288px / DINOv2 / focal-top-k / per-finding-pooling family ever
  addressed it; they all changed the encoder or the head while leaving the bag
  a bag. Testing it means slice positional encoding or a small transformer over
  the token axis, at 5-7 GPU-h a fold against ~7 h left this week. Recorded for
  a week with quota, not started with hours.

- **and inference has no budget problem at all**, measured from the last
  submission's manifest rather than assumed: 0.98 h projected for 1,300 test
  studies against the 9 h cap, of which the forward pass is 0.51 s/study across
  five members. An extra ensemble member costs **0.037 h** on the full test set
  — 2.2 minutes. Ten members with four-view TTA would come to roughly 2.3 h,
  still a quarter of the cap. **Ensemble size is limited by training quota
  alone**; the submission side has ~8 h of headroom and is not the constraint
  anyone should be designing around.


### E051 — DINOv2 is dead against a good teacher too, and further behind than before
- **date**: 2026-08-31. ~11 GPU-h, two folds, the probe E046 queued.
- **why**: every architecture lever this project owns was measured against a
  0.78 teacher. The probe asked whether a second family becomes competitive
  once the teacher is 0.89.

| fold | DINOv2 gold at its saved epoch | resnet34 gold, same fold | gap |
|---|---:|---:|---:|
| 0 | 0.8178 (best val 0.8240 @ 27) | 0.8526 | **−0.035** |
| 1 | 0.7765 (best val 0.8141 @ 29) | 0.9249 | **−0.148** |

- **the verdict is the same as E036 and slightly worse.** E036 put DINOv2 0.074
  behind against the weak teacher; against the good one it is 0.035 and 0.148
  behind on the two folds measured. A better teacher does not rescue it.
- **folds 2-4 are not being run.** ~15 GPU-h saved, which is more than remains
  this week. This is the second time a one-fold-first probe has paid for
  itself, and the reason the second slot was filled with fold 1 rather than
  left idle: one fold could not have carried this, at a ~0.19 interval.
- **an incidental finding worth more than the verdict.** On both folds the
  report-label validation AUC keeps *climbing* while the gold AUC *falls* —
  fold 1 runs val 0.8074 → 0.8141 across epochs 22-29 while gold goes 0.8121 →
  0.7765. `FINDINGS.md` §11 recorded that report-label CV mis-ranks *models*;
  this is the same disagreement appearing *within one run's trajectory*, which
  means it also mis-picks the epoch. Early stopping on val AUC selected weights
  0.036 worse on expert truth than the ones eight epochs earlier. Every lineage
  in this project selects its export that way.


### E052 — the untested region pays: per-finding pooling separates against the 0.89 teacher
- **date**: 2026-08-31. **CPU only — zero GPU hours**, ~4 min per arm.
- **why**: E046 named it outright — "every architecture lever was tested
  against the *old* weak labels; **none has been retested against a 0.89
  teacher**, and that is the one large untested region left." `head_lab.py`
  trains everything above a frozen backbone in minutes, so the region costs
  coffee rather than quota. All three A/Bs re-run with `--labels
  artifacts/phase1_public`.

- **first, the instrument was wrong and had to be fixed.** Run one seed per arm,
  focal top-k against the public labels reported +0.0256 [+0.001, +0.051],
  "A is better". Repeating it on three more seeds gave +0.0136, +0.0443,
  +0.0071 — the same comparison swinging by a factor of six. The reason is
  visible in the arms rather than the deltas:

  | seed | focal | baseline |
  |---|---:|---:|
  | 0 | 0.7361 | 0.7106 |
  | 1 | 0.7428 | 0.7292 |
  | 2 | 0.7433 | 0.6990 |
  | 3 | 0.7419 | 0.7347 |

  Focal spans 0.007 across restarts; **the baseline spans 0.036**. A
  single-seed A/B on 58 gold studies was reading the initialisation draw as if
  it were the architecture. `--seeds N` now averages out-of-fold predictions
  over restarts before scoring, which removes it from both arms.

- **the three levers, four restarts each, against the public teacher:**

| lever | Δ vs baseline | 95% CI | verdict |
|---|---:|---|---|
| **per-finding attention maps** | **+0.0338** | **[+0.009, +0.061]** | **separated** |
| focal top-k (k=3) | +0.0163 | [−0.001, +0.035] | not separated, all 4 seeds positive |
| slice positional embedding | +0.0035 | [−0.017, +0.025] | not separated |

- **per-finding pooling was recorded dead in E030 and is not dead.** What
  changed is the teacher, and the mechanism for why that matters was written
  down before the measurement, in the model's own comment: one attention map
  over twelve findings forces a single compromise about which slices matter,
  and the compromise is paid by the focal findings. Against a 0.78 teacher the
  focal findings had no signal left to sharpen. Against 0.89 they do — and they
  are still exactly where the 0.923 ensemble is weakest, at Synovitis 0.771,
  Lateral OA 0.830 and PF OA 0.849 against Medial OA 0.980 and Baker's 0.978.
- **focal top-k moves the same way and is not being taken**, because two levers
  at once stops being one variable. It is the next thing to try if pooling
  lands.
- **slice position buys nothing**, which lowers the prior on E050's
  architectural lead without closing it: a frozen-feature head at 0.74 may
  simply be floor-limited. It does say that lead is not worth 7 GPU-h today.
- **caveats, stated rather than buried.** These are frozen DINOv2 features, so
  the absolute numbers do not predict the board; what `head_lab` claims to
  transfer is comparisons *above the backbone*, which all three of these are.
  Three comparisons were run, so one separating at 95% is roughly a one-in-seven
  coincidence on its own — the reasons to believe it are the pre-registered
  mechanism and the second lever moving the same way, not the interval alone.
  The board is the only place this gets settled.

- **so the remaining GPU goes to `v1pubpool`**, five folds of the 0.923
  configuration with `per_finding_pool=True`, rather than to `v1publicB`'s
  reseed. Same cost, and it dominates: the folds join the ensemble either way,
  a different pooling is more diversity than a different initialisation, and
  the run answers a question while it does it.
