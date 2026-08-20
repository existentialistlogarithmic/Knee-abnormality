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
