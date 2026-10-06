# Handoff — RSNA Knee Abnormality Detection

**Written 2026-10-06. Target: the user wants 0.958. Read §0 before promising it.**

Everything below is measured in this repo unless marked otherwise. `docs/EXPERIMENTS.md`
is the full log; entries are cited as E###. **A run with no entry did not happen.**

---

## 0. The thing to understand before doing anything

**Board now: 0.945.** Ladder: 0.938 → 0.940 → 0.942 → 0.944 → 0.945 → 0.945 → 0.944.
Four +0.002 steps, then +0.001, then +0.000. **Cumulative gain of this entire
project is +0.007.** The ask is +0.013 on top.

**The public leaderboard is ~390 studies.** A commissioned literature review
(E136) computed its standard error: **±0.0046 if the twelve findings were
independent, ±0.006–0.008 at realistic correlation.** So:

- **0.945 and 0.958 are ~2 SE apart as standalone numbers.**
- A *paired* comparison of two near-identical ensembles is ~4× sharper
  (SE_Δ ≈ 0.0006–0.0015), which is the only reason the ladder above is readable.
- Trying k variants and keeping the best public one inflates by **+0.0015–0.003
  at k = 10** (winner's curse). Part of the "flattening" is that bias vanishing.
- **Private is ~910 studies.** Teams within ~0.002 on private are effectively tied.
  Chasing public points past the floor is how people lose private rank.

**Do not tell the user 0.958 is achievable.** Three sessions of mine ended in
"this lever returned +0.000". Say what is measured.

---

## 1. What ships, exactly

Blend = **four CoAtNet "Raptor" arms** (foreign, CC0) **+ three mattiaangeli
CoAtNet packages** (foreign, CC0) **+ ten of our own 2.5D resnet-family members**,
combined by equal-weight rank-mean inside one kernel.

Honest gold-58 numbers, **each gold study predicted by a model that never saw it**:

| arm | macro |
|---|---:|
| mattiaangeli 3-pack | 0.9275 |
| coat 4-arm | 0.9173 |
| **ours (v1), one resnet34** | **0.8716** |

**Our own arm is the weakest component**, and it sits below the ~0.90 bar used to
reject five foreign assets. Removing it reads +0.0038, CI [−0.0052, +0.0135] —
**not separated, and the test is biased toward removal** (its proxy is one
resnet34 on 80% of the data; the shipped arm is ten full fits on 100%). E140.

---

## 2. Two submissions are built, verified, and were never clicked

1. `knee-infer-raptor4` **version 7** — ten members + three pipelines, the union
   never submitted. Stub verified: ten distinct fingerprints, zero fallbacks.
2. `knee-infer-raptor-nov1` **version 2** — four CoAtNet arms + 3-pack, **v1 arm
   removed**. Pre-registered against 0.945: ≥0.948 the arm was diluting;
   0.943–0.947 it stays (incumbent wins a tie); ≤0.942 it contributes.

**Get these two numbers before building anything new.** They are free and #2
answers a question every submission since E105 has left open.

---

## 3. The instrument — the one genuinely new capability

`eda/pool_oof.py` + five fold trainers `knee-train-v1pubcv-f0..f4` (E137, E141).
Pooled: **4,407 studies, each exactly once**, partition-checked. 669–2,618
positives per finding, against gold-58's 9–35.

```
python eda/pool_oof.py <dir>/oof_fold*.csv.gz
python eda/pool_oof.py a/oof_fold*.csv.gz --vs b/oof_fold*.csv.gz   # paired bootstrap
```

**It ranks changes to OUR arm. It cannot score the shipped composite** — the
CoAtNet arms are foreign checkpoints, in-sample on the 4,349, and no honest OOF
for them can be built. Confusing those two is the likeliest way to misuse it.

**The lateral pair is excluded from its macro by default.** Report-label noise is
class-conditional for most findings (so AUC against them is rank-preserving) but
**instance-dependent for Lateral Meniscus and Lateral OA** — posterior-horn tears
go *unreported* rather than reported wrongly. Gold judges those two. `--include-lateral`
overrides deliberately.

**Per-finding, pooled (n=4,407):** MCL 0.7825, PF OA 0.8056, Lateral OA 0.8108,
Synovitis 0.8226, Effusion 0.8379, ACL 0.8383, Contusion 0.8405, Lateral Meniscus
0.8573, Fracture 0.8573, Medial OA 0.8680, Baker's 0.8956, Medial Meniscus 0.9012.

**MCL is the worst finding on 669 positives and was second-best on gold-58's 8.**
The two rankings correlate at only **+0.350** (E139). Gold-58's per-finding numbers,
which E133's whole "four weak findings" argument rested on, are not safe.

---

## 4. Closed, with numbers. Do not re-propose these.

| route | result |
|---|---|
| more resolution | −0.088; 288px **0.037 behind** 192px on the board |
| per-finding blend weights | board returned the identical number |
| blend weight tuning | 0.000, three separate times |
| public MaxSpan weights 0.60/0.10 | +0.0009 arm-alone, **+0.0000 shipped** (E138) |
| more members / families | 10 vs 8 = +0.002, inside the floor |
| reseeding | +0.000 (E064) |
| GBM / tree stacker | rank-mean 0.8635, GBM 0.8598, GBM+metadata 0.8636 (E135) |
| per-scanner normalisation | **−0.0209**, vs −0.0028 shuffled control — scanner is *signal* |
| AutoML | no trustworthy validation signal |
| **8 borrowed assets screened** | clean split at ~0.90: ≥0.909 pays, ≤0.892 costs |
| DINOv2 ×20 (the public recipe's base stage) | 0.8381, costs **−0.0171** (E132) |
| RadImageNet heads | 0.8543 / 0.8256, both negative |
| synovitis-via-effusion surrogate | closed (E121) |
| LR flip / slice reversal augmentation | closed — knees pre-mirrored; permutation-invariant |
| **CoAtNet for our own arm** | **host-OOM-killed 3 sessions running** (E143, E144) |

**No foreign asset beats us on a single finding, weak ones included.** Borrowing
is exhausted.

---

## 5. Live threads

**`maxvit_rmlp_tiny_rw_256` probe (`kaggle/110_train_v1probe_maxvit`) is built,
preflighted, never pushed.** It is the only remaining test of whether *any*
hybrid-attention backbone can train here. Prior is poor: depthwise-separable has
host-OOM-killed **5/5** (convnext_tiny, tf_efficientnet_b0, shufflenet_v2_x1_0,
coatnet_rmlp_nano, coatnet_rmlp_2) against dense/grouped surviving **6/6**. MaxViT
is also MBConv-based, so expect a kill. ~80 min to find out.

**The kill is host RAM, not GPU** (`Killed` = Linux OOM killer), fires **before
epoch 0**, right after `DataParallel across 2 GPUs`. One single-GPU run would
localise it. Not recommended as a priority — three sessions spent, and nothing
suggests CoAtNet at 192px clears the 0.8485 control anyway.

---

## 6. What I would actually do next, in order

1. **Click the two built submissions.** Free, and #2 resolves a standing question.
2. **Use the instrument on our own arm** — it has never been used for a real
   experiment. Candidates it can now resolve that gold-58 could not: augmentation
   magnitudes (untested), the intensity window / uint8 quantisation at cache build
   (untested), per-finding specialists for **MCL** (newly identified as worst).
3. **Protect private rank.** Two final picks: one robust core, one core+new. Never
   select on a public gap ≤0.002. Deadline 22 Oct.

**What I would not do:** another blend permutation. Six consecutive levers have
returned inside the floor.

---

## 7. Hard operational facts

- **`kaggle competitions submit` does not work** — HTTP 400, notebook-only
  competition. **Every submission needs a human click.** Do not retry, do not
  debug the CSV.
- **`kaggle competitions submissions` returns 403 and `leaderboard` returns 404**
  for this account. Scores must come from the user. You cannot read the board.
- Token: `KAGGLE_API_TOKEN=KGAT_958a35dbb329e4492d380aa50df76565`. **Never commit
  it**; a test asserts no literal token under `gcp/`.
- **Never hand-edit `kaggle/*/run.py`** — generated from `src/pipeline.py` via
  `python eda/generate_kernels.py --write`.
- **Run `bash eda/preflight.sh` before every push.** Lint, tests, kernel drift,
  patient-data check.
- **Commits must be authored `existentialistlogarithmic <achilleasd278@gmail.com>`
  in both author and `Co-Authored-By`. Never Claude, in any field.** The container
  defaults to Claude, so pass `-c user.name=... -c user.email=...` explicitly.
- Work on `main`. 5 submissions/day, 2 concurrent GPU sessions, 9 h runtime cap.
- **Background bash is capped at 600 s** — you cannot poll an 80-minute training
  run in one task. Check status on each turn instead.
- Competition-derived data (per-study predictions, caches) **must not enter the
  repo**. They live in the scratchpad. The preflight check is the last line of
  defence and it has already nearly been needed.

## 8. Three mistakes I made that are cheap to repeat

1. **A scripted manifest edit replaced a kernel instead of inserting beside it.**
   `knee-infer-raptor4` vanished from `src/pipeline.py`, its directory stayed on
   disk, and **preflight printed "tree matches the manifest" and passed** — the
   drift check only walked manifest → tree. Fixed; it now fails on undeclared
   directories, with six genuinely retired ones named individually.
2. **One `except` conflated two failures.** `build_model` caught timm's
   `TypeError` for an unsupported `dynamic_img_size` in the same handler as a
   failed weights download, silently pinning CoAtNet to its native 224 so it died
   an epoch in. **Two GPU sessions.** Fixed, with a forward pass at `TARGET_SIZE`
   before any build is kept.
3. **I published a lead from one fold.** Fold 0 peaked at epoch 17 and I wrote
   that every shipped member might be over-trained; folds 1–3 peak at 20/21/21,
   mean 19.8. It cost nothing only because I said explicitly I would not act on
   one fold. **Say that out loud when you do it.**
