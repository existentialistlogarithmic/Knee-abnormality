# Martijn — final comments

**Written 2026-10-16, 06:30 UTC.** Six days before the final deadline
(**2026-10-22**). This supersedes `COMMENTS_FOR_MARTIJN.md` at the repo root,
which dates from 17 Sep, when the board was 0.940.

Please put your answers in **[`COMMENTS.md`](COMMENTS.md)** in this folder. The
exact code we added to the public notebooks is in
**[`appended-cells.md`](appended-cells.md)**. The full experiment record is
`docs/EXPERIMENTS.md`, entries **E155–E158**, read from the bottom.

---

## 1. Where we are

| | |
|---|---|
| **Our best public score** | **0.958** (21 Oct, E below); A 0.957; 0.956 on 14 Oct |
| Our own pipeline's best | 0.945 (our resnets + public CoAtNets); closed as a base |
| Top of the board | 0.968 (6,065 teams) |
| 0.960 / 0.958 / 0.957 / 0.956 | rank ~51 / ~106 / ~209 / ~602 (0.957 has 393 teams tied; 0.956 has 797) |
| Submissions | 5 per day; 2 used today (16 Oct) |
| Final selection | Kaggle scores the private board (~910 studies) on **2 submissions we select**. Deadline 22 Oct |

**Why we switched to public notebooks.** About 1,400 teams are at ≥ 0.956, and
most of them run the same public pipeline: Raptor CoAtNets plus heliosli's
compact ConvNeXt. Our own models top out at 0.945. The only things that can still
separate us are (a) the newest public base, (b) a public model the base does not
use, and (c) our own models added as an extra vote.

## 2. In flight right now

Four private notebooks under `achelijndiamantidis`, built from heliosli's newest
public notebook (*"This is new public dataset version = 0.9576"*):

| | notebook | what it is | stub run | submission |
|---|---|---|---|---|
| **A** | `knee-pub9576` v2 | the 0.9576 notebook, unchanged except one dataset swap (§3.3) | COMPLETE | **57256809**: **0.957** |
| **B** | `knee-pub9576-v1arm` v2 | A + our ten-model v1 ensemble as a 25 % rank vote | COMPLETE | **57256817**: **0.956**, so our vote hurts this base and is dropped |
| **C** | `knee-pub9576-s6` v1 | A + nartaa's public **S6** ConvNeXt as a fourth model (R6 30 %, S6 20 %, Raptor 50 %) | COMPLETE | **57468079**: **0.957** = A, so S6 is closed |
| **D** | `knee-pub9576-s6-v1arm` v1 | C + our v1 vote at 25 % | COMPLETE | not submitted (B says the vote hurts) |
| **E** | `knee-pub9576-rw` v1 | A + the four per-finding Raptor weights behind the public 0.958 (Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35; E159) | COMPLETE | **57468070**: **0.958** |
| **F** | `knee-pub9576-f` v1 | E with heliosli's 0.959 weights + goodpjw's 2.5D reader at 0.3–3.5 % (E160) | COMPLETE | 57476951, pending |
| **G** | `knee-pub9576-g` v1 | E with heliosli's 0.959 weights only (E160) | COMPLETE | 57476961, pending |

Decision bands, written **before** any score (E158):
- **B vs A:** +0.002 or more means our vote pays, so try W = 0.35. +0.001 means hold and re-test. A tie means close it. −0.001 or less means drop it.
- **C vs A:** +0.001 or more means C becomes the base. A tie means keep A. −0.001 or less means drop S6.
- **D vs C:** read with B's bands.
- **E vs A:** 0.958 or more means the reweight carries to our draw. It may be fitted to the public board,
  so E is a final pick only **alongside** an unweighted one (A, or C).

Every appended stage **fails safe**: if S6 or our vote errors, times out or is
rejected, the notebook writes exactly the file from the stage before. A hard
time guard keeps the run under Kaggle's 9 h cap. Five independent reviewers checked
this before C/D were pushed (E158). They found one real defect, now fixed.

## 3. What I would most like you to check (highest value first)

### 3.1 Which two submissions to select for the private board ⭐
This is the decision with the most leverage left, and it is a judgment call.

- The public board is ~390 studies and the private is ~910. On our own pipeline,
  re-running with only the seed changed moved the score by up to 0.0012
  (E146). So public gaps of 0.001 between A–D are **not** evidence that one will
  win on private.
- The rule we fixed before the F/G scores (E160): **(1)** the best public scorer among A–G, and **(2)** the best
  candidate that differs most from (1). For example, if (1) is plain A, (2)
  should carry S6 or our vote, so that one bad bet cannot sink both picks.
- If we select nothing, Kaggle auto-selects. Please say what you'd pick once
  the scores are in, and why.

### 3.2 Rules check on external data ⚠️
The public pipeline's credits list training data from outside the competition:
- **OAI** (Osteoarthritis Initiative), via the NIMH Data Archive. That is
  *controlled access*: you register, then get approved.
- **fastMRI** (the R7 model's auxiliary data).
- **SKM-TEA** (the anatomy V-Net heliosli's code uses).

This applies to the **0.956 already on the board** and to all of A–D, and to the
~1,400 teams using the same notebooks. Please read the competition's **Rules →
External Data** section and any host answers in the discussion forum. Then tell
us whether any of it risks disqualification. If it does, our safe fallback is the
0.945 pipeline (CC0 CoAtNets + our own models).

### 3.3 The dataset swap in A–D
heliosli's dataset `heliosli/rsna-knee-r6-frozen-fe055406` is "frozen" by name
but **now holds the older R7 model**. Our first two runs died on the notebook's
own check: bundle `8c7b666e…` where `fe055406…` was expected. We mount
`rostigor/rsna-knee-r6-fe055406` instead. Its `manifest.json` has sha256
`6508ce6921eda986…c648`, **the exact hash heliosli's notebook names** for the
scored R6 bundle. The notebook checks every model file against that manifest, and
we added an assert on the manifest hash itself. Please sanity-check this
reasoning. We now depend on a third party's copy.

### 3.4 Code review of what we added
[`appended-cells.md`](appended-cells.md) has the three cells we append (S6
stage, four-model blend, our v1 vote), about 150 lines. Our blend helper is
`kaggle/_runners/v1arm_blend.py`. Look for anything that could make the hidden
re-run fail or silently write the wrong file.

## 4. How you can help hands-on

**First, please confirm you are on our Kaggle team**: the competition's *Team*
tab lists the members. The merge deadline was 15 Oct, and Kaggle forbids
sharing private notebooks or data outside a team. The options below assume
you are on it. If you are not, §3 is still fully in reach: everything it needs
is in this repo.

1. **Be the second pair of hands on submissions.** Kaggle will not let an API
   submit to this competition, so every submission is a human click. If the
   four notebooks are shared with you (`martijnlustig`), you can submit when
   `achelijndiamantidis` is away. The team shares one daily limit of 5, so coordinate who
   submits.
2. **Spare GPU.** Your account has its own weekly GPU quota and its own 2
   concurrent sessions. The next queued experiment is W = 0.35 for our vote
   if B pays. You can run it from a copy of B; it also needs our private
   dataset `achelijndiamantidis/knee-v1arm-addon` shared with you.
3. **Scout.** No GPU needed. Check the competition's *Code* tab, sorted by
   score, once a day for a public notebook that beats 0.9576 or adds a model
   family we do not have. Note it in `COMMENTS.md`.

## 5. Ground rules

- Never commit a Kaggle API token, and never commit per-study predictions or
  anything else derived from competition data; `eda/preflight.sh` checks the
  latter.
- Commit as `existentialistlogarithmic <achilleasd278@gmail.com>`, the same
  as the rest of the repo.
- `kaggle/*/run.py` is generated by `python eda/generate_kernels.py --write`.
  Don't edit it by hand.
