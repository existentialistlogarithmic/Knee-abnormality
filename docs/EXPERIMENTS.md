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
