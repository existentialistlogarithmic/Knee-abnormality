# The cells we append to the public 0.9576 notebook

Generated 2026-10-16 from the notebooks pushed as `knee-pub9576-s6-v1arm` v1. Each
block below was checked equal to the pushed cell before writing. Cells 0–8
are heliosli's published notebook, unchanged except for the two lines in §0.

- **A** = cells 0–8 · **B** = A + §3 · **C** = A + §1 + §2 · **D** = C + §3

## §0 — the dataset swap (cell 6, two lines; see README §3.3)

```python
C_SLUG = 'rsna-knee-r6-fe055406'   # [fix] rostigor's frozen copy: heliosli's 'frozen' dataset now holds R7 (8c7b666e)   # [d957] the frozen copy of the R6 bundle dataset (manifest sha256 6508ce6921eda986.. = 57068142's)
assert hashlib.sha256((M_SRC / 'manifest.json').read_bytes()).hexdigest() == '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648', 'not the scored R6 manifest'   # [fix]
```

## §1 — S6 stage (cell 9)

The `run_stage` helper and the stage logic are carried over from the public
four-model notebook (`laimerbotrolenk/rsna-four-model-ensemble`, cells 4 and
17), which in turn carries AlekseiProvorov's recipe. `S6_SRC_B64` holds that
notebook's embedded S6 script **byte for byte**: zlib + base64, about 66 kB.
It is omitted here. The decompressed script's sha256 is
`2e8c4dca6773588ee7a4246ba048e667c1e2516be1ec43a8b6e0269b5fa62c96`.
It is nartaa's `rsna-knee-s6fall-solo` with two changes. Its component A (the
Raptor-384) is read from this notebook's own `raptor_raw.csv` instead of being
recomputed. Its time budget comes from the `S6_GUARD_S` the cell passes in:
`GUARD_MAX_S` is read from it, with `A_TIMEOUT_S` = 60 and a `B_TIMEOUT_S`
floor of 3600.

```python
# [s6] Stage 4: nartaa's public all-data S6 ConvNeXt (nartaa/rsna-knee-s6fall-weights), run exactly as the public four-model
# notebook runs it (AlekseiProvorov's recipe, as carried by laimerbotrolenk/rsna-four-model-ensemble): the embedded script
# reuses this notebook's raptor_raw.csv as its component A, so only S6 itself is computed here.
# Optional: too little time or any failure keeps the three-model submission written above, byte for byte.
import base64, zlib, shutil, signal, threading
TOTAL_HOURS = 8.3
S6_PROBS = WORK / 'k1/B/B_probs.csv'
S6_OK, S6_SECONDS = False, None
shutil.copy(WORK / 'submission.csv', WORK / 'sub_base3.csv')


def run_stage(command, name, max_seconds, env=None):
    remaining = NB_T0 + TOTAL_HOURS * 3600 - time.time()
    if remaining < 60 or max_seconds < 60:
        raise RuntimeError('Insufficient time remaining for ' + name)
    with (WORK / (name + '.log')).open('w') as log:
        p = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        def pump():
            for line in p.stdout:
                log.write(line); log.flush()
                print(line, end='', flush=True)
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            rc = p.wait(timeout=min(max_seconds, remaining))
        except BaseException:
            try: os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            raise
        finally:
            reader.join(timeout=10)
        if rc:
            raise RuntimeError(f'{name} exited {rc}; see its log')


S6_SRC_B64 = "<embedded S6 script, omitted - see above>"

for directory in (WORK / 'k1', Path('/tmp/rsna_v33_s6')):
    if directory.exists():
        shutil.rmtree(directory)
_s6_left = NB_T0 + TOTAL_HOURS * 3600 - time.time() - 10 * 60
if _s6_left < 100 * 60:
    print('S6_SKIPPED: insufficient time (%.0f min left); the three-model submission stays' % (_s6_left / 60), flush=True)
else:
    _s6_t = time.time()
    try:
        Path('/tmp/s6_solo.py').write_bytes(zlib.decompress(base64.b64decode(S6_SRC_B64)))
        _env = dict(os.environ, S6_GUARD_S=str(int(_s6_left)),
                    K1_KAGGLE_ROOT='/kaggle', K1_TMP='/tmp/rsna_v33_s6')
        run_stage([sys.executable, '-u', '/tmp/s6_solo.py'], 'v33_s6', _s6_left + 120, env=_env)
        if not S6_PROBS.is_file() or not (WORK / 'k1/k1_receipt.json').is_file() or (WORK / 'k1/k1_error.json').exists():
            raise RuntimeError('S6 did not produce a clean completion receipt')
        S6_OK = True
    except Exception as e:
        print('S6_STAGE_FAILED', repr(e)[:400], flush=True)
    finally:
        if (WORK / 'submission.csv').exists():
            (WORK / 'submission.csv').replace(WORK / '_s6_solo_discarded.csv')
        shutil.copy(WORK / 'sub_base3.csv', WORK / 'submission.csv')
        S6_SECONDS = time.time() - _s6_t
        print('S6_STAGE_DONE', S6_OK, round(S6_SECONDS, 1), flush=True)
```

## §2 — four-model blend (cell 10)

```python
# [s6] Stage 5: the four-model blend, AlekseiProvorov's public formula with this notebook's R6 (flip TTA) in R7's place
# and this notebook's one-view Raptor-224 in place of his mirror-TTA 224 (so a skipped S6 leaves the 0.9576 file exactly):
#   Raptor half  = (1 - A224) x rank(Raptor-384) + A224 x rank(Raptor-224)
#   compact half = (1 - W_S6) x rank(R6) + W_S6 x rank(S6)
#   final        = W_RAPTOR x Raptor half + (1 - W_RAPTOR) x compact half      (R6 30 %, S6 20 %, Raptor 50 %)
# Weights fixed before any run, not fitted on any labels.  A member that failed on a study is dropped for that study and the
# rest renormalised.  S6 absent, unusable or rejected by the original sanity gate: the three-model submission above stays.
W_S6, W_RAPTOR = 0.40, 0.50
s6 = None
if S6_OK:
    try:
        s6 = pd.read_csv(S6_PROBS, dtype={'StudyInstanceUID': str})
        assert sorted(s6['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(s6.columns), 'S6 rows/columns differ'
        s6 = s6.set_index('StudyInstanceUID').loc[test_ids, labels]
    except Exception as e_:
        print('S6 member unusable, keeping the three-model submission:', repr(e_)[:300], flush=True)
        s6 = None
if s6 is not None:
    s6_fail = ~np.isfinite(s6.values.astype(np.float64)).all(axis=1)
    both = ~(s6_fail | our_fail)
    rho = [float(pd.Series(s6[c].values[both]).corr(pd.Series(ours[c].values[both]), method='spearman')) for c in labels] if both.sum() >= 20 else []
    print('S6 vs R6 spearman per finding:', dict(zip(labels, [round(x, 3) for x in rho])), flush=True)
    if rho and (np.nanmean(rho) < 0.6 or np.nanmin(rho) < 0.3) or s6_fail.sum() > 0.03 * n:
        print('S6 member rejected by the sanity check', flush=True)
        s6 = None


def _wavg(parts):
    num, den = np.zeros(n), np.zeros(n)
    for w_, r_ in parts:
        m_ = ~np.isnan(r_)
        num[m_] += w_ * r_[m_]
        den[m_] += w_
    o_ = np.full(n, np.nan)
    ok_ = den > 0
    o_[ok_] = num[ok_] / den[ok_]
    return o_


if s6 is None:
    print('S6_BLEND_SKIPPED: submission = the three-model blend', flush=True)
else:
    out4 = np.zeros((n, len(labels)))
    for j, c in enumerate(labels):
        r384, r224 = pct_on(raw[c].values, ~r384_fail), pct_on(raw224[c].values, ~r224_fail)
        raptor_half = _wavg([(1.0 - A224, r384), (A224, r224)])
        compact_half = _wavg([(1.0 - W_S6, pct_on(ours[c].values, ~our_fail)), (W_S6, pct_on(s6[c].values, ~s6_fail))])
        b = _wavg([(W_RAPTOR, raptor_half), (1.0 - W_RAPTOR, compact_half)])
        b[np.isnan(b)] = 0.5
        out4[:, j] = b
    sub4 = pd.DataFrame(out4.astype(np.float32), columns=labels)
    sub4.insert(0, 'StudyInstanceUID', test_ids)
    sub4 = sub4[cols]
    assert sub4['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub4[labels].values).all()
    assert ((sub4[labels].values >= 0) & (sub4[labels].values <= 1)).all()
    sub4.to_csv(WORK / 'submission.csv', index=False)
    print('S6_BLEND_APPLIED', json.dumps({'W_S6': W_S6, 'W_RAPTOR': W_RAPTOR, 'A224': A224, 's6_failed': int(s6_fail.sum()),
                                          's6_seconds': S6_SECONDS, 'total_seconds': round(time.time() - NB_T0, 1)}), flush=True)
```

## §3 — our v1 vote (last cell of B and D)

It runs `run_v1arm.py` (identical to `kaggle/118_infer_v1arm10_addon/run.py`)
and `v1arm_blend.py` (identical to `kaggle/_runners/v1arm_blend.py`). Both come
from our private dataset `achelijndiamantidis/knee-v1arm-addon`, together with
the ten model checkpoints.

```python
# [v1arm add-on] Rank-blend our ten-member v1 ensemble (dataset knee-v1arm-addon) into the submission written above:
# per finding, out = (1 - W) * rank(base) + W * rank(v1arm), W fixed before any run.
# Fails safe: add-on not mounted, too little time left before the 9 h cap, or any error -> the base submission stays.
import glob, os, shutil, subprocess, sys, time
W = 0.25
HARD_STOP_H = 8.6   # hours after NB_T0 by which the add-on must have finished
NEED_H = 1.1        # ten members measured at 0.70 h on 1,300 studies, plus model loading
WORK = '/kaggle/working'
_base, _v1 = WORK + '/sub_base.csv', WORK + '/sub_v1arm.csv'
shutil.copy(WORK + '/submission.csv', _base)
try:
    _runners = sorted(glob.glob('/kaggle/input/*/run_v1arm.py') + glob.glob('/kaggle/input/*/*/run_v1arm.py')
                      + glob.glob('/kaggle/input/*/*/*/run_v1arm.py'))
    if not _runners:
        raise RuntimeError('knee-v1arm-addon is not mounted')
    _addon = os.path.dirname(_runners[0])
    _left = HARD_STOP_H * 3600 - (time.time() - NB_T0)
    if _left < NEED_H * 3600:
        raise RuntimeError('only %.2f h left before the hard stop, the add-on needs %.1f h' % (_left / 3600, NEED_H))
    subprocess.run([sys.executable, '-u', _addon + '/run_v1arm.py'], check=True, timeout=_left, cwd=WORK)
    shutil.move(WORK + '/submission.csv', _v1)
    subprocess.run([sys.executable, '-u', _addon + '/v1arm_blend.py', _base, _v1, WORK + '/submission.csv', str(W)],
                   check=True, cwd=WORK)
    print('V1ARM_ADDON_APPLIED weight %.2f, total %.0f s' % (W, time.time() - NB_T0), flush=True)
except Exception as exc:
    shutil.copy(_base, WORK + '/submission.csv')
    print('V1ARM_ADDON_SKIPPED, base submission kept:', repr(exc), flush=True)
```
