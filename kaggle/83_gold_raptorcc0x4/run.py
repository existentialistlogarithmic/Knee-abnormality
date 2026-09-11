"""Run the public CC0 CoAtNet arm in OUR kernel, from CC0 weights we mount ourselves.

SOURCING. Read this before changing anything below.

The model, its geometry and its slot layout are NOT this project's work. They
come from a publicly shared notebook and a set of publicly shared weights: The model, its geometry and its slot layout are NOT this project's work. They
# come from a publicly shared notebook and a set of publicly shared weights:
#
  notebook  kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model
  training  kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model
  weights   kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan          (CC0-1.0)
            kaggle.com/datasets/dreaddevelopment/raptor-knee-native384        (CC0-1.0)
            kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense   (CC0-1.0)
  labels    kaggle.com/datasets/dreaddevelopment/rsna-knee-labels             (CC0-1.0)
  blend     the four sub-model weights are published in
            kaggle.com/code/nathanjacob/4-arm-ensemble-explained-rsna-knee-0-937

The competition rules permit this: "It's okay to share code if made available
to all Participants on the forums." Every weight file mounted here is CC0-1.0,
a public-domain dedication, verified by reading each dataset's licence on
2026-09-08 and recorded in `eda/public_asset_licences.json` (E088). Nothing in
E043's `not-declared` tier is mounted, and the `tonylica` asset that gates the
0.937 notebook's DINOv3 arm is deliberately NOT here — that arm is excluded.

WHY OUR OWN KERNEL RATHER THAN A FORK. Upstream persists no output files, so
its predictions cannot be mounted (E085 tried and the blend correctly refused
with one member). The weights, however, ARE published as datasets, so the
honest route is to mount those and run them here with the attribution above
rather than to copy a notebook and quietly drop the credit.

WHAT IS OURS: the mount guard, the manifest constants, and the refusal to run
on a partial set of weights. Everything else is theirs.

WHAT THIS IS AND IS NOT EVIDENCE OF. Upstream self-reports 0.9167 macro AUC on
the 58 gold with gold held out, and 0.924 on the board from a single model.
Those are the author's numbers on the author's split. E048's comparability rule
has now failed twice as a predictor of what pays (E081 offline, E087 on the
board), and four unions in this log returned +0.0046, +0.0022, +0.0036 and
+0.0027, none separated.

So run the SINGLE ARM FIRST and submit it on its own. It is the control: if it
does not come back near its self-reported score, every later blend has an
untrustworthy member and no reading of the blend means anything. That ordering
is E039's rule applied to someone else's work.
"""
import gc
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOT A SUBMISSION. Scores the four CC0 CoAtNet arms on this
# project's own 58 expert studies and dumps their raw per-study
# probabilities, so blend weights can be searched offline instead
# of on the board at 2.4 GPU-h and one submission per question.
#
# It writes no submission.csv on purpose: these 58 studies are
# TRAINING data. A notebook with no submission.csv cannot be
# submitted, which is the safe failure.
#
# COST: 58 studies against 82's 1,300, same four arms, so about
# a tenth of 82's scored time plus the same fixed setup. Run it,
# do not submit it, and download gold_probs.csv.
#
# WHAT TO READ FIRST in the log: the `[gold]` lines. If
# maxspan-v5 does not land near 0.9214 the self-reports do not
# transfer and E097's +0.004 was bought on a premise this
# project never checked.
#
# ATTRIBUTION: as `knee-infer-raptorcc0`.
#
MEMBERS_EXPECTED = 4
ARMS             = ({'name': 'maxspan-v5', 'file': 'raptor_ft_coatnet_v5_full_swa.pt', 'img': 336, 'slots': (('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)), 'span': (0.02, 0.98), 'k_eval': 62, 'reverse': False, 'w': 0.55, 'expect_gold': 0.9214}, {'name': 'native384dense-v10', 'file': 'raptor_ft_coatnet_v10_full.pt', 'img': 384, 'slots': (('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)), 'span': (0.02, 0.98), 'k_eval': 62, 'reverse': False, 'w': 0.1, 'expect_gold': 0.9174}, {'name': 'maxspan-v5-reverse', 'file': 'raptor_ft_coatnet_v5_full_swa.pt', 'img': 336, 'slots': (('Sagittal', 1, 18), ('Sagittal', 0, 14), ('Coronal', 1, 12), ('Coronal', 0, 8), ('Axial', -1, 12)), 'span': (0.02, 0.98), 'k_eval': 62, 'reverse': True, 'w': 0.15, 'expect_gold': 0.9214}, {'name': 'native384-v8', 'file': 'raptor_ft_coatnet_v8_full_swa.pt', 'img': 384, 'slots': (('Sagittal', 1, 12), ('Sagittal', 0, 10), ('Coronal', 1, 8), ('Coronal', 0, 6), ('Axial', -1, 8)), 'span': (0.06, 0.94), 'k_eval': 42, 'reverse': False, 'w': 0.2, 'expect_gold': 0.9067})
CROP_MM          = 140.0
LAB              = ('ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture')
FALLBACK_LIMIT   = 0.02
DECODE_AHEAD     = 32
EVAL_SPLIT       = "gold"
GOLD_EXPECTED    = 58
V1_MEMBERS       = None


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ============================================================================
# Model. Structure is fixed by the checkpoints: every file carries `arch`, `res`
# and a state dict whose head is norm + att + clsW/clsb, so the architecture is
# read from the file rather than assumed.
# ============================================================================
def build_backbone(arch, pretrained=False):
    # coatnet/maxvit/convnext are conv-attention hybrids: no CLS token and no
    # interpolatable pos-embed, so they pool by average. The substring "vit"
    # inside "coatnet"/"maxvit" must NOT route them down the ViT path.
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = {"pretrained": pretrained, "num_classes": 0, "in_chans": 3}
    kw.update(global_pool="token", dynamic_img_size=True) if is_vit else kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    """Per-finding attention pooling over slice windows.

    Each of the twelve findings gets its own attention map across windows, so a
    cruciate tear visible on two slices and osteoarthritis spread across many do
    not compete for one shared pooling weight. This project built the same idea
    (E057/E058) and could never resolve it against a +/-0.03 noise floor; these
    weights are what it looks like trained to convergence by someone else.
    """

    def __init__(self, backbone, F_dim, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        self.n = n

    def forward(self, x):
        b, k = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1)).view(b, k, -1)
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck["arch"]
    res = int(ck["res"])
    backbone = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(backbone, F_dim=backbone.num_features)
    # strict=True on purpose: a silently partial load would produce a random head
    # on top of a real backbone and still submit. That is the failure mode
    # MEMBERS_EXPECTED exists to stop on the mounting side.
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    gold = ck.get("gold_auc")
    gold = None if gold is None else round(float(gold), 4)
    print(f"  {os.path.basename(path)}: {arch} res {res} author's gold {gold}", flush=True)
    del ck
    gc.collect()
    return model, res, arch, gold


# ============================================================================
# Windowing and inference
# ============================================================================
def _eval_centers(mask, depth, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, depth))
    lo, hi = int(valid.min()), int(valid.max())
    centers = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not centers:
        centers = [max(1, min((lo + hi) // 2, depth - 2))]
    idx = np.linspace(0, len(centers) - 1, k).round().astype(int)
    return [centers[i] for i in idx]


def eval_windows(vol, mask, k, res):
    depth = vol.shape[0]
    centers = _eval_centers(mask, depth, k)
    wins = np.empty((len(centers), 3, res, res), np.float32)
    for j, c in enumerate(centers):
        c = max(1, min(c, depth - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear", align_corners=False)[0]
        wins[j] = t.numpy()
    return (torch.from_numpy(wins) - _MEAN) / _STD


@torch.no_grad()
def infer_probs(model, xwins, device, reverse):
    # THE "REVERSE" MEMBER, AND WHAT IT IS NOT. Upstream's prose calls it a
    # "horizontal flip", and the published summary tables repeat that. The code
    # does `xw.flip(1)` on a tensor shaped (K, 3, H, W), so dim 1 is the THREE
    # NEIGHBOURING SLICES, not width: it reverses the slice triplet (c-1, c, c+1
    # -> c+1, c, c-1) and leaves the image geometry untouched.
    #
    # This project first implemented the prose — flip(-1), a true left-right
    # mirror — which is a different operation and a far larger distribution
    # shift for a model never trained on mirrored knees. Caught before the first
    # push by reading the reference implementation instead of its description.
    x = xwins.flip(1).contiguous() if reverse else xwins
    x = x.unsqueeze(0).to(device)
    if str(device).startswith("cuda"):
        try:
            # fp16 conv is fully cuDNN-supported on T4; bf16 is not ("no engine").
            with torch.autocast("cuda", dtype=torch.float16):
                return torch.sigmoid(model(x).float())[0].cpu().numpy()
        except RuntimeError:
            torch.cuda.empty_cache()
    return torch.sigmoid(model(x).float())[0].cpu().numpy()


def rankpct(x):
    """Per-column percentile rank in [0, 1]. Rank-mean, never probability-mean."""
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Study construction from DICOM
# ============================================================================
def auc(y, p):
    """Mann-Whitney AUC with average ranks for ties.

    Written out rather than imported so the gold split scores through arithmetic
    this repo owns and its tests pin. `rankpct` above already does the ranking
    the blend needs; this reuses the same tie convention so a member's rank in
    the blend and its rank in the score cannot disagree.
    """
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    npos = float((y == 1).sum())
    nneg = float((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    # Average the ranks inside every tied run. Without this a member that
    # returns the same probability for many studies scores by argsort order,
    # which is arbitrary and flattering.
    ps = p[order]
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def macro_auc(truth, pred):
    per = [auc(truth[:, k], pred[:, k]) for k in range(truth.shape[1])]
    return float(np.mean(per)), per


def _make_reader():
    import cv2
    import pydicom
    try:
        # pydicom 3.x moved this and 4.0 removes the old path. Upstream imports
        # the old one only; on an image with pydicom 4.x that raises inside the
        # reader, which is to say on the first study, an hour into the session.
        from pydicom.pixels import apply_modality_lut
    except ImportError:
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        recs, spacings = [], []
        for f in glob.glob(sdir + "/*.dcm"):
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, "ImageOrientationPatient", None)
                ipp = getattr(h, "ImagePositionPatient", None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                    pos = float(np.dot(np.array(ipp, float), normal))
                else:
                    pos = float(getattr(h, "InstanceNumber", 0) or 0)
                ps = getattr(h, "PixelSpacing", None)
                ps = float(ps[0]) if ps is not None else 0.5
                spacings.append(ps)
                recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda r: r[0])
        median_ps = float(np.median(spacings)) if spacings else 0.5
        return [(f, ps) for _, f, ps in recs], median_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps, img):
        h, w = a.shape
        cpx = min(int(round(CROP_MM / max(ps, 1e-3))), min(h, w))
        y0, x0 = (h - cpx) // 2, (w - cpx) // 2
        return cv2.resize(a[y0:y0 + cpx, x0:x0 + cpx], (img, img), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series(rows, plane, fluid, used):
    cands = [r for r in rows if r["Anatomical_Plane"] == plane
             and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_raptor_study(sid, series, tsdir, reader, img, slots, span):
    """Fill the fixed slots into one (maxs, img, img) uint8 stack.

    GEOMETRY IS PER ARM, not shared. The four published sub-models disagree on
    every one of these: `maxspan-v5` caches at 336 px over a 64-slice layout
    spanning 0.02-0.98, `native384-v8` at 384 px over a 44-slice layout spanning
    0.06-0.94. An earlier version of this file applied one geometry to all four
    — and the one it applied was the *v4 legacy* config that none of them uses.
    That is the silent train/inference skew `HANDOFF.md` §4d warned about,
    reached by reading the v4 notebook and assuming the family shared its
    constants.
    """
    order_and_meta, read_px, mm_crop_resize = reader
    span_lo, span_hi = float(span[0]), float(span[1])
    maxs = sum(int(s[2]) for s in slots)
    rows = series.get(sid, [])
    vol = np.zeros((maxs, img, img), np.uint8)
    idx, used = 0, set()
    for plane, fluid, k in slots:
        rec = _pick_series(rows, plane, int(fluid), used)
        if rec is None:
            idx += int(k)
            continue
        used.add(rec["SeriesInstanceUID"])
        files, median_ps = order_and_meta(f"{tsdir}/{sid}/{rec['SeriesInstanceUID']}")
        if not files:
            idx += int(k)
            continue
        # Wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices a narrower crop throws away. Must match the corpus
        # these weights were trained on.
        n = len(files)
        lo, hi = int(n * span_lo), int(n * span_hi) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, int(k)).round().astype(int) if n > 1 else [0] * int(k)
        arrs, spacings = [], []
        for p in picks:
            fp, ps = files[min(int(p), n - 1)]
            try:
                arrs.append(read_px(fp))
                spacings.append(ps)
            except Exception:
                arrs.append(None)
                spacings.append(median_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            loq, hiq = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, spacings, strict=True):
            if idx >= maxs:
                break
            if a is None:
                idx += 1
                continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            vol[idx] = (mm_crop_resize(aw, ps if ps > 0 else median_ps, img) * 255).astype(np.uint8)
            idx += 1
        if idx >= maxs:
            break
    return vol, (vol.reshape(maxs, -1).sum(1) > 0).astype(np.uint8)


# ============================================================================
# Mounting, with the guard that has already earned its keep twice
# ============================================================================
def find_test_root():
    for c in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.isdir(c):
            return c
    for d, _, files in os.walk("/kaggle/input"):
        if "test.csv" in files and "sample_submission.csv" in files:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight(fname):
    for d in sorted(glob.glob("/kaggle/input/*/")):
        for root, _, files in os.walk(d):
            if fname in files:
                return os.path.join(root, fname)
    raise RuntimeError(f"{fname} not mounted under /kaggle/input")


def main():
    import pandas as pd
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device} | torch {torch.__version__} | timm {timm.__version__}", flush=True)

    # A kernel that never ran mounts as an EMPTY directory, and a missing weight
    # file would otherwise silently drop an arm and submit a smaller ensemble
    # under a larger claim. MEMBERS_EXPECTED stopped exactly that in E084, so
    # resolve every declared arm BEFORE any inference is spent.
    if len(ARMS) != MEMBERS_EXPECTED:
        raise RuntimeError(f"expected {MEMBERS_EXPECTED} arms, manifest declares {len(ARMS)}")
    paths = {a["file"]: find_weight(a["file"]) for a in ARMS}
    expect = {a["file"]: a["expect_gold"] for a in ARMS}
    for arm in ARMS:
        maxs = sum(int(s[2]) for s in arm["slots"])
        if arm["k_eval"] > maxs - 2:
            # `_eval_centers` can only place a window at c with c-1 and c+1 in
            # range, so asking for more windows than the stack holds silently
            # repeats slices instead of failing.
            raise RuntimeError(f"{arm['name']}: k_eval {arm['k_eval']} > {maxs - 2} usable centres")
    print(f"arms {len(ARMS)} | distinct checkpoints {len(paths)}", flush=True)

    for arm in ARMS:
        print(f"  {arm['name']:22s} w={arm['w']:.2f} img={arm['img']} "
              f"slices={sum(int(s[2]) for s in arm['slots'])} span={tuple(arm['span'])} "
              f"k_eval={arm['k_eval']} reverse={arm['reverse']}", flush=True)

    root = find_test_root()
    # TWO SPLITS, ONE INFERENCE PATH. `test` is the hidden set and writes a
    # submission; `gold` is the 58 expert-labelled TRAINING studies and writes a
    # scored dump instead. They share every line of preprocessing and every
    # forward pass below deliberately: an instrument that measures these arms
    # through a different code path than the one that submits them measures a
    # different system, and E090 recorded that this project has no offline gold
    # number for the CoAtNet arms at all — only their authors' self-reports.
    #
    # The gold split needs NO extra dataset. The competition's own `train.csv`
    # carries the twelve findings, and exactly 58 of its 4,407 rows have all
    # twelve filled in; the rest are blank. That set identity is asserted below
    # rather than assumed, because "58" arriving as 57 or 4,407 would still
    # produce a plausible-looking AUC.
    if EVAL_SPLIT == "gold":
        tsdir = root + "/train_series"
        if not os.path.isdir(tsdir):
            tsdir = root + "/train_images"
        tr = pd.read_csv(root + "/train.csv")
        tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
        labelled = tr[list(LAB)].notna().all(axis=1)
        gold = tr[labelled].reset_index(drop=True)
        if len(gold) != GOLD_EXPECTED:
            raise RuntimeError(
                f"train.csv has {len(gold)} fully-labelled studies, expected "
                f"{GOLD_EXPECTED}. The expert set this project scores on has "
                f"changed shape; every historical gold number is about a "
                f"different instrument until that is understood.")
        ids = gold["StudyInstanceUID"].tolist()
        truth = gold[list(LAB)].to_numpy(np.float64)
        # Both classes must be present per finding or AUC is undefined, and an
        # undefined column silently poisons the macro.
        degenerate = [f for k, f in enumerate(LAB)
                      if truth[:, k].min() == truth[:, k].max()]
        if degenerate:
            raise RuntimeError(f"single-class findings in gold: {degenerate}")
        tser = pd.read_csv(root + "/train_series.csv")
    else:
        tsdir = root + "/test_series"
        if not os.path.isdir(tsdir):
            tsdir = root + "/test_images"
        test = pd.read_csv(root + "/test.csv")
        test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
        ids = test["StudyInstanceUID"].tolist()
        truth = None
        tser = pd.read_csv(root + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    tser = tser[tser["StudyInstanceUID"].isin(set(ids))]
    series = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    missing = [s for s in ids if s not in series]
    if missing:
        raise RuntimeError(f"{len(missing)} studies have no series rows, first {missing[0]}")
    print(f"split {EVAL_SPLIT} | studies {len(ids)} | series {len(tser)}", flush=True)

    cols = ["StudyInstanceUID"] + list(LAB)
    ssub = os.path.join(root, "sample_submission.csv")
    if os.path.exists(ssub):
        cols = list(pd.read_csv(ssub, nrows=1).columns)

    reader = _make_reader()
    probs = [np.full((len(ids), len(LAB)), 0.5, np.float32) for _ in ARMS]
    ran = [None] * len(ARMS)
    # Setup — imports, CUDA init, the first model load — is a FIXED cost, not a
    # per-study one. On the 3-study visible stub it was 910 s of a 935 s run, so
    # dividing wall clock by studies projected 112 h instead of 3. Scale only the
    # per-study work and report setup beside it.
    setup_seconds = time.time() - t0
    scored_seconds = 0.0

    # Grouped so nothing is recomputed that two arms can share. Outer key is the
    # checkpoint, so each file is loaded ONCE and peak RAM stays at one model
    # (upstream hit a system-RAM OOM holding two). Inner key is the
    # preprocessing signature, so `maxspan-v5` and `maxspan-v5-reverse` — same
    # file, same geometry, differing only in the slice reversal — share one
    # build and one window tensor and cost one extra forward rather than a whole
    # second pass. Arms whose geometry differs get their own build, because it
    # must: that is the bug this structure exists to make impossible.
    for fname, path in paths.items():
        arms_here = [i for i, a in enumerate(ARMS) if a["file"] == fname]
        model, res, arch, gold = load_model(path, device)
        # VERIFY WHAT WE GOT, don't assume. Another account owns these files and
        # can re-upload different weights under the same name at any time; the
        # `gold_auc` each one carries is a fingerprint of the exact artefact this
        # manifest was written against. A mismatch means the arm is not the model
        # the notes describe, and every number downstream would be about
        # something else.
        if gold != expect[fname]:
            raise RuntimeError(
                f"{fname}: checkpoint reports gold_auc {gold}, manifest expects "
                f"{expect[fname]}. Upstream changed the file, or the wrong one is "
                f"mounted. Re-read it before trusting anything this run produces.")
        groups = {}
        for i in arms_here:
            a = ARMS[i]
            groups.setdefault((a["img"], a["slots"], tuple(a["span"]), a["k_eval"]), []).append(i)
        for (img, slots, span, k_eval), members in groups.items():
            names = "+".join(ARMS[i]["name"] for i in members)
            print(f"[{names}] img {img} span {span} k_eval {k_eval} res {res}", flush=True)
            bad = 0
            forward_seconds = 0.0

            # DECODE ON THREADS SO IT OVERLAPS THE GPU. E091 measured 8.27 s per
            # study single-threaded, which projects to 2.99 h for ONE arm and put
            # the four-arm blend over the 9 h cap. This project already hit the
            # same wall once and solved it the same way: `infer.py.in` records
            # 19.7 s/study single-threaded, 7.1 h projected, fixed by building
            # volumes on a pool while the GPU works.
            #
            # Threads decode to a uint8 volume (~7 MB) and the MAIN thread turns
            # it into windows. Returning the float32 window tensor instead would
            # be ~110 MB a study, and a pool running ahead of the GPU would then
            # hold gigabytes of them.
            def build_one(item, _img=img, _slots=slots, _span=span):
                j, sid = item
                try:
                    vol, mask = build_raptor_study(sid, series, tsdir, reader, _img, _slots, _span)
                    if not mask.any():
                        # No slot filled, and no exception raised. Without this the
                        # study contributes a 0.5 row that looks like a prediction;
                        # series selection failing everywhere is how a schema
                        # change would present.
                        return j, sid, None, None, "empty volume: no series matched any slot"
                    return j, sid, vol, mask, None
                except Exception as exc:                                  # noqa: BLE001
                    return j, sid, None, None, f"{type(exc).__name__}: {exc}"

            # WARM THE GPU BEFORE TIMING ANYTHING. `cudnn.benchmark` autotunes a
            # convolution algorithm the first time it sees a shape, and every
            # study here has the identical (k_eval, 3, res, res). On the 3-study
            # visible stub that one-off tuning lands inside the per-study average
            # and inflates it; on 1,300 studies it is invisible. Paying it here
            # makes the projection honest in both directions — and it is not free
            # accounting, the real run genuinely starts sooner.
            warm_t0 = time.time()
            infer_probs(model, torch.zeros(k_eval, 3, res, res), device, False)
            print(f"  [{names}] warmed cudnn in {time.time() - warm_t0:.1f}s "
                  f"(paid once, excluded from the per-study rate)", flush=True)
            group_t0 = time.time()

            workers = max(2, min(8, (os.cpu_count() or 4)))
            print(f"  [{names}] decoding on {workers} threads, {DECODE_AHEAD} studies ahead",
                  flush=True)
            indexed = list(enumerate(ids))
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # Chunked rather than one `map` over every study: `Executor.map`
                # submits all of them at once and holds every result until it is
                # consumed, so on 1,300 studies the decoded volumes would pile up
                # faster than the GPU drains them. A bounded window keeps decode
                # ahead of compute without letting it run away.
                for start in range(0, len(indexed), DECODE_AHEAD):
                    for j, sid, vol, mask, error in pool.map(
                            build_one, indexed[start:start + DECODE_AHEAD]):
                        done += 1
                        if error is not None:
                            # Never drop a study: a 0.5 row ranks mid-pack, a
                            # missing row fails the submission outright. But see
                            # the limit below — a few is robustness, all of them
                            # is a broken kernel that would submit 0.500 and look
                            # like it worked.
                            bad += 1
                            if bad <= 20:
                                print(f"  [{names}] study {j} {sid[:16]} FALLBACK "
                                      f"({error})", flush=True)
                        else:
                            xw = eval_windows(vol, mask, k=k_eval, res=res)
                            forward_t0 = time.time()
                            for i in members:
                                probs[i][j] = infer_probs(model, xw, device,
                                                          bool(ARMS[i]["reverse"]))
                            if str(device).startswith("cuda"):
                                # Otherwise this times queueing, not work.
                                torch.cuda.synchronize()
                            forward_seconds += time.time() - forward_t0
                            del vol, mask, xw
                        if done % 100 == 0 or done == len(ids):
                            # The hidden set is ~1,300 studies (FINDINGS 2.12)
                            # against a 9 h cap. A run that will not fit should be
                            # visible at study 100, not at hour eight.
                            rate = (time.time() - group_t0) / done
                            print(f"  [{names}] {done}/{len(ids)} | "
                                  f"{time.time() - t0:.0f}s | {rate:.2f}s/study, "
                                  f"this group projects {rate * 1300 / 3600:.2f} h "
                                  f"on 1,300", flush=True)
            group_seconds = time.time() - group_t0
            scored = max(1, len(ids) - bad)
            print(f"  [{names}] forward {forward_seconds:.1f}s "
                  f"({forward_seconds / scored:.2f}s/study), everything else "
                  f"(decode, threaded and overlapped) "
                  f"{max(group_seconds - forward_seconds, 0) / scored:.2f}s/study",
                  flush=True)
            scored_seconds += group_seconds
            for i in members:
                ran[i] = {"name": ARMS[i]["name"], "file": fname, "arch": arch,
                          "author_gold_auc": gold, "res": res, "img": img,
                          "slices": sum(int(x[2]) for x in slots),
                          "span": list(span), "k_eval": k_eval,
                          "reverse": bool(ARMS[i]["reverse"]), "w": ARMS[i]["w"],
                          "group_seconds": round(group_seconds, 1),
                          "forward_seconds": round(forward_seconds, 1),
                          "fallbacks": bad}
            rate = bad / max(1, len(ids))
            print(f"[{names}] fallbacks {bad}/{len(ids)} ({rate:.1%}) | "
                  f"{group_seconds:.0f}s", flush=True)
            if rate > FALLBACK_LIMIT:
                # THE GUARD THAT MATTERS MOST HERE. Every failure mode this kernel
                # has — a renamed column in the hidden test's series table, a
                # missing series directory, a DICOM the reader cannot open — lands
                # in the same `except` and produces a well-formed submission of
                # constant 0.5, which scores 0.500 and is indistinguishable from
                # "the model is bad" on the leaderboard. E084's MEMBERS_EXPECTED
                # stopped that on the mounting side; this stops it on the data
                # side. Refuse to write rather than submit a coin flip.
                raise RuntimeError(
                    f"{names}: {rate:.1%} of studies fell back, limit is "
                    f"{FALLBACK_LIMIT:.1%}. Refusing to write a submission that "
                    f"would score like a coin flip and read like a result.")
        del model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- THE SECOND ARCHITECTURE'S PASS ------------------------------------
    v1_probs = None

    weights = np.array([float(a["w"]) for a in ARMS], dtype=np.float64)
    weights = weights / weights.sum()
    named = dict(zip([a["name"] for a in ARMS], weights.round(4), strict=True))
    print(f"[blend] weighted rank-mean {named}", flush=True)
    ranks = np.tensordot(weights, np.stack([rankpct(np.clip(p, 0, 1)) for p in probs]), axes=(0, 0))
    ranks[~np.isfinite(ranks)] = 0.5

    scores = None
    if EVAL_SPLIT == "gold":
        # NO submission.csv IS WRITTEN HERE, AND THAT IS THE POINT. These 58
        # studies are training data; a submission built from them would be
        # scored against the hidden set it does not contain. A notebook with no
        # submission.csv simply cannot be submitted, which is the safe failure.
        #
        # What it writes instead is every arm's RAW probabilities, per study, so
        # that blend weights can be searched offline for nothing. Until now every
        # blend question in this project cost a board submission — E097 spent one
        # to learn that four arms beat one by +0.004 — and E098 closed six blends
        # on an instrument that had never once seen a blend gain.
        scores = {"split": "gold", "n": len(ids), "arms": {}}
        for i, a in enumerate(ARMS):
            m, per = macro_auc(truth, probs[i])
            scores["arms"][a["name"]] = {
                "macro": round(m, 4),
                "author_gold_auc": a["expect_gold"],
                "delta_vs_author": round(m - float(a["expect_gold"]), 4),
                "per_finding": {f: round(per[k], 4) for k, f in enumerate(LAB)}}
            print(f"[gold] {a['name']:22s} macro {m:.4f}  "
                  f"author says {a['expect_gold']}  "
                  f"delta {m - float(a['expect_gold']):+.4f}", flush=True)
        bm, bper = macro_auc(truth, ranks)
        best = max(v["macro"] for v in scores["arms"].values())
        scores["blend"] = {"macro": round(bm, 4),
                           "best_single_arm": round(best, 4),
                           "gain_over_best_single": round(bm - best, 4),
                           "weights": {k: float(v) for k, v in named.items()},
                           "per_finding": {f: round(bper[k], 4) for k, f in enumerate(LAB)}}
        print(f"[gold] BLEND macro {bm:.4f} | best single {best:.4f} | "
              f"gain {bm - best:+.4f}", flush=True)
        rows = []
        for i, a in enumerate(ARMS):
            d = pd.DataFrame(probs[i].astype(np.float32), columns=list(LAB))
            d.insert(0, "arm", a["name"])
            d.insert(0, "StudyInstanceUID", ids)
            rows.append(d)
        out = pd.concat(rows, ignore_index=True)
        assert len(out) == len(ids) * len(ARMS)
        out.to_csv("/kaggle/working/gold_probs.csv", index=False)
        truth_df = pd.DataFrame(truth.astype(np.int8), columns=list(LAB))
        truth_df.insert(0, "StudyInstanceUID", ids)
        truth_df.to_csv("/kaggle/working/gold_truth.csv", index=False)
        Path("/kaggle/working/gold_scores.json").write_text(json.dumps(scores, indent=2))
        print(f"wrote gold_probs.csv rows={len(out)} and gold_scores.json", flush=True)
    else:
        sub = pd.DataFrame(ranks.astype(np.float32), columns=list(LAB))
        sub.insert(0, "StudyInstanceUID", ids)
        sub = sub[cols]
        assert list(sub.columns) == cols, "column order drift"
        assert sub["StudyInstanceUID"].tolist() == ids, "row identity drift"
        assert np.isfinite(sub[list(LAB)].to_numpy()).all()
        sub.to_csv("/kaggle/working/submission.csv", index=False)

    # Same shape as `infer`'s manifest so the two lineages can be compared
    # without reading logs. `prediction_spread` is the degenerate-model check:
    # a member that collapsed to one value per finding still writes a valid
    # submission, and the spread is where that shows.
    elapsed = time.time() - t0
    # Measured on the RAW probabilities, per arm, not on the rank-blended output.
    # `rankpct` maps any non-constant column onto 0..1, and argsort breaks ties
    # arbitrarily, so post-rank spread reads 1.0 even for a member that returned
    # the same value for every study — the exact failure it was meant to catch.
    spread = {ARMS[i]["name"]: {f: round(float(probs[i][:, k].max() - probs[i][:, k].min()), 4)
                               for k, f in enumerate(LAB)}
              for i in range(len(ARMS))}
    flat = [name for name, per in spread.items()
            if max(per.values()) < 1e-6 and len(ids) > 1]
    if flat:
        raise RuntimeError(
            f"{flat} returned a constant prediction for every study. That still "
            f"writes a valid submission, so it is refused here instead.")
    Path("/kaggle/working/infer_manifest.json").write_text(json.dumps({
        "n_studies": len(ids),
        "wall_clock_seconds": round(elapsed, 1),
        "setup_seconds": round(setup_seconds, 1),
        "scored_seconds": round(scored_seconds, 1),
        "seconds_per_study": round(scored_seconds / max(1, len(ids)), 3),
        "projected_hours_1300_studies":
            round(setup_seconds / 3600 + scored_seconds / max(1, len(ids)) * 1300 / 3600, 3),
        "n_arms": len(ARMS),
        "n_checkpoints": len(paths),
        "total_fallbacks": sum(r["fallbacks"] for r in ran if r),
        "fallback_limit": FALLBACK_LIMIT,
        "arms": ran,
        "eval_split": EVAL_SPLIT,
        "gold_scores": scores,
        "prediction_spread": spread}, indent=2))
    print(f"prediction spread: {spread}", flush=True)
    if EVAL_SPLIT != "gold":
        print(f"wrote /kaggle/working/submission.csv  rows={len(sub)}", flush=True)
    print(f"DONE {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
