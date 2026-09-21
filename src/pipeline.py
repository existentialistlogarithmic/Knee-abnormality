"""Declarative description of the Kaggle pipeline — one source of truth.

Kaggle script kernels are single files, which is why this project accumulated
29 `run.py` files that were only 12 distinct programs. Measured rather than
guessed: `build_study` is byte-identical in 4 of them, `find_marker` in 7,
`read_series_volume` in 4. A bug in any of those had to be fixed everywhere it
had been pasted, and in practice it wasn't — `build_model` had **drifted into
two variants across 5 files**, the newer one applying ImageNet normalisation
and the older one not. The training kernel and the inference kernel that scores
its weights were on opposite sides of that split.

So the point of this file is not tidiness. Declaring the pipeline makes three
classes of bug unrepresentable rather than merely tested for.

**1. Geometry mismatch.** A cache is built at some mm/px, size and slice count,
and the model that trains on it and the kernel that runs inference must use
exactly those three numbers. Here a `Lineage` does not *have* a geometry — it
has a `Cache`, and the geometry belongs to the cache. Every kernel in the
lineage renders its constants from that one object, so a trainer cannot
disagree with the cache it reads. There is nothing to disagree with.

**2. Code drift between kernels.** The bodies live in `kaggle/_templates/` and
are spliced in at generation time (`@@INCLUDE@@`). `build_study` exists once.
Fixing it fixes fourteen kernels, because there is only one of it.

**3. Silently invalid ensembling.** Two properties of a trained model are
invisible in its weights and fatal if ensemble members disagree: how many
slices it saw (`slice_subsample`) and whether its input was ImageNet-normalised
(`input_norm`). Both are declared here, written into every checkpoint, and
re-checked by the inference kernel, which refuses to average models that were
fed differently rather than losing AUC quietly.

`input_norm` is False for the lineages that have already run. That is not a
preference — it is a record. The 0.725 leaderboard model was trained on raw
0..1 inputs, and rewriting history here would make the manifest a lie and the
ensemble guard useless.

The generator is `eda/generate_kernels.py`. `--check` fails when the tree and
the manifest disagree, which is what keeps this file honest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ACCOUNT owns the ASSETS -- the label datasets and the cache/trainer kernels that
# everything here mounts. It is NOT "whoever is running this".
ACCOUNT = "achelijndiamantidis"

# PUSH_ACCOUNT owns the kernel being CREATED, and it is the only one of the two a
# collaborator changes. A teammate lending GPU sets KAGGLE_PUSH_ACCOUNT to their
# own username: their kernels are then created under their account while still
# mounting OUR caches, datasets and trainers, which is the whole point -- the
# assets do not move and no credential changes hands.
#
# Why this is two names and not one. Setting a single ACCOUNT to a teammate's
# username would rewrite `kernel_sources` and every dataset id to
# `their-name/knee-cache-build-0`, which does not exist, and the kernel would
# fail at mount time with a message that looks like a permissions problem.
#
# PREREQUISITE, and it is a real one: a kernel or dataset can only be mounted by
# another account if it is PUBLIC or explicitly shared with that account. Ours
# are private by default (`is_private: true`), so anything a collaborator needs
# to mount has to be shared with them FIRST or their run dies at startup.
PUSH_ACCOUNT = os.environ.get("KAGGLE_PUSH_ACCOUNT", ACCOUNT)

# DEPENDS_ACCOUNT owns the kernels whose OUTPUTS get mounted -- the caches and
# trainers named in `depends`. It is a third name because it answers a third
# question, and a collaborator needs it set differently from the other two.
#
# Someone bootstrapping from a bare competition account builds their OWN caches,
# so their trainer must mount `their-name/knee-cache-build-0`. Someone who has
# been given access to ours leaves it alone and mounts ours. Folding this into
# PUSH_ACCOUNT would force the first case on everyone who sets a push account,
# and folding it into ACCOUNT would move the datasets too.
DEPENDS_ACCOUNT = os.environ.get("KAGGLE_DEPENDS_ACCOUNT", ACCOUNT)
COMPETITION = "rsna-knee-abnormality-detection"
# The two datasets a collaborator cannot be given before a team merge, because
# both are derived from competition data -- headers from the DICOMs, labels from
# the reports. Overridable so someone with their own competition access can point
# the pipeline at datasets THEY built and depend on nothing of ours:
#
#   KAGGLE_ARTIFACTS_DATASET=their-name/their-artifacts
#   KAGGLE_PUBLIC_DATASET=their-name/their-public-labels
#
# `kaggle/00_dicom_header_scan/` mounts the competition and NOTHING else, so the
# whole chain is reproducible from a bare competition account: header scan (CPU)
# -> package its `series_headers.parquet` as their artifacts dataset -> cache
# build (CPU) -> trainer (GPU). Nothing private of ours appears anywhere in it.
ARTIFACTS_DATASET = os.environ.get(
    "KAGGLE_ARTIFACTS_DATASET", f"{ACCOUNT}/knee-phase1-artifacts")
# Same headers, but soft_labels.parquet is the FUSION of the lexicon labeler
# and the LLM reader. A separate dataset rather than a new version of the
# one above, so every run already made stays comparable — replacing the
# labels in place would silently change what every earlier number meant.
FUSED_DATASET = f"{ACCOUNT}/knee-phase1-fused"
PUBLIC_DATASET = os.environ.get(
    "KAGGLE_PUBLIC_DATASET", f"{ACCOUNT}/knee-phase1-public")
DISTILLED_DATASET = f"{ACCOUNT}/knee-phase1-distilled"
# THE TWO ARMS OF E104, AND WHY THERE ARE TWO.
#
# `dreaddevelopment/rsna-knee-labels` (CC0) carries soft labels for 4,349
# studies -- every study except EXACTLY the 58 gold, verified as a set identity
# in E088. E089 scored the public label sets against gold and recorded this one
# as "unscoreable": zero overlap, so the labels cannot be compared to the expert
# values at all. That is true of the LABELS and it is where the log stopped.
#
# It is not true of a MODEL TRAINED ON THEM. A label set that excludes the gold
# by construction means a model trained on all of it has never seen a gold
# study, so gold-58 is a clean holdout for a model fitted on 98.7% of the
# corpus. This project has never had that: the 5-fold lineage is honest but each
# member misses a fifth of the data, and the full-fit lineage sees everything and
# cannot be scored at all (E083, and `knee-infer-v1pubfull5`'s own note).
#
# DREAD is that arm. PUB4349 is its control: the incumbent `stevenleehans`
# labels restricted to the SAME 4,349 studies, so the studies, the architecture,
# the geometry, the epochs and the seed are all identical and the label source
# is the only difference. Both parquets are structurally identical -- the
# incumbent's channel and weight columns are constants (`asserted` / 1.0 across
# all 52,884 cells), so they were copied and only the values differ.
#
# The two sets share NO cell value (0.0% identical) and their per-finding
# Spearman runs 0.52 to 0.945, lowest on **Synovitis (0.520)** and Fracture
# (0.609). Synovitis is this project's floor since E059 and the weakest finding
# in the CoAtNet blend (0.809), so the label sets disagree most exactly where
# there is the most to win.
DREAD_DATASET = f"{ACCOUNT}/knee-phase1-dread"
PUB4349_DATASET = f"{ACCOUNT}/knee-phase1-pub4349"
"""Publicly shared LLM report labels, repackaged into this pipeline's schema.

NOT this project's labels. Source: `stevenleehans/rsna-knee-llm-report-labels`,
file `llm_labels_v4_blend.csv`, shared publicly on Kaggle. Credit to that
author. The competition permits freely and publicly available external data.

Measured on the 58 expert studies with this project's convention (E041):
macro **0.8927** against this project's own fused labels at **0.7827**, and
**100%** slot coverage against 65.5%. Rank-unioning the two makes it *worse*
(0.8717), so it is used alone.
"""
T4 = "NvidiaTeslaT4"


@dataclass(frozen=True)
class Geometry:
    """How pixels reach the model.

    `mm_per_pixel` is the one that must never drift silently: pixel spacing in
    this dataset ranges 0.156–0.50 mm, so resampling to a fixed millimetre
    scale is what stops the model reading scanner identity off the image size
    (`FINDINGS.md` §9 measured that leak at 0.087 macro AUC).
    """

    mm_per_pixel: float
    size: int
    slices: int
    infer_batch: int = 4     # studies per forward pass; 288px does not fit 4
    note: str = ""

    def constants(self) -> dict[str, object]:
        return {"TARGET_MM_PER_PIXEL": self.mm_per_pixel,
                "TARGET_SIZE": self.size,
                "SLICES_PER_PLANE": self.slices}


@dataclass(frozen=True)
class Cache:
    """A built cache and the kernels that build it.

    The geometry lives here rather than on the lineage, so every consumer
    reaches it through the cache it actually reads.
    """

    geometry: Geometry
    shards: int
    slug: str            # "knee-cache-build-{shard}"
    directory: str       # "03_cache_build_shard{shard}"

    def kernels(self) -> list[Kernel]:
        return [
            Kernel(
                slug=self.slug.format(shard=shard),
                directory=self.directory.format(shard=shard),
                template="cache_build",
                datasets=[ARTIFACTS_DATASET],
                constants={"RUN_SPLIT": "train", "RUN_SHARD": shard,
                           "RUN_OF": self.shards, "RUN_LIMIT": 0,
                           **self.geometry.constants()},
                note=self.geometry.note,
            )
            for shard in range(self.shards)
        ]


@dataclass(frozen=True)
class TrainConfig:
    backbone: str
    epochs: int
    batch: int
    lr: float
    accum: int = 1
    slice_subsample: int | None = None
    input_norm: bool = False
    per_finding_pool: bool = False
    focal_k: int = 0
    full_fit_epoch: int = 20
    """Which epoch a full-fit run exports, since it has no honest validation.

    Measured rather than chosen: across the five v1public folds the mean val
    AUC peaks at epoch 20 and the mean gold AUC at 21, both flat over 18-21
    (E055). A full-fit model trains on every study including all 58 gold, so
    there is no held-out set left to early-stop on and picking the epoch from
    the monitor set would be picking it from training data.
    """
    full_fit_epoch_early: int | None = None
    """A SECOND full-fit export, from the same trajectory, or None for one.

    `full_fit_epoch` above is 20 because that is where the FOLD models peak
    (E055). Those models train on 3,526 studies; a full-fit model trains on
    4,407, so the same epoch number buys it 25% more optimisation steps. If the
    peak is governed by steps rather than by passes over the data, the fold
    optimum of 20 x 3,526 = 70,520 study-visits falls at epoch 16 here — and
    E055 measured the fold curves DECAYING past 21, so the standing setting may
    be exporting every full-fit member a quarter past its best.

    The opposite is equally arguable: more data per pass can support more
    passes. Nothing in this repo settles it, and the full-fit models cannot be
    validated offline by construction, so only the board can.

    Setting this writes `early_fold{tag}.pt` alongside the normal export.
    Because both come from ONE run, the export epoch is the only difference
    between the two arms — no seed draw, no data-order change. E060 measured a
    pure reseed at -0.0284 on gold, which is larger than any effect expected
    here, so two separate runs would not have been able to answer this.
    """
    seed: int | None = None
    """Explicit RNG seed, or None to leave the process unseeded.

    None is not an oversight and not a synonym for 0: it reproduces exactly
    what every run before this field existed did, which is what keeps those
    checkpoints comparable to the source that made them. A lineage that sets
    an integer here is one whose weights are reproducible, and — the reason
    the field exists — one that is *provably* a different draw from another
    lineage with a different integer, rather than different by the accident
    of two processes seeding themselves from OS entropy.
    """
    note: str = ""

    def constants(self) -> dict[str, object]:
        return {"RUN_EPOCHS": self.epochs, "RUN_BATCH": self.batch,
                "ACCUM_STEPS": self.accum, "RUN_LR": self.lr,
                "RUN_BACKBONE": self.backbone,
                "SLICE_SUBSAMPLE": self.slice_subsample,
                "INPUT_NORM": self.input_norm,
                "PER_FINDING_POOL": self.per_finding_pool,
                "FOCAL_K": self.focal_k,
                "RUN_SEED": self.seed,
                "FULL_FIT_EPOCH": self.full_fit_epoch,
                "FULL_FIT_EPOCH_EARLY": self.full_fit_epoch_early,
                "RUN_TIME_BUDGET": Raw("7.5 * 3600"),
                "GOLD_WEIGHT": 8.0, "ABSTAIN_MASKS_LOSS": True,
                "WARMUP_EPOCHS": 2, "EMA_DECAY": 0.999, "LABEL_SMOOTH": 0.02}


class Raw(str):
    """A constant to emit as source rather than as a repr'd literal."""


@dataclass(frozen=True)
class Trainer:
    fold: int
    slug: str
    directory: str
    resume_from: str | None = None
    """Another trainer whose output this one mounts, to continue its run.

    Kaggle caps a session at 9 hours, and the training kernel already resumes
    from a mounted `checkpoint_fold{n}.pt`. Naming that source here is what
    turns "the curve had not flattened yet" into a second session rather than a
    restart.
    """


@dataclass
class Kernel:
    """One pushed kernel: a directory, a slug, and what it mounts."""

    slug: str
    directory: str
    template: str
    gpu: bool = False
    internet: bool = False
    depends: list[str] = field(default_factory=list)     # slugs
    external_kernels: list[str] = field(default_factory=list)
    """Public kernels owned by SOMEONE ELSE, as full "owner/slug" strings.

    Separate from `depends` because the manifest check resolves every entry in
    `depends` against this file and must keep doing so — an unknown slug there
    is a typo or a deleted kernel, and silently allowing foreign names would
    turn that check off. These are deliberately outside the manifest: nothing
    here can regenerate them, and their contents can change without notice
    because another account owns them.

    Mounting a public notebook's output is Kaggle's own mechanism for reusing
    shared work, and the competition rules allow publicly shared code and data
    as inputs for every participant with attribution. It is preferred over
    republishing someone else's code under this account: no copy is taken, the
    upstream author keeps their authorship, and the credit is unambiguous.

    A foreign kernel can be deleted, made private, or re-run with different
    outputs at any time, so anything mounting one MUST verify what it got
    rather than assume — see `MEMBERS_EXPECTED` in `rank_blend`.
    """
    datasets: list[str] = field(default_factory=list)
    constants: dict[str, object] = field(default_factory=dict)
    note: str = ""

    def metadata(self) -> dict:
        return {
            "id": f"{PUSH_ACCOUNT}/{self.slug}",
            "title": self.slug,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": self.gpu,
            "enable_tpu": False,
            "enable_internet": self.internet,
            "machine_shape": T4 if self.gpu else "",
            "dataset_sources": list(self.datasets),
            "competition_sources": [COMPETITION],
            "kernel_sources": ([f"{DEPENDS_ACCOUNT}/{d}" for d in self.depends]
                               + list(self.external_kernels)),
            "model_sources": [],
        }


@dataclass(frozen=True)
class Lineage:
    """A cache and everything trained on it.

    Naming is spelled out rather than derived because the live slugs are
    historical — Kaggle identifies a kernel by its slug, so renaming one
    creates a second kernel instead of updating the first (`kaggle/README.md`).
    """

    name: str
    cache: Cache
    train: TrainConfig
    trainers: tuple[Trainer, ...]
    infer_slug: str | None = None
    infer_directory: str | None = None
    labels: str = ARTIFACTS_DATASET
    """Which dataset supplies soft_labels.parquet.

    Exactly one must be mounted: the training kernel finds the file by search,
    so mounting two would make the targets depend on directory order.
    """

    @property
    def geometry(self) -> Geometry:
        return self.cache.geometry          # the only place a geometry comes from

    def kernels(self) -> list[Kernel]:
        out = []
        cache_slugs = [k.slug for k in self.cache.kernels()]
        for trainer in self.trainers:
            out.append(Kernel(
                slug=trainer.slug,
                directory=trainer.directory,
                template="train",
                gpu=True,
                internet=True,      # pretrained weights; only submissions go offline
                depends=cache_slugs + ([trainer.resume_from]
                                       if trainer.resume_from else []),
                datasets=[self.labels],
                constants={"RUN_FOLD": trainer.fold,
                           **self.geometry.constants(),
                           **self.train.constants()},
                note=self.train.note,
            ))
        if self.infer_slug:
            out.append(Kernel(
                slug=self.infer_slug,
                directory=self.infer_directory,
                template="infer",
                gpu=True,
                internet=False,     # this is the submission kernel
                depends=[t.slug for t in self.trainers],
                constants={**self.geometry.constants(),
                           "BATCH_STUDIES": self.geometry.infer_batch,
                           "SLICE_SUBSAMPLE_EXPECTED": self.train.slice_subsample,
                           "INPUT_NORM_EXPECTED": self.train.input_norm,
                           # A lineage mounts every trainer it declares, so the
                           # count is derived rather than restated. The EXTRAS
                           # ensembles cross lineages and declare theirs by hand.
                           "MEMBERS_EXPECTED": len(self.trainers)},
                note=self.geometry.note,
            ))
        return out


# --------------------------------------------------------------------------- #
# The pipeline as it actually stands. Every number below has either been run or
# is queued to run; nothing here is aspirational.
# --------------------------------------------------------------------------- #

# ---------------------------------------------------------------------- #
# The CC0 CoAtNet arm. NOT this project's geometry — it is fixed by the
# published weights and must match them exactly or the model sees inputs it was
# never trained on.
#
# THE FOUR SUB-MODELS DISAGREE ON ALMOST EVERYTHING, and that is the trap. An
# earlier version of this manifest gave all four one shared geometry, taken from
# the v4 notebook — which is the LEGACY arm none of these four uses. Reading one
# notebook and assuming the family shares its constants is exactly the silent
# train/inference skew `HANDOFF.md` §4d warned about, and it was caught by
# reading the reference implementations rather than by running anything.
#
# Sourced from `evgendvorkin/rsna-baseline` and `hyakumanben2025/
# rsna-knee-0937-meniscus-resid-repro`, which agree line for line, and from the
# public training notebook `dreaddevelopment/knee-mri-training-the-twelve-
# finding-model` (E088, E090).
#
# Slots are (plane, fluid-sensitive, count); -1 means no preference. Preferring
# a fluid-sensitive series for some slots and not others is deliberate upstream:
# fluid-sensitive sequences show swelling, effusion and acute injury, the others
# show anatomy and cartilage, and the twelve findings split across both.
RAPTOR_SLOTS64 = (("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
                  ("Coronal", 0, 8), ("Axial", -1, 12))          # sums to 64
RAPTOR_SLOTS44 = (("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
                  ("Coronal", 0, 6), ("Axial", -1, 8))           # sums to 44

RAPTOR_LAB = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
              "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
              "Contusion", "Fracture")

# `img` is the cache resolution the volume is built at; the model input is `res`
# from the checkpoint (384 for all four), so `img` 336 means the windows are
# upsampled and `img` 384 means they are native. `reverse` reverses the
# three-slice channel triplet — upstream's prose calls it a horizontal flip and
# its code does `flip(1)` on (K, 3, H, W), which is not the same thing.
#
# Weights are PUBLISHED, not fitted here. Their author's own gold-58 figures:
# maxspan-v5 0.9198, native384dense-v10 0.9170, maxspan-v5-reverse 0.9167,
# native384-v8 0.9116, and the four-arm blend 0.9254 against our pooled 0.8980.
# Every one of those is a self-report on the author's split.
# `expect_gold` is the `gold_auc` each file carries, read on 2026-09-08. It is a
# MOUNT FINGERPRINT, not a claim: the Kernel docstring already says a foreign
# asset "can be deleted, made private, or re-run with different outputs at any
# time, so anything mounting one MUST verify what it got rather than assume".
# If upstream re-uploads different weights under the same filename, this is what
# notices — the same trick that verified E086's arms by matching each mounted
# checkpoint back to its trainer by val AUC.
RAPTOR_V5 = {"name": "maxspan-v5", "file": "raptor_ft_coatnet_v5_full_swa.pt",
             "img": 336, "slots": RAPTOR_SLOTS64, "span": (0.02, 0.98),
             "k_eval": 62, "reverse": False, "w": 0.55, "expect_gold": 0.9214}
# E117's CANDIDATE ARMS. `dreaddevelopment` publishes FIFTEEN CC0 datasets
# carrying FOURTEEN checkpoints; E086's inventory catalogued them and no
# experiment has touched the eleven this project does not mount. Their stored
# `gold_auc`, read 2026-09-17 from the files themselves:
#
#   coatnet_v4_full      0.9167     coatnet_v9_full   0.9031
#   coatnet_v4_full_swa  0.9150     coatnet384        0.9025
#   coatnet_v7_full_swa  0.9073     coatnet_v6_full   0.8997
#   cnv2b336             0.8901     effv2l480         0.8716
#
# **v4_full at 0.9167 outscores native384-v8 at 0.9067, which the blend already
# weights at 0.20.** Six sit inside E048's +/-0.02 comparability band against the
# incumbents' 0.9067-0.9214; `effv2l480` at 0.8716 is 0.035 behind the weakest
# incumbent, outside the band, and is NOT screened -- E048's four failed unions
# all added a member 0.03-0.06 back.
#
# THE GEOMETRY IS NOT IN THE FILES. Each checkpoint stores `arch`, `res`,
# `gold_auc` and `epoch`, but `src` is only 'timm-pretrained' -- the volume
# geometry came from upstream's write-up for the three arms this project mounts,
# and the dataset names (widedense, widefov, fullspan, finespacing) are all that
# is known for the rest. So every candidate is screened AT THE DENSE GEOMETRY,
# v10's, rather than at a guessed one. That is deliberate:
#
# - it costs ONE preprocessing signature. The template groups by signature, so
#   screening six candidates at a shared geometry adds forward passes to caches
#   the incumbents already build, and no new cache pass at all.
# - it makes the comparison readable. Each candidate's measured gold against its
#   own stored `gold_auc` is then a statement about THIS geometry, and a
#   candidate that holds up at a geometry it was not trained for is a stronger
#   member than one that needs its own.
# - a shortfall is diagnosis, not rejection: it separates "wrong geometry" from
#   "weak checkpoint", and the write-up can be read afterwards for any arm worth
#   the second pass.
#
# `expect_gold` stays the MOUNT FINGERPRINT it is for the incumbents -- the value
# the file carries, so a silent re-upload is caught -- and is NOT a prediction of
# what these will score here.
RAPTOR_DENSE = {"slots": RAPTOR_SLOTS64, "span": (0.02, 0.98),
                "k_eval": 62, "img": 384, "reverse": False}
RAPTOR_CANDIDATES = (
    {"name": "cand-widedense-v4", "file": "raptor_ft_coatnet_v4_full.pt",
     "w": 0.25, "expect_gold": 0.9167, **RAPTOR_DENSE},
    {"name": "cand-fullspan-v7", "file": "raptor_ft_coatnet_v7_full_swa.pt",
     "w": 0.25, "expect_gold": 0.9073, **RAPTOR_DENSE},
    {"name": "cand-finespacing-v9", "file": "raptor_ft_coatnet_v9_full.pt",
     "w": 0.25, "expect_gold": 0.9031, **RAPTOR_DENSE},
    {"name": "cand-coatnet384", "file": "raptor_ft_coatnet384.pt",
     "w": 0.25, "expect_gold": 0.9025, **RAPTOR_DENSE},
    {"name": "cand-widefov-v6", "file": "raptor_ft_coatnet_v6_full_swa.pt",
     "w": 0.25, "expect_gold": 0.8997, **RAPTOR_DENSE},
    # The only candidate from a DIFFERENT BACKBONE FAMILY, and the reason it is
    # screened at 0.8901 when a CoAtNet at that number would be marginal: the
    # four CoAtNet arms correlate 0.905-0.986 with each other and 0.542 with the
    # v1 arm, so what a union pays for here is disagreement, not rank. Its `res`
    # is 336, so `img` 336 keeps the windows native and reuses maxspan's cache.
    {"name": "cand-cnv2b336", "file": "raptor_ft_cnv2b336.pt",
     "w": 0.25, "expect_gold": 0.8901,
     "slots": RAPTOR_SLOTS64, "span": (0.02, 0.98), "k_eval": 62,
     "img": 336, "reverse": False},
)

RAPTOR_ARMS = (
    RAPTOR_V5,
    {"name": "native384dense-v10", "file": "raptor_ft_coatnet_v10_full.pt",
     "img": 384, "slots": RAPTOR_SLOTS64, "span": (0.02, 0.98),
     "k_eval": 62, "reverse": False, "w": 0.10, "expect_gold": 0.9174},
    {"name": "maxspan-v5-reverse", "file": "raptor_ft_coatnet_v5_full_swa.pt",
     "img": 336, "slots": RAPTOR_SLOTS64, "span": (0.02, 0.98),
     "k_eval": 62, "reverse": True, "w": 0.15, "expect_gold": 0.9214},
    {"name": "native384-v8", "file": "raptor_ft_coatnet_v8_full_swa.pt",
     "img": 384, "slots": RAPTOR_SLOTS44, "span": (0.06, 0.94),
     "k_eval": 42, "reverse": False, "w": 0.20, "expect_gold": 0.9067},
)

# TEST-TIME AUGMENTATION FROM THE CHECKPOINTS ALREADY MOUNTED (E103).
#
# Two axes, and the point of both is that they introduce NO new asset, NO new
# training and NO hyperparameter chosen on the 58 gold studies.
#
# AXIS 1 -- REVERSE THE OTHER TWO CHECKPOINTS. Upstream publishes four arms and
# exactly one of them is a transform rather than a file: `maxspan-v5-reverse`,
# the same weights with the three-slice triplet reversed. They never applied it
# to `native384-v8` or `native384dense-v10`. Applying it there adds two members
# for two extra forward passes on a volume that is already built -- the template
# groups by preprocessing signature, so a reverse member costs one forward and
# not a second pass. Zero new hyperparameters: the mechanism is upstream's own
# and it is already in the blend that scored 0.932.
#
# AXIS 2 -- JITTER THE SPAN. `span` selects which slices of each source series
# fill the stack: `lo, hi = int(n * span_lo), int(n * span_hi) - 1`, then k of
# them evenly. Changing it resamples the volume from different source slices,
# which is ordinary multi-crop TTA along the slice axis.
#
#   THE JITTER MAGNITUDE IS TAKEN FROM UPSTREAM, NOT FROM GOLD. v5 and v10 span
#   (0.02, 0.98); v8 spans (0.06, 0.94). The authors' own arms differ by exactly
#   0.04 at each end, so 0.04 and 0.08 are their step and twice their step. No
#   number here was chosen by looking at a score.
#
# WHAT IS DELIBERATELY ABSENT: a horizontal flip. Upstream's prose calls their
# reverse member one and the template's comment already records that the code
# does something else. A true left-right mirror is not merely a distribution
# shift here, it is WRONG FOR THIS LABEL SET: four of the twelve findings are
# Medial/Lateral Meniscus and Medial/Lateral OA, and medial versus lateral is
# which SIDE of the knee a structure sits on. Mirroring a left knee turns it
# into a right knee and swaps medial with lateral, so those four findings would
# be read off the wrong compartment. That is presumably why upstream's transform
# is a slice reversal and not a mirror.
#
# ALSO ABSENT: varying `k_eval`. `_eval_centers` takes `k` points evenly over
# the interior of the filled mask, and v5 already asks for 62 of at most 62. It
# is not subsampling, so a smaller k removes information and a larger one
# duplicates slices. There is no resampling diversity on that axis to have.


def _span(lo, hi, step):
    return (round(lo + step, 2), round(hi - step, 2))


def _variant(arm, name, span=None, reverse=None):
    return {**arm, "name": name,
            "span": arm["span"] if span is None else span,
            "reverse": arm["reverse"] if reverse is None else reverse,
            # Flat weights: this manifest exists to DUMP per-arm probabilities,
            # and the blend weights are searched offline afterwards. The kernel's
            # own blend line is a flat average and is not the thing under test.
            "w": 1.0}


_V10 = RAPTOR_ARMS[1]
_V8 = RAPTOR_ARMS[3]
RAPTOR_TTA_ARMS = (
    # the four published arms, unchanged, so the sweep has its own control
    _variant(RAPTOR_V5, "maxspan-v5"),
    _variant(_V10, "native384dense-v10"),
    _variant(RAPTOR_V5, "maxspan-v5-reverse", reverse=True),
    _variant(_V8, "native384-v8"),
    # axis 1: upstream's own transform, applied to the two files they left out
    _variant(_V10, "native384dense-v10-reverse", reverse=True),
    _variant(_V8, "native384-v8-reverse", reverse=True),
    # axis 2: one step and two steps inward, at upstream's own step size
    _variant(RAPTOR_V5, "maxspan-v5-span04", span=_span(0.02, 0.98, 0.04)),
    _variant(RAPTOR_V5, "maxspan-v5-span08", span=_span(0.02, 0.98, 0.08)),
    _variant(_V10, "native384dense-v10-span04", span=_span(0.02, 0.98, 0.04)),
    _variant(_V10, "native384dense-v10-span08", span=_span(0.02, 0.98, 0.08)),
    _variant(_V8, "native384-v8-span04", span=_span(0.06, 0.94, 0.04)),
    _variant(_V8, "native384-v8-span08", span=_span(0.06, 0.94, 0.08)),
)

# E108. ONE STEP NARROWER, ON EVERY CHECKPOINT, TESTED AT 75x GOLD'S SAMPLE.
#
# E103 found all SIX span-narrowed variants beat their parent on the 58 gold --
# six for six, one direction -- and `maxspan-v5-span04` at 0.9253 is the highest
# single arm this project has measured, above the published four-arm blend. It
# closed anyway, because gold-58 both produced that hypothesis and would have
# been the only thing testing it.
#
# E108's finding reopens it: the report-label proxy ranks the four published
# CoAtNet arms at Spearman +0.800 against gold, where E106 measured +0.052 for
# the cross-architecture case. E106's bias is that the proxy rewards a model for
# agreeing with the labels it was trained on -- and that is common-mode between
# two inference geometries of the SAME checkpoint. It cancels. So the 4,349
# non-gold studies can judge this, and gold-58 does not have to judge its own
# hypothesis.
#
# ONE STEP FOR ALL THREE, not the per-checkpoint best. Gold's own per-arm optima
# disagree (span04 for v5 and v10, span08 for v8); picking each checkpoint's best
# would be three parameters fitted on 58 studies. 0.04 is upstream's own spacing
# and is applied uniformly.
RAPTOR_SPAN04_ARMS = tuple(
    _variant(a, f"{a['name']}-span04", span=_span(a["span"][0], a["span"][1], 0.04))
    for a in (RAPTOR_V5, RAPTOR_ARMS[1], RAPTOR_ARMS[3])
)

V1 = Geometry(
    mm_per_pixel=0.6, size=192, slices=20,
    note="0.6 mm/px over 192 px covers ~115 mm, which contains the knee joint\n"
         "with margin. Chosen against the inference budget: 3 planes x 20 slices\n"
         "at 192px is ~2.2 MB per study, so the training cache stays under 10 GB.\n"
         "This is the geometry that scored 0.725 on the leaderboard.",
)

V2 = Geometry(
    mm_per_pixel=0.40, size=288, slices=24, infer_batch=2,
    note="v2 geometry, chosen after measuring what v1 discarded: native pixel\n"
         "spacing has a median of 0.312 mm and 96% of series are finer than 0.60,\n"
         "so v1 downsampled almost every study ~2x, and kept 20 of a median 30\n"
         "slices. 0.40 mm/px over 288 px keeps the same ~115 mm field of view at\n"
         "1.5x the in-plane detail. Cost: ~6 MB per study, ~26 GB, so 8 shards.\n"
         "MEASURED RESULT: 0.668 on the leaderboard, below v1's 0.725. The\n"
         "resolution thesis is unconfirmed.",
)

CACHE_V1 = Cache(geometry=V1, shards=4,
                 slug="knee-cache-build-{shard}",
                 directory="03_cache_build_shard{shard}")
CACHE_V2 = Cache(geometry=V2, shards=8,
                 slug="knee-cache-v2-{shard}",
                 directory="06_cache_v2_shard{shard}")

LINEAGES = [
    Lineage(
        name="v1",
        cache=CACHE_V1,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4,
            note="The configuration that scored 0.725. Batch 16 is affordable\n"
                 "because of AMP across both T4s; the LR is scaled to it.\n"
                 "input_norm is False because that is what these weights were\n"
                 "trained with — see the module docstring.",
        ),
        trainers=(
            Trainer(0, "knee-train", "04_train"),
            Trainer(1, "knee-train-fold1", "10_train_fold1"),
            Trainer(2, "knee-train-fold2", "10_train_fold2"),
            Trainer(3, "knee-train-fold3", "10_train_fold3"),
            Trainer(4, "knee-train-fold4", "10_train_fold4"),
        ),
        infer_slug="knee-infer-folds",
        infer_directory="11_infer_folds",
    ),
    Lineage(
        name="v2",
        cache=CACHE_V2,
        train=TrainConfig(
            backbone="resnet34", epochs=30, batch=4, lr=6e-4, accum=4,
            slice_subsample=18,
            note="288px is 2.25x the pixels of v1 and each study is 3 planes x 24\n"
                 "slices, so batch 16 does not fit a T4 even with AMP. Batch 4 with\n"
                 "4-step accumulation reproduces the effective batch that worked,\n"
                 "and subsampling 18 of 24 slices buys back throughput while acting\n"
                 "as augmentation.",
        ),
        trainers=(Trainer(0, "knee-train-v2", "07_train_v2"),),
        infer_slug="knee-infer-v2",
        infer_directory="08_infer_v2",
    ),
    Lineage(
        # The label experiment. Identical to knee-train in every constant; the
        # only difference is which dataset supplies soft_labels.parquet.
        #
        # Measured on the 58 gold studies (E023): neither labeler beats the
        # other — the paired interval on their difference contains zero — but
        # their union beats both by +0.070, because they abstain on different
        # findings. Across the corpus the fusion drops unsupervised slots from
        # 49.4% to 34.6%. The 0.725 imaging model was trained by a 0.769
        # teacher, so a better teacher is the change most likely to move the
        # board, and nothing else here has that much headroom.
        name="v1fused",
        cache=CACHE_V1,
        labels=FUSED_DATASET,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4,
            note="Identical to the 0.725 configuration. The labels are the\n"
                 "single variable.",
        ),
        trainers=(
            Trainer(0, "knee-train-v1fused", "21_train_v1fused"),
            # Folds 1-4, added after E031. Fold 0 alone gave +0.0721 paired on
            # gold with CI [-0.009, +0.183] — the right direction, and NOT
            # separated, because n=12 carries a ~0.19 interval. These four take
            # the paired comparison to n=58, where FINDINGS.md 13 measured the
            # interval at ~0.044 by simulation. That resolution separates a gap
            # this size; nothing cheaper does.
            Trainer(1, "knee-train-v1fused-fold1", "27_train_v1fused_fold1"),
            Trainer(2, "knee-train-v1fused-fold2", "28_train_v1fused_fold2"),
            Trainer(3, "knee-train-v1fused-fold3", "29_train_v1fused_fold3"),
            Trainer(4, "knee-train-v1fused-fold4", "30_train_v1fused_fold4"),
        ),
        infer_slug="knee-infer-v1fused",
        infer_directory="22_infer_v1fused",
    ),
    Lineage(
        # The public-label lineage. Identical to v1fused in every constant —
        # same cache, same geometry, same 0.725-era hyperparameters — so the
        # labels remain the single variable, exactly as E032 established them
        # to be the dominant lever (+0.089 of the +0.121 board move, E036).
        #
        # The teacher this mounts is 0.114 better than the one that produced
        # 0.846, and covers every slot rather than 65.5%, so ABSTAIN_MASKS_LOSS
        # now masks nothing and the model sees ~53% more supervised targets.
        name="v1public",
        cache=CACHE_V1,
        labels=PUBLIC_DATASET,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4,
            note="Identical to the 0.846 configuration. The labels are the\n"
                 "single variable, and they are not this project's own.",
        ),
        trainers=(
            Trainer(0, "knee-train-v1pub", "37_train_v1pub_fold0"),
            Trainer(1, "knee-train-v1pub-fold1", "38_train_v1pub_fold1"),
            Trainer(2, "knee-train-v1pub-fold2", "39_train_v1pub_fold2"),
            Trainer(3, "knee-train-v1pub-fold3", "40_train_v1pub_fold3"),
            Trainer(4, "knee-train-v1pub-fold4", "41_train_v1pub_fold4"),
        ),
        infer_slug="knee-infer-v1pub",
        infer_directory="42_infer_v1pub",
    ),
    Lineage(
        # The distilled lineage. Byte-identical to v1public in every constant —
        # same cache, same geometry, same hyperparameters, same five folds — so
        # the TEACHER is the single variable, exactly as it was when the public
        # labels replaced the fused ones and the board paid +0.077.
        #
        # The teacher is the 50/50 rank union of the public report labels and
        # the out-of-fold predictions of the v1public models those labels
        # trained. On the 58 expert studies: labels 0.8927, model 0.8980, union
        # 0.9188 — a paired +0.0261 over the labels, CI [+0.009, +0.046],
        # separated (E069).
        #
        # It is worth GPU because of WHY it separated. E048 established that a
        # union pays when its members are comparable and imports errors when
        # they are not: two readers 0.002 apart were worth +0.070, and four
        # later unions each added a member 0.03-0.06 behind and were worth
        # nothing. Teacher and student here are 0.005 apart. This is the first
        # candidate member in five attempts that met the condition, and it is
        # the first to separate.
        #
        # What gold-58 CANNOT see, stated here rather than discovered later: it
        # measures agreement with 58 expert answers, not whether the union is a
        # better training target on the other 4,349 studies — which is what a
        # distilled teacher is for. The board is the only instrument that prices
        # that, which is the whole reason this lineage exists.
        name="v1pubdistil",
        cache=CACHE_V1,
        labels=DISTILLED_DATASET,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4,
            note="Identical to the 0.923 configuration. The teacher is the\n"
                 "single variable: report labels united with the model those\n"
                 "labels produced.",
        ),
        trainers=(
            Trainer(0, "knee-train-v1distil", "65_train_v1distil_fold0"),
            Trainer(1, "knee-train-v1distil-fold1", "66_train_v1distil_fold1"),
            Trainer(2, "knee-train-v1distil-fold2", "67_train_v1distil_fold2"),
            Trainer(3, "knee-train-v1distil-fold3", "68_train_v1distil_fold3"),
            Trainer(4, "knee-train-v1distil-fold4", "69_train_v1distil_fold4"),
        ),
        infer_slug="knee-infer-v1distil",
        infer_directory="70_infer_v1distil",
    ),
    Lineage(
        # A ONE-FOLD PROBE of the LAST untested region, not a campaign.
        #
        # E046 ended its list with the observation that every geometry and
        # architecture lever this project tested — 288px, DINOv2, focal top-k,
        # per-finding pooling — was measured against a 0.78 teacher and came
        # back zero or negative, and that NONE had been retested against a good
        # one. The teacher is now 0.9188 (E069), which is 0.14 better than the
        # one that judged 288px in E019.
        #
        # The premise is measured rather than hoped, and it is in this file:
        # V2's own note records that native pixel spacing has a median of
        # 0.312 mm and 96% of series are finer than 0.60, so V1 downsamples
        # almost every study about 2x. This lineage is the same ~115 mm field of
        # view at 1.5x the in-plane detail — it does not crop differently, it
        # stops discarding what the scanner recorded.
        #
        # Why the answer might differ now: against a noisy teacher, extra
        # resolution is extra capacity to fit label noise, which is what E019
        # measured (0.688 against 192px's 0.725). Against a clean teacher the
        # same capacity can carry signal. That is a mechanism, not a hope, and
        # it is the only reason to revisit a lever the board already rejected.
        #
        # ONE FOLD, because a single fold's gold subset is n~12 and E031 put its
        # interval at ~0.19. This probe cannot confirm a gain. It can only rule
        # out a disaster — DINOv2 failed by 0.07 and that was visible at one
        # fold — which is exactly what E046's dinov2pub probe was for, and that
        # probe saved ~20 GPU-h.
        name="v2distil",
        cache=CACHE_V2,
        labels=DISTILLED_DATASET,
        train=TrainConfig(
            # Two variables had to be pinned, and finding the second cost two
            # OOMs. Both are held EQUAL to v1distil so that RESOLUTION is the
            # only thing that differs:
            #
            #   effective batch  2 x 8 = 16, matching v1distil's 16 x 1.
            #   slices per study slice_subsample=20 x 3 planes = 60, matching
            #                    v1distil's 20 x 3 = 60.
            #
            # V2 geometry carries 24 slices per plane against V1's 20, so
            # leaving slice_subsample None fed 72 slices and made SLICE COUNT a
            # silent second variable — as well as OOMing, because memory scales
            # with per-step batch x slices. The historical 288px runs that DID
            # fit (07_train_v2, 14_train_v2_long) used batch 4 with
            # slice_subsample 18, i.e. 216 slice-units per step; this is 120,
            # comfortably inside that.
            backbone="resnet34", epochs=24, batch=2, lr=6e-4, accum=8,
            slice_subsample=20,
            note="The distilled teacher at v2 geometry, effective batch 16 via\n"
                 "4-step accumulation so it matches v1distil exactly. Against\n"
                 "v1pubdistil the geometry is then the single variable; against\n"
                 "v1public it is two, so the comparison that counts is against\n"
                 "v1pubdistil fold 0.",
        ),
        trainers=(
            Trainer(0, "knee-train-v2distil", "72_train_v2distil_fold0"),
        ),
    ),
    Lineage(
        # A ONE-FOLD PROBE, not a campaign. Every architecture lever this
        # project tested — 288px, DINOv2, focal top-k, per-finding pooling —
        # was measured against a 0.78 teacher and came back zero or negative
        # (E029, E030, E036, E039). None has been retested against the 0.89
        # teacher that produced the 0.923 board result, and that is the one
        # large untested region left.
        #
        # Fold 0 only, ~5 GPU-h. If its gold lands near the resnet34's 0.8477
        # on the same fold, a second family is competitive and worth four more
        # folds. If it is 0.07 behind as in E036, it is dead again and the
        # probe saved ~20 GPU-h. Constants are otherwise identical to the
        # dinov2fused lineage so the comparison stays one-variable.
        name="dinov2public",
        cache=CACHE_V1,
        labels=PUBLIC_DATASET,
        train=TrainConfig(
            backbone="vit_small_patch14_dinov2.lvd142m",
            epochs=40, batch=6, lr=1e-4, accum=3, input_norm=True,
            note="Identical to dinov2fused except the labels. A one-fold probe\n"
                 "of whether a second architecture is competitive once the\n"
                 "teacher is good.",
        ),
        trainers=(
            Trainer(0, "knee-train-dinov2pub", "43_train_dinov2pub_fold0"),
            # Fold 1 fills the second GPU slot, which is idle while fold 0
            # runs (concurrency is 2). Two folds is a materially better read
            # than one — E031 showed a single fold's gold subset carries a
            # ~0.19 interval — and it costs no wall clock. Folds 2-4 stay
            # undeclared until the probe reports.
            Trainer(1, "knee-train-dinov2pub-fold1", "44_train_dinov2pub_fold1"),
        ),
    ),
    Lineage(
        # Per-finding attention pooling, which this project has called dead
        # three times and measured at roughly the same positive value each
        # time. On frozen embeddings, out-of-fold on the 58 expert studies:
        #
        #   E030, old labels, 1 seed    +0.039   [-0.009, +0.090]
        #   E053, old labels, 4 seeds   +0.0371  [-0.001, +0.077]
        #   E052, public labels, 4 seeds +0.0338 [+0.009, +0.061]  separated
        #
        # E052 first read the last row as the teacher unlocking the lever.
        # E053 withdrew that: the effect is the same size against both
        # teachers and only the interval moved. What changed was averaging
        # restarts before scoring — one seed per arm put the BASELINE's spread
        # at 0.036 while the treated arm moved 0.007, so a single-seed A/B on
        # 58 studies was reading an initialisation as an architecture.
        #
        # Three agreeing magnitudes across two independent teachers is why
        # this is worth GPU. The mechanism was also written down before any of
        # it was measured, in the model's own comment: one attention map over
        # twelve findings forces a single compromise about which slices
        # matter, and the focal findings pay it. That is still exactly where
        # this ensemble is weakest — Synovitis 0.771, Lateral OA 0.830, PF OA
        # 0.849, against Medial OA 0.980 and Baker's 0.978.
        #
        # Focal top-k moves the same way (+0.0163, [-0.001, +0.035]) and is
        # NOT taken here, because two levers at once is no longer one variable.
        #
        # It is also a better use of the last GPU hours than v1publicB's
        # reseed. The folds join the ensemble either way, a different pooling
        # is more diversity than a different initialisation, and this answers
        # a question while it does it.
        name="v1pubpool",
        cache=CACHE_V1,
        labels=PUBLIC_DATASET,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4,
            per_finding_pool=True, seed=2,
            note="v1public with per-finding attention pooling, which is the\n"
                 "single variable against the five folds behind 0.923.",
        ),
        trainers=(
            Trainer(0, "knee-train-v1pubpool", "51_train_v1pubpool_fold0"),
            Trainer(1, "knee-train-v1pubpool-fold1", "52_train_v1pubpool_fold1"),
            Trainer(2, "knee-train-v1pubpool-fold2", "53_train_v1pubpool_fold2"),
            Trainer(3, "knee-train-v1pubpool-fold3", "54_train_v1pubpool_fold3"),
            Trainer(4, "knee-train-v1pubpool-fold4", "55_train_v1pubpool_fold4"),
        ),
    ),
    Lineage(
        # The only lever with a measured coefficient. E036 measured ensembling
        # at +0.032 on the board for 1 fold -> 5; log-scaling puts 5 -> 10 at
        # roughly +0.010. Same config, same labels, different seeds — nothing
        # here is a new hypothesis, which is the point: every hypothesis this
        # project still had has been measured and is dead (E046-E049).
        #
        # Folds re-run with a different seed give genuinely different models
        # because init, augmentation order and batch composition all change,
        # while the fold split stays fixed so gold OOF remains valid.
        name="v1publicB",
        cache=CACHE_V1,
        labels=PUBLIC_DATASET,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4, seed=1,
            note="Second seed of the 0.923 configuration. Nothing changes but\n"
                 "the random seed; this buys ensemble diversity, not a new idea.\n"
                 "\n"
                 "seed=1 is an explicit mechanism, not a label. Before the seed\n"
                 "field existed this lineage would have differed from v1public\n"
                 "only because two unseeded processes draw different entropy —\n"
                 "true in practice, but nothing in the source said so and\n"
                 "nothing would have caught it if it stopped being true.",
        ),
        trainers=(
            Trainer(0, "knee-train-v1pubB", "45_train_v1pubB_fold0"),
            Trainer(1, "knee-train-v1pubB-fold1", "46_train_v1pubB_fold1"),
            Trainer(2, "knee-train-v1pubB-fold2", "47_train_v1pubB_fold2"),
            Trainer(3, "knee-train-v1pubB-fold3", "48_train_v1pubB_fold3"),
            Trainer(4, "knee-train-v1pubB-fold4", "49_train_v1pubB_fold4"),
        ),
    ),
    Lineage(
        # DINOv2 reached 0.6878 in 16 epochs and NEVER FLATTENED — it climbed
        # 0.588 to 0.688 with the last three epochs still adding 0.002 each,
        # where the resnet34 run plateaued by epoch 18 of 24. So 0.6878 is not a
        # measurement of this backbone, it is where the clock stopped. This
        # continues it to 40 epochs.
        #
        # It also earns the run a gold dump: knee-train-dinov2 predates that
        # output, so there is currently no way to compare this backbone against
        # the resnet34 baseline on anything except CV, which mis-ranks.
        name="dinov2long",
        cache=CACHE_V1,
        train=TrainConfig(
            backbone="vit_small_patch14_dinov2.lvd142m",
            epochs=40, batch=6, lr=1e-4, accum=3, input_norm=True,
            note="Continues knee-train-dinov2. As with v2-long the cosine\n"
                 "schedule is absolute in epoch, so resuming at 16 of 40 puts\n"
                 "the LR back up — a warm restart, not a smooth continuation.\n"
                 "The run inherits the 0.6878 checkpoint as its best, so a\n"
                 "restart that never recovers exports the old weights.",
        ),
        trainers=(Trainer(0, "knee-train-dinov2-long", "19_train_dinov2_long",
                          resume_from="knee-train-dinov2"),),
        infer_slug="knee-infer-dinov2-long",
        infer_directory="20_infer_dinov2_long",
    ),
    Lineage(
        # PATH.md Phase C. The largest untried lever: DINOv2 reached 0.7041 at
        # epoch 34 and had NOT flattened — that number is where the clock
        # stopped, not where the backbone converges. Published evidence puts
        # backbone adaptation at +0.09, roughly five times what resolution was
        # worth, and it has never had a fair run here.
        #
        # Fresh runs, NOT resumed from knee-train-dinov2: that checkpoint was
        # trained on lexicon labels, and inheriting it would confound the
        # backbone question with the label question that E032 just settled.
        #
        # These exist to be rank-blended with the resnet34 family (Phase D).
        # E033 measured that blending two LABEL SETS on one backbone fails —
        # worse than its better half at every weight. Blending two
        # ARCHITECTURES is the version with published support, and this is the
        # second architecture.
        name="dinov2fused",
        cache=CACHE_V1,
        labels=FUSED_DATASET,
        train=TrainConfig(
            backbone="vit_small_patch14_dinov2.lvd142m",
            epochs=40, batch=6, lr=1e-4, accum=3, input_norm=True,
            note="Identical to the dinov2long configuration except that it\n"
                 "starts from pretrained weights rather than resuming, and is\n"
                 "supervised by the fused labels. 40 epochs is what the budget\n"
                 "affords, not where the curve was shown to flatten.",
        ),
        trainers=(
            Trainer(0, "knee-train-dinov2f", "31_train_dinov2f_fold0"),
            Trainer(1, "knee-train-dinov2f-fold1", "32_train_dinov2f_fold1"),
            Trainer(2, "knee-train-dinov2f-fold2", "33_train_dinov2f_fold2"),
            Trainer(3, "knee-train-dinov2f-fold3", "34_train_dinov2f_fold3"),
            Trainer(4, "knee-train-dinov2f-fold4", "35_train_dinov2f_fold4"),
        ),
        infer_slug="knee-infer-dinov2f",
        infer_directory="36_infer_dinov2f",
    ),
    Lineage(
        # The change the gold measurement actually asked for. Against the same
        # 58 expert-labelled studies this configuration BEATS its own teacher on
        # every diffuse finding and LOSES to it on every focal one — focal 0.632
        # against a 0.798 teacher, diffuse 0.783 against a 0.688 teacher. Losing
        # 0.228 on Medial Meniscus, which the teacher scores 0.744, is not a hard
        # problem being lost to; it is signal present in the targets and thrown
        # away by pooling that averages over sixty slice embeddings.
        #
        # FOCAL_K keeps the top three slices per finding alongside the weighted
        # mean and learns, per finding, how much to lean on each. Twelve extra
        # parameters. Closing only the recoverable gaps is worth +0.060 macro.
        name="v1focal",
        cache=CACHE_V1,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4, focal_k=3,
            note="Identical to the 0.725 configuration except FOCAL_K=3, so the\n"
                 "result attributes to that and nothing else. k=3 because a\n"
                 "meniscal tear or a ligament tear is visible on roughly three\n"
                 "slices of the twenty kept per plane.",
        ),
        trainers=(Trainer(0, "knee-train-v1focal", "24_train_v1focal"),),
        infer_slug="knee-infer-v1focal",
        infer_directory="25_infer_v1focal",
    ),
    Lineage(
        # A one-variable A/B against knee-train: same cache, same fold, same
        # everything except that each finding gets its own attention map. Fold 0
        # scored 0.7001 CV and 0.725 on the board, so the comparison is against
        # a number that has been measured on the thing that counts.
        name="v1pool",
        cache=CACHE_V1,
        train=TrainConfig(
            backbone="resnet34", epochs=24, batch=16, lr=6e-4,
            per_finding_pool=True,
            note="Identical to the v1 configuration except PER_FINDING_POOL.\n"
                 "input_norm stays False so this differs from knee-train in\n"
                 "exactly one respect — the discipline the 288px run failed.\n"
                 "The four findings this is aimed at are the focal ones:\n"
                 "Medial Meniscus, PF OA, Synovitis and MCL.",
        ),
        trainers=(Trainer(0, "knee-train-v1pool", "17_train_v1pool"),),
        infer_slug="knee-infer-v1pool",
        infer_directory="18_infer_v1pool",
    ),
    Lineage(
        # Same cache, same config, more epochs. v2 fold 0 was still climbing at
        # epoch 29 of 30 — the last three were 0.725, 0.727, 0.7282 — so 0.7282
        # is a floor for that configuration rather than its ceiling. This mounts
        # the finished run and continues it instead of paying for the first 30
        # epochs again.
        name="v2long",
        cache=CACHE_V2,
        train=TrainConfig(
            backbone="resnet34", epochs=60, batch=4, lr=6e-4, accum=4,
            slice_subsample=18,
            note="Continues knee-train-v2 from its epoch-29 checkpoint. The\n"
                 "cosine schedule is absolute in epoch, so resuming at 30 of 60\n"
                 "puts the LR back at ~3.2e-4 against the ~1.2e-5 floor the\n"
                 "first run finished on — a 26x jump. That is a WARM RESTART,\n"
                 "not a smooth continuation, and it will get worse before it\n"
                 "gets better. The run inherits the 0.7282 checkpoint as its\n"
                 "best, so a restart that never recovers exports the old weights\n"
                 "rather than its own worse ones.",
        ),
        trainers=(Trainer(0, "knee-train-v2-long", "14_train_v2_long",
                          resume_from="knee-train-v2"),),
        infer_slug="knee-infer-v2-long",
        infer_directory="15_infer_v2_long",
    ),
    Lineage(
        # Deliberately on CACHE_V1: this is a backbone experiment, and holding
        # the input fixed is the only way its result attributes to the backbone.
        # That is the discipline the 288px run failed — it moved resolution,
        # epochs, batch, LR and slice count at once, so its number meant nothing.
        name="dinov2",
        cache=CACHE_V1,
        train=TrainConfig(
            backbone="vit_small_patch14_dinov2.lvd142m",
            epochs=16, batch=6, lr=1e-4, accum=3, input_norm=True,
            note="ViT-S/14 at 196px is heavier per image than resnet34, so batch 6\n"
                 "with 3-step accumulation lands near the effective 16 that worked.\n"
                 "ViTs want a lower LR than convnets and DINOv2 features are strong\n"
                 "already, so this is fine-tuning rather than training.\n"
                 "input_norm is True here and False for v1/v2, which is exactly why\n"
                 "the inference kernel refuses to average across lineages.",
        ),
        trainers=(Trainer(0, "knee-train-dinov2", "12_train_dinov2"),),
        infer_slug="knee-infer-dinov2",
        infer_directory="13_infer_dinov2",
    ),
]


# --------------------------------------------------------------------------- #
# Kernels that belong to no lineage: they produce or evaluate the TARGETS rather
# than consume the cache. They are declared here rather than hand-written so
# they inherit the shared code and the metadata contract like everything else.
# --------------------------------------------------------------------------- #
EXTRAS = [
    Kernel(
        slug="knee-embed",
        directory="26_embed",
        template="embed",
        gpu=False,          # 2.2 h frozen on CPU against 191 h fine-tuning
        internet=True,      # pretrained backbone weights
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[ARTIFACTS_DATASET],
        constants={**V1.constants(),
                   "RUN_BACKBONE": "resnet34",
                   "INPUT_NORM": True,     # frozen features want the right input
                   "RUN_MAX_STUDIES": 0,   # 0 = every study
                   "EMBED_THREADS": 4,
                   "RUN_TIME_BUDGET": Raw("10.0 * 3600")},
        note="Writes frozen backbone embeddings for the whole corpus once, so\n"
             "that everything above the backbone can be trained in minutes\n"
             "instead of hours. Measured: fine-tuning on CPU is 191 hours,\n"
             "frozen extraction is 2.2 hours, and a five-fold run of the 73,380\n"
             "parameters above the backbone is 2.6 minutes.\n"
             "\n"
             "input_norm is True here even though the 0.725 model was trained\n"
             "without it. A FROZEN backbone has no chance to adapt to the wrong\n"
             "input distribution, so feeding it what it was pretrained on\n"
             "matters more here than it did there.",
    ),
    Kernel(
        slug="knee-gold-eval",
        directory="23_gold_eval",
        template="gold_eval",
        gpu=False,          # twelve studies through one backbone; CPU is ample
        internet=False,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3", "knee-train"],
        datasets=[ARTIFACTS_DATASET],
        constants={**V1.constants(), "TTA_VIEWS": ("identity",),
                   "OOF_SCOPE": "gold"},
        note="Scores already-trained checkpoints against the expert labels they\n"
             "never saw, on CPU, because the weekly GPU allowance is spent and\n"
             "this is twelve studies through one backbone.\n"
             "\n"
             "knee-train predates the gold dump later runs emit, which leaves a\n"
             "hole in the middle of the only offline signal this project trusts:\n"
             "folds 1-4 pool to n=46 and fold 0 holds the other 12. It also\n"
             "leaves the two fold-0 experiments — per-finding pooling and\n"
             "DINOv2 — with no like-for-like baseline to be measured against.",
    ),
    Kernel(
        slug="knee-tta-eval",
        directory="50_tta_eval",
        template="gold_eval",
        # CPU. This is the whole point: the weekly GPU allowance is 30 hours
        # and every other lever left costs 5-7 of them per measurement, while
        # CPU sessions draw on a separate allowance. 58 studies x 4 views
        # through resnet34 is well inside a CPU session, so this measures a
        # real lever for a GPU cost of zero.
        gpu=False,
        internet=False,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3",
                 "knee-train-v1pub", "knee-train-v1pub-fold1",
                 "knee-train-v1pub-fold2", "knee-train-v1pub-fold3",
                 "knee-train-v1pub-fold4"],
        # PUBLIC_DATASET, not ARTIFACTS_DATASET, and this is load-bearing
        # rather than incidental. The cohort is the intersection of the cache
        # with the label file's index, so a different label set is a different
        # study list and therefore a different GroupKFold split. Mounting the
        # artifacts labels here would score each v1public checkpoint on studies
        # it had trained on, and the number would look held-out and not be.
        datasets=[PUBLIC_DATASET],
        # OOF_SCOPE="gold" restores what this kernel did before the constant
        # existed. It was added to the gold_eval template for the OOF dumps the
        # distillation work needed, and this kernel — which predates it — was
        # never given a value, so its generated run.py has referenced an
        # undefined name ever since. It reads held-out gold only, which is what
        # a TTA measurement wants: the 58 expert studies and nothing else.
        constants={**V1.constants(),
                   "OOF_SCOPE": "gold",
                   "TTA_VIEWS": ("identity", "reverse", "shift_pos", "shift_neg")},
        note="Test-time augmentation, measured out-of-fold on the 58 expert\n"
             "studies, for zero GPU hours.\n"
             "\n"
             "Inference has never used TTA and no experiment has ever tested\n"
             "it, which makes it the last untried lever that is not simply\n"
             "'train more models'. The views are the two geometric symmetries\n"
             "training already teaches — slice-order reversal (p=0.5) and a\n"
             "pixel roll of up to TARGET_SIZE//16 — so a gain here is the\n"
             "model being asked the same question four ways, not four\n"
             "different questions.\n"
             "\n"
             "It mounts the five v1public folds, the 0.923 board ensemble, and\n"
             "writes every view's predictions separately. Which subset to\n"
             "average is then decided offline on the pooled n=58 rather than\n"
             "costing another session to revisit. An unweighted mean of all\n"
             "four views is the default answer, because it fits nothing: the\n"
             "project has twice declined a gain that required a free parameter\n"
             "tuned on 58 studies (E048).",
    ),
    Kernel(
        slug="knee-train-v1pubfull",
        directory="57_train_v1pubfull",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,          # negative = train on everything
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, seed=3).constants()},
        note="Full fit: the 0.923 configuration trained on every study, with\n"
             "nothing held out.\n"
             "\n"
             "The fold split exists to produce an out-of-fold score, not\n"
             "because the model needs it. Once the configuration is settled,\n"
             "holding out a fifth of the corpus spends a fifth of the training\n"
             "data on a number already measured — and it costs more than that\n"
             "where it matters most, because each fold model never sees ~12 of\n"
             "the 58 expert studies. Those carry GOLD_WEIGHT=8.0 and are the\n"
             "only labels known to match what the leaderboard scores. This one\n"
             "sees all 58.\n"
             "\n"
             "It is a DATA lever, not an architecture one, which is the whole\n"
             "reason it is worth GPU: labels and data account for +0.166 of the\n"
             "+0.198 this project has gained, while every architecture lever\n"
             "measured zero, negative, or inside the +-0.03 noise floor E060\n"
             "established.\n"
             "\n"
             "The trade is that it cannot be validated. It exports at a FIXED\n"
             "epoch (E055: the five honest folds peak at 20-21 and plateau over\n"
             "18-21), writes no gold dump, and is named checkpoint_foldall.pt so\n"
             "that inference picks it up as an ensemble member while gold_eval\n"
             "skips it — a model that trained on every gold study must never be\n"
             "scored as though it had not.",
    ),
    Kernel(
        slug="knee-infer-v1pub10",
        directory="56_infer_v1pub10",
        template="infer",
        gpu=True,
        internet=False,     # a submission kernel
        # Ten members: the five folds behind the 0.923 board result and the
        # five second-seed folds. Nothing in the inference kernel needed
        # changing to allow this — it discovers checkpoints by globbing the
        # mounted notebooks rather than listing them, so a member joins by
        # being mounted. The two lineages agree on both properties the
        # ensemble guard checks (slice_subsample None, input_norm False), so
        # all ten load rather than being refused.
        depends=["knee-train-v1pub", "knee-train-v1pub-fold1",
                 "knee-train-v1pub-fold2", "knee-train-v1pub-fold3",
                 "knee-train-v1pub-fold4",
                 "knee-train-v1pubB", "knee-train-v1pubB-fold1",
                 "knee-train-v1pubB-fold2", "knee-train-v1pubB-fold3",
                 "knee-train-v1pubB-fold4"],
        constants={**V1.constants(),
                   "BATCH_STUDIES": V1.infer_batch,
                   "SLICE_SUBSAMPLE_EXPECTED": None,
                   "INPUT_NORM_EXPECTED": False,
                   "MEMBERS_EXPECTED": 10},
        note="Ten members: v1public plus its second seed.\n"
             "\n"
             "The only lever with a board-measured coefficient that survived\n"
             "the sweep of E050-E058. E036 measured ensembling at +0.032 going\n"
             "from one fold to five; log-scaling puts five to ten at roughly\n"
             "+0.010. Nothing here is a new hypothesis, which is the point.\n"
             "\n"
             "E054 measured a ten-member ensemble WORSE than its five when the\n"
             "added members were 0.107 behind (v1fused, -0.0296 [-0.046,\n"
             "-0.015]). That result is why this one is not assumed: the\n"
             "difference is that a reseed is an equal-strength member rather\n"
             "than a weak one, and E054's rule decides it on gold before this\n"
             "kernel is pushed, not after.\n"
             "\n"
             "Budget is not a constraint and was measured rather than guessed:\n"
             "0.98 h projected for 1,300 test studies with five members\n"
             "against a 9 h cap, and 0.037 h per extra member. Ten members is\n"
             "~1.2 h.",
    ),
    Kernel(
        slug="knee-infer-v1pubfull",
        directory="58_infer_v1pubfull",
        template="infer",
        gpu=True,
        internet=False,     # a submission kernel
        # The five folds behind 0.923 PLUS one full-fit member, which makes
        # this a one-variable submission against that score: same five models,
        # one addition. Deliberately not combined with v1publicB's five in the
        # same submission — two changes at once would leave a board move
        # unattributable, and the board is the only instrument that can see
        # either of them.
        depends=["knee-train-v1pub", "knee-train-v1pub-fold1",
                 "knee-train-v1pub-fold2", "knee-train-v1pub-fold3",
                 "knee-train-v1pub-fold4",
                 "knee-train-v1pubfull"],
        constants={**V1.constants(),
                   "BATCH_STUDIES": V1.infer_batch,
                   "SLICE_SUBSAMPLE_EXPECTED": None,
                   "INPUT_NORM_EXPECTED": False,
                   "MEMBERS_EXPECTED": 6},
        note="Six members: the five v1public folds and one full-fit model.\n"
             "\n"
             "The full-fit member cannot be evaluated offline at all — it\n"
             "trained on all 58 gold studies, so every out-of-fold instrument\n"
             "this project owns is blind to it by construction. That is not a\n"
             "gap in the measurement, it is the measurement: the board is the\n"
             "only judge, which is why this exists as its own submission\n"
             "rather than folded in with anything else.\n"
             "\n"
             "Cost is not a consideration: E050 measured an extra member at\n"
             "0.037 h on the full test set, so six is ~1.0 h against a 9 h cap.",
    ),
    Kernel(
        slug="knee-train-v1pubfull-s4",
        directory="59_train_v1pubfull_s4",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,          # negative = train on everything
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, seed=4).constants()},
        note="Full-fit member 1 of the four that turn E064's probe into a\n"
             "lineage. Byte-identical to knee-train-v1pubfull except for the\n"
             "seed, which is the only way to draw a second full-fit model at\n"
             "all: there are no folds left to vary when every study is in\n"
             "training.\n"
             "\n"
             "E064 measured the seed=3 full fit at 0.924 on the board against\n"
             "0.923, as one member in six. That is +0.001 at a sixth of the\n"
             "weight and it is the smallest move the board can show, so it\n"
             "is a direction rather than a size. These four exist so the\n"
             "lever can be measured at full weight instead.",
    ),
    Kernel(
        slug="knee-train-v1pubfull-s5",
        directory="60_train_v1pubfull_s5",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,          # negative = train on everything
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, seed=5).constants()},
        note="Full-fit member 2 of the four that turn E064's probe into a\n"
             "lineage. Byte-identical to knee-train-v1pubfull except for the\n"
             "seed, which is the only way to draw a second full-fit model at\n"
             "all: there are no folds left to vary when every study is in\n"
             "training.\n"
             "\n"
             "E064 measured the seed=3 full fit at 0.924 on the board against\n"
             "0.923, as one member in six. That is +0.001 at a sixth of the\n"
             "weight and it is the smallest move the board can show, so it\n"
             "is a direction rather than a size. These four exist so the\n"
             "lever can be measured at full weight instead.",
    ),
    Kernel(
        slug="knee-train-v1pubfull-s6",
        directory="61_train_v1pubfull_s6",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,          # negative = train on everything
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, seed=6).constants()},
        note="Full-fit member 3 of the four that turn E064's probe into a\n"
             "lineage. Byte-identical to knee-train-v1pubfull except for the\n"
             "seed, which is the only way to draw a second full-fit model at\n"
             "all: there are no folds left to vary when every study is in\n"
             "training.\n"
             "\n"
             "E064 measured the seed=3 full fit at 0.924 on the board against\n"
             "0.923, as one member in six. That is +0.001 at a sixth of the\n"
             "weight and it is the smallest move the board can show, so it\n"
             "is a direction rather than a size. These four exist so the\n"
             "lever can be measured at full weight instead.",
    ),
    Kernel(
        slug="knee-train-v1pubfull-s7",
        directory="62_train_v1pubfull_s7",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,          # negative = train on everything
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, seed=7).constants()},
        note="Full-fit member 4 of the four that turn E064's probe into a\n"
             "lineage. Byte-identical to knee-train-v1pubfull except for the\n"
             "seed, which is the only way to draw a second full-fit model at\n"
             "all: there are no folds left to vary when every study is in\n"
             "training.\n"
             "\n"
             "E064 measured the seed=3 full fit at 0.924 on the board against\n"
             "0.923, as one member in six. That is +0.001 at a sixth of the\n"
             "weight and it is the smallest move the board can show, so it\n"
             "is a direction rather than a size. These four exist so the\n"
             "lever can be measured at full weight instead.",
    ),
    *[
        Kernel(
            slug=f"knee-train-v1pubfe-s{seed}",
            directory=f"{74 + index}_train_v1pubfe_s{seed}",
            template="train",
            gpu=True,
            internet=True,
            depends=["knee-cache-build-0", "knee-cache-build-1",
                     "knee-cache-build-2", "knee-cache-build-3"],
            datasets=[PUBLIC_DATASET],
            constants={"RUN_FOLD": -1,          # negative = train on everything
                       **V1.constants(),
                       **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                     lr=6e-4, seed=seed,
                                     full_fit_epoch_early=16).constants()},
            note=f"Full-fit member {index + 1} of five, writing TWO exports from\n"
                 "one trajectory: checkpoint_foldall.pt at epoch 20 and\n"
                 "early_foldall.pt at epoch 16.\n"
                 "\n"
                 "THE QUESTION. FULL_FIT_EPOCH=20 was read off the FOLD models\n"
                 "(E055), which train on 3,526 studies. A full-fit model trains\n"
                 "on 4,407, so the same epoch number buys it 25% more\n"
                 "optimisation steps. If the peak is governed by steps rather\n"
                 "than by passes, the fold optimum of 20 x 3,526 = 70,520\n"
                 "study-visits lands at epoch 16 here — and E055 measured the\n"
                 "fold curves DECAYING past 21. On that reading every full-fit\n"
                 "member behind the 0.926 board score is a quarter past its\n"
                 "best. On the other reading, more data per pass supports more\n"
                 "passes and 16 is simply undertrained. Nothing in this repo\n"
                 "settles it, and a full-fit model cannot be validated offline\n"
                 "by construction, so the board decides.\n"
                 "\n"
                 "WHY BOTH EXPORTS COME FROM ONE RUN. Two separate runs would\n"
                 "differ in seed as well as epoch, and E060 measured a pure\n"
                 "reseed at -0.0284 on gold — larger than any effect expected\n"
                 "here. Same seed, same data order, same weights up to epoch\n"
                 "16 makes the export epoch the only difference between the\n"
                 "two arms.\n"
                 "\n"
                 "THE BONUS ARM, and it is free. These are new seeds, so the\n"
                 "epoch-20 ensemble is also a five-member SEED CONTROL against\n"
                 "the standing 0.926. E061 priced a reseed on gold and E064\n"
                 "priced one at ten members against five, but the board has\n"
                 "never seen a like-for-like reseed of a full-weight ensemble.\n"
                 "\n"
                 "Cost: ~1.5 GPU-h each. The early export is written during a\n"
                 "run that was happening anyway, so the second arm is free —\n"
                 "E039's rule that a probe must not cost anything when it\n"
                 "succeeds.",
        )
        for index, seed in enumerate((11, 12, 13, 14, 15))
    ],
    Kernel(
        slug="knee-infer-v1pubfe-early",
        directory="79_infer_v1pubfe_early",
        template="infer",
        gpu=True,
        internet=False,     # a submission kernel
        # The epoch-16 arm. CHECKPOINT_GLOB is why this and the epoch-20 arm
        # can share five training runs without contaminating each other: the
        # two exports are named so neither glob matches the other, and a kernel
        # that mounted both would double-count every model while its member
        # count still looked right.
        depends=[f"knee-train-v1pubfe-s{seed}" for seed in (11, 12, 13, 14, 15)],
        constants={**V1.constants(),
                   "BATCH_STUDIES": V1.infer_batch,
                   "SLICE_SUBSAMPLE_EXPECTED": None,
                   "INPUT_NORM_EXPECTED": False,
                   "CHECKPOINT_GLOB": "early_fold*.pt",
                   "MEMBERS_EXPECTED": 5},
        note="Five full-fit members exported at epoch 16 instead of 20.\n"
             "\n"
             "Against knee-infer-v1pubfe — the same five models, same seeds,\n"
             "same trajectory, exported at epoch 20 — the export epoch is the\n"
             "ONLY variable. That is the comparison. The standing 0.926 is a\n"
             "different seed draw and is NOT it.\n"
             "\n"
             "PRE-REGISTERED READING:\n"
             "  early > late   the fold-derived epoch was overtraining every\n"
             "                 full-fit member and 0.926 was left short\n"
             "  early < late   more data per pass supports more passes, 20 is\n"
             "                 right, and the step-count argument is wrong\n"
             "  within 0.001   the curve is as flat here as E055 found it over\n"
             "                 18-21, and the export epoch does not matter",
    ),
    Kernel(
        slug="knee-infer-v1pubfe",
        directory="80_infer_v1pubfe",
        template="infer",
        gpu=True,
        internet=False,     # a submission kernel
        # The epoch-20 arm from the same five runs. Serves twice: as the
        # control for the epoch comparison, and as the board's first
        # like-for-like reseed of a full-weight five-member ensemble.
        depends=[f"knee-train-v1pubfe-s{seed}" for seed in (11, 12, 13, 14, 15)],
        constants={**V1.constants(),
                   "BATCH_STUDIES": V1.infer_batch,
                   "SLICE_SUBSAMPLE_EXPECTED": None,
                   "INPUT_NORM_EXPECTED": False,
                   "MEMBERS_EXPECTED": 5},
        note="Five full-fit members at epoch 20 — the control arm for\n"
             "knee-infer-v1pubfe-early, and a seed control for 0.926.\n"
             "\n"
             "Against the standing 0.926 (knee-infer-v1pubfull5, seeds 3-7)\n"
             "this changes only the seed. E061 priced a reseed on gold at\n"
             "0.8827 against 0.8980 and E064 priced one on the board at ten\n"
             "members against five, but no like-for-like reseed of a\n"
             "full-weight ensemble has ever been submitted. Whatever it\n"
             "returns bounds how much of any future full-fit result is draw\n"
             "rather than lever.",
    ),
    Kernel(
        slug="knee-infer-v1pubfull5",
        directory="63_infer_v1pubfull5",
        template="infer",
        gpu=True,
        internet=False,     # a submission kernel
        # Five full-fit models and NOTHING ELSE. The point of the lineage is
        # that every member saw all 4,407 studies and all 58 expert ones;
        # mounting the fold models alongside would dilute that back to the
        # sixth-weight mixture E064 already priced at +0.001.
        depends=["knee-train-v1pubfull", "knee-train-v1pubfull-s4",
                 "knee-train-v1pubfull-s5", "knee-train-v1pubfull-s6",
                 "knee-train-v1pubfull-s7"],
        constants={**V1.constants(),
                   "BATCH_STUDIES": V1.infer_batch,
                   "SLICE_SUBSAMPLE_EXPECTED": None,
                   "INPUT_NORM_EXPECTED": False,
                   "MEMBERS_EXPECTED": 5},
        note="The full-fit lever at full weight: five members, every one of\n"
             "them trained on the whole corpus.\n"
             "\n"
             "Against the 0.923 five-fold ensemble this changes exactly one\n"
             "thing — how much data each member saw. Same architecture, same\n"
             "geometry, same labels, same cache, same member count. Each fold\n"
             "model trains on 80% of the studies and misses ~12 of the 58\n"
             "expert ones; each of these sees all of both. It is the data\n"
             "lever isolated, which is the category worth +0.166 of this\n"
             "project's +0.198.\n"
             "\n"
             "It cannot be scored offline and that is not a defect: a model\n"
             "trained on every gold study makes every out-of-fold instrument\n"
             "here blind by construction. The board is the only judge, so\n"
             "this is its own submission and shares it with nothing.\n"
             "\n"
             "Cost: E050 measured 0.037 h per member on the full test set, so\n"
             "five is ~1.0 h against a 9 h cap.",
    ),
    Kernel(
        slug="knee-infer-v1pubmix",
        directory="73_infer_v1pubmix",
        template="infer",
        gpu=True,
        internet=False,     # a submission kernel
        # Ten members: the five fold models behind 0.923 and the five full-fit
        # models behind 0.926. No training — every checkpoint already exists,
        # so this costs inference and a free submission and nothing else.
        depends=["knee-train-v1pub", "knee-train-v1pub-fold1",
                 "knee-train-v1pub-fold2", "knee-train-v1pub-fold3",
                 "knee-train-v1pub-fold4",
                 "knee-train-v1pubfull", "knee-train-v1pubfull-s4",
                 "knee-train-v1pubfull-s5", "knee-train-v1pubfull-s6",
                 "knee-train-v1pubfull-s7"],
        constants={**V1.constants(),
                   "BATCH_STUDIES": V1.infer_batch,
                   "SLICE_SUBSAMPLE_EXPECTED": None,
                   "INPUT_NORM_EXPECTED": False,
                   "MEMBERS_EXPECTED": 10},
        note="The two ensembles the board has priced, united. E083 put the\n"
             "fold five at 0.923 and the full-fit five at 0.926.\n"
             "\n"
             "WHY THIS IS NOT E064's TEN-MEMBER NULL, which scored 0.923 for\n"
             "ten members against 0.923 for five. Those ten were v1public and\n"
             "a RESEED of v1public — same configuration, same data exposure,\n"
             "same kind. E075 established what that costs: same-kind members\n"
             "make correlated errors, so averaging inherits the weaker one's\n"
             "bias without cancelling anything, and the reseed was weaker\n"
             "everywhere it was looked at. These ten differ in the one thing\n"
             "the board says matters — 80% of the corpus per member against\n"
             "100% — which is the different-kind case.\n"
             "\n"
             "AND THE COMPARABILITY IS THE TIGHTEST IN THE LOG. E048's rule is\n"
             "that a union pays when its members are comparable: E023's two\n"
             "readers sat 0.0025 apart and their union beat BOTH of them by\n"
             "0.070. These sit 0.003 apart. Every union that failed since added\n"
             "a member 0.03-0.06 adrift.\n"
             "\n"
             "THE HONEST CASE AGAINST, because one data point already argues\n"
             "the other way. Five folds plus ONE full-fit scored 0.924 (E064).\n"
             "Linear interpolation on mixing fraction predicts 0.9235 for that\n"
             "point and ~0.9245 here — below the 0.926 that pure full-fit\n"
             "already banks. If the ensemble is linear in mixture, this is a\n"
             "dilution and not a union. What that single point cannot do is\n"
             "separate linear from mildly super-additive: the board reports\n"
             "three decimals, and 0.9235 and 0.924 are the same number to it.\n"
             "E023 is this project's proof that unions here are not always\n"
             "linear, so the question is open and cheap.\n"
             "\n"
             "PRE-REGISTERED READING, so the answer is not chosen afterwards:\n"
             "  > 0.926  the mix wins and different-kind ensembling is live\n"
             "  0.924-0.926  linear dilution; pure full-fit stands and the\n"
             "               mixing question is closed for good\n"
             "  < 0.923  suspect the kernel, not the science, and read the log\n"
             "\n"
             "Cost: no training at all. E050 measured 0.037 h per member, so\n"
             "ten is ~1.2 h against a 9 h cap, out of 10.75 h left this week.",
    ),
    # ------------------------------------------------------------------ #
    # THE CC0 CoAtNet ARM. Someone else's model, our kernel, their credit.
    # ------------------------------------------------------------------ #
    # E088 changed what is reachable here. The blocker was never the licence on
    # the weights — those are CC0-1.0 — it was that upstream persists no output
    # files, so its predictions cannot be mounted (E085 tried; the blend
    # correctly refused with one member). The weights ARE published as datasets,
    # so the arm can be RUN here from CC0 inputs with attribution, instead of
    # forking a notebook or mounting an output that does not exist.
    #
    # E088 also found the model is fully specified by the checkpoint: `arch`
    # names a stock timm backbone, `res` the geometry, and the head is eight
    # tensors. HANDOFF.md §4d's "reproducing RaptorClassifier and build_backbone
    # is copying while claiming otherwise" is therefore only half true — the
    # model is given. What is genuinely theirs is the input pipeline, and the
    # template says so at the top and names every source.
    Kernel(
        slug="knee-infer-raptorcc0",
        directory="81_infer_raptorcc0",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan"],
        constants={
            "MEMBERS_EXPECTED": 1,
            "ARMS": ({**RAPTOR_V5, "w": 1.0},),
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            # A few unreadable studies is robustness; most of them is a broken
            # kernel writing a constant-0.5 submission that scores 0.500 and is
            # indistinguishable from a bad model.
            "FALLBACK_LIMIT": 0.02,
            # How many studies the decode pool may run ahead of the GPU. Each
            # decoded volume is ~7 MB, so 32 is ~230 MB of slack; an unbounded
            # map over 1,300 studies would hold them all.
            "DECODE_AHEAD": 32,
            # The hidden set. Writes submission.csv and scores nothing offline.
            "EVAL_SPLIT": "test",
            "GOLD_EXPECTED": 58,
            # The second architecture is off: CoAtNet arms only.
            "V1_MEMBERS": None,
        },
        note="THE CONTROL, and it must be submitted before any blend.\n"
             "\n"
             "One CC0 CoAtNet checkpoint (maxspan-v5), no blending, no TTA.\n"
             "\n"
             "THE NUMBERS THAT BOUND IT, and whose they are. 0.924 on the\n"
             "board is upstream's for the SINGLE v4 model, which is a\n"
             "different checkpoint from this one. For v5 there are two self-\n"
             "reports and they disagree: the file itself carries gold_auc\n"
             "0.9214, while the 4-arm write-up re-measures it at 0.9198 at\n"
             "inference geometry. v4's file says 0.9167 and scored 0.924, so\n"
             "v5 should land at or a little above 0.924 if it reproduces.\n"
             "\n"
             "PRE-REGISTERED READING, before the score arrives:\n"
             "  0.92-0.93   the arm reproduces and is a trustworthy member\n"
             "  under 0.90  something in OUR mounting or preprocessing is\n"
             "              wrong, and no later blend can be read at all\n"
             "  over 0.935  better than any single-model self-report in this\n"
             "              family; re-read before believing it\n"
             "\n"
             "The two self-reports for v5 differ by 0.0016 and v8's differ by\n"
             "0.0049. That spread is the floor on how precisely ANY of these\n"
             "numbers can be read, and it is the same order as the effects\n"
             "the blend is being justified by.\n"
             "\n"
             "E039: run the thing that costs nothing if it succeeds. This\n"
             "costs one submission and answers whether the member is real.\n"
             "\n"
             "ATTRIBUTION: model, geometry and slot layout are Dread\n"
             "Development's, from the public notebooks named in the template\n"
             "header. Weights CC0-1.0, licence read 2026-09-08 (E088).",
    ),
    Kernel(
        slug="knee-infer-raptorcc0x4",
        directory="82_infer_raptorcc0x4",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        constants={
            "MEMBERS_EXPECTED": 4,
            # The four sub-models published with the 0.937 system, each with the
            # geometry it was trained at. maxspan-v5 appears twice — once
            # straight and once with its slice triplet reversed — so four
            # members cost three checkpoints.
            "ARMS": RAPTOR_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            # A few unreadable studies is robustness; most of them is a broken
            # kernel writing a constant-0.5 submission that scores 0.500 and is
            # indistinguishable from a bad model.
            "FALLBACK_LIMIT": 0.02,
            # How many studies the decode pool may run ahead of the GPU. Each
            # decoded volume is ~7 MB, so 32 is ~230 MB of slack; an unbounded
            # map over 1,300 studies would hold them all.
            "DECODE_AHEAD": 32,
            # The hidden set. Writes submission.csv and scores nothing offline.
            "EVAL_SPLIT": "test",
            "GOLD_EXPECTED": 58,
            # The second architecture is off: CoAtNet arms only.
            "V1_MEMBERS": None,
        },
        note="The CoAtNet arm of the public 0.937 system, at its published\n"
             "weights: maxspan-v5 0.55, native384-v8 0.20, maxspan-v5 flipped\n"
             "0.15, native384dense-v10 0.10. Its authors call CoAtNet their\n"
             "strongest single arm and give it base weight 0.60 of four.\n"
             "\n"
             "EVERY WEIGHT FILE HERE IS CC0-1.0. The `tonylica` asset that\n"
             "gates that system's DINOv3 arm is deliberately NOT mounted: its\n"
             "licence reads `Other (specified in description)` and the\n"
             "description is EMPTY, so no grant exists and E043's avoid tier\n"
             "applies on the merits rather than on the label (E088).\n"
             "\n"
             "DO NOT PUSH THIS BEFORE `knee-infer-raptorcc0` HAS SCORED.\n"
             "Three checkpoints and a flip is four ways for a mounting bug to\n"
             "hide; the single-arm control is what makes this readable.\n"
             "\n"
             "Weights are published, not fitted here. A weight tuned on 58\n"
             "studies is a free parameter fitted to 58 studies, which this\n"
             "project has declined four times (E048, E069, E081, E084).\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorcc0`, plus the blend weights\n"
             "from the public write-up `4-arm-ensemble-explained-rsna-knee-0-937`.",
    ),
    # THE INSTRUMENT THIS PROJECT HAS NEVER HAD.
    #
    # Every CoAtNet number in the log is an author's self-report. E090 verified
    # the checkpoints carry `gold_auc` 0.9214 / 0.9174 / 0.9067 and recorded, in
    # the same entry, that *"there is no offline gold evaluation available for
    # this arm"*. E097 then spent a board submission to learn the four-arm blend
    # beats one arm by +0.004, and E098 closed six foreign blends on gold-58
    # while noting that gold-58 has never once seen a blend gain — a caution it
    # had no way to test.
    #
    # This kernel tests it. Same template, same preprocessing, same forward
    # pass, pointed at the 58 expert-labelled TRAINING studies instead of the
    # hidden set. It writes no submission and costs no submission: the gold
    # labels are in the competition's own `train.csv` (exactly 58 of 4,407 rows
    # have all twelve findings filled), so no extra dataset is mounted and the
    # only inputs are the three CC0 `dreaddevelopment` weight sets.
    #
    # THE ARMS HELD GOLD OUT OF TRAINING. Upstream's write-up states it and the
    # per-file `gold_auc` values only mean anything under it. If that is wrong
    # the measurement here is contaminated upward, and the tell is a macro far
    # ABOVE 0.9214 rather than near it.
    #
    # WHAT IT DECIDES, pre-registered in E099 before it runs:
    #   1. measured v5 within +/-0.01 of 0.9214 -> the self-reports transfer and
    #      E048's gap table stands as written.
    #   2. measured v5 below 0.90 -> they do not transfer, and every `expect_gold`
    #      in this file is a mount fingerprint only, never a score.
    #   3. blend beats best single arm on gold-58 -> gold-58 can see a blend gain
    #      and E098's caution retires. It does not -> gold-58 is blind to a gain
    #      the board measured at +0.004, and the four "not separated" blend
    #      readings (E033, E039, E046, E048) are void.
    # No outcome here is uninformative, which is the whole reason to run it.
    Kernel(
        slug="knee-gold-raptorcc0x10",
        directory="98_gold_raptorcc0x10",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense",
                  # The six screened candidates (E117). All CC0-1.0, all from the
                  # same account the four incumbent arms already come from.
                  "dreaddevelopment/raptor-knee-widedense",
                  "dreaddevelopment/raptor-knee-fullspan",
                  "dreaddevelopment/raptor-knee-finespacing",
                  "dreaddevelopment/raptor-knee-widefov",
                  "dreaddevelopment/raptor-knee-arms"],
        constants={
            "MEMBERS_EXPECTED": 10,
            "ARMS": RAPTOR_ARMS + RAPTOR_CANDIDATES,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "gold",
            "GOLD_EXPECTED": 58,
            "V1_MEMBERS": None,
        },
        note="NOT A SUBMISSION. E117's SCREEN. Scores the four incumbent CC0\n"
             "arms AND six unused CC0 checkpoints from the same account on the\n"
             "58 expert studies, and dumps every arm's per-study probabilities.\n"
             "\n"
             "WHY THIS EXISTS. The blend runs THREE distinct checkpoints. The\n"
             "account publishes FOURTEEN, all CC0, all licence-cleared in E086,\n"
             "and no experiment has ever tested the other eleven. One of them,\n"
             "`coatnet_v4_full` at a stored 0.9167, outscores `native384-v8` at\n"
             "0.9067 which the blend already weights at 0.20.\n"
             "\n"
             "AND THE RUNTIME IS ALREADY PAID FOR. The submission uses 2.80 h\n"
             "of a 9 h cap (1.13 + 0.58 + 0.41 CoAtNet, 0.68 v1), so there are\n"
             "~6 h idle. Adding members is also the ONLY operation this project\n"
             "has measured moving the board: five members scored 0.938 at two\n"
             "blend weights, six scored 0.940 at two, so member count moved it\n"
             "+0.002 twice while blend weight moved it 0.000 twice.\n"
             "\n"
             "WHAT TO READ FIRST in the log: the `[gold]` lines, and read them\n"
             "against each file's own `author's gold` printed at load. The four\n"
             "incumbents are the CONTROL -- if maxspan-v5 does not land near\n"
             "0.9214 the run is not readable and nothing else in it means\n"
             "anything. Every candidate runs at the DENSE geometry because the\n"
             "per-variant geometry is not in the files, so a candidate scoring\n"
             "below its stored number may be mis-geometried rather than weak.\n"
             "\n"
             "It writes no submission.csv on purpose: these 58 studies are\n"
             "TRAINING data, and a notebook with no submission.csv cannot be\n"
             "submitted by accident.\n"
             "\n"
             "COST: 58 studies, three preprocessing signatures, all of which\n"
             "the incumbent arms already build. Budget ~1.5 GPU-h.\n"
             "\n"
             "ATTRIBUTION: all weights from `dreaddevelopment`, CC0-1.0.",
    ),
    Kernel(
        slug="knee-gold-raptorcc0x4",
        directory="83_gold_raptorcc0x4",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        constants={
            "MEMBERS_EXPECTED": 4,
            "ARMS": RAPTOR_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            # The 58 expert studies. Writes gold_probs.csv, gold_truth.csv and
            # gold_scores.json; writes NO submission.csv, so it cannot be
            # submitted by accident against a set it does not contain.
            "EVAL_SPLIT": "gold",
            "GOLD_EXPECTED": 58,
            "V1_MEMBERS": None,
        },
        note="NOT A SUBMISSION. Scores the four CC0 CoAtNet arms on this\n"
             "project's own 58 expert studies and dumps their raw per-study\n"
             "probabilities, so blend weights can be searched offline instead\n"
             "of on the board at 2.4 GPU-h and one submission per question.\n"
             "\n"
             "It writes no submission.csv on purpose: these 58 studies are\n"
             "TRAINING data. A notebook with no submission.csv cannot be\n"
             "submitted, which is the safe failure.\n"
             "\n"
             "COST: 58 studies against 82's 1,300, same four arms, so about\n"
             "a tenth of 82's scored time plus the same fixed setup. Run it,\n"
             "do not submit it, and download gold_probs.csv.\n"
             "\n"
             "WHAT TO READ FIRST in the log: the `[gold]` lines. If\n"
             "maxspan-v5 does not land near 0.9214 the self-reports do not\n"
             "transfer and E097's +0.004 was bought on a premise this\n"
             "project never checked.\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorcc0`.",
    ),
    Kernel(
        slug="knee-train-lab-dread",
        directory="88_train_lab_dread",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[DREAD_DATASET],
        # RUN_FOLD=-1 is a full fit, and here it costs nothing in honesty: the
        # label set has no gold rows, so `build_cohort` finds no gold study to
        # put in `studies` at all and the 58 stay outside training whatever this
        # is set to. The log will read "gold studies in cache: 0", which is the
        # tell that the holdout is real.
        constants={"RUN_FOLD": -1,
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, seed=3).constants()},
        note="ARM A of the first label comparison this project has measured on a\nMODEL rather than on the labels themselves.\n\nTrains the 0.923 configuration on `dreaddevelopment`'s CC0 soft\nlabels over 4,349 studies - every study except exactly the 58\ngold. So this is a FULL FIT that is still honestly scoreable: the\n58 are outside the training set by construction, not by a split.\n\nE089 called this label set unscoreable and it is, as LABELS. The\nmodel trained on them is not, and that is the whole point.\n\nIts authors' own headline for these labels is +0.013 AUC (E088),\nself-reported. This measures it.\n\nONE VARIABLE. `knee-train-lab-pub4349` is the control: same 4,349\nstudies, same resnet34, same 192px/0.6mm geometry, same 24 epochs,\nsame batch, same LR, same seed 3. Only the label source differs.\n\nATTRIBUTION: labels from `dreaddevelopment/rsna-knee-labels`,\nCC0-1.0, repackaged into this pipeline's schema with credit.",
    ),
    Kernel(
        slug="knee-train-lab-pub4349",
        directory="89_train_lab_pub4349",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUB4349_DATASET],
        # RUN_FOLD=-1 is a full fit, and here it costs nothing in honesty: the
        # label set has no gold rows, so `build_cohort` finds no gold study to
        # put in `studies` at all and the 58 stay outside training whatever this
        # is set to. The log will read "gold studies in cache: 0", which is the
        # tell that the holdout is real.
        constants={"RUN_FOLD": -1,
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, seed=3).constants()},
        note="ARM B, THE CONTROL, and it must be run. Same 4,349 studies as\n`knee-train-lab-dread`, labelled by the incumbent\n`stevenleehans` set instead.\n\nWithout it the comparison is against `knee-infer-v1pub`'s 0.8980,\nwhich is a FIVE-FOLD OUT-OF-FOLD number from models that each saw\n80% of the corpus. Reading a full-fit arm against it would confound\nthe label change with the data change - and E060 is in this log\nprecisely because a claim was made without a control arm.\n\nIt is also worth having on its own: the incumbent labels have never\nbeen measured at full corpus with an honest holdout, because the\nfull-fit lineage trains on the gold and cannot be scored.\n\nATTRIBUTION: labels from `stevenleehans/rsna-knee-llm-report-labels`,\nCC0-1.0, as `knee-train-v1pub`.",
    ),
    # E110b. A SIXTH v1 MEMBER, BUILT TO REACH THE BOARD RATHER THAN TO SETTLE
    # A QUESTION.
    #
    # The shipped blend's v1 half is five full-fit resnet34. This adds a sixth
    # member that is a DIFFERENT DEPTH on the same labels and geometry. It is not
    # an experiment: a full-fit model trains on all 58 gold and so cannot be
    # scored offline at all (E083, `knee-infer-v1pubfull5`). The board is the only
    # judge, the board keeps a team's best submission, and 0.938 is banked — so
    # the cost of being wrong is one click.
    #
    # WHY resnet50 AND NOT A FOLD PROBE. A fold-0 resnet50 would be the readable
    # experiment, and it cannot ship: a fold-0 win needs five folds (~7 GPU-h)
    # before anything reaches the board, which does not fit the quota left. This
    # is the version that can.
    #
    # input_norm=False, AND THAT IS E109's ONE DELIVERED FINDING BEING USED.
    # E109's control measured ImageNet normalisation at **-0.0064** on fold 0's
    # 882 held-out studies, CI [-0.0113, -0.0014] — it HURTS this lineage. The
    # incumbent had it off by default rather than by measurement; this arm has it
    # off on purpose.
    #
    # batch 8 x 2 accumulation rather than 16 x 1: resnet50's activations are
    # roughly twice resnet34's and each study is 60 images, so batch 16 is 960
    # forward passes of them — the size that OOMed the T4 in E109. UNLIKE that
    # entry's convnext this is a BatchNorm backbone, so accumulation is NOT
    # exactly equivalent to the larger batch. That is acceptable here and would
    # not have been in E109: this is an ensemble member, not a controlled
    # comparison, so differing batch statistics cost nothing readable.
    Kernel(
        slug="knee-train-v1pubfull-rx50",
        directory="99_train_v1pubfull_rx50",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,
                   **V1.constants(),
                   **TrainConfig(backbone="resnext50_32x4d", epochs=24, batch=8, accum=2,
                                 lr=6e-4, input_norm=False, seed=3).constants()},
        note="E119. A NEW v1 MEMBER FROM A DIFFERENT BACKBONE FAMILY.\n"
             "\n"
             "resnext50_32x4d: grouped convolutions at resnet50's depth and\nparameter count, so it is the smallest possible step OFF the\nresnet trunk - the family changes, the capacity does not.\n"
             "\n"
             "WHY FAMILY AND NOT SEED. E064 submitted a second seed and the\n"
             "board returned +0.000. E111 changed DEPTH (resnet34 -> resnet50)\n"
             "and the board returned +0.002, twice, at two different blend\n"
             "weights. So reseeding this lineage is measured worthless and\n"
             "changing what the network IS is the part that paid. This goes\n"
             "one step further than depth: a different family.\n"
             "\n"
             "EVERYTHING ELSE IS HELD EQUAL to the resnet50 member - 24\n"
             "epochs, batch 8 x 2 accumulation, LR 6e-4, seed 3,\n"
             "input_norm=False, V1 geometry, the same 4,349 public CC0\n"
             "labels. The backbone is the only difference.\n"
             "\n"
             "batch 8 x 2 rather than 16 because E109 OOMed a T4 at batch 16\n"
             "on a 25M-parameter backbone, and convnext_tiny then failed\n"
             "twice at 192px x 60 images per study. Memory here is tight and\n"
             "the effective batch stays 16 either way.\n"
             "\n"
             "input_norm=False by MEASUREMENT, not default: E109 put ImageNet\n"
             "normalisation at -0.0064 on 882 held-out studies, CI\n"
             "[-0.0113, -0.0014], on this lineage.\n"
             "\n"
             "FULL FIT, so it cannot be scored offline - it trains on all 58\n"
             "gold. The board is the only judge. 0.940 is banked and the\n"
             "leaderboard keeps a team's best, so being wrong costs one click.\n"
             "\n"
             "ATTRIBUTION: labels from `dreaddevelopment/rsna-knee-labels`,\n"
             "CC0-1.0, repackaged as `knee-phase1-public`.",
    ),
    Kernel(
        slug="knee-train-v1pubfull-shuffle",
        directory="100_train_v1pubfull_shuffle",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,
                   **V1.constants(),
                   **TrainConfig(backbone="shufflenet_v2_x1_0", epochs=24, batch=8, accum=2,
                                 lr=6e-4, input_norm=False, seed=3).constants()},
        note="E119. A NEW v1 MEMBER FROM A DIFFERENT BACKBONE FAMILY.\n"
             "\n"
             "tf_efficientnet_b0: inverted residuals and squeeze-excite at 5M\nparameters, a genuinely different inductive bias and a fifth of\nthe capacity. If diversity is what the blend buys, this is the\nmember most likely to disagree with the resnets.\n"
             "\n"
             "WHY FAMILY AND NOT SEED. E064 submitted a second seed and the\n"
             "board returned +0.000. E111 changed DEPTH (resnet34 -> resnet50)\n"
             "and the board returned +0.002, twice, at two different blend\n"
             "weights. So reseeding this lineage is measured worthless and\n"
             "changing what the network IS is the part that paid. This goes\n"
             "one step further than depth: a different family.\n"
             "\n"
             "EVERYTHING ELSE IS HELD EQUAL to the resnet50 member - 24\n"
             "epochs, batch 8 x 2 accumulation, LR 6e-4, seed 3,\n"
             "input_norm=False, V1 geometry, the same 4,349 public CC0\n"
             "labels. The backbone is the only difference.\n"
             "\n"
             "batch 8 x 2 rather than 16 because E109 OOMed a T4 at batch 16\n"
             "on a 25M-parameter backbone, and convnext_tiny then failed\n"
             "twice at 192px x 60 images per study. Memory here is tight and\n"
             "the effective batch stays 16 either way.\n"
             "\n"
             "input_norm=False by MEASUREMENT, not default: E109 put ImageNet\n"
             "normalisation at -0.0064 on 882 held-out studies, CI\n"
             "[-0.0113, -0.0014], on this lineage.\n"
             "\n"
             "FULL FIT, so it cannot be scored offline - it trains on all 58\n"
             "gold. The board is the only judge. 0.940 is banked and the\n"
             "leaderboard keeps a team's best, so being wrong costs one click.\n"
             "\n"
             "ATTRIBUTION: labels from `dreaddevelopment/rsna-knee-labels`,\n"
             "CC0-1.0, repackaged as `knee-phase1-public`.",
    ),
    Kernel(
        slug="knee-train-v1pubfull-regnet",
        directory="101_train_v1pubfull_regnet",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,
                   **V1.constants(),
                   **TrainConfig(backbone="regnet_y_3_2gf", epochs=24, batch=8, accum=2,
                                 lr=6e-4, input_norm=False, seed=3).constants()},
        note="E119. A NEW v1 MEMBER FROM A DIFFERENT BACKBONE FAMILY.\n"
             "\n"
             "regnety_032: a designed-space network with squeeze-excite at 19M\nparameters. Third family, chosen to sit between the other two in\ncapacity so the three are not all making the same bet.\n"
             "\n"
             "WHY FAMILY AND NOT SEED. E064 submitted a second seed and the\n"
             "board returned +0.000. E111 changed DEPTH (resnet34 -> resnet50)\n"
             "and the board returned +0.002, twice, at two different blend\n"
             "weights. So reseeding this lineage is measured worthless and\n"
             "changing what the network IS is the part that paid. This goes\n"
             "one step further than depth: a different family.\n"
             "\n"
             "EVERYTHING ELSE IS HELD EQUAL to the resnet50 member - 24\n"
             "epochs, batch 8 x 2 accumulation, LR 6e-4, seed 3,\n"
             "input_norm=False, V1 geometry, the same 4,349 public CC0\n"
             "labels. The backbone is the only difference.\n"
             "\n"
             "batch 8 x 2 rather than 16 because E109 OOMed a T4 at batch 16\n"
             "on a 25M-parameter backbone, and convnext_tiny then failed\n"
             "twice at 192px x 60 images per study. Memory here is tight and\n"
             "the effective batch stays 16 either way.\n"
             "\n"
             "input_norm=False by MEASUREMENT, not default: E109 put ImageNet\n"
             "normalisation at -0.0064 on 882 held-out studies, CI\n"
             "[-0.0113, -0.0014], on this lineage.\n"
             "\n"
             "FULL FIT, so it cannot be scored offline - it trains on all 58\n"
             "gold. The board is the only judge. 0.940 is banked and the\n"
             "leaderboard keeps a team's best, so being wrong costs one click.\n"
             "\n"
             "ATTRIBUTION: labels from `dreaddevelopment/rsna-knee-labels`,\n"
             "CC0-1.0, repackaged as `knee-phase1-public`.",
    ),
    Kernel(
        slug="knee-train-v1pubfull-r50",
        directory="97_train_v1pubfull_r50",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": -1,
                   **V1.constants(),
                   **TrainConfig(backbone="resnet50", epochs=24, batch=8, accum=2,
                                 lr=6e-4, input_norm=False, seed=3).constants()},
        note="A SIXTH v1 MEMBER FOR THE SHIPPED BLEND, not an experiment.\n"
             "\n"
             "Full fit on all 4,407 studies, so it cannot be scored offline -\n"
             "it trains on all 58 gold. The board is the only judge, it keeps\n"
             "our best submission, and 0.938 is banked, so being wrong costs\n"
             "one click.\n"
             "\n"
             "resnet50 rather than a deeper change because it is the one new\n"
             "member buildable in the quota left. A fold-0 probe would be the\n"
             "readable experiment and could not ship: a win there still needs\n"
             "five folds before anything reaches the board.\n"
             "\n"
             "input_norm=False ON PURPOSE. E109 measured ImageNet\n"
             "normalisation at -0.0064 on 882 held-out studies, CI\n"
             "[-0.0113, -0.0014] - it hurts this lineage. The incumbent had it\n"
             "off by default; this has it off by measurement.\n"
             "\n"
             "batch 8 x 2 accumulation because resnet50 at batch 16 is the\n"
             "size that OOMed the T4 in E109. resnet50 is BatchNorm, so that\n"
             "is not exactly equivalent to batch 16 - acceptable for an\n"
             "ensemble member, and it would not have been for E109's\n"
             "controlled comparison.\n"
             "\n"
             "~1.5 h.",
    ),
    # E109. THE OLDEST CLAIM IN THE LOG, RETESTED UNDER THE LABELS IT PREDATES.
    #
    # `PATH.md` §1 records "architecture, every attempt: 0.000" and that line
    # governs how every hour of GPU here gets allocated. Every experiment behind
    # it — E012-E024, 288px, DINOv2 twice, focal top-k, per-finding pooling —
    # ran on or before **2026-08-19**, and E041/E044 replaced the labels
    # afterwards for **+0.1067 on the 58 gold**. So the claim was measured under
    # labels that E044 proved were costing 0.107 of macro AUC, which is three
    # times E060's own +/-0.03 noise floor. **An architecture effect could not
    # have been seen through that.**
    #
    # AND E020 DISOWNS ITSELF. Its own entry: *"0.6878 is not a measurement of
    # this backbone; it is where the clock stopped"* — the DINOv2 curve was
    # still gaining 0.002 an epoch when the budget ended it at 16. Half the
    # evidence for the project's largest standing closure is an experiment the
    # log says was not a comparison.
    #
    # ONE FOLD, ONE VARIABLE. Byte-identical to `v1public` fold 0 — same cache,
    # same 192px/0.6mm geometry, same public labels, same 24 epochs, batch and
    # LR — with `backbone` the only difference. convnext_tiny rather than
    # something larger, so this tests the architecture FAMILY and not capacity:
    # 28M parameters against resnet34's 21M.
    #
    # THE INSTRUMENT IS THE POINT, and it is why this is worth running when a
    # 58-study comparison would not be. Fold 0 holds out **882 studies**, every
    # one of them report-labelled, and `knee-infer-v1pub`'s existing dump has
    # resnet34's honest out-of-fold predictions for exactly those. The
    # comparison is 882 paired studies, not the n~12 gold subset a single fold
    # carries.
    #
    # THE ARBITER'S STATUS, STATED BEFORE THE RUN. E106 invalidated report
    # labels for ranking OUR model against a FOREIGN one; E108 validated them
    # within a checkpoint family at Spearman +0.800. Two of our own models,
    # same labels, same data, same fold, differing only in backbone, is an
    # UNTESTED MIDDLE CASE — the "rewarded for agreeing with what it was trained
    # on" bias is common-mode here, which argues it cancels, but that argument
    # has not been measured for this class. Read the result accordingly.
    Kernel(
        slug="knee-train-v1pub-cnx",
        directory="93_train_v1pub_cnx_fold0",
        template="train",
        gpu=True,
        internet=True,          # timm downloads the pretrained backbone
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": 0,
                   **V1.constants(),
                   # batch 4 x 4 accumulation = the SAME effective batch of 16
                   # the control uses. convnext_tiny OOMed the T4 at batch 16
                   # (each study is 3 planes x 20 slices = 60 images, so a batch
                   # of 16 is 960 forward passes of activations). Accumulation is
                   # EXACTLY equivalent here rather than approximately: convnext
                   # normalises with LayerNorm, which is per-sample, so unlike a
                   # BatchNorm backbone the split carries no batch-statistics
                   # difference. The comparison stays one-variable.
                   **TrainConfig(backbone="convnext_tiny", epochs=24, batch=4,
                                 accum=4, lr=6e-4, input_norm=True,
                                 seed=3).constants()},
        note="Retests the oldest standing claim in this project: that\n"
             "architecture has measured zero, every time.\n"
             "\n"
             "Every experiment behind that claim ran on or before 2026-08-19,\n"
             "and the labels changed afterwards for +0.1067 on gold (E044).\n"
             "The claim was measured under labels costing 0.107 of macro AUC,\n"
             "which is 3x E060's own noise floor - an architecture effect\n"
             "could not have been seen through that. And E020's own entry\n"
             "says its DINOv2 number was not a measurement of the backbone\n"
             "but of where the epoch budget stopped.\n"
             "\n"
             "ONE VARIABLE: byte-identical to `knee-train-v1pub` fold 0 except\n"
             "the backbone. input_norm=True because convnext expects ImageNet\n"
             "normalisation and resnet34 here was trained without it - that is\n"
             "a second difference and it is forced, not chosen; it is recorded\n"
             "in E109 rather than hidden.\n"
             "\n"
             "Read it on the 882 held-out studies of fold 0, NOT on the ~12\n"
             "gold ones a single fold carries. `knee-oof-v1pub-cnx` builds\n"
             "that dump on CPU for zero GPU quota.\n"
             "\n"
             "~2-3 h. If it separates, five folds is the follow-up; if it does\n"
             "not, PATH.md's line stands and is no longer stale.",
    ),
    # THE CONTROL ARM, and it exists because without it this repeats E020's own
    # unresolved flaw. convnext needs ImageNet normalisation; the resnet34
    # baseline was trained without it. Comparing them directly confounds the
    # backbone with the input scaling — which is precisely what E020 flagged in
    # 2026-08-19 (*"ImageNet normalisation on, which the resnet34 runs did not
    # have"*) and then never separated. Repeating that would leave the stale
    # claim replaced by an equally unreadable one.
    #
    # So resnet34 is re-run at fold 0 WITH normalisation. Three arms then bound
    # both variables: the existing `knee-train-v1pub` (resnet34, no norm), this
    # (resnet34, norm), and the convnext (norm). Backbone is read from the last
    # two; normalisation from the first two. Cost is one extra fold, ~1.4 h.
    Kernel(
        slug="knee-train-v1pub-norm",
        directory="95_train_v1pub_norm_fold0",
        template="train",
        gpu=True,
        internet=True,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3"],
        datasets=[PUBLIC_DATASET],
        constants={"RUN_FOLD": 0,
                   **V1.constants(),
                   **TrainConfig(backbone="resnet34", epochs=24, batch=16,
                                 lr=6e-4, input_norm=True, seed=3).constants()},
        note="THE CONTROL FOR `knee-train-v1pub-cnx`, and it must run.\n"
             "\n"
             "convnext needs ImageNet normalisation and the resnet34 baseline\n"
             "was trained without it, so comparing them directly confounds\n"
             "the backbone with the input scaling. E020 hit exactly this in\n"
             "August - `ImageNet normalisation on, which the resnet34 runs did\n"
             "not have` - and never separated it, which is half of why the\n"
             "architecture claim is unreadable today.\n"
             "\n"
             "Three arms bound both variables: `knee-train-v1pub` (resnet34,\n"
             "no norm), this one (resnet34, norm), and the convnext (norm).\n"
             "Backbone is read from the last two; normalisation from the\n"
             "first two.\n"
             "\n"
             "~1.4 h. Skipping it would replace a stale claim with an\n"
             "equally unreadable one.",
    ),
    Kernel(
        slug="knee-oof-v1pub-norm",
        directory="96_oof_v1pub_norm",
        template="gold_eval",
        gpu=False,
        internet=False,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3", "knee-train-v1pub-norm"],
        datasets=[PUBLIC_DATASET],
        constants={**V1.constants(), "TTA_VIEWS": ("identity",),
                   "OOF_SCOPE": "all"},
        note="Out-of-fold predictions from the resnet34+normalisation fold-0\n"
             "control, for the 882 studies it held out. CPU, zero GPU quota.",
    ),
    Kernel(
        slug="knee-oof-v1pub-cnx",
        directory="94_oof_v1pub_cnx",
        template="gold_eval",
        gpu=False,          # 882 forward passes on CPU; zero GPU quota
        internet=False,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3", "knee-train-v1pub-cnx"],
        datasets=[PUBLIC_DATASET],
        constants={**V1.constants(), "TTA_VIEWS": ("identity",),
                   "OOF_SCOPE": "all"},
        note="Out-of-fold predictions from the convnext fold-0 model, for the\n"
             "882 studies it held out. The comparison arm is\n"
             "`knee-oof-v1pub`'s existing dump, which carries resnet34's\n"
             "honest predictions for the same 882.\n"
             "\n"
             "CPU, so it costs no GPU quota - the same reason\n"
             "`knee-oof-v1pub` runs this way.",
    ),
    # E108's READOUT: the three narrowed arms on all 4,407, to sit beside E106's
    # dump of the three published ones and be compared on the 4,349 that are not
    # the 58.
    Kernel(
        slug="knee-trainall-span04",
        directory="92_trainall_span04",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        constants={
            "MEMBERS_EXPECTED": 3,
            "ARMS": RAPTOR_SPAN04_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "trainall",
            "GOLD_EXPECTED": 58,
            "V1_MEMBERS": None,
        },
        note="NOT A SUBMISSION. The three CoAtNet checkpoints run one span\n"
             "step narrower - 0.04 at each end, upstream's own spacing - over\n"
             "all 4,407 studies.\n"
             "\n"
             "WHY IT IS BACK. E103 found six of six narrowed variants beat\n"
             "their parent on the 58 gold and closed the route anyway,\n"
             "because gold-58 had produced that hypothesis and would have\n"
             "been the only thing testing it. E108 measured the report-label\n"
             "proxy ranking the four published arms at Spearman +0.800\n"
             "against gold, where E106 got +0.052 across architectures: the\n"
             "proxy's bias is agreement with what a model was TRAINED on, and\n"
             "that is common-mode between two geometries of one checkpoint.\n"
             "\n"
             "So the 4,349 non-gold studies decide, and gold-58 no longer has\n"
             "to judge its own hypothesis.\n"
             "\n"
             "ONE STEP FOR ALL THREE. Gold's per-arm optima disagree (span04\n"
             "for v5 and v10, span08 for v8) and picking each checkpoint's\n"
             "best would be three parameters fitted on 58 studies.\n"
             "\n"
             "~6.6 h at E106's measured 1.01 s/study over three build groups.\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorcc0`.",
    ),
    # E106. THE CoAtNet ARMS ON ALL 4,407 STUDIES, so a blend rule can be FITTED
    # somewhere other than the 58 studies it will be tested on.
    #
    # E105's oracle bound said the available headroom in the two shipped arms is
    # +0.0076 gold, and that it is concentrated in three findings where the
    # uniform 0.5 weight DESTROYS AUC: MCL 0.982 -> 0.950, Medial Meniscus
    # 0.968 -> 0.948, ACL 0.977 -> 0.973. CoAtNet is far ahead there and
    # averaging a weaker member in costs real ordering.
    #
    # Fitting twelve per-finding weights on 58 studies is what this log has
    # declined six times and it stays declined. What makes this different is that
    # the weights are derived on 4,349 studies the 58 are NOT in, so gold-58
    # becomes a HELD-OUT TEST of the rule rather than its source. This project
    # has never had a train/validate split for a blend weight.
    #
    # The arbiter offline is the public report labels (`stevenleehans`, CC0),
    # which E041 measured at 0.8927 macro against the 58 expert studies. E093
    # called that proxy's validity unverified and it still is -- but it carries
    # 75x the sample and, unlike gold-58, using it leaves an honest test set.
    #
    # The v1 side already exists: `knee-infer-v1pub`'s five folds dumped honest
    # out-of-fold predictions covering all 4,407 (verified: 882+882+881+881+881).
    # Only the CoAtNet side is missing, and this is it.
    #
    # COST: E099 measured 4.65 s/study for these four arms, so 4,407 studies is
    # ~5.7 h against the 9 h cap. No submission. The dump is reusable: every
    # future blend question about this arm becomes arithmetic instead of a run.
    Kernel(
        slug="knee-trainall-raptor",
        directory="91_trainall_raptor",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        constants={
            "MEMBERS_EXPECTED": 4,
            "ARMS": RAPTOR_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "trainall",
            "GOLD_EXPECTED": 58,
            "V1_MEMBERS": None,
        },
        note="NOT A SUBMISSION, and it writes none. Runs the four CC0 CoAtNet\n"
             "arms over all 4,407 TRAINING studies and dumps raw per-study\n"
             "probabilities to trainall_probs.parquet.\n"
             "\n"
             "WHY: E105 bounded the headroom left in the two shipped arms at\n"
             "+0.0076 gold and found it concentrated where the uniform 0.5\n"
             "weight destroys AUC - MCL 0.982 -> 0.950, Medial Meniscus\n"
             "0.968 -> 0.948. A per-finding weight would recover it, and\n"
             "fitting twelve weights on 58 studies is refused here.\n"
             "\n"
             "So the weights get fitted on the 4,349 NON-GOLD studies against\n"
             "the public report labels, and the 58 gold become a held-out\n"
             "test of the rule. That split is the entire point: this project\n"
             "has never validated a blend weight on data that did not\n"
             "produce it.\n"
             "\n"
             "~5.7 h at E099's measured 4.65 s/study, against a 9 h cap.\n"
             "Watch the projection line at study 100; it should read ~1.7 h\n"
             "per 1,300.\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorcc0`.",
    ),
    # E104's READOUT. Both label arms scored on the 58, in ONE run, separately.
    #
    # The gold dump emits one block per v1 member rather than their average,
    # which is the whole reason this is one kernel and not two: averaging the two
    # arms would answer nothing, and running them separately would pay for the
    # CoAtNet pass twice.
    #
    # The four CoAtNet arms ride along as the control. They have read
    # 0.9198 / 0.9170 / 0.9167 / 0.9116 on four consecutive runs; a fifth
    # confirms this run is measuring the same instrument, and their column is
    # what the winning label arm gets blended against afterwards.
    Kernel(
        slug="knee-gold-labarms",
        directory="90_gold_labarms",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        depends=["knee-train-lab-dread", "knee-train-lab-pub4349"],
        constants={
            "MEMBERS_EXPECTED": 4,
            "ARMS": RAPTOR_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "gold",
            "GOLD_EXPECTED": 58,
            "V1_MEMBERS": 2,
            "V1_BLEND_W": 0.40,
            "V1_BATCH_STUDIES": V1.infer_batch,
            "V1_SLICE_SUBSAMPLE": None,
            "V1_INPUT_NORM": False,
            "CHECKPOINT_GLOB": "checkpoint_fold*.pt",
            "SKIP_DIRECTORIES": Raw('{"train_series", "test_series"}'),
            **V1.constants(),
            "PLANES": ("Sagittal", "Coronal", "Axial"),
        },
        note="NOT A SUBMISSION. Scores E104's two label arms on the 58 gold\n"
             "studies, separately, with the four CoAtNet arms as the control.\n"
             "\n"
             "THIS IS AN HONEST NUMBER FOR A FULL-FIT MODEL, which this\n"
             "project has never had. Both arms trained on 4,349 studies and\n"
             "both logs print `gold studies in cache: 0` - the label sets\n"
             "contain no gold rows, so the 58 were never in training.\n"
             "\n"
             "Read the two `[gold] v1 member` lines. The pre-registered rule\n"
             "is E104's: a gap under 0.015 is NOT SEPARATED, because E031\n"
             "puts gold-58's absolute interval at +/-0.0153. A point estimate\n"
             "is not a tie-break.\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorcc0`; labels from\n"
             "`dreaddevelopment/rsna-knee-labels` and\n"
             "`stevenleehans/rsna-knee-llm-report-labels`, both CC0-1.0.",
    ),
    # TWELVE ARMS FROM THREE FILES, TO PRICE TTA THE WAY E101 PRICED PARTNERS.
    #
    # E099's instrument turned "what blend?" from a board submission into
    # arithmetic, and E101 spent it on eight candidate PARTNERS. This spends it
    # on the arms themselves: the three mounted checkpoints can produce more
    # members than upstream published, and nobody has checked whether the extra
    # ones are diverse enough to pay.
    #
    # The four published arms are included unchanged, so the sweep carries its
    # own control and must reproduce 0.9198 / 0.9170 / 0.9167 / 0.9116 for a
    # fourth consecutive run before anything else in it is readable.
    #
    # NOTHING IS SELECTED ON GOLD. The acceptance rule is fixed in E103 before
    # the run: each axis is taken WHOLE or not at all, at flat weight within its
    # checkpoint family. Picking the variants that happened to score well on 58
    # studies is fitting 12 free parameters to 58 studies, which this project has
    # declined five times.
    Kernel(
        slug="knee-gold-raptortta",
        directory="87_gold_raptortta",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        constants={
            "MEMBERS_EXPECTED": 12,
            "ARMS": RAPTOR_TTA_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "gold",
            "GOLD_EXPECTED": 58,
            "V1_MEMBERS": None,
        },
        note="NOT A SUBMISSION. Twelve arms from the same three CC0\n"
             "checkpoints, scored on the 58 expert studies, dumping raw\n"
             "per-study probabilities so TTA can be priced offline.\n"
             "\n"
             "AXIS 1: upstream publishes ONE transform arm - maxspan-v5 with\n"
             "its slice triplet reversed - and never applies it to the other\n"
             "two files. This does. Two extra members for two extra forward\n"
             "passes on volumes already built.\n"
             "\n"
             "AXIS 2: span jitter. `span` picks which source slices fill the\n"
             "stack, so shifting it resamples the volume. The step is 0.04,\n"
             "which is the gap between upstream's OWN spans (v5 and v10 at\n"
             "0.02-0.98, v8 at 0.06-0.94) - taken from their choices, not\n"
             "from a score.\n"
             "\n"
             "NO HORIZONTAL FLIP, on purpose. Four of the twelve findings\n"
             "are Medial/Lateral Meniscus and Medial/Lateral OA, and medial\n"
             "versus lateral is which SIDE of the knee a structure sits on.\n"
             "Mirroring a left knee makes it a right knee and swaps the two,\n"
             "so those four would be read off the wrong compartment.\n"
             "\n"
             "Weights are flat because the blend is searched offline; the\n"
             "kernel's own blend line is a flat average, not the result.\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorcc0`.",
    ),
    # THE FIRST UNION THIS PROJECT EVER PRICED OFFLINE BEFORE SUBMITTING IT.
    #
    # E099 built the gold instrument; E101 swept eight candidate partners for the
    # CoAtNet arms on the 58 expert studies and exactly one has an interior
    # optimum rather than a monotonic decline away from w=0 -- this project's own
    # resnet34 2.5D at 192 px.
    #
    #   w      0.00    0.10    0.20    0.30    0.50
    #   macro  0.9225  0.9243  0.9267  0.9277  0.9255
    #
    # Every foreign system E098 could reach declines from the first step:
    # pilkwang's three at w=0.1 already read 0.9211, 0.9217, 0.9223.
    #
    # WHY THIS PAIR AND NOT THOSE. E048's rule wants members comparable in
    # strength and different in kind, and this is the only pair that is both:
    #   gap        0.9223 vs 0.8980 = 0.0243, against 0.079-0.130 for the foreign
    #              systems.
    #   kind       resnet34 2.5D at 192 px on this project's report labels
    #              against CoAtNet at 336/384 px on upstream's. Cross-architecture
    #              rank correlation 0.793, where CoAtNet's own four arms sit at
    #              0.905-0.986.
    #   mechanism  8 of 12 findings improve, and the gain lands where CoAtNet is
    #              WEAKEST -- Synovitis +0.016 (this project's floor since E059),
    #              Baker's +0.022, Effusion +0.012 -- while all four losses are
    #              findings where CoAtNet already reads 0.92-0.98 and the second
    #              member is far behind. Decorrelation doing work, not an average
    #              of noise.
    #
    # WHY IT IS ONE KERNEL AND NOT A BLEND OF TWO. `rank_blend`'s own guard says
    # it: a mounted kernel supplies its LAST SAVED output, frozen at the 3-study
    # visible run, so a CSV-chained blend submits three rows and spends a slot.
    # Of 626 public notebooks, 152 mount another for its CHECKPOINTS and 3 read a
    # submission.csv. The v1 path here is SPLICED FROM THE SAME `_shared`
    # fragments every v1 kernel has always used, not reimplemented -- a second
    # copy of the preprocessing is the exact skew E088 caught in kernel 81's
    # first draft.
    #
    # WHAT IT IS NOT EVIDENCE OF. +0.0053 on gold-58 with a paired CI of
    # [-0.0016, +0.0118] is NOT separated, and the board floor is +/-0.003 (E092),
    # so this is marginal by construction. E042 found the same shape at the same
    # w=0.3 against a different partner and never got to submit it.
    Kernel(
        slug="knee-gold-v1integrity",
        directory="102_gold_v1integrity",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        depends=["knee-train-v1pubfull", "knee-train-v1pubfull-s4",
                 "knee-train-v1pubfull-s5", "knee-train-v1pubfull-s6",
                 "knee-train-v1pubfull-s7", "knee-train-v1pubfull-r50",
                 "knee-train-v1pubfull-rx50", "knee-train-v1pubfull-regnet"],
        constants={
            "MEMBERS_EXPECTED": 4,
            "ARMS": RAPTOR_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "gold",
            "GOLD_EXPECTED": 58,
            "V1_MEMBERS": 8,
            "V1_BLEND_W": 0.50,
            "V1_BATCH_STUDIES": V1.infer_batch,
            "V1_SLICE_SUBSAMPLE": None,
            "V1_INPUT_NORM": False,
            "CHECKPOINT_GLOB": "checkpoint_fold*.pt",
            "SKIP_DIRECTORIES": Raw('{"train_series", "test_series"}'),
            **V1.constants(),
            "PLANES": ("Sagittal", "Coronal", "Axial"),
        },
        note="NOT A SUBMISSION. E120's INTEGRITY CHECK, and it needs no\n"
             "validation set because it exploits MEMORISATION.\n"
             "\n"
             "Every v1 member is a FULL FIT trained on all 4,407 studies,\n"
             "including all 58 gold. Each one has therefore SEEN every study\n"
             "it is scored on here. Run through a correct inference path each\n"
             "must come back at ~0.99 macro AUC on the 58. That is normally\n"
             "the reason gold is unusable for a full fit; here it is exactly\n"
             "what makes the test sharp.\n"
             "\n"
             "ANY MEMBER WELL SHORT OF ~0.99 HAS A LOAD OR MAPPING FAULT,\n"
             "not a quality problem. A model cannot fail to recognise data it\n"
             "memorised unless something between the checkpoint and the\n"
             "forward pass is wrong - a partial state_dict, a backbone\n"
             "rebuilt with different feature widths, a normalisation flag\n"
             "that differs from training, or rows joined by position.\n"
             "\n"
             "WHY THIS RUN EXISTS. E119 shipped eight members and the board\n"
             "returned 0.921/0.923 against six members' 0.940 - BELOW BOTH\n"
             "arms alone (v1 0.926, CoAtNet 0.932). A rank-mean under its own\n"
             "weakest component is breakage, not a weak union, and nothing\n"
             "confined to two members at 1/8 weight can move a blend that far.\n"
             "The stub could not see it: at a 40% per-study failure rate,\n"
             "`fallbacks 0/3` still appears 22% of the time, and eight\n"
             "distinct fingerprints prove eight FILES were read, not that\n"
             "each file's weights reached its network.\n"
             "\n"
             "WHAT TO READ: gold_probs.csv carries one row per member per\n"
             "study. Score each member separately. The five resnet34s and the\n"
             "resnet50 are the CONTROL - they shipped in the 0.940 blend, so\n"
             "if they do not memorise either, the fault is in shared code and\n"
             "not in the two new members at all.\n"
             "\n"
             "Writes no submission.csv: these 58 are training data.\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorv1`.",
    ),
    Kernel(
        slug="knee-gold-raptorv1",
        directory="85_gold_raptorv1",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        depends=["knee-train-v1pubfull", "knee-train-v1pubfull-s4",
                 "knee-train-v1pubfull-s5", "knee-train-v1pubfull-s6",
                 "knee-train-v1pubfull-s7",
                 # E111's resnet50, restored. E064 priced a reseed of this
                 # lineage at +0.000 and this depth change at +0.002, twice, at
                 # two different blend weights -- it is the one added member
                 # whose gain has a mechanism behind it rather than a draw.
                 "knee-train-v1pubfull-r50",
                 # E119's two survivors are REVERTED. Eight members scored
                 # 0.921/0.923 against six members' 0.940 -- below BOTH arms
                 # alone (v1 0.926, CoAtNet 0.932), which is breakage rather
                 # than weak members, and far outside the +/-0.003 floor. The
                 # checkpoints and trainers are kept; only the mount is removed.
                 ],
        constants={
            "MEMBERS_EXPECTED": 4,
            "ARMS": RAPTOR_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "gold",
            "GOLD_EXPECTED": 58,
            # THE SECOND ARCHITECTURE, ON. Five full-fit resnet34 members at
            # this project's own v1 geometry, run in the same kernel because a
            # CSV-chained blend cannot work here (see `rank_blend`'s own guard:
            # a mounted kernel supplies its last SAVED output, frozen at the
            # 3-study visible run).
            #
            # BACK TO FIVE, per E111's own pre-registered rule. The resnet50
            # sixth member moved the board 0.938 -> 0.940, which is inside the
            # +/-0.003 reseed floor E092 established, so it has not been shown to
            # do anything. The 0.940 stays banked whatever this says -- the board
            # keeps a team's best -- so the revert costs nothing and keeping it
            # would be banking a number the instrument cannot resolve.
            "V1_MEMBERS": 6,
            # BACK TO THE SCALAR, per E116's pre-registration. E116 shipped one
            # weight per finding, derived without touching gold-58 and without
            # touching CoAtNet's predictions -- leakage-free by construction,
            # worth +0.0023 on held-out gold with P(better) 0.897, and recovering
            # 72% of the loss E105's oracle had measured on MCL specifically.
            #
            # THE BOARD RETURNED 0.940, THE SAME NUMBER THE SCALAR 0.50 SCORED.
            # Not worse -- identical. So on 1,300 studies the per-finding
            # structure is worth nothing measurable, and the pre-registered rule
            # for that bracket was revert, which needs no justification.
            #
            # `one_sided_weights` in `eda/per_finding_weights.py` keeps the rule
            # and its input table, so reinstating the vector is one line if a
            # future instrument can resolve +0.002. The reason it is not shipped
            # is a measurement, not a doubt about the derivation.
            "V1_BLEND_W": 0.50,
            "V1_BATCH_STUDIES": V1.infer_batch,
            "V1_SLICE_SUBSAMPLE": None,
            "V1_INPUT_NORM": False,
            "CHECKPOINT_GLOB": "checkpoint_fold*.pt",
            "SKIP_DIRECTORIES": Raw('{"train_series", "test_series"}'),
            **V1.constants(),
            "PLANES": ("Sagittal", "Coronal", "Axial"),
        },
        note="THE PLUMBING CHECK, and it must run before the submission.\n"
             "\n"
             "Same kernel as `knee-infer-raptorv1`, pointed at the 58 expert\n"
             "studies. It writes no submission and costs no submission.\n"
             "\n"
             "WHAT IT CAN AND CANNOT SHOW. The v1 members here are FULL-FIT:\n"
             "every one trained on all 58 gold studies, so their gold number\n"
             "is contaminated upward and is NOT a score. What it checks is\n"
             "that the second architecture is wired correctly at all - a\n"
             "skewed v1 path reads near 0.5, a working one reads high. That\n"
             "is the failure mode worth a free run, because silent\n"
             "train/inference skew is what E088 caught in kernel 81's first\n"
             "draft and it runs, writes a submission, and is wrong.\n"
             "\n"
             "The four CoAtNet arms should reproduce E099 exactly:\n"
             "0.9198 / 0.9170 / 0.9167 / 0.9116, blend 0.9223. If they move,\n"
             "splicing the second architecture in changed the first one.\n"
             "\n"
             "ATTRIBUTION: as `knee-infer-raptorcc0`.",
    ),
    Kernel(
        slug="knee-infer-raptorv1",
        directory="86_infer_raptorv1",
        template="raptor_infer",
        gpu=True,
        internet=False,
        datasets=["dreaddevelopment/raptor-knee-maxspan",
                  "dreaddevelopment/raptor-knee-native384",
                  "dreaddevelopment/raptor-knee-native384dense"],
        depends=["knee-train-v1pubfull", "knee-train-v1pubfull-s4",
                 "knee-train-v1pubfull-s5", "knee-train-v1pubfull-s6",
                 "knee-train-v1pubfull-s7",
                 # E111's resnet50, restored. E064 priced a reseed of this
                 # lineage at +0.000 and this depth change at +0.002, twice, at
                 # two different blend weights -- it is the one added member
                 # whose gain has a mechanism behind it rather than a draw.
                 "knee-train-v1pubfull-r50",
                 # E119's two survivors are REVERTED. Eight members scored
                 # 0.921/0.923 against six members' 0.940 -- below BOTH arms
                 # alone (v1 0.926, CoAtNet 0.932), which is breakage rather
                 # than weak members, and far outside the +/-0.003 floor. The
                 # checkpoints and trainers are kept; only the mount is removed.
                 ],
        constants={
            "MEMBERS_EXPECTED": 4,
            "ARMS": RAPTOR_ARMS,
            "CROP_MM": 140.0,
            "LAB": RAPTOR_LAB,
            "FALLBACK_LIMIT": 0.02,
            "DECODE_AHEAD": 32,
            "EVAL_SPLIT": "test",
            "GOLD_EXPECTED": 58,
            # THE SECOND ARCHITECTURE, ON. Five full-fit resnet34 members at
            # this project's own v1 geometry, run in the same kernel because a
            # CSV-chained blend cannot work here (see `rank_blend`'s own guard:
            # a mounted kernel supplies its last SAVED output, frozen at the
            # 3-study visible run).
            #
            # BACK TO FIVE, per E111's own pre-registered rule. The resnet50
            # sixth member moved the board 0.938 -> 0.940, which is inside the
            # +/-0.003 reseed floor E092 established, so it has not been shown to
            # do anything. The 0.940 stays banked whatever this says -- the board
            # keeps a team's best -- so the revert costs nothing and keeping it
            # would be banking a number the instrument cannot resolve.
            "V1_MEMBERS": 6,
            # BACK TO THE SCALAR, per E116's pre-registration. E116 shipped one
            # weight per finding, derived without touching gold-58 and without
            # touching CoAtNet's predictions -- leakage-free by construction,
            # worth +0.0023 on held-out gold with P(better) 0.897, and recovering
            # 72% of the loss E105's oracle had measured on MCL specifically.
            #
            # THE BOARD RETURNED 0.940, THE SAME NUMBER THE SCALAR 0.50 SCORED.
            # Not worse -- identical. So on 1,300 studies the per-finding
            # structure is worth nothing measurable, and the pre-registered rule
            # for that bracket was revert, which needs no justification.
            #
            # `one_sided_weights` in `eda/per_finding_weights.py` keeps the rule
            # and its input table, so reinstating the vector is one line if a
            # future instrument can resolve +0.002. The reason it is not shipped
            # is a measurement, not a doubt about the derivation.
            "V1_BLEND_W": 0.50,
            "V1_BATCH_STUDIES": V1.infer_batch,
            "V1_SLICE_SUBSAMPLE": None,
            "V1_INPUT_NORM": False,
            "CHECKPOINT_GLOB": "checkpoint_fold*.pt",
            "SKIP_DIRECTORIES": Raw('{"train_series", "test_series"}'),
            **V1.constants(),
            "PLANES": ("Sagittal", "Coronal", "Axial"),
        },
        note="THE SUBMISSION. Four CC0 CoAtNet arms (board 0.932) and five\n"
             "full-fit resnet34 members (board 0.926), rank-blended 50/50\n"
             "inside one kernel.\n"
             "\n"
             "DO NOT SUBMIT BEFORE `knee-gold-raptorv1` HAS RUN. Two\n"
             "architectures in one kernel is two ways for a preprocessing\n"
             "skew to hide, and the gold run is what makes this readable.\n"
             "\n"
             "PRE-REGISTERED, per E101:\n"
             "  > 0.935   the offline sweep transferred and then some\n"
             "  0.933-0.935  a real gain above the +/-0.003 board floor\n"
             "  0.930-0.932  inside the floor; gold-58 saw a gain the board\n"
             "               cannot, which is the E083 failure again\n"
             "  < 0.929   below the CoAtNet arm alone, so the second member\n"
             "            DILUTES, and E048's rule is wrong at a 0.024 gap\n"
             "\n"
             "The weight is 0.5 and nothing is fitted, for the fifth time.\n"
             "The sweep's peak at w=0.3 beats 0.5 by +0.0022, which is below\n"
             "the board's own reseed floor: a fitted weight would buy a\n"
             "difference the board cannot measure.\n"
             "\n"
             "Cost: 82's four arms (2.43 h on 1,300) plus 63's five members\n"
             "(~1.0 h), so ~3.5 h against a 9 h cap.\n"
             "\n"
             "ATTRIBUTION: the CoAtNet arms are Dread Development's, run\n"
             "from the CC0 datasets `raptor-knee-maxspan`,\n"
             "`raptor-knee-native384` and `raptor-knee-native384dense`, with\n"
             "blend weights from the public write-up\n"
             "`4-arm-ensemble-explained-rsna-knee-0-937`.",
    ),
    Kernel(
        slug="knee-blend-raptor",
        directory="74_blend_raptor",
        template="rank_blend",
        gpu=False,          # two CSVs and a rank average; there is nothing to accelerate
        internet=False,     # a submission kernel
        # THE ONE UNION LEFT THAT MEETS BOTH HALVES OF E048's RULE.
        #
        # `knee-infer-raptor` is the public CC0 Raptor widedense model, which
        # upstream reports at 0.924 on the board from ONE model with no
        # ensembling and no TTA, and at 0.9167 on the same 58 expert studies
        # (gold held out of its training). `knee-infer-v1pubfull5` is this
        # project's five full-fit members at 0.926.
        #
        # 0.002 apart is the tightest comparability in the log — E023's pair sat
        # 0.0025 apart and paid +0.070 — and they are maximally different in
        # kind: 64 slices in five fixed plane slots at 336px on a 140 mm
        # physical crop against our 2.5D resnet34 at 192px on our own report
        # labels. E075's correction is the reason that matters: E064's ten
        # members failed because the extra five were a RESEED, and same-kind
        # members make correlated errors. Nothing about these two is same-kind.
        #
        # WHY NOT THE STRONGER PUBLIC SYSTEMS. The 0.936 cluster (492 teams) and
        # 0.937 cluster (163) both mount `tonylica/rsna-knee-bend-dinov3-0917-
        # repro-assets`, licensed `other`. The four-arm ensembles add
        # `prvsiyan/...radimagenet-heads` (`other`) and `marwanmath` /
        # `antoinegg1` RadImageNet heads (CC-BY-NC-SA). E043's licence rule
        # excludes all of them, as it already excluded
        # `mattiaangeli/rsna-knee-cnx-m448-f0-public`. Raptor is the strongest
        # arm that clears the gate, not the strongest arm on the board.
        #
        # PRE-REGISTERED READING, so it is not chosen after the score lands:
        #   > 0.926   the union pays and borrowing public work is a live lever
        #   0.924-0.926  dilution between two members; both stand alone and the
        #                blend adds nothing, which closes different-kind
        #                ensembling the way E064 closed the same-kind case
        #   < 0.924   below BOTH members, which means the blend is broken
        #             rather than useless — read the agreement numbers it prints
        #
        # WHAT STANDS IN FOR A CONTROL ARM, and its limit. Upstream's 0.924 is
        # self-reported rather than measured here, and this project does not
        # own a submission of the Raptor arm alone — mounting a public
        # notebook's output takes no copy, but it also produces no standalone
        # score. So the Raptor member is trusted on upstream's word plus the
        # 900+ teams whose board positions corroborate the family, and NOT on
        # anything measured in this repo.
        #
        # That is a real weakness and it shapes the reading below: a blend that
        # lands under BOTH members is the signature of a bad member, not of a
        # bad blend, and the per-finding agreement numbers the kernel prints
        # are what distinguish them. If the Raptor arm ever needs a score of
        # its own, that requires forking the upstream notebook under this
        # account, which is the account owner's decision to make and not
        # something this file should quietly assume.
        # BLOCKED, and the blocker is not technical. Upstream's notebook
        # persists NO output files — the API reports `files: []`, only a log —
        # so mounting its submission.csv is impossible, and E085 confirmed it by
        # running: the kernel mounted one member and refused, exactly as
        # MEMBERS_EXPECTED is there to make it do.
        #
        # The Raptor arm therefore has to be RUN, which means a fork of the
        # upstream notebook living under this account. That is ordinary Kaggle
        # practice and the rules permit it with attribution — 655 teams sit at
        # 0.936/0.937 having done exactly that — but it republishes someone
        # else's code under the account owner's name, and that is the owner's
        # call to make rather than something this file should assume. See
        # HANDOFF.md §4d for the exact steps if the answer is yes.
        #
        # RETARGETED 2026-09-08 (E088). This used to mount upstream's notebook
        # directly, which cannot work: it persists no output files, so there is
        # no submission.csv to blend and E085's run correctly refused with one
        # member. `knee-infer-raptorcc0` is that same model run in OUR kernel
        # from the CC0 weights, so the arm is now a real dependency inside this
        # manifest rather than a foreign kernel that can never supply one.
        #
        # The attribution below does not change and must not: the model is still
        # Dread Development's. What changed is only where it runs.
        #
        # DO NOT PUSH until `knee-infer-raptorcc0` has SCORED on the board. Its
        # own note says why — an unpriced member makes the blend unreadable
        # rather than merely unproven.
        depends=["knee-infer-v1pubfull5", "knee-infer-raptorcc0"],
        constants={"MEMBERS_EXPECTED": 2},
        note="A 50/50 rank blend of the public CC0 Raptor model and this\n"
             "project's full-fit ensemble. No model, no GPU, no training —\n"
             "two finished submission.csv files and a rank average.\n"
             "\n"
             "ATTRIBUTION, required by the competition rules and by E043:\n"
             "the Raptor arm is Dread Development's, reproduced from the\n"
             "public notebook `knee-mri-twelve-findings-from-a-single-model`\n"
             "with weights from `dreaddevelopment/raptor-knee-widedense`,\n"
             "licensed CC0-1.0. The rules permit publicly shared code and\n"
             "data as inputs for every participant, with credit.\n"
             "\n"
             "Every member here has already been priced by the board, so no\n"
             "offline instrument is in the loop and there is nothing for a\n"
             "leak to get into. That is deliberate: E083 is three days old and\n"
             "cost a board point to the gold rig separating something at\n"
             "+0.0221 that the board scored at −0.013.\n"
             "\n"
             "The weight is 0.5 and nothing is fitted. E048, E069 and E081\n"
             "each declined a weight chosen on 58 studies; this declines it a\n"
             "fourth time. The blend ranks each finding before averaging\n"
             "because the metric reads ordering only, and the two members are\n"
             "calibrated differently enough that raw averaging would let the\n"
             "wider-spread member decide the order.",
    ),
    Kernel(
        slug="knee-oof-v1pub",
        directory="64_oof_v1pub",
        template="gold_eval",
        gpu=False,          # 4,407 forward passes; CPU is a separate allowance
        internet=False,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3",
                 "knee-train-v1pub", "knee-train-v1pub-fold1",
                 "knee-train-v1pub-fold2", "knee-train-v1pub-fold3",
                 "knee-train-v1pub-fold4"],
        datasets=[PUBLIC_DATASET],
        constants={**V1.constants(), "TTA_VIEWS": ("identity",),
                   "OOF_SCOPE": "all"},
        note="Out-of-fold predictions for every study in the corpus, from the\n"
             "five checkpoints behind 0.923. Each study is predicted exactly\n"
             "once, by the one model that held it out.\n"
             "\n"
             "This is the raw material for a distillation teacher, and the\n"
             "reason to want one is measured rather than assumed: the model\n"
             "scores 0.8980 on the 58 gold studies and the labels that trained\n"
             "it score 0.8927. The student has overtaken its teacher, and by\n"
             "0.005 — which puts the two COMPARABLE, the condition E048\n"
             "identified as the difference between E023's union paying +0.070\n"
             "and the four unions since paying nothing. Every one of those four\n"
             "failures added a member 0.03-0.06 behind the incumbent. This is\n"
             "the first candidate that is not.\n"
             "\n"
             "It runs on CPU and therefore costs ZERO GPU quota — the whole\n"
             "point of building it from checkpoints that already exist rather\n"
             "than re-running five folds with a wider dump, which would have\n"
             "cost ~7.5 GPU-h for the same file.\n"
             "\n"
             "It also self-verifies. The gold macro is still computed from the\n"
             "gold subset, so this run must reproduce 0.8980. If it does not,\n"
             "the fold split here differs from the trainer's and the OOF is not\n"
             "out-of-fold — a teacher built on that would leak, train cleanly,\n"
             "and score worse for no visible reason.\n"
             "\n"
             "NOTHING IS BLENDED HERE. What to mix with the report labels, and\n"
             "at what weight, is decided offline on the 58 — not baked into a\n"
             "two-hour CPU run that would have to be repeated to revisit it.",
    ),
    Kernel(
        slug="knee-oof-v1pubb",
        directory="71_oof_v1pubb",
        template="gold_eval",
        gpu=False,          # 4,407 forward passes; CPU is a separate allowance
        internet=False,
        depends=["knee-cache-build-0", "knee-cache-build-1", "knee-cache-build-2",
                 "knee-cache-build-3",
                 "knee-train-v1pubB", "knee-train-v1pubB-fold1",
                 "knee-train-v1pubB-fold2", "knee-train-v1pubB-fold3",
                 "knee-train-v1pubB-fold4"],
        datasets=[PUBLIC_DATASET],
        constants={**V1.constants(), "TTA_VIEWS": ("identity",),
                   "OOF_SCOPE": "all"},
        note="A SECOND out-of-fold prediction per study, from the reseeded\n"
             "lineage, to strengthen the model arm of the distilled teacher.\n"
             "\n"
             "E069's teacher is the rank union of the public report labels\n"
             "(0.8927 gold) with ONE prediction per study from v1public\n"
             "(0.8980). Each study is predicted once, by the single fold model\n"
             "that held it out, so that arm carries the full single-model\n"
             "variance. Averaging a second independent prediction reduces it\n"
             "without changing anything else.\n"
             "\n"
             "The reason to expect it to pay is E048's rule rather than\n"
             "optimism: a union pays when its members are COMPARABLE. E061\n"
             "pooled v1publicB at 0.8827 against v1public's 0.8980 — 0.015\n"
             "apart, inside the 0.02 band, where E023's union paid +0.070 and\n"
             "the five unions with a member 0.03-0.06 behind paid nothing.\n"
             "\n"
             "CPU only, so it costs ZERO GPU quota — which matters because the\n"
             "weekly allowance is spent and the distilled lineage is queued\n"
             "behind the reset. If this improves the teacher, it improves it\n"
             "BEFORE those five folds train, not after.\n"
             "\n"
             "Self-verifying like knee-oof-v1pub: the gold macro is still\n"
             "computed from the gold subset, so this run must reproduce\n"
             "v1publicB's 0.8827. If it does not, the folds were cut\n"
             "differently from the trainer and the predictions are not\n"
             "out-of-fold.",
    ),
    Kernel(
        slug="knee-llm-labeler",
        directory="16_llm_labeler",
        template="llm_label",
        gpu=True,
        internet=True,      # a training-time kernel: it fetches open weights
        datasets=[ARTIFACTS_DATASET],
        constants={"RUN_MODEL": "Qwen/Qwen2.5-7B-Instruct",
                   # Both T4s, with headroom. The first attempt used
                   # device_map="auto", which put every weight on GPU 0 —
                   # 13.59 of 14.56 GiB — and left nothing for activations, so
                   # every batch raised OutOfMemoryError while the second card
                   # sat idle.
                   "RUN_GPU_BUDGET": "9GiB",
                   "RUN_MAX_REPORTS": 0,      # 0 = every report
                   # Measured locally rather than guessed (E022): prompts are
                   # 716-1,138 tokens, not the ~3,000 the truncation limit
                   # allows for, and completions are 109-128 tokens. The first
                   # run's OOM came from batch 32 on a single card, not from
                   # long sequences. 16 is affordable at ~1.1k tokens across two
                   # sharded cards, and run_batch halves on an OOM anyway, so a
                   # wrong guess costs one retry rather than the session.
                   "RUN_BATCH": 16,
                   "RUN_MAX_NEW_TOKENS": 160,
                   "RUN_TIME_BUDGET": Raw("8.0 * 3600")},
        note="Reads every report into a closed 7-state ladder with an\n"
             "open-weights model, then maps states to soft targets in Python.\n"
             "The lexicon labeler reaches 0.769 against the 58 expert-labelled\n"
             "studies; published systems using this method reach 0.881, and the\n"
             "gap is paraphrase and negation scope across ten languages rather\n"
             "than missing vocabulary.\n"
             "\n"
             "COMPLIANCE: open weights, inside a Kaggle kernel. No report text\n"
             "leaves this kernel and no hosted API is contacted — see\n"
             "docs/STRATEGY.md on competition Rule 4.b.",
    ),
]


def caches() -> list[Cache]:
    seen, out = set(), []
    for lineage in LINEAGES:
        if id(lineage.cache) not in seen:
            seen.add(id(lineage.cache))
            out.append(lineage.cache)
    return out


def all_kernels() -> list[Kernel]:
    out: list[Kernel] = []
    for cache in caches():
        out.extend(cache.kernels())
    for lineage in LINEAGES:
        out.extend(lineage.kernels())
    out.extend(EXTRAS)
    return out


def check() -> list[str]:
    """Problems a declaration can have, as human-readable strings."""
    problems = []
    kernels = all_kernels()
    known = {k.slug for k in kernels}

    for kernel in kernels:
        for dependency in kernel.depends:
            if dependency not in known:
                problems.append(f"{kernel.slug} mounts unknown kernel {dependency}")
        if kernel.template == "infer" and kernel.internet:
            problems.append(f"{kernel.slug} is a submission kernel with internet on")

    for attribute in ("slug", "directory"):
        seen = {}
        for kernel in kernels:
            value = getattr(kernel, attribute)
            if value in seen:
                problems.append(f"two kernels share {attribute} {value}")
            seen[value] = kernel

    # An inference kernel averages its mounted trainers, so they must agree on
    # everything that changes what the model was fed.
    by_slug = {k.slug: k for k in kernels}
    for kernel in kernels:
        if kernel.template != "infer":
            continue
        if not kernel.depends:
            problems.append(f"{kernel.slug} is an inference kernel mounting nothing")
            continue
        for key, expected in (("SLICE_SUBSAMPLE", "SLICE_SUBSAMPLE_EXPECTED"),
                              ("INPUT_NORM", "INPUT_NORM_EXPECTED"),
                              ("TARGET_SIZE", "TARGET_SIZE"),
                              ("TARGET_MM_PER_PIXEL", "TARGET_MM_PER_PIXEL"),
                              ("SLICES_PER_PLANE", "SLICES_PER_PLANE")):
            want = kernel.constants[expected]
            for slug in kernel.depends:
                got = by_slug[slug].constants[key]
                if got != want:
                    problems.append(
                        f"{kernel.slug} expects {key}={want!r} but mounts {slug} "
                        f"trained with {got!r}")
    return problems


if __name__ == "__main__":
    kernels = all_kernels()
    print(f"{len(LINEAGES)} lineages, {len(caches())} caches, {len(kernels)} kernels")
    for kernel in kernels:
        print(f"  {kernel.directory:24s} {kernel.slug:22s} {kernel.template}")
    for problem in check():
        print(f"PROBLEM: {problem}")
