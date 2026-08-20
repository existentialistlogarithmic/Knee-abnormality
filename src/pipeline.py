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

from dataclasses import dataclass, field

ACCOUNT = "achelijndiamantidis"
COMPETITION = "rsna-knee-abnormality-detection"
ARTIFACTS_DATASET = f"{ACCOUNT}/knee-phase1-artifacts"
# Same headers, but soft_labels.parquet is the FUSION of the lexicon labeler
# and the LLM reader. A separate dataset rather than a new version of the
# one above, so every run already made stays comparable — replacing the
# labels in place would silently change what every earlier number meant.
FUSED_DATASET = f"{ACCOUNT}/knee-phase1-fused"
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
    note: str = ""

    def constants(self) -> dict[str, object]:
        return {"RUN_EPOCHS": self.epochs, "RUN_BATCH": self.batch,
                "ACCUM_STEPS": self.accum, "RUN_LR": self.lr,
                "RUN_BACKBONE": self.backbone,
                "SLICE_SUBSAMPLE": self.slice_subsample,
                "INPUT_NORM": self.input_norm,
                "PER_FINDING_POOL": self.per_finding_pool,
                "FOCAL_K": self.focal_k,
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
    datasets: list[str] = field(default_factory=list)
    constants: dict[str, object] = field(default_factory=dict)
    note: str = ""

    def metadata(self) -> dict:
        return {
            "id": f"{ACCOUNT}/{self.slug}",
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
            "kernel_sources": [f"{ACCOUNT}/{d}" for d in self.depends],
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
                           "INPUT_NORM_EXPECTED": self.train.input_norm},
                note=self.geometry.note,
            ))
        return out


# --------------------------------------------------------------------------- #
# The pipeline as it actually stands. Every number below has either been run or
# is queued to run; nothing here is aspirational.
# --------------------------------------------------------------------------- #

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
        trainers=(Trainer(0, "knee-train-v1fused", "21_train_v1fused"),),
        infer_slug="knee-infer-v1fused",
        infer_directory="22_infer_v1fused",
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
                   # DINOv2, not resnet34. The resnet34 pass was run first and
                   # its features turned out to be AT CHANCE on the focal
                   # findings — ACL 0.532, MCL 0.478, Medial Meniscus 0.538 —
                   # so no pooling change above them could be tested, because
                   # there was nothing there to pool. Measured at 2.13 s/study
                   # against resnet34's 1.76, so 2.6 hours rather than 2.2.
                   "RUN_BACKBONE": "vit_small_patch14_dinov2.lvd142m",
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
        constants={**V1.constants()},
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
