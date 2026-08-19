"""The 2.5D model: a 2D backbone over slices, attention-pooled.

Included verbatim into generated kernels by eda/generate_kernels.py.
Kaggle script kernels are single files, so sharing code means splicing it
at generation time. Editing it here changes every kernel that includes it.
"""

# ImageNet statistics. Every pretrained backbone here — torchvision and DINOv2
# alike — was trained on inputs normalised this way. The earliest runs fed raw
# 0..1 values straight in, which shifts the input distribution away from what
# the pretrained filters expect and quietly costs transfer quality. It never
# errors; it just makes the pretrained weights worth less than they should be.
#
# It is therefore a property of a trained model, not a preference: INPUT_NORM
# comes from the manifest at training time and from the checkpoint at inference
# time, and mixing the two in an ensemble is refused rather than averaged.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2 uses 14-pixel patches, so its input side must be a multiple of 14.
# A 192px cache becomes 196 here — a 2% resize — which keeps one cache usable
# by every architecture instead of forcing a rebuild per backbone.
PATCH_MULTIPLE = 14


def build_model(backbone: str, n_planes: int, n_out: int, normalise_input: bool,
                per_finding_pool: bool = False):
    """2.5D: a 2D backbone over slices, attention-pooled to a study.

    Attention pooling rather than mean pooling because a finding is usually
    visible on a handful of slices; averaging over twenty dilutes it.

    Accepts either a torchvision name or a timm name. DINOv2 is the reason:
    the public baseline for this competition reportedly reaches ~0.809 with
    DINOv2 features while this project's ImageNet resnet34 reached 0.725, and
    self-supervised features transfer to medical imaging far better than
    ImageNet classification features do.
    """
    import torch
    import torch.nn as nn

    net = None
    features = None
    if "." in backbone or backbone.startswith(("vit_", "convnext", "tf_efficientnet")):
        import timm

        try:
            net = timm.create_model(backbone, pretrained=True, num_classes=0,
                                    dynamic_img_size=True)
        except Exception:  # noqa: BLE001 - offline, or no dynamic_img_size support
            try:
                net = timm.create_model(backbone, pretrained=True, num_classes=0)
            except Exception:  # noqa: BLE001 - internet off (inference kernels)
                print("no pretrained download (internet off) — random init; "
                      "at inference the checkpoint replaces all of it")
                net = timm.create_model(backbone, pretrained=False, num_classes=0,
                                        dynamic_img_size=True)
        features = net.num_features
    else:
        import torchvision

        try:
            net = getattr(torchvision.models, backbone)(weights="DEFAULT")
        except Exception:  # noqa: BLE001
            print("no pretrained download (internet off) — random init; "
                  "at inference the checkpoint replaces all of it")
            net = getattr(torchvision.models, backbone)(weights=None)
        features = net.fc.in_features
        net.fc = nn.Identity()

    is_patch_model = backbone.startswith("vit_")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = net
            self.patch_multiple = PATCH_MULTIPLE if is_patch_model else 0
            self.normalise = normalise_input
            # persistent=False deliberately: these are constants, not learned
            # state. Persisting them would add two keys to every state_dict and
            # the strict-load check at inference would then reject every
            # checkpoint written before they existed — including the folds
            # currently training.
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                                 persistent=False)
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                                 persistent=False)
            # One attention map for twelve findings forces a single compromise
            # over which slices matter. A meniscal tear occupies a handful of
            # sagittal slices and a large effusion occupies most of the study,
            # so the compromise is paid mostly by the focal findings — which are
            # exactly the four weakest here (Medial Meniscus 0.656, PF OA 0.659,
            # Synovitis 0.663, MCL 0.669 on fold 1).
            #
            # With per_finding_pool each finding gets its own attention map over
            # the same slice embeddings and its own weight vector over features.
            # Cost is one 128xN linear and an einsum over ~60 tokens: negligible
            # against the backbone, which is why this is worth trying before
            # anything that buys AUC with runtime.
            self.per_finding = per_finding_pool
            maps = n_out if per_finding_pool else 1
            self.attention = nn.Sequential(nn.Linear(features, 128), nn.Tanh(),
                                           nn.Linear(128, maps))
            if per_finding_pool:
                self.head_weight = nn.Parameter(torch.zeros(n_out, features))
                nn.init.trunc_normal_(self.head_weight, std=0.02)
                self.head_bias = nn.Parameter(torch.zeros(n_out))
            else:
                self.head = nn.Linear(features, n_out)

        def forward(self, x):                      # x: (B, P, S, H, W)
            b, p, s, h, w = x.shape
            flat = x.reshape(b * p * s, 1, h, w).repeat(1, 3, 1, 1)

            # Patch-based backbones need a side length divisible by the patch
            # size. Resizing here rather than in the cache keeps one cache
            # usable by every architecture.
            if self.patch_multiple and (h % self.patch_multiple or w % self.patch_multiple):
                side = int(round(h / self.patch_multiple)) * self.patch_multiple
                flat = torch.nn.functional.interpolate(
                    flat, size=(side, side), mode="bilinear", align_corners=False)

            if self.normalise:
                flat = (flat - self.mean) / self.std
            embedded = self.backbone(flat).reshape(b, p * s, -1)
            scores = self.attention(embedded).softmax(dim=1)   # (B, T, maps)
            if self.per_finding:
                # (B, T, F) x (B, T, C) -> (B, F, C): one pooled vector per
                # finding, each attending wherever that finding actually lives.
                pooled = torch.einsum("btf,btc->bfc", scores, embedded)
                return (pooled * self.head_weight).sum(-1) + self.head_bias
            pooled = (embedded * scores).sum(dim=1)
            return self.head(pooled)

    return Model()
