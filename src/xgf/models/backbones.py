"""Frozen feature extractors.

Design rationale (this is what makes the project fit a 4 GB laptop GPU):
every detector is ``frozen backbone -> small trainable head``. Backbone weights
never change, so

* features can be extracted once and cached,
* head training is logistic regression on cached features -- seconds on CPU,
* the only expensive step is the faithfulness evaluation, which is inference.

The architecture axis (CNN vs ViT vs CLIP-ViT) comes for free, because all three
are just different frozen backbones behind an identical head.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..data.datasets import CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD


@dataclass
class BackboneSpec:
    name: str
    kind: str                 # "cnn" | "vit" | "clip"
    feat_dim: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    # Name of the module whose activations Grad-CAM should hook.
    cam_layer: Optional[str] = None


class Normalize(nn.Module):
    """Pixel-space [0,1] -> backbone-normalised tensor, as a module.

    Keeping this inside the model (rather than in the dataloader) is what lets
    every perturbation operator work in plain [0,1] pixel space.
    """

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class TimmBackbone(nn.Module):
    """Wraps a timm model as a pooled-feature extractor."""

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feat_dim = self.model.num_features
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class OpenClipBackbone(nn.Module):
    """CLIP image tower via open_clip, returning the pooled image embedding."""

    def __init__(self, model_name: str = "ViT-B-16",
                 pretrained: str = "openai"):
        super().__init__()
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.visual = model.visual
        for p in self.visual.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.feat_dim = int(self.visual(dummy).shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_SPECS: Dict[str, BackboneSpec] = {
    # CNN arm -- the Tsigos et al. style detector family.
    "resnet50": BackboneSpec(
        "resnet50", "cnn", 2048, IMAGENET_MEAN, IMAGENET_STD, cam_layer="layer4"
    ),
    "efficientnet_b0": BackboneSpec(
        "efficientnet_b0", "cnn", 1280, IMAGENET_MEAN, IMAGENET_STD,
        cam_layer="conv_head",
    ),
    "xception": BackboneSpec(
        "legacy_xception", "cnn", 2048, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
        cam_layer="conv4",
    ),
    # Plain-ViT arm.
    "vit_b16": BackboneSpec(
        "vit_base_patch16_224", "vit", 768, IMAGENET_MEAN, IMAGENET_STD,
        cam_layer="blocks.11.norm1",
    ),
    # CLIP arm -- the UnivFD / Effort recipe, strongest cross-generator baseline.
    "clip_vitb16": BackboneSpec(
        "ViT-B-16", "clip", 512, CLIP_MEAN, CLIP_STD,
        cam_layer="transformer.resblocks.11.ln_1",
    ),
    "clip_vitl14": BackboneSpec(
        "ViT-L-14", "clip", 768, CLIP_MEAN, CLIP_STD,
        cam_layer="transformer.resblocks.23.ln_1",
    ),
}


def list_backbones() -> List[str]:
    return sorted(_SPECS)


def get_spec(name: str) -> BackboneSpec:
    if name not in _SPECS:
        raise KeyError(
            "Unknown backbone '{}'. Available: {}".format(name, list_backbones())
        )
    return _SPECS[name]


def build_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, BackboneSpec]:
    spec = get_spec(name)
    if spec.kind == "clip":
        bb = OpenClipBackbone(spec.name, pretrained="openai")
    else:
        bb = TimmBackbone(spec.name, pretrained=pretrained)
    # Trust the constructed model over the hard-coded dim.
    spec = BackboneSpec(
        spec.name, spec.kind, int(bb.feat_dim), spec.mean, spec.std, spec.cam_layer
    )
    bb.eval()
    return bb, spec
