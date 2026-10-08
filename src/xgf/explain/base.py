"""Common interface for saliency methods.

An explainer maps ``(detector, image)`` to a single-channel saliency map,
normalised to [0, 1] at the image resolution. Everything downstream --
deletion/insertion, ROAD, pointing game -- consumes only that.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_REGISTRY: Dict[str, type] = {}


def register_explainer(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def list_explainers() -> List[str]:
    return sorted(_REGISTRY)


def build_explainer(name: str, detector, **kw):
    if name not in _REGISTRY:
        raise KeyError(
            "Unknown explainer '{}'. Available: {}".format(name, list_explainers())
        )
    return _REGISTRY[name](detector, **kw)


def normalize_map(sal: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max to [0,1]. Shape (B, H, W) in, same out.

    Faithfulness metrics only use the *ranking* of pixels, so the affine
    normalisation is cosmetic -- but it makes maps comparable in figures and
    keeps the pointing-game threshold meaningful.
    """
    B = sal.shape[0]
    flat = sal.reshape(B, -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo).clamp_min(1e-8)
    return flat.reshape(sal.shape)


def resize_map(sal: torch.Tensor, size: int) -> torch.Tensor:
    """Bilinear-resize a (B, h, w) map to (B, size, size)."""
    if sal.shape[-1] == size and sal.shape[-2] == size:
        return sal
    return F.interpolate(
        sal.unsqueeze(1), size=(size, size), mode="bilinear", align_corners=False
    ).squeeze(1)


class Explainer:
    """Base class. Subclasses implement ``_explain``."""

    name: str = "base"
    #: Rough number of forward passes per image -- used by the cost estimator.
    cost_per_image: int = 1

    def __init__(self, detector, **kw):
        self.detector = detector
        self.device = next(detector.parameters()).device
        self.kw = kw

    def _explain(self, x01: torch.Tensor, target: int) -> torch.Tensor:
        raise NotImplementedError

    def reseed(self, seed: int) -> None:
        """Reset any RNG the explainer carries between calls.

        Needed for resumable runs: a map must depend on ``(seed, input)`` only,
        never on how many batches were explained before it. Methods that already
        reseed inside every call (RISE, LIME) need nothing here.
        """

    def __call__(self, x01: torch.Tensor, target: int = 1) -> torch.Tensor:
        """``x01``: (B, 3, H, W) in [0,1]. Returns (B, H, W) in [0,1]."""
        x01 = x01.to(self.device)
        sal = self._explain(x01, target)
        sal = resize_map(sal, x01.shape[-1])
        return normalize_map(sal).detach().cpu()
