"""The detector: frozen backbone + linear head, exposed as one nn.Module.

Everything downstream (explainers, perturbation metrics) sees a single callable
that maps **[0,1] pixel tensors** to a fake-logit. That uniform interface is
what lets the same faithfulness code run over CNN, ViT and CLIP detectors
without special-casing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .backbones import BackboneSpec, Normalize, build_backbone, get_spec


class Detector(nn.Module):
    """``pixels in [0,1]`` -> ``logit`` (positive = fake).

    A 2-class head is used rather than a single output so that Grad-CAM and
    Captum attributions have a conventional class dimension to target.
    """

    def __init__(self, backbone_name: str, pretrained: bool = True,
                 head: Optional[nn.Module] = None):
        super().__init__()
        self.backbone_name = backbone_name
        bb, spec = build_backbone(backbone_name, pretrained=pretrained)
        self.spec: BackboneSpec = spec
        self.normalize = Normalize(spec.mean, spec.std)
        self.backbone = bb
        self.head = head if head is not None else nn.Linear(spec.feat_dim, 2)

    # -- forward paths -------------------------------------------------------
    def features(self, x01: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.normalize(x01))

    def forward(self, x01: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, 2). Input must be [0,1] pixels."""
        return self.head(self.features(x01))

    @torch.no_grad()
    def fake_prob(self, x01: torch.Tensor) -> torch.Tensor:
        """P(fake) for a batch of [0,1] pixel tensors."""
        return torch.softmax(self(x01), dim=1)[:, 1]

    # -- persistence ---------------------------------------------------------
    def load_head(self, path: str | Path) -> "Detector":
        state = torch.load(path, map_location="cpu")
        w = state["weight"] if isinstance(state, dict) and "weight" in state else state
        self.head.load_state_dict(
            w if isinstance(w, dict) else {"weight": w, "bias": state["bias"]}
        )
        return self

    def save_head(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path)
        return path

    def set_head_from_sklearn(self, clf) -> "Detector":
        """Copy a fitted sklearn ``LogisticRegression`` into the torch head.

        sklearn's binary LR stores a single (1, D) weight row for class 1. We
        expand it to a 2-logit head as [-w/2, +w/2] so the softmax matches
        sklearn's sigmoid exactly.
        """
        w = np.asarray(clf.coef_, dtype=np.float32).reshape(-1)
        b = float(np.asarray(clf.intercept_, dtype=np.float32).reshape(-1)[0])
        W = np.stack([-w / 2.0, w / 2.0], axis=0)
        B = np.array([-b / 2.0, b / 2.0], dtype=np.float32)
        with torch.no_grad():
            self.head.weight.copy_(torch.from_numpy(W))
            self.head.bias.copy_(torch.from_numpy(B))
        return self


def build_detector(backbone_name: str, head_path: Optional[str | Path] = None,
                   device: str = "cpu", eval_mode: bool = True) -> Detector:
    det = Detector(backbone_name)
    if head_path is not None and Path(head_path).exists():
        det.head.load_state_dict(torch.load(head_path, map_location="cpu"))
    det = det.to(device)
    if eval_mode:
        det.eval()
        for p in det.parameters():
            p.requires_grad_(False)
        # Grad-CAM needs gradients w.r.t. activations, not parameters; input
        # grads are enabled per-call by the explainer.
    return det


@torch.no_grad()
def batched_fake_prob(det: Detector, x01: torch.Tensor, batch_size: int = 64,
                      amp: bool = True) -> torch.Tensor:
    """Score a (possibly large) stack of perturbed variants in chunks.

    The faithfulness metrics generate hundreds of variants per image; this is
    the single hottest function in the codebase.
    """
    device = next(det.parameters()).device
    use_amp = amp and device.type == "cuda"
    out = []
    for i in range(0, x01.shape[0], batch_size):
        chunk = x01[i : i + batch_size].to(device, non_blocking=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.softmax(det(chunk), dim=1)[:, 1]
        else:
            p = torch.softmax(det(chunk), dim=1)[:, 1]
        out.append(p.float().cpu())
    return torch.cat(out) if out else torch.empty(0)
