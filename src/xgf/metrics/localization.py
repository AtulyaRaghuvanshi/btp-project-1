"""Localization / plausibility metrics against ground-truth manipulation masks.

Faithfulness and plausibility are different questions and this project keeps
them apart on purpose:

* **Faithfulness** (``metrics.faithfulness``) -- does the map reflect what the
  *model* used? Needs no ground truth.
* **Plausibility** (here) -- does the map point at what was *actually*
  manipulated? Needs a mask, so it only applies to sources that ship one
  (CocoGlide, AutoSplice, FF++).

A detector can be perfectly faithful and completely implausible: the map
honestly reports that the model keyed on a JPEG grid in the background. Papers
that show only heatmap screenshots routinely conflate the two.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


def pointing_game(saliency: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """1 if the saliency argmax falls inside the manipulated region.

    ``saliency``: (N, H, W), ``mask``: (N, H, W) binary. Returns (N,) of 0/1.
    """
    N, H, W = saliency.shape
    flat = saliency.reshape(N, -1)
    idx = flat.argmax(dim=1)
    ys, xs = idx // W, idx % W
    hits = mask[torch.arange(N), ys, xs]
    return (hits > 0.5).float().numpy()


def mask_iou(saliency: torch.Tensor, mask: torch.Tensor,
             threshold: Optional[float] = None) -> np.ndarray:
    """IoU between a binarised saliency map and the GT mask.

    When ``threshold`` is None the map is binarised at its own top-k, where k
    is the true mask area. That removes the (large) confound of a method simply
    producing broader maps than another.
    """
    N, H, W = saliency.shape
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        gt = mask[i] > 0.5
        k = int(gt.sum().item())
        if k == 0:
            out[i] = np.nan
            continue
        if threshold is None:
            flat = saliency[i].reshape(-1)
            order = torch.argsort(flat, descending=True)
            pred = torch.zeros_like(flat, dtype=torch.bool)
            pred[order[:k]] = True
            pred = pred.reshape(H, W)
        else:
            pred = saliency[i] > threshold
        inter = (pred & gt).sum().item()
        union = (pred | gt).sum().item()
        out[i] = inter / union if union > 0 else np.nan
    return out


def mask_auc(saliency: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Per-image pixel-level ROC-AUC of saliency vs the GT mask.

    Threshold-free, so it is the fairest single plausibility number.
    """
    from sklearn.metrics import roc_auc_score

    N = saliency.shape[0]
    out = np.zeros(N, dtype=np.float32)
    for i in range(N):
        y = (mask[i].reshape(-1).numpy() > 0.5).astype(np.int8)
        if y.min() == y.max():
            out[i] = np.nan
            continue
        out[i] = roc_auc_score(y, saliency[i].reshape(-1).numpy())
    return out


def localization_summary(saliency: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    pg = pointing_game(saliency, mask)
    iou = mask_iou(saliency, mask)
    auc = mask_auc(saliency, mask)
    return {
        "pointing_game": float(np.nanmean(pg)),
        "mask_iou": float(np.nanmean(iou)),
        "mask_auc": float(np.nanmean(auc)),
        "n": int(len(pg)),
    }
