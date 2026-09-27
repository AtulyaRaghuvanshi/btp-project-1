"""Perturbation-based faithfulness metrics, parameterised by the operator.

The one structural difference from every existing implementation (Quantus,
Captum, the Tsigos et al. harness) is that the **fill operator is an explicit
argument**, not a hard-coded constant. That is the whole experiment: run the
same metric under ``zero`` and under ``real_transplant`` and see whether the
ranking of explanation methods survives.

Each function returns per-image curves *and* their summary AUCs, because the
curves are what reveal the leakage signature (a leaky operator shows a rising
p(fake) even as evidence is removed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..utils.compat import trapezoid

from ..models.detector import Detector
from ..perturb.operators import DonorBank, apply_operator


def _rank_masks(sal: torch.Tensor, fractions: Sequence[float],
                most_first: bool = True) -> torch.Tensor:
    """Build cumulative top-k masks from a saliency map.

    ``sal``: (H, W). Returns (len(fractions), 1, H, W) with 1 = perturb.
    Ties are broken by flat index, deterministically.
    """
    H, W = sal.shape
    n = H * W
    flat = sal.reshape(-1)
    order = torch.argsort(flat, descending=most_first)
    masks = torch.zeros(len(fractions), n, device=sal.device, dtype=sal.dtype)
    for i, f in enumerate(fractions):
        k = int(round(f * n))
        if k > 0:
            masks[i, order[:k]] = 1.0
    return masks.reshape(len(fractions), 1, H, W)


def _auc(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """Normalised area under a curve sampled at ``x`` (default: uniform)."""
    y = np.asarray(y, dtype=np.float64)
    if x is None:
        x = np.linspace(0.0, 1.0, len(y))
    return float(trapezoid(y, x) / max(x[-1] - x[0], 1e-12)) if len(y) > 1 else float(y[0])


@dataclass
class CurveResult:
    fractions: np.ndarray
    curves: np.ndarray           # (N_images, n_steps) of P(fake)
    auc: np.ndarray              # (N_images,)
    meta: Dict = field(default_factory=dict)

    def mean_curve(self) -> np.ndarray:
        return self.curves.mean(axis=0)

    def summary(self) -> Dict[str, float]:
        return {
            "auc_mean": float(self.auc.mean()),
            "auc_std": float(self.auc.std()),
            "auc_sem": float(self.auc.std() / max(np.sqrt(len(self.auc)), 1.0)),
            "n": int(len(self.auc)),
        }


@torch.no_grad()
def perturbation_curve(detector: Detector, images: torch.Tensor,
                       saliency: torch.Tensor, operator: str = "zero",
                       mode: str = "deletion", n_steps: int = 25,
                       max_fraction: float = 1.0,
                       donor_bank: Optional[DonorBank] = None,
                       batch_size: int = 32, amp: bool = True,
                       seed: int = 1337) -> CurveResult:
    """Core routine behind deletion / insertion.

    ``mode="deletion"``  perturb the most-salient pixels first; a faithful map
                         makes P(fake) fall fast, so **lower AUC is better**.
    ``mode="insertion"`` start fully perturbed and restore the most-salient
                         pixels first; **higher AUC is better**.

    ``images``: (N, 3, H, W) in [0,1]. ``saliency``: (N, H, W) in [0,1].
    """
    device = next(detector.parameters()).device
    use_amp = amp and device == torch.device("cuda") or (amp and device.type == "cuda")
    fractions = np.linspace(0.0, max_fraction, n_steps)
    N = images.shape[0]
    curves = np.zeros((N, n_steps), dtype=np.float32)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    for i in range(N):
        x = images[i : i + 1].to(device)
        sal = saliency[i].to(device)
        masks = _rank_masks(sal, fractions, most_first=True)

        if mode == "deletion":
            m_all = masks                       # (n_steps, 1, H, W)
        elif mode == "insertion":
            # Keep the top-k salient pixels, perturb everything else.
            m_all = 1.0 - masks
        else:
            raise ValueError("mode must be 'deletion' or 'insertion'")

        # One operator call for all masking levels: road_linear and friends are
        # far too expensive to invoke per step.
        x_rep = x.expand(n_steps, -1, -1, -1).contiguous()
        batch = apply_operator(operator, x_rep, m_all, donor_bank=donor_bank)
        batch = torch.where(
            (m_all.flatten(1).sum(1) > 0).view(-1, 1, 1, 1), batch, x_rep
        )

        probs = []
        for k in range(0, batch.shape[0], batch_size):
            chunk = batch[k : k + batch_size]
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    p = torch.softmax(detector(chunk), dim=1)[:, 1]
            else:
                p = torch.softmax(detector(chunk), dim=1)[:, 1]
            probs.append(p.float().cpu())
        curves[i] = torch.cat(probs).numpy()

    auc = np.array([_auc(curves[i], fractions) for i in range(N)], dtype=np.float32)
    return CurveResult(fractions, curves, auc,
                       meta={"operator": operator, "mode": mode})


def deletion(detector, images, saliency, **kw) -> CurveResult:
    return perturbation_curve(detector, images, saliency, mode="deletion", **kw)


def insertion(detector, images, saliency, **kw) -> CurveResult:
    return perturbation_curve(detector, images, saliency, mode="insertion", **kw)


@torch.no_grad()
def comprehensiveness_sufficiency(detector: Detector, images: torch.Tensor,
                                  saliency: torch.Tensor, operator: str = "zero",
                                  fractions: Sequence[float] = (0.05, 0.1, 0.2, 0.5),
                                  donor_bank: Optional[DonorBank] = None,
                                  amp: bool = True) -> Dict[str, np.ndarray]:
    """DeYoung et al.'s pair, in pixel space.

    comprehensiveness = p(x) - p(x with top-k removed)   -> higher is better
    sufficiency       = p(x) - p(only top-k kept)        -> lower is better
    """
    device = next(detector.parameters()).device
    N = images.shape[0]
    comp = np.zeros((N, len(fractions)), dtype=np.float32)
    suff = np.zeros((N, len(fractions)), dtype=np.float32)

    for i in range(N):
        x = images[i : i + 1].to(device)
        base = float(torch.softmax(detector(x), dim=1)[0, 1])
        masks = _rank_masks(saliency[i].to(device), fractions, most_first=True)
        for j in range(len(fractions)):
            m = masks[j : j + 1]
            removed = apply_operator(operator, x, m, donor_bank=donor_bank)
            kept = apply_operator(operator, x, 1.0 - m, donor_bank=donor_bank)
            p_rem = float(torch.softmax(detector(removed), dim=1)[0, 1])
            p_keep = float(torch.softmax(detector(kept), dim=1)[0, 1])
            comp[i, j] = base - p_rem
            suff[i, j] = base - p_keep
    return {
        "fractions": np.asarray(fractions),
        "comprehensiveness": comp,
        "sufficiency": suff,
        "comp_mean": comp.mean(axis=0),
        "suff_mean": suff.mean(axis=0),
    }


def corrected_faithfulness(salient: CurveResult, random_ctrl: CurveResult) -> Dict:
    """**Leakage-corrected faithfulness** -- the metric this project proposes.

    A raw deletion AUC conflates two things:

      1. the explanation genuinely locating pixels the model relies on, and
      2. the fill operator injecting an artifact that moves the score by itself.

    Term 2 is present in the random-location control too, so the difference
    isolates term 1::

        DeltaAUC = AUC_random - AUC_salient

    Positive and large = the explanation beats chance *after* discounting the
    operator. Near zero = the raw score was mostly the operator talking.
    Reported alongside a paired t-test, since both arms use the same images.
    """
    from scipy import stats

    d = random_ctrl.auc - salient.auc
    t, p = stats.ttest_rel(random_ctrl.auc, salient.auc)
    pooled = np.sqrt((random_ctrl.auc.var() + salient.auc.var()) / 2.0)
    return {
        "delta_auc_mean": float(d.mean()),
        "delta_auc_sem": float(d.std() / max(np.sqrt(len(d)), 1.0)),
        "auc_salient": float(salient.auc.mean()),
        "auc_random": float(random_ctrl.auc.mean()),
        "cohens_d": float(d.mean() / pooled) if pooled > 1e-9 else 0.0,
        "t_stat": float(t),
        "p_value": float(p),
        "n": int(len(d)),
    }
