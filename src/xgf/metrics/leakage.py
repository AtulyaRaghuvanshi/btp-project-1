"""Perturbation leakage: does the measuring instrument emit its own signal?

**This is the project's primary methodological contribution.**

Claim: perturbation-based faithfulness metrics are systematically biased when
the model under test is a forgery detector, because the fill operator creates
low-level statistical discontinuities -- hard edges, flat regions, alien
frequency content -- of the same kind the detector was trained to fire on.

The test is a null experiment, and it needs no explanation method at all:

    Take **real** images the detector confidently calls real.
    Mask a random region (not a salient one -- there is no explanation here).
    Measure how far P(fake) moves.

Under an unbiased operator that quantity is ~0: scrambling arbitrary pixels of
a genuine photograph should not manufacture evidence of forgery. Whatever it
actually is, is the operator's leakage, and it contaminates every deletion and
insertion curve ever computed with that operator.

Three numbers come out:

``leakage_auc``     area under P(fake) vs masked-fraction, on real images.
                    0 = clean instrument, 1 = the operator alone convinces the
                    detector everything is fake.
``flip_rate``       fraction of real images pushed across the decision
                    threshold by masking alone.
``leakage_slope``   dP(fake)/d(fraction) near the origin -- the regime where
                    deletion curves are actually read.

Reported per (detector, operator, generator-family). The family axis matters:
if leakage differs across families, then cross-generator faithfulness
comparisons made with a leaky operator are not comparing like with like, which
is exactly the confound in the cross-generator experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..utils.compat import trapezoid

from ..models.detector import Detector
from ..perturb.operators import DonorBank, apply_operator


def random_masks(n: int, size: int, fractions: Sequence[float], seed: int = 1337,
                 blob_scale: int = 14) -> torch.Tensor:
    """Spatially-coherent random masks at prescribed area fractions.

    Coherent blobs rather than scattered pixels, so the mask geometry matches
    what a saliency map produces -- otherwise the control would face an easier
    (or harder) perturbation than the salient arm and the subtraction in
    ``corrected_faithfulness`` would be invalid.
    """
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(seed)
    low = torch.rand(n, 1, blob_scale, blob_scale, generator=g)
    field = F.interpolate(low, size=(size, size), mode="bilinear",
                          align_corners=False).squeeze(1)
    out = torch.zeros(n, len(fractions), size, size)
    flat = field.reshape(n, -1)
    order = torch.argsort(flat, dim=1, descending=True)
    npix = size * size
    for i in range(n):
        for j, f in enumerate(fractions):
            k = int(round(f * npix))
            m = torch.zeros(npix)
            if k > 0:
                m[order[i, :k]] = 1.0
            out[i, j] = m.reshape(size, size)
    return out


@dataclass
class LeakageResult:
    operator: str
    fractions: np.ndarray
    curves: np.ndarray            # (N, n_steps) P(fake) on REAL images
    baseline: np.ndarray          # (N,) P(fake) of the untouched real image
    meta: Dict

    @property
    def delta_curves(self) -> np.ndarray:
        return self.curves - self.baseline[:, None]

    def summary(self, threshold: float = 0.5) -> Dict[str, float]:
        d = self.delta_curves
        mean_curve = self.curves.mean(axis=0)
        leak_auc = float(trapezoid(mean_curve, self.fractions) /
                         max(self.fractions[-1] - self.fractions[0], 1e-12))
        # Slope over the first 20% of masking -- the regime deletion curves use.
        lo = self.fractions <= 0.2
        slope = 0.0
        if lo.sum() >= 2:
            slope = float(np.polyfit(self.fractions[lo],
                                     d.mean(axis=0)[lo], 1)[0])
        flipped = (self.curves.max(axis=1) > threshold) & (self.baseline <= threshold)
        return {
            "operator": self.operator,
            "leakage_auc": leak_auc,
            "leakage_delta_auc": float(
                trapezoid(d.mean(axis=0), self.fractions)
                / max(self.fractions[-1] - self.fractions[0], 1e-12)
            ),
            "leakage_slope": slope,
            "flip_rate": float(flipped.mean()),
            "baseline_pfake": float(self.baseline.mean()),
            "max_pfake": float(self.curves.mean(axis=0).max()),
            "n": int(self.curves.shape[0]),
            **self.meta,
        }


@torch.no_grad()
def measure_leakage(detector: Detector, real_images: torch.Tensor,
                    operator: str = "zero", n_steps: int = 21,
                    max_fraction: float = 1.0, n_repeats: int = 3,
                    donor_bank: Optional[DonorBank] = None,
                    batch_size: int = 32, amp: bool = True,
                    seed: int = 1337, meta: Optional[Dict] = None) -> LeakageResult:
    """Run the null test for one (detector, operator) pair on real images.

    ``real_images``: (N, 3, H, W) in [0,1], label == 0 rows only. Passing fakes
    here would be meaningless: the question is whether the operator *creates*
    fakeness, so the input must be genuinely real.
    """
    device = next(detector.parameters()).device
    use_amp = amp and device.type == "cuda"
    N, _, H, _ = real_images.shape
    fractions = np.linspace(0.0, max_fraction, n_steps)

    baseline = np.zeros(N, dtype=np.float32)
    for i in range(0, N, batch_size):
        chunk = real_images[i : i + batch_size].to(device)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.softmax(detector(chunk), dim=1)[:, 1]
        else:
            p = torch.softmax(detector(chunk), dim=1)[:, 1]
        baseline[i : i + chunk.shape[0]] = p.float().cpu().numpy()

    acc = np.zeros((N, n_steps), dtype=np.float64)
    for r in range(n_repeats):
        masks = random_masks(N, H, fractions, seed=seed + r)
        for i in range(N):
            x = real_images[i : i + 1].to(device)
            # Build every masking level in ONE operator call. Operators like
            # road_linear run a 40-iteration relaxation, so calling them once
            # per step instead of once per image was the dominant cost.
            m_all = masks[i].to(device).unsqueeze(1)          # (n_steps,1,H,H)
            x_rep = x.expand(n_steps, -1, -1, -1).contiguous()
            batch = apply_operator(operator, x_rep, m_all, donor_bank=donor_bank)
            batch = torch.where(
                (m_all.flatten(1).sum(1) > 0).view(-1, 1, 1, 1), batch, x_rep
            )
            probs = []
            for k in range(0, batch.shape[0], batch_size):
                ch = batch[k : k + batch_size]
                if use_amp:
                    with torch.autocast("cuda", dtype=torch.float16):
                        p = torch.softmax(detector(ch), dim=1)[:, 1]
                else:
                    p = torch.softmax(detector(ch), dim=1)[:, 1]
                probs.append(p.float().cpu())
            acc[i] += torch.cat(probs).numpy()
    curves = (acc / n_repeats).astype(np.float32)

    return LeakageResult(operator, fractions, curves, baseline,
                         meta=meta or {})


def leakage_table(detector: Detector, real_images: torch.Tensor,
                  operators: Sequence[str], donor_bank: Optional[DonorBank] = None,
                  **kw) -> List[Dict[str, float]]:
    """Sweep the operator set and return one summary row per operator."""
    rows = []
    for op in operators:
        res = measure_leakage(detector, real_images, operator=op,
                              donor_bank=donor_bank, **kw)
        rows.append(res.summary())
        print("  [leakage] {:>16s}  auc={:.4f}  slope={:+.4f}  flip={:.3f}".format(
            op, rows[-1]["leakage_auc"], rows[-1]["leakage_slope"],
            rows[-1]["flip_rate"]))
    return rows


def rank_agreement(scores_a: Dict[str, float], scores_b: Dict[str, float],
                   higher_is_better: bool = False) -> Dict[str, float]:
    """Do two operators rank explanation methods the same way?

    The headline test for the operator-bias claim: if Spearman rho between the
    method ranking under a leaky operator and under an artifact-neutral one is
    low, then published conclusions of the form "method X is the most faithful"
    are statements about the operator, not about X.
    """
    from scipy import stats

    keys = sorted(set(scores_a) & set(scores_b))
    if len(keys) < 3:
        return {"spearman": float("nan"), "kendall": float("nan"), "n": len(keys)}
    a = np.array([scores_a[k] for k in keys])
    b = np.array([scores_b[k] for k in keys])
    if not higher_is_better:
        a, b = -a, -b
    rho, p_rho = stats.spearmanr(a, b)
    tau, p_tau = stats.kendalltau(a, b)
    top_a = keys[int(np.argmax(a))]
    top_b = keys[int(np.argmax(b))]
    return {
        "spearman": float(rho), "spearman_p": float(p_rho),
        "kendall": float(tau), "kendall_p": float(p_tau),
        "top_method_a": top_a, "top_method_b": top_b,
        "top_method_agrees": bool(top_a == top_b),
        "methods": keys, "n": len(keys),
    }
