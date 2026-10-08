"""Perturbation operators -- the scientific core of this project.

Every perturbation-based faithfulness metric (deletion, insertion, ROAD,
pixel-flipping, comprehensiveness/sufficiency) has the same shape: remove the
pixels an explanation calls important, and watch the prediction move. "Remove"
is not defined by the metric -- it is defined by an *operator* that decides what
to put in the hole.

For an ordinary object classifier the choice of operator is a second-order
detail. For a **forgery detector** it is not, because a forgery detector is an
artifact detector: filling a region with flat grey, blurring it, or pasting in
generated content all create exactly the kind of low-level statistical
discontinuity the detector was trained to fire on. The measurement instrument
emits the signal it is trying to measure.

This module implements operators spanning that spectrum so the bias can be
quantified (see ``xgf.metrics.leakage``) rather than assumed away:

Artifact-INJECTING (the field standard, expected to leak badly)
    ``zero``, ``mean``, ``gray``, ``uniform_noise``, ``gaussian_blur``

Artifact-NEUTRAL candidates (our proposed corrections)
    ``real_transplant``   paste co-located pixels from a *real* photograph, so
                          the fill carries authentic sensor/JPEG statistics
    ``spectral_match``    noise shaped to the image's own local power spectrum,
                          preserving the frequency fingerprint the detector reads
    ``road_linear``       ROAD's noisy linear imputation from surviving
                          neighbours (the strongest published OOD-aware fill)

Deliberately NOT offered: generative (diffusion) inpainting. It is the standard
fix for the OOD-confound problem in natural-image XAI, and it is *invalid here* --
the filler is itself a generator, so it stamps the hole with precisely the
synthesis artifacts under investigation. That argument is a contribution in its
own right; see docs/PROPOSAL.md.

All operators share the signature::

    op(x, mask, **kw) -> x_perturbed

where ``x`` is (B, 3, H, W) float in [0, 1] and ``mask`` is (B, 1, H, W) or
(B, H, W) float in [0, 1], with **1 = replace this pixel**.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

Operator = Callable[..., torch.Tensor]
_REGISTRY: Dict[str, Operator] = {}


def register(name: str) -> Callable[[Operator], Operator]:
    def deco(fn: Operator) -> Operator:
        _REGISTRY[name] = fn
        fn.op_name = name  # type: ignore[attr-defined]
        return fn
    return deco


def get_operator(name: str) -> Operator:
    if name not in _REGISTRY:
        raise KeyError(
            "Unknown operator '{}'. Available: {}".format(name, list_operators())
        )
    return _REGISTRY[name]


def list_operators() -> list:
    return sorted(_REGISTRY)


# Operators grouped by whether we expect them to inject forensic artifacts.
ARTIFACT_INJECTING = ["zero", "gray", "mean", "uniform_noise", "gaussian_blur"]
ARTIFACT_NEUTRAL = ["road_linear", "spectral_match", "real_transplant"]


def _safe_gen(generator: Optional[torch.Generator],
              device: torch.device) -> Optional[torch.Generator]:
    """Drop the generator when its device does not match the target device.

    torch raises if a CPU generator is handed to a CUDA allocation. Randomness
    here only needs to be reproducible on CPU runs (where all the debugging
    happens), so silently falling back to the global RNG on CUDA is fine.
    """
    if generator is None:
        return None
    return generator if generator.device.type == device.type else None


def _as_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast a mask to (B, 1, H, W) matching ``x``'s dtype/device."""
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.to(dtype=x.dtype, device=x.device)


def _blend(x: torch.Tensor, fill: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x * (1.0 - mask) + fill * mask


# --------------------------------------------------------------------------- #
# Artifact-injecting operators (the field standard)
# --------------------------------------------------------------------------- #
@register("zero")
def zero_fill(x: torch.Tensor, mask: torch.Tensor, **_) -> torch.Tensor:
    """Replace with black. Maximal edge energy at the mask boundary."""
    m = _as_mask(mask, x)
    return _blend(x, torch.zeros_like(x), m)


@register("gray")
def gray_fill(x: torch.Tensor, mask: torch.Tensor, value: float = 0.5,
              **_) -> torch.Tensor:
    """Replace with mid-grey -- the most common choice in the literature."""
    m = _as_mask(mask, x)
    return _blend(x, torch.full_like(x, value), m)


@register("mean")
def mean_fill(x: torch.Tensor, mask: torch.Tensor, **_) -> torch.Tensor:
    """Replace with each image's own per-channel mean."""
    m = _as_mask(mask, x)
    mu = x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    return _blend(x, mu, m)


@register("uniform_noise")
def uniform_noise_fill(x: torch.Tensor, mask: torch.Tensor,
                       generator: Optional[torch.Generator] = None,
                       **_) -> torch.Tensor:
    """Replace with white uniform noise -- flat spectrum, very detectable."""
    m = _as_mask(mask, x)
    noise = torch.rand(x.shape, device=x.device, dtype=x.dtype,
                       generator=_safe_gen(generator, x.device))
    return _blend(x, noise, m)


@register("gaussian_blur")
def gaussian_blur_fill(x: torch.Tensor, mask: torch.Tensor, sigma: float = 6.0,
                       **_) -> torch.Tensor:
    """Replace with a heavily blurred copy: destroys high-frequency evidence
    while keeping colour, so it isolates the frequency channel."""
    m = _as_mask(mask, x)
    radius = max(1, int(3 * sigma))
    ksize = 2 * radius + 1
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - radius
    kern1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kern1d = kern1d / kern1d.sum()
    c = x.shape[1]
    kx = kern1d.view(1, 1, 1, ksize).expand(c, 1, 1, ksize)
    ky = kern1d.view(1, 1, ksize, 1).expand(c, 1, ksize, 1)
    blurred = F.conv2d(F.pad(x, (radius,) * 2 + (0, 0), mode="reflect"), kx, groups=c)
    blurred = F.conv2d(F.pad(blurred, (0, 0) + (radius,) * 2, mode="reflect"), ky,
                       groups=c)
    return _blend(x, blurred, m)


# --------------------------------------------------------------------------- #
# Artifact-neutral candidates (our corrections)
# --------------------------------------------------------------------------- #
@register("road_linear")
def road_linear_fill(x: torch.Tensor, mask: torch.Tensor, n_iter: int = 40,
                     noise_std: float = 0.01,
                     generator: Optional[torch.Generator] = None,
                     **_) -> torch.Tensor:
    """ROAD-style noisy linear imputation.

    Iteratively relaxes masked pixels toward a weighted average of their
    surviving neighbours (a Jacobi solve of the Laplace equation on the hole),
    then adds a small amount of noise so the fill is not implausibly smooth.
    Reference: Rong et al., "A Consistent and Efficient Evaluation Strategy for
    Attribution Methods" (ICML 2022).
    """
    m = _as_mask(mask, x)
    keep = 1.0 - m
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 0.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=x.dtype, device=x.device,
    ).view(1, 1, 3, 3)

    filled = x * keep
    w = keep.clone()
    kc = kernel.expand(x.shape[1], 1, 3, 3)
    for _ in range(n_iter):
        num = F.conv2d(F.pad(filled, (1, 1, 1, 1), mode="reflect"), kc,
                       groups=x.shape[1])
        den = F.conv2d(F.pad(w, (1, 1, 1, 1), mode="reflect"), kernel)
        avg = num / den.clamp_min(1e-6)
        filled = torch.where(m.expand_as(x) > 0.5, avg, x)
        w = torch.clamp(w + (den > 1e-6).to(w.dtype) * m, max=1.0)
    if noise_std > 0:
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype,
                            generator=_safe_gen(generator, x.device)) * noise_std
        filled = filled + noise * m
    return filled.clamp(0.0, 1.0)


@register("spectral_match")
def spectral_match_fill(x: torch.Tensor, mask: torch.Tensor,
                        generator: Optional[torch.Generator] = None,
                        **_) -> torch.Tensor:
    """Fill with noise whose power spectrum matches the image's own.

    Motivation specific to this project: a large share of synthetic-image
    detectors read the frequency fingerprint (up-sampling peaks, spectral
    roll-off). A fill with the *same* radial power spectrum as the host image
    removes the local content without handing the detector a new, alien
    spectral signature. Phase is randomised; magnitude is preserved.
    """
    m = _as_mask(mask, x)
    B, C, H, W = x.shape
    Xf = torch.fft.rfft2(x, norm="ortho")
    mag = Xf.abs()
    phase = torch.rand(mag.shape, device=x.device, dtype=x.dtype,
                       generator=_safe_gen(generator, x.device)) * (2 * torch.pi)
    surrogate = torch.fft.irfft2(
        torch.polar(mag, phase), s=(H, W), norm="ortho"
    ).to(x.dtype)
    # Re-match first two moments per image/channel so brightness is preserved.
    mu_x = x.mean(dim=(2, 3), keepdim=True)
    sd_x = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    mu_s = surrogate.mean(dim=(2, 3), keepdim=True)
    sd_s = surrogate.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    surrogate = (surrogate - mu_s) / sd_s * sd_x + mu_x
    return _blend(x, surrogate.clamp(0.0, 1.0), m)


@register("real_transplant")
def real_transplant_fill(x: torch.Tensor, mask: torch.Tensor,
                         donor: Optional[torch.Tensor] = None, **_) -> torch.Tensor:
    """Paste co-located pixels from an authentic photograph.

    The strongest artifact-neutral option available: the fill carries genuine
    camera noise, demosaicing and JPEG statistics, so a detector that has
    learned "real pixel statistics" sees real pixel statistics in the hole.
    The residual confound is the seam at the mask boundary, which every
    non-generative operator shares and which the leakage null test measures.

    ``donor`` is a batch of real images, broadcast/cycled to match ``x``.
    Callers should supply donors drawn from the *same* real distribution as the
    dataset's negatives (see ``DonorBank``).
    """
    m = _as_mask(mask, x)
    if donor is None:
        raise ValueError(
            "real_transplant needs `donor` real images; use DonorBank.sample()"
        )
    donor = donor.to(dtype=x.dtype, device=x.device)
    if donor.shape[0] != x.shape[0]:
        reps = (x.shape[0] + donor.shape[0] - 1) // donor.shape[0]
        donor = donor.repeat(reps, 1, 1, 1)[: x.shape[0]]
    if donor.shape[-2:] != x.shape[-2:]:
        donor = F.interpolate(donor, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
    return _blend(x, donor, m)


class DonorBank:
    """Holds a pool of real images for ``real_transplant``.

    Kept resident on CPU and sampled per call; a few hundred 224x224 images is
    ~100 MB, which is fine, and it avoids re-reading from disk inside the
    inner loop.
    """

    def __init__(self, images: torch.Tensor, seed: int = 1337):
        self.images = images.cpu()
        self.g = torch.Generator().manual_seed(seed)

    def reseed(self, seed: int) -> None:
        """Restart the donor sequence, so the donors a curve draws depend on
        its own seed and not on how many curves were computed before it."""
        self.g.manual_seed(seed)

    def sample(self, n: int) -> torch.Tensor:
        idx = torch.randint(0, self.images.shape[0], (n,), generator=self.g)
        return self.images[idx]

    @classmethod
    def from_manifest(cls, df, image_size: int = 224, n: int = 256,
                      seed: int = 1337) -> "DonorBank":
        from ..data.datasets import ManifestDataset, subset

        real = subset(df, label=0, n=n, seed=seed)
        ds = ManifestDataset(real, image_size=image_size)
        imgs = torch.stack([ds[i]["image"] for i in range(len(ds))])
        return cls(imgs, seed=seed)


def apply_operator(name: str, x: torch.Tensor, mask: torch.Tensor,
                   donor_bank: Optional[DonorBank] = None,
                   generator: Optional[torch.Generator] = None,
                   same_donor: bool = True, **kw) -> torch.Tensor:
    """Dispatch by name, wiring up the donor bank when needed.

    ``same_donor=True`` draws a single donor and repeats it across the batch.
    That is what the metrics want: a batch is usually the *same image* at
    increasing masking levels, and letting the donor change between steps would
    inject step-to-step noise into the deletion curve that has nothing to do
    with the explanation.
    """
    op = get_operator(name)
    if name == "real_transplant":
        if donor_bank is None:
            raise ValueError("real_transplant requires a DonorBank")
        n = 1 if same_donor else x.shape[0]
        kw["donor"] = donor_bank.sample(n).to(x.device)
    if name in ("uniform_noise", "spectral_match", "road_linear"):
        kw.setdefault("generator", generator)
    return op(x, mask, **kw)
