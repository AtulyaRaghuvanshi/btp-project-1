"""Saliency methods, implemented against the uniform ``Detector`` interface.

Method set is chosen to (a) reproduce the Tsigos et al. MAD'24 comparison
(Grad-CAM++, RISE, LIME, SHAP-ish, plus gradients) and (b) extend it to the
architectures that paper did not cover -- plain ViT and CLIP-ViT, which need
attention-based and reshape-aware variants.

Every method is written so it works for CNN *and* transformer backbones, since
the whole point is a like-for-like comparison across architectures.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import Explainer, normalize_map, register_explainer


def _get_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    mod = root
    for part in path.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def _cam_target_module(detector) -> torch.nn.Module:
    """Resolve the layer Grad-CAM should hook for this backbone."""
    spec = detector.spec
    inner = detector.backbone.visual if spec.kind == "clip" else detector.backbone.model
    if spec.cam_layer:
        try:
            return _get_module(inner, spec.cam_layer)
        except (AttributeError, IndexError, KeyError):
            pass
    # Fallback: last module that produces a 4-D or 3-D activation.
    candidates = [m for m in inner.modules()
                  if isinstance(m, (torch.nn.Conv2d, torch.nn.LayerNorm))]
    if not candidates:
        raise RuntimeError("No hookable layer found for " + spec.name)
    return candidates[-1]


def _reshape_tokens(act: torch.Tensor) -> torch.Tensor:
    """(B, T, C) transformer tokens -> (B, C, h, w), dropping the CLS token."""
    B, T, C = act.shape
    n = T - 1
    side = int(round(n ** 0.5))
    if side * side != n:            # some towers have no CLS token
        side = int(round(T ** 0.5))
        if side * side != T:
            raise RuntimeError("Cannot reshape {} tokens to a grid".format(T))
        return act.permute(0, 2, 1).reshape(B, C, side, side)
    return act[:, 1:, :].permute(0, 2, 1).reshape(B, C, side, side)


class _HookedCAM(Explainer):
    """Shared plumbing for Grad-CAM family methods."""

    cost_per_image = 2

    def __init__(self, detector, **kw):
        super().__init__(detector, **kw)
        self.layer = _cam_target_module(detector)
        self._act: Optional[torch.Tensor] = None
        self._grad: Optional[torch.Tensor] = None
        self.layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, _m, _i, out):
        self._act = out
        if out.requires_grad:
            out.register_hook(self._bwd_hook)

    def _bwd_hook(self, grad):
        self._grad = grad

    def _run(self, x01: torch.Tensor, target: int):
        self._act = self._grad = None
        with torch.enable_grad():
            x = x01.clone().requires_grad_(True)
            logits = self.detector(x)
            score = logits[:, target].sum()
            self.detector.zero_grad(set_to_none=True)
            score.backward()
        act, grad = self._act, self._grad
        if act is None or grad is None:
            raise RuntimeError("CAM hooks did not fire for " + self.detector.spec.name)
        if act.dim() == 3:
            act, grad = _reshape_tokens(act), _reshape_tokens(grad)
        return act.detach(), grad.detach()


@register_explainer("gradcam")
class GradCAM(_HookedCAM):
    def _explain(self, x01, target):
        act, grad = self._run(x01, target)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        return F.relu((weights * act).sum(dim=1))


@register_explainer("gradcampp")
class GradCAMPlusPlus(_HookedCAM):
    """Grad-CAM++ (Chattopadhay et al.) -- the Tsigos et al. baseline."""

    def _explain(self, x01, target):
        act, grad = self._run(x01, target)
        g2, g3 = grad.pow(2), grad.pow(3)
        denom = 2.0 * g2 + (act * g3).sum(dim=(2, 3), keepdim=True)
        alpha = g2 / denom.clamp_min(1e-8)
        weights = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        return F.relu((weights * act).sum(dim=1))


@register_explainer("inputgrad")
class InputGradient(Explainer):
    """Gradient x Input -- cheapest possible faithfulness reference point."""

    cost_per_image = 2

    def _explain(self, x01, target):
        with torch.enable_grad():
            x = x01.clone().requires_grad_(True)
            logits = self.detector(x)
            logits[:, target].sum().backward()
            grad = x.grad
        return (grad * x01).abs().sum(dim=1)


@register_explainer("intgrad")
class IntegratedGradients(Explainer):
    """Integrated Gradients against a black baseline.

    Note the irony this project is about: IG's baseline choice is the same
    decision as the deletion operator's fill choice, and for a forgery detector
    a black baseline is not a neutral "absence of evidence".
    """

    cost_per_image = 32

    def __init__(self, detector, steps: int = 32, **kw):
        super().__init__(detector, **kw)
        self.steps = steps

    def _explain(self, x01, target):
        base = torch.zeros_like(x01)
        total = torch.zeros_like(x01)
        for a in torch.linspace(1.0 / self.steps, 1.0, self.steps):
            with torch.enable_grad():
                pt = (base + a * (x01 - base)).requires_grad_(True)
                self.detector(pt)[:, target].sum().backward()
                total = total + pt.grad
        return ((x01 - base) * total / self.steps).abs().sum(dim=1)


@register_explainer("rise")
class RISE(Explainer):
    """RISE (Petsiuk et al.): random masking, weight maps by the score.

    The dominant cost in the whole pipeline -- ``n_masks`` forwards per image.
    ``n_masks`` is the knob to turn when a run must fit in an evening.
    """

    def __init__(self, detector, n_masks: int = 2000, s: int = 8, p1: float = 0.5,
                 batch_size: int = 64, seed: int = 1337, **kw):
        super().__init__(detector, **kw)
        self.n_masks, self.s, self.p1 = n_masks, s, p1
        self.batch_size = batch_size
        self.cost_per_image = n_masks
        self.seed = seed
        self._masks: Optional[torch.Tensor] = None

    def _make_masks(self, size: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        cell = int(np.ceil(size / self.s))
        up = (self.s + 1) * cell
        grid = (torch.rand(self.n_masks, 1, self.s, self.s, generator=g) < self.p1)
        grid = grid.float()
        big = F.interpolate(grid, size=(up, up), mode="bilinear", align_corners=False)
        out = torch.empty(self.n_masks, 1, size, size)
        offs = torch.randint(0, cell, (self.n_masks, 2), generator=g)
        for i in range(self.n_masks):
            oy, ox = int(offs[i, 0]), int(offs[i, 1])
            out[i] = big[i, :, oy : oy + size, ox : ox + size]
        return out

    @torch.no_grad()
    def _explain(self, x01, target):
        size = x01.shape[-1]
        if self._masks is None or self._masks.shape[-1] != size:
            self._masks = self._make_masks(size)      # kept on CPU
        # 2000 masks at 224px is ~400 MB; on a 4 GB card that must not live on
        # the GPU alongside the model, so chunks are streamed across instead.
        sal = torch.zeros(x01.shape[0], size, size, device=self.device)
        for b in range(x01.shape[0]):
            acc = torch.zeros(size, size, device=self.device)
            for i in range(0, self.n_masks, self.batch_size):
                mb = self._masks[i : i + self.batch_size].to(self.device,
                                                             non_blocking=True)
                logits = self.detector(x01[b : b + 1] * mb)
                sc = torch.softmax(logits, dim=1)[:, target].view(-1, 1, 1)
                acc += (sc * mb.squeeze(1)).sum(0)
            sal[b] = acc / (self.n_masks * self.p1)
        return sal


@register_explainer("lime")
class LIME(Explainer):
    """LIME over SLIC superpixels -- the method Tsigos et al. found most faithful.

    Re-testing that verdict under artifact-neutral operators is one of this
    project's headline experiments: LIME's own perturbation (grey-out
    superpixels) is exactly the operator family we argue is biased for forgery
    detectors, so it may be winning on a technicality.
    """

    def __init__(self, detector, n_samples: int = 1000, n_segments: int = 64,
                 batch_size: int = 64, seed: int = 1337, fill: str = "mean", **kw):
        super().__init__(detector, **kw)
        self.n_samples, self.n_segments = n_samples, n_segments
        self.batch_size, self.seed, self.fill = batch_size, seed, fill
        self.cost_per_image = n_samples

    def _grid_segments(self, H: int, W: int) -> "np.ndarray":
        """Regular square grid, used when SLIC fails to find structure."""
        side = max(2, int(round(self.n_segments ** 0.5)))
        gy = np.minimum((np.arange(H) * side) // H, side - 1)
        gx = np.minimum((np.arange(W) * side) // W, side - 1)
        return (gy[:, None] * side + gx[None, :]).astype(np.int64)

    def _grid_segments(self, H: int, W: int) -> np.ndarray:
        """Regular square tiling with roughly ``n_segments`` cells."""
        side = max(2, int(round(self.n_segments ** 0.5)))
        ys = np.minimum(np.arange(H) * side // H, side - 1)
        xs = np.minimum(np.arange(W) * side // W, side - 1)
        return (ys[:, None] * side + xs[None, :]).astype(np.int64)

    @torch.no_grad()
    def _explain(self, x01, target):
        from skimage.segmentation import slic
        from sklearn.linear_model import Ridge

        rng = np.random.default_rng(self.seed)
        B, _, H, W = x01.shape
        out = torch.zeros(B, H, W, device=self.device)
        for b in range(B):
            img = x01[b].permute(1, 2, 0).cpu().numpy()
            segs = slic(img, n_segments=self.n_segments, compactness=10.0,
                        start_label=0, channel_axis=2)
            n_seg = int(segs.max()) + 1
            if n_seg < 4:
                # SLIC collapses to one or two regions on low-structure images
                # (flat sky, heavy noise, some synthetic textures). The Ridge
                # fit would then return a near-constant map, which normalises
                # to zero and scores as "unfaithful" when it is really
                # "unmeasurable". Fall back to a regular grid instead.
                segs = self._grid_segments(H, W)
                n_seg = int(segs.max()) + 1
            seg_t = torch.from_numpy(segs).to(self.device)
            z = rng.binomial(1, 0.5, size=(self.n_samples, n_seg)).astype(np.float32)
            z[0] = 1.0
            fill_val = x01[b].mean(dim=(1, 2)).view(3, 1, 1)
            scores = np.zeros(self.n_samples, dtype=np.float32)
            for i in range(0, self.n_samples, self.batch_size):
                zb = torch.from_numpy(z[i : i + self.batch_size]).to(self.device)
                keep = zb[:, seg_t]                      # (b, H, W)
                keep = keep.unsqueeze(1)
                inp = x01[b : b + 1] * keep + fill_val * (1 - keep)
                logits = self.detector(inp)
                p = torch.softmax(logits, dim=1)[:, target]
                scores[i : i + zb.shape[0]] = p.cpu().numpy()
            dist = np.sqrt(((1 - z) ** 2).mean(axis=1))
            weights = np.exp(-(dist ** 2) / 0.25)
            reg = Ridge(alpha=1.0).fit(z, scores, sample_weight=weights)
            coef = torch.from_numpy(reg.coef_.astype(np.float32)).to(self.device)
            out[b] = coef[seg_t]
        return out


def _unfuse_attention(inner: torch.nn.Module) -> int:
    """Force explicit-softmax attention and expose the weights.

    Both timm and open_clip default to fused scaled-dot-product attention,
    which never materialises an attention matrix. A naive forward hook
    therefore captures *nothing* and a rollout implementation silently
    degrades to something that is not rollout at all.

    Two backbone families, two fixes:

    * **timm ViT** -- set ``fused_attn = False`` on each ``Attention`` module.
      Its slow path computes ``attn = softmax(q @ k^T)`` and passes it through
      ``attn_drop``, so a hook on ``attn_drop`` sees the real matrix.
    * **open_clip** -- blocks call ``nn.MultiheadAttention`` with
      ``need_weights=False``. Wrap ``forward`` to flip that flag so the module
      returns per-layer weights. The flip is skipped while the module's
      ``_xgf_capture`` attribute is False (see ``AttentionRollout``).

    Returns the number of attention modules successfully instrumented.
    """
    n = 0
    for m in inner.modules():
        if hasattr(m, "fused_attn") and hasattr(m, "attn_drop"):
            m.fused_attn = False
            n += 1
        elif isinstance(m, torch.nn.MultiheadAttention):
            if not getattr(m, "_xgf_patched", False):
                orig = m.forward

                def fwd(*a, _orig=orig, _m=m, **kw):
                    if getattr(_m, "_xgf_capture", True):
                        kw["need_weights"] = True
                        kw["average_attn_weights"] = True
                    return _orig(*a, **kw)

                m.forward = fwd
                m._xgf_patched = True
            n += 1
    return n


@register_explainer("attention_rollout")
class AttentionRollout(Explainer):
    """Attention rollout (Abnar & Zuidema) for ViT / CLIP-ViT detectors.

    Included because "just look at the attention" is the standard intrinsic
    explanation claim for transformer detectors (ISTVT, M2TR). Putting it on
    the same faithfulness footing as post-hoc maps is part of the point -- so
    it has to be *actual* rollout. If the real attention weights cannot be
    captured this class raises rather than quietly substituting a proxy;
    use the separate ``token_norm`` explainer if you want that proxy, under
    its own honest name.

    The detector is shared with every other explainer and with the
    perturbation curves, so instrumentation is live only inside ``_explain``.
    Outside it the backbone runs its default fused attention and nothing is
    captured: otherwise every later forward would append 12 attention maps
    (an OOM within seconds on the GPU), and the curves' numerics would depend
    on whether this explainer happened to be built in the current process --
    which a resumed run does not reproduce.
    """

    cost_per_image = 1

    def __init__(self, detector, discard_ratio: float = 0.0, **kw):
        super().__init__(detector, **kw)
        if detector.spec.kind == "cnn":
            raise ValueError("attention_rollout requires a transformer backbone")
        self.discard_ratio = discard_ratio
        self._attns = []
        self._capturing = False
        inner = (detector.backbone.visual if detector.spec.kind == "clip"
                 else detector.backbone.model)
        self._timm_attn = [m for m in inner.modules()
                           if hasattr(m, "fused_attn") and hasattr(m, "attn_drop")]
        self._mha = [m for m in inner.modules()
                     if isinstance(m, torch.nn.MultiheadAttention)]
        self._fused_default = [m.fused_attn for m in self._timm_attn]
        self._n_patched = _unfuse_attention(inner)
        self._set_capture(False)
        for m in self._timm_attn:
            m.attn_drop.register_forward_hook(self._drop_hook)
        for m in self._mha:
            m.register_forward_hook(self._mha_hook)

    def _set_capture(self, on: bool) -> None:
        self._capturing = on
        for m, fused in zip(self._timm_attn, self._fused_default):
            m.fused_attn = False if on else fused
        for m in self._mha:
            m._xgf_capture = on

    def _drop_hook(self, _m, _i, out):
        if self._capturing and torch.is_tensor(out) and out.dim() == 4:
            self._attns.append(out.detach())

    def _mha_hook(self, _m, _i, out):
        if (self._capturing and isinstance(out, tuple) and len(out) > 1
                and torch.is_tensor(out[1])):
            a = out[1]
            self._attns.append((a if a.dim() == 4 else a.unsqueeze(1)).detach())

    @torch.no_grad()
    def _explain(self, x01, target):
        self._attns = []
        self._set_capture(True)
        try:
            self.detector(x01)
        finally:
            self._set_capture(False)
        if not self._attns:
            raise RuntimeError(
                "attention_rollout captured no attention weights for backbone "
                "'{}' ({} modules instrumented). Do not silently substitute a "
                "proxy -- use the 'token_norm' explainer instead.".format(
                    self.detector.backbone_name, self._n_patched)
            )
        B = x01.shape[0]
        result = None
        attns, self._attns = self._attns, []      # don't pin them on the GPU
        for a in attns:
            a = a.mean(dim=1)                                  # average heads
            if self.discard_ratio > 0:                         # drop weak edges
                k = int(a.shape[-1] * self.discard_ratio)
                if k > 0:
                    idx = a.topk(k, dim=-1, largest=False).indices
                    a = a.scatter(-1, idx, 0.0)
            a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            result = a if result is None else torch.bmm(a, result)
        cls_attn = result[:, 0, 1:]
        side = int(round(cls_attn.shape[-1] ** 0.5))
        if side * side != cls_attn.shape[-1]:                  # no CLS token
            cls_attn = result.mean(dim=1)
            side = int(round(cls_attn.shape[-1] ** 0.5))
        return cls_attn[:, : side * side].reshape(B, side, side)


@register_explainer("token_norm")
class TokenNorm(Explainer):
    """Patch-token L2 norm at the final block -- a cheap intrinsic proxy.

    Not attention rollout, and deliberately registered under its own name so
    results can never be mislabelled as rollout. Works on any transformer
    backbone, including ones whose attention is fused and unhookable.
    """

    cost_per_image = 1

    def __init__(self, detector, **kw):
        super().__init__(detector, **kw)
        if detector.spec.kind == "cnn":
            raise ValueError("token_norm requires a transformer backbone")
        inner = (detector.backbone.visual if detector.spec.kind == "clip"
                 else detector.backbone.model)
        self._ln = [m for m in inner.modules()
                    if isinstance(m, torch.nn.LayerNorm)][-1]
        self._act = None
        self._ln.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self._act = out

    @torch.no_grad()
    def _explain(self, x01, target):
        self._act = None
        self.detector(x01)
        a = self._act
        if a is None or a.dim() != 3:
            raise RuntimeError("token_norm: unexpected activation shape")
        toks = a[:, 1:, :].norm(dim=-1)
        side = int(round(toks.shape[-1] ** 0.5))
        if side * side != toks.shape[-1]:
            toks = a.norm(dim=-1)
            side = int(round(toks.shape[-1] ** 0.5))
        return toks[:, : side * side].reshape(a.shape[0], side, side)


@register_explainer("random")
class RandomBaseline(Explainer):
    """A smooth random map. The floor every real method must beat.

    Critical for this project: under a leaky operator a *random* map can post a
    healthy deletion score, because the score is dominated by the fill's own
    artifact. Reporting it is how the bias becomes visible.
    """

    cost_per_image = 0

    def __init__(self, detector, seed: int = 1337, **kw):
        super().__init__(detector, **kw)
        self.g = torch.Generator().manual_seed(seed)

    def reseed(self, seed: int) -> None:
        self.g.manual_seed(seed)

    def _explain(self, x01, target):
        B, _, H, W = x01.shape
        low = torch.rand(B, 1, 14, 14, generator=self.g)
        up = F.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        return up.squeeze(1).to(x01.device)
