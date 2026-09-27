"""Unit + smoke tests. Run: python -m pytest tests/ -v"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from xgf.metrics.faithfulness import (
    CurveResult, _auc, _rank_masks, corrected_faithfulness,
)
from xgf.metrics.leakage import LeakageResult, random_masks, rank_agreement
from xgf.metrics.localization import mask_auc, mask_iou, pointing_game
from xgf.perturb.operators import (
    ARTIFACT_INJECTING, ARTIFACT_NEUTRAL, DonorBank, apply_operator,
    list_operators,
)


# --------------------------------------------------------------------------- #
# Perturbation operators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("op", ARTIFACT_INJECTING + ARTIFACT_NEUTRAL)
def test_operator_shape_and_range(op):
    x = torch.rand(2, 3, 32, 32)
    m = torch.zeros(2, 1, 32, 32)
    m[:, :, 8:24, 8:24] = 1.0
    bank = DonorBank(torch.rand(4, 3, 32, 32))
    out = apply_operator(op, x, m, donor_bank=bank)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.min() >= -1e-4 and out.max() <= 1.0 + 1e-4, op


@pytest.mark.parametrize("op", ARTIFACT_INJECTING + ARTIFACT_NEUTRAL)
def test_operator_leaves_unmasked_pixels_untouched(op):
    """The whole subtraction in corrected_faithfulness assumes this."""
    x = torch.rand(1, 3, 32, 32)
    m = torch.zeros(1, 1, 32, 32)
    m[:, :, :16, :] = 1.0
    bank = DonorBank(torch.rand(4, 3, 32, 32))
    out = apply_operator(op, x, m, donor_bank=bank)
    torch.testing.assert_close(out[:, :, 16:, :], x[:, :, 16:, :],
                               rtol=1e-4, atol=1e-4)


def test_empty_mask_is_identity():
    x = torch.rand(1, 3, 16, 16)
    m = torch.zeros(1, 1, 16, 16)
    for op in ["zero", "gray", "mean", "spectral_match"]:
        torch.testing.assert_close(apply_operator(op, x, m), x,
                                   rtol=1e-4, atol=1e-4)


def test_real_transplant_requires_donor():
    x = torch.rand(1, 3, 16, 16)
    m = torch.ones(1, 1, 16, 16)
    with pytest.raises(ValueError):
        apply_operator("real_transplant", x, m)


def test_spectral_match_preserves_brightness():
    """The fill should not shift global luminance, or leakage would just be
    measuring a brightness change."""
    x = torch.rand(2, 3, 64, 64) * 0.5 + 0.25
    m = torch.ones(2, 1, 64, 64)
    out = apply_operator("spectral_match", x, m)
    assert abs(float(out.mean() - x.mean())) < 0.1


def test_registry_complete():
    names = set(list_operators())
    assert set(ARTIFACT_INJECTING + ARTIFACT_NEUTRAL) <= names


# --------------------------------------------------------------------------- #
# Mask ranking
# --------------------------------------------------------------------------- #
def test_rank_masks_area_matches_fraction():
    sal = torch.rand(20, 20)
    fr = [0.0, 0.25, 0.5, 1.0]
    m = _rank_masks(sal, fr)
    assert m.shape == (4, 1, 20, 20)
    for i, f in enumerate(fr):
        assert abs(float(m[i].sum()) - round(f * 400)) <= 1


def test_rank_masks_are_nested():
    """Deletion curves assume cumulative removal."""
    sal = torch.rand(16, 16)
    m = _rank_masks(sal, [0.1, 0.3, 0.6])
    assert ((m[1] - m[0]) >= 0).all()
    assert ((m[2] - m[1]) >= 0).all()


def test_rank_masks_picks_the_most_salient():
    sal = torch.zeros(10, 10)
    sal[3, 4] = 1.0
    m = _rank_masks(sal, [0.01])
    assert m[0, 0, 3, 4] == 1.0


def test_random_masks_fractions():
    m = random_masks(3, 16, [0.0, 0.5, 1.0], seed=0)
    assert m.shape == (3, 3, 16, 16)
    assert float(m[0, 0].sum()) == 0
    assert float(m[0, 2].sum()) == 256
    assert abs(float(m[0, 1].sum()) - 128) <= 1


# --------------------------------------------------------------------------- #
# Metric maths
# --------------------------------------------------------------------------- #
def test_auc_of_constant():
    assert _auc(np.full(10, 0.7)) == pytest.approx(0.7, abs=1e-6)


def test_auc_of_linear_ramp():
    assert _auc(np.linspace(0, 1, 11)) == pytest.approx(0.5, abs=1e-6)


def test_corrected_faithfulness_sign():
    """A salient curve that drops faster than random must score positive."""
    fr = np.linspace(0, 1, 11)
    sal = CurveResult(fr, np.tile(np.linspace(0.9, 0.1, 11), (8, 1)),
                      np.full(8, 0.5, dtype=np.float32))
    rnd = CurveResult(fr, np.tile(np.linspace(0.9, 0.8, 11), (8, 1)),
                      np.full(8, 0.85, dtype=np.float32))
    out = corrected_faithfulness(sal, rnd)
    assert out["delta_auc_mean"] > 0
    assert out["auc_salient"] < out["auc_random"]


def test_leakage_summary_on_clean_operator():
    """A perfectly clean operator: P(fake) never moves off baseline."""
    fr = np.linspace(0, 1, 11)
    curves = np.full((10, 11), 0.05, dtype=np.float32)
    res = LeakageResult("ideal", fr, curves, np.full(10, 0.05, dtype=np.float32), {})
    s = res.summary()
    assert s["leakage_delta_auc"] == pytest.approx(0.0, abs=1e-6)
    assert s["flip_rate"] == 0.0


def test_leakage_summary_on_leaky_operator():
    fr = np.linspace(0, 1, 11)
    curves = np.tile(np.linspace(0.05, 0.95, 11), (10, 1)).astype(np.float32)
    res = LeakageResult("leaky", fr, curves, np.full(10, 0.05, dtype=np.float32), {})
    s = res.summary()
    assert s["leakage_delta_auc"] > 0.3
    assert s["flip_rate"] == 1.0
    assert s["leakage_slope"] > 0


def test_rank_agreement_detects_disagreement():
    a = {"m1": 0.1, "m2": 0.2, "m3": 0.3, "m4": 0.4}
    same = rank_agreement(a, a, higher_is_better=True)
    assert same["spearman"] == pytest.approx(1.0)
    assert same["top_method_agrees"]
    rev = {k: -v for k, v in a.items()}
    flipped = rank_agreement(a, rev, higher_is_better=True)
    assert flipped["spearman"] == pytest.approx(-1.0)
    assert not flipped["top_method_agrees"]


# --------------------------------------------------------------------------- #
# Localization
# --------------------------------------------------------------------------- #
def test_pointing_game_hit_and_miss():
    sal = torch.zeros(2, 8, 8)
    mask = torch.zeros(2, 8, 8)
    mask[:, 2:5, 2:5] = 1.0
    sal[0, 3, 3] = 1.0          # inside
    sal[1, 7, 7] = 1.0          # outside
    pg = pointing_game(sal, mask)
    assert pg[0] == 1.0 and pg[1] == 0.0


def test_mask_iou_perfect_and_disjoint():
    mask = torch.zeros(2, 8, 8)
    mask[:, :4, :] = 1.0
    perfect = mask.clone()
    disjoint = torch.zeros(2, 8, 8)
    disjoint[:, 4:, :] = 1.0
    assert mask_iou(perfect, mask)[0] == pytest.approx(1.0)
    assert mask_iou(disjoint, mask)[0] == pytest.approx(0.0)


def test_mask_auc_perfect():
    mask = torch.zeros(1, 8, 8)
    mask[:, :4, :] = 1.0
    assert mask_auc(mask.clone(), mask)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# End-to-end smoke test
# --------------------------------------------------------------------------- #
def test_leakage_end_to_end_tiny():
    """A detector that literally counts dark pixels must show huge `zero`
    leakage and near-zero `real_transplant` leakage. This is the project's
    central claim, reduced to a toy model where the answer is known."""
    import torch.nn as nn

    from xgf.metrics.leakage import measure_leakage

    class DarkCounter(nn.Module):
        """P(fake) rises with the fraction of near-black pixels."""

        def forward(self, x01):
            dark = (x01.mean(dim=1) < 0.15).float().mean(dim=(1, 2))
            logit = (dark - 0.1) * 30.0
            return torch.stack([-logit, logit], dim=1)

        def parameters(self, recurse=True):
            yield torch.nn.Parameter(torch.zeros(1))

    det = DarkCounter()
    real = torch.rand(6, 3, 32, 32) * 0.5 + 0.45      # bright -> "real"
    donors = DonorBank(torch.rand(8, 3, 32, 32) * 0.5 + 0.45)

    leaky = measure_leakage(det, real, operator="zero", n_steps=6, n_repeats=1,
                            amp=False).summary()
    clean = measure_leakage(det, real, operator="real_transplant", n_steps=6,
                            n_repeats=1, donor_bank=donors, amp=False).summary()

    assert leaky["leakage_delta_auc"] > clean["leakage_delta_auc"]
    assert leaky["flip_rate"] >= clean["flip_rate"]


# --------------------------------------------------------------------------- #
# Attention rollout must never silently degrade to a proxy
# --------------------------------------------------------------------------- #
def test_attention_rollout_raises_instead_of_substituting_a_proxy():
    """Regression guard.

    An earlier version fell back to token-norm saliency whenever the attention
    weights could not be hooked (timm and open_clip both fuse attention by
    default). Results would then have been labelled "attention rollout" while
    containing no attention at all -- which would misstate the intrinsic-
    interpretability claim the transformer arm exists to test. The fallback now
    lives in a separately-named `token_norm` explainer and rollout must fail
    loudly.
    """
    import torch.nn as nn

    from xgf.explain.base import build_explainer
    from xgf.explain import methods  # noqa: F401

    class Opaque(nn.Module):
        """Transformer-shaped model exposing no hookable attention."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 8)
            self.norm = nn.LayerNorm(8)
            self.fc = nn.Linear(8, 2)

        def forward(self, x01):
            # (B,3,H,W) -> 17 tokens of width 8: 1 CLS + a 4x4 patch grid.
            t = self.proj(x01.flatten(2).transpose(1, 2)[:, :17, :])
            return self.fc(self.norm(t).mean(1))

    class Spec:
        kind, name, cam_layer = "vit", "opaque", None

    class Det(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = Spec()
            self.backbone = nn.Module()
            self.backbone.model = Opaque()
            self.backbone_name = "opaque"

        def forward(self, x):
            return self.backbone.model(x)

    det = Det().eval()
    x = torch.rand(1, 3, 8, 8)

    expl = build_explainer("attention_rollout", det)
    with pytest.raises(RuntimeError, match="captured no attention weights"):
        expl(x, target=1)

    # The honest proxy is still available, under its own name.
    assert build_explainer("token_norm", det)(x, target=1).shape == (1, 8, 8)
