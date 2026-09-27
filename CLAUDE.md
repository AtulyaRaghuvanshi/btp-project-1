# Project notes

BTP: **perturbation-artifact bias in explanation evaluation for cross-generator
deepfake detection**. Research plan in `docs/PROPOSAL.md`; read it before
changing anything scientific.

## Invariants — breaking these invalidates results

1. **Images stay in `[0,1]` pixel space until inside the model.**
   `ManifestDataset` yields unnormalised CHW tensors; `Detector` owns the
   `Normalize` layer. Every perturbation operator works on raw pixels, so
   `perturb -> normalise -> forward` is identical across backbones. Moving
   normalisation into the dataloader would silently make leakage numbers
   incomparable across architectures.

2. **Backbones are frozen. Always.** Heads are logistic regression fitted on
   cached features (`models/probe.py`). No fine-tuning anywhere — the 4 GB
   laptop GPU cannot do it, and the controlled-intervention design depends on
   head capacity being identical across probe configs.

3. **Splits are assigned within `(generator, label)` groups.** See
   `data/manifest.assign_splits`. If a generator straddles the train/test
   boundary unevenly, the cross-generator transfer matrix leaks.

4. **The random control must traverse the same code path as the salient arm.**
   `corrected_faithfulness` subtracts one from the other; that subtraction is
   only valid if mask geometry, operator and batching are identical. The
   control is implemented by feeding `perturbation_curve` a smooth random
   saliency map, not by a separate routine.

5. **Q3 is answered by single-family vs `loo_*` probes, never by `mixed`.**
   `mixed` trains on every family, so it has no unseen family and its
   generalization gap is `NaN` by construction. The valid contrast scores the
   *same* held-out family with a probe trained on one family (diversity 1) and
   one trained on the rest (diversity 2). `scripts/05` defaults to this.

6. **`attention_rollout` must capture real attention, or fail loudly.**
   timm and open_clip both default to fused SDPA, which materialises no
   attention matrix. `_unfuse_attention` sets `fused_attn=False` (timm) and
   forces `need_weights=True` (open_clip); the explainer raises if it still
   captures nothing. The token-norm proxy lives in a separate `token_norm`
   explainer so a result can never be mislabelled as rollout. Verified: 12
   attention layers captured on both vit_b16 and clip_vitb16.

7. **Never add a generative-inpainting operator as a "neutral" fill.** Its
   exclusion is an argument in the paper (`perturb/operators.py` docstring): the
   filler would be a generator stamping synthesis artifacts into the hole. It
   may be added only as a deliberately-bad arm, labelled injecting.

## Conventions

- New dataset -> one `scan_*` function in `data/manifest.py` returning the
  standard row dict. Nothing downstream changes.
- New explanation method -> subclass `Explainer`, decorate with
  `@register_explainer("name")`, return `(B, H, W)`. Set `cost_per_image` so
  the budget estimator stays honest.
- New fill operator -> decorate with `@register("name")` and add it to
  `ARTIFACT_INJECTING` or `ARTIFACT_NEUTRAL`. Operators must leave unmasked
  pixels bit-identical (there is a test for this).
- Operators are called **once per batch of masking levels**, never per step.
  `road_linear` runs a 40-iteration relaxation; per-step calls were ~3x slower.

## Environment gotchas

- `torch 2.2.x` is ABI-incompatible with `numpy 2.x`. Either pin `numpy<2` or
  upgrade torch. Upgrading is better — the CUDA wheel is needed anyway.
- `np.trapezoid` is numpy >= 2.0 only; use `xgf.utils.compat.trapezoid`.
- Windows: keep `num_workers=0` unless the caller has a `__main__` guard.
- `scripts/00_setup_check.py --benchmark` measures real throughput and prints
  wall-clock estimates. Use it rather than guessing before any long run.

## Commands

```bash
python -m pytest tests/ -q
python scripts/run_all.py --selftest          # full pipeline, synthetic data
python scripts/00_setup_check.py --benchmark
```
