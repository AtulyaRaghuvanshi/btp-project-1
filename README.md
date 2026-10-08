# Measuring the Instrument

**Perturbation-artifact bias in explanation evaluation for cross-generator
deepfake detection.**

BTP codebase. Full research plan and literature positioning: **[docs/PROPOSAL.md](docs/PROPOSAL.md)**.
Dataset acquisition: **[docs/DATA.md](docs/DATA.md)**.

---

## The idea in one paragraph

Perturbation-based faithfulness metrics — deletion, insertion, ROAD,
pixel-flipping — all work by removing the pixels an explanation calls important
and watching the prediction move. "Remove" is not defined by the metric; it is
defined by an **operator** that decides what goes in the hole. For an ImageNet
classifier that is a footnote. For a **forgery detector** it is not, because a
forgery detector *is* an artifact detector: a hard mask boundary is a splice
edge, flat grey has impossible sensor statistics, white noise has a flat power
spectrum no camera produces. The measuring instrument emits the signal it is
trying to measure. This project quantifies that bias, proposes artifact-neutral
operators that avoid it, and then — on a corrected instrument — asks whether
explanation faithfulness survives a shift from one generator family to another.

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/00_setup_check.py --benchmark            # measure YOUR hardware
python scripts/01_build_manifest.py --selftest          # synthetic data, no download
python scripts/02_extract_features.py --manifest manifest_selftest.csv --backbone resnet50
python scripts/03_train_probes.py     --backbone resnet50 --tag selftest
python scripts/04_run_leakage.py      --backbone resnet50 --tag selftest
# probes default to every single-family + loo_* config on disk; see the note below
python scripts/05_run_faithfulness.py --backbone resnet50 --tag selftest \
    --explainers gradcampp gradcam inputgrad random \
    --operators zero gray road_linear real_transplant --n-images 30
python scripts/06_make_figures.py     --backbone resnet50 --tag selftest
```

The `--selftest` path generates synthetic images whose "generator families" have
deliberately distinct frequency signatures. It is **not** a scientific artifact —
it exists so every metric, script and figure can be debugged while the 32 GB
ArtiFact download runs.

Then swap in real data (see [docs/DATA.md](docs/DATA.md)) and drop `--selftest` /
`--tag selftest`.

### Interrupted runs resume

If a run dies part-way (crash, Ctrl+C, the laptop goes to sleep), re-run the
same command and it continues from where it stopped:

| stage | unit committed to disk | where |
|---|---|---|
| 02 features | a shard every `--checkpoint-every` batches | `features/<bb>__<tag>.partial/` |
| 03 probes | each fitted probe | `results/checkpoints/` |
| 04 leakage | each operator | `results/progress/leakage__<bb>__<probe>.jsonl` |
| 05 faithfulness | each (probe, family, explainer, operator) row, plus saliency maps and control curves | `results/progress/faithfulness__<bb>__<tag>/` |

Each unit reseeds from its own identity, so a resumed run produces the same
numbers as an uninterrupted one. Settings that change results (`--n-images`,
`--n-steps`, the manifest, a retrained probe, fp16 vs fp32) are recorded with
the progress. If they differ on resume, the script refuses to continue rather
than mix two experiments; pass `--fresh` to start over. Batch size may be
changed freely on resume. Widening the grid (more explainers, operators,
families or probes) computes only the new cells.

---

## Repository map

```
src/xgf/
├── config.py               generator -> family taxonomy; paths; run config
├── data/
│   ├── manifest.py         one flat CSV over every dataset; add a scan_* fn per source
│   └── datasets.py         torch Dataset yielding UNNORMALISED [0,1] pixels
├── models/
│   ├── backbones.py        frozen ResNet50 / ViT-B/16 / CLIP-ViT (the architecture axis)
│   ├── detector.py         backbone + head as one module: [0,1] pixels -> fake logit
│   └── probe.py            feature caching + logistic-regression probes
├── explain/
│   ├── base.py             Explainer interface -> (B,H,W) saliency in [0,1]
│   └── methods.py          Grad-CAM, Grad-CAM++, IG, IntGrad, RISE, LIME,
│                           attention rollout, token-norm, random
├── perturb/
│   └── operators.py        *** the fill operators: injecting vs neutral ***
└── metrics/
    ├── faithfulness.py     deletion/insertion parameterised BY OPERATOR
    ├── leakage.py          *** the null test: primary contribution ***
    └── localization.py     pointing game / IoU / pixel-AUC vs GT masks
```

### Two design decisions worth knowing

**Pixels stay unnormalised until inside the model.** `ManifestDataset` yields
`[0,1]` tensors and `Detector` owns the normalisation layer. Every perturbation
operator therefore works in plain pixel space, and `perturb -> normalise ->
forward` is identical for all backbones — which is the only way leakage numbers
are comparable across architectures.

**Backbones are frozen; heads are logistic regression on cached features.** This
is what makes the project fit a 4 GB laptop GPU. Probe training is *seconds* on
CPU, so the full train-family × test-family matrix is cheap, and all real compute
goes to the faithfulness evaluation, which is the actual contribution.

---

## The operators

| Operator | Class | What goes in the hole |
|---|---|---|
| `zero` | injecting | black — maximal boundary edge energy |
| `gray` | injecting | mid-grey — the most common literature default |
| `mean` | injecting | per-image channel mean |
| `uniform_noise` | injecting | white noise — flat spectrum, very detectable |
| `gaussian_blur` | injecting | heavy blur — isolates the frequency channel |
| `road_linear` | neutral | ROAD noisy linear imputation from surviving neighbours |
| `spectral_match` | neutral | noise matched to the image's **own** power spectrum |
| `real_transplant` | neutral | co-located pixels from an authentic photograph |

**Deliberately absent: generative (diffusion) inpainting.** It is the standard
fix for the OOD confound in natural-image XAI and it is invalid here — the
filler is itself a generator, so it stamps the hole with exactly the synthesis
artifacts under investigation. Using a diffusion model to evaluate a
diffusion-image detector is measuring a ruler with itself. See
[docs/PROPOSAL.md §2.3](docs/PROPOSAL.md).

---

## Hardware

Run `python scripts/00_setup_check.py --benchmark` — it measures real throughput
on your machine and converts it into per-experiment wall-clock estimates.

- **Training needs no GPU.** Frozen backbones + logistic-regression heads: seconds.
- **Evaluation wants one.** Perturbation curves are forward-pass bound. At the
  measured 8.6 img/s a full-budget run is **~40 days on CPU** vs **~3 days** on a
  4 GB laptop GPU.
- **On CPU only**, use `configs/cpu_budget.yaml`: drop RISE/LIME, `--n-images
  150`, `--n-steps 15`. That lands at ~3 days for a complete result set. The
  primary contribution (leakage) costs ~18 h on CPU by itself.

If `cuda_available` is `False` while `nvidia-smi` works, you have the CPU-only
torch wheel. Update the NVIDIA driver, then:

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**numpy note:** torch 2.2.x is not ABI-compatible with numpy 2.x. Either pin
`numpy<2` or upgrade torch — upgrading is better, and you need to anyway for CUDA.

---

## Tests

```bash
python -m pytest tests/ -v
```

Covers operator shape/range invariants, the mask-ranking logic, metric maths on
synthetic inputs, and an end-to-end smoke test of the leakage pipeline.
