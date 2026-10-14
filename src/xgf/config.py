"""Central configuration. Paths are resolved relative to the repo root."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _p(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


# --- Generator taxonomy -------------------------------------------------------
# The whole project hangs off this grouping: every fake image is tagged with a
# `generator` (the concrete model) and a `family` (its synthesis paradigm).
# Faithfulness is then reported per family, and probes are trained on family
# subsets so we can build a train-family x test-family transfer matrix.
#
# Names are the *normalised* folder names the scanners see (manifest._norm), so
# ArtiFact's `pro_gan` and ForenSynths' `progan` are both listed. Every name
# must appear in exactly one family -- enforced below. A model that does two
# tasks gets one name per task (`glide` vs `glide_inpaint`), never two families.
FAMILIES: Dict[str, List[str]] = {
    "gan": [
        "progan", "pro_gan", "stylegan", "stylegan1", "stylegan2", "stylegan3",
        "biggan", "big_gan", "cyclegan", "cycle_gan", "stargan", "star_gan",
        "gaugan", "gau_gan", "projected_gan", "gansformer", "cips", "sfhq",
        "whichfaceisreal",
        # Diffusion-GAN is a one-step StyleGAN2/ProjectedGAN generator trained
        # with diffusion-noised discriminator inputs; DD-GAN runs a 4-step chain
        # whose every step is an adversarially trained generator. Both are GANs
        # at sampling time, so they sit here despite the names.
        "diffusion_gan", "denoising_diffusion_gan",
    ],
    "diffusion": [
        "stable_diffusion", "sd14", "sd15", "sd21", "sdxl", "glide", "adm",
        "ddpm", "ldm", "latent_diffusion", "vq_diffusion", "vqdm", "palette",
        "dalle2", "dalle3", "midjourney", "guided_diffusion", "wukong",
    ],
    "autoregressive": ["taming_transformer", "dalle_mini", "vqgan", "parti", "llamagen"],
    "inpainting": ["lama", "mat", "generative_inpainting", "glide_inpaint", "sd_inpaint", "cocoglide"],
    # Graphics-pipeline renders (Microsoft Face Synthetics): no learned generator.
    "rendered": ["face_synthetics"],
    "faceswap": ["deepfake", "deepfakes", "faceswap", "faceshifter", "simswap", "inswap", "blendface", "e4s"],
    "reenactment": ["face2face", "neuraltextures", "fomm", "facevid2vid", "wav2lip", "mraa", "one_shot_free"],
    "real": ["real"],
}

# Reverse lookup, built once. Refuses duplicates: with a first-wins lookup, a
# name listed under two families silently lands in whichever comes first, which
# is how stable_diffusion once ended up in "gan".
GENERATOR_TO_FAMILY: Dict[str, str] = {}
for _fam, _gens in FAMILIES.items():
    for _g in _gens:
        if _g in GENERATOR_TO_FAMILY:
            raise ValueError("generator {!r} listed under both {!r} and {!r}".format(
                _g, GENERATOR_TO_FAMILY[_g], _fam))
        GENERATOR_TO_FAMILY[_g] = _fam


@dataclass
class Paths:
    root: Path = REPO_ROOT
    data_raw: Path = field(default_factory=lambda: _p("data", "raw"))
    data_interim: Path = field(default_factory=lambda: _p("data", "interim"))
    manifests: Path = field(default_factory=lambda: _p("data", "manifests"))
    features: Path = field(default_factory=lambda: _p("features"))
    results: Path = field(default_factory=lambda: _p("results"))
    figures: Path = field(default_factory=lambda: _p("results", "figures"))
    # Q1/Q2/Q3 summary tables (xgf.metrics.analysis)
    analysis: Path = field(default_factory=lambda: _p("results", "analysis"))
    checkpoints: Path = field(default_factory=lambda: _p("results", "checkpoints"))
    # Resumable-run state (progress logs, cached saliency). Safe to delete; a
    # deleted entry only means that work is redone on the next run.
    progress: Path = field(default_factory=lambda: _p("results", "progress"))

    def ensure(self) -> None:
        for v in vars(self).values():
            if isinstance(v, Path):
                v.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    seed: int = 1337
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0          # Windows: keep 0 or 2; >0 needs __main__ guard
    device: str = "auto"          # auto | cuda | cpu
    amp: bool = True              # fp16 autocast on CUDA; ignored on CPU
    # Faithfulness evaluation budget
    n_eval_images: int = 500      # images per (detector, family) cell
    n_mask_steps: int = 25        # points on the deletion/insertion curve
    rise_n_masks: int = 2000      # RISE budget; the single biggest cost knob
    rise_s: int = 8
    rise_p1: float = 0.5
    lime_n_samples: int = 1000
    n_random_controls: int = 3    # random-location repeats for the leakage null test

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


PATHS = Paths()


def load_config(path: str | os.PathLike | None = None) -> RunConfig:
    """Load a RunConfig from YAML, falling back to dataclass defaults."""
    cfg = RunConfig()
    if path is None:
        return cfg
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg
