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
FAMILIES: Dict[str, List[str]] = {
    "gan": [
        "progan", "stylegan", "stylegan2", "stylegan3", "biggan", "cyclegan",
        "stargan", "gaugan", "projected_gan", "gansformer", "star_gan",
        "denoising_diffusion_gan", "big_gan", "face_synthetics", "generative_inpainting",
        "lama", "mat", "palette", "sfhq", "stable_diffusion", "taming_transformer",
    ],
    "diffusion": [
        "stable_diffusion", "sd14", "sd15", "sd21", "sdxl", "glide", "adm",
        "ldm", "dalle2", "dalle3", "midjourney", "guided_diffusion",
        "latent_diffusion", "vqdm", "wukong", "palette", "cips",
    ],
    "autoregressive": ["taming_transformer", "dalle_mini", "vqgan", "parti", "llamagen"],
    "inpainting": ["lama", "mat", "generative_inpainting", "glide_inpaint", "sd_inpaint", "cocoglide"],
    "faceswap": ["deepfakes", "faceswap", "faceshifter", "simswap", "inswap", "blendface", "e4s"],
    "reenactment": ["face2face", "neuraltextures", "fomm", "facevid2vid", "wav2lip", "mraa", "one_shot_free"],
    "real": ["real"],
}

# Reverse lookup, built once. A generator can legitimately appear in two
# families (e.g. stable_diffusion does both full synthesis and inpainting);
# the manifest builder resolves that from the source dataset, not this map,
# so this reverse map only supplies a *default*.
GENERATOR_TO_FAMILY: Dict[str, str] = {}
for _fam, _gens in FAMILIES.items():
    for _g in _gens:
        GENERATOR_TO_FAMILY.setdefault(_g, _fam)


@dataclass
class Paths:
    root: Path = REPO_ROOT
    data_raw: Path = field(default_factory=lambda: _p("data", "raw"))
    data_interim: Path = field(default_factory=lambda: _p("data", "interim"))
    manifests: Path = field(default_factory=lambda: _p("data", "manifests"))
    features: Path = field(default_factory=lambda: _p("features"))
    results: Path = field(default_factory=lambda: _p("results"))
    figures: Path = field(default_factory=lambda: _p("results", "figures"))
    checkpoints: Path = field(default_factory=lambda: _p("results", "checkpoints"))

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
