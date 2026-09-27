"""Torch datasets that read the manifest CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# CLIP normalisation. We use it for every backbone so that a single perturbation
# operator (which acts in [0,1] pixel space) composes with all of them.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_rgb(path: str | Path, size: int) -> np.ndarray:
    """Load an image as float32 HWC in [0, 1], resized to ``size``x``size``."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def load_mask(path: str | Path, size: int) -> np.ndarray:
    """Load a binary manipulation mask as float32 HW in {0, 1}."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.NEAREST)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


class ManifestDataset(Dataset):
    """Yields images in **unnormalised [0,1] CHW** form.

    Normalisation happens inside the model wrapper, not here. That is
    deliberate: every perturbation operator works on raw pixels, so the
    perturb -> normalise -> forward order is identical for all backbones and
    the leakage measurements are comparable across them.
    """

    def __init__(self, df: pd.DataFrame, image_size: int = 224,
                 with_mask: bool = False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.with_mask = with_mask

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = load_rgb(row["path"], self.image_size)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        out = {
            "image": x,
            "label": int(row["label"]),
            "index": idx,
            "generator": str(row["generator"]),
            "family": str(row["family"]),
        }
        if self.with_mask:
            mp = str(row.get("mask_path", "") or "")
            if mp and Path(mp).exists():
                m = load_mask(mp, self.image_size)
                out["mask"] = torch.from_numpy(m)
                out["has_mask"] = 1
            else:
                out["mask"] = torch.zeros(self.image_size, self.image_size)
                out["has_mask"] = 0
        return out


def make_loader(df: pd.DataFrame, image_size: int = 224, batch_size: int = 32,
                num_workers: int = 0, shuffle: bool = False,
                with_mask: bool = False) -> DataLoader:
    ds = ManifestDataset(df, image_size=image_size, with_mask=with_mask)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def subset(df: pd.DataFrame, families: Optional[Sequence[str]] = None,
           splits: Optional[Sequence[str]] = None,
           generators: Optional[Sequence[str]] = None,
           label: Optional[int] = None,
           n: Optional[int] = None, seed: int = 1337) -> pd.DataFrame:
    """Filter the manifest, then optionally take a balanced random ``n``.

    When ``families`` is given, rows with ``family == "real"`` are kept too --
    a detector needs negatives regardless of which fake family is in play.
    """
    out = df
    if families is not None:
        keep = set(families) | {"real"}
        out = out[out["family"].isin(keep)]
    if generators is not None:
        keep = set(generators) | {"real"}
        out = out[out["generator"].isin(keep)]
    if splits is not None:
        out = out[out["split"].isin(set(splits))]
    if label is not None:
        out = out[out["label"] == label]
    if n is not None and len(out) > n:
        # Balance across labels when both are present.
        if out["label"].nunique() > 1:
            per = max(1, n // 2)
            parts = [
                g.sample(min(per, len(g)), random_state=seed)
                for _, g in out.groupby("label")
            ]
            out = pd.concat(parts)
        else:
            out = out.sample(n, random_state=seed)
    return out.reset_index(drop=True)
