"""Build a single flat manifest CSV over every dataset we use.

The manifest is the contract between data acquisition and everything else:

    path, label, generator, family, source, split, mask_path

``label``     1 = fake/synthetic, 0 = real
``generator`` concrete model name, lowercased (e.g. "stylegan2", "sd14")
``family``    synthesis paradigm from config.FAMILIES (gan/diffusion/...)
``source``    which dataset the row came from (artifact/forensynths/...)
``split``     train | val | test
``mask_path`` ground-truth manipulation mask, or "" when the dataset has none

Adding a dataset means writing one ``scan_*`` function that returns rows.
Nothing downstream needs to change.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import pandas as pd

from ..config import GENERATOR_TO_FAMILY, PATHS

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
COLUMNS = ["path", "label", "generator", "family", "source", "split", "mask_path"]

# ArtiFact folders that hold authentic photographs rather than generated ones.
REAL_SOURCES = {
    "afhq", "coco", "ffhq", "imagenet", "landscape", "lsun", "metfaces", "celebahq",
}


def _iter_images(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            yield p


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _family_for(generator: str, override: Optional[str] = None) -> str:
    if override:
        return override
    return GENERATOR_TO_FAMILY.get(generator, "unknown")


def _row(path, label, generator, family, source, split="", mask_path=""):
    return dict(
        path=str(path),
        label=int(label),
        generator="real" if label == 0 else generator,
        family="real" if label == 0 else family,
        source=source,
        split=split,
        mask_path=mask_path,
    )


# --------------------------------------------------------------------------- #
# Dataset scanners
# --------------------------------------------------------------------------- #
def scan_artifact(root: Path) -> List[dict]:
    """ArtiFact (Kaggle ``awsaf49/artifact-dataset``).

    Layout: ``<root>/<generator_name>/<images...>`` plus a ``metadata.csv``
    per folder whose ``target`` column is 0/1.
    """
    rows: List[dict] = []
    if not root.exists():
        return rows
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generator = _norm(gen_dir.name)
        targets = {}
        meta = gen_dir / "metadata.csv"
        if meta.exists():
            try:
                md = pd.read_csv(meta)
                if {"image_path", "target"}.issubset(md.columns):
                    targets = {
                        Path(str(r.image_path)).name: int(r.target)
                        for r in md.itertuples()
                    }
            except Exception:
                targets = {}
        default_label = 0 if generator in REAL_SOURCES else 1
        family = _family_for(generator)
        for img in _iter_images(gen_dir):
            label = targets.get(img.name, default_label)
            rows.append(_row(img, label, generator, family, "artifact"))
    return rows


def scan_forensynths(root: Path) -> List[dict]:
    """ForenSynths / CNNDetection test set.

    Layout: ``<root>/<generator>/{0_real,1_fake}/<images...>``
    """
    rows: List[dict] = []
    if not root.exists():
        return rows
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generator = _norm(gen_dir.name)
        family = _family_for(generator, "gan")
        for sub, label in (("0_real", 0), ("1_fake", 1)):
            sub_dir = gen_dir / sub
            if not sub_dir.exists():
                continue
            for img in _iter_images(sub_dir):
                rows.append(_row(img, label, generator, family, "forensynths"))
    return rows


def scan_genimage(root: Path) -> List[dict]:
    """GenImage. Layout: ``<root>/<generator>/{train,val}/{nature,ai}/...``"""
    rows: List[dict] = []
    if not root.exists():
        return rows
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generator = _norm(gen_dir.name)
        family = _family_for(generator, "diffusion")
        for split_dir in (d for d in gen_dir.iterdir() if d.is_dir()):
            split = "train" if split_dir.name.lower().startswith("train") else "val"
            for sub, label in (("nature", 0), ("ai", 1)):
                sub_dir = split_dir / sub
                if not sub_dir.exists():
                    continue
                for img in _iter_images(sub_dir):
                    rows.append(
                        _row(img, label, generator, family, "genimage", split)
                    )
    return rows


def scan_cocoglide(root: Path) -> List[dict]:
    """CocoGlide (GRIP-UNINA).

    Diffusion inpainting *with* per-pixel masks -- the only source here that
    supports localization/plausibility metrics alongside faithfulness.
    Layout: ``<root>/{fake,real}/*.png`` and ``<root>/masks/*.png``.
    """
    rows: List[dict] = []
    if not root.exists():
        return rows
    mask_dir = next((d for d in (root / "masks", root / "mask") if d.exists()), None)
    for sub, label, gen, fam in (
        ("fake", 1, "glide_inpaint", "inpainting"),
        ("real", 0, "real", "real"),
    ):
        sub_dir = root / sub
        if not sub_dir.exists():
            continue
        for img in _iter_images(sub_dir):
            mask_path = ""
            if label == 1 and mask_dir is not None:
                cand = mask_dir / (img.stem + ".png")
                if cand.exists():
                    mask_path = str(cand)
            rows.append(
                _row(img, label, gen, fam, "cocoglide", "test", mask_path)
            )
    return rows


def scan_generic(root: Path, source: str, generator: str, label: int,
                 family: Optional[str] = None) -> List[dict]:
    """Escape hatch: treat a flat folder as one ``(generator, label)`` bucket.

    Use for FaceForensics++ / DF40 / Celeb-DF frame dumps, where the folder
    layout is whatever your extraction script produced.
    """
    rows: List[dict] = []
    if not root.exists():
        return rows
    generator = _norm(generator)
    fam = _family_for(generator, family)
    for img in _iter_images(root):
        rows.append(_row(img, label, generator, fam, source))
    return rows


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def assign_splits(df: pd.DataFrame, seed: int = 1337,
                  frac: tuple = (0.7, 0.1, 0.2)) -> pd.DataFrame:
    """Stratified split within each ``(generator, label)`` group.

    Splitting *within* generator matters: if a generator straddled the
    train/test boundary unevenly the cross-generator transfer matrix would
    leak, and the whole experiment would be meaningless.
    """
    df = df.copy()
    df["split"] = df["split"].fillna("").astype(str)
    need = df["split"].str.len() == 0
    if not need.any():
        return df

    assigned = {}
    for (_gen, _lab), grp in df[need].groupby(["generator", "label"], sort=False):
        shuffled = grp.sample(frac=1.0, random_state=seed)
        n = len(shuffled)
        n_tr = int(round(n * frac[0]))
        n_va = int(round(n * frac[1]))
        labels = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        for idx, lab in zip(shuffled.index, labels[:n]):
            assigned[idx] = lab
    df.loc[list(assigned.keys()), "split"] = list(assigned.values())
    return df


def build_manifest(rows: Iterable[dict], seed: int = 1337,
                   out_name: str = "manifest.csv") -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    if df.empty:
        raise RuntimeError(
            "No images found. Download at least one dataset into data/raw/ "
            "first -- see docs/DATA.md"
        )
    df["split"] = df["split"].fillna("").astype(str)
    df["mask_path"] = df["mask_path"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    df = assign_splits(df, seed=seed)

    PATHS.manifests.mkdir(parents=True, exist_ok=True)
    out = PATHS.manifests / out_name
    df.to_csv(out, index=False)
    print("[manifest] {:,} rows -> {}".format(len(df), out))
    print(df.groupby(["family", "label"]).size().to_string())
    return df


def load_manifest(name: str = "manifest.csv") -> pd.DataFrame:
    path = PATHS.manifests / name
    if not path.exists():
        raise FileNotFoundError(
            "{} missing. Run: python scripts/01_build_manifest.py".format(path)
        )
    df = pd.read_csv(path)
    df["mask_path"] = df["mask_path"].fillna("").astype(str)
    df["split"] = df["split"].fillna("").astype(str)
    return df
