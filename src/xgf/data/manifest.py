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


def _family_for(generator: str, override: Optional[str] = None,
                default: str = "unknown") -> str:
    """``override`` always wins; ``default`` applies only to unlisted generators.

    Scanners pass their dataset's dominant family as ``default``, never as
    ``override``: GenImage is mostly diffusion but ships a BigGAN subset.
    """
    if override:
        return override
    return GENERATOR_TO_FAMILY.get(generator, default)


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
def _artifact_subgenerator(generator: str, rel_parts: tuple) -> str:
    """ArtiFact's ``glide`` folder mixes text-to-image and inpainting samples
    (``glide-t2i/``, ``glide/glide/text2img/`` vs ``glide-in/``,
    ``glide/glide/inpainting/``). Inpainted images keep real pixels outside the
    hole, so they are a different paradigm and get their own generator name."""
    if generator == "glide" and (rel_parts[0] == "glide-in" or "inpainting" in rel_parts[:-1]):
        return "glide_inpaint"
    return generator


def scan_artifact(root: Path) -> List[dict]:
    """ArtiFact (Kaggle ``awsaf49/artifact-dataset``).

    Layout: ``<root>/<generator_name>/<images...>`` plus a ``metadata.csv`` per
    folder with ``image_path`` (relative to that folder) and ``target``.

    ``target`` is ArtiFact's *class code*, not a binary label: 0 = real,
    1 = StyleGAN2, 2 = StyleGAN3, 3 = GLIDE, 4 = generative inpainting,
    5 = taming transformer, 6 = every other generator. Anything non-zero is
    fake. ``cycle_gan`` and ``pro_gan`` contain both real and fake images, so
    the lookup has to be per file.
    """
    rows: List[dict] = []
    if not root.exists():
        return rows
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generator = _norm(gen_dir.name)
        by_relpath, by_name = {}, {}
        meta = gen_dir / "metadata.csv"
        if meta.exists():
            try:
                md = pd.read_csv(meta, usecols=["image_path", "target"])
            except Exception as exc:
                print("  [artifact] WARNING: unreadable {} ({}); labelling every "
                      "image in {}/ as {}".format(meta, exc, gen_dir.name,
                                                   "real" if generator in REAL_SOURCES else "fake"))
            else:
                fake = md["target"].astype(int).ne(0).astype(int).tolist()
                paths = md["image_path"].astype(str).str.replace("\\", "/", regex=False).tolist()
                by_relpath = dict(zip(paths, fake))
                # Filename fallback, only for names whose label is unambiguous.
                for p, lab in zip(paths, fake):
                    name = p.rsplit("/", 1)[-1]
                    by_name[name] = lab if by_name.get(name, lab) == lab else None
        default_label = 0 if generator in REAL_SOURCES else 1
        for img in _iter_images(gen_dir):
            rel_parts = img.relative_to(gen_dir).parts
            label = by_relpath.get("/".join(rel_parts))
            if label is None:
                label = by_name.get(img.name)
            if label is None:
                label = default_label
            gen = _artifact_subgenerator(generator, rel_parts)
            rows.append(_row(img, label, gen, _family_for(gen), "artifact"))
    return rows


def scan_forensynths(root: Path) -> List[dict]:
    """ForenSynths / CNNDetection test set.

    Layout: ``<root>/<generator>/[<category>/]{0_real,1_fake}/<images...>``

    Most generators in the official test zip are split by scene category
    (``progan/airplane/0_real``, ``cyclegan/horse/1_fake``); a few are flat
    (``biggan/0_real``). Both are collected under the top-level generator.

    No family default: the zip also ships non-GAN sources (``san``
    super-resolution, ``crn``/``imle`` non-adversarial synthesis,
    ``seeingdark`` low-light imaging). Defaulting those to ``gan`` would
    contaminate the GAN family, so unlisted generators come out ``unknown``
    and ``build_manifest`` drops them.
    """
    rows: List[dict] = []
    if not root.exists():
        return rows
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generator = _norm(gen_dir.name)
        family = _family_for(generator)
        for sub, label in (("0_real", 0), ("1_fake", 1)):
            for sub_dir in sorted(d for d in gen_dir.rglob(sub) if d.is_dir()):
                for img in _iter_images(sub_dir):
                    rows.append(_row(img, label, generator, family, "forensynths"))
    return rows


def scan_genimage(root: Path) -> List[dict]:
    """GenImage. Layout: ``<root>/<generator>/{train,val}/{nature,ai}/...``

    GenImage's own train/val split is kept only when a generator has both.
    The public ``genimage_test.zip`` ships ``val`` alone; labelling all of it
    ``val`` would leave nothing for probe training or testing, so a lone split
    is left blank and ``assign_splits`` re-splits it within
    ``(generator, label)`` like every other source.
    """
    rows: List[dict] = []
    if not root.exists():
        return rows
    for gen_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        generator = _norm(gen_dir.name)
        family = _family_for(generator, default="diffusion")
        split_dirs = [d for d in gen_dir.iterdir() if d.is_dir()]
        has_both = len({d.name.lower().startswith("train") for d in split_dirs}) == 2
        for split_dir in split_dirs:
            if not has_both:
                split = ""
            else:
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


def _subset_dir(path: str, source: str) -> str:
    """The folder directly under the dataset root (``<source>/<subset>/...``),
    or "" if the path does not contain the source's root folder."""
    parts = Path(path).parts
    idx = [i for i, p in enumerate(parts[:-2]) if p == source]
    return parts[idx[-1] + 1] if idx else ""


def cap_per_group(df: pd.DataFrame, max_per_group: int,
                  seed: int = 1337) -> pd.DataFrame:
    """Seeded subsample to at most ``max_per_group`` rows per
    ``(source, generator, label, subset folder)``.

    Applied *before* ``assign_splits`` so splits are still drawn within each
    ``(generator, label)`` group. Keyed on source too, so a generator shipped by
    two datasets (e.g. ArtiFact and ForenSynths ProGAN) keeps both.

    The subset folder matters for real images only, which all share generator
    ``real``: without it each dataset's reals form ONE group and the cap keeps
    a few hundred per content domain. ForenSynths pairs each generator folder
    with content-matched reals (faces for ``deepfake``, LSUN for ``progan``)
    and ArtiFact spreads its reals over 8 sub-datasets; collapsing them leaves
    face-only fake families facing mostly non-face reals, and a probe can then
    separate them by content. For fakes the folder is the generator's own.
    """
    keep = []
    sub = [_subset_dir(p, s) for p, s in zip(df["path"], df["source"])]
    for _, grp in df.assign(_sub=sub).groupby(
            ["source", "generator", "label", "_sub"], sort=True):
        keep.extend(grp.sample(n=min(len(grp), max_per_group),
                               random_state=seed).index)
    return df.loc[sorted(keep)].reset_index(drop=True)


def build_manifest(rows: Iterable[dict], seed: int = 1337,
                   out_name: str = "manifest.csv",
                   max_per_group: Optional[int] = None) -> pd.DataFrame:
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    if df.empty:
        raise RuntimeError(
            "No images found. Download at least one dataset into data/raw/ "
            "first -- see docs/DATA.md"
        )
    df["split"] = df["split"].fillna("").astype(str)
    df["mask_path"] = df["mask_path"].fillna("").astype(str)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)

    # Every non-real family becomes a probe config downstream, so an "unknown"
    # bucket would be trained and reported as if it were a synthesis paradigm.
    unknown = df["family"] == "unknown"
    if unknown.any():
        gens = sorted(df.loc[unknown, "generator"].unique())
        print("[manifest] WARNING: dropping {:,} images from generators with no "
              "family: {}. Add them to config.FAMILIES to keep them.".format(
                  int(unknown.sum()), gens))
        df = df[~unknown].reset_index(drop=True)

    if max_per_group:
        before = len(df)
        df = cap_per_group(df, max_per_group, seed=seed)
        print("[manifest] capped to {:,} per (source, generator, label, subset): "
              "{:,} -> {:,} rows".format(max_per_group, before, len(df)))

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
