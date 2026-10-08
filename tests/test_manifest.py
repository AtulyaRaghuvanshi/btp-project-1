"""Generator taxonomy and dataset scanners. Run: python -m pytest tests/ -v"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from xgf.config import FAMILIES, GENERATOR_TO_FAMILY, PATHS
from xgf.data import manifest as M

# Every generator folder in Kaggle awsaf49/artifact-dataset (verified against
# the per-folder metadata.csv files), with the family it must land in.
ARTIFACT_GENERATORS = {
    "big_gan": "gan", "cips": "gan", "cycle_gan": "gan", "diffusion_gan": "gan",
    "denoising_diffusion_gan": "gan", "gansformer": "gan", "gau_gan": "gan",
    "pro_gan": "gan", "projected_gan": "gan", "sfhq": "gan", "star_gan": "gan",
    "stylegan1": "gan", "stylegan2": "gan", "stylegan3": "gan",
    "ddpm": "diffusion", "glide": "diffusion", "latent_diffusion": "diffusion",
    "palette": "diffusion", "stable_diffusion": "diffusion",
    "vq_diffusion": "diffusion",
    "taming_transformer": "autoregressive",
    "generative_inpainting": "inpainting", "lama": "inpainting", "mat": "inpainting",
    "face_synthetics": "rendered",
}


def test_generator_names_are_unique_across_families():
    names = [g for gens in FAMILIES.values() for g in gens]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("gen,fam", sorted(ARTIFACT_GENERATORS.items()))
def test_every_artifact_generator_has_the_right_family(gen, fam):
    assert GENERATOR_TO_FAMILY.get(gen) == fam


def test_artifact_generator_count():
    assert len(ARTIFACT_GENERATORS) == 25
    assert not (set(ARTIFACT_GENERATORS) & M.REAL_SOURCES)


def test_dataset_default_does_not_override_known_generators():
    # GenImage defaults to diffusion but ships a BigGAN subset.
    assert M._family_for("biggan", default="diffusion") == "gan"
    assert M._family_for("stable_diffusion_v_1_4", default="diffusion") == "diffusion"
    # ForenSynths defaults to gan but its "deepfake" subset is a face swap.
    assert M._family_for("deepfake", default="gan") == "faceswap"
    # An explicit override (scan_generic's family=...) still wins.
    assert M._family_for("progan", override="custom") == "custom"
    assert M._family_for("never_heard_of_it") == "unknown"


# --------------------------------------------------------------------------- #
# scan_artifact on a miniature copy of the real layout
# --------------------------------------------------------------------------- #
def _touch(root: Path, folder: str, rel: str) -> None:
    p = root / folder / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _make_folder(root: Path, folder: str, entries) -> None:
    """entries: [(relative image path, ArtiFact target code)]"""
    for rel, _ in entries:
        _touch(root, folder, rel)
    pd.DataFrame({
        "filename": [r.rsplit("/", 1)[-1] for r, _ in entries],
        "image_path": [r for r, _ in entries],
        "target": [t for _, t in entries],
        "category": ["x"] * len(entries),
    }).to_csv(root / folder / "metadata.csv", index=False)


@pytest.fixture
def artifact_root(tmp_path):
    root = tmp_path / "artifact"
    _make_folder(root, "afhq", [("afhq/train/cat/img000000.jpg", 0),
                                ("afhq/train/dog/img000000.jpg", 0)])
    # Mixed folder, same filename under real and fake: only the relative path
    # disambiguates.
    _make_folder(root, "pro_gan", [("pro/airplane/0_real/img000001.jpg", 0),
                                   ("pro/airplane/1_fake/img000001.jpg", 6)])
    _make_folder(root, "stylegan2", [("car/img000002.jpg", 1)])
    _make_folder(root, "taming_transformer", [("tt-ffhq/img000003.jpg", 5)])
    _make_folder(root, "stable_diffusion", [("stable/images/img000004.jpg", 6)])
    _make_folder(root, "glide", [("glide-in/images/img000005.jpg", 3),
                                 ("glide-t2i/images/img000006.jpg", 3),
                                 ("glide/glide/inpainting/img000007.jpg", 3),
                                 ("glide/glide/text2img/img000008.jpg", 3)])
    _make_folder(root, "face_synthetics", [("facesyn/images/img000009.jpg", 6)])
    return root


def _by_rel(rows, root):
    return {Path(r["path"]).relative_to(root).as_posix(): r for r in rows}


def test_scan_artifact_labels_are_binary(artifact_root):
    rows = M.scan_artifact(artifact_root)
    assert len(rows) == 12
    assert {r["label"] for r in rows} == {0, 1}


def test_scan_artifact_mixed_folder_uses_per_file_labels(artifact_root):
    r = _by_rel(M.scan_artifact(artifact_root), artifact_root)
    real = r["pro_gan/pro/airplane/0_real/img000001.jpg"]
    fake = r["pro_gan/pro/airplane/1_fake/img000001.jpg"]
    assert (real["label"], real["generator"], real["family"]) == (0, "real", "real")
    assert (fake["label"], fake["generator"], fake["family"]) == (1, "pro_gan", "gan")


def test_scan_artifact_families(artifact_root):
    r = _by_rel(M.scan_artifact(artifact_root), artifact_root)
    expect = {
        "afhq/afhq/train/cat/img000000.jpg": ("real", "real"),
        "stylegan2/car/img000002.jpg": ("stylegan2", "gan"),
        "taming_transformer/tt-ffhq/img000003.jpg": ("taming_transformer", "autoregressive"),
        "stable_diffusion/stable/images/img000004.jpg": ("stable_diffusion", "diffusion"),
        "glide/glide-in/images/img000005.jpg": ("glide_inpaint", "inpainting"),
        "glide/glide-t2i/images/img000006.jpg": ("glide", "diffusion"),
        "glide/glide/glide/inpainting/img000007.jpg": ("glide_inpaint", "inpainting"),
        "glide/glide/glide/text2img/img000008.jpg": ("glide", "diffusion"),
        "face_synthetics/facesyn/images/img000009.jpg": ("face_synthetics", "rendered"),
    }
    for rel, (gen, fam) in expect.items():
        assert (r[rel]["generator"], r[rel]["family"]) == (gen, fam), rel


def test_scan_artifact_without_metadata_falls_back_to_folder_default(tmp_path):
    root = tmp_path / "artifact"
    _touch(root, "ffhq", "images/a.png")
    _touch(root, "lama", "x/b.png")
    r = _by_rel(M.scan_artifact(root), root)
    assert r["ffhq/images/a.png"]["label"] == 0
    assert (r["lama/x/b.png"]["label"], r["lama/x/b.png"]["family"]) == (1, "inpainting")


def test_build_manifest_drops_unknown_family(tmp_path, monkeypatch):
    monkeypatch.setattr(PATHS, "manifests", tmp_path)
    rows = []
    for i in range(10):
        rows.append(M._row("/r/{}.png".format(i), 0, "real", "real", "t"))
        rows.append(M._row("/g/{}.png".format(i), 1, "pro_gan", "gan", "t"))
        rows.append(M._row("/u/{}.png".format(i), 1, "mystery", "unknown", "t"))
    df = M.build_manifest(rows, out_name="m.csv")
    assert set(df["family"]) == {"real", "gan"}
    assert len(df) == 20
    assert (tmp_path / "m.csv").exists()


def test_scan_forensynths_finds_nested_categories(tmp_path):
    root = tmp_path / "forensynths"
    _touch(root, "progan", "airplane/0_real/a.png")
    _touch(root, "progan", "airplane/1_fake/b.png")
    _touch(root, "progan", "cat/1_fake/c.png")
    _touch(root, "biggan", "0_real/d.png")
    _touch(root, "biggan", "1_fake/e.png")
    r = _by_rel(M.scan_forensynths(root), root)
    assert len(r) == 5
    assert r["progan/airplane/0_real/a.png"]["label"] == 0
    assert r["progan/cat/1_fake/c.png"]["generator"] == "progan"
    assert r["biggan/1_fake/e.png"]["label"] == 1


def test_scan_forensynths_leaves_non_gan_sources_unknown(tmp_path):
    root = tmp_path / "forensynths"
    _touch(root, "san", "1_fake/a.png")
    _touch(root, "deepfake", "1_fake/b.png")
    r = _by_rel(M.scan_forensynths(root), root)
    assert r["san/1_fake/a.png"]["family"] == "unknown"
    assert r["deepfake/1_fake/b.png"]["family"] == "faceswap"


def test_cap_per_group_is_seeded_and_bounded():
    rows = [M._row("/a/{}.png".format(i), 1, "pro_gan", "gan", "artifact") for i in range(50)]
    rows += [M._row("/b/{}.png".format(i), 1, "pro_gan", "gan", "forensynths") for i in range(5)]
    df = pd.DataFrame(rows, columns=M.COLUMNS)
    a = M.cap_per_group(df, 10, seed=1)
    b = M.cap_per_group(df, 10, seed=1)
    assert list(a["path"]) == list(b["path"])
    assert a.groupby("source").size().to_dict() == {"artifact": 10, "forensynths": 5}


def test_cap_keeps_reals_from_every_subset_folder():
    """Reals all share generator 'real'; the cap must not pool their domains."""
    rows = [M._row("/d/raw/forensynths/{}/0_real/{}.png".format(g, i), 0, "real",
                   "real", "forensynths") for g in ("deepfake", "progan") for i in range(20)]
    rows += [M._row("/d/raw/forensynths/deepfake/1_fake/{}.png".format(i), 1,
                    "deepfake", "faceswap", "forensynths") for i in range(20)]
    df = pd.DataFrame(rows, columns=M.COLUMNS)
    out = M.cap_per_group(df, 5, seed=1)
    reals = out[out["label"] == 0]["path"].str.split("/").str[4]
    assert reals.value_counts().to_dict() == {"deepfake": 5, "progan": 5}
    assert (out["label"] == 1).sum() == 5


def test_scan_genimage_lone_val_split_is_left_for_reassignment(tmp_path):
    root = tmp_path / "genimage"
    _touch(root, "sd14", "val/ai/a.png")
    _touch(root, "sd14", "val/nature/b.png")
    _touch(root, "adm", "train/ai/c.png")
    _touch(root, "adm", "val/ai/d.png")
    r = _by_rel(M.scan_genimage(root), root)
    assert r["sd14/val/ai/a.png"]["split"] == ""
    assert r["sd14/val/nature/b.png"]["label"] == 0
    assert (r["adm/train/ai/c.png"]["split"], r["adm/val/ai/d.png"]["split"]) == ("train", "val")
