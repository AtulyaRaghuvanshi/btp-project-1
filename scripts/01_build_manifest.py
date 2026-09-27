"""Build data/manifests/manifest.csv from whatever is present in data/raw/.

    # scan real datasets you have downloaded
    python scripts/01_build_manifest.py

    # generate a tiny SYNTHETIC dataset and manifest so the whole pipeline can
    # be run end-to-end today, before any download finishes
    python scripts/01_build_manifest.py --selftest

The self-test data is not a scientific artifact -- it exists so that every
script, metric and figure can be debugged while the real downloads run. Its
"generators" are simple, separable image corruptions with deliberately
different frequency signatures, which is enough to exercise the cross-family
machinery.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from xgf.config import PATHS  # noqa: E402
from xgf.data import manifest as M  # noqa: E402
from xgf.utils.seed import set_seed  # noqa: E402


def make_selftest_data(root: Path, n_per_class: int = 60, size: int = 224,
                       seed: int = 1337) -> None:
    """Synthesise a stand-in dataset with three distinct 'generator families'.

    real          : smooth natural-ish texture (1/f noise)
    gan_like      : + periodic checkerboard (mimics up-sampling artifacts)
    diffusion_like: + spatially-correlated high-frequency residual
    inpaint_like  : real, but a rectangular region swapped from another image
                    (this one ships a ground-truth mask, so localization
                    metrics are exercised too)
    """
    rng = np.random.default_rng(seed)

    def pink(h, w):
        """1/f noise -- a crude but useful stand-in for natural image stats."""
        white = rng.normal(size=(h, w))
        f = np.fft.fftfreq(h)[:, None] ** 2 + np.fft.fftfreq(w)[None, :] ** 2
        f[0, 0] = 1e-6
        spec = np.fft.fft2(white) / np.sqrt(f)
        out = np.real(np.fft.ifft2(spec))
        out = (out - out.min()) / (np.ptp(out) + 1e-8)
        return out

    def base_img():
        return np.stack([pink(size, size) for _ in range(3)], axis=-1)

    def save(arr, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.clip(arr * 255, 0, 255).astype(np.uint8)
        ).save(path)

    print("[selftest] generating synthetic data in", root)

    for i in range(n_per_class):
        save(base_img(), root / "real" / "real_{:04d}.png".format(i))

    yy, xx = np.mgrid[0:size, 0:size]
    checker = ((np.sin(xx * np.pi / 2) * np.sin(yy * np.pi / 2)) > 0).astype(np.float32)
    for i in range(n_per_class):
        img = base_img() + 0.06 * checker[..., None]
        save(img, root / "gan_like" / "gan_{:04d}.png".format(i))

    for i in range(n_per_class):
        hf = rng.normal(size=(size, size, 3))
        hf = hf - np.stack(
            [np.convolve(hf[..., c].ravel(), np.ones(9) / 9, "same").reshape(size, size)
             for c in range(3)], axis=-1
        )
        img = base_img() + 0.05 * hf
        save(img, root / "diffusion_like" / "dif_{:04d}.png".format(i))

    mask_dir = root / "inpaint_like" / "masks"
    for i in range(n_per_class):
        img = base_img()
        donor = base_img()
        y0, x0 = rng.integers(20, size - 90, size=2)
        h, w = rng.integers(50, 80, size=2)
        img[y0:y0 + h, x0:x0 + w] = donor[y0:y0 + h, x0:x0 + w] + 0.04
        save(img, root / "inpaint_like" / "fake" / "inp_{:04d}.png".format(i))
        m = np.zeros((size, size), dtype=np.float32)
        m[y0:y0 + h, x0:x0 + w] = 1.0
        save(np.stack([m] * 3, -1), mask_dir / "inp_{:04d}.png".format(i))


def scan_selftest(root: Path) -> list:
    rows = []
    rows += M.scan_generic(root / "real", "selftest", "real", 0)
    rows += M.scan_generic(root / "gan_like", "selftest", "progan", 1, family="gan")
    rows += M.scan_generic(root / "diffusion_like", "selftest", "sd14", 1,
                           family="diffusion")
    fake_dir = root / "inpaint_like" / "fake"
    mask_dir = root / "inpaint_like" / "masks"
    for img in sorted(fake_dir.glob("*.png")):
        mp = mask_dir / img.name
        rows.append(M._row(img, 1, "glide_inpaint", "inpainting", "selftest",
                           "", str(mp) if mp.exists() else ""))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="generate + scan synthetic data instead of real datasets")
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    PATHS.ensure()
    raw = PATHS.data_raw

    if args.selftest:
        root = raw / "selftest"
        if not (root / "real").exists():
            make_selftest_data(root, n_per_class=args.n_per_class, seed=args.seed)
        rows = scan_selftest(root)
        M.build_manifest(rows, seed=args.seed, out_name="manifest_selftest.csv")
        print("\nNEXT: python scripts/02_extract_features.py "
              "--manifest manifest_selftest.csv --backbone resnet50")
        return

    rows = []
    scanners = [
        ("artifact", M.scan_artifact, raw / "artifact"),
        ("forensynths", M.scan_forensynths, raw / "forensynths"),
        ("genimage", M.scan_genimage, raw / "genimage"),
        ("cocoglide", M.scan_cocoglide, raw / "cocoglide"),
    ]
    for name, fn, path in scanners:
        found = fn(path)
        print("  [{:<12}] {:>8,} images  ({})".format(
            name, len(found), path if path.exists() else "not present"))
        rows += found

    if not rows:
        print("\nNothing found under {}.".format(raw))
        print("See docs/DATA.md for download instructions, or run with --selftest")
        sys.exit(1)

    M.build_manifest(rows, seed=args.seed, out_name=args.out)


if __name__ == "__main__":
    main()
