"""Run the whole pipeline for one or more backbones.

    # fast sanity pass on synthetic data (minutes)
    python scripts/run_all.py --selftest

    # the real thing, once data/raw is populated
    python scripts/run_all.py --backbones resnet50 vit_b16 clip_vitb16 \
        --n-images 200 --explainers gradcampp gradcam inputgrad rise random

Each stage is skipped if its output already exists, so the script is safe to
re-run after a crash or an interruption.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xgf.config import PATHS  # noqa: E402


def run(cmd: list, label: str) -> None:
    print("\n" + "=" * 78)
    print(">>> {}".format(label))
    print("    " + " ".join(str(c) for c in cmd))
    print("=" * 78)
    rc = subprocess.call([sys.executable] + [str(c) for c in cmd])
    if rc != 0:
        print("\n!! stage failed: {} (exit {})".format(label, rc))
        sys.exit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--backbones", nargs="*", default=["resnet50"])
    ap.add_argument("--probes", nargs="*", default=["mixed"])
    ap.add_argument("--explainers", nargs="*",
                    default=["gradcampp", "gradcam", "inputgrad", "random"])
    ap.add_argument("--operators", nargs="*",
                    default=["zero", "gray", "road_linear", "real_transplant"])
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--n-steps", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true", help="ignore existing outputs")
    args = ap.parse_args()

    S = ROOT / "scripts"
    tag = "selftest" if args.selftest else "all"
    manifest = "manifest_selftest.csv" if args.selftest else "manifest.csv"

    if args.force or not (PATHS.manifests / manifest).exists():
        cmd = [S / "01_build_manifest.py"]
        if args.selftest:
            cmd.append("--selftest")
        run(cmd, "01 build manifest")
    else:
        print("[skip] manifest exists: {}".format(manifest))

    for bb in args.backbones:
        feat = PATHS.features / "{}__{}.npz".format(bb, tag)
        if args.force or not feat.exists():
            run([S / "02_extract_features.py", "--backbone", bb,
                 "--manifest", manifest, "--tag", tag,
                 "--batch-size", args.batch_size, "--device", args.device],
                "02 extract features [{}]".format(bb))
        else:
            print("[skip] features exist: {}".format(feat.name))

        run([S / "03_train_probes.py", "--backbone", bb, "--tag", tag],
            "03 train probes [{}]".format(bb))

        for probe in args.probes:
            run([S / "04_run_leakage.py", "--backbone", bb, "--tag", tag,
                 "--probe", probe, "--n-images", min(args.n_images, 40),
                 "--n-steps", args.n_steps, "--n-repeats", 2,
                 "--batch-size", args.batch_size, "--device", args.device],
                "04 leakage [{} / {}]".format(bb, probe))

        run([S / "05_run_faithfulness.py", "--backbone", bb, "--tag", tag,
             "--probes", *args.probes, "--explainers", *args.explainers,
             "--operators", *args.operators, "--n-images", args.n_images,
             "--n-steps", args.n_steps, "--batch-size", args.batch_size,
             "--device", args.device],
            "05 faithfulness [{}]".format(bb))

        run([S / "06_make_figures.py", "--backbone", bb, "--tag", tag,
             "--probe", args.probes[0]],
            "06 figures [{}]".format(bb))

    print("\n" + "=" * 78)
    print("DONE.  results/ and results/figures/")
    print("=" * 78)


if __name__ == "__main__":
    main()
