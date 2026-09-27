"""Cache frozen-backbone features for every image in the manifest.

    python scripts/02_extract_features.py --backbone resnet50
    python scripts/02_extract_features.py --backbone clip_vitb16 --batch-size 16

One pass per backbone. Everything after this (probe training, the whole
train-family x test-family matrix) runs off the cache in seconds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from xgf.config import PATHS, RunConfig  # noqa: E402
from xgf.data.manifest import load_manifest  # noqa: E402
from xgf.models.detector import Detector  # noqa: E402
from xgf.models.probe import extract_features, save_features  # noqa: E402
from xgf.utils.seed import set_seed  # noqa: E402
from xgf.utils.timing import timer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--tag", default=None, help="cache tag (default: manifest stem)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N rows (smoke testing)")
    args = ap.parse_args()

    cfg = RunConfig(device=args.device, image_size=args.image_size,
                    batch_size=args.batch_size)
    set_seed(cfg.seed)
    PATHS.ensure()
    device = cfg.resolve_device()
    tag = args.tag or Path(args.manifest).stem.replace("manifest_", "") or "all"

    df = load_manifest(args.manifest)
    if args.limit:
        df = df.head(args.limit)
    print("[extract] {} rows | backbone={} | device={}".format(
        len(df), args.backbone, device))

    det = Detector(args.backbone).to(device).eval()
    for p in det.parameters():
        p.requires_grad_(False)

    with timer("extract_features"):
        cache = extract_features(
            det, df, image_size=cfg.image_size, batch_size=cfg.batch_size,
            num_workers=args.num_workers, amp=cfg.amp,
            desc="{}".format(args.backbone),
        )
    save_features(cache, args.backbone, tag=tag)
    print("\nNEXT: python scripts/03_train_probes.py --backbone {} --tag {}".format(
        args.backbone, tag))


if __name__ == "__main__":
    main()
