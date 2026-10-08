"""EXPERIMENT 0 -- the leakage null test.

Run this before any faithfulness number is believed. It asks a single question
with no explanation method involved:

    On images that are genuinely REAL, does masking a random region make the
    detector say "fake"?

Any answer above zero is bias that contaminates every deletion/insertion curve
computed with that operator.

    python scripts/04_run_leakage.py --backbone resnet50 --tag selftest
    python scripts/04_run_leakage.py --backbone resnet50 --tag selftest \
        --probe mixed --operators zero gray road_linear real_transplant

Output: results/leakage__<backbone>__<probe>.csv plus a JSON summary.

Resumable: each operator's result is committed to
``results/progress/leakage__<backbone>__<probe>.jsonl`` as it finishes, so a
re-run skips operators already measured. Adding operators to the list later
just measures the new ones.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from xgf.config import PATHS, RunConfig  # noqa: E402
from xgf.data.datasets import ManifestDataset, subset  # noqa: E402
from xgf.data.manifest import load_manifest  # noqa: E402
from xgf.metrics.leakage import measure_leakage  # noqa: E402
from xgf.models.detector import Detector  # noqa: E402
from xgf.models.probe import (  # noqa: E402
    feature_precision, load_probe, probe_path, probe_to_detector,
)
from xgf.perturb.operators import (  # noqa: E402
    ARTIFACT_INJECTING, ARTIFACT_NEUTRAL, DonorBank,
)
from xgf.utils.checkpoint import (  # noqa: E402
    RunLog, file_fingerprint, reseed_all, run_resumable, unit_seed,
)
from xgf.utils.io import save_json  # noqa: E402
from xgf.utils.seed import set_seed  # noqa: E402

DEFAULT_OPS = ARTIFACT_INJECTING + ARTIFACT_NEUTRAL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--tag", default="all", help="feature cache tag")
    ap.add_argument("--manifest", default=None,
                    help="default: manifest_<tag>.csv, else manifest.csv")
    ap.add_argument("--probe", default="mixed", help="probe config tag")
    ap.add_argument("--operators", nargs="*", default=DEFAULT_OPS)
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=21)
    ap.add_argument("--n-repeats", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fresh", action="store_true",
                    help="discard saved progress for this backbone/probe")
    args = ap.parse_args()

    cfg = RunConfig(device=args.device, seed=args.seed)
    set_seed(cfg.seed)
    PATHS.ensure()
    device = cfg.resolve_device()

    mname = args.manifest or (
        "manifest_{}.csv".format(args.tag) if args.tag != "all" else "manifest.csv"
    )
    df = load_manifest(mname)

    det = Detector(args.backbone).to(device).eval()
    clf, info = load_probe(args.backbone, args.probe)
    det = probe_to_detector(det, clf)
    print("[leakage] backbone={} probe={} (trained on {})".format(
        args.backbone, args.probe, info.get("train_families")))

    # Real, held-out images only. The premise of the null test is that these
    # contain no forgery, so any rise in P(fake) is manufactured by the operator.
    real_df = subset(df, label=0, splits=["test"], n=args.n_images, seed=cfg.seed)
    if len(real_df) < 10:
        real_df = subset(df, label=0, n=args.n_images, seed=cfg.seed)
    ds = ManifestDataset(real_df, image_size=cfg.image_size)
    images = torch.stack([ds[i]["image"] for i in range(len(ds))])
    print("[leakage] {} real test images".format(len(images)))

    donors = DonorBank.from_manifest(
        subset(df, label=0, splits=["train"]), image_size=cfg.image_size,
        n=min(256, max(32, len(df) // 8)), seed=cfg.seed,
    )

    # Batch size is left out: it only chunks forward passes, it does not change
    # which pixels are masked or filled.
    log = RunLog(
        PATHS.progress / "leakage__{}__{}.jsonl".format(args.backbone, args.probe),
        config={
            "backbone": args.backbone, "probe": args.probe,
            "probe_file": file_fingerprint(probe_path(args.backbone, args.probe)),
            "manifest": file_fingerprint(PATHS.manifests / mname),
            "n_images": args.n_images, "n_steps": args.n_steps,
            "n_repeats": args.n_repeats, "seed": cfg.seed,
            "image_size": cfg.image_size,
            "precision": feature_precision(torch.device(device), cfg.amp),
        },
        fresh=args.fresh,
    )
    n_done = sum(log.done(op) for op in args.operators)
    if n_done:
        print("[leakage] resuming: {}/{} operators already measured ({})".format(
            n_done, len(args.operators), log.path))

    rows, curves = [], {}
    for op in args.operators:
        if not log.done(op):
            # Reseed from the operator's name alone, so its stochastic fill and
            # donor draws are the same whether or not earlier operators ran in
            # this process.
            op_seed = unit_seed(cfg.seed, "leakage", op)
            reseed_all(op_seed)
            donors.reseed(op_seed)
            res = measure_leakage(
                det, images, operator=op, n_steps=args.n_steps,
                n_repeats=args.n_repeats, donor_bank=donors,
                batch_size=args.batch_size, amp=cfg.amp, seed=cfg.seed,
                meta={"backbone": args.backbone, "probe": args.probe},
            )
            s = res.summary()
            s["class"] = ("injecting" if op in ARTIFACT_INJECTING
                          else "neutral" if op in ARTIFACT_NEUTRAL else "other")
            log.commit(op, {"summary": s, "curves": {
                "fractions": res.fractions.tolist(),
                "mean_pfake": res.curves.mean(axis=0).tolist(),
                "mean_delta": res.delta_curves.mean(axis=0).tolist(),
            }})
        rec = log.get(op)
        s = rec["summary"]
        rows.append(s)
        curves[op] = rec["curves"]
        print("  {:<16} [{:<9}] leak_auc={:.4f}  delta={:+.4f}  "
              "slope={:+.4f}  flip={:.3f}".format(
                  op, s["class"], s["leakage_auc"], s["leakage_delta_auc"],
                  s["leakage_slope"], s["flip_rate"]))

    out = pd.DataFrame(rows).sort_values("leakage_auc", ascending=False)
    csv = PATHS.results / "leakage__{}__{}.csv".format(args.backbone, args.probe)
    out.to_csv(csv, index=False)
    save_json(curves, PATHS.results / "leakage_curves__{}__{}.json".format(
        args.backbone, args.probe))

    print("\n" + "=" * 70)
    print("LEAKAGE RANKING (lower = cleaner measuring instrument)")
    print("=" * 70)
    print(out[["operator", "class", "leakage_auc", "leakage_delta_auc",
               "leakage_slope", "flip_rate"]].round(4).to_string(index=False))

    inj = out[out["class"] == "injecting"]["leakage_delta_auc"].mean()
    neu = out[out["class"] == "neutral"]["leakage_delta_auc"].mean()
    print("\n  mean delta-AUC: artifact-injecting={:+.4f}  "
          "artifact-neutral={:+.4f}".format(inj, neu))
    if inj > neu:
        print("  -> injecting operators manufacture more fakeness, as predicted.")
    print("\n[leakage] -> {}".format(csv))
    print("\nNEXT: python scripts/05_run_faithfulness.py --backbone {} --tag {}".format(
        args.backbone, args.tag))


if __name__ == "__main__":
    run_resumable(main)
