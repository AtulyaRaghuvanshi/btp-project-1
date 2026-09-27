"""Train one linear probe per training-family configuration, and build the
detection-accuracy half of the cross-generator transfer matrix.

    python scripts/03_train_probes.py --backbone resnet50 --tag selftest

Configurations trained:
  * one probe per single family      ("gan", "diffusion", ...)
  * one probe on the MIXTURE of all families ("mixed")
  * leave-one-family-out probes       ("loo_gan", ...)

The mixed-vs-single contrast is the controlled intervention at the heart of the
cross-generator experiment: same backbone, same head capacity, same total
training budget -- only the *generator diversity* of the training set changes.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xgf.config import PATHS  # noqa: E402
from xgf.models.probe import (  # noqa: E402
    evaluate_probe_per_family, load_features, save_probe, train_probe,
)
from xgf.utils.io import save_json  # noqa: E402
from xgf.utils.seed import set_seed  # noqa: E402


def build_configs(families, include_loo: bool = True):
    """Return ``{tag: [train_families]}``."""
    cfgs = {f: [f] for f in families}
    cfgs["mixed"] = list(families)
    if include_loo and len(families) > 2:
        for f in families:
            cfgs["loo_" + f] = [g for g in families if g != f]
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--tag", default="all", help="feature cache tag")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-loo", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    PATHS.ensure()

    cache = load_features(args.backbone, tag=args.tag)
    fams = sorted(set(cache["family"].astype(str)) - {"real"})
    print("[probes] backbone={} families={}".format(args.backbone, fams))

    configs = build_configs(fams, include_loo=not args.no_loo)
    rows, infos = [], {}

    for tag, train_fams in configs.items():
        try:
            clf, info = train_probe(cache, train_fams, C=args.C, seed=args.seed)
        except RuntimeError as exc:
            print("  [skip] {}: {}".format(tag, exc))
            continue
        per_fam = evaluate_probe_per_family(clf, cache, split="test")
        per_fam["train_config"] = tag
        per_fam["train_families"] = ",".join(train_fams)
        per_fam["seen"] = per_fam["test_family"].isin(train_fams)
        per_fam["backbone"] = args.backbone
        rows.append(per_fam)

        save_probe(clf, info, args.backbone, tag)
        infos[tag] = info
        seen = per_fam.loc[per_fam["seen"], "auc"]
        unseen = per_fam.loc[~per_fam["seen"], "auc"]
        print("  {:<16} val_auc={:.3f}  seen_auc={:.3f}  unseen_auc={:.3f}".format(
            tag, info.get("val_auc", float("nan")),
            seen.mean() if len(seen) else float("nan"),
            unseen.mean() if len(unseen) else float("nan")))

    if not rows:
        print("No probes trained -- check the feature cache.")
        sys.exit(1)

    matrix = pd.concat(rows, ignore_index=True)
    out = PATHS.results / "accuracy_matrix__{}__{}.csv".format(args.backbone, args.tag)
    matrix.to_csv(out, index=False)
    save_json(infos, PATHS.results / "probe_info__{}__{}.json".format(
        args.backbone, args.tag))
    print("\n[probes] -> {}".format(out))

    piv = matrix.pivot_table(index="train_config", columns="test_family",
                             values="auc", aggfunc="mean")
    print("\nDetection AUC  (rows = train config, cols = test family)")
    print(piv.round(3).to_string())

    gen_gap = (
        matrix[matrix["seen"]].groupby("train_config")["auc"].mean()
        - matrix[~matrix["seen"]].groupby("train_config")["auc"].mean()
    ).dropna()
    if len(gen_gap):
        print("\nAccuracy generalization gap (seen - unseen AUC):")
        print(gen_gap.round(3).to_string())

    print("\nNEXT: python scripts/04_run_leakage.py --backbone {} --tag {}".format(
        args.backbone, args.tag))


if __name__ == "__main__":
    main()
