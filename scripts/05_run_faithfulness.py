"""EXPERIMENT 1+2 -- the cross-generator, cross-operator faithfulness grid.

For each (probe config, test family, explainer, operator) cell it computes

  * raw deletion AUC on the salient ranking,
  * deletion AUC on a matched random-location control,
  * the leakage-corrected DeltaAUC and its paired significance test,
  * localization metrics where ground-truth masks exist.

The three questions this answers:

  Q1  Does the ranking of explanation methods change when the fill operator
      changes from artifact-injecting to artifact-neutral?
      (If yes, published "method X is most faithful" claims are operator claims.)

  Q2  Does explanation faithfulness degrade under generator shift, and does it
      degrade faster or slower than detection accuracy?

  Q3  Does training on a MIXTURE of generator families buy faithfulness
      robustness over and above the accuracy robustness it is known to buy?

    python scripts/05_run_faithfulness.py --backbone resnet50 --tag selftest \
        --probes mixed gan --explainers gradcampp inputgrad random \
        --operators zero road_linear --n-images 40

Resumable: every finished (probe, family, explainer, operator) row is committed
to ``results/progress/faithfulness__<backbone>__<tag>/log.jsonl``, and saliency
maps and random-control curves are cached beside it. Re-running the same
command skips finished rows; widening the grid (more explainers, operators,
families or probes) computes only the new cells. Each curve reseeds from its
own cell identity, so resumed numbers equal uninterrupted ones.
"""
from __future__ import annotations

import argparse
import itertools
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from xgf.config import PATHS, RunConfig  # noqa: E402
from xgf.data.datasets import ManifestDataset, subset  # noqa: E402
from xgf.data.manifest import load_manifest  # noqa: E402
from xgf.explain.base import build_explainer  # noqa: E402
from xgf.explain import methods as _methods  # noqa: F401,E402  (registers)
from xgf.metrics.faithfulness import (  # noqa: E402
    CurveResult, corrected_faithfulness, perturbation_curve,
)
from xgf.metrics.grid import balanced_explainers, complete_cells  # noqa: E402
from xgf.metrics.leakage import random_masks  # noqa: E402
from xgf.metrics.localization import localization_summary  # noqa: E402
from xgf.models.detector import Detector  # noqa: E402
from xgf.models.probe import (  # noqa: E402
    feature_precision, load_probe, probe_path, probe_to_detector,
)
from xgf.perturb.operators import DonorBank  # noqa: E402
from xgf.utils.checkpoint import (  # noqa: E402
    RunLog, atomic_save_npz, atomic_save_torch, file_fingerprint, reseed_all,
    run_resumable, unit_seed,
)
from xgf.utils.io import save_json  # noqa: E402
from xgf.utils.seed import set_seed  # noqa: E402

# Thousands of forwards per image, against ~2 for the gradient methods.
HEAVY_EXPLAINERS = ("rise", "lime")


def random_control_curve(det, images, operator, n_steps, donor_bank, seed,
                         batch_size, amp):
    """Deletion curve using random (non-explanatory) rankings.

    Implemented by handing ``perturbation_curve`` a smooth random saliency map,
    so the control traverses the identical code path and mask geometry as the
    salient arm -- the subtraction is only valid if nothing else differs.
    """
    g = torch.Generator().manual_seed(seed)
    B, _, H, W = images.shape
    low = torch.rand(B, 1, 14, 14, generator=g)
    sal = torch.nn.functional.interpolate(
        low, size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(1)
    return perturbation_curve(
        det, images, sal, operator=operator, mode="deletion", n_steps=n_steps,
        donor_bank=donor_bank, batch_size=batch_size, amp=amp, seed=seed,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--tag", default="all")
    ap.add_argument("--manifest", default=None)
    # NOTE: "mixed" trains on every family, so by construction it has no unseen
    # family and cannot produce a generalization gap. Q3 is answered by pairing
    # a single-family probe against a leave-one-out probe on the SAME held-out
    # family -- same head capacity, same test set, only training diversity
    # differs. So the useful default is single-family + LOO, not "mixed".
    ap.add_argument("--probes", nargs="*", default=None,
                    help="probe config tags; default = every single-family "
                         "and loo_* probe found on disk, plus 'mixed'")
    ap.add_argument("--explainers", nargs="*",
                    default=["gradcampp", "gradcam", "inputgrad", "random"])
    ap.add_argument("--heavy-probes", nargs="*", default=None,
                    help="run {} only on these probes (default: on every "
                         "probe). Summaries stay like-for-like: see "
                         "xgf.metrics.grid".format("/".join(HEAVY_EXPLAINERS)))
    ap.add_argument("--operators", nargs="*", default=["zero", "gray", "road_linear"])
    ap.add_argument("--families", nargs="*", default=None,
                    help="test families (default: all fake families present)")
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--n-steps", type=int, default=21)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rise-masks", type=int, default=1000)
    ap.add_argument("--lime-samples", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--fresh", action="store_true",
                    help="discard saved progress for this backbone/tag")
    args = ap.parse_args()

    cfg = RunConfig(device=args.device, seed=args.seed)
    set_seed(cfg.seed)
    PATHS.ensure()
    device = cfg.resolve_device()

    mname = args.manifest or (
        "manifest_{}.csv".format(args.tag) if args.tag != "all" else "manifest.csv"
    )
    df = load_manifest(mname)
    fams = args.families or sorted(set(df["family"].astype(str)) - {"real"})

    probes = args.probes
    if probes is None:
        found = sorted(
            p.stem.split("__")[-1]
            for p in PATHS.checkpoints.glob("probe__{}__*.pkl".format(args.backbone))
        )
        # single-family + leave-one-out give the seen/unseen contrast; "mixed"
        # is kept only as the upper-bound reference point.
        probes = [p for p in found if p in fams or p.startswith("loo_")]
        if "mixed" in found:
            probes.append("mixed")
        if not probes:
            print("No probes found. Run scripts/03_train_probes.py first.")
            sys.exit(1)

    print("[faithfulness] families={} probes={} operators={} explainers={}".format(
        fams, probes, args.operators, args.explainers))

    donors = DonorBank.from_manifest(
        subset(df, label=0, splits=["train"]), image_size=cfg.image_size,
        n=min(256, max(32, len(df) // 8)), seed=cfg.seed,
    )

    # What changes the numbers. The grid axes (probes, families, explainers,
    # operators) are deliberately NOT here -- they are part of each row's key,
    # so the grid can be widened on a later run. Batch size only chunks forward
    # passes and is left out so it can be lowered after an OOM.
    prog_dir = PATHS.progress / "faithfulness__{}__{}".format(args.backbone, args.tag)
    if args.fresh and prog_dir.exists():
        shutil.rmtree(prog_dir)
    log = RunLog(prog_dir / "log.jsonl", config={
        "backbone": args.backbone, "tag": args.tag,
        "manifest": file_fingerprint(PATHS.manifests / mname),
        "n_images": args.n_images, "n_steps": args.n_steps,
        "rise_masks": args.rise_masks, "lime_samples": args.lime_samples,
        "seed": cfg.seed, "image_size": cfg.image_size,
        "precision": feature_precision(torch.device(device), cfg.amp),
    })

    def reseed_cell(fam, op):
        # Shared by the random control and every explainer in the same
        # (family, operator) cell: both arms of corrected_faithfulness then see
        # the same stochastic fill and the same donors, image for image.
        s_ = unit_seed(cfg.seed, "curve", fam, op)
        reseed_all(s_)
        donors.reseed(s_)

    rows, n_resumed = [], 0
    for probe_tag in probes:
        try:
            clf, info = load_probe(args.backbone, probe_tag)
        except FileNotFoundError as exc:
            print("  [skip probe {}] {}".format(probe_tag, exc))
            continue
        # A retrained probe gets a new id, so its cells are recomputed rather
        # than mixed with results from the old one.
        pid = "{}_{}".format(probe_tag, file_fingerprint(
            probe_path(args.backbone, probe_tag))[:10])
        train_fams = info.get("train_families", [])

        def key(fam, ename, op):
            return (pid, fam, ename, op)

        probe_expl = [e for e in args.explainers
                      if e not in HEAVY_EXPLAINERS or args.heavy_probes is None
                      or probe_tag in args.heavy_probes]
        todo = [(f, e, o) for f in fams for e in probe_expl
                for o in args.operators if not log.done(key(f, e, o))]
        explainers = {}
        if not todo:
            print("\n  probe={:<12} all cells already done".format(probe_tag))
        else:
            det = Detector(args.backbone).to(device).eval()
            det = probe_to_detector(det, clf)
            for name in [e for e in args.explainers if any(e == t[1] for t in todo)]:
                kw = {}
                if name == "rise":
                    kw = dict(n_masks=args.rise_masks, batch_size=args.batch_size)
                elif name == "lime":
                    kw = dict(n_samples=args.lime_samples, batch_size=args.batch_size)
                try:
                    explainers[name] = build_explainer(name, det, seed=cfg.seed, **kw)
                except Exception as exc:
                    print("  [skip explainer {}] {}".format(name, exc))

        for fam in fams:
            fam_todo = [(e, o) for f, e, o in todo if f == fam and e in explainers]
            if not fam_todo:
                continue
            fam_df = subset(df, families=[fam], splits=["test"], label=1,
                            n=args.n_images, seed=cfg.seed)
            if len(fam_df) < 5:
                fam_df = subset(df, families=[fam], label=1, n=args.n_images,
                                seed=cfg.seed)
            if len(fam_df) < 5:
                print("  [skip family {}] too few images".format(fam))
                continue
            ds = ManifestDataset(fam_df, image_size=cfg.image_size, with_mask=True)
            items = [ds[i] for i in range(len(ds))]
            images = torch.stack([it["image"] for it in items])
            masks = torch.stack([it["mask"] for it in items])
            has_mask = np.array([it["has_mask"] for it in items], dtype=bool)

            with torch.no_grad():
                base_p = torch.cat([
                    torch.softmax(det(images[i:i + args.batch_size].to(device)),
                                  dim=1)[:, 1].cpu()
                    for i in range(0, len(images), args.batch_size)
                ]).numpy()

            seen = fam in train_fams
            print("\n  probe={:<12} family={:<14} {:<7} n={} base_p(fake)={:.3f}"
                  "  [{} cells to run]".format(
                      probe_tag, fam, "SEEN" if seen else "UNSEEN",
                      len(images), base_p.mean(), len(fam_todo)))

            # Saliency maps are computed once per (probe, family, explainer) and
            # reused across every operator -- that is what makes the operator
            # sweep affordable. They are also cached on disk: for RISE and LIME
            # they are the most expensive thing in the whole run.
            sal_cache = {}
            for ename in [e for e in explainers if any(e == t[0] for t in fam_todo)]:
                sal_file = prog_dir / "saliency" / "{}__{}__{}.pt".format(
                    pid, fam, ename)
                if sal_file.exists():
                    sal_cache[ename] = torch.load(sal_file)
                    continue
                expl = explainers[ename]
                s_ = unit_seed(cfg.seed, "saliency", fam, ename)
                reseed_all(s_)
                expl.reseed(s_)
                maps = []
                for i in tqdm(range(0, len(images), 8),
                              desc="    {}".format(ename), leave=False):
                    maps.append(expl(images[i:i + 8], target=1))
                sal_cache[ename] = torch.cat(maps)
                atomic_save_torch(sal_file, sal_cache[ename])

            for op in args.operators:
                op_todo = [e for e, o in fam_todo if o == op]
                if not op_todo:
                    continue
                ctrl_file = prog_dir / "control" / "{}__{}__{}.npz".format(
                    pid, fam, op)
                if ctrl_file.exists():
                    with np.load(ctrl_file) as z:
                        ctrl = CurveResult(z["fractions"], z["curves"], z["auc"])
                else:
                    reseed_cell(fam, op)
                    ctrl = random_control_curve(
                        det, images, op, args.n_steps, donors, cfg.seed,
                        args.batch_size, cfg.amp,
                    )
                    atomic_save_npz(ctrl_file, fractions=ctrl.fractions,
                                    curves=ctrl.curves, auc=ctrl.auc)
                for ename in op_todo:
                    sal = sal_cache[ename]
                    reseed_cell(fam, op)
                    res = perturbation_curve(
                        det, images, sal, operator=op, mode="deletion",
                        n_steps=args.n_steps, donor_bank=donors,
                        batch_size=args.batch_size, amp=cfg.amp, seed=cfg.seed,
                    )
                    corr = corrected_faithfulness(res, ctrl)
                    row = {
                        "backbone": args.backbone, "probe": probe_tag,
                        "train_families": ",".join(train_fams),
                        "test_family": fam, "seen": seen,
                        "explainer": ename, "operator": op,
                        "base_pfake": float(base_p.mean()),
                        "deletion_auc": corr["auc_salient"],
                        "random_auc": corr["auc_random"],
                        "delta_auc": corr["delta_auc_mean"],
                        "delta_sem": corr["delta_auc_sem"],
                        "cohens_d": corr["cohens_d"],
                        "p_value": corr["p_value"],
                        "n": corr["n"],
                    }
                    if has_mask.any():
                        loc = localization_summary(sal[has_mask], masks[has_mask])
                        row.update({"loc_" + k: v for k, v in loc.items()})
                    log.commit(key(fam, ename, op), row)
                    print("      {:<10} {:<14} del_auc={:.4f} rand={:.4f} "
                          "DELTA={:+.4f} d={:+.2f} p={:.3g}".format(
                              ename, op, row["deletion_auc"], row["random_auc"],
                              row["delta_auc"], row["cohens_d"], row["p_value"]))

        # Collect in grid order -- rows finished this run and resumed ones alike.
        for fam in fams:
            for ename in probe_expl:
                for op in args.operators:
                    if log.done(key(fam, ename, op)):
                        rows.append(log.get(key(fam, ename, op)))
                        n_resumed += (fam, ename, op) not in todo

    if n_resumed:
        print("\n[faithfulness] {} of {} rows resumed from {}".format(
            n_resumed, len(rows), log.path))

    if not rows:
        print("No results produced.")
        sys.exit(1)

    out = pd.DataFrame(rows)
    csv = PATHS.results / "faithfulness__{}__{}.csv".format(args.backbone, args.tag)
    out.to_csv(csv, index=False)
    print("\n[faithfulness] {} rows -> {}".format(len(out), csv))

    print("\n" + "=" * 78)
    print("Q1  Does the operator change the ranking of explanation methods?")
    print("=" * 78)
    # Explainers are compared on the cells where all of them ran, so a method
    # run on one probe is not ranked against methods averaged over thirteen.
    q1 = complete_cells(out)
    if len(q1) < len(out):
        print("  (over the {} probe/family/operator cells where every explainer "
              "ran: probes {})".format(
                  q1[["probe", "test_family", "operator"]].drop_duplicates().shape[0],
                  sorted(q1["probe"].unique())))
    piv = q1.pivot_table(index="explainer", columns="operator",
                         values="delta_auc", aggfunc="mean")
    print(piv.round(4).to_string())
    if piv.shape[1] > 1:
        from scipy import stats
        ops = list(piv.columns)
        for a, b in itertools.combinations(ops, 2):
            sub = piv[[a, b]].dropna()
            if len(sub) >= 3:
                rho, p = stats.spearmanr(sub[a], sub[b])
                flag = "  <-- RANKING DISAGREES" if rho < 0.7 else ""
                print("  rank corr {:<14} vs {:<14} rho={:+.3f} (p={:.3g}){}".format(
                    a, b, rho, p, flag))

    print("\n" + "=" * 78)
    print("Q3  Does TRAINING DIVERSITY buy faithfulness on an unseen family?")
    print("=" * 78)
    print("    Each held-out family is scored twice: once by a probe trained on")
    print("    ONE other family, once by a probe trained on the REST. Same head")
    print("    capacity, same test images -- only training diversity differs.")
    # Probes are compared over the explainers every probe was scored with.
    bal = balanced_explainers(out)
    div = bal[(~bal["seen"]) & (bal["explainer"] != "random")].copy()
    div["n_train_fams"] = div["train_families"].apply(
        lambda s: len(str(s).split(",")) if s else 0
    )
    if len(div) and div["n_train_fams"].nunique() > 1:
        tab = div.pivot_table(index="test_family", columns="n_train_fams",
                              values="delta_auc", aggfunc="mean")
        print("\n  mean unseen-family delta_auc, by number of training families:")
        print(tab.round(4).to_string())
        lo, hi = tab.columns.min(), tab.columns.max()
        paired = tab[[lo, hi]].dropna()
        if len(paired) >= 2:
            from scipy import stats
            t, pv = stats.ttest_rel(paired[hi], paired[lo])
            print("\n  {} -> {} training families: mean delta_auc {:+.4f} -> {:+.4f}"
                  "  (paired t={:.2f}, p={:.3g}, n={})".format(
                      lo, hi, paired[lo].mean(), paired[hi].mean(), t, pv,
                      len(paired)))
            verdict = ("diversity HELPS faithfulness"
                       if paired[hi].mean() > paired[lo].mean()
                       else "diversity does NOT help faithfulness")
            print("  -> {}".format(verdict))
    else:
        print("\n  Not enough probe configs to compare. Train the loo_* probes:")
        print("  python scripts/03_train_probes.py --backbone {} --tag {}".format(
            args.backbone, args.tag))

    print("\n" + "=" * 78)
    print("Q2  Faithfulness on seen vs unseen families, by probe config")
    print("=" * 78)
    real = bal[bal["explainer"] != "random"]
    g = real.groupby(["probe", "seen"])["delta_auc"].agg(["mean", "std", "count"])
    print(g.round(4).to_string())
    gap = real.pivot_table(index="probe", columns="seen", values="delta_auc",
                           aggfunc="mean")
    if True in gap.columns and False in gap.columns:
        gap["faithfulness_gap"] = gap[True] - gap[False]
        print("\nFaithfulness generalization gap (seen - unseen):")
        print(gap[["faithfulness_gap"]].round(4).to_string())
        print("\nCompare against the accuracy gap in "
              "results/accuracy_matrix__*.csv -- if the two gaps diverge, the "
              "detector is generalizing without its explanation generalizing.")

    save_json({"args": vars(args)},
              PATHS.results / "faithfulness_args__{}__{}.json".format(
                  args.backbone, args.tag))
    print("\nNEXT: python scripts/06_make_figures.py --backbone {} --tag {}".format(
        args.backbone, args.tag))


if __name__ == "__main__":
    run_resumable(main)
