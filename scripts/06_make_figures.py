"""Generate the paper figures from the CSVs the experiment scripts wrote.

    python scripts/06_make_figures.py --backbone resnet50 --tag selftest

Figures:
  F1  leakage curves per operator (the C1 money figure)
  F2  leakage ranking bar chart, injecting vs neutral
  F3  explanation-method ranking under each operator (the C3 money figure)
  F4  accuracy gap vs faithfulness gap, per probe config (the C2 money figure)
  F5  faithfulness transfer heatmap
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from xgf.config import PATHS  # noqa: E402
from xgf.utils.io import load_json  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": True,
})
INJ = "#c1432f"      # artifact-injecting
NEU = "#2b6cb0"      # artifact-neutral


def _save(fig, name: str) -> None:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    out = PATHS.figures / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("  [fig] {}".format(out))


def fig_leakage_curves(backbone: str, probe: str) -> None:
    p = PATHS.results / "leakage_curves__{}__{}.json".format(backbone, probe)
    csv = PATHS.results / "leakage__{}__{}.csv".format(backbone, probe)
    if not p.exists():
        return
    curves = load_json(p)
    meta = pd.read_csv(csv).set_index("operator")["class"].to_dict()

    # Distinct hue per operator, with warm/cool families and solid/dashed
    # linestyle both encoding the injecting-vs-neutral hypothesis.
    warm = ["#8c1d12", "#c1432f", "#e06a3b", "#f0a04b", "#b5651d"]
    cool = ["#1a365d", "#2b6cb0", "#4fa3d1"]
    inj_ops = [o for o in curves if meta.get(o) == "injecting"]
    neu_ops = [o for o in curves if meta.get(o) != "injecting"]
    style = {o: (warm[i % len(warm)], "-") for i, o in enumerate(inj_ops)}
    style.update({o: (cool[i % len(cool)], "--") for i, o in enumerate(neu_ops)})

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    baseline = None
    for op in inj_ops + neu_ops:
        d = curves[op]
        c, ls = style[op]
        baseline = d["mean_pfake"][0]
        ax.plot(d["fractions"], d["mean_pfake"], color=c, ls=ls, lw=2.0,
                label="{}  ({})".format(op, meta.get(op, "other")), alpha=0.9)
    if baseline is not None:
        ax.axhline(baseline, color="k", lw=1.2, ls="-", alpha=0.7)
        ax.annotate("unperturbed baseline — an unbiased operator stays here",
                    xy=(0.03, baseline), xytext=(0.03, baseline + 0.008),
                    fontsize=7.5, color="#333", va="bottom")
        ax.set_ylim(bottom=baseline - 0.03)
    ax.set_xlabel("fraction of the image masked at random")
    ax.set_ylabel("P(fake)  on images that are REAL")
    ax.set_title("F1  Perturbation leakage: how much fakeness the measuring\n"
                 "instrument manufactures on its own", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5),
              framealpha=0.95, title="fill operator", title_fontsize=8)
    _save(fig, "F1_leakage_curves__{}__{}.png".format(backbone, probe))


def fig_leakage_ranking(backbone: str, probe: str) -> None:
    csv = PATHS.results / "leakage__{}__{}.csv".format(backbone, probe)
    if not csv.exists():
        return
    df = pd.read_csv(csv).sort_values("leakage_delta_auc")
    colors = [INJ if c == "injecting" else NEU for c in df["class"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(df["operator"], df["leakage_delta_auc"], color=colors)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_xlabel("leakage  (mean rise in P(fake) on real images)")
    ax.set_title("F2  How much fakeness each operator manufactures", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=INJ),
               plt.Rectangle((0, 0), 1, 1, color=NEU)]
    ax.legend(handles, ["artifact-injecting", "artifact-neutral"], fontsize=7)
    _save(fig, "F2_leakage_ranking__{}__{}.png".format(backbone, probe))


def fig_method_ranking(backbone: str, tag: str) -> None:
    csv = PATHS.results / "faithfulness__{}__{}.csv".format(backbone, tag)
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    piv = df.pivot_table(index="explainer", columns="operator",
                         values="delta_auc", aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(piv.index))
    w = 0.8 / max(len(piv.columns), 1)
    for i, op in enumerate(piv.columns):
        ax.bar(x + i * w, piv[op].values, w, label=op)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(piv.index, rotation=20, ha="right")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_ylabel("leakage-corrected faithfulness  $\\Delta$AUC")
    ax.set_title("F3  Does the fill operator change which method 'wins'?",
                 fontsize=10)
    ax.legend(fontsize=7, title="operator", title_fontsize=7)
    _save(fig, "F3_method_ranking__{}__{}.png".format(backbone, tag))


def fig_gap_scatter(backbone: str, tag: str) -> None:
    fcsv = PATHS.results / "faithfulness__{}__{}.csv".format(backbone, tag)
    acsv = PATHS.results / "accuracy_matrix__{}__{}.csv".format(backbone, tag)
    if not (fcsv.exists() and acsv.exists()):
        return
    fdf = pd.read_csv(fcsv)
    adf = pd.read_csv(acsv)
    fdf = fdf[fdf["explainer"] != "random"]

    fgap = fdf.pivot_table(index="probe", columns="seen", values="delta_auc",
                           aggfunc="mean")
    agap = adf.pivot_table(index="train_config", columns="seen", values="auc",
                           aggfunc="mean")
    if True not in fgap.columns or False not in fgap.columns:
        return
    if True not in agap.columns or False not in agap.columns:
        return
    f = (fgap[True] - fgap[False]).rename("faithfulness_gap")
    a = (agap[True] - agap[False]).rename("accuracy_gap")
    j = pd.concat([a, f], axis=1).dropna()
    if j.empty:
        return

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(j["accuracy_gap"], j["faithfulness_gap"], s=60,
               c=["#2b6cb0" if i == "mixed" else "#888" for i in j.index],
               zorder=3)
    for name, r in j.iterrows():
        ax.annotate(name, (r["accuracy_gap"], r["faithfulness_gap"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    lim = [min(j.min().min(), 0) - 0.05, max(j.max().max(), 0) + 0.05]
    ax.plot(lim, lim, ls=":", color="k", lw=0.9, label="equal degradation")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("accuracy gap  (seen - unseen AUC)")
    ax.set_ylabel("faithfulness gap  (seen - unseen $\\Delta$AUC)")
    ax.set_title("F4  Does the explanation generalize\nas well as the decision?",
                 fontsize=10)
    ax.legend(fontsize=7)
    _save(fig, "F4_gap_scatter__{}__{}.png".format(backbone, tag))


def fig_transfer_heatmap(backbone: str, tag: str) -> None:
    csv = PATHS.results / "faithfulness__{}__{}.csv".format(backbone, tag)
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df = df[df["explainer"] != "random"]
    piv = df.pivot_table(index="probe", columns="test_family",
                         values="delta_auc", aggfunc="mean")
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    im = ax.imshow(piv.values, cmap="RdYlBu", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, "{:.3f}".format(v), ha="center", va="center",
                        fontsize=7)
    ax.set_title("F5  Faithfulness transfer matrix ($\\Delta$AUC)", fontsize=10)
    ax.set_xlabel("test family")
    ax.set_ylabel("train config")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    _save(fig, "F5_transfer_heatmap__{}__{}.png".format(backbone, tag))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--tag", default="all")
    ap.add_argument("--probe", default="mixed")
    args = ap.parse_args()

    print("[figures] ->", PATHS.figures)
    fig_leakage_curves(args.backbone, args.probe)
    fig_leakage_ranking(args.backbone, args.probe)
    fig_method_ranking(args.backbone, args.tag)
    fig_gap_scatter(args.backbone, args.tag)
    fig_transfer_heatmap(args.backbone, args.tag)
    print("[figures] done")


if __name__ == "__main__":
    main()
