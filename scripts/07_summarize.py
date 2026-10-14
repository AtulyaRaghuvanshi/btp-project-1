"""Save the Q1/Q2/Q3 analysis tables from faithfulness CSVs already on disk.

scripts/05 prints and saves these at the end of a run; this rebuilds them
without rerunning anything, e.g. for results copied back from the server:

    python scripts/07_summarize.py --tag c2500 --results-dir results_server/results

Per backbone, into ``<results-dir>/analysis/``:
  q1_method_by_operator  explainer x operator mean DeltaAUC
  q1_ranks               the same, as per-operator ranks (1 = most faithful)
  q1_rank_agreement      Spearman rho between every pair of operators
  q2_seen_unseen         DeltaAUC on seen vs unseen families, per probe
  q2_gap                 faithfulness gap beside the accuracy gap, per probe
  q3_diversity           unseen-family DeltaAUC by number of training families
  analysis__*.json       scalars (Q3 paired t-test, which cells Q1 used)
  analysis__*.txt        the console report

With more than one backbone, also ``<table>__ALL__<tag>.csv`` (all backbones
stacked) and ``headline__<tag>.csv`` (one row per backbone).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from xgf.config import PATHS  # noqa: E402
from xgf.metrics.analysis import (  # noqa: E402
    flat_tables, report, save_analysis, summarize,
)
from xgf.utils.checkpoint import atomic_save_text  # noqa: E402


def headline(a) -> dict:
    ranks = a.tables["q1_ranks"]
    agree = a.tables["q1_rank_agreement"]
    row = {"backbone": a.backbone,
           "q1_n_cells": a.scalars["q1_n_cells"],
           "q1_rho_mean": agree["spearman_rho"].mean(),
           "q1_rho_min": agree["spearman_rho"].min(),
           "q1_pairs_disagree": int(agree["disagrees"].sum()),
           "q1_pairs": len(agree)}
    # which method each operator ranks first, and where LIME lands
    # (Tsigos et al. MAD'24 called LIME the most faithful)
    for op in ranks.columns:
        row["top__" + op] = ranks[op].idxmin()
    if "lime" in ranks.index:
        row["lime_rank_best"] = ranks.loc["lime"].min()
        row["lime_rank_worst"] = ranks.loc["lime"].max()
    t = a.scalars.get("q3_test")
    if t:
        row.update({"q3_mean_lo": t["mean_lo"], "q3_mean_hi": t["mean_hi"],
                    "q3_t": t["t"], "q3_p": t["p_value"], "q3_n": t["n"]})
    gap = a.tables.get("q2_gap")
    if gap is not None and "accuracy_gap" in gap:
        g = gap.dropna(subset=["faithfulness_gap", "accuracy_gap"])
        row["q2_gap_corr"] = g["faithfulness_gap"].corr(g["accuracy_gap"])
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="all")
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="default: every faithfulness__<bb>__<tag>.csv found")
    ap.add_argument("--results-dir", default=None,
                    help="where the CSVs are (default: results/)")
    ap.add_argument("--out-dir", default=None,
                    help="default: <results-dir>/analysis")
    ap.add_argument("--quiet", action="store_true", help="do not print reports")
    args = ap.parse_args()

    rdir = Path(args.results_dir) if args.results_dir else PATHS.results
    odir = Path(args.out_dir) if args.out_dir else rdir / "analysis"
    bbs = args.backbones or sorted(
        p.name.split("__")[1] for p in rdir.glob("faithfulness__*__{}.csv".format(args.tag)))
    if not bbs:
        sys.exit("no faithfulness__*__{}.csv in {}".format(args.tag, rdir))

    analyses = []
    for bb in bbs:
        fcsv = rdir / "faithfulness__{}__{}.csv".format(bb, args.tag)
        acsv = rdir / "accuracy_matrix__{}__{}.csv".format(bb, args.tag)
        df = pd.read_csv(fcsv)
        a = summarize(df, bb, args.tag,
                      acc=pd.read_csv(acsv) if acsv.exists() else None)
        if not args.quiet:
            print("\n#### {} ({} rows)".format(bb, len(df)))
            print(report(a))
        save_analysis(a, odir)
        analyses.append(a)
        print("[analysis] {} -> {}".format(bb, odir))

    if len(analyses) > 1:
        flat = [flat_tables(a) for a in analyses]
        for n in sorted({n for f in flat for n in f}):
            stacked = pd.concat([f[n] for f in flat if n in f], ignore_index=True)
            atomic_save_text(odir / "{}__ALL__{}.csv".format(n, args.tag),
                             stacked.to_csv(index=False))
        head = pd.DataFrame([headline(a) for a in analyses])
        atomic_save_text(odir / "headline__{}.csv".format(args.tag),
                         head.to_csv(index=False))
        print("\n[analysis] headline across backbones:")
        print(head.set_index("backbone").T.to_string())


if __name__ == "__main__":
    main()
