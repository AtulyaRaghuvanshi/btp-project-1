"""Q1/Q2/Q3 summaries of a finished faithfulness grid, as tables.

``scripts/05_run_faithfulness.py`` prints these at the end of a run and
``scripts/07_summarize.py`` rebuilds them from CSVs already on disk; both go
through ``summarize`` so the saved numbers and the printed ones cannot drift.

  Q1  explainer x operator mean DeltaAUC, per-operator ranks, and Spearman rank
      agreement between every pair of operators (over ``complete_cells``)
  Q2  seen vs unseen DeltaAUC per probe, and the faithfulness gap next to the
      accuracy gap from ``accuracy_matrix__*.csv`` (over ``balanced_explainers``)
  Q3  unseen-family DeltaAUC by number of training families, with a paired
      t-test between the fewest and the most (single-family vs ``loo_*``)
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from xgf.metrics.grid import balanced_explainers, complete_cells
from xgf.utils.checkpoint import atomic_save_text

# Below this, two operators are said to rank explanation methods differently.
RANK_DISAGREE_RHO = 0.7


@dataclass
class Analysis:
    backbone: str
    tag: str
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    scalars: Dict[str, object] = field(default_factory=dict)


def _n_train_fams(s) -> int:
    return len(str(s).split(",")) if isinstance(s, str) and s else 0


def summarize(df: pd.DataFrame, backbone: str, tag: str,
              acc: Optional[pd.DataFrame] = None) -> Analysis:
    a = Analysis(backbone, tag)

    # ---- Q1: does the operator change the ranking of explanation methods? --
    q1 = complete_cells(df)
    a.scalars["q1_probes"] = sorted(q1["probe"].unique().tolist())
    a.scalars["q1_n_cells"] = int(
        q1[["probe", "test_family", "operator"]].drop_duplicates().shape[0])
    a.scalars["q1_subset"] = len(q1) < len(df)
    piv = q1.pivot_table(index="explainer", columns="operator",
                         values="delta_auc", aggfunc="mean")
    a.tables["q1_method_by_operator"] = piv
    a.tables["q1_ranks"] = piv.rank(ascending=False, method="min")
    agree = []
    for o1, o2 in itertools.combinations(list(piv.columns), 2):
        sub = piv[[o1, o2]].dropna()
        if len(sub) < 3:
            continue
        rho, p = stats.spearmanr(sub[o1], sub[o2])
        agree.append({"operator_a": o1, "operator_b": o2, "spearman_rho": rho,
                      "p_value": p, "n_explainers": len(sub),
                      "disagrees": bool(rho < RANK_DISAGREE_RHO)})
    a.tables["q1_rank_agreement"] = pd.DataFrame(
        agree, columns=["operator_a", "operator_b", "spearman_rho", "p_value",
                        "n_explainers", "disagrees"])

    # ---- Q3: does training diversity buy faithfulness on an unseen family? -
    bal = balanced_explainers(df)
    div = bal[(~bal["seen"]) & (bal["explainer"] != "random")].copy()
    div["n_train_fams"] = div["train_families"].apply(_n_train_fams)
    a.scalars["q3_explainers"] = sorted(div["explainer"].unique().tolist())
    if len(div) and div["n_train_fams"].nunique() > 1:
        tab = div.pivot_table(index="test_family", columns="n_train_fams",
                              values="delta_auc", aggfunc="mean")
        a.tables["q3_diversity"] = tab
        lo, hi = tab.columns.min(), tab.columns.max()
        paired = tab[[lo, hi]].dropna()
        if len(paired) >= 2:
            t, pv = stats.ttest_rel(paired[hi], paired[lo])
            a.scalars["q3_test"] = {
                "n_fams_lo": int(lo), "n_fams_hi": int(hi),
                "mean_lo": float(paired[lo].mean()),
                "mean_hi": float(paired[hi].mean()),
                "t": float(t), "p_value": float(pv), "n": int(len(paired)),
                "diversity_helps": bool(paired[hi].mean() > paired[lo].mean()),
            }

    # ---- Q2: faithfulness on seen vs unseen families, by probe config ------
    real = bal[bal["explainer"] != "random"]
    a.tables["q2_seen_unseen"] = (
        real.groupby(["probe", "seen"])["delta_auc"]
        .agg(["mean", "std", "count"]).reset_index())
    fg = real.pivot_table(index="probe", columns="seen", values="delta_auc",
                          aggfunc="mean")
    if True in fg.columns and False in fg.columns:
        gap = pd.DataFrame({"faith_seen": fg[True], "faith_unseen": fg[False]})
        gap["faithfulness_gap"] = gap["faith_seen"] - gap["faith_unseen"]
        if acc is not None:
            ag = acc.pivot_table(index="train_config", columns="seen",
                                 values="auc", aggfunc="mean")
            if True in ag.columns and False in ag.columns:
                gap["auc_seen"] = ag[True]
                gap["auc_unseen"] = ag[False]
                gap["accuracy_gap"] = gap["auc_seen"] - gap["auc_unseen"]
        gap.index.name = "probe"
        a.tables["q2_gap"] = gap
    return a


def _fmt(df: pd.DataFrame) -> str:
    return df.round(4).to_string()


def report(a: Analysis) -> str:
    """The console text scripts/05 has always printed, built from ``a``."""
    bar = "=" * 78
    out: List[str] = []
    out += ["", bar,
            "Q1  Does the operator change the ranking of explanation methods?",
            bar]
    if a.scalars["q1_subset"]:
        out.append("  (over the {} probe/family/operator cells where every "
                   "explainer ran: probes {})".format(
                       a.scalars["q1_n_cells"], a.scalars["q1_probes"]))
    out.append(_fmt(a.tables["q1_method_by_operator"]))
    for r in a.tables["q1_rank_agreement"].itertuples():
        out.append("  rank corr {:<14} vs {:<14} rho={:+.3f} (p={:.3g}){}".format(
            r.operator_a, r.operator_b, r.spearman_rho, r.p_value,
            "  <-- RANKING DISAGREES" if r.disagrees else ""))

    out += ["", bar,
            "Q3  Does TRAINING DIVERSITY buy faithfulness on an unseen family?",
            bar,
            "    Each held-out family is scored twice: once by a probe trained on",
            "    ONE other family, once by a probe trained on the REST. Same head",
            "    capacity, same test images -- only training diversity differs."]
    if "q3_diversity" in a.tables:
        out += ["", "  mean unseen-family delta_auc, by number of training families:",
                _fmt(a.tables["q3_diversity"])]
        t = a.scalars.get("q3_test")
        if t:
            out += ["", "  {} -> {} training families: mean delta_auc {:+.4f} -> "
                    "{:+.4f}  (paired t={:.2f}, p={:.3g}, n={})".format(
                        t["n_fams_lo"], t["n_fams_hi"], t["mean_lo"], t["mean_hi"],
                        t["t"], t["p_value"], t["n"]),
                    "  -> diversity {} faithfulness".format(
                        "HELPS" if t["diversity_helps"] else "does NOT help")]
    else:
        out += ["", "  Not enough probe configs to compare. Train the loo_* probes:",
                "  python scripts/03_train_probes.py --backbone {} --tag {}".format(
                    a.backbone, a.tag)]

    out += ["", bar, "Q2  Faithfulness on seen vs unseen families, by probe config",
            bar, _fmt(a.tables["q2_seen_unseen"].set_index(["probe", "seen"]))]
    if "q2_gap" in a.tables:
        gap = a.tables["q2_gap"]
        cols = [c for c in ("faithfulness_gap", "accuracy_gap") if c in gap]
        out += ["", "Faithfulness generalization gap (seen - unseen){}:".format(
            ", beside the accuracy gap" if "accuracy_gap" in gap else ""),
                _fmt(gap[cols])]
        if "accuracy_gap" not in gap:
            out.append("\nCompare against the accuracy gap in "
                       "results/accuracy_matrix__*.csv -- if the two gaps "
                       "diverge, the detector is generalizing without its "
                       "explanation generalizing.")
    return "\n".join(out)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def flat_tables(a: Analysis) -> Dict[str, pd.DataFrame]:
    """Every table with its index as columns and a leading ``backbone``
    column, so tables from different backbones stack without bookkeeping."""
    out = {}
    for name, t in a.tables.items():
        t = t.reset_index() if not isinstance(t.index, pd.RangeIndex) else t.copy()
        t.columns = [str(c) for c in t.columns]
        t.insert(0, "backbone", a.backbone)
        out[name] = t
    return out


def save_analysis(a: Analysis, out_dir: str | Path) -> List[Path]:
    """Write every table as ``<table>__<backbone>__<tag>.csv``, plus the
    scalars as JSON and the text report."""
    out_dir = Path(out_dir)
    sfx = "{}__{}".format(a.backbone, a.tag)
    written = []
    for name, t in flat_tables(a).items():
        written.append(atomic_save_text(
            out_dir / "{}__{}.csv".format(name, sfx), t.to_csv(index=False)))
    written.append(atomic_save_text(
        out_dir / "analysis__{}.json".format(sfx),
        json.dumps({"backbone": a.backbone, "tag": a.tag, **a.scalars},
                   indent=2, default=_json_default)))
    written.append(atomic_save_text(
        out_dir / "analysis__{}.txt".format(sfx), report(a) + "\n"))
    return written
