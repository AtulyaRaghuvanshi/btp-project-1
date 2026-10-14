"""Q1/Q2/Q3 summaries are computed once and saved, not only printed."""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from xgf.metrics.analysis import save_analysis, summarize

FAMS = ["gan", "diffusion", "inpainting"]
PROBES = {"gan": "gan", "diffusion": "diffusion", "inpainting": "inpainting",
          "loo_gan": "diffusion,inpainting", "loo_diffusion": "gan,inpainting",
          "loo_inpainting": "diffusion,gan", "mixed": "diffusion,gan,inpainting"}
# zero ranks a > b > c, road_linear reverses it
SCORE = {"zero": {"a": 0.3, "b": 0.2, "c": 0.1, "random": 0.0},
         "road_linear": {"a": 0.1, "b": 0.2, "c": 0.3, "random": 0.0}}


def _grid():
    rows = []
    for (probe, train), fam, op, e in itertools.product(
            PROBES.items(), FAMS, SCORE, ["a", "b", "c", "random"]):
        seen = fam in train.split(",")
        k = len(train.split(","))
        rows.append({"probe": probe, "train_families": train, "test_family": fam,
                     "seen": seen, "explainer": e, "operator": op,
                     # unseen faithfulness rises with training diversity,
                     # by a family-dependent amount (mean 0.06 per family)
                     "delta_auc": SCORE[op][e] + (
                         0.0 if seen else (0.05 + 0.01 * FAMS.index(fam)) * k)})
    return pd.DataFrame(rows)


def _acc():
    return pd.DataFrame([{"train_config": p, "test_family": f,
                          "seen": f in t.split(","), "auc": 0.9 if f in t else 0.6}
                         for p, t in PROBES.items() for f in FAMS])


def test_rank_disagreement_is_flagged():
    a = summarize(_grid(), "bb", "t")
    agree = a.tables["q1_rank_agreement"]
    assert len(agree) == 1
    # a,b,c reverse; random is last under both -> rho = 1 - 6*8/(4*15)
    assert agree["spearman_rho"].iloc[0] == pytest.approx(0.2)
    assert agree["disagrees"].iloc[0]
    ranks = a.tables["q1_ranks"]
    assert ranks.loc["a", "zero"] == 1 and ranks.loc["c", "road_linear"] == 1


def test_diversity_contrast_pairs_single_family_with_loo():
    a = summarize(_grid(), "bb", "t")
    t = a.scalars["q3_test"]
    assert (t["n_fams_lo"], t["n_fams_hi"], t["n"]) == (1, 2, 3)
    assert t["mean_hi"] - t["mean_lo"] == pytest.approx(0.06)
    assert t["diversity_helps"]
    assert "random" not in a.scalars["q3_explainers"]


def test_gap_table_carries_accuracy_gap():
    gap = summarize(_grid(), "bb", "t", acc=_acc()).tables["q2_gap"]
    assert gap.loc["gan", "accuracy_gap"] == pytest.approx(0.3)
    # mixed has no unseen family: its gap is undefined, not zero
    assert pd.isna(gap.loc["mixed", "faithfulness_gap"])


def test_saved_tables_round_trip(tmp_path):
    a = summarize(_grid(), "bb", "t", acc=_acc())
    written = {p.name for p in save_analysis(a, tmp_path)}
    assert {"q1_rank_agreement__bb__t.csv", "q3_diversity__bb__t.csv",
            "q2_gap__bb__t.csv", "analysis__bb__t.json",
            "analysis__bb__t.txt"} <= written
    piv = pd.read_csv(tmp_path / "q1_method_by_operator__bb__t.csv")
    assert list(piv.columns[:2]) == ["backbone", "explainer"]
    assert piv.set_index("explainer").loc["a", "zero"] == pytest.approx(
        a.tables["q1_method_by_operator"].loc["a", "zero"])
