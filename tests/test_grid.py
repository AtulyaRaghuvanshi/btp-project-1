"""Like-for-like slicing of a partially run faithfulness grid."""
from __future__ import annotations

import itertools

import pandas as pd

from xgf.metrics.grid import balanced_explainers, complete_cells


def _grid(heavy_probes=("mixed",)):
    """Cheap explainers on every probe, rise only on ``heavy_probes``."""
    rows = []
    for probe, fam, op in itertools.product(
            ["gan", "loo_gan", "mixed"], ["gan", "diffusion"], ["zero", "road_linear"]):
        for e in ["gradcam", "random"] + (["rise"] if probe in heavy_probes else []):
            rows.append({"probe": probe, "test_family": fam, "operator": op,
                         "explainer": e, "delta_auc": 0.0})
    return pd.DataFrame(rows)


def test_complete_grid_is_unchanged():
    df = _grid(heavy_probes=("gan", "loo_gan", "mixed"))
    assert len(complete_cells(df)) == len(df)
    assert len(balanced_explainers(df)) == len(df)


def test_explainer_ranking_uses_only_cells_with_every_explainer():
    out = complete_cells(_grid())
    assert set(out["probe"]) == {"mixed"}
    # every explainer scored on the same 4 cells
    assert out.groupby("explainer").size().tolist() == [4, 4, 4]


def test_probe_comparison_drops_explainers_missing_from_some_probes():
    out = balanced_explainers(_grid())
    assert set(out["explainer"]) == {"gradcam", "random"}
    assert set(out["probe"]) == {"gan", "loo_gan", "mixed"}


def test_empty_frame_passes_through():
    empty = _grid().iloc[:0]
    assert complete_cells(empty).empty and balanced_explainers(empty).empty
