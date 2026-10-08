"""Like-for-like slices of a faithfulness grid that was not run in full.

The grid is (probe, test_family, explainer, operator), but it need not be
complete: the expensive explainers (RISE, LIME) are typically run on one probe
only (``05_run_faithfulness.py --heavy-probes``). Averaging such a grid over
everything compares explainers on different cells -- RISE scored on ``mixed``
against Grad-CAM averaged over every probe -- and the comparison is then partly
about which cells each method happened to be run on. Each summary must
therefore average over a slice in which the axis being compared is balanced:

  * comparing EXPLAINERS (method rankings)  -> ``complete_cells``
  * comparing PROBES or FAMILIES (gaps, transfer matrices) -> ``balanced_explainers``

On a complete grid both return the input unchanged.
"""
from __future__ import annotations

import pandas as pd

CELL = ["probe", "test_family", "operator"]


def complete_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the (probe, test_family, operator) cells that contain every
    explainer present in ``df`` -- so each explainer is averaged over the same
    cells."""
    if df.empty:
        return df
    n = df["explainer"].nunique()
    k = df.groupby(CELL)["explainer"].transform("nunique")
    return df[k == n]


def balanced_explainers(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the explainers that were run in every (probe, test_family,
    operator) cell -- so each probe or family is averaged over the same
    explainers."""
    if df.empty:
        return df
    n_cells = df[CELL].drop_duplicates().shape[0]
    per_expl = df[CELL + ["explainer"]].drop_duplicates().groupby("explainer").size()
    keep = per_expl[per_expl == n_cells].index
    return df[df["explainer"].isin(keep)]
