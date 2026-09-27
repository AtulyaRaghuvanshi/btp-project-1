"""Small compatibility shims.

numpy renamed ``trapz`` to ``trapezoid`` in 2.0 and deprecated the old name.
The project has to run on both, because torch wheels lag numpy releases.
"""
from __future__ import annotations

import numpy as np

if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:                                   # numpy < 2.0
    trapezoid = np.trapz


def ptp(a) -> float:
    """``np.ptp`` as a free function (the ndarray method went away in 2.0)."""
    return float(np.ptp(a))
