from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timer(name: str, sink: Dict[str, float] | None = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if sink is not None:
            sink[name] = sink.get(name, 0.0) + dt
        print(f"[timer] {name}: {dt:.2f}s")
