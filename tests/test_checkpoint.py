"""Resumable-run guarantees. Run: python -m pytest tests/test_checkpoint.py -v

The property that matters scientifically is the last group: a resumed run must
produce the *same numbers* as an uninterrupted one, otherwise the results table
silently depends on when the laptop happened to crash.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from xgf.utils.checkpoint import (
    RunLog, ShardCheckpoint, StaleCheckpointError, atomic_write, reseed_all,
    unit_seed,
)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def test_atomic_write_leaves_no_temp_file(tmp_path):
    p = tmp_path / "sub" / "out.bin"
    atomic_write(p, lambda fh: fh.write(b"hello"))
    assert p.read_bytes() == b"hello"
    assert not list(tmp_path.rglob("*.tmp"))


def test_atomic_write_failure_keeps_previous_file(tmp_path):
    p = tmp_path / "out.bin"
    atomic_write(p, lambda fh: fh.write(b"old"))

    def boom(fh):
        fh.write(b"half")
        raise RuntimeError("crash mid-write")

    with pytest.raises(RuntimeError):
        atomic_write(p, boom)
    assert p.read_bytes() == b"old"


def test_unit_seed_is_stable_and_distinct():
    assert unit_seed(1337, "curve", "gan", "zero") == unit_seed(1337, "curve", "gan", "zero")
    assert unit_seed(1337, "curve", "gan", "zero") != unit_seed(1337, "curve", "gan", "gray")
    assert unit_seed(1337, "a") != unit_seed(1338, "a")
    assert 0 <= unit_seed(1, "x") < 2 ** 31


# --------------------------------------------------------------------------- #
# RunLog
# --------------------------------------------------------------------------- #
def test_runlog_roundtrip(tmp_path):
    cfg = {"n_images": 50, "seed": 1337}
    log = RunLog(tmp_path / "log.jsonl", cfg)
    log.commit(("mixed", "gan", "gradcam", "zero"), {"delta_auc": np.float32(0.25)})
    log.commit("op2", {"x": [1, 2]})

    again = RunLog(tmp_path / "log.jsonl", cfg)
    assert len(again) == 2
    assert again.done(("mixed", "gan", "gradcam", "zero"))
    assert again.get(("mixed", "gan", "gradcam", "zero"))["delta_auc"] == pytest.approx(0.25)
    assert not again.done(("mixed", "gan", "gradcam", "gray"))


def test_runlog_drops_torn_last_line_and_keeps_appending(tmp_path):
    cfg = {"a": 1}
    path = tmp_path / "log.jsonl"
    log = RunLog(path, cfg)
    log.commit("k1", 1)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"key": "k2", "val')          # power cut mid-write

    log = RunLog(path, cfg)
    assert log.done("k1") and not log.done("k2")
    log.commit("k2", 2)
    assert RunLog(path, cfg).get("k2") == 2


def test_runlog_refuses_different_config(tmp_path):
    RunLog(tmp_path / "log.jsonl", {"n_images": 50}).commit("k", 1)
    with pytest.raises(StaleCheckpointError, match="n_images"):
        RunLog(tmp_path / "log.jsonl", {"n_images": 200})


def test_runlog_fresh_discards(tmp_path):
    RunLog(tmp_path / "log.jsonl", {"n": 1}).commit("k", 1)
    log = RunLog(tmp_path / "log.jsonl", {"n": 2}, fresh=True)
    assert len(log) == 0


# --------------------------------------------------------------------------- #
# Feature extraction resume
# --------------------------------------------------------------------------- #
class _Crash(Exception):
    pass


class _TinyBackbone(nn.Module):
    """Deterministic stand-in for a frozen backbone; optionally dies mid-run."""

    def __init__(self, die_after=None):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(0)
            self.proj = nn.Linear(3 * 4 * 4, 5)
        self.calls, self.die_after = 0, die_after

    def features(self, x):
        self.calls += 1
        if self.die_after is not None and self.calls > self.die_after:
            raise _Crash()
        x = nn.functional.adaptive_avg_pool2d(x, 4).flatten(1)
        return self.proj(x)


def _tiny_manifest(tmp_path, n=23):
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        p = tmp_path / "img_{:03d}.png".format(i)
        Image.fromarray(rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(p)
        rows.append({"path": str(p), "label": i % 2, "generator": "g",
                     "family": "gan" if i % 2 else "real", "split": "train",
                     "mask_path": ""})
    return pd.DataFrame(rows)


def test_feature_extraction_resume_is_bit_identical(tmp_path):
    from xgf.models.probe import extract_features

    df = _tiny_manifest(tmp_path)
    kw = dict(image_size=16, amp=False, desc="t")
    ref = extract_features(_TinyBackbone(), df, batch_size=4, **kw)

    cfg = {"backbone": "tiny"}
    ck = ShardCheckpoint(tmp_path / "partial", cfg, n_rows=len(df))
    with pytest.raises(_Crash):
        extract_features(_TinyBackbone(die_after=3), df, batch_size=4,
                         checkpoint=ck, checkpoint_every=2, **kw)
    assert ck.next_row() == 8                     # 2 batches of 4 made it to disk

    # Resume with a *different* batch size -- shards are row ranges, not batches.
    ck = ShardCheckpoint(tmp_path / "partial", cfg, n_rows=len(df))
    model = _TinyBackbone()
    out = extract_features(model, df, batch_size=5, checkpoint=ck,
                           checkpoint_every=2, **kw)
    assert model.calls == 3                       # only rows 8..22 recomputed
    np.testing.assert_array_equal(out["features"], ref["features"])
    np.testing.assert_array_equal(out["labels"], ref["labels"])


def test_shard_checkpoint_refuses_different_config(tmp_path):
    ShardCheckpoint(tmp_path / "p", {"precision": "fp32"}, n_rows=10)
    with pytest.raises(StaleCheckpointError, match="precision"):
        ShardCheckpoint(tmp_path / "p", {"precision": "fp16"}, n_rows=10)


# --------------------------------------------------------------------------- #
# Order independence of the stochastic arms
# --------------------------------------------------------------------------- #
class _MeanDetector(nn.Module):
    def forward(self, x01):
        logit = (x01.mean(dim=(1, 2, 3)) - 0.5) * 10.0
        return torch.stack([-logit, logit], dim=1)

    def parameters(self, recurse=True):
        yield torch.nn.Parameter(torch.zeros(1))


@pytest.mark.parametrize("op", ["real_transplant", "uniform_noise", "spectral_match"])
def test_reseeded_curve_does_not_depend_on_previous_cells(op):
    """Without per-cell reseeding, the donors and noise a cell draws depend on
    how many cells ran before it in the same process -- so a resumed run
    would not reproduce an uninterrupted one."""
    from xgf.metrics.faithfulness import perturbation_curve
    from xgf.perturb.operators import DonorBank

    det = _MeanDetector()
    images = torch.rand(3, 3, 16, 16)
    sal = torch.rand(3, 16, 16)
    bank = DonorBank(torch.rand(8, 3, 16, 16))

    def cell():
        s = unit_seed(1337, "curve", "gan", op)
        reseed_all(s)
        bank.reseed(s)
        return perturbation_curve(det, images, sal, operator=op, n_steps=5,
                                  donor_bank=bank, amp=False).curves

    fresh = cell()
    # Burn RNG state as earlier cells in an uninterrupted run would.
    torch.rand(1000)
    bank.sample(17)
    after_other_cells = cell()
    np.testing.assert_array_equal(fresh, after_other_cells)


def test_random_explainer_reseed():
    from xgf.explain.base import build_explainer
    from xgf.explain import methods  # noqa: F401

    expl = build_explainer("random", _MeanDetector(), seed=1)
    x = torch.rand(2, 3, 16, 16)
    expl.reseed(7)
    a = expl(x)
    expl(x)                                       # advance the internal RNG
    expl.reseed(7)
    torch.testing.assert_close(expl(x), a)


def test_save_features_accepts_a_path_array(tmp_path, monkeypatch):
    """Regression: the cache has an array called "path", which once collided
    with the save helper's own ``path`` argument."""
    from xgf.config import PATHS
    from xgf.models.probe import load_features, save_features

    monkeypatch.setattr(PATHS, "features", tmp_path)
    cache = {"features": np.ones((2, 3), np.float32),
             "path": np.array(["a.png", "b.png"], dtype=object)}
    save_features(cache, "bb", tag="t")
    assert list(load_features("bb", tag="t")["path"]) == ["a.png", "b.png"]
