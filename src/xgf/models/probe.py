"""Feature caching and linear-probe training.

This is what keeps the project inside a 4 GB laptop GPU (or no GPU at all):

    extract features once  ->  train probes in seconds  ->  spend all remaining
    compute on the faithfulness evaluation, which is the actual contribution.

Training a probe per *generator family* is cheap enough that the full
train-family x test-family transfer matrix is a few minutes of CPU time once
features exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import PATHS
from ..data.datasets import make_loader
from .detector import Detector


def feature_path(backbone: str, tag: str = "all") -> Path:
    return PATHS.features / "{}__{}.npz".format(backbone, tag)


@torch.no_grad()
def extract_features(detector: Detector, df: pd.DataFrame, image_size: int = 224,
                     batch_size: int = 32, num_workers: int = 0,
                     amp: bool = True, desc: str = "features") -> Dict[str, np.ndarray]:
    """Run the frozen backbone over a manifest slice and cache pooled features."""
    device = next(detector.parameters()).device
    use_amp = amp and device.type == "cuda"
    loader = make_loader(df, image_size=image_size, batch_size=batch_size,
                         num_workers=num_workers, shuffle=False)
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device, non_blocking=True)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.float16):
                f = detector.features(x)
        else:
            f = detector.features(x)
        feats.append(f.float().cpu().numpy())
        labels.append(batch["label"].numpy())
    return {
        "features": np.concatenate(feats).astype(np.float32),
        "labels": np.concatenate(labels).astype(np.int64),
        "generator": df["generator"].to_numpy(dtype=object),
        "family": df["family"].to_numpy(dtype=object),
        "split": df["split"].to_numpy(dtype=object),
        "path": df["path"].to_numpy(dtype=object),
    }


def save_features(cache: Dict[str, np.ndarray], backbone: str, tag: str = "all") -> Path:
    out = feature_path(backbone, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cache)
    print("[features] {} x {} -> {}".format(*cache["features"].shape, out))
    return out


def load_features(backbone: str, tag: str = "all") -> Dict[str, np.ndarray]:
    p = feature_path(backbone, tag)
    if not p.exists():
        raise FileNotFoundError(
            "{} missing. Run: python scripts/02_extract_features.py".format(p)
        )
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def train_probe(cache: Dict[str, np.ndarray], train_families: Sequence[str],
                C: float = 1.0, max_iter: int = 2000, seed: int = 1337,
                balance: bool = True):
    """Fit a logistic-regression probe on one family subset.

    Features are L2-normalised first (standard for CLIP probes, and it makes
    the regularisation strength comparable across backbones).

    Returns ``(classifier, info_dict)``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = cache["features"], cache["labels"]
    fam = cache["family"].astype(str)
    split = cache["split"].astype(str)

    keep_fam = set(train_families) | {"real"}
    tr = (split == "train") & np.isin(fam, list(keep_fam))
    va = (split == "val") & np.isin(fam, list(keep_fam))
    if tr.sum() < 20:
        raise RuntimeError(
            "Only {} train rows for families {}".format(int(tr.sum()), train_families)
        )

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    clf = LogisticRegression(
        C=C, max_iter=max_iter, random_state=seed,
        class_weight="balanced" if balance else None,
    )
    clf.fit(Xn[tr], y[tr])

    info = {
        "train_families": list(train_families),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "train_acc": float(clf.score(Xn[tr], y[tr])),
    }
    if va.sum() > 10 and len(np.unique(y[va])) > 1:
        p = clf.predict_proba(Xn[va])[:, 1]
        info["val_auc"] = float(roc_auc_score(y[va], p))
        info["val_acc"] = float(clf.score(Xn[va], y[va]))
    return clf, info


def evaluate_probe_per_family(clf, cache: Dict[str, np.ndarray],
                              split: str = "test") -> pd.DataFrame:
    """Per-family detection AUC on held-out data.

    This is the *accuracy* half of the central comparison; the faithfulness
    half comes from the perturbation experiments. Plotting one against the
    other is the cross-generator result.
    """
    from sklearn.metrics import roc_auc_score

    X, y = cache["features"], cache["labels"]
    fam = cache["family"].astype(str)
    sp = cache["split"].astype(str)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    sel = sp == split
    real = sel & (fam == "real")
    rows = []
    for f in sorted(set(fam[sel]) - {"real"}):
        idx = sel & (fam == f)
        both = idx | real
        if both.sum() < 10 or len(np.unique(y[both])) < 2:
            continue
        p = clf.predict_proba(Xn[both])[:, 1]
        rows.append({
            "test_family": f,
            "auc": float(roc_auc_score(y[both], p)),
            "acc": float(clf.score(Xn[both], y[both])),
            "n_fake": int(idx.sum()),
            "n_real": int(real.sum()),
        })
    return pd.DataFrame(rows)


def probe_to_detector(detector: Detector, clf) -> Detector:
    """Install a fitted sklearn probe into the torch detector head.

    After this the detector is a single differentiable module from [0,1] pixels
    to logits, which is what Grad-CAM and the perturbation metrics need.

    Note the L2 normalisation applied at fit time is folded in by rescaling --
    strictly the head becomes exact only for unit-norm features, so we wrap the
    head in a normalising layer instead of approximating.
    """
    import torch.nn as nn

    class _L2Head(nn.Module):
        def __init__(self, linear: nn.Linear):
            super().__init__()
            self.linear = linear

        def forward(self, f):
            f = f / (f.norm(dim=1, keepdim=True) + 1e-8)
            return self.linear(f)

    w = np.asarray(clf.coef_, dtype=np.float32).reshape(-1)
    b = float(np.asarray(clf.intercept_, dtype=np.float32).reshape(-1)[0])
    lin = nn.Linear(w.shape[0], 2)
    with torch.no_grad():
        lin.weight.copy_(torch.from_numpy(np.stack([-w / 2.0, w / 2.0])))
        lin.bias.copy_(torch.tensor([-b / 2.0, b / 2.0], dtype=torch.float32))
    device = next(detector.parameters()).device
    detector.head = _L2Head(lin).to(device)
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)
    return detector


def save_probe(clf, info: Dict, backbone: str, tag: str) -> Path:
    import pickle

    out = PATHS.checkpoints / "probe__{}__{}.pkl".format(backbone, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump({"clf": clf, "info": info}, fh)
    return out


def load_probe(backbone: str, tag: str) -> Tuple[object, Dict]:
    import pickle

    p = PATHS.checkpoints / "probe__{}__{}.pkl".format(backbone, tag)
    if not p.exists():
        raise FileNotFoundError(
            "{} missing. Run: python scripts/03_train_probes.py".format(p)
        )
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    return d["clf"], d["info"]
