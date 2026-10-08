"""Crash-safe, resumable execution for the long pipeline stages.

Nothing here trains anything -- backbones are frozen and probes fit in seconds.
What takes hours is feature extraction (02), the leakage sweep (04) and the
faithfulness grid (05). Each is broken into *work units* that are committed to
disk the moment they finish, so a crash, a Ctrl+C or a laptop going to sleep
costs at most the unit in flight. Re-running the same command resumes.

Three guarantees, each of which the stages rely on:

1. **Atomic writes.** Every output is written to ``<name>.tmp`` and moved into
   place with ``os.replace``. A file that exists is complete; a half-written
   ``.npz`` can never be mistaken for a finished cache.

2. **Fingerprinted progress.** A progress log records the configuration that
   produced it. Resuming with different settings (another ``--n-images``, a
   re-extracted feature cache, a retrained probe) is refused instead of
   silently splicing two experiments into one results table.

3. **Order-independent randomness.** Each unit reseeds every RNG from its own
   identity (:func:`unit_seed`), not from how many units ran before it. That is
   what makes a resumed run produce the *same numbers* as an uninterrupted one.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


class StaleCheckpointError(RuntimeError):
    """Saved progress was produced by a different configuration."""


# --------------------------------------------------------------------------- #
# Atomic file writes
# --------------------------------------------------------------------------- #
def _replace_with_retry(src: Path, dst: Path, attempts: int = 40,
                        delay: float = 0.25) -> None:
    # On Windows, OneDrive and antivirus scanners briefly hold files open, and
    # os.replace then raises PermissionError. It clears within a second or two.
    for k in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if k == attempts - 1:
                raise
            time.sleep(delay)


def atomic_write(path: str | Path, write_fn: Callable[[Any], None]) -> Path:
    """Call ``write_fn(binary_file_handle)`` and publish the result atomically.

    A file handle rather than a path is passed on purpose: ``np.savez`` appends
    ``.npz`` to paths that lack it, which would break the temp-file rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write_fn(fh)
        fh.flush()
        os.fsync(fh.fileno())
    _replace_with_retry(tmp, path)
    return path


def atomic_save_npz(path: str | Path, /, **arrays: np.ndarray) -> Path:
    # positional-only: feature caches carry an array literally named "path"
    return atomic_write(path, lambda fh: np.savez_compressed(fh, **arrays))


def atomic_save_torch(path: str | Path, obj: Any) -> Path:
    import torch

    return atomic_write(path, lambda fh: torch.save(obj, fh))


def atomic_save_pickle(path: str | Path, obj: Any) -> Path:
    import pickle

    return atomic_write(path, lambda fh: pickle.dump(obj, fh))


# --------------------------------------------------------------------------- #
# Fingerprints and seeds
# --------------------------------------------------------------------------- #
def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def fingerprint(obj: Any) -> str:
    """Short stable hash of a JSON-able object (dict order does not matter)."""
    return hashlib.sha1(_canonical(obj).encode("utf-8")).hexdigest()[:16]


def file_fingerprint(path: str | Path, chunk: int = 1 << 22) -> str:
    """Content hash of a file.

    Content, not mtime: OneDrive rewrites mtimes when it re-syncs a file, which
    would make every resume look stale.
    """
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def unit_seed(base: int, *keys: Any) -> int:
    """Deterministic 31-bit seed for one work unit.

    Python's ``hash()`` is salted per process, so it cannot be used here.
    """
    digest = hashlib.sha1(_canonical([int(base), list(keys)]).encode("utf-8"))
    return int(digest.hexdigest()[:8], 16) & 0x7FFFFFFF


def reseed_all(seed: int) -> None:
    """Reset python / numpy / torch (CPU and CUDA) global RNGs."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds every CUDA device


# --------------------------------------------------------------------------- #
# Progress log: append-only JSONL of finished work units
# --------------------------------------------------------------------------- #
def _key(key: Any) -> str:
    return key if isinstance(key, str) else _canonical(list(key))


def _describe_mismatch(old: Dict, new: Dict) -> str:
    keys = sorted(set(old) | set(new))
    diffs = ["    {}: saved={!r}  now={!r}".format(k, old.get(k), new.get(k))
             for k in keys if old.get(k) != new.get(k)]
    return "\n".join(diffs) or "    (fingerprint differs)"


class RunLog:
    """Durable record of which work units are finished, and their results.

    One JSON object per line. The first line is a header holding the run's
    configuration; every later line is ``{"key": ..., "value": ...}``. Lines
    are fsync'd as they are written, so a crash loses at most the unit that was
    running. A torn final line (power cut mid-write) is detected and dropped.
    """

    def __init__(self, path: str | Path, config: Dict, fresh: bool = False):
        self.path = Path(path)
        self.config = dict(config)
        self.fp = fingerprint(self.config)
        self._done: Dict[str, Any] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh and self.path.exists():
            self.path.unlink()
        if self.path.exists():
            self._load()
        else:
            header = {"_header": True, "fingerprint": self.fp,
                      "config": self.config,
                      "created": datetime.now().isoformat(timespec="seconds")}
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(_canonical(header) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def _load(self) -> None:
        good_bytes = 0
        header = None
        with open(self.path, "rb") as fh:
            for raw in fh:
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    break                       # torn write; drop the rest
                if not raw.endswith(b"\n"):
                    break
                good_bytes += len(raw)
                if header is None:
                    header = rec
                    continue
                self._done[rec["key"]] = rec["value"]
        if header is None or not header.get("_header"):
            raise StaleCheckpointError(
                "{} is not a progress log. Delete it or pass --fresh.".format(self.path))
        if header["fingerprint"] != self.fp:
            raise StaleCheckpointError(
                "Saved progress in {} was produced by a different configuration:\n{}\n"
                "Re-run with the original settings to resume, or pass --fresh to "
                "discard it and start over.".format(
                    self.path, _describe_mismatch(header.get("config", {}), self.config)))
        if good_bytes < self.path.stat().st_size:
            with open(self.path, "r+b") as fh:
                fh.truncate(good_bytes)

    def __len__(self) -> int:
        return len(self._done)

    def done(self, key: Any) -> bool:
        return _key(key) in self._done

    def get(self, key: Any, default: Any = None) -> Any:
        return self._done.get(_key(key), default)

    def commit(self, key: Any, value: Any) -> None:
        from .io import _NpEncoder

        k = _key(key)
        line = json.dumps({"key": k, "value": value}, cls=_NpEncoder,
                          separators=(",", ":"))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        # Round-trip through JSON so resumed and fresh values are identical.
        self._done[k] = json.loads(line)["value"]


# --------------------------------------------------------------------------- #
# Row-range shards (feature extraction)
# --------------------------------------------------------------------------- #
class ShardCheckpoint:
    """Contiguous row-range shards of a large array computation.

    Shards are named by the row range they cover, not by batch index, so the
    batch size may change between the interrupted run and the resumed one.
    """

    def __init__(self, directory: str | Path, config: Dict, n_rows: int,
                 fresh: bool = False):
        self.dir = Path(directory)
        self.n_rows = int(n_rows)
        self.config = dict(config, n_rows=self.n_rows)
        self.fp = fingerprint(self.config)
        if fresh and self.dir.exists():
            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "meta.json"
        if meta.exists():
            with open(meta, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
            if saved.get("fingerprint") != self.fp:
                raise StaleCheckpointError(
                    "Partial results in {} came from a different configuration:\n{}\n"
                    "Re-run with the original settings to resume, or pass --fresh."
                    .format(self.dir, _describe_mismatch(saved.get("config", {}),
                                                         self.config)))
        else:
            payload = json.dumps({"fingerprint": self.fp, "config": self.config},
                                 indent=2, default=str).encode("utf-8")
            atomic_write(meta, lambda fh: fh.write(payload))

    def _shards(self) -> List[Tuple[int, int, Path]]:
        out = []
        for p in self.dir.glob("shard_*.npz"):
            try:
                _, a, b = p.stem.split("_")
                out.append((int(a), int(b), p))
            except ValueError:
                continue
        return sorted(out)

    def next_row(self) -> int:
        """First row not covered by the contiguous run of shards from row 0.

        Anything past a gap (should not happen with atomic writes, but a user
        could delete a shard by hand) is discarded and recomputed.
        """
        pos = 0
        for a, b, p in self._shards():
            if a == pos:
                pos = b
            elif a > pos or b > pos:
                p.unlink()
        return pos

    def write(self, start: int, stop: int, **arrays: np.ndarray) -> None:
        atomic_save_npz(self.dir / "shard_{:09d}_{:09d}.npz".format(start, stop),
                        **arrays)

    def load_all(self) -> Dict[str, np.ndarray]:
        parts: Dict[str, List[np.ndarray]] = {}
        pos = 0
        for a, b, p in self._shards():
            if a != pos:
                raise RuntimeError("Shard gap at row {} in {}".format(pos, self.dir))
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    parts.setdefault(k, []).append(z[k])
            pos = b
        if pos != self.n_rows:
            raise RuntimeError("Shards cover {} of {} rows in {}".format(
                pos, self.n_rows, self.dir))
        return {k: np.concatenate(v) for k, v in parts.items()}

    def discard(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Entry-point wrapper
# --------------------------------------------------------------------------- #
def run_resumable(main: Callable[[], None]) -> None:
    """Run a script's ``main`` and turn Ctrl+C / stale state into clear exits."""
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[interrupted] Finished work is saved. Re-run the same command "
              "to resume from where it stopped.", file=sys.stderr)
        sys.exit(130)
    except StaleCheckpointError as exc:
        print("\n[stale checkpoint] {}".format(exc), file=sys.stderr)
        sys.exit(2)

