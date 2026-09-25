"""Crash-safe checkpointing and resumable work loops.

Written for an environment where the machine can lose power mid-write at any
instant (load-shedding), and for Kaggle sessions that are killed at a hard wall
clock limit. Two guarantees:

1. **No torn files.** Every write goes to a temporary path, is fsynced, then
   atomically renamed onto the target. A power cut therefore leaves either the
   old complete file or the new complete file -- never a half-written one that
   ``has_ckpt`` would later report as valid.

2. **Per-unit resume.** Work is enumerated as independent units (one model fit
   on one fold with one seed). Completed units are appended to a JSONL ledger,
   one fsynced line each, so an interruption costs at most the unit in flight.
   On restart the ledger is repaired (a torn trailing line is dropped) and
   completed units are skipped.

The unit of loss is therefore one fit -- typically 60-90 seconds -- rather than
one stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd


# --------------------------------------------------------------------------------------
# Durable, atomic writes
# --------------------------------------------------------------------------------------

def _fsync_dir(path: Path) -> None:
    """Flush the directory entry so the rename itself survives power loss.

    POSIX only. Windows has no directory handle to fsync and NTFS journals the
    rename, so this is a no-op there.
    """
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _tmp_path(path: Path) -> Path:
    # PID-tagged so two workers never collide on the same temporary file.
    return path.with_name(f"{path.name}.tmp{os.getpid()}")


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)          # atomic on POSIX and on NTFS
    _fsync_dir(path.parent)
    return path


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_save_df(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe crash-safely.

    Honours a ``.parquet`` suffix when an engine is installed, and falls back to
    pickle at the same path otherwise, so a missing pyarrow degrades rather than
    raising mid-run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    if path.suffix == ".parquet":
        try:
            # The index is preserved deliberately. The pickle fallback below keeps
            # it, so dropping it here would mean a cache entry read back with
            # pyarrow installed differs from the same entry read back without it
            # -- a difference that would appear only after someone installs a
            # package, which is a miserable thing to debug.
            df.to_parquet(tmp)
        except (ImportError, ValueError):
            df.to_pickle(tmp)      # no parquet engine; same path, pickle payload
    else:
        df.to_pickle(tmp)
    # pandas closed the handle without fsyncing; reopen to force the flush.
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return path


def safe_load_df(path: str | Path) -> pd.DataFrame | None:
    """Load a checkpoint, returning None (not raising) if it is unreadable.

    Belt and braces: atomic writes should make torn files impossible, but a
    checkpoint written by an older build may still be corrupt. Treat an
    unreadable checkpoint as absent so the work is simply redone.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            try:
                return pd.read_parquet(path)
            except (ImportError, ValueError):
                return pd.read_pickle(path)   # written by the pickle fallback above
        return pd.read_pickle(path)
    except Exception:
        quarantine = path.with_name(path.name + ".corrupt")
        try:
            os.replace(path, quarantine)
            print(f"  [resume] unreadable checkpoint quarantined -> {quarantine.name}")
        except OSError:
            pass
        return None


def prune_cache(directory: str | Path, budget_gb: float,
                pattern: str = "feat_*") -> int:
    """Keep a cache directory under a size budget, evicting least-recently-used.

    Feature tables are large and numerous: one per (dataset, fold, config), and
    at TabFormer scale each is several hundred megabytes. Caching all of them
    costs far more disk than a Kaggle session has, and filling the disk mid-run
    fails the write that was supposed to protect progress -- turning a space
    problem into a lost run.

    Eviction is safe by construction: a cache entry is a pure function of its
    key, so deleting one costs recomputation and nothing else. Never called on
    ledgers or results, only on entries matching ``pattern``.
    """
    directory = Path(directory)
    if not directory.exists() or budget_gb <= 0:
        return 0
    budget = budget_gb * (1024 ** 3)
    files = []
    for p in directory.glob(pattern):
        if p.suffix in (".parquet", ".pkl"):
            try:
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_atime, st.st_size, p))
    total = sum(f[1] for f in files)
    if total <= budget:
        return 0
    removed = 0
    for _, size, p in sorted(files):               # oldest access first
        if total <= budget:
            break
        try:
            p.unlink()
            total -= size
            removed += 1
        except OSError:
            pass
    if removed:
        print(f"  [cache] evicted {removed} cached feature tables "
              f"to stay under {budget_gb:g} GB")
    return removed


def dir_size_gb(directory: str | Path, pattern: str = "*") -> float:
    directory = Path(directory)
    if not directory.exists():
        return 0.0
    total = 0
    for p in directory.glob(pattern):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total / (1024 ** 3)


def sweep_stale_tmp(directory: str | Path, older_than_s: float = 3600) -> int:
    """Delete abandoned .tmp files left by a process that died mid-write."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    now, removed = time.time(), 0
    for p in directory.glob("*.tmp*"):
        try:
            if now - p.stat().st_mtime > older_than_s:
                p.unlink()
                removed += 1
        except OSError:
            pass
    return removed


# --------------------------------------------------------------------------------------
# Work units
# --------------------------------------------------------------------------------------

def work_key(unit: dict, config_hash: str = "") -> str:
    """Deterministic identity for one unit of work.

    ``config_hash`` is folded in so that changing a hyperparameter invalidates
    previously completed units rather than silently mixing two configurations
    into one results table.
    """
    payload = json.dumps(unit, sort_keys=True, default=str) + "|" + config_hash
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Append-only ledger
# --------------------------------------------------------------------------------------

class Ledger:
    """Crash-tolerant JSONL record of completed work units.

    Appending one fsynced line per unit is far safer under power loss than
    rewriting a CSV: the worst case is a torn final line, which ``repair``
    truncates on the next start.
    """

    def __init__(self, path: str | Path, config_hash: str = ""):
        self.path = Path(path)
        self.config_hash = config_hash
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        self._records: list[dict] = []
        self.repair()

    # ---- recovery --------------------------------------------------------------------
    def repair(self) -> int:
        """Load the ledger, discarding a truncated trailing line. Returns bytes dropped."""
        if not self.path.exists():
            return 0
        raw = self.path.read_bytes()
        offset, done, records = 0, set(), []
        while True:
            nl = raw.find(b"\n", offset)
            if nl == -1:
                break                          # trailing partial line -> discard
            line = raw[offset:nl]
            if line.strip():
                try:
                    rec = json.loads(line)
                    key = rec["key"]
                except (ValueError, KeyError, TypeError):
                    # ValueError covers both JSONDecodeError and UnicodeDecodeError.
                    # The latter matters more than it looks: an unclean shutdown
                    # commonly leaves the tail of a file padded with NUL bytes, and
                    # json.loads on bytes then guesses UTF-16 and raises a decode
                    # error rather than a JSON error. Catching only JSONDecodeError
                    # turns the single most likely form of power-cut damage into an
                    # exception that aborts the resume it was supposed to survive.
                    break                      # corrupt line -> stop, truncate here
                done.add(key)
                records.append(rec)
            offset = nl + 1

        dropped = len(raw) - offset
        if dropped:
            with open(self.path, "rb+") as fh:
                fh.truncate(offset)
                fh.flush()
                os.fsync(fh.fileno())
            print(f"  [resume] repaired ledger, dropped {dropped} trailing bytes")
        self._done, self._records = done, records
        return dropped

    # ---- use -------------------------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self._done

    def __len__(self) -> int:
        return len(self._done)

    def record(self, key: str, payload: dict) -> None:
        rec = {**payload, "key": key, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        line = json.dumps(rec, default=str) + "\n"
        with open(self.path, "ab") as fh:
            fh.write(line.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())              # the line that makes this power-safe
        self._done.add(key)
        self._records.append(rec)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)


# --------------------------------------------------------------------------------------
# Resumable driver
# --------------------------------------------------------------------------------------

def run_resumable(
    units: Sequence[dict],
    fn: Callable[[dict], dict],
    ledger: Ledger,
    *,
    max_hours: float | None = None,
    label: str = "run",
) -> pd.DataFrame:
    """Execute units, skipping completed ones, recording each on completion.

    Parameters
    ----------
    units
        Every unit of work, enumerated up front in a deterministic order.
    fn
        ``fn(unit) -> dict`` of metrics. Must be free of side effects that would
        make a re-run of an interrupted unit incorrect.
    max_hours
        Stop cleanly before this budget is spent. Set it just under a Kaggle
        session cap (e.g. 11.5), or to the time remaining before a scheduled
        load-shedding window, so the process exits having recorded its work
        rather than being killed mid-unit.

    Re-running after any interruption is always safe: completed units are
    skipped, and the unit that was in flight is simply redone.
    """
    t_start = time.perf_counter()
    todo = [(work_key(u, ledger.config_hash), u) for u in units]
    todo = [(k, u) for k, u in todo if k not in ledger]

    print(f"[{label}] {len(units)} units total, {len(units) - len(todo)} already done, "
          f"{len(todo)} to run")

    for i, (key, unit) in enumerate(todo, 1):
        if max_hours is not None:
            elapsed_h = (time.perf_counter() - t_start) / 3600
            if elapsed_h >= max_hours:
                print(f"[{label}] time budget reached ({elapsed_h:.2f}h); "
                      f"{len(todo) - i + 1} units remain. Re-run to continue.")
                break

        t0 = time.perf_counter()
        result = fn(unit)
        ledger.record(key, {
            **{f"u_{k}": v for k, v in unit.items()},
            **result,
            "seconds": round(time.perf_counter() - t0, 2),
        })
        print(f"[{label}] {i}/{len(todo)}  {time.perf_counter()-t0:6.1f}s  "
              f"{ {k: unit[k] for k in list(unit)[:3]} }")

    return ledger.to_frame()


def expand_grid(**axes: Iterable[Any]) -> list[dict]:
    """Cartesian product of named axes, in deterministic order.

    >>> expand_grid(rung=["r0", "r1"], seed=[0, 1])
    [{'rung': 'r0', 'seed': 0}, {'rung': 'r0', 'seed': 1}, ...]
    """
    import itertools
    keys = list(axes.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]
