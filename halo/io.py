"""Checkpointing and environment capture.

Kaggle sessions die. Every stage persists its output here so a timeout costs one
stage, never the whole run. Uses parquet when pyarrow is present, pickle otherwise.
"""
from __future__ import annotations

import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CKPT_DIR, RESULTS_DIR
from .resume import atomic_save_df, atomic_write_bytes, atomic_write_text, safe_load_df

try:  # pragma: no cover - environment dependent
    import pyarrow  # noqa: F401
    _HAVE_PARQUET = True
except ImportError:
    _HAVE_PARQUET = False


# --------------------------------------------------------------------------------------
# Dataframe checkpoints
# --------------------------------------------------------------------------------------

def save_df(df: pd.DataFrame, name: str) -> Path:
    """Crash-safe checkpoint write (tmp -> fsync -> atomic rename).

    Writing straight to the target path is unsafe where power can be lost
    mid-write: it leaves a truncated file that ``has_ckpt`` then reports as
    valid forever. See ``resume.atomic_save_df``.
    """
    ext = ".parquet" if _HAVE_PARQUET else ".pkl"
    return atomic_save_df(df, CKPT_DIR / f"{name}{ext}")


def load_df(name: str) -> pd.DataFrame | None:
    """Load a checkpoint. An unreadable file is quarantined and treated as absent."""
    for ext in (".parquet", ".pkl"):
        path = CKPT_DIR / f"{name}{ext}"
        if path.exists():
            df = safe_load_df(path)
            if df is not None:
                return df
    return None


def has_ckpt(name: str) -> bool:
    return any((CKPT_DIR / f"{name}{e}").exists() for e in (".parquet", ".pkl"))


def save_obj(obj, name: str) -> Path:
    return atomic_write_bytes(CKPT_DIR / f"{name}.pkl", pickle.dumps(obj))


def load_obj(name: str):
    path = CKPT_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


# --------------------------------------------------------------------------------------
# Result tables
# --------------------------------------------------------------------------------------

def save_table(df: pd.DataFrame, table_id: str, title: str = "") -> Path:
    """Persist a result table as CSV plus a sidecar of metadata."""
    path = RESULTS_DIR / f"{table_id}.csv"
    atomic_write_text(path, df.to_csv(index=False))
    meta = {"table_id": table_id, "title": title,
            "rows": int(len(df)), "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    atomic_write_text(RESULTS_DIR / f"{table_id}.meta.json", json.dumps(meta, indent=2))
    return path


def load_table(table_id: str) -> pd.DataFrame | None:
    path = RESULTS_DIR / f"{table_id}.csv"
    return pd.read_csv(path) if path.exists() else None


# --------------------------------------------------------------------------------------
# Environment manifest -- goes verbatim into the report
# --------------------------------------------------------------------------------------

def environment_manifest() -> dict:
    """Observe the actual runtime limits rather than trusting remembered numbers."""
    man = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import psutil
        vm = psutil.virtual_memory()
        man["ram_total_gb"] = round(vm.total / 1e9, 2)
        man["ram_available_gb"] = round(vm.available / 1e9, 2)
    except ImportError:
        man["ram_total_gb"] = None

    try:
        import shutil
        usage = shutil.disk_usage(str(CKPT_DIR))
        man["disk_free_gb"] = round(usage.free / 1e9, 2)
    except OSError:
        man["disk_free_gb"] = None

    for lib in ("numpy", "pandas", "sklearn", "scipy", "lightgbm",
                "xgboost", "catboost", "shap", "matplotlib"):
        try:
            mod = __import__(lib)
            man[f"v_{lib}"] = getattr(mod, "__version__", "?")
        except ImportError:
            man[f"v_{lib}"] = "MISSING"
    man["parquet_available"] = _HAVE_PARQUET
    return man


# --------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------

def downcast(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Shrink numeric dtypes in place. IEEE-CIS is ~1.5 GB after this, from ~3 GB."""
    before = df.memory_usage(deep=True).sum()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_integer_dtype(s):
            df[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            df[col] = s.astype(np.float32)
    after = df.memory_usage(deep=True).sum()
    if verbose:
        print(f"  downcast {before/1e6:.0f} MB -> {after/1e6:.0f} MB")
    return df


class Timer:
    """Context manager that records wall time for the report's runtime table."""

    _log: list[dict] = []

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        Timer._log.append({"stage": self.label, "seconds": round(dt, 2)})
        print(f"  [{dt:7.2f}s] {self.label}")
        return False

    @classmethod
    def table(cls) -> pd.DataFrame:
        return pd.DataFrame(cls._log)
