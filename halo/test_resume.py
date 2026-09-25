"""Crash-safety tests for the resume layer.

These are the tests that matter most on a machine that loses power without
warning. Everything else in the project can be recomputed; a corrupted ledger or
a half-written checkpoint silently poisons every result that follows it, and the
damage is not visible until someone checks a number by hand weeks later.

Each test simulates a specific way a process can die mid-write:

    T1  atomic write            a reader never observes a partial file
    T2  no debris               a completed write leaves no .tmp behind
    T3  torn ledger line        a power cut mid-append costs one unit, not the run
    T4  corrupt ledger line     garbage in the middle truncates, does not crash
    T5  completed units skipped a resumed run does not redo banked work
    T6  config invalidation     changing a parameter retires stale units
    T7  corrupt checkpoint      an unreadable parquet is quarantined, not trusted
    T8  stale tmp sweep         debris from an earlier death is cleared on start

Run: ``python -m halo.cli_v2 selftest``
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .resume import (Ledger, atomic_save_df, atomic_write_bytes, run_resumable,
                     safe_load_df, sweep_stale_tmp, work_key)


def _t1_atomic_write(tmp: Path) -> dict:
    p = tmp / "a.bin"
    atomic_write_bytes(p, b"hello" * 1000)
    return {"test": "T1 atomic write", "bytes": p.stat().st_size,
            "passed": p.read_bytes() == b"hello" * 1000}


def _t2_no_debris(tmp: Path) -> dict:
    p = tmp / "b.bin"
    atomic_write_bytes(p, b"x" * 64)
    leftovers = list(tmp.glob("*.tmp*"))
    return {"test": "T2 no debris", "leftover_tmp": len(leftovers),
            "passed": not leftovers}


def _t3_torn_line(tmp: Path) -> dict:
    """A power cut during the final append must cost exactly that one unit."""
    path = tmp / "led_torn.jsonl"
    led = Ledger(path, config_hash="c1")
    for i in range(5):
        led.record(work_key({"i": i}, "c1"), {"i": i, "auprc": 0.1 * i})

    # simulate: the 6th line reached the disk only partially
    with open(path, "ab") as fh:
        fh.write(b'{"i": 5, "auprc": 0.5, "key": "deadbe')

    reopened = Ledger(path, config_hash="c1")
    intact = len(reopened) == 5
    still_appendable = True
    try:
        reopened.record(work_key({"i": 5}, "c1"), {"i": 5, "auprc": 0.5})
    except Exception:
        still_appendable = False
    return {"test": "T3 torn ledger line", "survivors": 5, "recovered": len(reopened) - 1,
            "appendable_after_repair": still_appendable,
            "passed": intact and still_appendable and len(reopened) == 6}


def _t4_corrupt_line(tmp: Path) -> dict:
    """Garbage in the middle truncates from that point rather than raising."""
    path = tmp / "led_corrupt.jsonl"
    led = Ledger(path, config_hash="c1")
    for i in range(4):
        led.record(work_key({"i": i}, "c1"), {"i": i})
    with open(path, "ab") as fh:
        fh.write(b"\x00\xff not json at all \n")
        fh.write(json.dumps({"i": 99, "key": "later"}).encode() + b"\n")
    reopened = Ledger(path, config_hash="c1")
    return {"test": "T4 corrupt ledger line", "kept": len(reopened),
            "passed": len(reopened) == 4}


def _t5_skip_completed(tmp: Path) -> dict:
    """The whole point: a resumed run must not redo banked work."""
    path = tmp / "led_skip.jsonl"
    units = [{"i": i} for i in range(6)]
    calls: list[int] = []

    def fn(u):
        calls.append(u["i"])
        return {"auprc": 0.01 * u["i"]}

    led = Ledger(path, config_hash="c1")
    run_resumable(units[:3], fn, led, label="first")
    first = list(calls)

    calls.clear()
    led2 = Ledger(path, config_hash="c1")       # fresh process
    run_resumable(units, fn, led2, label="resumed")
    return {"test": "T5 completed units skipped",
            "first_pass": first, "second_pass": list(calls),
            "passed": first == [0, 1, 2] and calls == [3, 4, 5]}


def _t6_config_invalidation(tmp: Path) -> dict:
    """Changing configuration must retire stale units, not silently mix them."""
    path = tmp / "led_cfg.jsonl"
    units = [{"i": i} for i in range(3)]
    calls: list[int] = []

    def fn(u):
        calls.append(u["i"])
        return {"auprc": 0.0}

    run_resumable(units, fn, Ledger(path, config_hash="c1"), label="cfg1")
    calls.clear()
    run_resumable(units, fn, Ledger(path, config_hash="c2"), label="cfg2")
    return {"test": "T6 config invalidation", "rerun_under_new_config": len(calls),
            "passed": len(calls) == 3}


def _t7_corrupt_checkpoint(tmp: Path) -> dict:
    """An unreadable checkpoint must be quarantined, never half-trusted."""
    df = pd.DataFrame({"a": np.arange(50, dtype=np.float32)})
    p = atomic_save_df(df, tmp / "ckpt")
    back = safe_load_df(p)
    roundtrip = back is not None and len(back) == 50

    p.write_bytes(b"this is not a parquet file")
    recovered = safe_load_df(p)
    quarantined = list(tmp.glob("*.corrupt*"))
    return {"test": "T7 corrupt checkpoint",
            "roundtrip_ok": roundtrip,
            "returns_none_on_corrupt": recovered is None,
            "quarantined": len(quarantined),
            "passed": roundtrip and recovered is None and len(quarantined) == 1}


def _t8_stale_tmp_sweep(tmp: Path) -> dict:
    fresh = tmp / "fresh.parquet.tmp"
    old = tmp / "old.parquet.tmp"
    fresh.write_bytes(b"x")
    old.write_bytes(b"x")
    ancient = time.time() - 7200
    os.utime(old, (ancient, ancient))
    removed = sweep_stale_tmp(tmp, older_than_s=3600)
    return {"test": "T8 stale tmp sweep", "removed": removed,
            "kept_fresh": fresh.exists(),
            "passed": removed == 1 and fresh.exists() and not old.exists()}


TESTS = (_t1_atomic_write, _t2_no_debris, _t3_torn_line, _t4_corrupt_line,
         _t5_skip_completed, _t6_config_invalidation, _t7_corrupt_checkpoint,
         _t8_stale_tmp_sweep)


def run_all(verbose: bool = True) -> pd.DataFrame:
    import tempfile

    rows = []
    for fn in TESTS:
        with tempfile.TemporaryDirectory() as d:
            try:
                r = fn(Path(d))
            except Exception as e:
                r = {"test": fn.__name__, "passed": False,
                     "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        if verbose:
            mark = "PASS" if r.get("passed") else "FAIL"
            detail = {k: v for k, v in r.items() if k not in ("test", "passed")}
            print(f"  [{mark}] {r['test']}")
            for k, v in detail.items():
                print(f"          {k}: {v}")
    return pd.DataFrame(rows)
