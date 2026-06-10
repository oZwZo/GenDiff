"""
gendiff_dev.results.store — the single long-format results store (architecture §6.6).

Schema (one row per scalar): (dataset, target, model, metric, value, stratum).
Rows are appended to results/runs.jsonl and mirrored into results/runs.csv, so any (dataset, target,
model) combination from either direction is directly comparable in one table.
"""
from __future__ import annotations
import os, json, csv

RESULTS_ROOT = os.environ.get(
    "GENDIFF_RESULTS_ROOT", "/rds/user/wz369/hpc-work/GenDiff/gendiff_dev/results")
FIELDS = ["dataset", "target", "model", "metric", "value", "stratum"]


def append(rows, out=None):
    """Append result rows (list of dicts with FIELDS) to runs.jsonl + runs.csv."""
    if not rows: return
    root = out or RESULTS_ROOT; os.makedirs(root, exist_ok=True)
    jl = os.path.join(root, "runs.jsonl"); cs = os.path.join(root, "runs.csv")
    from .. import _now
    with open(jl, "a") as f:
        for r in rows: f.write(json.dumps({**r, "_created": _now()}) + "\n")
    new = not os.path.exists(cs)
    with open(cs, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new: w.writeheader()
        for r in rows: w.writerow(r)


def read(out=None):
    """Return all rows as a list of dicts (or a DataFrame if pandas is available)."""
    root = out or RESULTS_ROOT; jl = os.path.join(root, "runs.jsonl")
    if not os.path.exists(jl): return []
    rows = [json.loads(l) for l in open(jl)]
    try:
        import pandas as pd; return pd.DataFrame(rows)
    except Exception:
        return rows
