"""Small utilities shared across experiment runners.

Torch-free on purpose: safe to import from modules that defer torch loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _deep_merge(base, update):
    """Recursively merge `update` into `base`, preserving nested keys.

    Prevents partial reruns from nuking sibling results. Rerunning a single
    model in the scaling sweep merges into the existing models dict rather
    than replacing the entire key.
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _save_results(results, filename="transformer_observe.json"):
    """Deep-merge results into existing JSON file and save."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / filename
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
    _deep_merge(existing, results)
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {out_file} (keys: {sorted(existing.keys())})")


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=0):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return lo, hi
