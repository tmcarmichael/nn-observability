"""Analysis library for the nn-observability research program.

Loaders read committed JSONs from `../results/` into typed Python structures.
Statistical primitives operate on those structures.

Loaders:
    load_all_models             every paper-scope model with metadata
    load_model_means            (family, log10(params), pcorr) tuples
    load_per_seed               per-seed values for variance analysis
    load_control_sensitivity    control-strength rows for the waterfall
    load_random_head_baselines  random-probe baseline values

Statistical primitives:
    family_f_stat               family effect F-statistic with scale residualization

Schema validation:
    validate_results_json       check a single JSON against the required schema
    validate_all                check every paper-scope JSON, return failure count

Stable across the v3.x line.
"""

from __future__ import annotations

from analysis.load_results import (
    RESULTS_DIR,
    load_all_models,
    load_control_sensitivity,
    load_model_means,
    load_per_seed,
    load_random_head_baselines,
    validate_all,
    validate_results_json,
)
from analysis.permutation_test import family_f_stat

__all__ = [
    "RESULTS_DIR",
    "family_f_stat",
    "load_all_models",
    "load_control_sensitivity",
    "load_model_means",
    "load_per_seed",
    "load_random_head_baselines",
    "validate_all",
    "validate_results_json",
]
