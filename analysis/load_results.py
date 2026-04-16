"""Shared data loading for all analysis scripts.

Single source of truth for which result files to load and their
metadata. Update this file when results land or scope changes.

Usage: from load_results import load_all_models, load_per_seed, load_control_sensitivity
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── Schema for paper-scope results JSONs ──────────────────────────────

# Required fields and their types for any results file used in the paper.
# Checked on load; violations are printed as warnings (not exceptions)
# so partially complete files can still be used during development.

REQUIRED_FIELDS = {
    "model": str,
    "partial_corr.mean": (int, float),
    "partial_corr.per_seed": list,
    "output_controlled.mean": (int, float),
    "peak_layer_frac": (int, float),
    "seed_agreement": (dict, int, float),
    "baselines": dict,
}

# Fields where either name is acceptable (older vs newer convention)
REQUIRED_ALIASES = {
    "peak_layer_final": ["peak_layer_final", "peak_layer"],
}

RECOMMENDED_FIELDS = {
    "protocol.target_ex_per_dim": (int, float),
    "protocol.eval_seeds": list,
    "provenance.model_revision": str,
    "provenance.script": str,
    "provenance.timestamp": str,
    "flagging_6a": dict,
    "control_sensitivity": dict,
}


def _get_nested(d: dict, dotpath: str) -> Any:
    """Get a value from a nested dict using dot notation."""
    parts = dotpath.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_results_json(data: dict, filename: str, strict: bool = False) -> list[str]:
    """Validate a results JSON against the paper schema.

    Returns list of warning strings. If strict=True, also checks recommended fields.
    """
    warnings = []
    for field, expected_type in REQUIRED_FIELDS.items():
        val = _get_nested(data, field)
        if val is None:
            warnings.append(f"{filename}: missing required field '{field}'")
        elif not isinstance(val, expected_type):
            warnings.append(
                f"{filename}: field '{field}' has type {type(val).__name__}, expected {expected_type}"
            )

    for canonical, aliases in REQUIRED_ALIASES.items():
        found = any(_get_nested(data, a) is not None for a in aliases)
        if not found:
            warnings.append(
                f"{filename}: missing required field '{canonical}' (checked: {', '.join(aliases)})"
            )

    if strict:
        for field, _expected_type in RECOMMENDED_FIELDS.items():
            val = _get_nested(data, field)
            if val is None:
                warnings.append(f"{filename}: missing recommended field '{field}'")

    # Value range checks
    pc_mean = _get_nested(data, "partial_corr.mean")
    if isinstance(pc_mean, (int, float)) and abs(pc_mean) > 1.0:
        warnings.append(f"{filename}: partial_corr.mean={pc_mean} outside [-1, 1]")

    per_seed = _get_nested(data, "partial_corr.per_seed")
    if isinstance(per_seed, list) and len(per_seed) < 3:
        warnings.append(f"{filename}: partial_corr.per_seed has {len(per_seed)} seeds (minimum 3)")

    peak_frac = _get_nested(data, "peak_layer_frac")
    if isinstance(peak_frac, (int, float)) and not (0.0 <= peak_frac <= 1.0):
        warnings.append(f"{filename}: peak_layer_frac={peak_frac} outside [0, 1]")

    return warnings


# === V1 paper scope: models included in the analysis ===
# Update this when scope changes. All scripts import from here.

GPT2_MODELS = [
    ("gpt2", 0.124, "GPT2-124M"),
    ("gpt2-medium", 0.355, "GPT2-355M"),
    ("gpt2-large", 0.774, "GPT2-774M"),
    ("gpt2-xl", 1.558, "GPT2-1.5B"),
]

QWEN_MODELS = [
    # v3 results at 600 ex/dim (0.5B needs higher ex/dim)
    ("qwen05b_v3_results.json", 0.5, "Qwen-0.5B"),
    ("qwen1_5b_v3_results.json", 1.5, "Qwen-1.5B"),
    ("qwen3b_v3_results.json", 3.0, "Qwen-3B"),
    ("qwen7b_v3_results.json", 7.6, "Qwen-7B"),
    ("qwen14b_v3_results.json", 14.0, "Qwen-14B"),
    # 32B excluded from v1 (reconstructed data, incomplete battery)
]

# Fallbacks: if the primary file doesn't exist, try these
QWEN_FALLBACKS = {
    "qwen05b_v3_results.json": "qwen05b_v2_results.json",
    "qwen1_5b_v3_results.json": "qwen1_5b_v2_results.json",
    "qwen3b_v3_results.json": "qwen3b_v2_results.json",
    "qwen7b_v3_results.json": "qwen7b_comprehensive.json",
}

LLAMA_MODELS = [
    ("llama3b_v3_results.json", 3.0, "Llama-3B"),
    # Llama 1B (+0.286) excluded from family-level analysis because its
    # architecture (16L, 2048d) differs from 3B (28L, 3072d). Including it
    # inflates within-family variance and obscures the between-family effect.
    # Reported separately as within-family evidence in the architecture section.
]

GEMMA_MODELS = [
    ("gemma3_1b_results.json", 1.0, "Gemma-1B"),
]

MISTRAL_MODELS = [
    ("mistral7b_results.json", 7.25, "Mistral-7B"),
]

PHI_MODELS = [
    ("phi3_mini_results.json", 3.82, "Phi-3-Mini"),
]


def _load_gpt2() -> dict[str, dict[str, Any]]:
    """Load GPT-2 models from transformer_observe.json phase 8."""
    path = RESULTS_DIR / "transformer_observe.json"
    if not path.exists():
        return {}
    to = json.loads(path.read_text())
    p8 = to.get("8", {}).get("models", {})
    models = {}
    for name, params_b, label in GPT2_MODELS:
        if name not in p8:
            continue
        m = p8[name]
        models[label] = {
            "family": "GPT-2",
            "params_b": params_b,
            "label": label,
            "partial_corr": m.get("partial_corr", {}),
            "baselines": m.get("baselines", {}),
        }
    # Control sensitivity for GPT-2 124M is stored separately
    cs = to.get("control_sensitivity", {}).get("control_sets", {})
    if cs and "GPT2-124M" in models:
        models["GPT2-124M"]["control_sensitivity"] = {k: v["mean"] for k, v in cs.items()}
    return models


def _load_family(
    file_list: list[tuple], family_name: str, fallbacks: dict[str, str] | None = None
) -> dict[str, dict[str, Any]]:
    """Load a family of models from individual result files."""
    models = {}
    for fname, params_b, label in file_list:
        path = RESULTS_DIR / fname
        used_fallback = False
        if not path.exists() and fallbacks and fname in fallbacks:
            path = RESULTS_DIR / fallbacks[fname]
            used_fallback = True
        if not path.exists():
            print(f"  WARNING: {fname} not found for {label} ({family_name})")
            continue
        d = json.loads(path.read_text())
        schema_warnings = validate_results_json(d, path.name)
        for w in schema_warnings:
            print(f"  WARNING: {w}")
        pc = d.get("partial_corr", {})
        if not pc or "mean" not in pc:
            print(f"  WARNING: {path.name} has no partial_corr for {label}")
            continue
        mean_val = pc["mean"]
        if not isinstance(mean_val, (int, float)) or abs(mean_val) > 1.0:
            print(f"  WARNING: {path.name} has invalid partial_corr.mean={mean_val} for {label}")
            continue
        entry = {
            "family": family_name,
            "params_b": params_b,
            "label": label,
            "partial_corr": pc,
            "baselines": d.get("baselines", {}),
            "control_sensitivity": d.get("control_sensitivity", {}),
            "source_file": path.name,
        }
        if used_fallback:
            entry["fallback"] = True
            print(f"  NOTE: {label} using fallback {path.name} (primary {fname} not found)")
        models[label] = entry
    return models


def load_all_models(verbose: bool = False) -> dict[str, dict[str, Any]]:
    """Load all v1-scope models. Returns dict keyed by label."""
    models = {}
    models.update(_load_gpt2())
    models.update(_load_family(QWEN_MODELS, "Qwen", QWEN_FALLBACKS))
    models.update(_load_family(LLAMA_MODELS, "Llama"))
    models.update(_load_family(GEMMA_MODELS, "Gemma"))
    models.update(_load_family(MISTRAL_MODELS, "Mistral"))
    models.update(_load_family(PHI_MODELS, "Phi"))

    if verbose:
        # Report what loaded and what's missing
        expected = (
            [(l, "GPT-2") for _, _, l in GPT2_MODELS]
            + [(l, "Qwen") for _, _, l in QWEN_MODELS]
            + [(l, "Llama") for _, _, l in LLAMA_MODELS]
            + [(l, "Gemma") for _, _, l in GEMMA_MODELS]
            + [(l, "Mistral") for _, _, l in MISTRAL_MODELS]
            + [(l, "Phi") for _, _, l in PHI_MODELS]
        )
        loaded = set(models.keys())
        missing = [(l, f) for l, f in expected if l not in loaded]
        if missing:
            print(f"  WARNING: {len(missing)} expected models not found:")
            for label, family in missing:
                print(f"    {label} ({family}) - file missing or no partial_corr")

    return models


def load_per_seed() -> list[tuple[str, str, float, int, float]]:
    """Load per-seed observations: (family, label, params_b, seed_idx, pcorr).
    Models without per_seed data are excluded (not fabricated as n=1)."""
    rows = []
    models = load_all_models()
    for label, m in models.items():
        pc = m["partial_corr"]
        seeds = pc.get("per_seed", [])
        if not seeds:
            print(f"  WARNING: {label} has no per_seed data, excluded from per-seed analyses")
            continue
        for i, rho in enumerate(seeds):
            rows.append((m["family"], label, m["params_b"], i, float(rho)))
    return rows


def load_model_means() -> list[tuple[str, float, float]]:
    """Load one row per model: (family, log_params, mean_pcorr)."""
    models = load_all_models()
    rows = []
    for _label, m in models.items():
        mean = m["partial_corr"].get("mean")
        if mean is not None:
            rows.append((m["family"], np.log10(m["params_b"]), float(mean)))
    return rows


def load_control_sensitivity() -> list[dict[str, Any]]:
    """Load models that have control sensitivity data."""
    models = load_all_models()
    results = []
    for label, m in models.items():
        cs = m.get("control_sensitivity", {})
        if not cs or "none" not in cs:
            continue
        results.append(
            {
                "name": label,
                "family": m["family"],
                "params_b": m["params_b"],
                **{
                    k: cs[k]
                    for k in ["none", "softmax_only", "standard", "plus_entropy", "nonlinear"]
                    if k in cs
                },
            }
        )
    return results


def load_random_head_baselines() -> list[tuple[str, str, float, float]]:
    """Load random_head baseline: (label, family, params_b, value)."""
    models = load_all_models()
    results = []
    for label, m in models.items():
        rh = m.get("baselines", {}).get("random_head")
        if rh is not None:
            results.append((label, m["family"], m["params_b"], float(rh)))
    return results


def validate_all(strict: bool = False) -> int:
    """Validate all paper-scope results JSONs. Returns count of warnings."""

    all_files = (
        [(f, "Qwen") for f, _, _ in QWEN_MODELS]
        + [(f, "Llama") for f, _, _ in LLAMA_MODELS]
        + [(f, "Gemma") for f, _, _ in GEMMA_MODELS]
        + [(f, "Mistral") for f, _, _ in MISTRAL_MODELS]
        + [(f, "Phi") for f, _, _ in PHI_MODELS]
    )
    total_warnings = 0
    for fname, family in all_files:
        path = RESULTS_DIR / fname
        fallback = QWEN_FALLBACKS.get(fname)
        if not path.exists() and fallback:
            path = RESULTS_DIR / fallback
        if not path.exists():
            print(f"  MISSING: {fname} ({family})")
            total_warnings += 1
            continue
        d = json.loads(path.read_text())
        warnings = validate_results_json(d, path.name, strict=strict)
        if warnings:
            for w in warnings:
                print(f"  {w}")
            total_warnings += len(warnings)
        else:
            print(f"  OK: {path.name}")
    return total_warnings


if __name__ == "__main__":
    import sys

    strict = "--strict" in sys.argv
    print(f"Validating paper-scope results JSONs {'(strict)' if strict else ''}...\n")
    n = validate_all(strict=strict)
    if n:
        print(f"\n{n} warning(s)")
        sys.exit(1)
    else:
        print("\nAll files valid")
