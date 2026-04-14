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
    # v3 results with CS (will fall back to v2 if v3 not yet available)
    ("qwen3b_v3_results.json", 3.0, "Qwen-3B"),
    # v3 results (will fall back to older if not yet available)
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
    ("llama3b_v2_results.json", 3.0, "Llama-3B"),
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
        pc = d.get("partial_corr", {})
        if not pc or "mean" not in pc:
            print(f"  WARNING: {path.name} has no partial_corr for {label}")
            continue
        mean_val = pc["mean"]
        if not isinstance(mean_val, (int, float)) or abs(mean_val) > 1.0:
            print(f"  WARNING: {path.name} has invalid partial_corr.mean={mean_val} for {label}")
            continue
        if "per_seed" not in pc:
            print(f"  WARNING: {path.name} has no per_seed data for {label}")
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
