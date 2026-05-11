"""Shared data loading for all analysis scripts.

Single source of truth for which result files to load and their metadata.
Update this file when results land or scope changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"


# ── Schema for results JSONs ──────────────────────────────────────────

# Required fields and their types for any results file in the configured scope.
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
    "provenance.device": str,
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
    parts = dotpath.split(".")
    current = d
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def validate_results_json(data: dict, filename: str, strict: bool = False) -> list[str]:
    """Validate a results JSON against the schema.

    Returns list of warning strings. If strict=True, also checks recommended fields.
    """
    warnings = []
    for field, expected_type in REQUIRED_FIELDS.items():
        val = _get_nested(data, field)
        if val is None:
            warnings.append(f"{filename}: missing required field '{field}'")
        elif not isinstance(val, expected_type):  # type: ignore[arg-type]
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
    declared_n_seeds = _get_nested(data, "partial_corr.n_seeds")
    if isinstance(per_seed, list) and len(per_seed) < 3:
        # Empty per_seed with n_seeds>=3: mean and std measured.
        if not (len(per_seed) == 0 and isinstance(declared_n_seeds, int) and declared_n_seeds >= 3):
            warnings.append(f"{filename}: partial_corr.per_seed has {len(per_seed)} seeds (minimum 3)")

    peak_frac = _get_nested(data, "peak_layer_frac")
    if isinstance(peak_frac, (int, float)) and not (0.0 <= peak_frac <= 1.0):
        warnings.append(f"{filename}: peak_layer_frac={peak_frac} outside [0, 1]")

    device = _get_nested(data, "provenance.device")
    if isinstance(device, str) and device != "cuda":
        warnings.append(f"{filename}: provenance.device='{device}' (expected 'cuda')")

    return warnings


DYNAMICS_FILES = [
    ("pythia-1b_dynamics.json", "Pythia-1B (16L/8H) dynamics"),
    ("pythia-1.4b_dynamics.json", "Pythia-1.4B (24L/16H) dynamics"),
]

DYNAMICS_REQUIRED_TOP = {
    "model": str,
    "experiment": str,
    "n_layers": int,
    "hidden_dim": int,
    "heads": int,
    "architecture_class": str,
    "provenance": dict,
    "protocol": dict,
    "checkpoints": dict,
}

DYNAMICS_REQUIRED_CHECKPOINT: dict[str, type | tuple[type, ...]] = {
    "step": int,
    "tokens_seen": (int, float),
    "revision": str,
    "peak_layer": int,
    "peak_layer_frac": (int, float),
    "partial_corr": dict,
    "output_controlled": dict,
    "perplexity": (int, float),
}


def validate_dynamics_json(data: dict, filename: str) -> list[str]:
    """Validate a checkpoint-dynamics JSON. Returns list of warnings."""
    warnings = []
    for field, expected_type in DYNAMICS_REQUIRED_TOP.items():
        val = data.get(field)
        if val is None:
            warnings.append(f"{filename}: missing required field '{field}'")
        elif not isinstance(val, expected_type):
            warnings.append(
                f"{filename}: field '{field}' has type {type(val).__name__}, expected {expected_type}"
            )

    checkpoints = data.get("checkpoints", {})
    if not checkpoints:
        warnings.append(f"{filename}: no checkpoints found")
        return warnings

    for ck_name, ck_data in checkpoints.items():
        for field, expected in DYNAMICS_REQUIRED_CHECKPOINT.items():
            val = ck_data.get(field)
            if val is None:
                warnings.append(f"{filename}: checkpoint '{ck_name}' missing '{field}'")
            elif not isinstance(val, expected):  # type: ignore[arg-type]
                warnings.append(
                    f"{filename}: checkpoint '{ck_name}' field '{field}' has type {type(val).__name__}"
                )

        pc = ck_data.get("partial_corr", {})
        pc_mean = pc.get("mean")
        if isinstance(pc_mean, (int, float)) and abs(pc_mean) > 1.0:
            warnings.append(f"{filename}: checkpoint '{ck_name}' partial_corr.mean={pc_mean} outside [-1, 1]")
        per_seed = pc.get("per_seed")
        if isinstance(per_seed, list) and len(per_seed) < 3:
            warnings.append(
                f"{filename}: checkpoint '{ck_name}' partial_corr.per_seed has {len(per_seed)} seeds"
            )

    return warnings


GPT2_MODELS = [
    ("gpt2", 0.124, "GPT2-124M"),
    ("gpt2-medium", 0.355, "GPT2-355M"),
    ("gpt2-large", 0.774, "GPT2-774M"),
    ("gpt2-xl", 1.558, "GPT2-1.5B"),
]

QWEN_MODELS = [
    ("qwen2.5-0.5b_main.json", 0.5, "Qwen-0.5B"),
    ("qwen2.5-1.5b_main.json", 1.5, "Qwen-1.5B"),
    ("qwen2.5-3b_main.json", 3.0, "Qwen-3B"),
    ("qwen2.5-7b_main.json", 7.6, "Qwen-7B"),
    ("qwen2.5-14b_main.json", 14.0, "Qwen-14B"),
    ("qwen2.5-32b_main.json", 32.0, "Qwen-32B"),
]

LLAMA_MODELS = [
    ("llama-3.2-1b_main.json", 1.0, "Llama-1B"),
    ("llama-3.2-3b_main.json", 3.0, "Llama-3B"),
    ("llama-3.1-8b_main.json", 8.0, "Llama-8B"),
]

GEMMA_MODELS = [
    ("gemma-3-1b_main.json", 1.0, "Gemma-1B"),
    ("gemma-3-4b_main.json", 4.0, "Gemma-4B"),
]

MISTRAL_MODELS = [
    ("mistral-7b-v0.3_main.json", 7.25, "Mistral-7B"),
]

PHI_MODELS = [
    ("phi-3-mini_main.json", 3.82, "Phi-3-Mini"),
]

PYTHIA_MODELS = [
    ("pythia-70m_main.json", 0.07, "Pythia-70M"),
    ("pythia-160m_main.json", 0.16, "Pythia-160M"),
    ("pythia-410m_main.json", 0.41, "Pythia-410M"),
    ("pythia-1b_main.json", 1.0, "Pythia-1B"),
    ("pythia-1.4b_main.json", 1.4, "Pythia-1.4B"),
    ("pythia-1.4b-deduped_main.json", 1.4, "Pythia-1.4B-deduped"),
    ("pythia-2.8b_main.json", 2.8, "Pythia-2.8B"),
    ("pythia-6.9b_main.json", 6.9, "Pythia-6.9B"),
    ("pythia-12b_main.json", 12.0, "Pythia-12B"),
]


# ── Named analysis scopes ────────────────────────────────────────────
#
# Use these to reproduce specific paper-table numbers. The default loaders
# return everything; pass scope="<name>" to filter. Headline scripts
# (permutation_test, selectivity) default to the paper-Section-5 scope.

_CROSS_FAMILY_14 = frozenset(
    {
        "GPT2-124M",
        "GPT2-355M",
        "GPT2-774M",
        "GPT2-1.5B",
        "Qwen-0.5B",
        "Qwen-1.5B",
        "Qwen-3B",
        "Qwen-7B",
        "Qwen-14B",
        "Qwen-32B",
        "Llama-3B",
        "Gemma-1B",
        "Mistral-7B",
        "Phi-3-Mini",
    }
)
_PYTHIA_CONTROLLED_9 = frozenset(
    {
        "Pythia-70M",
        "Pythia-160M",
        "Pythia-410M",
        "Pythia-1B",
        "Pythia-1.4B",
        "Pythia-1.4B-deduped",
        "Pythia-2.8B",
        "Pythia-6.9B",
        "Pythia-12B",
    }
)
# Confidence-absorption headline cohort. Differs from _CROSS_FAMILY_14: one
# GPT-2 size (124M) rather than four, all evaluated Llama sizes, both Gemma
# sizes. This is the scope behind the paper's cross-family confidence-
# absorption headline; the canonical value lives in
# reports/paper_values.json under the confabsorbmean macro.
_ABSORPTION_COHORT_14 = frozenset(
    {
        "GPT2-124M",
        "Qwen-0.5B",
        "Qwen-1.5B",
        "Qwen-3B",
        "Qwen-7B",
        "Qwen-14B",
        "Qwen-32B",
        "Llama-1B",
        "Llama-3B",
        "Llama-8B",
        "Gemma-1B",
        "Gemma-4B",
        "Mistral-7B",
        "Phi-3-Mini",
    }
)

SCOPES: dict[str, frozenset[str] | None] = {
    "cross_family_14": _CROSS_FAMILY_14,
    "control_sensitivity_14": _CROSS_FAMILY_14,
    "absorption_cohort_14": _ABSORPTION_COHORT_14,
    "pythia_controlled_9": _PYTHIA_CONTROLLED_9,
    "all": None,
}


def _load_gpt2() -> dict[str, dict[str, Any]]:
    """Load GPT-2 scaling results from per-size matched-protocol files.

    All four sizes (124M, 355M, 774M, 1.5B) are read from gpt2-{size}_main.json
    under the canonical 350 ex/dim, 7-seed cross-family protocol. The legacy
    transformer_observe.json phase-8 entries (varying ex/dim per size) are no
    longer the source of truth; transformer_observe.json is retained only for
    the 124M hardening, hand-designed baselines, and analytical random-baseline
    flagging experiments that live outside this scaling test.
    """
    name_to_file = {
        "gpt2": "gpt2-124m_main.json",
        "gpt2-medium": "gpt2-medium_main.json",
        "gpt2-large": "gpt2-large_main.json",
        "gpt2-xl": "gpt2-xl_main.json",
    }
    models = {}
    for name, params_b, label in GPT2_MODELS:
        path = RESULTS_DIR / name_to_file[name]
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        models[label] = {
            "family": "GPT-2",
            "params_b": params_b,
            "label": label,
            "partial_corr": d.get("partial_corr", {}),
            "baselines": d.get("baselines", {}),
            "source_file": name_to_file[name],
        }
        cs = d.get("control_sensitivity", {})
        if cs:
            models[label]["control_sensitivity"] = {
                k: (v["mean"] if isinstance(v, dict) else v) for k, v in cs.items()
            }
    return models


def _load_family(
    file_list: list[tuple[str, float, str]],
    family_name: str,
) -> dict[str, dict[str, Any]]:
    models = {}
    for fname, params_b, label in file_list:
        path = RESULTS_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{fname} not found for {label} ({family_name}); expected at {path}")
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
            "baselines": d.get("baselines") or {},
            "control_sensitivity": d.get("control_sensitivity", {}),
            "source_file": path.name,
        }
        models[label] = entry
    return models


def _resolve_scope(scope: str | None) -> frozenset[str] | None:
    if scope is None:
        return None
    if scope not in SCOPES:
        raise ValueError(f"Unknown scope '{scope}'. Known scopes: {sorted(SCOPES)}")
    return SCOPES[scope]


def load_all_models(verbose: bool = False, scope: str | None = None) -> dict[str, dict[str, Any]]:
    """Load all paper-scope models, keyed by label.

    With scope=None (default), loads every model in the family lists. Pass a
    named scope from SCOPES (e.g. 'cross_family_14', 'pythia_controlled_9')
    to filter to the model set used by a specific paper table.
    """
    allowed = _resolve_scope(scope)
    models = {}
    models.update(_load_gpt2())
    models.update(_load_family(QWEN_MODELS, "Qwen"))
    models.update(_load_family(LLAMA_MODELS, "Llama"))
    models.update(_load_family(GEMMA_MODELS, "Gemma"))
    models.update(_load_family(MISTRAL_MODELS, "Mistral"))
    models.update(_load_family(PHI_MODELS, "Phi"))
    models.update(_load_family(PYTHIA_MODELS, "Pythia"))

    if allowed is not None:
        models = {label: m for label, m in models.items() if label in allowed}

    if verbose:
        # Report what loaded and what's missing (relative to scope when set)
        expected_all = (
            [(l, "GPT-2") for _, _, l in GPT2_MODELS]
            + [(l, "Qwen") for _, _, l in QWEN_MODELS]
            + [(l, "Llama") for _, _, l in LLAMA_MODELS]
            + [(l, "Gemma") for _, _, l in GEMMA_MODELS]
            + [(l, "Mistral") for _, _, l in MISTRAL_MODELS]
            + [(l, "Phi") for _, _, l in PHI_MODELS]
            + [(l, "Pythia") for _, _, l in PYTHIA_MODELS]
        )
        if allowed is not None:
            expected = [(l, f) for l, f in expected_all if l in allowed]
        else:
            expected = expected_all
        loaded = set(models.keys())
        missing = [(l, f) for l, f in expected if l not in loaded]
        if missing:
            print(f"  WARNING: {len(missing)} expected models not found:")
            for label, family in missing:
                print(f"    {label} ({family}) - file missing or no partial_corr")

    return models


def load_per_seed(scope: str | None = None) -> list[tuple[str, str, float, int, float]]:
    """Per-seed observations: (family, label, params_b, seed_idx, pcorr)."""
    rows = []
    models = load_all_models(scope=scope)
    for label, m in models.items():
        pc = m["partial_corr"]
        seeds = pc.get("per_seed", [])
        if not seeds:
            print(f"  WARNING: {label} has no per_seed data, excluded from per-seed analyses")
            continue
        for i, rho in enumerate(seeds):
            rows.append((m["family"], label, m["params_b"], i, float(rho)))
    return rows


def load_model_means(scope: str | None = None) -> list[tuple[str, float, float]]:
    """One row per model: (family, log10_params, mean_pcorr)."""
    models = load_all_models(scope=scope)
    rows = []
    for _label, m in models.items():
        mean = m["partial_corr"].get("mean")
        if mean is not None:
            rows.append((m["family"], np.log10(m["params_b"]), float(mean)))
    return rows


def load_control_sensitivity(scope: str | None = None) -> list[dict[str, Any]]:
    """Models with control sensitivity data for the waterfall plot."""
    models = load_all_models(scope=scope)
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
                    for k in [
                        "none",
                        "softmax_only",
                        "standard",
                        "plus_entropy",
                        "nonlinear",
                    ]
                    if k in cs
                },
            }
        )
    return results


def load_random_head_baselines(
    scope: str | None = None,
) -> list[tuple[str, str, float, float]]:
    """Random-probe baseline: (label, family, params_b, value)."""
    models = load_all_models(scope=scope)
    results = []
    for label, m in models.items():
        rh = m.get("baselines", {}).get("random_head")
        if rh is not None:
            results.append((label, m["family"], m["params_b"], float(rh)))
    return results


def validate_all(strict: bool = False) -> int:
    """Validate all configured results JSONs. Returns warning count."""
    all_files = (
        [(f, "Qwen") for f, _, _ in QWEN_MODELS]
        + [(f, "Llama") for f, _, _ in LLAMA_MODELS]
        + [(f, "Gemma") for f, _, _ in GEMMA_MODELS]
        + [(f, "Mistral") for f, _, _ in MISTRAL_MODELS]
        + [(f, "Phi") for f, _, _ in PHI_MODELS]
        + [(f, "Pythia") for f, _, _ in PYTHIA_MODELS]
    )
    total_warnings = 0
    for fname, family in all_files:
        path = RESULTS_DIR / fname
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

    for fname, label in DYNAMICS_FILES:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"  MISSING: {fname} ({label})")
            total_warnings += 1
            continue
        d = json.loads(path.read_text())
        warnings = validate_dynamics_json(d, path.name)
        if warnings:
            for w in warnings:
                print(f"  {w}")
            total_warnings += len(warnings)
        else:
            print(f"  OK: {path.name}")

    return total_warnings


CANONICAL_PROVENANCE_FIELDS = ("model_revision", "script", "timestamp", "value_source", "device")
CANONICAL_VALUE_SOURCE = {"runtime", "post_hoc_deterministic"}
# Paper-cited committed results require CUDA. MPS and CPU runs are
# accepted for local development only and are excluded from committed results.
CANONICAL_DEVICE = {"cuda"}
NON_RESULT_FILES = {
    "model_revisions.json",  # HF model revision registry, not a result
    "dataset_revisions.json",  # HF dataset revision registry, not a result
}


def _validate_one_provenance(prov: Any, label: str = "<test>") -> list[str]:
    """Validate a single provenance dict against the canonical contract.

    Returns a list of error messages (empty if valid). Pure function so
    mutation tests can call it directly without writing files.
    """
    errors: list[str] = []
    if not isinstance(prov, dict):
        return [f"{label}: missing provenance dict"]

    missing = [k for k in CANONICAL_PROVENANCE_FIELDS if k not in prov]
    if missing:
        errors.append(f"{label}: missing fields: {missing}")
        return errors  # downstream checks assume required keys present

    extras = set(prov.keys()) - set(CANONICAL_PROVENANCE_FIELDS)
    if extras:
        errors.append(f"{label}: extra fields not allowed: {sorted(extras)}")

    if tuple(prov.keys()) != CANONICAL_PROVENANCE_FIELDS:
        errors.append(f"{label}: field order {tuple(prov.keys())} != canonical")

    ts = prov["timestamp"]
    if not (isinstance(ts, str) and len(ts) == 25 and ts.endswith("+00:00") and "." not in ts):
        errors.append(f"{label}: timestamp not UTC second-precision: {ts!r}")

    rev = prov["model_revision"]
    if not (isinstance(rev, str) and (len(rev) == 40 or rev == "multi")):
        errors.append(f"{label}: model_revision not 40-char SHA / 'multi': {rev!r}")

    if prov["value_source"] not in CANONICAL_VALUE_SOURCE:
        errors.append(f"{label}: value_source not in enum: {prov['value_source']!r}")

    if prov["device"] not in CANONICAL_DEVICE:
        errors.append(f"{label}: device not in {CANONICAL_DEVICE}: {prov['device']!r}")

    script = prov["script"]
    if not isinstance(script, str) or not script:
        errors.append(f"{label}: script must be a non-empty string: {script!r}")
    elif not (script.startswith("scripts/") or script.startswith("src/")):
        errors.append(f"{label}: script must be under scripts/ or src/: {script!r}")
    elif not (REPO_ROOT / script).is_file():
        errors.append(f"{label}: script does not point at an existing file: {script!r}")

    return errors


def validate_canonical_provenance() -> int:
    """Scan every result JSON for canonical provenance shape.

    Complements validate_all(): the per-file-type schema checks above
    cover paper-cited probe runs only. This pass runs on every JSON in
    results/ and tests/fixtures/ and verifies the canonical 5-field
    provenance block shape regardless of file type. Returns warning count.
    """
    warnings = 0
    files: list[Path] = []
    for d in (RESULTS_DIR, FIXTURES_DIR):
        if d.exists():
            files.extend(sorted(d.glob("*.json")))
    scanned = 0
    for path in files:
        if path.name in NON_RESULT_FILES:
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            print(f"  CANONICAL FAIL: {path.name} JSON decode error: {e}")
            warnings += 1
            continue
        if not isinstance(data, dict):
            continue
        scanned += 1
        for msg in _validate_one_provenance(data.get("provenance"), path.name):
            print(f"  CANONICAL FAIL: {msg}")
            warnings += 1

    if warnings == 0:
        print(f"  Canonical provenance: {scanned} files all conform")
    return warnings


if __name__ == "__main__":
    import sys

    strict = "--strict" in sys.argv
    print(f"Validating results JSONs {'(strict)' if strict else ''}...\n")
    n = validate_all(strict=strict)
    print()
    n += validate_canonical_provenance()
    if n:
        print(f"\n{n} warning(s)")
        sys.exit(1)
    else:
        print("\nAll files valid")
