"""Tests for analysis/load_results.py -- the single source of truth for model scope.

Verifies that every model declared in load_results actually loads from
the committed results JSONs. Catches filename typos, missing files,
and schema changes before they propagate to downstream consumers.

Run: uv run pytest tests/test_load_results.py -v
"""

from analysis.load_results import (
    load_all_models,
    load_model_means,
    load_per_seed,
)


def test_all_models_load():
    """Every declared model should load without warnings."""
    models = load_all_models(verbose=False)
    assert len(models) > 0, "No models loaded"


def test_expected_families():
    """At least the v1 families should be present."""
    models = load_all_models()
    families = {m["family"] for m in models.values()}
    for expected in ("GPT-2", "Qwen", "Llama", "Gemma"):
        assert expected in families, f"Family {expected} missing from loaded models"


def test_model_means_match_all_models():
    """load_model_means should return one entry per model in load_all_models."""
    all_models = load_all_models()
    means = load_model_means()
    assert len(means) == len(all_models), (
        f"load_model_means returned {len(means)} but load_all_models has {len(all_models)}"
    )


def test_per_seed_has_data():
    """load_per_seed should return multiple observations."""
    rows = load_per_seed()
    assert len(rows) > 10, f"Only {len(rows)} per-seed observations loaded"


def test_partial_corr_schema():
    """Every model should have partial_corr with a mean."""
    models = load_all_models()
    for label, m in models.items():
        pc = m.get("partial_corr", {})
        assert "mean" in pc, f"{label} missing partial_corr.mean"
        assert isinstance(pc["mean"], (int, float)), f"{label} partial_corr.mean is not numeric"


def test_no_nan_values():
    """No model should have NaN partial correlation."""
    import math

    models = load_all_models()
    for label, m in models.items():
        val = m["partial_corr"]["mean"]
        assert not math.isnan(val), f"{label} has NaN partial_corr"


def test_pcorr_in_valid_range():
    """Partial correlations must be in [-1, 1]."""
    models = load_all_models()
    for label, m in models.items():
        val = m["partial_corr"]["mean"]
        assert -1.0 <= val <= 1.0, f"{label} partial_corr.mean={val} out of range"


def test_six_families():
    """v2.2.1+ should have six families."""
    models = load_all_models()
    families = {m["family"] for m in models.values()}
    for expected in ("GPT-2", "Qwen", "Llama", "Gemma", "Mistral", "Phi"):
        assert expected in families, f"Family {expected} missing"


def test_minimum_model_count():
    """v2.2.1+ should have at least 13 models."""
    models = load_all_models()
    assert len(models) >= 13, f"Only {len(models)} models loaded, expected >= 13"
