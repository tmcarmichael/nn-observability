"""Tests for analysis/load_results.py -- the single source of truth for model scope.

Verifies that every model declared in load_results actually loads from
the committed results JSONs. Catches filename typos, missing files,
and schema changes before they propagate to downstream consumers.

Run: uv run pytest tests/test_load_results.py -v
"""

import pytest

from analysis.load_results import (
    SCOPES,
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


def test_exactly_seven_families():
    """Default-scope family set is locked. Drops or additions trip this test."""
    models = load_all_models()
    families = {m["family"] for m in models.values()}
    expected = {"GPT-2", "Qwen", "Llama", "Gemma", "Mistral", "Phi", "Pythia"}
    assert families == expected, f"Family mismatch: got {sorted(families)}, expected {sorted(expected)}"


def test_exact_model_count():
    """Default-scope model count is locked. Drops (e.g. via a typo in
    QWEN_MODELS) or additions trip this test.
    """
    models = load_all_models()
    expected_count = 26
    assert len(models) == expected_count, f"Got {len(models)} models, expected {expected_count}"


# ── Named-scope membership contracts ────────────────────────────────
#
# The paper's headline numbers (confidence absorption in the abstract,
# cross-family permutation F-test in the architecture section) are
# produced by analysis scripts running on these named scopes. A typo in
# a frozenset literal in load_results.py would silently shift those
# headline numbers; these tests lock the contract.

EXPECTED_SCOPE_MEMBERSHIP = {
    "cross_family_14": frozenset(
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
    ),
    "control_sensitivity_14": frozenset(
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
    ),
    "absorption_cohort_14": frozenset(
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
    ),
    "pythia_controlled_9": frozenset(
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
    ),
}


@pytest.mark.parametrize("scope_name", sorted(EXPECTED_SCOPE_MEMBERSHIP))
def test_scope_membership(scope_name):
    """Each named scope's membership must match the exact expected set."""
    assert scope_name in SCOPES, f"{scope_name!r} not in SCOPES"
    actual = SCOPES[scope_name]
    expected = EXPECTED_SCOPE_MEMBERSHIP[scope_name]
    assert actual == expected, (
        f"{scope_name} membership drift: "
        f"missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}"
    )


def test_scopes_keys_complete():
    """SCOPES must contain exactly the named keys. A new scope addition
    requires a corresponding membership assertion above; a deletion
    requires removing it here. Catches silent introduction of new scopes.
    """
    expected_keys = set(EXPECTED_SCOPE_MEMBERSHIP) | {"all"}
    assert set(SCOPES) == expected_keys, (
        f"SCOPES key drift: got {sorted(SCOPES)}, expected {sorted(expected_keys)}"
    )
