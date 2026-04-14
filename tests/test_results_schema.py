"""Schema validation for primary results JSONs.

Catches missing fields, wrong types, and structural changes before
they propagate to the paper pipeline. Parametrized over every primary
result file so a new model with a missing field fails immediately.

Run: uv run pytest tests/test_results_schema.py -v
"""

import json
from pathlib import Path

import pytest

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Primary result files: v3 base, v3 instruct, and cross-family models.
# These are the files load_results.py reads for the paper.
PRIMARY_FILES = sorted(
    list(RESULTS_DIR.glob("qwen*v3*.json"))
    + list(RESULTS_DIR.glob("qwen*instruct_v3*.json"))
    + list(RESULTS_DIR.glob("gemma*.json"))
    + list(RESULTS_DIR.glob("llama*v2*.json"))
    + list(RESULTS_DIR.glob("mistral*.json"))
)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(params=PRIMARY_FILES, ids=lambda p: p.name)
def result(request):
    return _load(request.param)


def test_model_metadata(result):
    """Every result must identify the model and architecture."""
    assert "model" in result
    assert "n_layers" in result
    assert "hidden_dim" in result
    assert isinstance(result["n_layers"], int)
    assert isinstance(result["hidden_dim"], int)


def test_partial_corr_structure(result):
    """partial_corr must have mean, per_seed, and n_seeds."""
    pc = result["partial_corr"]
    assert "mean" in pc, "missing partial_corr.mean"
    assert "per_seed" in pc, "missing partial_corr.per_seed"
    assert "n_seeds" in pc, "missing partial_corr.n_seeds"
    assert isinstance(pc["mean"], (int, float))
    assert isinstance(pc["per_seed"], list)
    assert len(pc["per_seed"]) == pc["n_seeds"]
    assert len(pc["per_seed"]) >= 3, f"only {len(pc['per_seed'])} seeds"


def test_partial_corr_range(result):
    """pcorr should be between -1 and +1."""
    pc = result["partial_corr"]
    assert -1 <= pc["mean"] <= 1
    for val in pc["per_seed"]:
        assert -1 <= val <= 1


def test_output_controlled(result):
    """output_controlled must have mean."""
    oc = result["output_controlled"]
    assert "mean" in oc
    assert isinstance(oc["mean"], (int, float))


def test_seed_agreement(result):
    """seed_agreement must have mean >= 0."""
    sa = result["seed_agreement"]
    assert isinstance(sa, dict), "seed_agreement should be a dict"
    assert "mean" in sa
    assert sa["mean"] >= 0


def test_peak_layer(result):
    """Peak layer must exist and be within layer range."""
    peak = result.get("peak_layer_final", result.get("peak_layer"))
    assert peak is not None, "missing peak_layer_final or peak_layer"
    assert 0 <= peak < result["n_layers"]
    assert "peak_layer_frac" in result
    assert 0 <= result["peak_layer_frac"] <= 1


def test_baselines(result):
    """baselines dict must exist with at least one entry."""
    assert "baselines" in result
    assert len(result["baselines"]) > 0


def test_control_sensitivity(result):
    """control_sensitivity must have the standard control set."""
    cs = result["control_sensitivity"]
    if "_incomplete" in cs:
        pytest.skip("control_sensitivity incomplete (partial result)")
    for key in ("none", "softmax_only", "standard"):
        assert key in cs, f"missing control_sensitivity.{key}"
        assert isinstance(cs[key], (int, float))


def test_layer_profile(result):
    """layer_profile must exist with at least 3 entries."""
    lp = result["layer_profile"]
    assert isinstance(lp, dict)
    assert len(lp) >= 3, f"layer_profile has only {len(lp)} entries"


def test_protocol(result):
    """protocol must document the evaluation setup."""
    proto = result["protocol"]
    assert "eval_seeds" in proto or "layer_select_seed" in proto
