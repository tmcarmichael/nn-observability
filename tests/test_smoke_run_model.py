"""Smoke test for run_model.py using GPT-2 124M on CPU.

Runs with minimal token budget (50 ex/dim, batch 4) to verify the full
pipeline: data loading, layer sweep, multi-seed eval, test split, baselines,
output-controlled, cross-domain, control sensitivity, and flagging.

This test takes ~2-3 minutes on CPU and downloads GPT-2 124M (~500MB) on
first run. It validates the output JSON schema, not exact numeric values.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_FILE = "smoke_test_gpt2.json"


@pytest.fixture(scope="module")
def smoke_results():
    """Run run_model.py once for the entire test module."""
    out_path = RESULTS_DIR / OUTPUT_FILE
    if out_path.exists():
        out_path.unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "run_model.py"),
            "--model",
            "openai-community/gpt2",
            "--output",
            OUTPUT_FILE,
            "--ex-dim",
            "50",
            "--batch-size",
            "4",
            "--layers-per-pass",
            "4",
            "--attn-impl",
            "sdpa",
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(f"run_model.py failed:\n{result.stderr[-2000:]}")

    assert out_path.exists(), f"Output file not created: {out_path}"
    data = json.loads(out_path.read_text())
    yield data
    out_path.unlink(missing_ok=True)


def test_model_metadata(smoke_results):
    assert smoke_results["model"] == "openai-community/gpt2"
    assert smoke_results["n_layers"] == 12
    assert smoke_results["hidden_dim"] == 768


def test_partial_corr_schema(smoke_results):
    pc = smoke_results["partial_corr"]
    assert "mean" in pc
    assert "per_seed" in pc
    assert len(pc["per_seed"]) == 7
    assert all(isinstance(v, float) for v in pc["per_seed"])


def test_partial_corr_positive(smoke_results):
    assert smoke_results["partial_corr"]["mean"] > 0


def test_test_split_exists(smoke_results):
    tsc = smoke_results["test_split_comparison"]
    assert "mean" in tsc
    assert "per_seed" in tsc


def test_seed_agreement(smoke_results):
    sa = smoke_results["seed_agreement"]
    assert sa["mean"] > 0.5


def test_output_controlled(smoke_results):
    oc = smoke_results["output_controlled"]
    assert "mean" in oc
    assert "per_seed" in oc


def test_baselines(smoke_results):
    bl = smoke_results["baselines"]
    for key in ["ff_goodness", "active_ratio", "act_entropy", "activation_norm", "random_head"]:
        assert key in bl, f"Missing baseline: {key}"


def test_cross_domain(smoke_results):
    cd = smoke_results["cross_domain"]
    assert "wikitext" in cd
    assert "c4" in cd
    assert "c4_within" in cd


def test_control_sensitivity(smoke_results):
    cs = smoke_results["control_sensitivity"]
    for key in ["none", "softmax_only", "standard", "plus_entropy", "nonlinear"]:
        assert key in cs, f"Missing control: {key}"
    assert cs["none"] > cs["standard"]


def test_flagging(smoke_results):
    fl = smoke_results["flagging_6a"]
    assert "n_tokens" in fl
    assert "summary" in fl
    assert "0.1" in fl["summary"]


def test_layer_profile(smoke_results):
    lp = smoke_results["layer_profile"]
    assert len(lp) == 12
    assert all(isinstance(v, float) for v in lp.values())
