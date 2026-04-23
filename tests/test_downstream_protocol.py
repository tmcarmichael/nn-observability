"""Lock-in tests for the downstream task protocol.

The three downstream scripts (medqa_selective.py, rag_hallucination.py,
truthfulqa_hallucination.py) inline copies of the canonical probe functions.
This file tests that:
  1. Inlined compute_loss_residuals matches src/probe.py exactly.
  2. Inlined _get_layer_list resolves correctly for Qwen/Llama/Mistral/Phi/GPT-2.
  3. train_linear_binary is deterministic under fixed seed.
  4. Exclusive-catch flag-rate arithmetic matches a hand-computed fixture.
  5. The dataset loaded by each script matches the "dataset" field written to
     the output JSON (prevents the silent code/pod divergence that shipped a
     Natural Questions loader with a SQuAD output label).

Drift between copies was the root cause of several past bugs; these tests
catch it at CI time instead of during an expensive GPU run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.stats import pearsonr, rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
DOWNSTREAM_SCRIPTS = [
    REPO_ROOT / "scripts" / "medqa_selective.py",
    REPO_ROOT / "scripts" / "rag_hallucination.py",
    REPO_ROOT / "scripts" / "truthfulqa_hallucination.py",
]


def _extract_function(filepath: Path, func_name: str) -> str:
    lines = filepath.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start = i
            break
    if start is None:
        pytest.fail(f"{func_name} not found in {filepath.name}")
    func_lines = [lines[start]]
    for line in lines[start + 1 :]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def _compile_function(source: str, name: str, extra_globals=None):
    ns = {"np": np, "rankdata": rankdata, "pearsonr": pearsonr, "torch": torch, "nn": nn}
    if extra_globals:
        ns.update(extra_globals)
    exec(compile(source, f"<{name}>", "exec"), ns)
    return ns[name]


@pytest.fixture(scope="module")
def probe_src():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import probe

    return probe


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.RandomState(42)
    n = 500
    losses = rng.exponential(2.0, n).astype(np.float64)
    max_softmax = rng.beta(5, 2, n).astype(np.float64)
    activation_norm = rng.lognormal(0, 1, n).astype(np.float64)
    probe_scores = 0.3 * losses + 0.5 * rng.randn(n)
    return losses, max_softmax, activation_norm, probe_scores


@pytest.mark.parametrize("script_path", DOWNSTREAM_SCRIPTS, ids=lambda p: p.name)
def test_compute_loss_residuals_matches_src(script_path, probe_src, synthetic_data):
    losses, max_softmax, activation_norm, _ = synthetic_data
    fn = _compile_function(_extract_function(script_path, "compute_loss_residuals"), "compute_loss_residuals")
    expected = probe_src.compute_loss_residuals(losses, max_softmax, activation_norm)
    got = fn(losses, max_softmax, activation_norm)
    np.testing.assert_array_almost_equal(
        got, expected, decimal=12, err_msg=f"compute_loss_residuals drift in {script_path.name}"
    )


def _qwen_like():
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
    return m


def _gpt2_like():
    m = nn.Module()
    m.transformer = nn.Module()
    m.transformer.h = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
    return m


def _unsupported():
    m = nn.Module()
    m.foo = nn.Linear(8, 8)
    return m


@pytest.mark.parametrize("script_path", DOWNSTREAM_SCRIPTS, ids=lambda p: p.name)
def test_get_layer_list_covers_cross_family(script_path):
    fn = _compile_function(_extract_function(script_path, "_get_layer_list"), "_get_layer_list")

    qwen = _qwen_like()
    layers = fn(qwen)
    assert layers is qwen.model.layers
    assert len(layers) == 4

    gpt2 = _gpt2_like()
    layers = fn(gpt2)
    assert layers is gpt2.transformer.h

    with pytest.raises(ValueError, match="Unsupported architecture"):
        fn(_unsupported())


@pytest.mark.parametrize("script_path", DOWNSTREAM_SCRIPTS, ids=lambda p: p.name)
def test_get_layer_list_phi3_and_mistral_share_qwen_case(script_path):
    # Phi-3 and Mistral both expose .model.layers (confirmed by AutoConfig inspection,
    # see scripts/preflight/check_architecture.py). Verify the Qwen-shaped fixture
    # works, which is sufficient because the resolution path is identical.
    fn = _compile_function(_extract_function(script_path, "_get_layer_list"), "_get_layer_list")
    phi3_shaped = _qwen_like()
    phi3_shaped.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(32)])
    mistral_shaped = _qwen_like()
    mistral_shaped.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(32)])

    assert len(fn(phi3_shaped)) == 32
    assert len(fn(mistral_shaped)) == 32


def test_exclusive_catch_arithmetic():
    # Hand-computed fixture matching the pattern in all three downstream scripts:
    #   k = max(1, int(n * rate))
    #   obs_flagged  = obs  >= np.sort(obs)[-k]
    #   conf_flagged = conf <= np.sort(conf)[k]
    #
    # n=10 questions, 4 errors (~correct = True at indices 1,3,5,7).
    # rate=0.2 -> k=2, top-2 observer flags, bottom-2 confidence flags.
    n = 10
    obs = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.0])
    conf = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 1.0])
    correct = np.array([True, False, True, False, True, False, True, False, True, True])

    rate = 0.2
    k = max(1, int(n * rate))
    assert k == 2
    obs_flagged = obs >= np.sort(obs)[-k]
    conf_flagged = conf <= np.sort(conf)[k]

    # Observer flags indices {1, 3} (obs 0.9, 0.8). Confidence flags indices {1, 3} (conf 0.1, 0.2).
    # With conf_flagged = conf <= np.sort(conf)[k=2] = 0.3, that also includes index 5 (conf=0.3).
    assert set(np.where(obs_flagged)[0].tolist()) == {1, 3}
    assert set(np.where(conf_flagged)[0].tolist()) == {1, 3, 5}

    obs_exclusive = int((obs_flagged & ~conf_flagged & ~correct).sum())
    conf_exclusive = int((conf_flagged & ~obs_flagged & ~correct).sum())
    both = int((obs_flagged & conf_flagged & ~correct).sum())

    # obs_flagged = {1,3}; conf_flagged = {1,3,5}
    # both error-flagged: {1,3} (errors=False at 1,3); conf-only error-flagged: {5} (error=False at 5)
    assert both == 2
    assert conf_exclusive == 1
    assert obs_exclusive == 0  # observer-exclusive errors = 0 in this fixture


def test_train_linear_binary_deterministic():
    """Fixed seed must yield identical weights on identical data, per script."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import probe

    rng = np.random.RandomState(0)
    d = 32
    n = 1024
    train_data = {
        "activations": torch.from_numpy(rng.randn(n, d).astype(np.float32)),
        "losses": rng.exponential(1.0, n).astype(np.float64),
        "max_softmax": rng.beta(5, 2, n).astype(np.float64),
        "activation_norm": rng.lognormal(0, 1, n).astype(np.float64),
    }

    h1 = probe.train_linear_binary(train_data, seed=42, epochs=2, lr=1e-3, train_device="cpu")
    h2 = probe.train_linear_binary(train_data, seed=42, epochs=2, lr=1e-3, train_device="cpu")

    w1 = h1.weight.detach().cpu().numpy()
    w2 = h2.weight.detach().cpu().numpy()
    np.testing.assert_array_equal(w1, w2, err_msg="train_linear_binary not deterministic under fixed seed")


EXPECTED_DATASETS = {
    "medqa_selective.py": "GBaker/MedQA-USMLE-4-options",
    "rag_hallucination.py": "rajpurkar/squad_v2",
    "truthfulqa_hallucination.py": "truthfulqa/truthful_qa",
}


@pytest.mark.parametrize("script_path", DOWNSTREAM_SCRIPTS, ids=lambda p: p.name)
def test_loader_dataset_matches_output_label(script_path):
    """Loader's load_dataset(...) target must match the output 'dataset' string.

    The RAG script shipped for a while with a Natural Questions loader and a
    SQuAD 2.0 output label. This test locks the invariant: whatever dataset
    the script actually loads is what the saved JSON must claim.
    """
    import re

    src = script_path.read_text()
    loader_matches = re.findall(r'load_dataset\(\s*"([^"]+)"', src)
    loader_datasets = [m for m in loader_matches if m != "wikitext"]
    output_matches = re.findall(r'"dataset"\s*:\s*"([^"]+)"', src)

    assert loader_datasets, f"No task-dataset load_dataset call found in {script_path.name}"
    assert output_matches, f"No 'dataset' output field found in {script_path.name}"
    assert len(set(loader_datasets)) == 1, f"Multiple task datasets loaded: {loader_datasets}"
    assert len(set(output_matches)) == 1, f"Multiple output 'dataset' strings: {output_matches}"

    loader_ds = loader_datasets[0]
    output_ds = output_matches[0]
    assert loader_ds == output_ds, (
        f"{script_path.name}: loader uses '{loader_ds}' but output claims '{output_ds}'. "
        f"This was the RAG NQ-vs-SQuAD bug."
    )

    expected = EXPECTED_DATASETS[script_path.name]
    assert loader_ds == expected, (
        f"{script_path.name}: loads '{loader_ds}' but paper baseline is '{expected}'. "
        f"Paper Table 3 numbers assume the baseline dataset."
    )


def test_partial_spearman_matches_reference(probe_src, synthetic_data):
    losses, max_softmax, activation_norm, probe_scores = synthetic_data
    r, p = probe_src.partial_spearman(probe_scores, losses, [max_softmax, activation_norm])

    # Independent re-derivation: rank inputs, regress on covariates, Pearson on residuals.
    rx = rankdata(probe_scores)
    ry = rankdata(losses)
    rc = np.column_stack([rankdata(max_softmax), rankdata(activation_norm), np.ones(len(losses))])
    coef_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    coef_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    r_ref, p_ref = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)

    assert r == pytest.approx(float(r_ref), abs=1e-12)
    assert p == pytest.approx(float(p_ref), abs=1e-12)
