"""Verify that inlined probe functions in run_model.py match src/probe.py.

run_model.py inlines core functions for GPU portability (no local imports).
This test catches drift between the two copies by running both on the same
synthetic data and asserting identical output.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr, rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent


def _extract_function(filepath: Path, func_name: str) -> str:
    """Extract a function's source from a file by name."""
    lines = filepath.read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start = i
            break
    if start is None:
        pytest.fail(f"{func_name} not found in {filepath.name}")
    # Collect lines until next top-level def/class or end of file
    func_lines = [lines[start]]
    for line in lines[start + 1 :]:
        if line and not line[0].isspace() and not line.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines)


def _compile_function(source: str, name: str):
    """Compile a function source string and return the callable."""
    namespace = {"np": np, "rankdata": rankdata, "pearsonr": pearsonr}
    exec(compile(source, f"<{name}>", "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module")
def run_model_funcs():
    rm_path = REPO_ROOT / "scripts" / "run_model.py"
    return {
        name: _compile_function(_extract_function(rm_path, name), name)
        for name in ["partial_spearman", "compute_loss_residuals"]
    }


@pytest.fixture(scope="module")
def probe_funcs():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import probe

    return {
        "partial_spearman": probe.partial_spearman,
        "compute_loss_residuals": probe.compute_loss_residuals,
    }


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.RandomState(42)
    n = 500
    losses = rng.exponential(2.0, n)
    max_softmax = rng.beta(5, 2, n)
    activation_norm = rng.lognormal(0, 1, n)
    probe_scores = 0.3 * losses + 0.5 * rng.randn(n)
    return losses, max_softmax, activation_norm, probe_scores


def test_partial_spearman_sync(run_model_funcs, probe_funcs, synthetic_data):
    """partial_spearman must produce identical output in both copies."""
    losses, max_softmax, activation_norm, probe_scores = synthetic_data
    covariates = [max_softmax, activation_norm]

    r_src, p_src = probe_funcs["partial_spearman"](probe_scores, losses, covariates)
    r_rm, p_rm = run_model_funcs["partial_spearman"](probe_scores, losses, covariates)

    assert r_src == pytest.approx(r_rm, abs=1e-12), (
        f"partial_spearman drift: src/probe.py={r_src}, scripts/run_model.py={r_rm}"
    )
    assert p_src == pytest.approx(p_rm, abs=1e-12)


def test_compute_loss_residuals_sync(run_model_funcs, probe_funcs, synthetic_data):
    """compute_loss_residuals must produce identical output in both copies."""
    losses, max_softmax, activation_norm, _ = synthetic_data

    resid_src = probe_funcs["compute_loss_residuals"](losses, max_softmax, activation_norm)
    resid_rm = run_model_funcs["compute_loss_residuals"](losses, max_softmax, activation_norm)

    np.testing.assert_array_almost_equal(
        resid_src,
        resid_rm,
        decimal=12,
        err_msg="compute_loss_residuals drift between src/probe.py and scripts/run_model.py",
    )
