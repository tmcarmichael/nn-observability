"""Pin absolute outputs of producer-side statistical primitives.

`test_probe_sync.py` enforces that every inlined copy of `partial_spearman` and
`compute_loss_residuals` in `scripts/` matches the canonical `src/probe.py`
implementation. That test catches drift between copies but cannot catch
simultaneous drift across all copies (e.g., a numpy/scipy upgrade that
shifts both src and inlined outputs together). The tests here pin both
`src/probe.py` and `src/observe.py` against fixed expected values computed
under the current numerical stack, at 1e-12 tolerance.

Inputs use a different seed and length than `test_probe_sync.py` so the two
tests cover non-overlapping points in input space.
"""

from __future__ import annotations

import numpy as np
import pytest

# src/ is on pythonpath via pyproject.toml
import observe
import probe


def _fixture_partial_spearman() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    rng = np.random.RandomState(12345)
    n = 200
    x = rng.standard_normal(n)
    y = 0.4 * x + 0.6 * rng.standard_normal(n)
    c1 = 0.3 * x + 0.7 * rng.standard_normal(n)
    c2 = rng.standard_normal(n)
    return x, y, [c1, c2]


def _fixture_loss_residuals() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Same RandomState advances past the partial_spearman draws, so this fixture
    # rebuilds it and skips ahead to keep the residuals input deterministic
    # without coupling the two test paths.
    rng = np.random.RandomState(12345)
    n = 200
    rng.standard_normal(n)  # x
    rng.standard_normal(n)  # y noise
    rng.standard_normal(n)  # c1 noise
    rng.standard_normal(n)  # c2
    losses = rng.exponential(2.0, n)
    margins = rng.beta(5, 2, n)
    norms = rng.lognormal(0, 1, n)
    return losses, margins, norms


# Expected values captured under numpy 2.x / scipy 1.x on the v5.0.0 stack.
# Any drift here means the underlying numerical environment has shifted and
# previously-committed result JSONs may no longer reproduce bit-for-bit.
EXPECTED_PSPEARMAN_R = 0.47306051369433955
EXPECTED_PSPEARMAN_P = 1.5102919411349478e-12

EXPECTED_RESIDUALS_SUMSQ = 729.6346282943625
EXPECTED_RESIDUALS_FIRST5 = (
    -1.9067617734803675,
    -1.8352471826880532,
    2.11648178511346,
    -2.166140823790863,
    -1.2836855715087778,
)
EXPECTED_RESIDUALS_LAST5 = (
    -0.8630509987389472,
    -1.1221759235561606,
    -1.488811997960656,
    -1.8823846338509092,
    -1.0985881543135925,
)

TOL = 1e-12


@pytest.mark.parametrize(
    "partial_spearman",
    [probe.partial_spearman, observe.partial_spearman],
    ids=["probe", "observe"],
)
def test_partial_spearman_pinned_output(partial_spearman) -> None:
    """Both src copies must reproduce the captured (r, p) on the fixed input."""
    x, y, covariates = _fixture_partial_spearman()
    r, p = partial_spearman(x, y, covariates)
    assert r == pytest.approx(EXPECTED_PSPEARMAN_R, abs=TOL)
    assert p == pytest.approx(EXPECTED_PSPEARMAN_P, abs=TOL)


@pytest.mark.parametrize(
    "compute_loss_residuals",
    [probe.compute_loss_residuals, observe.compute_loss_residuals],
    ids=["probe", "observe"],
)
def test_compute_loss_residuals_pinned_output(compute_loss_residuals) -> None:
    """Both src copies must reproduce the captured residual vector on the fixed input."""
    losses, margins, norms = _fixture_loss_residuals()
    residuals = compute_loss_residuals(losses, margins, norms)

    # OLS residuals sum to zero up to floating-point error.
    assert residuals.sum() == pytest.approx(0.0, abs=1e-10)

    # Total energy in the residual is invariant to numerical reordering at this scale.
    assert (residuals**2).sum() == pytest.approx(EXPECTED_RESIDUALS_SUMSQ, abs=TOL)

    # Pin specific entries to detect element-wise drift that average-statistic
    # checks could miss.
    for actual, expected in zip(residuals[:5], EXPECTED_RESIDUALS_FIRST5):
        assert actual == pytest.approx(expected, abs=TOL)
    for actual, expected in zip(residuals[-5:], EXPECTED_RESIDUALS_LAST5):
        assert actual == pytest.approx(expected, abs=TOL)


def test_src_copies_agree_on_partial_spearman() -> None:
    """probe.partial_spearman and observe.partial_spearman are independently
    defined; if they ever diverge, this test fails before any committed
    paper number can shift."""
    x, y, covariates = _fixture_partial_spearman()
    r_probe, p_probe = probe.partial_spearman(x, y, covariates)
    r_observe, p_observe = observe.partial_spearman(x, y, covariates)
    assert r_probe == pytest.approx(r_observe, abs=TOL)
    assert p_probe == pytest.approx(p_observe, abs=TOL)


def test_src_copies_agree_on_compute_loss_residuals() -> None:
    """probe.compute_loss_residuals and observe.compute_loss_residuals are
    independently defined; if they ever diverge, this test fails before any
    committed paper number can shift."""
    losses, margins, norms = _fixture_loss_residuals()
    res_probe = probe.compute_loss_residuals(losses, margins, norms)
    res_observe = observe.compute_loss_residuals(losses, margins, norms)
    np.testing.assert_array_almost_equal(res_probe, res_observe, decimal=12)
