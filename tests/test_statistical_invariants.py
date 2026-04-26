"""Property-based tests for statistical primitives.

Verifies mathematical invariants of partial_spearman and family_f_stat
across random inputs using Hypothesis.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from analysis.permutation_test import family_f_stat

# Import from src/ (on pythonpath via pyproject.toml)
from observe import compute_loss_residuals, partial_spearman


def finite_floats(min_val: float = -100, max_val: float = 100) -> st.SearchStrategy:
    return st.floats(min_value=min_val, max_value=max_val, allow_nan=False, allow_infinity=False)


def float_arrays(n: int, min_val: float = -100, max_val: float = 100) -> st.SearchStrategy:
    return arrays(dtype=np.float64, shape=n, elements=finite_floats(min_val, max_val))


# ── partial_spearman ────────────────────────────────────────────────


@given(st.integers(min_value=10, max_value=50))
@settings(max_examples=30)
def test_partial_spearman_returns_bounded(n: int) -> None:
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    cov = [rng.standard_normal(n)]
    r, p = partial_spearman(x, y, cov)
    assert -1.0 <= r <= 1.0
    assert 0.0 <= p <= 1.0


@given(st.integers(min_value=10, max_value=50))
@settings(max_examples=30)
def test_partial_spearman_self_correlation(n: int) -> None:
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    cov = [rng.standard_normal(n)]
    r, _p = partial_spearman(x, x, cov)
    assert r > 0.9


@given(st.integers(min_value=10, max_value=50))
@settings(max_examples=30)
def test_partial_spearman_symmetric(n: int) -> None:
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    cov = [rng.standard_normal(n)]
    r_xy, _ = partial_spearman(x, y, cov)
    r_yx, _ = partial_spearman(y, x, cov)
    assert abs(r_xy - r_yx) < 1e-10


@given(st.integers(min_value=20, max_value=50))
@settings(max_examples=20)
def test_partial_spearman_covariate_removes_shared_signal(n: int) -> None:
    rng = np.random.default_rng(n)
    z = rng.standard_normal(n)
    x = z + rng.standard_normal(n) * 0.1
    y = z + rng.standard_normal(n) * 0.1
    r_raw, _ = partial_spearman(x, y, [rng.standard_normal(n)])
    r_controlled, _ = partial_spearman(x, y, [z])
    assert abs(r_controlled) < abs(r_raw) + 0.05


# ── family_f_stat ───────────────────────────────────────────────────


@given(st.integers(min_value=6, max_value=20))
@settings(max_examples=30)
def test_family_f_stat_nonnegative(n: int) -> None:
    rng = np.random.default_rng(n)
    families = [["A", "B", "C"][i % 3] for i in range(n)]
    log_params = rng.standard_normal(n).astype(np.float64)
    pcorrs = rng.standard_normal(n).astype(np.float64)
    f = family_f_stat(families, np.array(log_params), np.array(pcorrs))
    assert f >= 0.0


@given(st.integers(min_value=6, max_value=20))
@settings(max_examples=30)
def test_family_f_stat_single_family_is_zero(n: int) -> None:
    rng = np.random.default_rng(n)
    families = ["A"] * n
    log_params = rng.standard_normal(n).astype(np.float64)
    pcorrs = rng.standard_normal(n).astype(np.float64)
    f = family_f_stat(families, np.array(log_params), np.array(pcorrs))
    assert f == 0.0


@given(st.integers(min_value=10, max_value=30))
@settings(max_examples=20)
def test_family_f_stat_separated_groups_large(n: int) -> None:
    rng = np.random.default_rng(n)
    half = n // 2
    families = ["A"] * half + ["B"] * (n - half)
    log_params = rng.standard_normal(n).astype(np.float64)
    pcorrs = np.concatenate([rng.standard_normal(half) + 10, rng.standard_normal(n - half) - 10])
    f = family_f_stat(families, np.array(log_params), np.array(pcorrs))
    assert f > 1.0


# ── compute_loss_residuals ──────────────────────────────────────────


@given(st.integers(min_value=10, max_value=50))
@settings(max_examples=30)
def test_loss_residuals_mean_near_zero(n: int) -> None:
    rng = np.random.default_rng(n)
    losses = rng.standard_normal(n)
    margins = rng.standard_normal(n)
    norms = rng.standard_normal(n)
    resid = compute_loss_residuals(losses, margins, norms)
    assert abs(resid.mean()) < 1e-10


@given(st.integers(min_value=10, max_value=50))
@settings(max_examples=30)
def test_loss_residuals_uncorrelated_with_covariates(n: int) -> None:
    rng = np.random.default_rng(n)
    losses = rng.standard_normal(n)
    margins = rng.standard_normal(n)
    norms = rng.standard_normal(n)
    resid = compute_loss_residuals(losses, margins, norms)
    corr_margin = np.corrcoef(resid, margins)[0, 1]
    corr_norm = np.corrcoef(resid, norms)[0, 1]
    assert abs(corr_margin) < 1e-10
    assert abs(corr_norm) < 1e-10
