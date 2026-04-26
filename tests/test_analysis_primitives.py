"""Tests for the statistical primitives in analysis/.

Covers the pure functions the paper's inferential claims rest on:
family_f_stat, n_unique_permutations, partial_pearson, the partial-Spearman
in-sample and held-out variants, and Egger's test. Synthetic data with known
answers; no I/O, no plotting.

Run: uv run pytest tests/test_analysis_primitives.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.funnel_plot import eggers_test
from analysis.held_out_split import partial_spearman_held_out, partial_spearman_in_sample
from analysis.pearson_vs_spearman import partial_pearson
from analysis.permutation_test import family_f_stat, n_unique_permutations


class TestFamilyFStat:
    def test_zero_when_one_family(self):
        """One family, no between-group variance, F is undefined and returns 0."""
        families = ["A", "A", "A", "A"]
        log_params = np.array([0.1, 0.2, 0.3, 0.4])
        pcorrs = np.array([0.1, 0.2, 0.15, 0.25])
        assert family_f_stat(families, log_params, pcorrs) == 0.0

    def test_zero_when_no_within_variance(self):
        """All values identical within each family, ss_within is 0, returns 0."""
        families = ["A", "A", "B", "B"]
        log_params = np.array([0.0, 0.0, 0.0, 0.0])
        pcorrs = np.array([0.1, 0.1, 0.3, 0.3])
        assert family_f_stat(families, log_params, pcorrs) == 0.0

    def test_positive_when_groups_separate(self):
        """Two clearly separated families produce a large F."""
        families = ["A"] * 4 + ["B"] * 4
        log_params = np.array([0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3])
        pcorrs = np.array([0.10, 0.11, 0.09, 0.12, 0.30, 0.31, 0.29, 0.32])
        f = family_f_stat(families, log_params, pcorrs)
        assert f > 100.0

    def test_residualizes_against_scale(self):
        """A pure scaling effect across one family should not produce family signal."""
        families = ["A"] * 8
        log_params = np.linspace(0.0, 1.0, 8)
        pcorrs = 0.1 + 0.2 * log_params
        assert family_f_stat(families, log_params, pcorrs) == 0.0


class TestNUniquePermutations:
    def test_all_distinct_labels(self):
        assert n_unique_permutations(["A", "B", "C"]) == 6

    def test_with_repeats(self):
        """Multinomial: 4! / (2! * 2!) = 6."""
        assert n_unique_permutations(["A", "A", "B", "B"]) == 6

    def test_all_same(self):
        assert n_unique_permutations(["A", "A", "A"]) == 1

    def test_known_paper_case(self):
        """13 models in 6 families with paper's actual counts."""
        families = (
            ["Qwen"] * 5 + ["Llama"] * 3 + ["GPT-2"] * 1 + ["Mistral"] * 1 + ["Phi"] * 1 + ["Gemma"] * 2
        )
        result = n_unique_permutations(families)
        assert result > 0
        assert result < 6227020800


class TestPartialPearson:
    def test_zero_when_correlation_goes_through_covariate(self):
        """x and y correlate only through a shared covariate; partial corr is near 0."""
        rng = np.random.default_rng(0)
        c = rng.normal(size=500)
        x = c + 0.3 * rng.normal(size=500)
        y = c + 0.3 * rng.normal(size=500)
        r, _ = partial_pearson(x, y, [c])
        assert abs(r) < 0.15

    def test_preserves_direct_correlation(self):
        """A direct x-y signal independent of the covariate survives partialling."""
        rng = np.random.default_rng(0)
        c = rng.normal(size=500)
        signal = rng.normal(size=500)
        x = signal + 0.3 * c + 0.1 * rng.normal(size=500)
        y = signal + 0.3 * c + 0.1 * rng.normal(size=500)
        r, _ = partial_pearson(x, y, [c])
        assert r > 0.7

    def test_passes_through_when_no_covariates(self):
        """Empty covariates: partial pearson is ordinary Pearson against an intercept column."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = 0.5 * x + 0.5 * rng.normal(size=500)
        r, _ = partial_pearson(x, y, [])
        # Analytical expected: 0.5 / sqrt(1 * 0.5) = 0.707. Sample variation widens the band.
        assert 0.55 < r < 0.8


class TestPartialSpearmanHeldOut:
    def test_in_sample_matches_held_out_on_clean_signal(self):
        """When x and y share a clean signal independent of covariates,
        in-sample and held-out estimators agree closely."""
        rng = np.random.default_rng(1)
        n = 2000
        signal = rng.normal(size=n)
        c1 = rng.normal(size=n)
        c2 = rng.normal(size=n)
        x = signal + 0.1 * c1
        y = signal + 0.1 * c2
        r_in = partial_spearman_in_sample(x, y, [c1, c2])
        r_held, _ = partial_spearman_held_out(x, y, [c1, c2], seed=0)
        assert abs(r_in - r_held) < 0.05
        assert r_in > 0.5

    def test_returns_per_fold_values(self):
        """Held-out variant returns mean plus per-fold list of length 2."""
        rng = np.random.default_rng(2)
        x = rng.normal(size=500)
        y = rng.normal(size=500)
        c = rng.normal(size=500)
        _, per_fold = partial_spearman_held_out(x, y, [c], seed=0)
        assert len(per_fold) == 2


class TestEggersTest:
    def test_no_asymmetry_on_balanced_input(self):
        """Symmetric distribution of effects across precisions: intercept near 0."""
        rng = np.random.default_rng(3)
        n = 20
        ses = np.linspace(0.005, 0.05, n)
        means = rng.normal(loc=0.2, scale=ses)
        intercept, _, p_value = eggers_test(list(means), list(ses))
        assert abs(intercept) < 1.0
        assert 0.0 <= p_value <= 1.0

    def test_returns_three_values(self):
        means = [0.2, 0.25, 0.3, 0.18, 0.22]
        ses = [0.01, 0.02, 0.03, 0.015, 0.025]
        result = eggers_test(means, ses)
        assert len(result) == 3
        intercept, t_stat, p_value = result
        assert isinstance(intercept, float)
        assert 0.0 <= p_value <= 1.0

    def test_handles_two_studies_gracefully(self):
        """With n <= 2 the test is undefined; should return a sentinel rather than raise."""
        result = eggers_test([0.2, 0.3], [0.01, 0.02])
        assert result == (0, 0, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
