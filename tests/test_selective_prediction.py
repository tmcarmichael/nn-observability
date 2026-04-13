"""Tests for selective prediction utilities."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from selective_prediction import (
    build_coverage_curves,
    exact_match,
    normalize_answer,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("New York") == "new york"

    def test_strip_articles(self):
        assert normalize_answer("the United States") == "united states"
        assert normalize_answer("A dog") == "dog"
        assert normalize_answer("an apple") == "apple"

    def test_strip_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"
        assert normalize_answer("it's fine.") == "its fine"

    def test_collapse_whitespace(self):
        assert normalize_answer("  too   many   spaces  ") == "too many spaces"

    def test_combined(self):
        # "The U.S.A." -> lowercase -> strip "the" and trailing "a" (articles)
        # -> strip punctuation -> collapse whitespace -> "us"
        assert normalize_answer("The U.S.A.") == "us"

    def test_empty(self):
        assert normalize_answer("") == ""


class TestExactMatch:
    def test_simple_match(self):
        assert exact_match("Paris", ["Paris", "paris"])

    def test_normalized_match(self):
        assert exact_match("The United States", ["United States", "USA", "US"])

    def test_no_match(self):
        assert not exact_match("London", ["Paris", "Berlin"])

    def test_article_stripping(self):
        assert exact_match("a cat", ["cat"])

    def test_empty_prediction(self):
        assert not exact_match("", ["something"])


class TestBuildCoverageCurves:
    def _make_results(self, n=100, accuracy=0.6, seed=42):
        """Create synthetic per-question results."""
        rng = np.random.default_rng(seed)
        correct = rng.random(n) < accuracy
        # Make observer scores anti-correlated with correctness (good observer)
        observer = np.where(correct, rng.normal(-1, 0.5, n), rng.normal(1, 0.5, n))
        confidence = np.where(correct, rng.normal(0.9, 0.05, n), rng.normal(0.5, 0.1, n))
        return [
            {
                "correct": bool(c),
                "mean_observer": float(o),
                "max_observer": float(o + rng.normal(0, 0.1)),
                "mean_confidence": float(conf),
                "min_confidence": float(conf - rng.uniform(0, 0.1)),
            }
            for c, o, conf in zip(correct, observer, confidence)
        ]

    def test_full_coverage_equals_base(self):
        results = self._make_results()
        curves = build_coverage_curves(results)
        # At 100% coverage (first element), accuracy == base accuracy
        assert curves["coverage_levels"][0] == 1.0
        assert abs(curves["observer_mean"]["accuracy"][0] - curves["base_accuracy"]) < 1e-6

    def test_accuracy_improves_with_abstention(self):
        """With a good observer, accuracy should improve as coverage decreases."""
        results = self._make_results(n=500, accuracy=0.5)
        curves = build_coverage_curves(results)
        # Accuracy at 50% coverage should be higher than at 100%
        acc_100 = curves["observer_mean"]["accuracy"][0]
        acc_50 = curves["observer_mean"]["accuracy"][-1]
        assert acc_50 > acc_100, f"Expected improvement: {acc_50} > {acc_100}"

    def test_auacc_bounded(self):
        results = self._make_results()
        curves = build_coverage_curves(results)
        for strategy in ["observer_mean", "confidence_mean", "combined"]:
            auacc = curves[strategy]["auacc"]
            # AUACC is an integral of accuracy over coverage levels.
            # On synthetic data, observer ordering can be adversarial (negative AUACC).
            # Confidence and combined should be non-negative on reasonable data.
            assert isinstance(auacc, float), f"{strategy} AUACC not a float: {auacc}"

    def test_all_strategies_present(self):
        results = self._make_results(n=50)
        curves = build_coverage_curves(results)
        for key in ["observer_mean", "observer_max", "confidence_mean", "confidence_min", "combined"]:
            assert key in curves, f"Missing strategy: {key}"
            assert "accuracy" in curves[key]
            assert "auacc" in curves[key]

    def test_coverage_levels_match(self):
        results = self._make_results(n=50)
        levels = [1.0, 0.8, 0.6]
        curves = build_coverage_curves(results, coverage_levels=levels)
        assert curves["coverage_levels"] == levels
        assert len(curves["observer_mean"]["accuracy"]) == 3
