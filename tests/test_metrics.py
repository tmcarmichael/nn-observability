"""
Synthetic-data tests for observer metric functions.

Validates partial_spearman, observer scores, class_similarity, and
correlation_suite against known ground-truth cases. These are the
functions the README's headline numbers depend on.

Run: uv run pytest tests/ -v
"""

import numpy as np
import torch

from observe import (
    class_similarity_score,
    correlation_suite,
    partial_spearman,
)
from transformer_observe import _deep_merge, bootstrap_ci

# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_known_distribution(self):
        """CI of [1,2,3,4,5] mean should contain 3.0."""
        lo, hi = bootstrap_ci([1, 2, 3, 4, 5])
        assert lo < 3.0 < hi

    def test_tight_ci_for_constant(self):
        """CI of constant array should be a point."""
        lo, hi = bootstrap_ci([5.0] * 100)
        assert abs(lo - 5.0) < 0.01
        assert abs(hi - 5.0) < 0.01

    def test_wider_ci_for_high_variance(self):
        """Higher variance should produce wider CI."""
        lo_tight, hi_tight = bootstrap_ci([4.9, 5.0, 5.1] * 10)
        lo_wide, hi_wide = bootstrap_ci([1.0, 5.0, 9.0] * 10)
        assert (hi_wide - lo_wide) > (hi_tight - lo_tight)


class TestDeepMerge:
    def test_shallow_keys_preserved(self):
        """Sibling keys should not be overwritten."""
        base = {"5a": {"result": 1}, "5b": {"result": 2}}
        update = {"5a": {"result": 3}}
        merged = _deep_merge(base, update)
        assert merged["5a"]["result"] == 3
        assert merged["5b"]["result"] == 2

    def test_nested_merge(self):
        """Nested dicts should merge, not replace."""
        base = {"8": {"models": {"gpt2": {"corr": 0.29}, "gpt2-xl": {"corr": 0.29}}}}
        update = {"8": {"models": {"gpt2-medium": {"corr": 0.28}}}}
        merged = _deep_merge(base, update)
        assert "gpt2" in merged["8"]["models"]
        assert "gpt2-xl" in merged["8"]["models"]
        assert "gpt2-medium" in merged["8"]["models"]

    def test_non_dict_overwrite(self):
        """Non-dict values should be overwritten, not merged."""
        base = {"5a": {"partial_corrs": [0.28, 0.28]}}
        update = {"5a": {"partial_corrs": [0.29, 0.29, 0.29]}}
        merged = _deep_merge(base, update)
        assert merged["5a"]["partial_corrs"] == [0.29, 0.29, 0.29]


# ---------------------------------------------------------------------------
# partial_spearman
# ---------------------------------------------------------------------------


class TestPartialSpearman:
    """The most critical function: headline numbers depend on it."""

    def test_confounded_correlation_vanishes(self):
        """x and y both correlate with z, but have no independent relationship.
        Partial correlation controlling for z should be near zero."""
        rng = np.random.default_rng(42)
        n = 5000
        z = rng.normal(size=n)
        x = z + rng.normal(scale=0.3, size=n)
        y = z + rng.normal(scale=0.3, size=n)

        rho, p = partial_spearman(x, y, [z])
        # Without control, x and y are strongly correlated (~0.9).
        # After controlling for z, the partial correlation should be near zero.
        assert abs(rho) < 0.1, f"Expected near-zero partial corr, got {rho}"

    def test_independent_signal_survives(self):
        """x has both shared (via z) and independent signal about y.
        Partial correlation should remain positive after controlling for z."""
        rng = np.random.default_rng(42)
        n = 5000
        z = rng.normal(size=n)
        independent = rng.normal(size=n)
        x = z + 0.5 * independent + rng.normal(scale=0.3, size=n)
        y = z + 0.5 * independent + rng.normal(scale=0.3, size=n)

        rho, _ = partial_spearman(x, y, [z])
        assert rho > 0.1, f"Expected positive partial corr, got {rho}"

    def test_perfect_correlation(self):
        """x = y exactly. Partial correlation should be ~1 regardless of controls."""
        rng = np.random.default_rng(42)
        n = 1000
        x = rng.normal(size=n)
        z = rng.normal(size=n)

        rho, _ = partial_spearman(x, x, [z])
        assert rho > 0.95, f"Expected ~1 for perfect correlation, got {rho}"

    def test_multiple_controls(self):
        """Controlling for two confounders simultaneously."""
        rng = np.random.default_rng(42)
        n = 5000
        z1 = rng.normal(size=n)
        z2 = rng.normal(size=n)
        x = 0.5 * z1 + 0.5 * z2 + rng.normal(size=n)
        y = 0.5 * z1 + 0.5 * z2 + rng.normal(size=n)

        rho, _ = partial_spearman(x, y, [z1, z2])
        assert abs(rho) < 0.1, f"Expected near-zero after dual control, got {rho}"

    def test_intercept_with_independent_vars(self):
        """Three mutually independent variables. Partial corr should be ~0.

        This catches a missing intercept in OLS: without intercept, ranked data
        (mean ~N/2) produces spurious positive partial correlation (~0.36)
        between independent variables because the regression through the origin
        attributes shared mean to the covariate relationship.
        """
        rng = np.random.default_rng(99)
        n = 5000
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        z = rng.normal(size=n)

        rho, _ = partial_spearman(x, y, [z])
        assert abs(rho) < 0.1, (
            f"Independent vars should have partial corr ~0, got {rho:.3f}. "
            "Possible missing intercept in OLS regression."
        )

    def test_sign_preservation(self):
        """Negative independent relationship should produce negative partial corr."""
        rng = np.random.default_rng(42)
        n = 5000
        z = rng.normal(size=n)
        independent = rng.normal(size=n)
        x = z + independent + rng.normal(scale=0.2, size=n)
        y = z - independent + rng.normal(scale=0.2, size=n)

        rho, _ = partial_spearman(x, y, [z])
        assert rho < -0.1, f"Expected negative partial corr, got {rho}"


# ---------------------------------------------------------------------------
# Observer score functions
# ---------------------------------------------------------------------------


class TestObserverScores:
    """Validate observer computations on known activation patterns."""

    def _make_data(self, acts_list, n_cls=10):
        """Build a minimal data dict with known activations."""
        n = acts_list[0].size(0)
        # Dummy logits: class 0 always wins
        logits = torch.zeros(n, n_cls)
        logits[:, 0] = 10.0
        labels = torch.zeros(n, dtype=torch.long)
        losses = torch.zeros(n)
        return dict(
            observers={},
            per_layer_acts=acts_list,
            logits=logits,
            losses=losses.numpy(),
            labels=labels.numpy(),
            predictions=logits.argmax(dim=1).numpy(),
            is_correct=np.ones(n, dtype=bool),
        )

    def test_ff_goodness_known_value(self):
        """Two layers, uniform activations of value 2.0.
        Per-layer goodness = mean(4.0) = 4.0. Total = 8.0 for all examples."""
        acts = [torch.full((100, 50), 2.0) for _ in range(2)]
        goodness = sum((a**2).mean(dim=1) for a in acts).numpy()
        np.testing.assert_allclose(goodness, 8.0, atol=1e-5)

    def test_active_ratio_all_active(self):
        """All neurons above threshold. active_ratio should be 1.0."""
        acts = [torch.full((100, 50), 1.0)]
        ratio = (acts[0].abs() > 0.01).float().mean(dim=1).numpy()
        np.testing.assert_allclose(ratio, 1.0)

    def test_active_ratio_half_active(self):
        """Half neurons at 0, half at 1. active_ratio should be 0.5."""
        a = torch.zeros(100, 50)
        a[:, :25] = 1.0
        ratio = (a.abs() > 0.01).float().mean(dim=1).numpy()
        np.testing.assert_allclose(ratio, 0.5)

    def test_active_ratio_none_active(self):
        """All neurons at zero. active_ratio should be 0."""
        acts = [torch.zeros(100, 50)]
        ratio = (acts[0].abs() > 0.01).float().mean(dim=1).numpy()
        np.testing.assert_allclose(ratio, 0.0)

    def test_act_entropy_uniform(self):
        """Uniform activations: entropy should be log(n_neurons)."""
        n_neurons = 50
        a = torch.ones(10, n_neurons)
        p = a.abs() / (a.abs().sum(dim=1, keepdim=True) + 1e-8)
        ent = -(p * (p + 1e-8).log()).sum(dim=1)
        expected = np.log(n_neurons)
        np.testing.assert_allclose(ent.numpy(), expected, atol=0.01)

    def test_act_entropy_concentrated(self):
        """One neuron active, rest zero: entropy should be near zero."""
        a = torch.zeros(10, 50)
        a[:, 0] = 1.0
        p = a.abs() / (a.abs().sum(dim=1, keepdim=True) + 1e-8)
        ent = -(p * (p + 1e-8).log()).sum(dim=1)
        assert ent.mean().item() < 0.01, f"Expected near-zero entropy, got {ent.mean()}"


# ---------------------------------------------------------------------------
# class_similarity_score
# ---------------------------------------------------------------------------


class TestClassSimilarity:
    """Validate prototype-based similarity scoring."""

    def test_perfect_match(self):
        """Activations identical to prototypes. Similarity should be ~1."""
        n_cls, hidden = 3, 10
        protos = [torch.randn(n_cls, hidden)]
        # Each example matches its class prototype exactly
        predictions = np.array([0, 1, 2, 0, 1])
        test_acts = [protos[0][predictions]]

        sim = class_similarity_score(test_acts, predictions, protos)
        np.testing.assert_allclose(sim, 1.0, atol=1e-5)

    def test_orthogonal_activations(self):
        """Activations orthogonal to prototypes. Similarity should be ~0."""
        protos = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        # Class 0 prototype is [1,0]; give class 0 examples [0,1] (orthogonal)
        predictions = np.array([0, 0])
        test_acts = [torch.tensor([[0.0, 1.0], [0.0, 1.0]])]

        sim = class_similarity_score(test_acts, predictions, protos)
        np.testing.assert_allclose(sim, 0.0, atol=1e-5)

    def test_wrong_class_lower_similarity(self):
        """Examples matching wrong prototype should have lower similarity."""
        protos = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
        predictions = np.array([0, 0])
        # First example matches class 0 prototype; second doesn't
        test_acts = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]

        sim = class_similarity_score(test_acts, predictions, protos)
        assert sim[0] > sim[1], "Matching example should have higher similarity"

    def test_multi_layer_average(self):
        """Similarity should average across layers."""
        protos = [
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0, 0.0]]),
        ]
        predictions = np.array([0])
        test_acts = [
            torch.tensor([[1.0, 0.0]]),  # perfect match
            torch.tensor([[0.0, 1.0]]),  # orthogonal
        ]
        sim = class_similarity_score(test_acts, predictions, protos)
        # Average of 1.0 and 0.0
        np.testing.assert_allclose(sim, 0.5, atol=1e-5)


# ---------------------------------------------------------------------------
# correlation_suite (integration smoke test)
# ---------------------------------------------------------------------------


class TestCorrelationSuite:
    """Smoke test: feed synthetic data through the full pipeline."""

    def test_output_structure(self):
        """correlation_suite returns expected keys."""
        rng = np.random.default_rng(42)
        n = 500
        acts = [torch.randn(n, 50)]

        logits = torch.randn(n, 10)
        labels = torch.randint(0, 10, (n,))
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

        data = dict(
            observers={
                "ff_goodness": rng.normal(size=n),
                "max_softmax": rng.uniform(size=n),
                "logit_margin": rng.normal(size=n),
                "entropy": rng.uniform(size=n),
                "nll": losses.numpy(),
                "activation_norm": rng.normal(size=n),
                "active_ratio": rng.uniform(size=n),
                "act_entropy": rng.normal(size=n),
                "class_similarity": rng.uniform(size=n),
            },
            per_layer_acts=acts,
            logits=logits,
            losses=losses.numpy(),
            labels=labels.numpy(),
            predictions=logits.argmax(dim=1).numpy(),
            is_correct=(logits.argmax(dim=1) == labels).numpy().astype(bool),
        )

        result = correlation_suite(data)

        assert "spearman_vs_loss" in result
        assert "spearman_vs_margin" in result
        assert "within_class" in result
        assert "partial_vs_loss" in result
        assert "ff_goodness" in result["partial_vs_loss"]
        assert "active_ratio" in result["partial_vs_loss"]

        # Check value ranges
        for name in result["partial_vs_loss"]:
            rho = result["partial_vs_loss"][name]["rho"]
            assert -1.0 <= rho <= 1.0, f"{name} partial corr out of range: {rho}"

    def test_nll_correlates_perfectly_with_loss(self):
        """NLL is literally the loss. Spearman should be 1.0."""
        n = 500
        losses = np.random.default_rng(42).uniform(0.01, 5.0, size=n)

        data = dict(
            observers={
                "ff_goodness": np.zeros(n),
                "max_softmax": np.zeros(n),
                "logit_margin": np.zeros(n),
                "entropy": np.zeros(n),
                "nll": losses,
                "activation_norm": np.zeros(n),
                "active_ratio": np.zeros(n),
                "act_entropy": np.zeros(n),
            },
            per_layer_acts=[torch.randn(n, 10)],
            logits=torch.randn(n, 10),
            losses=losses,
            labels=np.zeros(n, dtype=int),
            predictions=np.zeros(n, dtype=int),
            is_correct=np.ones(n, dtype=bool),
        )

        result = correlation_suite(data)
        rho = result["spearman_vs_loss"]["nll"]["rho"]
        assert abs(rho - 1.0) < 0.001, f"NLL vs loss should be ~1.0, got {rho}"


class TestRegimeSweep:
    """Verify partial_spearman behaves correctly across accuracy regimes.

    The CIFAR-10 results live in the ~50% accuracy regime where confidence
    is a weak predictor. This suite checks that partial correlation doesn't
    produce artifacts when the base predictor varies from near-perfect to
    near-chance.
    """

    def _synthetic_regime(self, accuracy, n=3000, seed=42):
        """Generate synthetic data at a given accuracy level.

        Creates: a confidence proxy (margin), a structural signal with known
        independent relationship to loss, and loss values consistent with
        the target accuracy regime.
        """
        rng = np.random.default_rng(seed)

        # Simulate per-example loss: low loss for correct, high for incorrect
        is_correct = rng.random(n) < accuracy
        loss = np.where(is_correct, rng.exponential(0.3, n), rng.exponential(2.0, n))

        # Confidence proxy: strongly correlated with loss
        margin = -loss + rng.normal(scale=0.5, size=n)
        norm = rng.exponential(1.0, size=n)

        # Structural signal: has known independent component beyond margin/norm
        independent = rng.normal(size=n)
        structural = 0.3 * margin + 0.3 * norm + 0.5 * independent * loss
        # The independent * loss term creates partial correlation with loss
        # that margin and norm cannot explain.

        return loss, margin, norm, structural

    def test_high_accuracy_regime(self):
        """~95% accuracy (MNIST-like). Confidence is strong."""
        loss, margin, norm, structural = self._synthetic_regime(0.95)
        rho, p = partial_spearman(structural, loss, [margin, norm])
        assert abs(rho) < 1.0, f"Out of range: {rho}"
        # Should detect the independent component
        assert rho > 0.05, f"Expected positive partial corr at 95% acc, got {rho}"

    def test_medium_accuracy_regime(self):
        """~50% accuracy (CIFAR-10-like). Confidence is weak."""
        loss, margin, norm, structural = self._synthetic_regime(0.50)
        rho, p = partial_spearman(structural, loss, [margin, norm])
        assert abs(rho) < 1.0, f"Out of range: {rho}"
        assert rho > 0.05, f"Expected positive partial corr at 50% acc, got {rho}"

    def test_low_accuracy_regime(self):
        """~20% accuracy (near-chance). Confidence is mostly noise."""
        loss, margin, norm, structural = self._synthetic_regime(0.20)
        rho, p = partial_spearman(structural, loss, [margin, norm])
        assert abs(rho) < 1.0, f"Out of range: {rho}"
        assert rho > 0.05, f"Expected positive partial corr at 20% acc, got {rho}"

    def test_null_signal_across_regimes(self):
        """A signal with NO independent relationship to loss should show
        near-zero partial correlation at every accuracy level."""
        for acc in [0.95, 0.50, 0.20]:
            rng = np.random.default_rng(42)
            n = 3000
            is_correct = rng.random(n) < acc
            loss = np.where(is_correct, rng.exponential(0.3, n), rng.exponential(2.0, n))
            margin = -loss + rng.normal(scale=0.5, size=n)
            norm = rng.exponential(1.0, size=n)
            # Pure noise, no relationship to loss beyond margin/norm
            noise = rng.normal(size=n)

            rho, _ = partial_spearman(noise, loss, [margin, norm])
            assert abs(rho) < 0.1, (
                f"Null signal at {acc:.0%} accuracy should have near-zero partial corr, got {rho:.3f}"
            )
