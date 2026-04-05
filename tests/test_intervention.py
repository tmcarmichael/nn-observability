"""
Tests for intervention_curves ranking behavior.

The test-set leakage bug: neuron importance was computed from test data,
then evaluated on the same test data. These tests verify the ranking_acts
parameter controls which data is used for neuron ranking.

Run: uv run pytest tests/test_intervention.py -v
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from observe import intervention_curves
from train import BPNet


def _make_model_and_loader(in_dim=10, hidden=20, n_cls=3, n_examples=50):
    """Create a small BPNet and synthetic DataLoader."""
    torch.manual_seed(42)
    model = BPNet([in_dim, hidden], n_cls)
    X = torch.randn(n_examples, in_dim)
    Y = torch.randint(0, n_cls, (n_examples,))
    loader = DataLoader(TensorDataset(X, Y), batch_size=n_examples)
    return model, loader, hidden


class TestRankingActs:
    """Verify intervention_curves uses ranking_acts for neuron importance."""

    def test_backward_compat_without_ranking_acts(self):
        """Without ranking_acts, function should use data['per_layer_acts']."""
        model, loader, hidden = _make_model_and_loader()
        test_acts = [torch.randn(50, hidden)]
        data = {
            "per_layer_acts": test_acts,
            "labels": np.zeros(50, dtype=int),
        }

        # Should not raise
        result = intervention_curves(
            model,
            loader,
            data,
            "cpu",
            fractions=(0.0, 0.5),
            n_random_trials=1,
        )
        assert "layers" in result
        assert "0" in result["layers"]

    def test_same_data_same_results(self):
        """Passing test data as ranking_acts should match default behavior."""
        model, loader, hidden = _make_model_and_loader()
        acts = [torch.randn(50, hidden)]
        labels = np.zeros(50, dtype=int)
        data = {"per_layer_acts": acts, "labels": labels}

        result_default = intervention_curves(
            model,
            loader,
            data,
            "cpu",
            fractions=(0.0, 0.5),
            n_random_trials=1,
        )
        result_explicit = intervention_curves(
            model,
            loader,
            data,
            "cpu",
            fractions=(0.0, 0.5),
            n_random_trials=1,
            ranking_acts=acts,
            ranking_labels=labels,
        )

        for strategy in ["ff_targeted", "magnitude", "sparsity"]:
            assert result_default["layers"]["0"][strategy] == result_explicit["layers"]["0"][strategy], (
                f"{strategy}: explicit ranking_acts == test data should match default"
            )

    def test_different_ranking_data_changes_ordering(self):
        """Neuron rankings from different data should produce different
        ablation results, confirming ranking_acts is actually used."""
        model, loader, hidden = _make_model_and_loader()
        data = {
            "per_layer_acts": [torch.randn(50, hidden)],
            "labels": np.zeros(50, dtype=int),
        }

        # Ranking A: energy concentrated in first 5 neurons
        ranking_a = [torch.zeros(50, hidden)]
        ranking_a[0][:, :5] = 10.0

        # Ranking B: energy concentrated in last 5 neurons
        ranking_b = [torch.zeros(50, hidden)]
        ranking_b[0][:, -5:] = 10.0

        result_a = intervention_curves(
            model,
            loader,
            data,
            "cpu",
            fractions=(0.0, 0.5),
            n_random_trials=1,
            ranking_acts=ranking_a,
            ranking_labels=np.zeros(50, dtype=int),
        )
        result_b = intervention_curves(
            model,
            loader,
            data,
            "cpu",
            fractions=(0.0, 0.5),
            n_random_trials=1,
            ranking_acts=ranking_b,
            ranking_labels=np.zeros(50, dtype=int),
        )

        # ff_targeted ablates highest-importance neurons first.
        # With opposite importance patterns, the ablation order differs,
        # so the resulting accuracy should differ.
        a_targeted = result_a["layers"]["0"]["ff_targeted"]
        b_targeted = result_b["layers"]["0"]["ff_targeted"]
        assert a_targeted != b_targeted, "Opposite ranking patterns should produce different ablation results"
