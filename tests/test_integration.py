"""
End-to-end pipeline tests with synthetic data.

Verifies that training loops produce valid output structures and that
key algorithmic properties hold. Uses tiny models and 2 epochs to
stay fast (~5s per test on CPU).

Run: uv run pytest tests/test_integration.py -v
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from observe import (
    collect_activations,
    compute_observers,
    correlation_suite,
    fit_probe,
    intervention_curves,
    prediction_aucs,
)
from train import (
    BPNet,
    FFLayer,
    FFNet,
    collect,
    eval_bp,
    eval_ff,
    eval_layer,
    train_bp,
    train_ff,
)


def _synthetic_loaders(n=200, in_dim=784, n_cls=10, batch=100):
    """Tiny synthetic data matching MNIST dimensions."""
    torch.manual_seed(0)
    X = torch.randn(n, in_dim)
    Y = torch.randint(0, n_cls, (n,))
    train_dl = DataLoader(TensorDataset(X, Y), batch_size=batch, shuffle=True)
    test_dl = DataLoader(TensorDataset(X, Y), batch_size=batch)
    return train_dl, test_dl, in_dim, n_cls


# ---------------------------------------------------------------------------
# FFLayer.train_step correctness
# ---------------------------------------------------------------------------


class TestTrainStep:
    """The train_step bug: post-update activations passed to next layer."""

    def test_returns_pre_update_activations(self):
        """train_step should return activations computed BEFORE the weight
        update, not after re-forwarding through updated weights."""
        torch.manual_seed(42)
        layer = FFLayer(10, 20)

        pos = torch.randn(5, 10)
        neg = torch.randn(5, 10)

        # Snapshot pre-update activations
        with torch.no_grad():
            expected_pos = layer.forward(pos).clone()
            expected_neg = layer.forward(neg).clone()

        # train_step updates weights, then returns activations
        out_pos, out_neg, loss = layer.train_step(pos, neg)

        torch.testing.assert_close(out_pos, expected_pos)
        torch.testing.assert_close(out_neg, expected_neg)

    def test_weights_change_after_train_step(self):
        """Verify the weight update actually happened."""
        torch.manual_seed(42)
        layer = FFLayer(10, 20)
        w_before = layer.linear.weight.data.clone()

        pos = torch.randn(5, 10)
        neg = torch.randn(5, 10)
        layer.train_step(pos, neg)

        assert not torch.equal(w_before, layer.linear.weight.data), "Weights should change after train_step"

    def test_returned_activations_are_detached(self):
        """Returned activations should not carry gradients."""
        torch.manual_seed(42)
        layer = FFLayer(10, 20)
        pos = torch.randn(5, 10)
        neg = torch.randn(5, 10)

        out_pos, out_neg, _ = layer.train_step(pos, neg)
        assert not out_pos.requires_grad
        assert not out_neg.requires_grad


# ---------------------------------------------------------------------------
# Train pipeline (Phase 1)
# ---------------------------------------------------------------------------


class TestTrainPipeline:
    """End-to-end: train variants, verify output structure."""

    def test_ff_bp_train_and_eval(self):
        """Train FF and BP on synthetic data, verify valid accuracy."""
        train_dl, test_dl, in_dim, n_cls = _synthetic_loaders()

        torch.manual_seed(42)
        ff = FFNet([in_dim, 32, 32])
        train_ff(ff, train_dl, epochs=2, n_cls=n_cls, device="cpu")
        ff_acc = eval_ff(ff, test_dl, n_cls, "cpu")
        assert 0 <= ff_acc <= 1

        torch.manual_seed(42)
        bp = BPNet([in_dim, 32, 32], n_cls)
        train_bp(bp, train_dl, epochs=2, lr=0.001, device="cpu")
        bp_acc = eval_bp(bp, test_dl, "cpu")
        assert 0 <= bp_acc <= 1

    def test_collect_and_eval_layer(self):
        """Collect activations and compute per-layer metrics."""
        train_dl, test_dl, in_dim, n_cls = _synthetic_loaders()

        torch.manual_seed(42)
        bp = BPNet([in_dim, 32, 32], n_cls)
        train_bp(bp, train_dl, epochs=2, lr=0.001, device="cpu")

        tr_acts, tr_y = collect(bp, train_dl, "bp", n_cls, "cpu")
        te_acts, te_y = collect(bp, test_dl, "bp", n_cls, "cpu")
        assert len(tr_acts) == 2

        metrics = eval_layer(tr_acts[0], tr_y, te_acts[0], te_y, "layer_0", n_cls)
        for key in ["probe_acc", "sparsity", "dead_frac", "eff_rank", "polysemanticity"]:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], float)


# ---------------------------------------------------------------------------
# Observe pipeline (Phase 2/3)
# ---------------------------------------------------------------------------


class TestObservePipeline:
    """End-to-end: observer suite on a trained BP model."""

    def test_observer_output_schema(self):
        """compute_observers returns all expected keys and shapes."""
        train_dl, test_dl, in_dim, n_cls = _synthetic_loaders()

        torch.manual_seed(42)
        model = BPNet([in_dim, 32, 32], n_cls)
        train_bp(model, train_dl, epochs=2, lr=0.001, device="cpu")

        data = compute_observers(model, test_dl, "cpu")

        expected_observers = [
            "ff_goodness",
            "max_softmax",
            "logit_margin",
            "entropy",
            "nll",
            "activation_norm",
            "active_ratio",
            "act_entropy",
        ]
        for name in expected_observers:
            assert name in data["observers"], f"Missing observer: {name}"
            assert data["observers"][name].shape == (200,)

        assert len(data["per_layer_acts"]) == 2
        assert data["labels"].shape == (200,)
        assert data["is_correct"].dtype == bool

    def test_full_observe_pipeline(self):
        """Run correlation suite, intervention, and prediction AUCs."""
        train_dl, test_dl, in_dim, n_cls = _synthetic_loaders()

        torch.manual_seed(42)
        model = BPNet([in_dim, 32, 32], n_cls)
        train_bp(model, train_dl, epochs=2, lr=0.001, device="cpu")

        data = compute_observers(model, test_dl, "cpu")

        # Add probe and class_similarity (as run_once does)
        probe = fit_probe(model, train_dl, "cpu")
        data["observers"]["probe_confidence"] = probe.predict_proba(data["per_layer_acts"][-1].numpy()).max(
            axis=1
        )

        # Correlation suite
        corr = correlation_suite(data)
        assert "partial_vs_loss" in corr
        assert "ff_goodness" in corr["partial_vs_loss"]
        for name in corr["partial_vs_loss"]:
            rho = corr["partial_vs_loss"][name]["rho"]
            assert -1.0 <= rho <= 1.0, f"{name} out of range: {rho}"

        # Intervention (with training-data ranking)
        tr_acts, tr_labels = collect_activations(model, train_dl, "cpu")
        intervention = intervention_curves(
            model,
            test_dl,
            data,
            "cpu",
            fractions=(0.0, 0.5),
            n_random_trials=1,
            ranking_acts=tr_acts,
            ranking_labels=tr_labels,
        )
        assert "layers" in intervention
        assert "0" in intervention["layers"]
        for strategy in ["ff_targeted", "magnitude", "random_mean"]:
            assert strategy in intervention["layers"]["0"]
            assert len(intervention["layers"]["0"][strategy]) == 2

        # Prediction AUCs
        pred = prediction_aucs(data)
        assert "ff_goodness" in pred
        assert "auc" in pred["ff_goodness"]
