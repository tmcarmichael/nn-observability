"""
Shared-weight faithfulness testing.

Train a BP model, then test whether FF-style goodness on its activations
faithfully tracks BP decision-making. Pure observer mode: frozen BP model,
no label overlay, no separate model.

Three evaluation axes:
  1. Correlation: does goodness track decision-relevant signals beyond confidence?
  2. Intervention: do FF-guided ablations degrade performance faster than random?
  3. Prediction: can goodness rank likely failures?

All observer scores compared against cheap baselines (max softmax, logit margin,
entropy, NLL, activation norm, linear probe confidence).

Usage:
    uv run observe.py                          # MNIST, 50 epochs, 3 seeds
    uv run observe.py --dataset cifar10        # harder benchmark
    uv run observe.py --seeds 1 --epochs 5     # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import TypedDict

import matplotlib
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _import_train():
    """Lazy import of train.py to avoid torchvision dependency at import time.

    Metric functions (partial_spearman, compute_loss_residuals) don't need
    train.py. Only the training/main pipeline does.
    """
    from train import BPNet, eval_bp, get_data, overlay_label, train_bp, wrong_labels

    return BPNet, eval_bp, get_data, overlay_label, train_bp, wrong_labels


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ObserverData(TypedDict):
    observers: dict[str, npt.NDArray[np.floating]]
    per_layer_acts: list[torch.Tensor]
    logits: torch.Tensor
    losses: npt.NDArray[np.floating]
    labels: npt.NDArray[np.integer]
    predictions: npt.NDArray[np.integer]
    is_correct: npt.NDArray[np.bool_]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def compute_observers(model, loader, device) -> ObserverData:
    """Single pass over dataset: collect per-example observer scores and metadata."""
    model.eval()
    all_acts, all_logits, all_losses, all_labels = None, [], [], []
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            acts = model.get_activations(x)
            logits = model.head(acts[-1])
            losses = criterion(logits, y)

            if all_acts is None:
                all_acts = [[] for _ in acts]
            for i, a in enumerate(acts):
                all_acts[i].append(a.cpu())
            all_logits.append(logits.cpu())
            all_losses.append(losses.cpu())
            all_labels.append(y.cpu())

    acts = [torch.cat(a) for a in all_acts]
    logits = torch.cat(all_logits)
    losses = torch.cat(all_losses)
    labels = torch.cat(all_labels)
    predictions = logits.argmax(dim=1)
    is_correct = (predictions == labels).numpy().astype(bool)

    # Observer scores (all 1-D numpy arrays, length N)
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    sorted_logits = logits.sort(dim=1, descending=True).values

    # Per-example active neuron ratio (fraction above eps), averaged across layers
    n_layers = len(acts)
    active_ratio = sum((a.abs() > 0.01).float().mean(dim=1) for a in acts).numpy() / n_layers

    # Per-example activation entropy (concentration of activation mass), averaged
    act_entropy = torch.zeros(acts[0].size(0))
    for a in acts:
        p = a.abs() / (a.abs().sum(dim=1, keepdim=True) + 1e-8)
        act_entropy += -(p * (p + 1e-8).log()).sum(dim=1)
    act_entropy = (act_entropy / n_layers).numpy()

    observers = {
        "ff_goodness": sum((a**2).mean(dim=1) for a in acts).numpy(),
        "max_softmax": probs.max(dim=1).values.numpy(),
        "logit_margin": (sorted_logits[:, 0] - sorted_logits[:, 1]).numpy(),
        "entropy": -(probs * log_probs).sum(dim=1).numpy(),
        "nll": losses.numpy(),
        "activation_norm": torch.stack([a.norm(dim=1) for a in acts]).mean(dim=0).numpy(),
        "active_ratio": active_ratio,
        "act_entropy": act_entropy,
    }

    return dict(
        observers=observers,
        per_layer_acts=acts,
        logits=logits,
        losses=losses.numpy(),
        labels=labels.numpy(),
        predictions=predictions.numpy(),
        is_correct=is_correct,
    )


def fit_probe(model, loader, device, max_n=5000):
    """Fit logistic regression on last-layer training activations."""
    model.eval()
    xs, ys, n = [], [], 0
    with torch.inference_mode():
        for x, y in loader:
            if n >= max_n:
                break
            x = x.to(device)
            acts = model.get_activations(x)
            xs.append(acts[-1].cpu())
            ys.append(y)
            n += x.size(0)
    X = torch.cat(xs)[:max_n].numpy()
    y = torch.cat(ys)[:max_n].numpy()
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
    clf.fit(X, y)
    return clf


def compute_class_prototypes(model, loader, device, n_cls, max_n=5000):
    """Per-class mean activation at each layer, from training data."""
    model.eval()
    all_acts, all_y, n = None, [], 0
    with torch.inference_mode():
        for x, y in loader:
            if n >= max_n:
                break
            x = x.to(device)
            acts = model.get_activations(x)
            if all_acts is None:
                all_acts = [[] for _ in acts]
            for i, a in enumerate(acts):
                all_acts[i].append(a.cpu())
            all_y.append(y)
            n += x.size(0)

    layer_acts = [torch.cat(a)[:max_n] for a in all_acts]
    labels = torch.cat(all_y)[:max_n]

    prototypes = []
    for la in layer_acts:
        protos = torch.zeros(n_cls, la.size(1))
        for c in range(n_cls):
            mask = labels == c
            if mask.sum() > 0:
                protos[c] = la[mask].mean(dim=0)
        prototypes.append(protos)
    return prototypes


def class_similarity_score(test_acts, predictions, prototypes):
    """Per-example cosine similarity to predicted class prototype, averaged across layers."""
    preds = torch.from_numpy(predictions).long()
    total = torch.zeros(test_acts[0].size(0))
    for la, protos in zip(test_acts, prototypes, strict=True):
        pred_protos = protos[preds]
        total += F.cosine_similarity(la, pred_protos, dim=1)
    return (total / len(test_acts)).numpy()


def collect_activations(model, loader, device):
    """Collect per-layer activations and labels from a data loader."""
    model.eval()
    all_acts, all_labels = None, []
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            acts = model.get_activations(x)
            if all_acts is None:
                all_acts = [[] for _ in acts]
            for i, a in enumerate(acts):
                all_acts[i].append(a.cpu())
            all_labels.append(y)
    return [torch.cat(a) for a in all_acts], torch.cat(all_labels).numpy()


# ---------------------------------------------------------------------------
# Observer head
# ---------------------------------------------------------------------------


class ObserverHead(nn.Module):
    """Small MLP that reads BP activations and outputs a scalar quality score."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_loss_residuals(losses, margins, norms):
    """Remove the component of per-example loss explained by confidence proxies.

    Fits OLS: loss = a * margin + b * norm + c
    Returns the residual (what confidence cannot predict).
    """
    X = np.column_stack([margins, norms, np.ones(len(margins))])
    coef, _, _, _ = np.linalg.lstsq(X, losses, rcond=None)
    return losses - X @ coef


def train_observer_head(model, head, loader, device, epochs=20, lr=1e-3):
    """Train observer head on frozen BP activations to predict loss residuals.

    Collects all training activations, computes the component of per-example
    loss not explained by logit margin and activation norm, then trains the
    head to predict that residual from activations alone.
    """
    model.eval()
    all_acts, all_logits, all_losses = [], [], []
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            acts = model.get_activations(x)
            logits = model.head(acts[-1])
            losses = criterion(logits, y)
            all_acts.append(acts[-1].cpu())
            all_logits.append(logits.cpu())
            all_losses.append(losses.cpu())

    acts_t = torch.cat(all_acts)
    logits_t = torch.cat(all_logits)
    losses_np = torch.cat(all_losses).numpy()

    # Confidence metrics for residual computation
    sorted_logits = logits_t.sort(dim=1, descending=True).values
    margins = (sorted_logits[:, 0] - sorted_logits[:, 1]).numpy()
    norms = acts_t.norm(dim=1).numpy()

    # Loss residuals: what confidence can't explain
    residuals = compute_loss_residuals(losses_np, margins, norms)
    residuals_t = torch.from_numpy(residuals).float()

    # Train
    head.to(device)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(acts_t, residuals_t)
    dl = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    for ep in range(epochs):
        tot = n = 0
        for batch_x, batch_y in dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            pred = head(batch_x)
            loss = F.mse_loss(pred, batch_y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    ObsHead ep {ep + 1:3d}/{epochs}  mse={tot / n:.6f}")


# ---------------------------------------------------------------------------
# Auxiliary loss training
# ---------------------------------------------------------------------------


def train_bp_auxiliary(model, loader, epochs, bp_lr, device, n_cls, ff_weight=0.1, threshold=2.0):
    """Train BP with auxiliary FF goodness loss.

    BP loss on original inputs for classification. FF contrastive loss on
    label-overlaid inputs: positive (correct label) should produce high
    per-layer goodness, negative (wrong label) should produce low goodness.

    Evaluation later uses non-overlaid inputs only, testing whether the
    FF shaping persists in the raw representation.
    """
    _, _, _, overlay_label, _, wrong_labels = _import_train()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=bp_lr)
    bp_crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        tot_bp = tot_ff = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # BP loss on clean input
            logits = model(x)
            bp_loss = bp_crit(logits, y)

            # FF auxiliary: goodness contrast on overlaid inputs
            pos_acts = model.get_activations(overlay_label(x, y, n_cls))
            neg_acts = model.get_activations(overlay_label(x, wrong_labels(y, n_cls), n_cls))

            ff_loss = torch.zeros(1, device=device)
            for pos_h, neg_h in zip(pos_acts, neg_acts, strict=True):
                pos_g = (pos_h**2).mean(dim=1)
                neg_g = (neg_h**2).mean(dim=1)
                ff_loss = ff_loss + (
                    F.softplus(-(pos_g - threshold)).mean() + F.softplus(neg_g - threshold).mean()
                )

            loss = bp_loss + ff_weight * ff_loss.squeeze()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_bp += bp_loss.item()
            tot_ff += ff_loss.item()
            n += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  AUX  ep {ep + 1:3d}/{epochs}  bp={tot_bp / n:.4f}  ff={tot_ff / n:.4f}")


def train_bp_denoise(model, loader, epochs, bp_lr, device, ff_weight=0.1, threshold=2.0, noise_std=0.3):
    """Train BP with denoising FF goodness loss (no label overlay).

    Positive = clean input activations (should produce high goodness).
    Negative = corrupted input activations (should produce low goodness).
    Training and evaluation domains match: both use non-overlaid inputs.
    Goodness learns to measure activation coherence, not overlay identity.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=bp_lr)
    bp_crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        tot_bp = tot_ff = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # BP loss on clean input
            logits = model(x)
            bp_loss = bp_crit(logits, y)

            # FF auxiliary: clean vs corrupted
            pos_acts = model.get_activations(x)
            neg_x = x + noise_std * torch.randn_like(x)
            neg_acts = model.get_activations(neg_x)

            ff_loss = torch.zeros(1, device=device)
            for pos_h, neg_h in zip(pos_acts, neg_acts, strict=True):
                pos_g = (pos_h**2).mean(dim=1)
                neg_g = (neg_h**2).mean(dim=1)
                ff_loss = ff_loss + (
                    F.softplus(-(pos_g - threshold)).mean() + F.softplus(neg_g - threshold).mean()
                )

            loss = bp_loss + ff_weight * ff_loss.squeeze()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tot_bp += bp_loss.item()
            tot_ff += ff_loss.item()
            n += 1

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  DEN  ep {ep + 1:3d}/{epochs}  bp={tot_bp / n:.4f}  ff={tot_ff / n:.4f}")


# ---------------------------------------------------------------------------
# Correlation suite
# ---------------------------------------------------------------------------

# Direction: +1 means higher score = more confident / better.
# -1 means higher score = less confident / worse.
DIRECTION = {
    "ff_goodness": 1,
    "max_softmax": 1,
    "logit_margin": 1,
    "entropy": -1,
    "nll": -1,
    "activation_norm": 1,
    "probe_confidence": 1,
    "active_ratio": -1,
    "act_entropy": -1,
    "class_similarity": 1,
    "observer_head": 1,
    "random_head": 1,
}


def partial_spearman(x, y, covariates):
    """Spearman partial correlation: rank x and y, regress out covariates, Pearson residuals."""
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates])
    rc = np.column_stack([rc, np.ones(len(rc))])  # intercept
    # OLS residuals
    coef_x, _, _, _ = np.linalg.lstsq(rc, rx, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(rc, ry, rcond=None)
    rx_resid = rx - rc @ coef_x
    ry_resid = ry - rc @ coef_y
    r, p = pearsonr(rx_resid, ry_resid)
    return float(r), float(p)


def correlation_suite(data: ObserverData):
    """Compute correlation metrics for all observers."""
    obs = data["observers"]
    losses = data["losses"]
    is_correct = data["is_correct"]

    logit_margin = obs["logit_margin"]
    act_norm = obs["activation_norm"]

    results = {"spearman_vs_loss": {}, "spearman_vs_margin": {}, "within_class": {}}

    for name, scores in obs.items():
        r_loss, p_loss = spearmanr(scores, losses)
        r_margin, p_margin = spearmanr(scores, logit_margin)
        results["spearman_vs_loss"][name] = {"rho": float(r_loss), "p": float(p_loss)}
        results["spearman_vs_margin"][name] = {"rho": float(r_margin), "p": float(p_margin)}

        # Within-class ranking: among correct examples, does observer track margin?
        if is_correct.sum() > 100:
            r_wc, p_wc = spearmanr(scores[is_correct], logit_margin[is_correct])
            results["within_class"][name] = {"rho": float(r_wc), "p": float(p_wc)}

    # Partial correlations: each observer vs loss, controlling for margin + norm.
    # The key diagnostic: does this observer carry independent signal?
    controls = [logit_margin, act_norm]
    results["partial_vs_loss"] = {}
    for name in [
        "ff_goodness",
        "active_ratio",
        "act_entropy",
        "class_similarity",
        "observer_head",
        "random_head",
    ]:
        if name in obs:
            r_part, p_part = partial_spearman(obs[name], losses, controls)
            results["partial_vs_loss"][name] = {"rho": r_part, "p": p_part}

    return results


# ---------------------------------------------------------------------------
# Intervention: dose-response curves
# ---------------------------------------------------------------------------


def eval_ablated(model, loader, device, layer_idx, neuron_indices):
    """Evaluate model accuracy with specific neurons zeroed at one layer."""
    model.eval()
    n_layers = len(model.linears)
    # Normalize layer_idx (support negative indexing)
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    ablate_set = set(neuron_indices) if len(neuron_indices) > 0 else set()

    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            h = x
            for i, linear in enumerate(model.linears):
                h = model.act(linear(model._norm(h)))
                if i == layer_idx and ablate_set:
                    h = h.clone()
                    h[:, list(ablate_set)] = 0.0
            logits = model.head(h)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def intervention_curves(
    model,
    loader,
    data: ObserverData,
    device,
    prototypes=None,
    fractions=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    n_random_trials=5,
    ranking_acts=None,
    ranking_labels=None,
):
    """Dose-response curves for multiple ablation strategies.

    Strategies:
      ff_targeted / anti_targeted: rank by mean(h²) per neuron
      magnitude: rank by mean(|h|) per neuron (identical to ff under ReLU)
      sparsity: rank by neuron activity rate (fraction of examples where |h|>eps)
      class_disc: rank by class-discriminativeness (variance of per-class means)
      random: random neuron selection, averaged over trials

    If ranking_acts/ranking_labels are provided, neuron importance is computed
    from those (e.g. training data) to avoid test-set leakage in the ranking.
    """
    acts_list = ranking_acts if ranking_acts is not None else data["per_layer_acts"]
    labels = ranking_labels if ranking_labels is not None else data["labels"]
    n_layers = len(acts_list)
    n_cls = labels.max() + 1

    strategies = ["ff_targeted", "anti_targeted", "magnitude", "sparsity", "class_disc", "random_mean"]
    results = {"fractions": list(fractions), "layers": {}}

    for layer_idx in range(n_layers):
        acts = acts_list[layer_idx]
        n_neurons = acts.size(1)

        # Per-neuron importance rankings (descending = most important first)
        ff_importance = (acts**2).mean(dim=0).numpy()
        mag_importance = acts.abs().mean(dim=0).numpy()
        # Sparsity: activity rate per neuron (higher = fires more often = more important)
        activity_rate = (acts.abs() > 0.01).float().mean(dim=0).numpy()
        # Class-discriminativeness: variance of per-class mean activation
        class_means = torch.zeros(n_cls, n_neurons)
        for c in range(n_cls):
            mask = torch.from_numpy(labels) == c
            if mask.sum() > 0:
                class_means[c] = acts[mask].mean(dim=0)
        class_disc = class_means.var(dim=0).numpy()

        rankings = {
            "ff_targeted": np.argsort(-ff_importance),
            "anti_targeted": np.argsort(ff_importance),
            "magnitude": np.argsort(-mag_importance),
            "sparsity": np.argsort(-activity_rate),
            "class_disc": np.argsort(-class_disc),
        }

        layer_result = {s: [] for s in strategies}
        layer_result["random_std"] = []

        for frac in fractions:
            n_ablate = int(n_neurons * frac)
            if n_ablate == 0:
                base_acc = eval_ablated(model, loader, device, layer_idx, [])
                for key in strategies:
                    layer_result[key].append(base_acc)
                layer_result["random_std"].append(0.0)
                continue

            for name, rank in rankings.items():
                layer_result[name].append(
                    eval_ablated(model, loader, device, layer_idx, rank[:n_ablate].tolist())
                )

            # Random trials
            rng = np.random.default_rng(42 + layer_idx * 100 + int(frac * 10))
            rand_accs = []
            for _ in range(n_random_trials):
                idx = rng.choice(n_neurons, n_ablate, replace=False).tolist()
                rand_accs.append(eval_ablated(model, loader, device, layer_idx, idx))
            layer_result["random_mean"].append(float(np.mean(rand_accs)))
            layer_result["random_std"].append(float(np.std(rand_accs)))

        results["layers"][str(layer_idx)] = layer_result

        # Rank divergence: how different is each strategy's neuron ordering from magnitude?
        mag_rank = rankings["magnitude"]
        for name in ["sparsity", "class_disc"]:
            overlap = len(set(rankings[name][:50]) & set(mag_rank[:50]))
            print(f"    layer {layer_idx} {name}: top-50 overlap with magnitude = {overlap}/50")

        print(
            f"    layer {layer_idx}: ff@90%={layer_result['ff_targeted'][-1]:.4f}  "
            f"sparsity@90%={layer_result['sparsity'][-1]:.4f}  "
            f"class_disc@90%={layer_result['class_disc'][-1]:.4f}  "
            f"random@90%={layer_result['random_mean'][-1]:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Prediction: error detection AUC
# ---------------------------------------------------------------------------


def prediction_aucs(data: ObserverData):
    """AUC for predicting correctness, per observer."""
    is_correct = data["is_correct"].astype(float)
    if is_correct.mean() in (0.0, 1.0):
        return {name: {"auc": float("nan")} for name in data["observers"]}

    results = {}
    for name, scores in data["observers"].items():
        direction = DIRECTION.get(name, 1)
        # For higher-is-better: high score should predict correct=1, use score directly.
        # For higher-is-worse: high score should predict correct=0, negate.
        oriented = scores * direction
        auc = roc_auc_score(is_correct, oriented)
        results[name] = {"auc": float(auc), "discrimination": float(max(auc, 1 - auc))}
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_intervention(all_runs, n_layers, dataset, out_path):
    """The killer figure: dose-response curves per layer, averaged across seeds."""
    n_seeds = len(all_runs)
    fracs = all_runs[0]["intervention"]["fractions"]

    ncols = min(n_layers, 4)
    nrows = math.ceil(n_layers / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), facecolor="white", squeeze=False)
    fig.suptitle(
        f"Intervention: FF-guided neuron ablation on BP model ({dataset.upper()})",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    STYLES = {
        "ff_targeted": {"color": "#2563eb", "ls": "-", "marker": "o", "label": "FF-targeted"},
        "magnitude": {"color": "#9333ea", "ls": "--", "marker": "s", "label": "Magnitude"},
        "sparsity": {"color": "#ea580c", "ls": "-", "marker": "D", "label": "Sparsity"},
        "class_disc": {"color": "#0891b2", "ls": "-", "marker": "P", "label": "Class-disc"},
        "anti_targeted": {"color": "#16a34a", "ls": "-", "marker": "^", "label": "Anti-targeted"},
        "random_mean": {"color": "#888888", "ls": "-", "marker": "", "label": "Random"},
    }

    for layer_idx in range(n_layers):
        row, col = divmod(layer_idx, ncols)
        ax = axes[row, col]
        li = str(layer_idx)

        for key, style in STYLES.items():
            vals = np.array([r["intervention"]["layers"][li][key] for r in all_runs])
            mean = vals.mean(axis=0)
            ax.plot(
                fracs,
                mean,
                color=style["color"],
                ls=style["ls"],
                marker=style["marker"] or None,
                ms=5,
                lw=2,
                label=style["label"],
                zorder=3,
            )

            if key == "random_mean" and n_seeds == 1:
                # Show within-seed std for random trials
                stds = np.array(all_runs[0]["intervention"]["layers"][li]["random_std"])
                ax.fill_between(fracs, mean - stds, mean + stds, alpha=0.15, color=style["color"], zorder=1)
            elif n_seeds > 1:
                std = vals.std(axis=0)
                ax.fill_between(fracs, mean - std, mean + std, alpha=0.1, color=style["color"], zorder=1)

        ax.set_xlabel("Fraction ablated")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Layer {layer_idx}", fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3, zorder=0)

    # Hide unused subplots
    for idx in range(n_layers, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_once(args, seed):
    """One seed: train BP, compute all observer metrics."""
    BPNet, eval_bp, get_data, _, train_bp, _ = _import_train()
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_dl, te_dl, in_dim, n_cls = get_data(args.dataset, args.batch, args.device)
    sizes = [in_dim] + [args.hidden] * args.layers

    # Train
    model = BPNet(sizes, n_cls)
    t0 = time.time()
    if args.mode == "auxiliary":
        train_bp_auxiliary(
            model, tr_dl, args.epochs, args.bp_lr, args.device, n_cls, args.ff_weight, args.threshold
        )
    elif args.mode == "denoise":
        train_bp_denoise(
            model, tr_dl, args.epochs, args.bp_lr, args.device, args.ff_weight, args.threshold, args.noise_std
        )
    else:
        train_bp(model, tr_dl, args.epochs, args.bp_lr, args.device)
    train_time = time.time() - t0
    bp_acc = eval_bp(model, te_dl, args.device)
    print(f"  BP accuracy: {bp_acc:.4f} ({train_time:.0f}s)")

    # Observers
    print("  Computing observer scores...")
    data = compute_observers(model, te_dl, args.device)

    # Probe baseline
    probe = fit_probe(model, tr_dl, args.device)
    last_acts = data["per_layer_acts"][-1].numpy()
    data["observers"]["probe_confidence"] = probe.predict_proba(last_acts).max(axis=1)

    # Class-conditional similarity (needs training prototypes)
    prototypes = compute_class_prototypes(model, tr_dl, args.device, n_cls)
    data["observers"]["class_similarity"] = class_similarity_score(
        data["per_layer_acts"], data["predictions"], prototypes
    )

    # Observer head: trained to predict loss residuals on frozen BP
    if args.mode == "observer_head":
        print("  Training observer head...")
        head = ObserverHead(args.hidden, hidden_dim=64)
        train_observer_head(model, head, tr_dl, args.device)
        head.eval()
        with torch.inference_mode():
            test_acts = data["per_layer_acts"][-1].to(args.device)
            data["observers"]["observer_head"] = head(test_acts).cpu().numpy()

        # Random-head baseline: same architecture, random weights, no training
        random_head = ObserverHead(args.hidden, hidden_dim=64).to(args.device)
        random_head.eval()
        with torch.inference_mode():
            data["observers"]["random_head"] = random_head(test_acts).cpu().numpy()

        print(
            f"    observer_head: mean={data['observers']['observer_head'].mean():.4f}, "
            f"std={data['observers']['observer_head'].std():.4f}"
        )
        print(
            f"    random_head:   mean={data['observers']['random_head'].mean():.4f}, "
            f"std={data['observers']['random_head'].std():.4f}"
        )

    # Correlation
    print("  Correlation suite...")
    corr = correlation_suite(data)
    for obs_name in ["ff_goodness", "active_ratio", "act_entropy", "class_similarity"]:
        pv = corr["partial_vs_loss"].get(obs_name, {})
        if pv:
            print(f"    partial({obs_name}, loss | margin, norm): rho={pv['rho']:+.4f}, p={pv['p']:.4f}")

    # Collect training activations for intervention neuron ranking
    tr_acts, tr_labels = collect_activations(model, tr_dl, args.device)

    # Intervention
    print("  Intervention curves...")
    intervention = intervention_curves(
        model, te_dl, data, args.device, prototypes, ranking_acts=tr_acts, ranking_labels=tr_labels
    )

    # Prediction
    print("  Prediction AUCs...")
    prediction = prediction_aucs(data)
    for name in ["ff_goodness", "max_softmax", "logit_margin"]:
        print(f"    {name}: AUC={prediction[name]['auc']:.4f}")

    # Pack results (numpy arrays to lists for JSON)
    observer_summary = {}
    for name, scores in data["observers"].items():
        observer_summary[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores))}

    return {
        "seed": seed,
        "bp_accuracy": bp_acc,
        "train_time": train_time,
        "observers": observer_summary,
        "correlation": corr,
        "intervention": intervention,
        "prediction": prediction,
    }


# ---------------------------------------------------------------------------
# Summary and stop/go gate
# ---------------------------------------------------------------------------


def print_summary(all_runs, args):
    """Print aggregate results and stop/go gate."""
    n = len(all_runs)

    bp_accs = [r["bp_accuracy"] for r in all_runs]
    print(f"\n{'=' * 72}")
    print(f"  OBSERVER FAITHFULNESS ({n} seed{'s' if n > 1 else ''}, {args.dataset.upper()})")
    print(f"{'=' * 72}")
    print(f"\n  BP accuracy: {np.mean(bp_accs):.4f} +/- {np.std(bp_accs):.4f}")

    # Observer comparison table
    obs_names = [
        "ff_goodness",
        "max_softmax",
        "logit_margin",
        "entropy",
        "nll",
        "activation_norm",
        "probe_confidence",
        "active_ratio",
        "act_entropy",
        "class_similarity",
        "observer_head",
        "random_head",
    ]
    print(f"\n  {'Observer':<20} {'rho(loss)':>10} {'AUC':>8} {'within-class':>14}")
    print(f"  {'-' * 54}")

    for name in obs_names:
        rhos = [r["correlation"]["spearman_vs_loss"].get(name, {}).get("rho", float("nan")) for r in all_runs]
        aucs = [r["prediction"].get(name, {}).get("auc", float("nan")) for r in all_runs]
        wcs = [r["correlation"]["within_class"].get(name, {}).get("rho", float("nan")) for r in all_runs]
        print(f"  {name:<20} {np.nanmean(rhos):>+10.4f} {np.nanmean(aucs):>8.4f} {np.nanmean(wcs):>+14.4f}")

    # Partial correlations (all structural observers)
    print("\n  Partial correlations (vs loss | margin, norm):")
    structural_obs = [
        "ff_goodness",
        "active_ratio",
        "act_entropy",
        "class_similarity",
        "observer_head",
        "random_head",
    ]
    for obs_name in structural_obs:
        rhos = [
            r["correlation"]["partial_vs_loss"].get(obs_name, {}).get("rho", float("nan")) for r in all_runs
        ]
        ps = [r["correlation"]["partial_vs_loss"].get(obs_name, {}).get("p", float("nan")) for r in all_runs]
        print(
            f"    {obs_name:<20} rho={np.nanmean(rhos):+.4f} +/- {np.nanstd(rhos):.4f}, "
            f"p={np.nanmean(ps):.4f}"
        )

    partials = [r["correlation"]["partial_vs_loss"].get("ff_goodness", {}).get("rho", -1) for r in all_runs]

    # Intervention summary: compare all strategies at last layer, 90% ablation
    last_layer = str(args.layers - 1)
    print(f"\n  Intervention at 90% ablation (last layer, {n} seeds):")
    for key in ["ff_targeted", "sparsity", "class_disc", "magnitude", "random_mean", "anti_targeted"]:
        vals = [r["intervention"]["layers"][last_layer].get(key, [0])[-1] for r in all_runs]
        label = key.replace("_mean", "").replace("_", "-")
        print(f"    {label:<16} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Stop/go gate
    targ_90 = [r["intervention"]["layers"][last_layer]["ff_targeted"][-1] for r in all_runs]
    rand_90 = [r["intervention"]["layers"][last_layer]["random_mean"][-1] for r in all_runs]
    targeted_faster = np.mean(targ_90) < np.mean(rand_90)
    partial_positive = np.mean(partials) > 0

    print(f"\n  {'=' * 50}")
    print("  STOP/GO GATE")
    print(f"  {'=' * 50}")
    gate1 = "PASS" if targeted_faster else "FAIL"
    gate2 = "PASS" if partial_positive else "FAIL"
    print(f"  [{gate1}] FF-targeted ablation degrades faster than random")
    print(f"  [{gate2}] Partial correlation of ff_goodness > 0")
    verdict = "GO" if (targeted_faster and partial_positive) else "STOP"
    print(f"  Verdict: {verdict}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    P.add_argument("--epochs", type=int, default=50)
    P.add_argument("--hidden", type=int, default=500)
    P.add_argument("--layers", type=int, default=4)
    P.add_argument("--batch", type=int, default=512)
    P.add_argument("--bp-lr", type=float, default=0.001)
    P.add_argument("--device", default="auto")
    P.add_argument("--seeds", type=int, default=3)
    P.add_argument(
        "--mode",
        default="pure_observer",
        choices=["pure_observer", "auxiliary", "denoise", "observer_head"],
        help="Wiring mode",
    )
    P.add_argument("--ff-weight", type=float, default=0.1, help="Weight for FF auxiliary loss")
    P.add_argument("--threshold", type=float, default=2.0, help="FF goodness threshold")
    P.add_argument("--noise-std", type=float, default=0.3, help="Corruption noise std (denoise mode only)")
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    sizes = [{"mnist": 784, "cifar10": 3072}[a.dataset]] + [a.hidden] * a.layers
    print(f"Observer faithfulness ({a.mode})")
    print(f"Dataset: {a.dataset}  Arch: {' -> '.join(map(str, sizes))}")
    print(f"Epochs: {a.epochs}  Seeds: {a.seeds}  Device: {a.device}\n")

    all_runs = []
    for i, seed in enumerate(range(42, 42 + a.seeds)):
        print(f"\n--- Seed {seed} ({i + 1}/{a.seeds}) ---")
        result = run_once(a, seed)
        all_runs.append(result)

    # Save
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    output = dict(config=vars(a), runs=all_runs)
    suffix = f"_{a.mode}" if a.mode != "pure_observer" else ""
    out_file = out / f"observe_{a.dataset}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved {out_file}")

    # Plot
    chart_path = (
        Path(__file__).resolve().parent.parent / "assets" / f"observe_{a.dataset}_intervention{suffix}.png"
    )
    plot_intervention(all_runs, a.layers, a.dataset, str(chart_path))

    # Summary
    print_summary(all_runs, a)


if __name__ == "__main__":
    main()
