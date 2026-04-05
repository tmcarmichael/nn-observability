"""
Observer head variant sweep with seed-agreement built in.

Three variants tested in parallel:
  1. Linear head (no hidden layer): tests whether the informative direction
     is a linear combination that Phase 3's hand-designed readouts missed.
  2. Bottleneck (500→4→1): forces convergence through a low-dimensional
     subspace. Tests whether seeds can agree through a narrow channel.
  3. Binary target (predict correct/incorrect residual sign instead of
     continuous loss residuals): tests whether regression is too unconstrained.

Each variant: train 3 BP models (seeds 42-44), fit observer head on frozen
activations, compute partial correlation AND pairwise seed agreement.

Usage: uv run src/observer_variants.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from observe import (
    ObserverHead,
    compute_loss_residuals,
    partial_spearman,
)
from train import BPNet, get_data, train_bp

# ---------------------------------------------------------------------------
# Head architectures
# ---------------------------------------------------------------------------


class LinearHead(nn.Module):
    """Single linear projection: activations → scalar."""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class BottleneckHead(nn.Module):
    """500 → bottleneck_dim → 1. Forces low-dimensional encoding."""

    def __init__(self, input_dim, bottleneck_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def collect_head_data(model, loader, device):
    """Collect activations, losses, margins, norms from frozen model."""
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

    acts = torch.cat(all_acts)
    logits = torch.cat(all_logits)
    losses = torch.cat(all_losses).numpy()

    sorted_l = logits.sort(dim=1, descending=True).values
    margins = (sorted_l[:, 0] - sorted_l[:, 1]).numpy()
    norms = acts.norm(dim=1).numpy()

    return acts, losses, margins, norms


def train_head_regression(head, acts, residuals, device, epochs=20, lr=1e-3):
    """Train head to predict continuous loss residuals."""
    head.to(device)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    targets = torch.from_numpy(residuals).float()
    dataset = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    for _ep in range(epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            loss = F.mse_loss(head(bx), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def train_head_binary(head, acts, residuals, device, epochs=20, lr=1e-3):
    """Train head to predict sign of loss residual (correct/incorrect relative to confidence)."""
    head.to(device)
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    # Binary target: 1 if residual > 0 (worse than confidence predicts), 0 otherwise
    targets = torch.from_numpy((residuals > 0).astype(np.float32))
    dataset = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

    for _ep in range(epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            loss = F.binary_cross_entropy_with_logits(head(bx), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_head(head, test_acts, test_losses, test_margins, test_norms, device):
    """Compute partial correlation of head output vs loss, controlling for margin+norm."""
    head.eval()
    with torch.inference_mode():
        scores = head(test_acts.to(device)).cpu().numpy()
    rho, p = partial_spearman(scores, test_losses, [test_margins, test_norms])
    return scores, rho, p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_variant(
    name,
    make_head,
    train_fn,
    device,
    seeds,
    tr_dl,
    te_dl,
    in_dim,
    n_cls,
    hidden,
    layers,
    epochs_bp,
    epochs_head,
):
    """Run one variant across all seeds, return per-seed scores and partial correlations."""
    sizes = [in_dim] + [hidden] * layers
    all_scores = []
    all_rhos = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train BP
        model = BPNet(sizes, n_cls)
        train_bp(model, tr_dl, epochs_bp, 0.001, device)

        # Collect data
        tr_acts, tr_losses, tr_margins, tr_norms = collect_head_data(model, tr_dl, device)
        te_acts, te_losses, te_margins, te_norms = collect_head_data(model, te_dl, device)

        residuals = compute_loss_residuals(tr_losses, tr_margins, tr_norms)

        # Train head
        head = make_head()
        train_fn(head, tr_acts, residuals, device, epochs=epochs_head)

        # Evaluate
        scores, rho, p = eval_head(head, te_acts, te_losses, te_margins, te_norms, device)
        all_scores.append(scores)
        all_rhos.append(rho)
        print(f"    seed {seed}: partial_corr={rho:+.4f} (p={p:.4f})")

    # Seed agreement
    pairwise = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            r, _ = spearmanr(all_scores[i], all_scores[j])
            pairwise.append(r)
    mean_agree = np.mean(pairwise)

    print(f"    mean partial_corr: {np.mean(all_rhos):+.4f} +/- {np.std(all_rhos):.4f}")
    print(f"    seed agreement:    {mean_agree:+.4f}")
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            r, _ = spearmanr(all_scores[i], all_scores[j])
            print(f"      {seeds[i]} vs {seeds[j]}: {r:+.4f}")

    return all_rhos, pairwise


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [42, 43, 44]
    hidden = 500
    layers = 4
    epochs_bp = 50
    epochs_head = 20

    tr_dl, te_dl, in_dim, n_cls = get_data("mnist", 512, device)

    variants = {
        "linear_regression": {
            "make_head": lambda: LinearHead(hidden).to(device),
            "train_fn": train_head_regression,
        },
        "bottleneck_regression": {
            "make_head": lambda: BottleneckHead(hidden, bottleneck_dim=4).to(device),
            "train_fn": train_head_regression,
        },
        "mlp_binary": {
            "make_head": lambda: ObserverHead(hidden, hidden_dim=64).to(device),
            "train_fn": train_head_binary,
        },
        "linear_binary": {
            "make_head": lambda: LinearHead(hidden).to(device),
            "train_fn": train_head_binary,
        },
    }

    print("Observer head variant sweep")
    print(f"Device: {device}  Seeds: {seeds}  BP epochs: {epochs_bp}  Head epochs: {epochs_head}\n")

    results = {}
    for name, cfg in variants.items():
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")
        rhos, agreements = run_variant(
            name,
            cfg["make_head"],
            cfg["train_fn"],
            device,
            seeds,
            tr_dl,
            te_dl,
            in_dim,
            n_cls,
            hidden,
            layers,
            epochs_bp,
            epochs_head,
        )
        results[name] = {"rhos": rhos, "agreements": agreements}

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Variant':<25} {'Partial corr':>14} {'Seed agree':>12}")
    print(f"  {'-' * 53}")
    for name, r in results.items():
        pc = f"{np.mean(r['rhos']):+.4f}+/-{np.std(r['rhos']):.3f}"
        sa = f"{np.mean(r['agreements']):+.4f}"
        print(f"  {name:<25} {pc:>14} {sa:>12}")

    # Reference: original MLP regression (from prior run)
    print("\n  Reference (prior run):")
    print(f"  {'mlp_regression':<25} {'~+0.139':>14} {'-0.061':>12}")

    # Diagnostic
    print("\n  Interpretation:")
    lr = results.get("linear_regression", {})
    bn = results.get("bottleneck_regression", {})
    mb = results.get("mlp_binary", {})
    if lr and np.mean(lr["rhos"]) > 0.05:
        print("  - Linear head finds signal: Phase 3 tested wrong linear directions")
    if bn and np.mean(bn["agreements"]) > 0.3:
        print("  - Bottleneck converges: shared signal in low-dimensional subspace")
    if mb and np.mean(mb["agreements"]) > 0.3:
        print("  - Binary target converges: regression framing was too unconstrained")
    if all(np.mean(r["agreements"]) < 0.1 for r in results.values()):
        print("  - All variants unstable: learned component is irreducibly fragmented")


if __name__ == "__main__":
    main()
