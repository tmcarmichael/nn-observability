"""
Seed-agreement test for observer heads.

Trains 3 BP models (different seeds), fits an observer head on each,
then checks whether the heads agree on which test examples are high-risk.
High rank correlation = one stable signal. Low = per-seed artifacts.

Usage: uv run src/seed_agreement.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from observe import (
    ObserverHead,
    train_observer_head,
)
from train import BPNet, eval_bp, get_data, train_bp


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "mnist"
    epochs = 50
    hidden = 500
    layers = 4
    batch = 512
    seeds = [42, 43, 44]

    tr_dl, te_dl, in_dim, n_cls = get_data(dataset, batch, device)
    sizes = [in_dim] + [hidden] * layers

    all_scores = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train BP model
        model = BPNet(sizes, n_cls)
        train_bp(model, tr_dl, epochs, 0.001, device)
        acc = eval_bp(model, te_dl, device)
        print(f"  BP accuracy: {acc:.4f}")

        # Train observer head
        head = ObserverHead(hidden, hidden_dim=64)
        train_observer_head(model, head, tr_dl, device)

        # Get per-example scores on test set
        head.eval()
        model.eval()
        scores = []
        with torch.inference_mode():
            for x, _y in te_dl:
                x = x.to(device)
                acts = model.get_activations(x)
                s = head(acts[-1])
                scores.append(s.cpu().numpy())
        scores = np.concatenate(scores)
        all_scores.append(scores)
        print(f"  Observer head scores: mean={scores.mean():.4f}, std={scores.std():.4f}")

    # Seed-agreement: pairwise rank correlations
    print(f"\n{'=' * 50}")
    print("  SEED AGREEMENT (pairwise Spearman rank correlation)")
    print(f"{'=' * 50}")

    n_seeds = len(seeds)
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            rho, p = spearmanr(all_scores[i], all_scores[j])
            print(f"  Seed {seeds[i]} vs {seeds[j]}: rho = {rho:+.4f}, p = {p:.4f}")

    # Mean pairwise
    rhos = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            rho, _ = spearmanr(all_scores[i], all_scores[j])
            rhos.append(rho)
    mean_rho = np.mean(rhos)

    print(f"\n  Mean pairwise rank correlation: {mean_rho:+.4f}")
    if mean_rho > 0.5:
        print("  --> STABLE: heads agree on which examples are high-risk")
    elif mean_rho > 0.2:
        print("  --> MODERATE: some shared signal, significant per-seed variation")
    else:
        print("  --> UNSTABLE: each seed found different structure")


if __name__ == "__main__":
    main()
