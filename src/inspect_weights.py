"""Inspect linear binary observer head weight vectors across seeds."""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from observe import compute_loss_residuals
from train import BPNet, get_data, train_bp


def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    tr_dl, te_dl, in_dim, n_cls = get_data("mnist", 512, device)
    sizes = [in_dim] + [500] * 4
    seeds = [42, 43, 44]

    weights = []
    all_scores = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = BPNet(sizes, n_cls)
        train_bp(model, tr_dl, 50, 0.001, device)
        model.eval()

        # Collect training data
        all_acts, all_logits, all_losses = [], [], []
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        with torch.inference_mode():
            for x, y in tr_dl:
                x, y = x.to(device), y.to(device)
                acts = model.get_activations(x)
                logits = model.head(acts[-1])
                losses = criterion(logits, y)
                all_acts.append(acts[-1].cpu())
                all_logits.append(logits.cpu())
                all_losses.append(losses.cpu())

        tr_acts = torch.cat(all_acts)
        tr_logits = torch.cat(all_logits)
        tr_losses = torch.cat(all_losses).numpy()
        sorted_l = tr_logits.sort(dim=1, descending=True).values
        tr_margins = (sorted_l[:, 0] - sorted_l[:, 1]).numpy()
        tr_norms = tr_acts.norm(dim=1).numpy()
        residuals = compute_loss_residuals(tr_losses, tr_margins, tr_norms)
        targets = (residuals > 0).astype(np.float32)

        # Train linear binary head
        head = torch.nn.Linear(500, 1).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        dataset = torch.utils.data.TensorDataset(tr_acts, torch.from_numpy(targets))
        dl = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
        for _ep in range(20):
            for bx, by in dl:
                bx, by = bx.to(device), by.to(device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        w = head.weight.data.cpu().squeeze().numpy()
        weights.append(w)

        # Test scores
        head.eval()
        te_scores = []
        with torch.inference_mode():
            for x, _y in te_dl:
                x = x.to(device)
                acts = model.get_activations(x)
                te_scores.append(head(acts[-1]).squeeze(-1).cpu().numpy())
        all_scores.append(np.concatenate(te_scores))

        sorted_abs = np.sort(np.abs(w))[::-1]
        cumsum = np.cumsum(sorted_abs) / sorted_abs.sum()
        n_90 = int(np.searchsorted(cumsum, 0.9)) + 1
        print(
            f"Seed {seed}: norm={np.linalg.norm(w):.4f}, "
            f"{n_90}/500 neurons carry 90% of |weight|, "
            f"top-10: {np.argsort(-np.abs(w))[:10].tolist()}"
        )

    # Q1: Weight vector agreement
    print(f"\n{'=' * 50}")
    print("  WEIGHT VECTOR COSINE SIMILARITY")
    print(f"{'=' * 50}")
    for i in range(3):
        for j in range(i + 1, 3):
            cos = np.dot(weights[i], weights[j]) / (np.linalg.norm(weights[i]) * np.linalg.norm(weights[j]))
            print(f"  Seed {seeds[i]} vs {seeds[j]}: {cos:+.4f}")

    # Q2: Top neuron overlap
    print(f"\n{'=' * 50}")
    print("  TOP NEURON OVERLAP")
    print(f"{'=' * 50}")
    for k in [20, 50]:
        top_sets = [set(np.argsort(-np.abs(w))[:k]) for w in weights]
        for i in range(3):
            for j in range(i + 1, 3):
                overlap = len(top_sets[i] & top_sets[j])
                print(f"  Top-{k}: seed {seeds[i]} vs {seeds[j]}: {overlap}/{k} shared")

    # Q3: Score ranking agreement
    print(f"\n{'=' * 50}")
    print("  SCORE RANKING AGREEMENT")
    print(f"{'=' * 50}")
    for i in range(3):
        for j in range(i + 1, 3):
            rho, _ = spearmanr(all_scores[i], all_scores[j])
            print(f"  Seed {seeds[i]} vs {seeds[j]}: {rho:+.4f}")

    # Q4: Projection onto simple directions
    print(f"\n{'=' * 50}")
    print("  PROJECTION ONTO SIMPLE DIRECTIONS")
    print(f"{'=' * 50}")
    uniform = np.ones(500)
    uniform /= np.linalg.norm(uniform)
    for si, seed in enumerate(seeds):
        w_unit = weights[si] / (np.linalg.norm(weights[si]) + 1e-8)
        print(f"  Seed {seed}: cos(w, uniform) = {np.dot(w_unit, uniform):+.4f}")


if __name__ == "__main__":
    main()
