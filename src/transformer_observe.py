"""
Phase 5: Transformer transfer.

Test whether Phase 4's learned observer heads transfer to transformers.
Freeze pretrained GPT-2 124M, collect per-position residual stream
activations, train linear binary observer heads, measure partial
correlation and seed agreement.

Experiments:
  5a: Direct replication at last layer (primary result)
  5b: Layer sweep across all 12 layers
  5c: Hand-designed baseline comparison
  5d: Neuron ablation intervention

Usage:
    uv run --extra transformer src/transformer_observe.py
    uv run --extra transformer src/transformer_observe.py --layer-sweep
    uv run --extra transformer src/transformer_observe.py --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from observe import compute_loss_residuals, partial_spearman

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def load_wikitext(split="test", max_docs=None):
    """Load WikiText-103 documents."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    docs = []
    current = []
    for row in ds:
        text = row["text"]
        if text.strip() == "" and current:
            docs.append("\n".join(current))
            current = []
            if max_docs and len(docs) >= max_docs:
                break
        elif text.strip():
            current.append(text)
    if current:
        docs.append("\n".join(current))
    return docs


def collect_layer_data(model, tokenizer, docs, layer, device, max_tokens=200000, max_length=512):
    """Collect per-position data at one layer.

    Returns activations, per-position loss, max softmax, and activation norm.
    Train/test split is by document (caller provides the right split).
    """
    model.eval()
    all_acts, all_losses, all_softmax, all_norms = [], [], [], []
    total_tokens = 0

    for doc in docs:
        if total_tokens >= max_tokens:
            break
        if not doc.strip():
            continue

        tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tokens["input_ids"].to(device)

        if input_ids.size(1) < 2:
            continue

        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)

        # Hidden state at target layer (layer+1 because index 0 is embedding)
        h = outputs.hidden_states[layer + 1][0, :-1, :].cpu()
        # Per-position loss
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        losses = F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu()
        # Confidence
        probs = F.softmax(shift_logits, dim=-1)
        max_sm = probs.max(dim=-1).values.cpu()
        norms = h.norm(dim=-1)

        all_acts.append(h)
        all_losses.append(losses)
        all_softmax.append(max_sm)
        all_norms.append(norms)
        total_tokens += h.size(0)

    print(f"    collected {total_tokens} positions from {len(all_acts)} documents")
    return {
        "activations": torch.cat(all_acts),
        "losses": torch.cat(all_losses).numpy(),
        "max_softmax": torch.cat(all_softmax).numpy(),
        "activation_norm": torch.cat(all_norms).numpy(),
    }


# ---------------------------------------------------------------------------
# Observer heads
# ---------------------------------------------------------------------------


def train_linear_binary(train_data, seed=42, epochs=20, lr=1e-3):
    """Train linear binary observer head on loss residuals."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    acts = train_data["activations"]
    residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    targets = torch.from_numpy((residuals > 0).astype(np.float32))

    n_features = acts.size(1)
    head = torch.nn.Linear(n_features, 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    head.train()
    for _ep in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return head


def evaluate_head(head, test_data):
    """Partial correlation of head output vs loss, controlling for confidence."""
    head.eval()
    with torch.inference_mode():
        scores = head(test_data["activations"]).squeeze(-1).numpy()
    rho, p = partial_spearman(
        scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
    )
    return scores, rho, p


# ---------------------------------------------------------------------------
# Hand-designed baselines (Phase 3 replication)
# ---------------------------------------------------------------------------


def compute_hand_designed(data):
    """Compute Phase 3 statistics on transformer activations."""
    acts = data["activations"]
    return {
        "ff_goodness": (acts**2).mean(dim=1).numpy(),
        "active_ratio": (acts.abs() > 0.01).float().mean(dim=1).numpy(),
        "act_entropy": _activation_entropy(acts),
        "activation_norm": data["activation_norm"],
    }


def _activation_entropy(acts):
    p = acts.abs() / (acts.abs().sum(dim=1, keepdim=True) + 1e-8)
    return -(p * (p + 1e-8).log()).sum(dim=1).numpy()


# ---------------------------------------------------------------------------
# Main experiments
# ---------------------------------------------------------------------------


def run_5a(model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test):
    """5a: Direct replication at last layer."""
    layer = 11  # last transformer block
    print(f"\n{'=' * 60}")
    print(f"  5a: Linear binary observer head at layer {layer}")
    print(f"{'=' * 60}")

    print("  Collecting train activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
    print("  Collecting test activations...")
    test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)

    all_scores = []
    all_rhos = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        head = train_linear_binary(train_data, seed=seed)
        scores, rho, p = evaluate_head(head, test_data)
        all_scores.append(scores)
        all_rhos.append(rho)
        print(f"    partial corr = {rho:+.4f} (p={p:.4f})")

    # Seed agreement
    pairwise = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            r, _ = spearmanr(all_scores[i], all_scores[j])
            pairwise.append(r)
            print(f"    seed {seeds[i]} vs {seeds[j]}: rank corr = {r:+.4f}")

    mean_rho = np.mean(all_rhos)
    mean_agree = np.mean(pairwise) if pairwise else 0.0

    print("\n  5a RESULT:")
    print(f"    mean partial corr: {mean_rho:+.4f} +/- {np.std(all_rhos):.4f}")
    print(f"    mean seed agreement: {mean_agree:+.4f}")

    if mean_rho > 0.15 and mean_agree > 0.30:
        print("    --> STRONG POSITIVE: MLP result transfers")
    elif mean_rho > 0.05 and mean_agree > 0.15:
        print("    --> WEAK POSITIVE: signal exists but weaker")
    else:
        print("    --> NEGATIVE: signal does not transfer")

    return {
        "layer": layer,
        "partial_corrs": all_rhos,
        "seed_agreement": pairwise,
        "mean_partial_corr": float(mean_rho),
        "mean_seed_agreement": float(mean_agree),
    }


def run_5b(model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test):
    """5b: Layer sweep across all 12 layers."""
    n_layers = 12
    print(f"\n{'=' * 60}")
    print(f"  5b: Layer sweep (0-{n_layers - 1})")
    print(f"{'=' * 60}")

    layer_results = {}

    for layer in range(n_layers):
        print(f"\n  Layer {layer}:")
        train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
        test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)

        rhos = []
        for seed in seeds:
            head = train_linear_binary(train_data, seed=seed)
            _, rho, p = evaluate_head(head, test_data)
            rhos.append(rho)

        mean_rho = np.mean(rhos)
        print(f"    mean partial corr: {mean_rho:+.4f} +/- {np.std(rhos):.4f}")
        layer_results[layer] = {"mean": float(mean_rho), "std": float(np.std(rhos)), "per_seed": rhos}

    # Summary
    print("\n  5b LAYER PROFILE:")
    print(f"  {'Layer':>6} {'Partial corr':>14}")
    print(f"  {'-' * 22}")
    for layer in range(n_layers):
        r = layer_results[layer]
        print(f"  {layer:>6} {r['mean']:>+14.4f}")

    peak_layer = max(range(n_layers), key=lambda l: layer_results[l]["mean"])
    print(f"\n  Peak at layer {peak_layer}: {layer_results[peak_layer]['mean']:+.4f}")

    return layer_results


def run_5c(model, tokenizer, device, test_docs, max_tokens_test, layer=11):
    """5c: Hand-designed baselines at target layer."""
    print(f"\n{'=' * 60}")
    print(f"  5c: Hand-designed baselines at layer {layer}")
    print(f"{'=' * 60}")

    test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)
    observers = compute_hand_designed(test_data)

    results = {}
    for name, scores in observers.items():
        rho, p = partial_spearman(
            scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
        )
        results[name] = {"rho": float(rho), "p": float(p)}
        print(f"    {name:<20} partial corr = {rho:+.4f} (p={p:.4f})")

    return results


# ---------------------------------------------------------------------------
# Intervention (5d)
# ---------------------------------------------------------------------------


def run_5e(model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test):
    """5e: Full-output control.

    Train an MLP on layer 11 activations to predict loss (strongest possible
    output-derived baseline). Then train the observer at layer 8 and partial
    out the layer-11 predictor. If partial correlation survives, layer 8
    contains decision-quality information that the output never reveals.
    """
    print(f"\n{'=' * 60}")
    print("  5e: Full-output control (layer 8 vs layer 11 predictor)")
    print(f"{'=' * 60}")

    # Collect activations at both layers
    print("  Collecting layer 11 train activations...")
    train_11 = collect_layer_data(model, tokenizer, train_docs, 11, device, max_tokens_train)
    print("  Collecting layer 11 test activations...")
    test_11 = collect_layer_data(model, tokenizer, test_docs, 11, device, max_tokens_test)
    print("  Collecting layer 8 train activations...")
    train_8 = collect_layer_data(model, tokenizer, train_docs, 8, device, max_tokens_train)
    print("  Collecting layer 8 test activations...")
    test_8 = collect_layer_data(model, tokenizer, test_docs, 8, device, max_tokens_test)

    all_rhos_controlled = []
    all_rhos_standard = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train layer 11 loss predictor (MLP, strongest output baseline)
        acts_11 = train_11["activations"]
        losses_tr = train_11["losses"]
        targets_11 = torch.from_numpy(losses_tr).float()
        n_feat = acts_11.size(1)

        predictor = torch.nn.Sequential(
            torch.nn.Linear(n_feat, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )
        opt = torch.optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
        dataset = torch.utils.data.TensorDataset(acts_11, targets_11)
        dl = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
        for _ep in range(20):
            for bx, by in dl:
                loss = F.mse_loss(predictor(bx).squeeze(-1), by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        # Get layer 11 predictor scores on test set
        predictor.eval()
        with torch.inference_mode():
            pred_11_scores = predictor(test_11["activations"]).squeeze(-1).numpy()

        # Train layer 8 observer head (linear binary, same as 5a)
        head_8 = train_linear_binary(train_8, seed=seed)
        head_8.eval()
        with torch.inference_mode():
            obs_8_scores = head_8(test_8["activations"]).squeeze(-1).numpy()

        # Standard partial correlation (controlling for softmax + norm)
        rho_std, p_std = partial_spearman(
            obs_8_scores, test_8["losses"], [test_8["max_softmax"], test_8["activation_norm"]]
        )

        # Full-output control: partial out layer 11 predictor too
        rho_ctrl, p_ctrl = partial_spearman(
            obs_8_scores,
            test_8["losses"],
            [test_8["max_softmax"], test_8["activation_norm"], pred_11_scores],
        )

        all_rhos_standard.append(rho_std)
        all_rhos_controlled.append(rho_ctrl)
        print(f"\n  Seed {seed}:")
        print(f"    standard controls:    partial corr = {rho_std:+.4f}")
        print(f"    + layer 11 predictor: partial corr = {rho_ctrl:+.4f}")

    mean_std = np.mean(all_rhos_standard)
    mean_ctrl = np.mean(all_rhos_controlled)
    print("\n  5e RESULT:")
    print(f"    layer 8 (standard controls): {mean_std:+.4f} +/- {np.std(all_rhos_standard):.4f}")
    print(f"    layer 8 (+ layer 11 output): {mean_ctrl:+.4f} +/- {np.std(all_rhos_controlled):.4f}")

    if mean_ctrl > 0.05:
        print("    --> Layer 8 contains information the output does not carry")
    elif mean_ctrl > 0.01:
        print("    --> Small residual signal survives output control")
    else:
        print("    --> Signal collapses: layer 8 reads early output information")

    return {
        "standard": {"mean": float(mean_std), "per_seed": all_rhos_standard},
        "output_controlled": {"mean": float(mean_ctrl), "per_seed": all_rhos_controlled},
    }


def _eval_ablated_transformer(model, tokenizer, block, ablate_indices, docs, device, max_length=512):
    """Evaluate model loss with specific MLP intermediate neurons zeroed."""

    def make_hook(indices):
        idx_list = list(indices)

        def hook(module, input, output):
            output[:, :, idx_list] = 0.0
            return output

        return hook

    handle = block.mlp.act.register_forward_hook(make_hook(ablate_indices))
    losses = []
    with torch.inference_mode():
        for doc in docs:
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.logits[0, :-1, :], input_ids[0, 1:])
            losses.append(loss.item())
    handle.remove()
    return float(np.mean(losses))


def run_5d(model, tokenizer, device, train_docs, test_docs, max_tokens_train, max_tokens_test, layer=8):
    """5d: Neuron ablation at target layer, observer-guided vs random.

    Ranks MLP intermediate neurons by their contribution to the observer
    head's learned direction. Ablates top-ranked neurons via forward hooks
    and measures loss increase versus random ablation.
    """
    print(f"\n{'=' * 60}")
    print(f"  5d: Intervention at layer {layer}")
    print(f"{'=' * 60}")

    # Train observer head at target layer
    print("  Collecting train activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
    head = train_linear_binary(train_data, seed=42)
    obs_weight = head.weight.data.cpu().squeeze().numpy()  # [768]

    # Observer-guided neuron ranking:
    # MLP output = W_out @ GELU(W_in @ x + b_in) + b_out
    # Contribution of intermediate neuron j to observer direction ≈ obs_weight @ W_out[:, j]
    block = model.transformer.h[layer]
    W_out = block.mlp.c_proj.weight.data.cpu().numpy()  # [768, 3072] (GPT-2 uses Conv1D)
    # GPT-2 Conv1D stores weights as [in, out], so W_out is [3072, 768]
    # The projection is: output = input @ W_out, so column j of W_out is the direction neuron j writes
    # Observer contribution of neuron j = obs_weight @ W_out[j, :] ... need to check GPT-2 weight layout

    # GPT-2 Conv1D: weight shape is [n_in, n_out], forward is x @ weight + bias
    # c_proj: n_in=3072 (intermediate), n_out=768 (residual)
    # So W_out[j, :] is the 768-dim direction that intermediate neuron j writes to residual stream
    # Observer contribution = obs_weight . W_out[j, :] for each j
    n_intermediate = W_out.shape[0]  # 3072
    obs_contribution = np.array([np.dot(obs_weight, W_out[j, :]) for j in range(n_intermediate)])

    obs_rank_desc = np.argsort(-np.abs(obs_contribution))  # most important first
    mag_rank_desc = None  # compute from activations below

    # Collect baseline loss on test data
    print("  Collecting baseline test loss...")
    baseline_losses = []
    with torch.inference_mode():
        for doc in test_docs[:200]:
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids)
            loss = F.cross_entropy(outputs.logits[0, :-1, :], input_ids[0, 1:])
            baseline_losses.append(loss.item())
    baseline_loss = np.mean(baseline_losses)
    print(f"    baseline loss: {baseline_loss:.4f}")

    # Also collect intermediate activations to get magnitude ranking
    print("  Collecting MLP intermediate activations for magnitude ranking...")
    intermediates = []

    def capture_hook(module, input, output):
        intermediates.append(output.detach().cpu())

    # Hook into the GELU activation output (after c_fc, before c_proj)
    hook_handle = block.mlp.act.register_forward_hook(capture_hook)
    with torch.inference_mode():
        for doc in test_docs[:100]:
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            model(input_ids)
    hook_handle.remove()

    all_inter = torch.cat(intermediates, dim=1).squeeze(0)  # [total_positions, 3072]
    mag_importance = all_inter.abs().mean(dim=0).numpy()
    mag_rank_desc = np.argsort(-mag_importance)
    intermediates.clear()

    # Ablation sweep
    fractions = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    strategies = {
        "observer": obs_rank_desc,
        "magnitude": mag_rank_desc,
    }

    results = {"fractions": fractions, "baseline_loss": float(baseline_loss)}

    eval_docs = test_docs[:200]

    for name, ranking in strategies.items():
        results[name] = []
        for frac in fractions:
            n_ablate = int(n_intermediate * frac)
            if n_ablate == 0:
                results[name].append(float(baseline_loss))
                continue
            indices = set(ranking[:n_ablate].tolist())
            results[name].append(
                _eval_ablated_transformer(model, tokenizer, block, indices, eval_docs, device)
            )

    # Random ablation
    results["random_mean"] = []
    results["random_std"] = []
    n_random = 3
    for frac in fractions:
        n_ablate = int(n_intermediate * frac)
        if n_ablate == 0:
            results["random_mean"].append(float(baseline_loss))
            results["random_std"].append(0.0)
            continue
        rng = np.random.default_rng(42)
        rand_losses = []
        for _ in range(n_random):
            indices = set(rng.choice(n_intermediate, n_ablate, replace=False).tolist())
            rand_losses.append(_eval_ablated_transformer(model, tokenizer, block, indices, eval_docs, device))
        results["random_mean"].append(float(np.mean(rand_losses)))
        results["random_std"].append(float(np.std(rand_losses)))

    # Summary
    print(f"\n  5d INTERVENTION (layer {layer}, MLP intermediate neurons):")
    print(f"  {'Fraction':<10} {'Observer':>10} {'Magnitude':>10} {'Random':>10}")
    print(f"  {'-' * 42}")
    for i, frac in enumerate(fractions):
        obs_v = results["observer"][i]
        mag_v = results["magnitude"][i]
        rnd_v = results["random_mean"][i]
        print(f"  {frac:<10.2f} {obs_v:>10.4f} {mag_v:>10.4f} {rnd_v:>10.4f}")

    obs_50 = results["observer"][fractions.index(0.5)]
    rnd_50 = results["random_mean"][fractions.index(0.5)]
    if obs_50 > rnd_50 * 1.05:
        print("\n  --> Observer-guided ablation is more destructive than random")
    else:
        print("\n  --> Observer-guided ablation is NOT more destructive than random")

    return results


# ---------------------------------------------------------------------------
# Phase 6a: early flagging
# ---------------------------------------------------------------------------


def run_6a(model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test):
    """6a: Mid-layer flagging vs output-confidence flagging.

    Train observer at layer 8. On test data, flag the top-k% of tokens by
    observer score (predicted high residual error). Compare against flagging
    by low max-softmax (output confidence). Measure:
      - Precision: fraction of flagged tokens that actually have above-median loss
      - Exclusive catches: errors the observer flags that confidence does not
    """
    print(f"\n{'=' * 60}")
    print("  6a: Early flagging (layer 8 observer vs output confidence)")
    print(f"{'=' * 60}")

    # Collect data
    print("  Collecting layer 8 train activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, 8, device, max_tokens_train)
    print("  Collecting layer 8 test activations...")
    test_data = collect_layer_data(model, tokenizer, test_docs, 8, device, max_tokens_test)

    # Also need layer 11 test data for the output-confidence baseline
    # (max_softmax from test_data is computed at layer 8's logit position,
    # but we want the model's actual output confidence)
    print("  Collecting layer 11 test activations (for output confidence)...")
    test_11 = collect_layer_data(model, tokenizer, test_docs, 11, device, max_tokens_test)

    # Align: both should have the same number of positions from the same docs
    n_test = min(len(test_data["losses"]), len(test_11["losses"]))
    test_losses = test_data["losses"][:n_test]
    output_softmax = test_11["max_softmax"][:n_test]
    test_acts_8 = test_data["activations"][:n_test]

    # Binary ground truth: above-median loss = "error-prone"
    median_loss = np.median(test_losses)
    is_high_loss = test_losses > median_loss

    flag_rates = [0.05, 0.10, 0.20, 0.30]
    all_results = []

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        head = train_linear_binary(train_data, seed=seed)
        head.eval()
        with torch.inference_mode():
            obs_scores = head(test_acts_8).squeeze(-1).numpy()

        seed_result = {"flag_rates": flag_rates, "observer": {}, "confidence": {}, "exclusive": {}}

        for rate in flag_rates:
            k = int(n_test * rate)

            # Observer flags: highest observer scores (predicts high residual = error-prone)
            obs_threshold = np.sort(obs_scores)[-k]
            obs_flagged = obs_scores >= obs_threshold

            # Confidence flags: lowest softmax (least confident = error-prone)
            conf_threshold = np.sort(output_softmax)[k]
            conf_flagged = output_softmax <= conf_threshold

            # Precision: fraction of flagged tokens that are actually high-loss
            obs_precision = is_high_loss[obs_flagged].mean() if obs_flagged.sum() > 0 else 0.0
            conf_precision = is_high_loss[conf_flagged].mean() if conf_flagged.sum() > 0 else 0.0

            # Exclusive catches: high-loss tokens flagged by observer but NOT by confidence
            obs_exclusive = obs_flagged & ~conf_flagged & is_high_loss
            conf_exclusive = conf_flagged & ~obs_flagged & is_high_loss
            # Combined: flag if EITHER flags
            combined_flagged = obs_flagged | conf_flagged
            combined_precision = is_high_loss[combined_flagged].mean() if combined_flagged.sum() > 0 else 0.0

            seed_result["observer"][rate] = float(obs_precision)
            seed_result["confidence"][rate] = float(conf_precision)
            seed_result["exclusive"][rate] = {
                "observer_only": int(obs_exclusive.sum()),
                "confidence_only": int(conf_exclusive.sum()),
                "combined_precision": float(combined_precision),
            }

            print(
                f"    flag {rate:.0%}: observer prec={obs_precision:.3f}  "
                f"confidence prec={conf_precision:.3f}  "
                f"obs-exclusive catches={obs_exclusive.sum()}"
            )

        all_results.append(seed_result)

    # Aggregate
    print(f"\n  6a RESULT (averaged over {len(seeds)} seeds):")
    print(f"  {'Flag rate':<12} {'Observer prec':>14} {'Confidence prec':>16} {'Obs-exclusive':>14}")
    print(f"  {'-' * 58}")

    for rate in flag_rates:
        obs_p = np.mean([r["observer"][rate] for r in all_results])
        conf_p = np.mean([r["confidence"][rate] for r in all_results])
        excl = np.mean([r["exclusive"][rate]["observer_only"] for r in all_results])
        print(f"  {rate:<12.0%} {obs_p:>14.3f} {conf_p:>16.3f} {excl:>14.0f}")

    # Key question: does the observer catch errors confidence misses?
    total_exclusive_10 = np.mean([r["exclusive"][0.10]["observer_only"] for r in all_results])
    total_tokens = n_test
    print(
        f"\n  At 10% flag rate: observer catches {total_exclusive_10:.0f} high-loss tokens "
        f"that confidence misses ({total_exclusive_10 / total_tokens:.1%} of test set)"
    )

    combined_10 = np.mean([r["exclusive"][0.10]["combined_precision"] for r in all_results])
    obs_10 = np.mean([r["observer"][0.10] for r in all_results])
    conf_10 = np.mean([r["confidence"][0.10] for r in all_results])
    print(
        f"  Combined (flag if either flags): precision={combined_10:.3f} "
        f"(obs={obs_10:.3f}, conf={conf_10:.3f})"
    )

    if total_exclusive_10 > 100:
        print("\n  --> Observer catches substantial errors that confidence misses")
    else:
        print("\n  --> Observer adds limited exclusive coverage")

    return {
        "seeds": seeds,
        "n_test_tokens": n_test,
        "median_loss": float(median_loss),
        "per_seed": all_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--seeds", type=int, default=3, help="Number of observer head seeds")
    P.add_argument("--device", default="auto")
    P.add_argument("--max-tokens-train", type=int, default=200000)
    P.add_argument("--max-tokens-test", type=int, default=100000)
    P.add_argument("--layer-sweep", action="store_true", help="Run 5b layer sweep")
    P.add_argument("--baselines", action="store_true", help="Run 5c hand-designed baselines")
    P.add_argument("--output-control", action="store_true", help="Run 5e full-output control")
    P.add_argument("--intervention", action="store_true", help="Run 5d intervention")
    P.add_argument("--flagging", action="store_true", help="Run 6a early flagging")
    P.add_argument("--all", action="store_true", help="Run 5a-5e + 6a")
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    seeds = list(range(42, 42 + a.seeds))

    print("Phase 5: Transformer transfer")
    print(f"Model: GPT-2 124M  Device: {a.device}  Seeds: {seeds}")
    print(f"Train tokens: {a.max_tokens_train}  Test tokens: {a.max_tokens_test}")

    # Load model
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("\nLoading GPT-2 124M...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(a.device)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    # Load data (by document for proper train/test separation)
    print("\nLoading WikiText-103...")
    train_docs = load_wikitext("train", max_docs=2000)
    test_docs = load_wikitext("test", max_docs=500)
    print(f"  {len(train_docs)} train docs, {len(test_docs)} test docs")

    results = {}
    t0 = time.time()

    # 5a: always run
    results["5a"] = run_5a(
        model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
    )

    # 5b: layer sweep
    if a.layer_sweep or a.all:
        results["5b"] = run_5b(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # 5c: baselines
    if a.baselines or a.all:
        results["5c"] = run_5c(model, tokenizer, a.device, test_docs, a.max_tokens_test)

    # 5e: full-output control
    if a.output_control or a.all:
        results["5e"] = run_5e(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # 5d: intervention
    if a.intervention or a.all:
        results["5d"] = run_5d(
            model, tokenizer, a.device, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # 6a: early flagging
    if a.flagging or a.all:
        results["6a"] = run_6a(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save (merge with existing results to avoid overwriting prior runs)
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    out_file = out / "transformer_observe.json"
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
    existing.update(results)
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {out_file} (keys: {sorted(existing.keys())})")


if __name__ == "__main__":
    main()
