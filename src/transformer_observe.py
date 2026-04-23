"""
Transformer observer experiments.

Tests whether learned observer heads transfer to transformers, catch errors
confidence misses, and persist across the GPT-2 scaling curve and across
architecture families.

Experiments:
  5a: Direct replication at last layer (primary result)
  5b: Layer sweep across all 12 layers
  5c: Hand-designed baseline comparison
  5d: Neuron ablation intervention (null result: skip connections buffer damage)
  5e: Full-output control (layer 8 vs layer 11 predictor)
  5f: Directional ablation (residual stream projection)
  6a: Early flagging (observer vs confidence at 10% flag rate)
  8:  Scale characterization (GPT-2 124M through 1.5B)

Usage:
    uv run --extra transformer src/transformer_observe.py
    uv run --extra transformer src/transformer_observe.py --layer-sweep
    uv run --extra transformer src/transformer_observe.py --scale
    uv run --extra transformer src/transformer_observe.py --seeds 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from probe import (
    compute_hand_designed,
    evaluate_head,
    load_wikitext,
    partial_spearman,
    train_linear_binary,
)
from utils import _save_results, bootstrap_ci

# ---------------------------------------------------------------------------
# Architecture-agnostic helpers
# ---------------------------------------------------------------------------


def _get_layer_modules(model, layer_idx):
    """Return (attn_module, mlp_module) for a given layer index.

    Supports GPT-2 (model.transformer.h) and HuggingFace AutoModel
    architectures (Qwen, Llama, Mistral: model.model.layers).
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        block = model.transformer.h[layer_idx]
        return block.attn, block.mlp
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        block = model.model.layers[layer_idx]
        return block.self_attn, block.mlp
    raise ValueError(f"Unsupported model architecture: {type(model).__name__}")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_layer_data(model, tokenizer, docs, layer, device, max_tokens=200000, max_length=512):
    """Collect per-position data at one layer.

    Returns activations, per-position loss, max softmax, and activation norm.
    Train/test split is by document (caller provides the right split).
    """
    model.eval()
    all_acts, all_losses, all_softmax, all_norms, all_logit_entropy = [], [], [], [], []
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
        shift_logits = outputs.logits[0, :-1, :].float()
        shift_labels = input_ids[0, 1:]
        losses = F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu()
        # Confidence
        probs = F.softmax(shift_logits, dim=-1)
        max_sm = probs.max(dim=-1).values.cpu()
        logit_ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
        norms = h.norm(dim=-1)

        all_acts.append(h)
        all_losses.append(losses)
        all_softmax.append(max_sm)
        all_logit_entropy.append(logit_ent)
        all_norms.append(norms)
        total_tokens += h.size(0)

    print(f"    collected {total_tokens} positions from {len(all_acts)} documents")
    return {
        "activations": torch.cat(all_acts).float(),
        "losses": torch.cat(all_losses).float().numpy(),
        "max_softmax": torch.cat(all_softmax).float().numpy(),
        "logit_entropy": torch.cat(all_logit_entropy).float().numpy(),
        "activation_norm": torch.cat(all_norms).float().numpy(),
    }


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


def run_5b(
    model,
    tokenizer,
    device,
    seeds,
    train_docs,
    val_docs,
    test_docs,
    max_tokens_train,
    max_tokens_val,
    max_tokens_test,
):
    """5b: Layer sweep across all 12 layers.

    Sweeps on val_docs for layer selection, confirms peak on held-out test_docs.
    """
    n_layers = 12
    print(f"\n{'=' * 60}")
    print(f"  5b: Layer sweep (0-{n_layers - 1})")
    print(f"{'=' * 60}")

    # Sweep all layers on validation split
    layer_results = {}

    for layer in range(n_layers):
        print(f"\n  Layer {layer} [val]:")
        train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
        val_data = collect_layer_data(model, tokenizer, val_docs, layer, device, max_tokens_val)

        rhos = []
        for seed in seeds:
            head = train_linear_binary(train_data, seed=seed)
            _, rho, p = evaluate_head(head, val_data)
            rhos.append(rho)

        mean_rho = np.mean(rhos)
        print(f"    mean partial corr: {mean_rho:+.4f} +/- {np.std(rhos):.4f}")
        layer_results[layer] = {
            "mean": float(mean_rho),
            "std": float(np.std(rhos)),
            "per_seed": rhos,
            "split": "validation",
        }

    # Summary
    print("\n  5b LAYER PROFILE (validation):")
    print(f"  {'Layer':>6} {'Partial corr':>14}")
    print(f"  {'-' * 22}")
    for layer in range(n_layers):
        r = layer_results[layer]
        print(f"  {layer:>6} {r['mean']:>+14.4f}")

    peak_layer = max(range(n_layers), key=lambda l: layer_results[l]["mean"])
    print(f"\n  Val-selected peak at layer {peak_layer}: {layer_results[peak_layer]['mean']:+.4f}")

    # Confirm peak on held-out test split
    print(f"\n  Confirming layer {peak_layer} on held-out test split:")
    train_data = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, max_tokens_train)
    test_data = collect_layer_data(model, tokenizer, test_docs, peak_layer, device, max_tokens_test)
    test_rhos = []
    for seed in seeds:
        head = train_linear_binary(train_data, seed=seed)
        _, rho, p = evaluate_head(head, test_data)
        test_rhos.append(rho)
    test_mean = float(np.mean(test_rhos))
    print(f"    held-out test partial corr: {test_mean:+.4f} +/- {np.std(test_rhos):.4f}")

    layer_results["held_out_test"] = {
        "peak_layer": peak_layer,
        "peak_layer_source": "validation",
        "val_partial_corr": float(layer_results[peak_layer]["mean"]),
        "test_partial_corr": test_mean,
        "test_std": float(np.std(test_rhos)),
        "test_per_seed": [float(r) for r in test_rhos],
    }

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
# Intervention (5d: neuron ablation, 5f: directional ablation)
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
            loss = F.cross_entropy(outputs.logits[0, :-1, :].float(), input_ids[0, 1:])
            losses.append(loss.item())
    handle.remove()
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Directional ablation helpers (5f)
# ---------------------------------------------------------------------------


def _compute_confidence_direction(model, tokenizer, docs, layer, device, max_positions=10000):
    """Average gradient of max softmax w.r.t. hidden states at target layer.

    Gives a single unit vector capturing the direction in activation space
    that most affects output confidence. Used as a baseline: if projecting
    out the observer direction hurts more than this, the observer reads
    something beyond confidence.
    """
    model.eval()
    hidden_dim = model.config.hidden_size
    grad_sum = torch.zeros(hidden_dim, device=device)
    n_pos = 0

    for doc in docs:
        if n_pos >= max_positions:
            break
        if not doc.strip():
            continue

        tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(device)
        if input_ids.size(1) < 2:
            continue

        captured = [None]

        def capture_hook(module, input, output, _c=captured):
            h = output[0] if isinstance(output, tuple) else output
            h.retain_grad()
            _c[0] = h
            return output

        handle = model.transformer.h[layer].register_forward_hook(capture_hook)
        outputs = model(input_ids)
        handle.remove()

        probs = F.softmax(outputs.logits[0, :-1, :], dim=-1)
        max_sm = probs.max(dim=-1).values
        max_sm.sum().backward()

        h = captured[0]
        if h.grad is not None:
            grad_sum += h.grad[0, :-1, :].sum(dim=0)
            n_pos += h.grad.size(1) - 1

        model.zero_grad()

    if n_pos == 0:
        d = torch.randn(hidden_dim, device=device)
        return d / d.norm()

    d = grad_sum / n_pos
    return d / d.norm()


def _eval_direction_intervention(
    model, tokenizer, layer, direction, alpha, docs, device, max_tokens=50000, max_length=512
):
    """Run model with a directional projection at the residual stream.

    Intervention: h' = h - alpha * (h . d) * d
    Positive alpha removes the direction; negative alpha amplifies it.
    Returns per-position losses as a numpy array.
    """
    d = direction.to(device)

    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            proj = (h * d).sum(dim=-1, keepdim=True)
            return (h - alpha * proj * d,) + output[1:]
        proj = (output * d).sum(dim=-1, keepdim=True)
        return output - alpha * proj * d

    handle = model.transformer.h[layer].register_forward_hook(hook)
    all_losses = []
    total = 0

    with torch.inference_mode():
        for doc in docs:
            if total >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids)
            losses = (
                F.cross_entropy(outputs.logits[0, :-1, :].float(), input_ids[0, 1:], reduction="none")
                .cpu()
                .numpy()
            )
            all_losses.append(losses)
            total += len(losses)

    handle.remove()
    return np.concatenate(all_losses)[:max_tokens]


def _intervention_summary(losses, flags, baseline):
    """Summarize intervention effect overall and split by observer-flagged status."""
    return {
        "mean_loss": float(losses.mean()),
        "delta": float(losses.mean() - baseline.mean()),
        "flagged_mean": float(losses[flags].mean()) if flags.any() else 0.0,
        "unflagged_mean": float(losses[~flags].mean()) if (~flags).any() else 0.0,
        "flagged_delta": float(losses[flags].mean() - baseline[flags].mean()) if flags.any() else 0.0,
        "unflagged_delta": float(losses[~flags].mean() - baseline[~flags].mean()) if (~flags).any() else 0.0,
    }


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
            loss = F.cross_entropy(outputs.logits[0, :-1, :].float(), input_ids[0, 1:])
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


def run_5f(
    model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test, layer=8
):
    """5f: Directional ablation at the residual stream.

    Projects out the observer head's learned direction from the residual
    stream at the target layer and measures loss increase. Three baselines
    (observer, random, confidence direction), dose-response curve,
    per-example targeting, and bidirectional steering.

    Motivated by the 5d null result: neuron ablation failed because the
    signal is distributed and skip connections buffer MLP-level damage.
    Directional ablation intervenes on the residual stream directly,
    at the level of abstraction the observer reads from.
    """
    print(f"\n{'=' * 60}")
    print(f"  5f: Directional ablation at layer {layer}")
    print(f"{'=' * 60}")

    # --- Train observer, extract direction ---
    print("  Training observer head...")
    train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
    test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)

    head = train_linear_binary(train_data, seed=42)
    w = head.weight.data.squeeze()
    obs_dir = (w / w.norm()).to(device)

    # Observer scores for per-example targeting (top 20% = flagged)
    head.eval()
    with torch.inference_mode():
        obs_scores = head(test_data["activations"]).squeeze(-1).numpy()
    flagged_threshold = np.percentile(obs_scores, 80)
    is_flagged_full = obs_scores >= flagged_threshold

    # --- Confidence direction ---
    print("  Computing confidence direction (grad of max softmax)...")
    conf_dir = _compute_confidence_direction(model, tokenizer, train_docs[:100], layer, device)
    cosine_obs_conf = float((obs_dir * conf_dir).sum())
    print(f"    cosine(observer, confidence) = {cosine_obs_conf:.4f}")

    # --- Random directions ---
    rng = np.random.default_rng(42)
    hidden_dim = int(obs_dir.shape[0])
    n_random = 5
    random_dirs = []
    for _ in range(n_random):
        v = torch.from_numpy(rng.standard_normal(hidden_dim).astype(np.float32)).to(device)
        random_dirs.append(v / v.norm())

    # --- Evaluation setup ---
    eval_docs = test_docs[:300]
    eval_budget = min(max_tokens_test, 50000)

    # Baseline (no intervention)
    print("  Baseline evaluation...")
    baseline = _eval_direction_intervention(
        model, tokenizer, layer, obs_dir, 0.0, eval_docs, device, eval_budget
    )
    n_eval = len(baseline)
    flags = is_flagged_full[:n_eval]
    print(f"    baseline loss: {baseline.mean():.4f}  ({n_eval} positions, {flags.sum()} flagged)")

    removal_alphas = [0.25, 0.5, 0.75, 1.0]
    amplify_alphas = [-0.25, -0.5, -0.75, -1.0]

    # --- Observer direction: removal + amplification ---
    print("\n  Observer direction (removal):")
    obs_removal = {0.0: _intervention_summary(baseline, flags, baseline)}
    for alpha in removal_alphas:
        losses = _eval_direction_intervention(
            model, tokenizer, layer, obs_dir, alpha, eval_docs, device, eval_budget
        )
        obs_removal[alpha] = _intervention_summary(losses, flags, baseline)
        s = obs_removal[alpha]
        print(
            f"    remove {alpha:.0%}: loss={s['mean_loss']:.4f}  "
            f"delta={s['delta']:+.4f}  "
            f"flagged={s['flagged_delta']:+.4f}  unflagged={s['unflagged_delta']:+.4f}"
        )

    print("  Observer direction (amplification):")
    obs_amplify = {}
    for alpha in amplify_alphas:
        losses = _eval_direction_intervention(
            model, tokenizer, layer, obs_dir, alpha, eval_docs, device, eval_budget
        )
        obs_amplify[alpha] = _intervention_summary(losses, flags, baseline)
        s = obs_amplify[alpha]
        print(
            f"    amplify {abs(alpha):.0%}: loss={s['mean_loss']:.4f}  "
            f"delta={s['delta']:+.4f}  flagged={s['flagged_delta']:+.4f}"
        )

    # --- Random directions (removal only) ---
    print("\n  Random directions:")
    random_removal = {0.0: {"mean_loss": float(baseline.mean()), "delta": 0.0, "std": 0.0}}
    for alpha in removal_alphas:
        rand_means = []
        for rd in random_dirs:
            losses = _eval_direction_intervention(
                model, tokenizer, layer, rd, alpha, eval_docs, device, eval_budget
            )
            rand_means.append(float(losses.mean()))
        random_removal[alpha] = {
            "mean_loss": float(np.mean(rand_means)),
            "delta": float(np.mean(rand_means) - baseline.mean()),
            "std": float(np.std(rand_means)),
        }
        s = random_removal[alpha]
        print(
            f"    remove {alpha:.0%}: loss={s['mean_loss']:.4f} +/- {s['std']:.4f}  delta={s['delta']:+.4f}"
        )

    # --- Confidence direction (removal only) ---
    print("\n  Confidence direction:")
    conf_removal = {0.0: _intervention_summary(baseline, flags, baseline)}
    for alpha in removal_alphas:
        losses = _eval_direction_intervention(
            model, tokenizer, layer, conf_dir, alpha, eval_docs, device, eval_budget
        )
        conf_removal[alpha] = _intervention_summary(losses, flags, baseline)
        s = conf_removal[alpha]
        print(f"    remove {alpha:.0%}: loss={s['mean_loss']:.4f}  delta={s['delta']:+.4f}")

    # --- Summary tables ---
    print(f"\n  5d DOSE-RESPONSE (layer {layer}):")
    print(
        f"  {'Alpha':<8} {'Observer':>10} {'Random':>10} {'Confidence':>12}"
        f" {'Obs flagged':>12} {'Obs unflag':>12}"
    )
    print(f"  {'-' * 66}")
    for alpha in [0.0] + removal_alphas:
        o = obs_removal[alpha]["delta"]
        r = random_removal[alpha]["delta"]
        c = conf_removal[alpha]["delta"]
        fd = obs_removal[alpha]["flagged_delta"]
        ud = obs_removal[alpha]["unflagged_delta"]
        print(f"  {alpha:<8.2f} {o:>+10.4f} {r:>+10.4f} {c:>+12.4f} {fd:>+12.4f} {ud:>+12.4f}")

    print("\n  AMPLIFICATION (observer direction):")
    print(f"  {'Alpha':<8} {'Overall':>10} {'Flagged':>10} {'Unflagged':>10}")
    print(f"  {'-' * 40}")
    for alpha in amplify_alphas:
        s = obs_amplify[alpha]
        print(
            f"  {alpha:<+8.2f} {s['delta']:>+10.4f} {s['flagged_delta']:>+10.4f}"
            f" {s['unflagged_delta']:>+10.4f}"
        )

    # --- Interpretive tests ---
    obs_delta_100 = obs_removal[1.0]["delta"]
    rand_delta_100 = random_removal[1.0]["delta"]
    rand_std_100 = random_removal[1.0]["std"]
    conf_delta_100 = conf_removal[1.0]["delta"]
    flagged_delta_100 = obs_removal[1.0]["flagged_delta"]
    unflagged_delta_100 = obs_removal[1.0]["unflagged_delta"]

    # 1. Observer > random? (z-test against random distribution)
    obs_gt_random = (
        (obs_delta_100 > rand_delta_100 + 2 * rand_std_100)
        if rand_std_100 > 0
        else (obs_delta_100 > rand_delta_100)
    )

    # 2. Observer > confidence?
    obs_gt_conf = obs_delta_100 > conf_delta_100

    # 3. Dose-response monotonic?
    removal_deltas = [obs_removal[a]["delta"] for a in [0.0] + removal_alphas]
    monotonic = all(removal_deltas[i] <= removal_deltas[i + 1] for i in range(len(removal_deltas) - 1))

    # 4. Flagged tokens degrade more?
    targeting = flagged_delta_100 > unflagged_delta_100 and unflagged_delta_100 >= 0

    # 5. Amplification helps flagged tokens?
    amp_helps = obs_amplify[-1.0]["flagged_delta"] < 0

    targeting_ratio = flagged_delta_100 / unflagged_delta_100 if unflagged_delta_100 > 1e-6 else float("inf")

    tests = [
        (
            "observer > random (2-sigma)",
            obs_gt_random,
            f"{obs_delta_100:+.4f} vs {rand_delta_100:+.4f} +/- {rand_std_100:.4f}",
        ),
        ("observer > confidence", obs_gt_conf, f"{obs_delta_100:+.4f} vs {conf_delta_100:+.4f}"),
        ("dose-response monotonic", monotonic, ""),
        ("flagged degrade more", targeting, f"ratio={targeting_ratio:.2f}"),
        (
            "amplification helps flagged",
            amp_helps,
            f"flagged delta={obs_amplify[-1.0]['flagged_delta']:+.4f}",
        ),
    ]

    print("\n  INTERPRETIVE TESTS:")
    n_pass = 0
    for name, passed, detail in tests:
        tag = "PASS" if passed else "FAIL"
        n_pass += passed
        print(f"    {name:<32} {tag}  {detail}")

    if n_pass >= 4:
        print(f"\n  --> STRONG CAUSAL ({n_pass}/5): direction is functional, not decorative")
    elif n_pass >= 2:
        print(f"\n  --> PARTIAL CAUSAL ({n_pass}/5): some functional signal")
    else:
        print(f"\n  --> WEAK/NEGATIVE ({n_pass}/5): direction may be epiphenomenal")

    return {
        "layer": layer,
        "cosine_obs_conf": cosine_obs_conf,
        "baseline_loss": float(baseline.mean()),
        "n_eval_positions": n_eval,
        "n_flagged": int(flags.sum()),
        "observer_removal": {str(k): v for k, v in obs_removal.items()},
        "observer_amplify": {str(k): v for k, v in obs_amplify.items()},
        "random_removal": {str(k): v for k, v in random_removal.items()},
        "confidence_removal": {str(k): v for k, v in conf_removal.items()},
        "monotonic": monotonic,
        "targeting_ratio": float(targeting_ratio) if targeting_ratio != float("inf") else None,
        "n_tests_passed": n_pass,
    }


# ---------------------------------------------------------------------------
# Early flagging
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
# Generalization and statistical hardening
# ---------------------------------------------------------------------------


def run_8a(model, tokenizer, device, train_docs, test_docs, max_tokens_train, max_tokens_test):
    """8a: 20-seed statistical hardening at layer 11.

    Identical to 5a but with 20 seeds and bootstrap 95% confidence intervals.
    Hardcodes seed count to 20 regardless of --seeds flag.
    """
    seeds = list(range(42, 62))
    layer = 11
    print(f"\n{'=' * 60}")
    print(f"  8a: 20-seed statistical hardening at layer {layer}")
    print(f"{'=' * 60}")

    print("  Collecting train activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
    print("  Collecting test activations...")
    test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)

    all_scores = []
    all_rhos = []

    for seed in seeds:
        head = train_linear_binary(train_data, seed=seed)
        scores, rho, p = evaluate_head(head, test_data)
        all_scores.append(scores)
        all_rhos.append(rho)
        print(f"    seed {seed}: partial corr = {rho:+.4f}")

    # Pairwise seed agreement
    pairwise = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            r, _ = spearmanr(all_scores[i], all_scores[j])
            pairwise.append(r)

    mean_rho = float(np.mean(all_rhos))
    std_rho = float(np.std(all_rhos))
    rho_ci = bootstrap_ci(all_rhos)
    mean_agree = float(np.mean(pairwise))
    std_agree = float(np.std(pairwise))
    agree_ci = bootstrap_ci(pairwise)

    print(f"\n  8a RESULT ({len(seeds)} seeds):")
    print(
        f"    partial corr:    {mean_rho:+.4f} +/- {std_rho:.4f}  95% CI [{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]"
    )
    print(
        f"    seed agreement:  {mean_agree:+.4f} +/- {std_agree:.4f}  95% CI [{agree_ci[0]:+.4f}, {agree_ci[1]:+.4f}]"
    )

    return {
        "layer": layer,
        "n_seeds": len(seeds),
        "partial_corrs": [float(r) for r in all_rhos],
        "mean_partial_corr": mean_rho,
        "std_partial_corr": std_rho,
        "partial_corr_ci_95": list(rho_ci),
        "seed_agreement": [float(r) for r in pairwise],
        "mean_seed_agreement": mean_agree,
        "std_seed_agreement": std_agree,
        "seed_agreement_ci_95": list(agree_ci),
    }


def run_8b(model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test):
    """8b: Control sensitivity analysis.

    Train observer heads at layer 11, then evaluate each under six different
    control specifications to test robustness of the partial correlation.
    """
    layer = 11
    print(f"\n{'=' * 60}")
    print(f"  8b: Control sensitivity analysis at layer {layer}")
    print(f"{'=' * 60}")

    print("  Collecting train activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
    print("  Collecting test activations...")
    test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)

    # Train nonlinear confidence predictor (MLP on [max_softmax, activation_norm] -> loss)
    print("  Training nonlinear confidence predictor...")
    conf_features_train = torch.from_numpy(
        np.column_stack([train_data["max_softmax"], train_data["activation_norm"]])
    ).float()
    loss_targets_train = torch.from_numpy(train_data["losses"]).float()

    torch.manual_seed(42)
    mlp_pred = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(mlp_pred.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(conf_features_train, loss_targets_train)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    for _ep in range(20):
        for bx, by in dl:
            loss = F.mse_loss(mlp_pred(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    mlp_pred.eval()
    conf_features_test = torch.from_numpy(
        np.column_stack([test_data["max_softmax"], test_data["activation_norm"]])
    ).float()
    with torch.inference_mode():
        mlp_pred_scores = mlp_pred(conf_features_test).squeeze(-1).numpy()

    # Define control sets
    control_sets = {
        "softmax_only": [test_data["max_softmax"]],
        "norm_only": [test_data["activation_norm"]],
        "standard": [test_data["max_softmax"], test_data["activation_norm"]],
        "plus_entropy": [test_data["max_softmax"], test_data["activation_norm"], test_data["logit_entropy"]],
        "nonlinear": [mlp_pred_scores],
    }

    results = {"layer": layer, "n_seeds": len(seeds), "control_sets": {}}

    # Train observer heads and evaluate under each control set
    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        head = train_linear_binary(train_data, seed=seed)
        head.eval()
        with torch.inference_mode():
            scores = head(test_data["activations"]).squeeze(-1).numpy()

        # Raw Spearman (no controls)
        raw_rho, raw_p = spearmanr(scores, test_data["losses"])
        results["control_sets"].setdefault("none", {"per_seed": []})
        results["control_sets"]["none"]["per_seed"].append(float(raw_rho))
        print(f"    none:          rho = {raw_rho:+.4f}")

        # Partial correlations under each control set
        for name, covariates in control_sets.items():
            rho, p = partial_spearman(scores, test_data["losses"], covariates)
            results["control_sets"].setdefault(name, {"per_seed": []})
            results["control_sets"][name]["per_seed"].append(float(rho))
            print(f"    {name:<15} rho = {rho:+.4f}")

    # Aggregate
    print(f"\n  8b RESULT ({len(seeds)} seeds):")
    print(f"  {'Control set':<18} {'Mean':>8} {'95% CI':>20}")
    print(f"  {'-' * 48}")
    for name in ["none", "softmax_only", "norm_only", "standard", "plus_entropy", "nonlinear"]:
        vals = results["control_sets"][name]["per_seed"]
        mean = float(np.mean(vals))
        ci = bootstrap_ci(vals) if len(vals) >= 3 else (mean, mean)
        results["control_sets"][name]["mean"] = mean
        results["control_sets"][name]["ci_95"] = list(ci)
        print(f"  {name:<18} {mean:>+8.4f} [{ci[0]:>+8.4f}, {ci[1]:>+8.4f}]")

    # Interpretive summary
    std_mean = results["control_sets"]["standard"]["mean"]
    ent_mean = results["control_sets"]["plus_entropy"]["mean"]
    nl_mean = results["control_sets"]["nonlinear"]["mean"]

    if abs(std_mean - ent_mean) < 0.01:
        print("\n  Entropy is redundant with existing controls.")
    if nl_mean < 0.05:
        print("\n  WARNING: Signal collapses under nonlinear deconfounding.")
    elif nl_mean > 0.15:
        print(f"\n  Signal survives nonlinear deconfounding (+{nl_mean:.3f}).")

    return results


def run_mechanism_probes(
    model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test
):
    """Mechanism probes: what is the observer reading?

    Compute three representation-derived proxies and test whether they
    absorb the observer signal when added as partial correlation controls.

    1. Representational coherence: Mahalanobis distance from layer mean
    2. Trajectory instability: sensitivity of loss to activation perturbation
    3. Computation difficulty: magnitude of residual stream update at this layer
    """
    layer = 8
    print(f"\n{'=' * 60}")
    print(f"  Mechanism probes at layer {layer}")
    print(f"{'=' * 60}")

    # Collect activations at layer 8 (train + test) and layer 7 (test only, for update magnitude)
    print("  Collecting layer 8 train activations...")
    train_8 = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
    print("  Collecting layer 8 test activations...")
    test_8 = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)
    print("  Collecting layer 7 test activations...")
    test_7 = collect_layer_data(model, tokenizer, test_docs, layer - 1, device, max_tokens_test)

    n_test = len(test_8["losses"])

    # --- Proxy 1: Representational coherence (Mahalanobis distance) ---
    print("\n  Computing representational coherence...")
    acts_train = train_8["activations"].numpy()
    acts_test = test_8["activations"].numpy()

    mean_act = acts_train.mean(axis=0)
    centered = acts_train - mean_act
    # Regularized covariance for numerical stability
    cov = np.cov(centered, rowvar=False) + 1e-4 * np.eye(centered.shape[1])
    cov_inv = np.linalg.inv(cov)

    test_centered = acts_test - mean_act
    # Mahalanobis: sqrt((x - mu)^T Sigma^-1 (x - mu))
    # Compute efficiently: (X @ L) where L L^T = Sigma^-1
    L = np.linalg.cholesky(cov_inv)
    projected = test_centered @ L
    mahalanobis = np.sqrt((projected**2).sum(axis=1))
    print(f"    Mahalanobis: mean={mahalanobis.mean():.2f}, std={mahalanobis.std():.2f}")

    # --- Proxy 2: Trajectory instability (perturbation sensitivity) ---
    print("  Computing trajectory instability...")
    noise_std = 0.1
    rng = np.random.default_rng(42)

    # Approximate: perturb layer-8 activations, measure change in observer score
    head = train_linear_binary(train_8, seed=42)
    head.eval()
    with torch.inference_mode():
        base_scores = head(test_8["activations"]).squeeze(-1).numpy()

    perturbation_deltas = []
    n_samples = 5
    for _ in range(n_samples):
        noise = torch.from_numpy(
            rng.normal(0, noise_std, size=(n_test, test_8["activations"].shape[1])).astype(np.float32)
        )
        perturbed = test_8["activations"] + noise
        with torch.inference_mode():
            perturbed_scores = head(perturbed).squeeze(-1).numpy()
        perturbation_deltas.append(np.abs(perturbed_scores - base_scores))

    instability = np.mean(perturbation_deltas, axis=0)
    print(f"    Instability: mean={instability.mean():.4f}, std={instability.std():.4f}")

    # --- Proxy 3: Computation difficulty (update magnitude) ---
    print("  Computing computation difficulty...")
    acts_7 = test_7["activations"][:n_test].numpy()
    acts_8 = test_8["activations"][:n_test].numpy()

    # Residual stream update: layer_8 - layer_7
    update = acts_8 - acts_7
    update_magnitude = np.sqrt((update**2).sum(axis=1))

    # Also cosine similarity between consecutive layers
    norm_7 = np.sqrt((acts_7**2).sum(axis=1, keepdims=True)) + 1e-8
    norm_8 = np.sqrt((acts_8**2).sum(axis=1, keepdims=True)) + 1e-8
    cosine_sim = (acts_7 * acts_8).sum(axis=1) / (norm_7.squeeze() * norm_8.squeeze())
    print(f"    Update magnitude: mean={update_magnitude.mean():.2f}, std={update_magnitude.std():.2f}")
    print(f"    Cosine(L7, L8): mean={cosine_sim.mean():.4f}, std={cosine_sim.std():.4f}")

    # --- Partial correlation analysis ---
    print(f"\n  Partial correlation analysis ({len(seeds)} seeds):")
    standard_controls = [test_8["max_softmax"], test_8["activation_norm"]]

    proxy_sets = {
        "standard": standard_controls,
        "+ mahalanobis": standard_controls + [mahalanobis],
        "+ instability": standard_controls + [instability],
        "+ update_magnitude": standard_controls + [update_magnitude],
        "+ cosine_sim": standard_controls + [cosine_sim],
        "+ all_proxies": standard_controls + [mahalanobis, update_magnitude, cosine_sim],
    }

    results = {"layer": layer, "n_seeds": len(seeds), "proxies": {}}

    for seed in seeds:
        print(f"\n  --- Seed {seed} ---")
        obs_head = train_linear_binary(train_8, seed=seed)
        obs_head.eval()
        with torch.inference_mode():
            obs_scores = obs_head(test_8["activations"]).squeeze(-1).numpy()

        for name, controls in proxy_sets.items():
            rho, p = partial_spearman(obs_scores, test_8["losses"], controls)
            results["proxies"].setdefault(name, {"per_seed": []})
            results["proxies"][name]["per_seed"].append(float(rho))
            print(f"    {name:<22} rho = {rho:+.4f}")

    # Summary
    print("\n  MECHANISM PROBE RESULTS:")
    print(f"  {'Control':<24} {'Mean':>8} {'Delta vs std':>14}")
    print(f"  {'-' * 48}")
    std_mean = float(np.mean(results["proxies"]["standard"]["per_seed"]))
    for name in proxy_sets:
        vals = results["proxies"][name]["per_seed"]
        mean = float(np.mean(vals))
        ci = bootstrap_ci(vals) if len(vals) >= 3 else (mean, mean)
        results["proxies"][name]["mean"] = mean
        results["proxies"][name]["ci_95"] = list(ci)
        delta = mean - std_mean
        absorbed = f"absorbs {abs(delta) / std_mean * 100:.0f}%" if delta < -0.005 else "no effect"
        print(f"  {name:<24} {mean:>+8.4f} {delta:>+10.4f}  ({absorbed})")

    # Raw correlations of proxies with loss
    print("\n  Raw proxy correlations with loss:")
    from scipy.stats import spearmanr as _sp

    for pname, pvals in [
        ("mahalanobis", mahalanobis),
        ("update_magnitude", update_magnitude),
        ("cosine_sim", cosine_sim),
    ]:
        r, p = _sp(pvals, test_8["losses"][: len(pvals)])
        print(f"    {pname:<22} Spearman = {r:+.4f}")

    return results


def run_signal_decomposition(
    model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test
):
    """Signal decomposition: break the observer signal into named components.

    Extends the mechanism probes with:
    1. Token frequency as control (is the observer reading word rarity?)
    2. Confident-error analysis (where is the signal concentrated?)
    3. Cross-layer residual tracking (where does output-independent signal form?)
    4. Observer weight PCA alignment (what directions is it reading?)
    """
    layer = 8
    print(f"\n{'=' * 60}")
    print(f"  Signal decomposition at layer {layer}")
    print(f"{'=' * 60}")

    print("  Collecting train activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
    print("  Collecting test activations...")
    test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)

    n_test = len(test_data["losses"])

    # Train observer
    head = train_linear_binary(train_data, seed=42)
    head.eval()
    with torch.inference_mode():
        obs_scores = head(test_data["activations"]).squeeze(-1).numpy()

    # --- 1. Token frequency control ---
    print("\n  1. Token frequency analysis")

    # Compute token frequencies from training documents
    from collections import Counter

    token_counts = Counter()
    for doc in train_docs:
        tokens = tokenizer(doc, truncation=True, max_length=512)["input_ids"]
        token_counts.update(tokens)
    total_tokens = sum(token_counts.values())

    # Get test token IDs and their log-frequencies
    test_token_ids = []
    for doc in test_docs:
        tokens = tokenizer(doc, truncation=True, max_length=512)["input_ids"]
        if len(tokens) >= 2:
            test_token_ids.extend(tokens[1:])  # shift by 1 to match loss positions
    test_token_ids = test_token_ids[:n_test]
    log_freq = np.array([np.log(token_counts.get(tid, 1) / total_tokens + 1e-10) for tid in test_token_ids])

    from scipy.stats import spearmanr as _sp

    freq_loss_corr, _ = _sp(log_freq, test_data["losses"])
    print(f"    log_freq vs loss Spearman: {freq_loss_corr:+.4f}")

    standard_controls = [test_data["max_softmax"], test_data["activation_norm"]]
    rho_standard, _ = partial_spearman(obs_scores, test_data["losses"], standard_controls)
    rho_freq, _ = partial_spearman(obs_scores, test_data["losses"], standard_controls + [log_freq])
    print(f"    standard control:    {rho_standard:+.4f}")
    print(f"    + token frequency:   {rho_freq:+.4f}")
    freq_absorbed = (rho_standard - rho_freq) / rho_standard * 100
    print(f"    frequency absorbs:   {freq_absorbed:.1f}%")

    # --- 2. Confident-error decomposition ---
    print("\n  2. Confident-error analysis")

    losses = test_data["losses"]
    softmax = test_data["max_softmax"]
    median_loss = np.median(losses)
    median_conf = np.median(softmax)

    quadrants = {
        "high_conf_low_loss": (softmax >= median_conf) & (losses <= median_loss),
        "high_conf_high_loss": (softmax >= median_conf) & (losses > median_loss),
        "low_conf_low_loss": (softmax < median_conf) & (losses <= median_loss),
        "low_conf_high_loss": (softmax < median_conf) & (losses > median_loss),
    }

    print(f"    {'Quadrant':<25} {'N':>6} {'Obs corr w/ loss':>18} {'Mean obs score':>16}")
    print(f"    {'-' * 67}")
    quadrant_results = {}
    for qname, mask in quadrants.items():
        n_q = mask.sum()
        if n_q > 100:
            r, _ = _sp(obs_scores[mask], losses[mask])
            mean_score = obs_scores[mask].mean()
        else:
            r, mean_score = 0.0, 0.0
        quadrant_results[qname] = {"n": int(n_q), "spearman": float(r), "mean_score": float(mean_score)}
        print(f"    {qname:<25} {n_q:>6} {r:>+18.4f} {mean_score:>16.4f}")

    # --- 3. Cross-layer residual tracking ---
    print("\n  3. Cross-layer output-independent signal")

    # Train output-side predictor from last layer
    print("    Collecting layer 11 data for output control...")
    train_11 = collect_layer_data(model, tokenizer, train_docs, 11, device, max_tokens_train)
    test_11 = collect_layer_data(model, tokenizer, test_docs, 11, device, max_tokens_test)

    torch.manual_seed(42)
    np.random.seed(42)
    n_feat = train_11["activations"].size(1)
    predictor = torch.nn.Sequential(torch.nn.Linear(n_feat, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(train_11["activations"], torch.from_numpy(train_11["losses"]).float())
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    for _ep in range(20):
        for bx, by in dl:
            loss = F.mse_loss(predictor(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    predictor.eval()
    with torch.inference_mode():
        pred_11_scores = predictor(test_11["activations"]).squeeze(-1).numpy()

    layer_residuals = {}
    for probe_layer in [0, 2, 4, 6, 8, 10, 11]:
        layer_train = collect_layer_data(model, tokenizer, train_docs, probe_layer, device, max_tokens_train)
        layer_test = collect_layer_data(model, tokenizer, test_docs, probe_layer, device, max_tokens_test)
        layer_head = train_linear_binary(layer_train, seed=42)
        layer_head.eval()
        with torch.inference_mode():
            layer_scores = layer_head(layer_test["activations"]).squeeze(-1).numpy()

        rho_std, _ = partial_spearman(layer_scores, layer_test["losses"], standard_controls)
        rho_ctrl, _ = partial_spearman(
            layer_scores,
            layer_test["losses"],
            [layer_test["max_softmax"], layer_test["activation_norm"], pred_11_scores],
        )
        layer_residuals[probe_layer] = {"standard": float(rho_std), "output_controlled": float(rho_ctrl)}
        print(f"    layer {probe_layer:>2}: standard={rho_std:+.4f}  output-controlled={rho_ctrl:+.4f}")

    # --- 4. Observer weight PCA alignment ---
    print("\n  4. Observer weight PCA alignment")

    acts_test = test_data["activations"].numpy()
    # PCA on test activations
    mean_a = acts_test.mean(axis=0)
    centered = acts_test - mean_a
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Sort descending
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Observer weight direction
    w = head.weight.data.cpu().squeeze().numpy()
    w_unit = w / np.linalg.norm(w)

    # Project observer direction onto PCs
    projections = np.abs(eigenvectors.T @ w_unit)
    print("    Top-10 PC alignment (|cos| with observer direction):")
    for i in range(10):
        print(
            f"      PC{i:>3} (var {eigenvalues[i] / eigenvalues.sum() * 100:>5.1f}%): |cos| = {projections[i]:.4f}"
        )

    # How much of the observer direction lives in the top-k PCs?
    for k in [10, 50, 100, 200]:
        frac = (projections[:k] ** 2).sum()
        print(f"    Top-{k} PCs capture {frac * 100:.1f}% of observer direction")

    # --- 5. Shapley-robust decomposition ---
    print("\n  5. Shapley-robust decomposition (4 controls, 24 orderings)")

    # Compute Mahalanobis distance for geometric typicality control
    acts_train = train_data["activations"].numpy()
    mean_act = acts_train.mean(axis=0)
    centered_train = acts_train - mean_act
    cov = np.cov(centered_train, rowvar=False) + 1e-4 * np.eye(centered_train.shape[1])
    cov_inv = np.linalg.inv(cov)
    L = np.linalg.cholesky(cov_inv)
    test_centered = acts_test - mean_act
    projected_mahal = test_centered @ L
    mahalanobis = np.sqrt((projected_mahal**2).sum(axis=1))

    # Named controls for Shapley decomposition
    control_map = {
        "confidence": test_data["max_softmax"],
        "entropy": test_data["logit_entropy"],
        "typicality": mahalanobis,
        "frequency": log_freq,
    }
    control_names = list(control_map.keys())

    # Baseline: no controls (raw Spearman)
    from itertools import permutations

    rho_raw, _ = _sp(obs_scores, test_data["losses"])

    # Compute partial correlation for every subset via all 24 orderings
    ordering_results = []
    for perm in permutations(control_names):
        row = {"ordering": list(perm)}
        cumulative_controls = []
        prev_rho = float(rho_raw)
        for ctrl_name in perm:
            cumulative_controls.append(control_map[ctrl_name])
            rho_after, _ = partial_spearman(obs_scores, test_data["losses"], cumulative_controls)
            marginal = prev_rho - float(rho_after)
            row[ctrl_name] = float(marginal)
            prev_rho = float(rho_after)
        row["unexplained"] = float(prev_rho)
        ordering_results.append(row)

    # Shapley values: average marginal contribution per control
    shapley_values = {}
    for ctrl_name in control_names:
        vals = [r[ctrl_name] for r in ordering_results]
        shapley_values[ctrl_name] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    unexplained_vals = [r["unexplained"] for r in ordering_results]
    total_explained = sum(sv["mean"] for sv in shapley_values.values())
    total_explained_pct = total_explained / float(rho_raw) * 100
    unexplained_pct = 100 - total_explained_pct

    print(f"\n  {'Control':<16} {'Shapley mean':>13} {'Min':>8} {'Max':>8} {'% of raw':>10}")
    print(f"  {'-' * 57}")
    for ctrl_name in control_names:
        sv = shapley_values[ctrl_name]
        pct = sv["mean"] / float(rho_raw) * 100
        print(f"  {ctrl_name:<16} {sv['mean']:>+13.4f} {sv['min']:>+8.4f} {sv['max']:>+8.4f} {pct:>9.1f}%")
    print(f"  {'unexplained':<16} {float(np.mean(unexplained_vals)):>+13.4f}")
    print(f"\n  Total explained: {total_explained_pct:.1f}%, unexplained: {unexplained_pct:.1f}%")

    # --- Summary ---
    print("\n  SIGNAL DECOMPOSITION SUMMARY")
    print(f"  Raw Spearman: {rho_raw:+.4f}")
    print(f"  Shapley-explained: {total_explained_pct:.1f}%, unexplained: {unexplained_pct:.1f}%")

    return {
        "layer": layer,
        "token_frequency": {
            "standard": float(rho_standard),
            "with_frequency": float(rho_freq),
            "absorbed_pct": float(freq_absorbed),
            "freq_loss_corr": float(freq_loss_corr),
        },
        "quadrants": quadrant_results,
        "cross_layer_residuals": {str(k): v for k, v in layer_residuals.items()},
        "pca_alignment": {
            "top_10_projections": [float(p) for p in projections[:10]],
            "top_10_capture": float((projections[:10] ** 2).sum()),
            "top_50_capture": float((projections[:50] ** 2).sum()),
            "top_100_capture": float((projections[:100] ** 2).sum()),
        },
        "shapley_decomposition": {
            "raw_spearman": float(rho_raw),
            "shapley_values": shapley_values,
            "total_explained_pct": float(total_explained_pct),
            "unexplained_pct": float(unexplained_pct),
            "all_orderings": ordering_results,
        },
    }


def _run_matched_component_sweep(model, observer, pairs, peak_layer):
    """Sweep layers 0-peak, amplifying each component at low-score positions."""
    print(f"\n  Component-level matched patching (layers 0-{peak_layer}):")
    print(f"  {'Layer':<8} {'Comp':<6} {'Obs delta':>11} {'Loss delta':>12} {'Interpretation':>20}")
    print(f"  {'-' * 59}")

    component_results = {}
    for layer in range(peak_layer + 1):
        layer_results = {}
        for comp in ["attn", "mlp"]:
            obs_deltas, loss_deltas = _patch_one_component(
                model, observer, pairs[:50], peak_layer, layer, comp
            )

            mean_obs = float(np.mean(obs_deltas)) if obs_deltas else 0.0
            mean_loss = float(np.mean(loss_deltas)) if loss_deltas else 0.0

            if mean_obs > 0.005 and mean_loss < -0.005:
                interp = "CAUSAL +"
            elif mean_obs > 0.005:
                interp = "obs only"
            elif mean_loss < -0.005:
                interp = "loss only"
            else:
                interp = "no effect"

            layer_results[comp] = {"obs_delta": mean_obs, "loss_delta": mean_loss, "interp": interp}
            print(f"  {layer:<8} {comp:<6} {mean_obs:>+11.4f} {mean_loss:>+12.4f} {interp:>20}")
        component_results[layer] = layer_results
    return component_results


def _patch_one_component(model, observer, pairs, peak_layer, layer, comp):
    """Amplify one component at low-score positions, measure observer and loss deltas."""
    obs_deltas = []
    loss_deltas = []

    for pair in pairs:
        input_ids = pair["input_ids"]
        low_pos = pair["low_positions"]
        block = model.transformer.h[layer]
        target = block.attn if comp == "attn" else block.mlp

        # Baseline
        with torch.inference_mode():
            base_out = model(input_ids, output_hidden_states=True)
        base_h = base_out.hidden_states[peak_layer + 1][0, :-1, :].cpu().float()
        base_obs = observer(base_h).squeeze(-1).detach().numpy()
        base_loss = (
            F.cross_entropy(base_out.logits[0, :-1, :].float(), input_ids[0, 1:], reduction="none")
            .cpu()
            .numpy()
        )

        # Capture clean component output
        clean_out = [None]

        def capture(module, inp, out, _c=clean_out):
            _c[0] = (out[0] if isinstance(out, tuple) else out).detach().clone()
            return out

        h1 = target.register_forward_hook(capture)
        with torch.inference_mode():
            model(input_ids, output_hidden_states=True)
        h1.remove()

        # Amplify at low-score positions
        def amplify(module, inp, out, _cl=clean_out[0], _pos=low_pos):
            mod = (out[0] if isinstance(out, tuple) else out).clone()
            for p in _pos:
                if p < mod.size(1):
                    mod[0, p, :] += 0.5 * _cl[0, p, :]
            return (mod,) + out[1:] if isinstance(out, tuple) else mod

        h2 = target.register_forward_hook(amplify)
        with torch.inference_mode():
            patch_out = model(input_ids, output_hidden_states=True)
        h2.remove()

        patch_h = patch_out.hidden_states[peak_layer + 1][0, :-1, :].cpu().float()
        patch_obs = observer(patch_h).squeeze(-1).detach().numpy()
        patch_loss = (
            F.cross_entropy(patch_out.logits[0, :-1, :].float(), input_ids[0, 1:], reduction="none")
            .cpu()
            .numpy()
        )

        valid = low_pos[low_pos < len(base_obs)]
        if len(valid) > 0:
            obs_deltas.append(float((patch_obs[valid] - base_obs[valid]).mean()))
            loss_deltas.append(float((patch_loss[valid] - base_loss[valid]).mean()))

    return obs_deltas, loss_deltas


def run_matched_pair_patching(
    model, tokenizer, device, train_docs, test_docs, max_tokens_train, max_tokens_test
):
    """Matched-pair activation patching: causal localization of the observer signal.

    Finds token positions where the observer scores differ substantially
    (high-score vs low-score positions within the same documents), then
    patches component outputs from high-score contexts into low-score
    contexts. Measures both observer score change and loss change.

    If patching a component causes the observer score to increase AND
    loss to decrease in the recipient context, that component causally
    carries decision-quality information.
    """
    peak_layer = 8

    print(f"\n{'=' * 60}")
    print(f"  Matched-pair activation patching (layers 0-{peak_layer})")
    print(f"{'=' * 60}")

    # Train observer
    print("  Training observer head...")
    train_data = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, max_tokens_train)
    observer = train_linear_binary(train_data, seed=42)
    observer.eval()

    # Collect document-level data: for each document, get per-position observer scores and losses
    print("  Collecting paired data from test documents...")
    eval_docs = test_docs[:200]
    doc_data = []

    model.eval()
    for doc in eval_docs:
        if not doc.strip():
            continue
        tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(device)
        if input_ids.size(1) < 4:
            continue

        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)

        h = outputs.hidden_states[peak_layer + 1][0, :-1, :].cpu().float()
        scores = observer(h).squeeze(-1).detach().numpy()
        losses = (
            F.cross_entropy(outputs.logits[0, :-1, :].float(), input_ids[0, 1:], reduction="none")
            .cpu()
            .numpy()
        )

        doc_data.append({"input_ids": input_ids, "scores": scores, "losses": losses, "n_pos": len(scores)})

    # Build matched pairs: high-score and low-score positions from same document
    # Use documents with enough positions and score variance
    pairs = []
    for dd in doc_data:
        if dd["n_pos"] < 10:
            continue
        scores = dd["scores"]
        p20 = np.percentile(scores, 20)
        p80 = np.percentile(scores, 80)
        low_idx = np.where(scores <= p20)[0]
        high_idx = np.where(scores >= p80)[0]
        if len(low_idx) >= 2 and len(high_idx) >= 2:
            pairs.append(
                {
                    "input_ids": dd["input_ids"],
                    "low_positions": low_idx,
                    "high_positions": high_idx,
                    "low_mean_score": float(scores[low_idx].mean()),
                    "high_mean_score": float(scores[high_idx].mean()),
                    "low_mean_loss": float(dd["losses"][low_idx].mean()),
                    "high_mean_loss": float(dd["losses"][high_idx].mean()),
                }
            )

    print(f"    {len(pairs)} documents with valid high/low pairs")
    if not pairs:
        print("    No valid pairs found")
        return {"error": "no valid pairs"}

    # For each component, patch high-score component outputs into the full forward pass
    component_results = _run_matched_component_sweep(model, observer, pairs, peak_layer)

    # Summary
    print("\n  CAUSAL COMPONENTS (obs increase + loss decrease):")
    causal_found = False
    for layer, comps in sorted(component_results.items()):
        for comp, vals in comps.items():
            if vals["interp"] == "CAUSAL +":
                print(
                    f"    layer {layer} {comp}: obs={vals['obs_delta']:+.4f} loss={vals['loss_delta']:+.4f}"
                )
                causal_found = True
    if not causal_found:
        print("    None found with dual-metric criteria")

    return {
        "peak_layer": peak_layer,
        "n_pairs": len(pairs),
        "component_results": {str(k): v for k, v in component_results.items()},
    }


def run_activation_patching(
    model, tokenizer, device, train_docs, test_docs, max_tokens_train, max_tokens_test
):
    """Activation patching: which components causally produce the observer signal?

    For each attention layer and MLP at layers 0 through peak_layer, zero
    out the component's contribution to the residual stream and measure
    the change in observer score. Components with the largest effect are
    causally responsible for the signal.

    Then do head-level patching at the top-contributing layers.
    """
    peak_layer = 8
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    print(f"\n{'=' * 60}")
    print(f"  Activation patching (layers 0-{peak_layer})")
    print(f"{'=' * 60}")

    # Train observer head
    print("  Training observer head at layer 8...")
    train_data = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, max_tokens_train)
    head = train_linear_binary(train_data, seed=42)
    head.eval()

    # Select a subset of test documents for patching (forward passes are expensive)
    eval_docs = test_docs[:100]
    eval_budget = 20000

    # Baseline: observer scores without intervention
    print("  Computing baseline observer scores...")
    baseline_scores = _collect_observer_scores(
        model, tokenizer, head, peak_layer, eval_docs, device, eval_budget
    )
    n_eval = len(baseline_scores)
    print(f"    {n_eval} positions, mean score: {baseline_scores.mean():.4f}")

    # --- Component-level patching (attention + MLP per layer) ---
    print(f"\n  Component-level patching (layers 0-{peak_layer}):")
    component_effects = {}

    for layer in range(peak_layer + 1):
        # Zero attention output at this layer
        attn_scores = _patch_component(
            model,
            tokenizer,
            head,
            peak_layer,
            eval_docs,
            device,
            eval_budget,
            target_layer=layer,
            component="attn",
        )
        attn_delta = float((attn_scores - baseline_scores).mean())

        # Zero MLP output at this layer
        mlp_scores = _patch_component(
            model,
            tokenizer,
            head,
            peak_layer,
            eval_docs,
            device,
            eval_budget,
            target_layer=layer,
            component="mlp",
        )
        mlp_delta = float((mlp_scores - baseline_scores).mean())

        component_effects[layer] = {"attn": attn_delta, "mlp": mlp_delta}
        print(f"    layer {layer:>2}: attn={attn_delta:+.4f}  mlp={mlp_delta:+.4f}")

    # Find top-contributing layers
    attn_effects = [(l, e["attn"]) for l, e in component_effects.items()]
    mlp_effects = [(l, e["mlp"]) for l, e in component_effects.items()]
    top_attn_layer = max(attn_effects, key=lambda x: abs(x[1]))
    top_mlp_layer = max(mlp_effects, key=lambda x: abs(x[1]))
    print(f"\n    Top attention: layer {top_attn_layer[0]} ({top_attn_layer[1]:+.4f})")
    print(f"    Top MLP: layer {top_mlp_layer[0]} ({top_mlp_layer[1]:+.4f})")

    # --- Head-level patching at top attention layers ---
    head_layers_to_probe = sorted(set([top_attn_layer[0], peak_layer]))
    head_effects = {}

    for layer in head_layers_to_probe:
        print(f"\n  Head-level patching at layer {layer} ({n_heads} heads):")
        layer_head_effects = {}
        for h_idx in range(n_heads):
            patched_scores = _patch_head(
                model,
                tokenizer,
                head,
                peak_layer,
                eval_docs,
                device,
                eval_budget,
                target_layer=layer,
                head_idx=h_idx,
                n_heads=n_heads,
                head_dim=head_dim,
            )
            delta = float((patched_scores - baseline_scores).mean())
            layer_head_effects[h_idx] = delta
            print(f"    head {h_idx:>2}: {delta:+.4f}")

        head_effects[layer] = layer_head_effects

        # Top heads
        sorted_heads = sorted(layer_head_effects.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"    Top 3: {[(h, f'{d:+.4f}') for h, d in sorted_heads[:3]]}")

    # Summary
    print("\n  ACTIVATION PATCHING SUMMARY")
    print(f"  {'Layer':<8} {'Attn effect':>12} {'MLP effect':>12} {'Total':>12}")
    print(f"  {'-' * 46}")
    for layer in range(peak_layer + 1):
        a = component_effects[layer]["attn"]
        m = component_effects[layer]["mlp"]
        print(f"  {layer:<8} {a:>+12.4f} {m:>+12.4f} {a + m:>+12.4f}")

    return {
        "peak_layer": peak_layer,
        "n_eval_positions": n_eval,
        "component_effects": {str(k): v for k, v in component_effects.items()},
        "head_effects": {str(k): {str(h): d for h, d in v.items()} for k, v in head_effects.items()},
    }


def _collect_observer_scores(model, tokenizer, head, peak_layer, docs, device, max_tokens):
    """Collect observer scores from test documents."""
    all_scores = []
    total = 0
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            if total >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[peak_layer + 1][0, :-1, :].cpu().float()
            scores = head(h).squeeze(-1).numpy()
            all_scores.append(scores)
            total += len(scores)
    return np.concatenate(all_scores)[:max_tokens]


def _patch_component(model, tokenizer, head, peak_layer, docs, device, max_tokens, target_layer, component):
    """Zero out attention or MLP output at target_layer, collect observer scores at peak_layer."""

    def make_hook(comp):
        def hook_fn(module, input, output):
            if comp == "attn":
                # GPT-2 attention returns (attn_output, present, attn_weights) or (attn_output,)
                if isinstance(output, tuple):
                    zeroed = torch.zeros_like(output[0])
                    return (zeroed,) + output[1:]
                return torch.zeros_like(output)
            else:  # mlp
                return torch.zeros_like(output)

        return hook_fn

    block = model.transformer.h[target_layer]
    if component == "attn":
        handle = block.attn.register_forward_hook(make_hook("attn"))
    else:
        handle = block.mlp.register_forward_hook(make_hook("mlp"))

    scores = _collect_observer_scores(model, tokenizer, head, peak_layer, docs, device, max_tokens)
    handle.remove()
    return scores


def _patch_head(
    model,
    tokenizer,
    observer_head,
    peak_layer,
    docs,
    device,
    max_tokens,
    target_layer,
    head_idx,
    n_heads,
    head_dim,
):
    """Zero a single attention head via c_proj pre-hook (zeroes its slice before projection)."""
    block = model.transformer.h[target_layer]
    start = head_idx * head_dim
    end = start + head_dim

    def hook_fn(module, inp):
        x = inp[0].clone()
        x[:, :, start:end] = 0.0
        return (x,)

    handle = block.attn.c_proj.register_forward_pre_hook(hook_fn)
    scores = _collect_observer_scores(model, tokenizer, observer_head, peak_layer, docs, device, max_tokens)
    handle.remove()
    return scores


def run_mechanistic_analysis(
    model, tokenizer, device, train_docs, test_docs, max_tokens_train, max_tokens_test
):
    """Best-practice mechanistic analysis of the observer signal.

    Three improvements over the initial patching:
    1. Mean ablation (replace with dataset mean) instead of zero ablation
    2. Residualized observer metric (partial out confidence before measuring delta)
    3. Composition tests (patch layer groups, check additivity)

    Uses the activation-patching best-practices recommendations:
    - Mean ablation preserves activation statistics
    - Multiple metrics (residualized observer score + raw loss)
    - Aggregate over many positions, not single examples
    """
    peak_layer = 8
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads

    print("\n" + "=" * 60)
    print("  Mechanistic analysis (best-practice patching)")
    print("=" * 60)

    # Collect data and train observer
    print("  Collecting train data at layer 8...")
    train_data = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, max_tokens_train)
    observer = train_linear_binary(train_data, seed=42)
    observer.eval()

    # Compute mean component outputs across training data (for mean ablation)
    print("  Computing mean component activations...")
    eval_docs = test_docs[:150]
    eval_budget = 30000
    component_means = _compute_component_means(model, tokenizer, eval_docs, device, peak_layer, eval_budget)

    # Baseline: collect per-position observer scores, losses, and confidence
    print("  Computing baseline scores...")
    baseline = _collect_full_baseline(model, tokenizer, observer, peak_layer, eval_docs, device, eval_budget)
    n_eval = baseline["n"]
    print(f"    {n_eval} positions")

    # --- Part 1: Mean ablation per component ---
    print("\n  Part 1: Mean ablation (layers 0-8)")
    print(f"  {'Layer':<8} {'Comp':<6} {'Obs resid delta':>16} {'Loss delta':>12} {'Raw obs delta':>15}")
    print(f"  {'-' * 59}")

    ablation_results = {}
    for layer in range(peak_layer + 1):
        layer_r = {}
        for comp in ["attn", "mlp"]:
            patched = _mean_ablate_component(
                model,
                tokenizer,
                observer,
                peak_layer,
                eval_docs,
                device,
                eval_budget,
                layer,
                comp,
                component_means,
            )
            # Residualized observer delta: partial out confidence change
            obs_resid_delta = _residualized_delta(
                baseline["obs_scores"], patched["obs_scores"], baseline["max_softmax"], patched["max_softmax"]
            )
            loss_delta = float(patched["losses"].mean() - baseline["losses"].mean())
            raw_obs_delta = float(patched["obs_scores"].mean() - baseline["obs_scores"].mean())

            layer_r[comp] = {
                "obs_resid_delta": obs_resid_delta,
                "loss_delta": loss_delta,
                "raw_obs_delta": raw_obs_delta,
            }
            print(
                f"  {layer:<8} {comp:<6} {obs_resid_delta:>+16.4f} {loss_delta:>+12.4f} {raw_obs_delta:>+15.4f}"
            )

        ablation_results[layer] = layer_r

    # --- Part 2: Composition tests ---
    print("\n  Part 2: Composition tests")
    groups = {
        "attn_5_6": [(5, "attn"), (6, "attn")],
        "attn_7_8": [(7, "attn"), (8, "attn")],
        "attn_5_8": [(5, "attn"), (6, "attn"), (7, "attn"), (8, "attn")],
        "mlp_3_4": [(3, "mlp"), (4, "mlp")],
        "all_mid": [(5, "attn"), (6, "attn"), (7, "attn"), (8, "attn"), (3, "mlp"), (4, "mlp")],
    }

    composition_results = {}
    for gname, components in groups.items():
        patched = _mean_ablate_group(
            model,
            tokenizer,
            observer,
            peak_layer,
            eval_docs,
            device,
            eval_budget,
            components,
            component_means,
        )
        obs_resid = _residualized_delta(
            baseline["obs_scores"], patched["obs_scores"], baseline["max_softmax"], patched["max_softmax"]
        )
        loss_d = float(patched["losses"].mean() - baseline["losses"].mean())

        # Expected from individual ablations (additivity check)
        expected = sum(ablation_results[l][c]["obs_resid_delta"] for l, c in components)

        composition_results[gname] = {
            "obs_resid_delta": obs_resid,
            "loss_delta": loss_d,
            "expected_additive": expected,
            "interaction": obs_resid - expected,
        }
        print(
            f"    {gname:<14}: obs_resid={obs_resid:+.4f}  loss={loss_d:+.4f}  "
            f"expected={expected:+.4f}  interaction={obs_resid - expected:+.4f}"
        )

    # --- Part 3: Head-level at top attention layers ---
    top_attn = sorted(
        [(l, ablation_results[l]["attn"]["obs_resid_delta"]) for l in range(peak_layer + 1)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    probe_layers = sorted(set([top_attn[0][0], top_attn[1][0], peak_layer]))

    head_results = {}
    for layer in probe_layers:
        print(f"\n  Part 3: Head-level mean ablation at layer {layer}")
        block = model.transformer.h[layer]
        layer_heads = {}

        for h_idx in range(n_heads):
            start = h_idx * head_dim
            end = start + head_dim

            # Compute mean for this head's slice
            head_mean = component_means[(layer, "attn")][:, :, start:end].mean(dim=0, keepdim=True)

            def make_hook(s, e, hm):
                def hook_fn(module, inp):
                    x = inp[0].clone()
                    x[:, :, s:e] = hm.to(x.device)
                    return (x,)

                return hook_fn

            handle = block.attn.c_proj.register_forward_pre_hook(make_hook(start, end, head_mean))
            patched = _collect_full_baseline(
                model, tokenizer, observer, peak_layer, eval_docs, device, eval_budget
            )
            handle.remove()

            obs_resid = _residualized_delta(
                baseline["obs_scores"], patched["obs_scores"], baseline["max_softmax"], patched["max_softmax"]
            )
            loss_d = float(patched["losses"].mean() - baseline["losses"].mean())
            layer_heads[h_idx] = {"obs_resid_delta": obs_resid, "loss_delta": loss_d}

        head_results[layer] = layer_heads
        sorted_h = sorted(layer_heads.items(), key=lambda x: abs(x[1]["obs_resid_delta"]), reverse=True)
        for h_idx, vals in sorted_h:
            print(
                f"    head {h_idx:>2}: obs_resid={vals['obs_resid_delta']:+.4f}  loss={vals['loss_delta']:+.4f}"
            )
        top3 = [(h, round(v["obs_resid_delta"], 4)) for h, v in sorted_h[:3]]
        print(f"    Top 3: {top3}")

    # --- Summary ---
    print("\n  MECHANISTIC ANALYSIS SUMMARY")
    print(f"  {'Layer':<8} {'Attn obs_resid':>15} {'MLP obs_resid':>15} {'Attn loss':>11} {'MLP loss':>11}")
    print(f"  {'-' * 62}")
    for layer in range(peak_layer + 1):
        ar = ablation_results[layer]
        print(
            f"  {layer:<8} {ar['attn']['obs_resid_delta']:>+15.4f} {ar['mlp']['obs_resid_delta']:>+15.4f}"
            f" {ar['attn']['loss_delta']:>+11.4f} {ar['mlp']['loss_delta']:>+11.4f}"
        )

    print("\n  Composition:")
    for gname, vals in composition_results.items():
        additive = (
            "additive" if abs(vals["interaction"]) < 0.01 else f"interaction={vals['interaction']:+.4f}"
        )
        print(f"    {gname:<14}: {vals['obs_resid_delta']:+.4f} ({additive})")

    return {
        "peak_layer": peak_layer,
        "n_eval": n_eval,
        "ablation_results": {str(k): v for k, v in ablation_results.items()},
        "composition_results": composition_results,
        "head_results": {str(k): {str(h): v for h, v in heads.items()} for k, heads in head_results.items()},
    }


def _compute_component_means(model, tokenizer, docs, device, max_layer, max_tokens):
    """Collect mean activation for each attention and MLP output at each layer."""
    means = {}
    accumulators = {}

    def make_accumulator_hook(key):
        def hook_fn(module, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            if key not in accumulators:
                accumulators[key] = []
            accumulators[key].append(val.detach().cpu())
            return out

        return hook_fn

    handles = []
    for layer in range(max_layer + 1):
        block = model.transformer.h[layer]
        handles.append(block.attn.register_forward_hook(make_accumulator_hook((layer, "attn"))))
        handles.append(block.mlp.register_forward_hook(make_accumulator_hook((layer, "mlp"))))

    total = 0
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            if total >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            model(input_ids)
            total += input_ids.size(1) - 1

    for h in handles:
        h.remove()

    for key, tensors in accumulators.items():
        cat = torch.cat(tensors, dim=1)
        means[key] = cat.mean(dim=1, keepdim=True)  # [1, 1, hidden_dim]

    return means


def _collect_full_baseline(model, tokenizer, observer, peak_layer, docs, device, max_tokens):
    """Collect observer scores, losses, and confidence for all positions."""
    all_obs, all_loss, all_sm = [], [], []
    total = 0
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            if total >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids, output_hidden_states=True)
            h = outputs.hidden_states[peak_layer + 1][0, :-1, :].cpu().float()
            scores = observer(h).squeeze(-1).detach().numpy()
            losses = (
                F.cross_entropy(outputs.logits[0, :-1, :].float(), input_ids[0, 1:], reduction="none")
                .cpu()
                .numpy()
            )
            sm = F.softmax(outputs.logits[0, :-1, :].float(), dim=-1).max(dim=-1).values.cpu().numpy()
            all_obs.append(scores)
            all_loss.append(losses)
            all_sm.append(sm)
            total += len(scores)

    return {
        "obs_scores": np.concatenate(all_obs)[:max_tokens],
        "losses": np.concatenate(all_loss)[:max_tokens],
        "max_softmax": np.concatenate(all_sm)[:max_tokens],
        "n": min(total, max_tokens),
    }


def _mean_ablate_component(
    model, tokenizer, observer, peak_layer, docs, device, max_tokens, target_layer, component, means
):
    """Replace a component's output with its dataset mean, collect scores."""
    mean_val = means[(target_layer, component)]
    block = model.transformer.h[target_layer]
    target = block.attn if component == "attn" else block.mlp

    def hook_fn(module, inp, out):
        m = mean_val.to(out[0].device if isinstance(out, tuple) else out.device)
        expanded = m.expand_as(out[0] if isinstance(out, tuple) else out)
        if isinstance(out, tuple):
            return (expanded,) + out[1:]
        return expanded

    handle = target.register_forward_hook(hook_fn)
    result = _collect_full_baseline(model, tokenizer, observer, peak_layer, docs, device, max_tokens)
    handle.remove()
    return result


def _mean_ablate_group(model, tokenizer, observer, peak_layer, docs, device, max_tokens, components, means):
    """Mean-ablate multiple components simultaneously."""
    handles = []
    for layer, comp in components:
        mean_val = means[(layer, comp)]
        block = model.transformer.h[layer]
        target = block.attn if comp == "attn" else block.mlp

        def make_hook(mv):
            def hook_fn(module, inp, out):
                m = mv.to(out[0].device if isinstance(out, tuple) else out.device)
                expanded = m.expand_as(out[0] if isinstance(out, tuple) else out)
                if isinstance(out, tuple):
                    return (expanded,) + out[1:]
                return expanded

            return hook_fn

        handles.append(target.register_forward_hook(make_hook(mean_val)))

    result = _collect_full_baseline(model, tokenizer, observer, peak_layer, docs, device, max_tokens)
    for h in handles:
        h.remove()
    return result


def _residualized_delta(base_obs, patched_obs, base_sm, patched_sm):
    """Compute the change in observer score after partialling out confidence change.

    If confidence shifts under patching, raw observer score change conflates
    the observer-specific effect with the confidence-mediated effect.
    This removes the confidence-mediated component.
    """
    # Estimate how much of the observer change is explained by confidence change
    sm_delta = patched_sm - base_sm
    obs_delta = patched_obs - base_obs

    # Regress obs_delta on sm_delta to get the confidence-explained component
    if np.std(sm_delta) > 1e-8:
        beta = np.cov(obs_delta, sm_delta)[0, 1] / np.var(sm_delta)
        residualized = obs_delta - beta * sm_delta
    else:
        residualized = obs_delta

    return float(residualized.mean())


# ---------------------------------------------------------------------------
# Model-agnostic mechanistic analysis (generalizes run_mechanistic_analysis)
# ---------------------------------------------------------------------------


def _compute_component_means_general(model, tokenizer, docs, device, max_layer, max_tokens):
    """Collect mean activation for each attn and MLP output, architecture-agnostic."""
    accumulators = {}

    def make_accumulator_hook(key):
        def hook_fn(module, inp, out):
            val = out[0] if isinstance(out, tuple) else out
            if key not in accumulators:
                accumulators[key] = []
            accumulators[key].append(val.detach().cpu())
            return out

        return hook_fn

    handles = []
    for layer in range(max_layer + 1):
        attn, mlp = _get_layer_modules(model, layer)
        handles.append(attn.register_forward_hook(make_accumulator_hook((layer, "attn"))))
        handles.append(mlp.register_forward_hook(make_accumulator_hook((layer, "mlp"))))

    total = 0
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            if total >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            model(input_ids)
            total += input_ids.size(1) - 1

    for h in handles:
        h.remove()

    means = {}
    for key, tensors in accumulators.items():
        cat = torch.cat(tensors, dim=1)
        means[key] = cat.mean(dim=1, keepdim=True)  # [1, 1, hidden_dim]

    return means


def _mean_ablate_component_general(
    model, tokenizer, observer, peak_layer, docs, device, max_tokens, target_layer, component, means
):
    """Replace a component's output with its dataset mean, architecture-agnostic."""
    mean_val = means[(target_layer, component)]
    attn, mlp = _get_layer_modules(model, target_layer)
    target = attn if component == "attn" else mlp

    def hook_fn(module, inp, out):
        m = mean_val.to(out[0].device if isinstance(out, tuple) else out.device)
        expanded = m.expand_as(out[0] if isinstance(out, tuple) else out)
        if isinstance(out, tuple):
            return (expanded,) + out[1:]
        return expanded

    handle = target.register_forward_hook(hook_fn)
    result = _collect_full_baseline(model, tokenizer, observer, peak_layer, docs, device, max_tokens)
    handle.remove()
    return result


def _mean_ablate_group_general(
    model, tokenizer, observer, peak_layer, docs, device, max_tokens, components, means
):
    """Mean-ablate multiple components simultaneously, architecture-agnostic."""
    handles = []
    for layer, comp in components:
        mean_val = means[(layer, comp)]
        attn, mlp = _get_layer_modules(model, layer)
        target = attn if comp == "attn" else mlp

        def make_hook(mv):
            def hook_fn(module, inp, out):
                m = mv.to(out[0].device if isinstance(out, tuple) else out.device)
                expanded = m.expand_as(out[0] if isinstance(out, tuple) else out)
                if isinstance(out, tuple):
                    return (expanded,) + out[1:]
                return expanded

            return hook_fn

        handles.append(target.register_forward_hook(make_hook(mean_val)))

    result = _collect_full_baseline(model, tokenizer, observer, peak_layer, docs, device, max_tokens)
    for h in handles:
        h.remove()
    return result


def run_mechanistic_general(
    model,
    tokenizer,
    device,
    train_docs,
    test_docs,
    max_tokens_train,
    max_tokens_test,
    peak_layer,
    eval_budget=15000,
    skip_heads=True,
):
    """Model-agnostic mechanistic analysis via mean-ablation patching.

    Same methodology as run_mechanistic_analysis but works on any HF causal LM.
    Composition groups are computed from Part 1 results rather than hardcoded.
    Head-level analysis is skipped by default (expensive at 7B+).
    """
    print("\n" + "=" * 60)
    print("  Mechanistic analysis (model-agnostic)")
    print(f"  Peak layer: {peak_layer}, eval budget: {eval_budget}")
    print("=" * 60)

    # Train observer at peak layer
    print(f"  Collecting train data at layer {peak_layer}...")
    train_data = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, max_tokens_train)
    observer = train_linear_binary(train_data, seed=42)
    observer.eval()

    # Compute mean component outputs
    print("  Computing mean component activations...")
    eval_docs = test_docs[:200]
    component_means = _compute_component_means_general(
        model, tokenizer, eval_docs, device, peak_layer, eval_budget
    )

    # Baseline scores
    print("  Computing baseline scores...")
    baseline = _collect_full_baseline(model, tokenizer, observer, peak_layer, eval_docs, device, eval_budget)
    n_eval = baseline["n"]
    print(f"    {n_eval} positions")

    # --- Part 1: Per-component mean ablation ---
    print(f"\n  Part 1: Mean ablation (layers 0-{peak_layer})")
    print(f"  {'Layer':<8} {'Comp':<6} {'Obs resid delta':>16} {'Loss delta':>12} {'Raw obs delta':>15}")
    print(f"  {'-' * 59}")

    ablation_results = {}
    for layer in range(peak_layer + 1):
        layer_r = {}
        for comp in ["attn", "mlp"]:
            patched = _mean_ablate_component_general(
                model,
                tokenizer,
                observer,
                peak_layer,
                eval_docs,
                device,
                eval_budget,
                layer,
                comp,
                component_means,
            )
            obs_resid_delta = _residualized_delta(
                baseline["obs_scores"],
                patched["obs_scores"],
                baseline["max_softmax"],
                patched["max_softmax"],
            )
            loss_delta = float(patched["losses"].mean() - baseline["losses"].mean())
            raw_obs_delta = float(patched["obs_scores"].mean() - baseline["obs_scores"].mean())

            layer_r[comp] = {
                "obs_resid_delta": obs_resid_delta,
                "loss_delta": loss_delta,
                "raw_obs_delta": raw_obs_delta,
            }
            print(
                f"  {layer:<8} {comp:<6} {obs_resid_delta:>+16.4f} {loss_delta:>+12.4f} {raw_obs_delta:>+15.4f}"
            )

        ablation_results[layer] = layer_r

    # --- Part 2: Composition tests (data-driven groups) ---
    print("\n  Part 2: Composition tests")

    # Rank layers by absolute residualized delta
    attn_ranked = sorted(
        [(l, ablation_results[l]["attn"]["obs_resid_delta"]) for l in range(peak_layer + 1)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    mlp_ranked = sorted(
        [(l, ablation_results[l]["mlp"]["obs_resid_delta"]) for l in range(peak_layer + 1)],
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top2_attn = sorted([attn_ranked[0][0], attn_ranked[1][0]])
    top4_attn = sorted([r[0] for r in attn_ranked[:4]])
    top2_mlp = sorted([mlp_ranked[0][0], mlp_ranked[1][0]])

    groups = {
        f"attn_{top2_attn[0]}_{top2_attn[1]}": [(l, "attn") for l in top2_attn],
        "attn_top4": [(l, "attn") for l in top4_attn],
        f"mlp_{top2_mlp[0]}_{top2_mlp[1]}": [(l, "mlp") for l in top2_mlp],
        "all_top": [(l, "attn") for l in top4_attn] + [(l, "mlp") for l in top2_mlp],
    }

    composition_results = {}
    for gname, components in groups.items():
        patched = _mean_ablate_group_general(
            model,
            tokenizer,
            observer,
            peak_layer,
            eval_docs,
            device,
            eval_budget,
            components,
            component_means,
        )
        obs_resid = _residualized_delta(
            baseline["obs_scores"],
            patched["obs_scores"],
            baseline["max_softmax"],
            patched["max_softmax"],
        )
        loss_d = float(patched["losses"].mean() - baseline["losses"].mean())
        expected = sum(ablation_results[l][c]["obs_resid_delta"] for l, c in components)

        composition_results[gname] = {
            "obs_resid_delta": obs_resid,
            "loss_delta": loss_d,
            "expected_additive": expected,
            "interaction": obs_resid - expected,
            "components": [[l, c] for l, c in components],
        }
        print(
            f"    {gname:<18}: obs_resid={obs_resid:+.4f}  loss={loss_d:+.4f}  "
            f"expected={expected:+.4f}  interaction={obs_resid - expected:+.4f}"
        )

    # --- Summary ---
    print(f"\n  MECHANISTIC ANALYSIS SUMMARY (peak layer {peak_layer})")
    print(f"  {'Layer':<8} {'Attn obs_resid':>15} {'MLP obs_resid':>15} {'Attn loss':>11} {'MLP loss':>11}")
    print(f"  {'-' * 62}")
    for layer in range(peak_layer + 1):
        ar = ablation_results[layer]
        print(
            f"  {layer:<8} {ar['attn']['obs_resid_delta']:>+15.4f} {ar['mlp']['obs_resid_delta']:>+15.4f}"
            f" {ar['attn']['loss_delta']:>+11.4f} {ar['mlp']['loss_delta']:>+11.4f}"
        )

    print("\n  Composition:")
    for gname, vals in composition_results.items():
        additive = (
            "additive" if abs(vals["interaction"]) < 0.01 else f"interaction={vals['interaction']:+.4f}"
        )
        print(f"    {gname:<18}: {vals['obs_resid_delta']:+.4f} ({additive})")

    # Top attention layers for comparison with GPT-2
    top_attn_layers = [r[0] for r in attn_ranked[:3]]
    n_layers = model.config.num_hidden_layers
    depth_range = f"{min(top_attn_layers) / n_layers:.2f}-{max(top_attn_layers) / n_layers:.2f}"

    return {
        "peak_layer": peak_layer,
        "n_eval": n_eval,
        "ablation_results": {str(k): v for k, v in ablation_results.items()},
        "composition_results": composition_results,
        "top_attn_layers": top_attn_layers,
        "top_attn_depth_range": depth_range,
    }


def run_8c(model, tokenizer, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test):
    """8c: Cross-domain transfer.

    Train observer heads on WikiText at layer 8, evaluate on WikiText,
    OpenWebText, and code. Tests whether the signal is a general property
    of the residual stream or task-specific.
    """
    layer = 8
    print(f"\n{'=' * 60}")
    print(f"  8c: Cross-domain transfer at layer {layer}")
    print(f"{'=' * 60}")

    # Train observer heads on WikiText
    print("  Collecting WikiText train activations (layer 8)...")
    train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)

    heads = []
    for seed in seeds:
        heads.append(train_linear_binary(train_data, seed=seed))

    # Evaluate on each domain
    eval_domains = {
        "wikitext": test_docs,
        "openwebtext": None,  # loaded on demand
        "code": None,
    }

    results = {
        "train_domain": "wikitext",
        "layer": layer,
        "n_seeds": len(seeds),
        "domains": {},
        "seed_agreement": {},
    }

    for domain_name, docs in eval_domains.items():
        print(f"\n  --- Domain: {domain_name} ---")
        if docs is None:
            print(f"  Loading {domain_name}...")
            docs = load_domain(domain_name, "test", max_docs=500)
            print(f"    loaded {len(docs)} documents")

        test_data = collect_layer_data(model, tokenizer, docs, layer, device, max_tokens_test)

        domain_rhos = []
        domain_scores = []
        for seed, head in zip(seeds, heads):
            scores, rho, p = evaluate_head(head, test_data)
            domain_rhos.append(float(rho))
            domain_scores.append(scores)
            print(f"    seed {seed}: partial corr = {rho:+.4f}")

        # Seed agreement within this domain
        pairwise = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                r, _ = spearmanr(domain_scores[i], domain_scores[j])
                pairwise.append(float(r))

        mean_rho = float(np.mean(domain_rhos))
        ci = bootstrap_ci(domain_rhos) if len(domain_rhos) >= 3 else (mean_rho, mean_rho)
        mean_agree = float(np.mean(pairwise)) if pairwise else 0.0

        results["domains"][domain_name] = {
            "n_tokens": len(test_data["losses"]),
            "partial_corrs": domain_rhos,
            "mean": mean_rho,
            "ci_95": list(ci),
        }
        results["seed_agreement"][domain_name] = mean_agree
        print(f"    mean: {mean_rho:+.4f}  seed agreement: {mean_agree:+.4f}")

    # Transfer ratios
    wiki_mean = results["domains"]["wikitext"]["mean"]
    results["transfer_ratios"] = {}
    print("\n  8c RESULT:")
    print(f"  {'Domain':<15} {'Partial corr':>14} {'Transfer ratio':>16} {'Seed agreement':>16}")
    print(f"  {'-' * 63}")
    for domain_name in results["domains"]:
        d = results["domains"][domain_name]
        if domain_name == "wikitext":
            ratio_str = "(reference)"
            print(
                f"  {domain_name:<15} {d['mean']:>+14.4f} {ratio_str:>16} {results['seed_agreement'][domain_name]:>+16.4f}"
            )
        else:
            ratio = d["mean"] / wiki_mean if wiki_mean != 0 else 0.0
            results["transfer_ratios"][domain_name] = float(ratio)
            print(
                f"  {domain_name:<15} {d['mean']:>+14.4f} {ratio:>16.2f} {results['seed_agreement'][domain_name]:>+16.4f}"
            )

    return results


# ---------------------------------------------------------------------------
# Domain loaders
# ---------------------------------------------------------------------------


def load_openwebtext(split="test", max_docs=None):
    """Load OpenWebText documents (streamed from HuggingFace)."""
    from datasets import load_dataset

    ds = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)
    docs = []
    skip = 100000 if split == "test" else 0
    for i, row in enumerate(ds):
        if i < skip:
            continue
        text = row["text"].strip()
        if len(text) > 100:
            docs.append(text)
        if max_docs and len(docs) >= max_docs:
            break
    return docs


def load_code_dataset(split="test", max_docs=None):
    """Load Python code documents from CodeSearchNet."""
    from datasets import load_dataset

    ds = load_dataset("code_search_net", "python", split="test", streaming=True, trust_remote_code=True)
    docs = []
    for row in ds:
        text = row["whole_func_string"].strip()
        if len(text) > 100:
            docs.append(text)
        if max_docs and len(docs) >= max_docs:
            break
    return docs


DOMAIN_LOADERS = {
    "wikitext": load_wikitext,
    "openwebtext": load_openwebtext,
    "code": load_code_dataset,
}


def load_domain(domain, split, max_docs=None):
    """Load documents from a named domain."""
    if domain not in DOMAIN_LOADERS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(DOMAIN_LOADERS.keys())}")
    return DOMAIN_LOADERS[domain](split, max_docs)


# ---------------------------------------------------------------------------
# Scale characterization
# ---------------------------------------------------------------------------

GPT2_MODELS = [
    ("gpt2", 124),
    ("gpt2-medium", 355),
    ("gpt2-large", 774),
    ("gpt2-xl", 1558),
]


def _coarse_layer_sweep(
    model, tokenizer, device, train_docs, val_docs, n_layers, max_tokens_train, max_tokens_val
):
    """Sweep layers with 1 seed to find peak partial correlation.

    Layer selection uses val_docs (not test_docs) to avoid selection bias.
    Callers report final numbers on a held-out test split.

    For models with many layers, sweep every Nth layer first (coarse),
    then sweep densely around the peak region.
    """
    stride = max(1, n_layers // 12)
    coarse_layers = list(range(0, n_layers, stride))
    if (n_layers - 1) not in coarse_layers:
        coarse_layers.append(n_layers - 1)

    profile = {}
    print(f"  Coarse sweep: {len(coarse_layers)} layers (stride {stride}) [on validation split]")
    for layer in coarse_layers:
        train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
        val_data = collect_layer_data(model, tokenizer, val_docs, layer, device, max_tokens_val)
        head = train_linear_binary(train_data, seed=42)
        _, rho, _ = evaluate_head(head, val_data)
        profile[layer] = float(rho)
        print(f"    layer {layer:>3}: {rho:+.4f}")

    coarse_peak = max(profile, key=profile.get)

    # Dense sweep around peak if stride > 1
    if stride > 1:
        dense_lo = max(0, coarse_peak - stride)
        dense_hi = min(n_layers - 1, coarse_peak + stride)
        dense_layers = [l for l in range(dense_lo, dense_hi + 1) if l not in profile]
        if dense_layers:
            print(f"  Dense sweep: layers {dense_lo}-{dense_hi}")
            for layer in dense_layers:
                train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
                val_data = collect_layer_data(model, tokenizer, val_docs, layer, device, max_tokens_val)
                head = train_linear_binary(train_data, seed=42)
                _, rho, _ = evaluate_head(head, val_data)
                profile[layer] = float(rho)
                print(f"    layer {layer:>3}: {rho:+.4f}")

    peak_layer = max(profile, key=profile.get)
    return peak_layer, profile


def run_scale(
    device,
    seeds,
    train_docs,
    val_docs,
    test_docs,
    max_tokens_train,
    max_tokens_val,
    max_tokens_test,
    model_names=None,
):
    """Scale characterization across GPT-2 family.

    For each model: coarse layer sweep on val_docs to find peak, full battery
    at peak on held-out test_docs (partial correlation + seed agreement),
    output-controlled residual. Models loaded and unloaded sequentially to
    manage memory.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if model_names is None:
        model_names = [m[0] for m in GPT2_MODELS]

    models_to_run = [(name, params) for name, params in GPT2_MODELS if name in model_names]

    print(f"\n{'=' * 60}")
    print("  Scale characterization")
    print(f"  Models: {[m[0] for m in models_to_run]}")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 60}")

    all_results = {}

    for model_id, n_params_m in models_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_id} ({n_params_m}M)")
        print(f"{'=' * 60}")

        # Load model
        print(f"  Loading {model_id}...")
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        model.eval()

        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        output_layer = n_layers - 1
        print(f"  {n_layers} layers, hidden dim {hidden_dim}")

        # Scale token counts inversely with hidden dim to keep memory bounded
        # Scale token budget to maintain examples-per-dimension ratio.
        # GPT-2 baseline: 200k tokens / 768 dim = ~260 ex/dim.
        # Floor at 150 ex/dim for credible linear probe fitting.
        min_ex_per_dim = 150
        min_train = min_ex_per_dim * hidden_dim
        adj_train = max(min_train, int(max_tokens_train * (768 / hidden_dim)))
        adj_val = max(min_train // 2, int(max_tokens_val * (768 / hidden_dim)))
        adj_test = max(min_train // 2, int(max_tokens_test * (768 / hidden_dim)))
        ex_per_dim = adj_train / hidden_dim
        print(f"  Token budget: {adj_train} train, {adj_val} val, {adj_test} test ({ex_per_dim:.0f} ex/dim)")

        # Step 1: coarse layer sweep on validation split to find peak
        print("\n  Step 1: Layer sweep (validation split)")
        peak_layer, layer_profile = _coarse_layer_sweep(
            model, tokenizer, device, train_docs, val_docs, n_layers, adj_train, adj_val
        )

        # Guard: if peak is within 2 layers of output, the output-control test
        # becomes degenerate (comparing a layer against itself). Use the best
        # layer at <=80% depth instead, and report both.
        mid_peak = peak_layer
        if peak_layer >= output_layer - 1:
            max_mid = int(0.8 * n_layers)
            mid_candidates = {l: r for l, r in layer_profile.items() if l <= max_mid}
            if mid_candidates:
                mid_peak = max(mid_candidates, key=mid_candidates.get)
                print(
                    f"  Global peak at layer {peak_layer} ({peak_layer / n_layers:.0%} depth) is near output."
                )
                print(
                    f"  Using mid-depth peak at layer {mid_peak} ({mid_peak / n_layers:.0%} depth) for output control."
                )
        peak_layer = mid_peak
        val_rho_at_peak = layer_profile[peak_layer]
        print(f"  Peak layer: {peak_layer} (val partial corr {val_rho_at_peak:+.4f})")

        # Step 2: full battery at peak layer (held-out test split)
        print(f"\n  Step 2: Full battery at layer {peak_layer} ({len(seeds)} seeds) [held-out test]")
        train_peak = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, adj_train)
        test_peak = collect_layer_data(model, tokenizer, test_docs, peak_layer, device, adj_test)

        all_scores = []
        all_rhos = []
        for seed in seeds:
            head = train_linear_binary(train_peak, seed=seed)
            scores, rho, p = evaluate_head(head, test_peak)
            all_scores.append(scores)
            all_rhos.append(float(rho))
            print(f"    seed {seed}: partial corr = {rho:+.4f}")

        pairwise = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                r, _ = spearmanr(all_scores[i], all_scores[j])
                pairwise.append(float(r))

        mean_rho = float(np.mean(all_rhos))
        rho_ci = bootstrap_ci(all_rhos) if len(all_rhos) >= 3 else (mean_rho, mean_rho)
        mean_agree = float(np.mean(pairwise)) if pairwise else 0.0
        agree_ci = bootstrap_ci(pairwise) if len(pairwise) >= 3 else (mean_agree, mean_agree)

        # Step 3: output-controlled residual
        print(f"\n  Step 3: Output control (layer {peak_layer} vs layer {output_layer})")
        train_last = collect_layer_data(model, tokenizer, train_docs, output_layer, device, adj_train)
        test_last = collect_layer_data(model, tokenizer, test_docs, output_layer, device, adj_test)

        all_rhos_controlled = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Train output-layer loss predictor
            acts_last = train_last["activations"]
            targets = torch.from_numpy(train_last["losses"]).float()
            n_feat = acts_last.size(1)
            predictor = torch.nn.Sequential(
                torch.nn.Linear(n_feat, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
            )
            opt = torch.optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
            ds = torch.utils.data.TensorDataset(acts_last, targets)
            dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
            for _ep in range(20):
                for bx, by in dl:
                    loss = F.mse_loss(predictor(bx).squeeze(-1), by)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            predictor.eval()
            with torch.inference_mode():
                pred_scores = predictor(test_last["activations"]).squeeze(-1).numpy()

            # Evaluate peak-layer observer with output control
            head = train_linear_binary(train_peak, seed=seed)
            head.eval()
            with torch.inference_mode():
                obs_scores = head(test_peak["activations"]).squeeze(-1).numpy()

            rho_ctrl, _ = partial_spearman(
                obs_scores,
                test_peak["losses"],
                [test_peak["max_softmax"], test_peak["activation_norm"], pred_scores],
            )
            all_rhos_controlled.append(float(rho_ctrl))
            print(f"    seed {seed}: output-controlled = {rho_ctrl:+.4f}")

        mean_ctrl = float(np.mean(all_rhos_controlled))
        ctrl_ci = (
            bootstrap_ci(all_rhos_controlled) if len(all_rhos_controlled) >= 3 else (mean_ctrl, mean_ctrl)
        )

        # Summary for this model
        print(f"\n  {model_id} SUMMARY:")
        print(f"    peak layer:       {peak_layer} ({peak_layer / n_layers:.0%} depth, selected on val)")
        print(f"    val partial corr: {val_rho_at_peak:+.4f}  (layer selection split)")
        print(
            f"    partial corr:     {mean_rho:+.4f}  95% CI [{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]  (held-out test)"
        )
        print(f"    seed agreement:   {mean_agree:+.4f}  95% CI [{agree_ci[0]:+.4f}, {agree_ci[1]:+.4f}]")
        print(f"    output-controlled:{mean_ctrl:+.4f}  95% CI [{ctrl_ci[0]:+.4f}, {ctrl_ci[1]:+.4f}]")

        all_results[model_id] = {
            "model_id": model_id,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "n_params_m": n_params_m,
            "peak_layer": peak_layer,
            "peak_layer_source": "validation",
            "peak_layer_frac": round(peak_layer / n_layers, 2),
            "layer_profile_split": "validation",
            "layer_profile": {str(k): v for k, v in sorted(layer_profile.items())},
            "val_partial_corr_at_peak": val_rho_at_peak,
            "partial_corr": {
                "mean": mean_rho,
                "std": float(np.std(all_rhos)),
                "per_seed": all_rhos,
                "ci_95": list(rho_ci),
                "split": "test",
            },
            "seed_agreement": {
                "mean": mean_agree,
                "pairwise": pairwise,
                "ci_95": list(agree_ci),
            },
            "output_controlled": {
                "mean": mean_ctrl,
                "std": float(np.std(all_rhos_controlled)),
                "per_seed": all_rhos_controlled,
                "ci_95": list(ctrl_ci),
            },
            "n_train_tokens": adj_train,
            "n_val_tokens": adj_val,
            "n_test_tokens": adj_test,
        }

        # Free memory before loading next model
        del model, train_peak, test_peak, train_last, test_last
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Scaling summary table
    print(f"\n{'=' * 60}")
    print("  SCALING SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"  {'Model':<16} {'Params':>7} {'Peak':>6} {'Partial corr':>14} {'Output ctrl':>13} {'Seed agree':>12}"
    )
    print(f"  {'-' * 70}")
    summary = {"model_sizes_m": [], "partial_corr": [], "output_controlled": [], "seed_agreement": []}
    for model_id, n_params_m in models_to_run:
        if model_id not in all_results:
            continue
        r = all_results[model_id]
        pc = r["partial_corr"]["mean"]
        oc = r["output_controlled"]["mean"]
        sa = r["seed_agreement"]["mean"]
        print(
            f"  {model_id:<16} {n_params_m:>6}M  L{r['peak_layer']:<4} {pc:>+14.4f} {oc:>+13.4f} {sa:>+12.4f}"
        )
        summary["model_sizes_m"].append(n_params_m)
        summary["partial_corr"].append(pc)
        summary["output_controlled"].append(oc)
        summary["seed_agreement"].append(sa)

    return {"models": all_results, "scaling_summary": summary}


# ---------------------------------------------------------------------------
# Cross-family replication
# ---------------------------------------------------------------------------

CROSS_FAMILY_MODELS = {
    "9a": [("meta-llama/Llama-3.2-1B", 1236)],
    "9b": [("Qwen/Qwen2.5-0.5B", 495), ("Qwen/Qwen2.5-1.5B", 1544)],
}


def run_cross_family(
    phase,
    device,
    seeds,
    train_docs,
    val_docs,
    test_docs,
    max_tokens_train,
    max_tokens_val,
    max_tokens_test,
    model_id_override=None,
):
    """Cross-family replication of the observer signal.

    Same evaluation protocol as the GPT-2 scaling sweep (layer sweep on
    val_docs, three-seed battery on held-out test_docs, output-controlled
    residual), plus negative baselines (hand-designed observers, random head).
    Uses AutoModel for architecture-agnostic loading.

    Saves to cross_family.json with deep merge (safe for partial reruns).

    9a: Llama 3.2 1B (headline cross-family test)
    9b: Qwen 2.5 0.5B + 1.5B (second family replication)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id_override:
        models_to_run = [(model_id_override, 0)]
    else:
        models_to_run = CROSS_FAMILY_MODELS.get(phase, [])

    if not models_to_run:
        print(f"  No models configured for group {phase}")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  Cross-family replication ({phase})")
    print(f"  Models: {[m[0] for m in models_to_run]}")
    print(f"  Seeds: {seeds}")
    print(f"{'=' * 60}")

    all_results = {}

    for model_id, n_params_m in models_to_run:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_id}")
        print(f"{'=' * 60}")

        print(f"  Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # Use float16 for faster inference on MPS/CUDA; probes train in float32 on CPU tensors
        dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(
            device
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        n_params_m = n_params_m or int(sum(p.numel() for p in model.parameters()) / 1e6)
        output_layer = n_layers - 1
        print(f"  {n_params_m}M params, {n_layers} layers, hidden dim {hidden_dim}")

        # Scale token budget inversely with hidden dim
        # Scale token budget to maintain examples-per-dimension ratio.
        # GPT-2 baseline: 200k tokens / 768 dim = ~260 ex/dim.
        # Floor at 150 ex/dim for credible linear probe fitting.
        min_ex_per_dim = 150
        min_train = min_ex_per_dim * hidden_dim
        adj_train = max(min_train, int(max_tokens_train * (768 / hidden_dim)))
        adj_val = max(min_train // 2, int(max_tokens_val * (768 / hidden_dim)))
        adj_test = max(min_train // 2, int(max_tokens_test * (768 / hidden_dim)))
        ex_per_dim = adj_train / hidden_dim
        print(f"  Token budget: {adj_train} train, {adj_val} val, {adj_test} test ({ex_per_dim:.0f} ex/dim)")

        # Ensure tokenizer has a pad token (some models lack one)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Step 1: Layer sweep on validation split ---
        print("\n  Step 1: Layer sweep (validation split)")
        peak_layer, layer_profile = _coarse_layer_sweep(
            model, tokenizer, device, train_docs, val_docs, n_layers, adj_train, adj_val
        )

        # Guard against peak at output layer
        if peak_layer >= output_layer - 1:
            max_mid = int(0.8 * n_layers)
            mid_candidates = {l: r for l, r in layer_profile.items() if l <= max_mid}
            if mid_candidates:
                mid_peak = max(mid_candidates, key=mid_candidates.get)
                print(
                    f"  Global peak at layer {peak_layer} ({peak_layer / n_layers:.0%} depth) is near output."
                )
                print(
                    f"  Using mid-depth peak at layer {mid_peak} ({mid_peak / n_layers:.0%} depth) for output control."
                )
                peak_layer = mid_peak
        val_rho_at_peak = layer_profile[peak_layer]
        print(f"  Peak layer: {peak_layer} (val partial corr {val_rho_at_peak:+.4f})")

        # --- Step 2: Full battery at peak layer (held-out test split) ---
        print(f"\n  Step 2: Full battery at layer {peak_layer} ({len(seeds)} seeds) [held-out test]")
        train_peak = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, adj_train)
        test_peak = collect_layer_data(model, tokenizer, test_docs, peak_layer, device, adj_test)

        all_scores = []
        all_rhos = []
        for seed in seeds:
            head = train_linear_binary(train_peak, seed=seed)
            scores, rho, p = evaluate_head(head, test_peak)
            all_scores.append(scores)
            all_rhos.append(float(rho))
            print(f"    seed {seed}: partial corr = {rho:+.4f}")

        pairwise = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                r, _ = spearmanr(all_scores[i], all_scores[j])
                pairwise.append(float(r))

        mean_rho = float(np.mean(all_rhos))
        rho_ci = bootstrap_ci(all_rhos) if len(all_rhos) >= 3 else (mean_rho, mean_rho)
        mean_agree = float(np.mean(pairwise)) if pairwise else 0.0
        agree_ci = bootstrap_ci(pairwise) if len(pairwise) >= 3 else (mean_agree, mean_agree)

        # --- Step 3: Negative baselines at peak layer ---
        print(f"\n  Step 3: Negative baselines at layer {peak_layer}")
        hand_designed = compute_hand_designed(test_peak)
        baseline_results = {}
        for name, scores in hand_designed.items():
            rho_hd, p_hd = partial_spearman(
                scores, test_peak["losses"], [test_peak["max_softmax"], test_peak["activation_norm"]]
            )
            baseline_results[name] = float(rho_hd)
            print(f"    {name:<20} partial corr = {rho_hd:+.4f}")

        # Random head baseline
        torch.manual_seed(99)
        random_head = torch.nn.Linear(hidden_dim, 1)
        random_head.eval()
        with torch.inference_mode():
            random_scores = random_head(test_peak["activations"]).squeeze(-1).numpy()
        rho_rand, _ = partial_spearman(
            random_scores, test_peak["losses"], [test_peak["max_softmax"], test_peak["activation_norm"]]
        )
        baseline_results["random_head"] = float(rho_rand)
        print(f"    {'random_head':<20} partial corr = {rho_rand:+.4f}")

        # --- Step 4: Output-controlled residual ---
        print(f"\n  Step 4: Output control (layer {peak_layer} vs layer {output_layer})")
        train_last = collect_layer_data(model, tokenizer, train_docs, output_layer, device, adj_train)
        test_last = collect_layer_data(model, tokenizer, test_docs, output_layer, device, adj_test)

        all_rhos_controlled = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            acts_last = train_last["activations"]
            targets = torch.from_numpy(train_last["losses"]).float()
            n_feat = acts_last.size(1)
            predictor = torch.nn.Sequential(
                torch.nn.Linear(n_feat, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
            )
            opt = torch.optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
            ds = torch.utils.data.TensorDataset(acts_last, targets)
            dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
            for _ep in range(20):
                for bx, by in dl:
                    loss = F.mse_loss(predictor(bx).squeeze(-1), by)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

            predictor.eval()
            with torch.inference_mode():
                pred_scores = predictor(test_last["activations"]).squeeze(-1).numpy()

            head = train_linear_binary(train_peak, seed=seed)
            head.eval()
            with torch.inference_mode():
                obs_scores = head(test_peak["activations"]).squeeze(-1).numpy()

            rho_ctrl, _ = partial_spearman(
                obs_scores,
                test_peak["losses"],
                [test_peak["max_softmax"], test_peak["activation_norm"], pred_scores],
            )
            all_rhos_controlled.append(float(rho_ctrl))
            print(f"    seed {seed}: output-controlled = {rho_ctrl:+.4f}")

        mean_ctrl = float(np.mean(all_rhos_controlled))
        ctrl_ci = (
            bootstrap_ci(all_rhos_controlled) if len(all_rhos_controlled) >= 3 else (mean_ctrl, mean_ctrl)
        )

        # --- Summary ---
        print(f"\n  {model_id} SUMMARY:")
        print(f"    peak layer:       {peak_layer} ({peak_layer / n_layers:.0%} depth, selected on val)")
        print(f"    val partial corr: {val_rho_at_peak:+.4f}  (layer selection split)")
        print(
            f"    partial corr:     {mean_rho:+.4f}  95% CI [{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]  (held-out test)"
        )
        print(f"    seed agreement:   {mean_agree:+.4f}  95% CI [{agree_ci[0]:+.4f}, {agree_ci[1]:+.4f}]")
        print(f"    output-controlled:{mean_ctrl:+.4f}  95% CI [{ctrl_ci[0]:+.4f}, {ctrl_ci[1]:+.4f}]")
        print(f"    baselines: {baseline_results}")

        # Compare to GPT-2 band
        gpt2_band = (0.279, 0.290)
        if mean_rho >= gpt2_band[0] * 0.7:
            if mean_rho >= gpt2_band[0]:
                print("    --> In GPT-2 band: strong cross-family replication")
            else:
                print("    --> Below GPT-2 band but nontrivial: weaker cross-family signal")
        else:
            print("    --> Well below GPT-2 band: signal may be family-dependent")

        model_key = model_id.split("/")[-1]
        all_results[model_key] = {
            "model_id": model_id,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "n_params_m": n_params_m,
            "peak_layer": peak_layer,
            "peak_layer_source": "validation",
            "peak_layer_frac": round(peak_layer / n_layers, 2),
            "layer_profile_split": "validation",
            "layer_profile": {str(k): v for k, v in sorted(layer_profile.items())},
            "val_partial_corr_at_peak": val_rho_at_peak,
            "partial_corr": {
                "mean": mean_rho,
                "std": float(np.std(all_rhos)),
                "per_seed": all_rhos,
                "ci_95": list(rho_ci),
                "split": "test",
            },
            "seed_agreement": {
                "mean": mean_agree,
                "pairwise": pairwise,
                "ci_95": list(agree_ci),
            },
            "output_controlled": {
                "mean": mean_ctrl,
                "std": float(np.std(all_rhos_controlled)),
                "per_seed": all_rhos_controlled,
                "ci_95": list(ctrl_ci),
            },
            "baselines": baseline_results,
            "n_train_tokens": adj_train,
            "n_val_tokens": adj_val,
            "n_test_tokens": adj_test,
        }

        # Free memory
        del model, train_peak, test_peak, train_last, test_last
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  CROSS-FAMILY SUMMARY ({phase})")
    print(f"{'=' * 60}")
    print(
        f"  {'Model':<20} {'Params':>7} {'Peak':>6} {'Partial corr':>14}"
        f" {'Output ctrl':>13} {'Seed agree':>12}"
    )
    print(f"  {'-' * 74}")
    for key, r in all_results.items():
        pc = r["partial_corr"]["mean"]
        oc = r["output_controlled"]["mean"]
        sa = r["seed_agreement"]["mean"]
        print(
            f"  {key:<20} {r['n_params_m']:>6}M  L{r['peak_layer']:<4} {pc:>+14.4f} {oc:>+13.4f} {sa:>+12.4f}"
        )

    return {"models": all_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--seeds", type=int, default=3, help="Number of observer head seeds")
    P.add_argument("--device", default="auto")
    P.add_argument("--max-tokens-train", type=int, default=200000)
    P.add_argument("--max-tokens-val", type=int, default=100000)
    P.add_argument("--max-tokens-test", type=int, default=100000)
    P.add_argument("--layer-sweep", action="store_true", help="Run 5b layer sweep")
    P.add_argument("--baselines", action="store_true", help="Run 5c hand-designed baselines")
    P.add_argument("--output-control", action="store_true", help="Run 5e full-output control")
    P.add_argument("--intervention", action="store_true", help="Run 5d neuron ablation")
    P.add_argument("--directional-ablation", action="store_true", help="Run 5f directional ablation")
    P.add_argument("--flagging", action="store_true", help="Run 6a early flagging")
    P.add_argument("--all", action="store_true", help="Run 5a-5f + 6a")
    P.add_argument(
        "--statistical-hardening", action="store_true", help="20-seed hardening with bootstrap CIs"
    )
    P.add_argument("--control-sensitivity", action="store_true", help="Control sensitivity analysis")
    P.add_argument(
        "--mechanism-probes", action="store_true", help="Mechanism probes (what is the observer reading?)"
    )
    P.add_argument(
        "--signal-decomposition", action="store_true", help="Signal decomposition (named components)"
    )
    P.add_argument(
        "--activation-patching",
        action="store_true",
        help="Activation patching (zero-ablation causal scan)",
    )
    P.add_argument(
        "--matched-patching",
        action="store_true",
        help="Matched-pair patching (dual-metric causal localization)",
    )
    P.add_argument(
        "--mechanistic",
        action="store_true",
        help="Best-practice mechanistic analysis (mean ablation + composition)",
    )
    P.add_argument("--cross-domain", action="store_true", help="Cross-domain transfer test")
    P.add_argument("--scale", "--phase8", action="store_true", help="Scaling across GPT-2 family")
    P.add_argument("--model", default="gpt2", help="Model for single-model scaling run")
    P.add_argument("--phase9a", action="store_true", help="Cross-family test: Llama 3.2 1B")
    P.add_argument("--phase9b", action="store_true", help="Cross-family test: Qwen 2.5 0.5B and 1.5B")
    P.add_argument("--phase9", action="store_true", help="All cross-family experiments (9a and 9b)")
    P.add_argument(
        "--mechanistic-7b",
        action="store_true",
        help="Mechanistic analysis on Qwen 7B (mean-ablation patching at scale)",
    )
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    seeds = list(range(42, 42 + a.seeds))

    print("Transformer observer experiments")
    print(f"Device: {a.device}  Seeds: {seeds}")
    print(
        f"Train tokens: {a.max_tokens_train}  Val tokens: {a.max_tokens_val}  Test tokens: {a.max_tokens_test}"
    )

    # Load data (shared across all experiments)
    # Three-way split: train for probe fitting, validation for layer selection,
    # held-out test for final reporting. Prevents selection bias from choosing
    # the peak layer on the same data used to report partial correlations.
    print("\nLoading WikiText-103...")
    train_docs = load_wikitext("train", max_docs=2000)
    val_docs = load_wikitext("validation", max_docs=500)
    test_docs = load_wikitext("test", max_docs=500)
    print(f"  {len(train_docs)} train docs, {len(val_docs)} val docs, {len(test_docs)} test docs")

    results = {}
    t0 = time.time()

    # Scaling (handles its own model loading)
    if a.scale:
        model_names = [m[0] for m in GPT2_MODELS] if a.model == "gpt2" else [a.model]
        results["8"] = run_scale(
            a.device,
            seeds,
            train_docs,
            val_docs,
            test_docs,
            a.max_tokens_train,
            a.max_tokens_val,
            a.max_tokens_test,
            model_names,
        )
        elapsed = time.time() - t0
        print(f"\nTotal time: {elapsed:.0f}s")
        _save_results(results)
        return

    # Cross-family replication (handles its own model loading)
    if a.phase9a or a.phase9 or a.phase9b:
        if a.phase9a or a.phase9:
            results["9a"] = run_cross_family(
                "9a",
                a.device,
                seeds,
                train_docs,
                val_docs,
                test_docs,
                a.max_tokens_train,
                a.max_tokens_val,
                a.max_tokens_test,
            )
        if a.phase9b or a.phase9:
            results["9b"] = run_cross_family(
                "9b",
                a.device,
                seeds,
                train_docs,
                val_docs,
                test_docs,
                a.max_tokens_train,
                a.max_tokens_val,
                a.max_tokens_test,
            )
        elapsed = time.time() - t0
        print(f"\nTotal time: {elapsed:.0f}s")
        _save_results(results, filename="cross_family.json")
        return

    # Mechanistic analysis at 7B (handles its own model loading)
    if a.mechanistic_7b:
        import gc

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen2.5-7B"
        print(f"\nLoading {model_id} for mechanistic analysis...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if a.device in ("mps", "cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(a.device)
        model.eval()

        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {n_params / 1e9:.1f}B params, {n_layers} layers, hidden dim {hidden_dim}")

        # Scale token budgets for 7B
        min_ex_per_dim = 150
        adj_train = max(min_ex_per_dim * hidden_dim, int(a.max_tokens_train * (768 / hidden_dim)))
        adj_test = max(min_ex_per_dim * hidden_dim // 2, int(a.max_tokens_test * (768 / hidden_dim)))

        peak_layer = 18  # Qwen 7B base default
        mech_result = run_mechanistic_general(
            model,
            tokenizer,
            a.device,
            train_docs,
            test_docs,
            adj_train,
            adj_test,
            peak_layer=peak_layer,
            eval_budget=15000,
        )
        mech_result["model"] = model_id
        mech_result["n_params_b"] = round(n_params / 1e9, 1)
        mech_result["n_layers"] = n_layers
        mech_result["hidden_dim"] = hidden_dim

        elapsed = time.time() - t0
        print(f"\nTotal time: {elapsed:.0f}s")
        _save_results({"mechanistic_7b": mech_result}, filename="mechanistic_7b.json")

        del model, tokenizer
        gc.collect()
        if a.device == "cuda":
            torch.cuda.empty_cache()
        return

    # Phases 5-8: load GPT-2 124M
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("\nLoading GPT-2 124M...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(a.device)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    hardening_only = (
        (
            a.statistical_hardening
            or a.control_sensitivity
            or a.mechanism_probes
            or a.signal_decomposition
            or a.activation_patching
            or a.matched_patching
            or a.mechanistic
            or a.cross_domain
        )
        and not a.all
        and not a.layer_sweep
        and not a.baselines
        and not a.output_control
        and not a.intervention
        and not a.directional_ablation
        and not a.flagging
    )

    # 5a: always run unless hardening-only
    if not hardening_only:
        results["5a"] = run_5a(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # 5b: layer sweep
    if a.layer_sweep or a.all:
        results["5b"] = run_5b(
            model,
            tokenizer,
            a.device,
            seeds,
            train_docs,
            val_docs,
            test_docs,
            a.max_tokens_train,
            a.max_tokens_val,
            a.max_tokens_test,
        )

    # 5c: baselines
    if a.baselines or a.all:
        results["5c"] = run_5c(model, tokenizer, a.device, test_docs, a.max_tokens_test)

    # 5e: full-output control
    if a.output_control or a.all:
        results["5e"] = run_5e(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # 5d: neuron ablation
    if a.intervention or a.all:
        results["5d"] = run_5d(
            model, tokenizer, a.device, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # 5f: directional ablation
    if a.directional_ablation or a.all:
        results["5f"] = run_5f(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # 6a: early flagging
    if a.flagging or a.all:
        results["6a"] = run_6a(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    # Methodology hardening (not a numbered phase)
    if a.statistical_hardening:
        results["hardening"] = run_8a(
            model, tokenizer, a.device, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    if a.control_sensitivity:
        results["control_sensitivity"] = run_8b(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    if a.mechanism_probes:
        results["mechanism_probes"] = run_mechanism_probes(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    if a.signal_decomposition:
        results["signal_decomposition"] = run_signal_decomposition(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    if a.activation_patching:
        results["activation_patching"] = run_activation_patching(
            model, tokenizer, a.device, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    if a.matched_patching:
        results["matched_patching"] = run_matched_pair_patching(
            model, tokenizer, a.device, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    if a.mechanistic:
        results["mechanistic"] = run_mechanistic_analysis(
            model, tokenizer, a.device, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    if a.cross_domain:
        results["cross_domain"] = run_8c(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    _save_results(results)


if __name__ == "__main__":
    main()
