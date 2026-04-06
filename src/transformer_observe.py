"""
Phases 5, 6, 8: Transformer observer experiments.

Test whether learned observer heads transfer to transformers (Phase 5),
catch errors confidence misses (Phase 6), and persist across the GPT-2
scaling curve (Phase 8).

Experiments:
  5a: Direct replication at last layer (primary result)
  5b: Layer sweep across all 12 layers
  5c: Hand-designed baseline comparison
  5d: Neuron ablation intervention (null result: skip connections buffer damage)
  5e: Full-output control (layer 8 vs layer 11 predictor)
  5f: Directional ablation (residual stream projection, addresses 5d failure)
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
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from observe import compute_loss_residuals, partial_spearman

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def bootstrap_ci(values, n_boot=10000, ci=0.95, seed=0):
    """Bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    lo = float(np.percentile(means, 100 * (1 - ci) / 2))
    hi = float(np.percentile(means, 100 * (1 + ci) / 2))
    return lo, hi


def _deep_merge(base, update):
    """Recursively merge `update` into `base`, preserving nested keys.

    Prevents partial reruns from nuking sibling results. For example,
    rerunning a single model in Phase 8 merges into the existing
    models dict rather than replacing the entire phase key.
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _save_results(results, filename="transformer_observe.json"):
    """Deep-merge results into existing JSON file and save."""
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    out_file = out / filename
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
    _deep_merge(existing, results)
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {out_file} (keys: {sorted(existing.keys())})")


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
        shift_logits = outputs.logits[0, :-1, :]
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
        "activations": torch.cat(all_acts).float(),  # ensure float32 for probe training
        "losses": torch.cat(all_losses).numpy(),
        "max_softmax": torch.cat(all_softmax).numpy(),
        "logit_entropy": torch.cat(all_logit_entropy).numpy(),
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
            loss = F.cross_entropy(outputs.logits[0, :-1, :], input_ids[0, 1:])
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
                F.cross_entropy(outputs.logits[0, :-1, :], input_ids[0, 1:], reduction="none").cpu().numpy()
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
# Phase 8: Generalization and statistical hardening
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
# Phase 9: Scale characterization
# ---------------------------------------------------------------------------

GPT2_MODELS = [
    ("gpt2", 124),
    ("gpt2-medium", 355),
    ("gpt2-large", 774),
    ("gpt2-xl", 1558),
]


def _coarse_layer_sweep(
    model, tokenizer, device, train_docs, test_docs, n_layers, max_tokens_train, max_tokens_test
):
    """Sweep layers with 1 seed to find peak partial correlation.

    For models with many layers, sweep every Nth layer first (coarse),
    then sweep densely around the peak region.
    """
    stride = max(1, n_layers // 12)
    coarse_layers = list(range(0, n_layers, stride))
    if (n_layers - 1) not in coarse_layers:
        coarse_layers.append(n_layers - 1)

    profile = {}
    print(f"  Coarse sweep: {len(coarse_layers)} layers (stride {stride})")
    for layer in coarse_layers:
        train_data = collect_layer_data(model, tokenizer, train_docs, layer, device, max_tokens_train)
        test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)
        head = train_linear_binary(train_data, seed=42)
        _, rho, _ = evaluate_head(head, test_data)
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
                test_data = collect_layer_data(model, tokenizer, test_docs, layer, device, max_tokens_test)
                head = train_linear_binary(train_data, seed=42)
                _, rho, _ = evaluate_head(head, test_data)
                profile[layer] = float(rho)
                print(f"    layer {layer:>3}: {rho:+.4f}")

    peak_layer = max(profile, key=profile.get)
    return peak_layer, profile


def run_scale(device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test, model_names=None):
    """Phase 8: Scale characterization across GPT-2 family.

    For each model: coarse layer sweep to find peak, full battery at peak
    (partial correlation + seed agreement), output-controlled residual.
    Models loaded and unloaded sequentially to manage memory.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if model_names is None:
        model_names = [m[0] for m in GPT2_MODELS]

    models_to_run = [(name, params) for name, params in GPT2_MODELS if name in model_names]

    print(f"\n{'=' * 60}")
    print("  Phase 9: Scale characterization")
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
        adj_test = max(min_train // 2, int(max_tokens_test * (768 / hidden_dim)))
        ex_per_dim = adj_train / hidden_dim
        print(f"  Token budget: {adj_train} train, {adj_test} test ({ex_per_dim:.0f} ex/dim)")

        # Step 1: coarse layer sweep to find peak
        print("\n  Step 1: Layer sweep")
        peak_layer, layer_profile = _coarse_layer_sweep(
            model, tokenizer, device, train_docs, test_docs, n_layers, adj_train, adj_test
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
        print(f"  Peak layer: {peak_layer} (partial corr {layer_profile[peak_layer]:+.4f})")

        # Step 2: full battery at peak layer
        print(f"\n  Step 2: Full battery at layer {peak_layer} ({len(seeds)} seeds)")
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
        print(f"    peak layer:       {peak_layer} ({peak_layer / n_layers:.0%} depth)")
        print(f"    partial corr:     {mean_rho:+.4f}  95% CI [{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]")
        print(f"    seed agreement:   {mean_agree:+.4f}  95% CI [{agree_ci[0]:+.4f}, {agree_ci[1]:+.4f}]")
        print(f"    output-controlled:{mean_ctrl:+.4f}  95% CI [{ctrl_ci[0]:+.4f}, {ctrl_ci[1]:+.4f}]")

        all_results[model_id] = {
            "model_id": model_id,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "n_params_m": n_params_m,
            "peak_layer": peak_layer,
            "peak_layer_frac": round(peak_layer / n_layers, 2),
            "layer_profile": {str(k): v for k, v in sorted(layer_profile.items())},
            "partial_corr": {
                "mean": mean_rho,
                "std": float(np.std(all_rhos)),
                "per_seed": all_rhos,
                "ci_95": list(rho_ci),
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
    print("  Phase 9: SCALING SUMMARY")
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
# Phase 9: Cross-family replication
# ---------------------------------------------------------------------------

CROSS_FAMILY_MODELS = {
    "9a": [("meta-llama/Llama-3.2-1B", 1236)],
    "9b": [("Qwen/Qwen2.5-0.5B", 495), ("Qwen/Qwen2.5-1.5B", 1544)],
}


def run_cross_family(
    phase, device, seeds, train_docs, test_docs, max_tokens_train, max_tokens_test, model_id_override=None
):
    """Phase 9: Cross-family replication of the observer signal.

    Same evaluation protocol as Phase 8 (layer sweep, three-seed battery,
    output-controlled residual), plus negative baselines (hand-designed
    observers, random head). Uses AutoModel for architecture-agnostic loading.

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
        print(f"  No models configured for phase {phase}")
        return {}

    print(f"\n{'=' * 60}")
    print(f"  Phase {phase}: Cross-family replication")
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
        adj_test = max(min_train // 2, int(max_tokens_test * (768 / hidden_dim)))
        ex_per_dim = adj_train / hidden_dim
        print(f"  Token budget: {adj_train} train, {adj_test} test ({ex_per_dim:.0f} ex/dim)")

        # Ensure tokenizer has a pad token (some models lack one)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Step 1: Layer sweep ---
        print("\n  Step 1: Layer sweep")
        peak_layer, layer_profile = _coarse_layer_sweep(
            model, tokenizer, device, train_docs, test_docs, n_layers, adj_train, adj_test
        )

        # Guard against peak at output layer (same fix as Phase 8 medium)
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
        print(f"  Peak layer: {peak_layer} (partial corr {layer_profile[peak_layer]:+.4f})")

        # --- Step 2: Full battery at peak layer ---
        print(f"\n  Step 2: Full battery at layer {peak_layer} ({len(seeds)} seeds)")
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
        print(f"    peak layer:       {peak_layer} ({peak_layer / n_layers:.0%} depth)")
        print(f"    partial corr:     {mean_rho:+.4f}  95% CI [{rho_ci[0]:+.4f}, {rho_ci[1]:+.4f}]")
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
            "peak_layer_frac": round(peak_layer / n_layers, 2),
            "layer_profile": {str(k): v for k, v in sorted(layer_profile.items())},
            "partial_corr": {
                "mean": mean_rho,
                "std": float(np.std(all_rhos)),
                "per_seed": all_rhos,
                "ci_95": list(rho_ci),
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
    print(f"  Phase {phase}: CROSS-FAMILY SUMMARY")
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
    P.add_argument("--cross-domain", action="store_true", help="Cross-domain transfer test")
    P.add_argument("--scale", "--phase8", action="store_true", help="Phase 8: scaling across GPT-2 family")
    P.add_argument("--model", default="gpt2", help="Model for single-model scaling run")
    P.add_argument("--phase9a", action="store_true", help="Phase 9a: Gemma 2 2B cross-family test")
    P.add_argument("--phase9b", action="store_true", help="Phase 9b: Qwen 2.5 0.5B + 1.5B replication")
    P.add_argument("--phase9", action="store_true", help="Phase 9: all cross-family experiments (9a + 9b)")
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    seeds = list(range(42, 42 + a.seeds))

    print("Transformer observer experiments")
    print(f"Device: {a.device}  Seeds: {seeds}")
    print(f"Train tokens: {a.max_tokens_train}  Test tokens: {a.max_tokens_test}")

    # Load data (shared across all experiments)
    print("\nLoading WikiText-103...")
    train_docs = load_wikitext("train", max_docs=2000)
    test_docs = load_wikitext("test", max_docs=500)
    print(f"  {len(train_docs)} train docs, {len(test_docs)} test docs")

    results = {}
    t0 = time.time()

    # Phase 8: scaling (handles its own model loading)
    if a.scale:
        model_names = [m[0] for m in GPT2_MODELS] if a.model == "gpt2" else [a.model]
        results["8"] = run_scale(
            a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test, model_names
        )
        elapsed = time.time() - t0
        print(f"\nTotal time: {elapsed:.0f}s")
        _save_results(results)
        return

    # Phase 9: cross-family replication (handles its own model loading)
    if a.phase9a or a.phase9 or a.phase9b:
        if a.phase9a or a.phase9:
            results["9a"] = run_cross_family(
                "9a", a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
            )
        if a.phase9b or a.phase9:
            results["9b"] = run_cross_family(
                "9b", a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
            )
        elapsed = time.time() - t0
        print(f"\nTotal time: {elapsed:.0f}s")
        _save_results(results, filename="cross_family.json")
        return

    # Phases 5-8: load GPT-2 124M
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("\nLoading GPT-2 124M...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(a.device)
    model.eval()
    print(f"  {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")

    hardening_only = (
        (a.statistical_hardening or a.control_sensitivity or a.cross_domain)
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

    if a.cross_domain:
        results["cross_domain"] = run_8c(
            model, tokenizer, a.device, seeds, train_docs, test_docs, a.max_tokens_train, a.max_tokens_test
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    _save_results(results)


if __name__ == "__main__":
    main()
