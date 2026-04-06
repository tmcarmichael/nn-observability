"""
Phase 7: SAE comparison.

Does an SAE feature space recover the same decision-quality signal as the
direct linear observer? Apples-to-apples: same model, same hookpoint, same
train/test split, same binary target, same partial correlation evaluation.

Experiments:
  7a: SAE-code linear binary vs raw-residual linear binary
  7b: Three-channel causal decomposition via directional ablation
  7c: Rank overlap between SAE probe and linear observer
  7d: SAE probe in the Phase 6a flagging framework

Usage:
    uv run --extra transformer src/sae_compare.py
    uv run --extra transformer src/sae_compare.py --all
    uv run --extra transformer src/sae_compare.py --causal
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
from transformer_observe import _deep_merge, collect_layer_data, load_wikitext, train_linear_binary

# ---------------------------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------------------------


def encode_with_sae(sae, activations):
    """Encode activations through a pretrained SAE, return sparse feature activations."""
    sae.eval()
    device = next(sae.parameters()).device
    all_codes = []
    batch_size = 4096

    with torch.inference_mode():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            codes = sae.encode(batch)
            all_codes.append(codes.cpu())

    return torch.cat(all_codes)


def train_sae_probe(sae_codes, losses, softmax, norms, seed=42, epochs=20, lr=1e-3):
    """Train linear binary probe on SAE feature codes."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    residuals = compute_loss_residuals(losses, softmax, norms)
    targets = torch.from_numpy((residuals > 0).astype(np.float32))

    n_features = sae_codes.size(1)
    head = torch.nn.Linear(n_features, 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(sae_codes, targets)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    head.train()
    for _ep in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return head


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_7a(sae, train_data, test_data, seeds):
    """7a: SAE-code linear binary vs raw-residual linear binary."""
    print(f"\n{'=' * 60}")
    print("  7a: SAE probe vs linear observer (layer 8)")
    print(f"{'=' * 60}")

    # Encode through SAE
    print("  Encoding train activations through SAE...")
    train_codes = encode_with_sae(sae, train_data["activations"])
    print(f"    shape: {train_codes.shape} (sparsity: {(train_codes == 0).float().mean():.1%})")
    print("  Encoding test activations through SAE...")
    test_codes = encode_with_sae(sae, test_data["activations"])

    raw_results = []
    sae_results = []

    for seed in seeds:
        # Raw residual linear binary (replication of Phase 5a at layer 8)
        raw_head = train_linear_binary(train_data, seed=seed)
        raw_head.eval()
        with torch.inference_mode():
            raw_scores = raw_head(test_data["activations"]).squeeze(-1).numpy()
        raw_rho, raw_p = partial_spearman(
            raw_scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
        )

        # SAE-code linear binary
        sae_head = train_sae_probe(
            train_codes,
            train_data["losses"],
            train_data["max_softmax"],
            train_data["activation_norm"],
            seed=seed,
        )
        sae_head.eval()
        with torch.inference_mode():
            sae_scores = sae_head(test_codes).squeeze(-1).numpy()
        sae_rho, sae_p = partial_spearman(
            sae_scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
        )

        raw_results.append({"rho": raw_rho, "scores": raw_scores})
        sae_results.append({"rho": sae_rho, "scores": sae_scores})
        print(f"  Seed {seed}: raw={raw_rho:+.4f}  SAE={sae_rho:+.4f}")

    # Seed agreement
    seed_agreements = {}
    for label, results in [("raw", raw_results), ("SAE", sae_results)]:
        pairs = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                r, _ = spearmanr(results[i]["scores"], results[j]["scores"])
                pairs.append(r)
        seed_agreements[label] = float(np.mean(pairs))
        print(f"  {label} seed agreement: {seed_agreements[label]:+.4f}")

    raw_mean = np.mean([r["rho"] for r in raw_results])
    sae_mean = np.mean([r["rho"] for r in sae_results])
    print("\n  7a RESULT:")
    print(f"    raw linear binary:  {raw_mean:+.4f}")
    print(f"    SAE linear binary:  {sae_mean:+.4f}")

    if abs(sae_mean - raw_mean) < 0.03:
        print("    --> Similar: signal lives in SAE feature space")
    elif sae_mean > raw_mean:
        print("    --> SAE outperforms: richer basis for decision quality")
    else:
        print("    --> Raw outperforms: SAE compression loses signal")

    return {
        "raw": {
            "mean": float(raw_mean),
            "per_seed": [r["rho"] for r in raw_results],
            "seed_agreement": seed_agreements["raw"],
        },
        "sae": {
            "mean": float(sae_mean),
            "per_seed": [r["rho"] for r in sae_results],
            "seed_agreement": seed_agreements["SAE"],
        },
        "raw_scores": [r["scores"].tolist() for r in raw_results],
        "sae_scores": [r["scores"].tolist() for r in sae_results],
    }


def _project_out_direction(h, direction):
    """Remove a direction from activations: h' = h - (h . d) * d."""
    d = direction / direction.norm()
    return h - torch.outer(h @ d, d)


def _eval_with_hook(model, tokenizer, docs, layer, direction, device, max_tokens=100000, max_length=512):
    """Evaluate model loss with a direction projected out of the residual stream at `layer`."""
    d = (direction / direction.norm()).to(device)

    def hook_fn(module, input, output):
        # GPT-2 block output is a tuple (hidden_states, ...) or just a tensor
        if isinstance(output, tuple):
            h = output[0]
            proj = (h @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
            return (h - proj,) + output[1:]
        proj = (output @ d).unsqueeze(-1) * d.unsqueeze(0).unsqueeze(0)
        return output - proj

    handle = model.transformer.h[layer].register_forward_hook(hook_fn)

    all_losses = []
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            if total_tokens >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids)
            shift_logits = outputs.logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            losses = F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu().numpy()
            all_losses.append(losses)
            total_tokens += len(losses)

    handle.remove()
    return np.concatenate(all_losses)[:max_tokens]


def _eval_baseline(model, tokenizer, docs, device, max_tokens=100000, max_length=512):
    """Evaluate model loss without any intervention."""
    all_losses = []
    total_tokens = 0
    model.eval()
    with torch.inference_mode():
        for doc in docs:
            if total_tokens >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = tokens["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue
            outputs = model(input_ids)
            shift_logits = outputs.logits[0, :-1, :]
            shift_labels = input_ids[0, 1:]
            losses = F.cross_entropy(shift_logits, shift_labels, reduction="none").cpu().numpy()
            all_losses.append(losses)
            total_tokens += len(losses)

    return np.concatenate(all_losses)[:max_tokens]


def run_7b(model, tokenizer, sae, train_data, test_data, test_11, device, seeds, max_tokens_test):
    """7b: Three-channel causal decomposition via directional ablation.

    Train three channel directions (raw observer, SAE probe projected into
    activation space, confidence predictor). Remove each from the residual
    stream at layer 8 via projection hook. Measure loss delta on each
    channel's exclusive token catches from 7d-style flagging.

    The 3x3 matrix of (direction removed) x (token subset affected) tests
    whether Phase 7's correlational structure is causally real.
    """
    layer = 8
    print(f"\n{'=' * 60}")
    print(f"  7b: Three-channel causal decomposition (layer {layer})")
    print(f"{'=' * 60}")

    seed = seeds[0]

    # --- Train the three channel directions ---

    # 1. Raw observer direction
    print("  Training raw observer head...")
    raw_head = train_linear_binary(train_data, seed=seed)
    raw_direction = raw_head.weight.data.cpu().squeeze()  # [768]

    # 2. SAE probe direction (projected into activation space)
    print("  Training SAE probe...")
    train_codes = encode_with_sae(sae, train_data["activations"])
    sae_head = train_sae_probe(
        train_codes, train_data["losses"], train_data["max_softmax"], train_data["activation_norm"], seed=seed
    )
    sae_probe_weights = sae_head.weight.data.cpu().squeeze()  # [d_sae]

    # Project SAE probe direction into activation space via decoder
    # SAE.W_dec maps from sparse features back to activation space: [d_sae, d_in]
    W_dec = sae.W_dec.data.cpu()  # [d_sae, d_in]
    sae_direction = sae_probe_weights @ W_dec  # [d_in]
    print(f"    SAE direction norm: {sae_direction.norm():.4f}")

    # 3. Confidence direction (linear predictor of max_softmax from activations)
    print("  Training confidence predictor...")
    torch.manual_seed(seed)
    conf_targets = torch.from_numpy(train_data["max_softmax"]).float()
    conf_head = torch.nn.Linear(train_data["activations"].size(1), 1)
    opt = torch.optim.Adam(conf_head.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(train_data["activations"], conf_targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    for _ep in range(20):
        for bx, by in dl:
            loss = F.mse_loss(conf_head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    conf_direction = conf_head.weight.data.cpu().squeeze()  # [768]

    # Check pairwise cosine similarities between directions
    directions = {"raw_observer": raw_direction, "sae_probe": sae_direction, "confidence": conf_direction}
    print("\n  Direction cosine similarities:")
    dir_names = list(directions.keys())
    for i in range(len(dir_names)):
        for j in range(i + 1, len(dir_names)):
            d1 = directions[dir_names[i]]
            d2 = directions[dir_names[j]]
            cos = float(F.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)))
            print(f"    {dir_names[i]} vs {dir_names[j]}: {cos:+.4f}")

    # --- Identify token subsets from 7d-style flagging ---

    n_test = min(len(test_data["losses"]), len(test_11["losses"]))
    test_losses = test_data["losses"][:n_test]
    output_softmax = test_11["max_softmax"][:n_test]
    test_acts = test_data["activations"][:n_test]

    median_loss = np.median(test_losses)
    is_high_loss = test_losses > median_loss

    flag_rate = 0.10
    k = int(n_test * flag_rate)

    # Score test set with each channel
    raw_head.eval()
    with torch.inference_mode():
        raw_scores = raw_head(test_acts).squeeze(-1).numpy()
    test_codes = encode_with_sae(sae, test_acts)
    sae_head.eval()
    with torch.inference_mode():
        sae_scores = sae_head(test_codes).squeeze(-1).numpy()

    # Flag tokens
    raw_flagged = raw_scores >= np.sort(raw_scores)[-k]
    sae_flagged = sae_scores >= np.sort(sae_scores)[-k]
    conf_flagged = output_softmax <= np.sort(output_softmax)[k]

    # Exclusive catches (high-loss tokens flagged by only one channel)
    raw_exclusive = raw_flagged & ~sae_flagged & ~conf_flagged & is_high_loss
    sae_exclusive = sae_flagged & ~raw_flagged & ~conf_flagged & is_high_loss
    conf_exclusive = conf_flagged & ~raw_flagged & ~sae_flagged & is_high_loss
    shared = (
        (raw_flagged.astype(int) + sae_flagged.astype(int) + conf_flagged.astype(int)) >= 2
    ) & is_high_loss

    print(f"\n  Token subsets (10% flag rate, n={n_test}):")
    print(f"    raw-exclusive:  {raw_exclusive.sum()}")
    print(f"    SAE-exclusive:  {sae_exclusive.sum()}")
    print(f"    conf-exclusive: {conf_exclusive.sum()}")
    print(f"    shared (2+):    {shared.sum()}")

    # --- Directional ablation: remove each direction, measure per-subset loss ---

    # Need per-position losses from the model, with and without intervention
    # Use test_docs directly to get aligned per-position losses
    print("\n  Computing baseline losses...")
    from transformer_observe import load_wikitext

    test_docs = load_wikitext("test", max_docs=500)
    baseline_losses = _eval_baseline(model, tokenizer, test_docs, device, max_tokens_test)
    n_eval = min(n_test, len(baseline_losses))

    # Trim everything to aligned length
    baseline_losses = baseline_losses[:n_eval]
    subsets = {
        "raw_exclusive": raw_exclusive[:n_eval],
        "sae_exclusive": sae_exclusive[:n_eval],
        "conf_exclusive": conf_exclusive[:n_eval],
        "shared": shared[:n_eval],
    }

    # Random direction baselines (3 random directions, average)
    rng = np.random.default_rng(42)
    random_directions = [
        torch.from_numpy(rng.standard_normal(raw_direction.shape[0]).astype(np.float32)) for _ in range(3)
    ]

    interventions = {
        "raw_observer": raw_direction,
        "sae_probe": sae_direction,
        "confidence": conf_direction,
    }
    for i, rd in enumerate(random_directions):
        interventions[f"random_{i}"] = rd

    causal_matrix = {}
    for int_name, direction in interventions.items():
        print(f"\n  Intervening: remove {int_name} direction...")
        intervened_losses = _eval_with_hook(
            model, tokenizer, test_docs, layer, direction, device, max_tokens_test
        )
        intervened_losses = intervened_losses[:n_eval]

        loss_delta = intervened_losses - baseline_losses
        mean_delta = float(loss_delta.mean())

        subset_deltas = {}
        for sub_name, mask in subsets.items():
            if mask.sum() > 0:
                delta = float(loss_delta[mask].mean())
            else:
                delta = 0.0
            subset_deltas[sub_name] = delta

        causal_matrix[int_name] = {"mean_delta": mean_delta, "subset_deltas": subset_deltas}
        print(f"    mean loss delta: {mean_delta:+.4f}")
        for sub_name, delta in subset_deltas.items():
            print(f"    {sub_name}: {delta:+.4f}")

    # Average random baselines
    random_mean = {
        sub: float(np.mean([causal_matrix[f"random_{i}"]["subset_deltas"][sub] for i in range(3)]))
        for sub in subsets
    }
    random_overall = float(np.mean([causal_matrix[f"random_{i}"]["mean_delta"] for i in range(3)]))

    # Summary table
    print("\n  7b RESULT: Causal decomposition matrix")
    print("  Loss delta by (direction removed) x (token subset)")
    header = f"  {'Removed':<16}"
    for sub in subsets:
        header += f" {sub:>16}"
    header += f" {'overall':>10}"
    print(header)
    print(f"  {'-' * (16 + 16 * len(subsets) + 10)}")

    for int_name in ["raw_observer", "sae_probe", "confidence"]:
        row = f"  {int_name:<16}"
        for sub in subsets:
            row += f" {causal_matrix[int_name]['subset_deltas'][sub]:>+16.4f}"
        row += f" {causal_matrix[int_name]['mean_delta']:>+10.4f}"
        print(row)

    row = f"  {'random (mean)':<16}"
    for sub in subsets:
        row += f" {random_mean[sub]:>+16.4f}"
    row += f" {random_overall:>+10.4f}"
    print(row)

    # Key test: does removing a channel's direction disproportionately hurt its exclusive catches?
    print("\n  Diagonal dominance test (does each direction disproportionately affect its own catches?):")
    pairs = [
        ("raw_observer", "raw_exclusive"),
        ("sae_probe", "sae_exclusive"),
        ("confidence", "conf_exclusive"),
    ]
    for int_name, sub_name in pairs:
        own = causal_matrix[int_name]["subset_deltas"][sub_name]
        others = [
            causal_matrix[int_name]["subset_deltas"][s] for s in subsets if s != sub_name and s != "shared"
        ]
        mean_other = float(np.mean(others)) if others else 0.0
        ratio = own / mean_other if mean_other != 0 else float("inf")
        above_random = own - random_mean[sub_name]
        print(
            f"    {int_name}: own={own:+.4f}  others={mean_other:+.4f}  ratio={ratio:.2f}  above random={above_random:+.4f}"
        )

    # Clean up random entries from results
    clean_matrix = {k: v for k, v in causal_matrix.items() if not k.startswith("random_")}
    clean_matrix["random_mean"] = {"mean_delta": random_overall, "subset_deltas": random_mean}

    return {
        "layer": layer,
        "seed": seed,
        "flag_rate": flag_rate,
        "n_eval_tokens": n_eval,
        "token_subsets": {name: int(mask.sum()) for name, mask in subsets.items()},
        "direction_cosines": {
            f"{dir_names[i]}_vs_{dir_names[j]}": float(
                F.cosine_similarity(
                    directions[dir_names[i]].unsqueeze(0), directions[dir_names[j]].unsqueeze(0)
                )
            )
            for i in range(len(dir_names))
            for j in range(i + 1, len(dir_names))
        },
        "causal_matrix": clean_matrix,
    }


def run_7c(results_7a, seeds):
    """7c: Rank overlap between SAE probe and linear observer."""
    print(f"\n{'=' * 60}")
    print("  7c: Rank overlap (SAE probe vs linear observer)")
    print(f"{'=' * 60}")

    for i, seed in enumerate(seeds):
        raw_scores = np.array(results_7a["raw_scores"][i])
        sae_scores = np.array(results_7a["sae_scores"][i])
        rho, _ = spearmanr(raw_scores, sae_scores)
        print(f"  Seed {seed}: rank corr = {rho:+.4f}")

    # Average
    all_rhos = []
    for i in range(len(seeds)):
        raw_scores = np.array(results_7a["raw_scores"][i])
        sae_scores = np.array(results_7a["sae_scores"][i])
        r, _ = spearmanr(raw_scores, sae_scores)
        all_rhos.append(r)
    mean_rho = np.mean(all_rhos)

    print(f"\n  7c RESULT: mean rank correlation = {mean_rho:+.4f}")
    if mean_rho > 0.7:
        print("    --> High overlap: reading the same signal through different decompositions")
    elif mean_rho > 0.3:
        print("    --> Moderate overlap: partially shared signal, combining may help")
    else:
        print("    --> Low overlap: reading different things, strong case for multi-channel")

    return {"mean_rank_correlation": float(mean_rho), "per_seed": all_rhos}


def run_7d(sae, test_data, test_11, train_data, seeds):
    """7d: SAE probe in the Phase 6a flagging framework."""
    print(f"\n{'=' * 60}")
    print("  7d: SAE probe flagging vs confidence")
    print(f"{'=' * 60}")

    train_codes = encode_with_sae(sae, train_data["activations"])
    test_codes = encode_with_sae(sae, test_data["activations"])

    n_test = min(len(test_data["losses"]), len(test_11["losses"]))
    test_losses = test_data["losses"][:n_test]
    output_softmax = test_11["max_softmax"][:n_test]
    test_codes_aligned = test_codes[:n_test]
    test_acts_aligned = test_data["activations"][:n_test]

    median_loss = np.median(test_losses)
    is_high_loss = test_losses > median_loss

    flag_rate = 0.10
    k = int(n_test * flag_rate)

    # Confidence flags
    conf_threshold = np.sort(output_softmax)[k]
    conf_flagged = output_softmax <= conf_threshold

    all_results = []
    for seed in seeds:
        # SAE probe
        sae_head = train_sae_probe(
            train_codes,
            train_data["losses"],
            train_data["max_softmax"],
            train_data["activation_norm"],
            seed=seed,
        )
        sae_head.eval()
        with torch.inference_mode():
            sae_scores = sae_head(test_codes_aligned).squeeze(-1).numpy()

        # Raw observer
        raw_head = train_linear_binary(train_data, seed=seed)
        raw_head.eval()
        with torch.inference_mode():
            raw_scores = raw_head(test_acts_aligned).squeeze(-1).numpy()

        sae_threshold = np.sort(sae_scores)[-k]
        sae_flagged = sae_scores >= sae_threshold
        raw_threshold = np.sort(raw_scores)[-k]
        raw_flagged = raw_scores >= raw_threshold

        sae_precision = is_high_loss[sae_flagged].mean()
        raw_precision = is_high_loss[raw_flagged].mean()
        conf_precision = is_high_loss[conf_flagged].mean()

        sae_exclusive = (sae_flagged & ~conf_flagged & is_high_loss).sum()
        raw_exclusive = (raw_flagged & ~conf_flagged & is_high_loss).sum()

        # Three-channel: flag if any of the three flags
        triple_flagged = sae_flagged | raw_flagged | conf_flagged
        triple_precision = is_high_loss[triple_flagged].mean() if triple_flagged.sum() > 0 else 0.0
        triple_catches = (triple_flagged & is_high_loss).sum()

        all_results.append(
            {
                "sae_precision": float(sae_precision),
                "raw_precision": float(raw_precision),
                "conf_precision": float(conf_precision),
                "sae_exclusive": int(sae_exclusive),
                "raw_exclusive": int(raw_exclusive),
                "triple_precision": float(triple_precision),
                "triple_catches": int(triple_catches),
            }
        )

        print(
            f"  Seed {seed}: SAE prec={sae_precision:.3f}  raw prec={raw_precision:.3f}  "
            f"conf prec={conf_precision:.3f}  SAE-exclusive={sae_exclusive}"
        )

    print(f"\n  7d RESULT (10% flag rate, {len(seeds)} seeds):")
    print(f"    SAE probe precision:  {np.mean([r['sae_precision'] for r in all_results]):.3f}")
    print(f"    Raw observer precision: {np.mean([r['raw_precision'] for r in all_results]):.3f}")
    print(f"    Confidence precision: {np.mean([r['conf_precision'] for r in all_results]):.3f}")
    print(f"    SAE-exclusive catches: {np.mean([r['sae_exclusive'] for r in all_results]):.0f}")
    print(f"    Raw-exclusive catches: {np.mean([r['raw_exclusive'] for r in all_results]):.0f}")
    triple_c = np.mean([r["triple_catches"] for r in all_results])
    triple_p = np.mean([r["triple_precision"] for r in all_results])
    print(f"    Three-channel: {triple_c:.0f} catches, {triple_p:.3f} precision")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--seeds", type=int, default=3)
    P.add_argument("--device", default="auto")
    P.add_argument("--max-tokens-train", type=int, default=200000)
    P.add_argument("--max-tokens-test", type=int, default=100000)
    P.add_argument("--all", action="store_true", help="Run 7a + 7c + 7d")
    P.add_argument("--causal", action="store_true", help="Run 7b: three-channel causal decomposition")
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    seeds = list(range(42, 42 + a.seeds))
    print("Phase 7: SAE comparison")
    print(f"Device: {a.device}  Seeds: {seeds}")

    # Load model
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("\nLoading GPT-2 124M...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(a.device)
    model.eval()

    # Load SAE for layer 8 (= blocks.9.hook_resid_pre in SAE Lens convention)
    from sae_lens import SAE

    print("Loading pretrained SAE (blocks.9.hook_resid_pre)...")
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.9.hook_resid_pre")
    sae = sae.to(a.device)
    print(f"  SAE: {sae.cfg.d_in} -> {sae.cfg.d_sae} features")

    # Load data
    print("\nLoading WikiText-103...")
    train_docs = load_wikitext("train", max_docs=2000)
    test_docs = load_wikitext("test", max_docs=500)

    # Collect activations at layer 8 (same as Phase 5)
    print("\nCollecting layer 8 activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, 8, a.device, a.max_tokens_train)
    test_data = collect_layer_data(model, tokenizer, test_docs, 8, a.device, a.max_tokens_test)

    results = {}
    t0 = time.time()

    # 7a: core comparison
    results["7a"] = run_7a(sae, train_data, test_data, seeds)

    # 7c: rank overlap
    results["7c"] = run_7c(results["7a"], seeds)

    # 7d: flagging (needs layer 11 data for confidence baseline)
    if a.all or a.causal:
        print("\nCollecting layer 11 test activations...")
        test_11 = collect_layer_data(model, tokenizer, test_docs, 11, a.device, a.max_tokens_test)

    if a.all:
        results["7d"] = run_7d(sae, test_data, test_11, train_data, seeds)

    # 7b: three-channel causal decomposition
    if a.causal:
        results["7b"] = run_7b(
            model, tokenizer, sae, train_data, test_data, test_11, a.device, seeds, a.max_tokens_test
        )

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save (deep-merge to avoid nuking prior results on partial reruns)
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    # Don't save raw scores (too large for JSON)
    save_results = {k: v for k, v in results.items() if k != "7a"}
    if "7a" in results:
        save_results["7a"] = {k: v for k, v in results["7a"].items() if k not in ("raw_scores", "sae_scores")}
    out_file = out / "sae_compare.json"
    existing = {}
    if out_file.exists():
        with open(out_file) as f:
            existing = json.load(f)
    _deep_merge(existing, save_results)
    with open(out_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
