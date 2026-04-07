"""Test whether the output-independent scaling trend survives a proportionally scaled predictor.

The Phase 8 output-side predictor uses a fixed 64-unit bottleneck (hidden_dim -> 64 -> 1).
As hidden_dim grows from 768 (GPT-2) to 1600 (GPT-2 XL), the bottleneck becomes more
constrained, which could inflate the output-independent fraction.

This script re-runs the output-control experiment with the bottleneck scaled proportionally
to hidden_dim, maintaining the 124M ratio (768/64 = 12:1).

Usage:
    uv run --extra transformer scripts/bottleneck_scaling.py
    uv run --extra transformer scripts/bottleneck_scaling.py --device mps
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from observe import partial_spearman
from transformer_observe import (
    GPT2_MODELS,
    collect_layer_data,
    load_wikitext,
    train_linear_binary,
)

SEEDS = [42, 43, 44]
RATIO = 12  # 768 / 64 = 12, the GPT-2 124M compression ratio


def train_output_predictor(train_data, bottleneck, seed, epochs=20):
    """Train MLP predictor on last-layer activations with given bottleneck size."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    acts = train_data["activations"]
    targets = torch.from_numpy(train_data["losses"]).float()
    n_feat = acts.size(1)

    predictor = torch.nn.Sequential(
        torch.nn.Linear(n_feat, bottleneck),
        torch.nn.ReLU(),
        torch.nn.Linear(bottleneck, 1),
    )
    opt = torch.optim.Adam(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)

    predictor.train()
    for _ep in range(epochs):
        for bx, by in dl:
            loss = F.mse_loss(predictor(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return predictor


def run_output_control(train_peak, test_peak, train_last, test_last, bottleneck, seeds):
    """Run output-control experiment with a specific bottleneck size."""
    controlled = []
    for seed in seeds:
        predictor = train_output_predictor(train_last, bottleneck, seed)
        predictor.eval()
        with torch.inference_mode():
            pred_scores = predictor(test_last["activations"]).squeeze(-1).numpy()

        head = train_linear_binary(train_peak, seed=seed)
        head.eval()
        with torch.inference_mode():
            obs_scores = head(test_peak["activations"]).squeeze(-1).numpy()

        rho, _ = partial_spearman(
            obs_scores,
            test_peak["losses"],
            [test_peak["max_softmax"], test_peak["activation_norm"], pred_scores],
        )
        controlled.append(float(rho))
    return controlled


def main():
    parser = argparse.ArgumentParser(description="Bottleneck scaling experiment")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    train_docs = load_wikitext("train", max_docs=2000)
    test_docs = load_wikitext("test", max_docs=500)

    # Known peak layers from Phase 8 results
    peak_layers = {"gpt2": 8, "gpt2-medium": 16, "gpt2-large": 24, "gpt2-xl": 34}

    results = {}

    for model_id, n_params_m in GPT2_MODELS:
        print(f"\n{'=' * 60}")
        print(f"  {model_id} ({n_params_m}M)")
        print(f"{'=' * 60}")

        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        model.eval()

        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        output_layer = n_layers - 1
        peak_layer = peak_layers[model_id]

        # Scale token budget as in Phase 8
        min_train = 150 * hidden_dim
        adj_train = max(min_train, int(200000 * (768 / hidden_dim)))
        adj_test = max(min_train // 2, int(100000 * (768 / hidden_dim)))
        print(f"  {n_layers} layers, hidden dim {hidden_dim}")
        print(f"  Tokens: {adj_train} train, {adj_test} test")

        fixed_bottleneck = 64
        proportional_bottleneck = hidden_dim // RATIO
        print(f"  Fixed bottleneck: {fixed_bottleneck}")
        print(f"  Proportional bottleneck: {proportional_bottleneck} (dim/{RATIO})")

        print(f"\n  Collecting layer {peak_layer} (peak)...")
        train_peak = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, adj_train)
        test_peak = collect_layer_data(model, tokenizer, test_docs, peak_layer, device, adj_test)

        print(f"  Collecting layer {output_layer} (output)...")
        train_last = collect_layer_data(model, tokenizer, train_docs, output_layer, device, adj_train)
        test_last = collect_layer_data(model, tokenizer, test_docs, output_layer, device, adj_test)

        # Standard partial correlation (no output control)
        standard = []
        for seed in SEEDS:
            head = train_linear_binary(train_peak, seed=seed)
            head.eval()
            with torch.inference_mode():
                scores = head(test_peak["activations"]).squeeze(-1).numpy()
            rho, _ = partial_spearman(
                scores, test_peak["losses"],
                [test_peak["max_softmax"], test_peak["activation_norm"]],
            )
            standard.append(float(rho))

        print(f"\n  Standard partial corr: {np.mean(standard):+.4f} ({[f'{r:+.4f}' for r in standard]})")

        # Fixed bottleneck (replication of Phase 8)
        print(f"\n  Output control (fixed {fixed_bottleneck}-unit bottleneck):")
        fixed = run_output_control(train_peak, test_peak, train_last, test_last, fixed_bottleneck, SEEDS)
        for seed, rho in zip(SEEDS, fixed):
            print(f"    seed {seed}: {rho:+.4f}")
        print(f"    mean: {np.mean(fixed):+.4f}")

        # Proportional bottleneck
        print(f"\n  Output control (proportional {proportional_bottleneck}-unit bottleneck):")
        proportional = run_output_control(train_peak, test_peak, train_last, test_last, proportional_bottleneck, SEEDS)
        for seed, rho in zip(SEEDS, proportional):
            print(f"    seed {seed}: {rho:+.4f}")
        print(f"    mean: {np.mean(proportional):+.4f}")

        fixed_mean = float(np.mean(fixed))
        prop_mean = float(np.mean(proportional))
        std_mean = float(np.mean(standard))
        fixed_frac = 1 - fixed_mean / std_mean if std_mean else 0
        prop_frac = 1 - prop_mean / std_mean if std_mean else 0

        print(f"\n  Output absorbs (fixed):        {fixed_frac:.0%}")
        print(f"  Output absorbs (proportional): {prop_frac:.0%}")

        results[model_id] = {
            "hidden_dim": hidden_dim,
            "peak_layer": peak_layer,
            "fixed_bottleneck": fixed_bottleneck,
            "proportional_bottleneck": proportional_bottleneck,
            "standard": {"mean": std_mean, "per_seed": standard},
            "output_controlled_fixed": {"mean": fixed_mean, "per_seed": fixed},
            "output_controlled_proportional": {"mean": prop_mean, "per_seed": proportional},
            "absorbed_fraction_fixed": round(fixed_frac, 3),
            "absorbed_fraction_proportional": round(prop_frac, 3),
        }

        del model, train_peak, test_peak, train_last, test_last
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<16} {'Dim':>5} {'Standard':>10} {'Fixed-64':>10} {'Prop':>10} {'Absorb(F)':>10} {'Absorb(P)':>10}")
    print(f"  {'-' * 73}")
    for model_id, r in results.items():
        print(
            f"  {model_id:<16} {r['hidden_dim']:>5} "
            f"{r['standard']['mean']:>+10.4f} "
            f"{r['output_controlled_fixed']['mean']:>+10.4f} "
            f"{r['output_controlled_proportional']['mean']:>+10.4f} "
            f"{r['absorbed_fraction_fixed']:>9.0%} "
            f"{r['absorbed_fraction_proportional']:>9.0%}"
        )

    out = Path("results/bottleneck_scaling.json")
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
