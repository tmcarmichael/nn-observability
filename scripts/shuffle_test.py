"""Shuffle test: train probes on randomized labels to verify target validity.

Trains 10 probes on GPT-2 124M with randomly permuted binary targets.
If the real signal is genuine, shuffled probes should produce near-zero
partial correlation. The paper claims +0.008 +/- 0.012 (shuffled) vs
+0.334 (real, single seed on this data).

Runs on CPU/MPS.

Usage: cd nn-observability && uv run python scripts/shuffle_test.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from probe import (
    build_batches,
    collect_single_layer_fast,
    compute_loss_residuals,
    evaluate_head,
    load_wikitext,
    partial_spearman,
    pretokenize,
    train_linear_binary,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_PERMUTATIONS = 10
LAYER = 11  # GPT-2 124M peak layer (hardening layer)
SEED_BASE = 100  # seeds for permutation RNG (distinct from probe seeds)

print(f"=== Shuffle test: GPT-2 124M, layer {LAYER}, {N_PERMUTATIONS} permutations ===")
print(f"Device: {DEVICE}")
t0 = time.time()

# Load model
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(DEVICE)
model.eval()
print(f"Model loaded ({time.time() - t0:.0f}s)")

# Load data
train_docs = load_wikitext("train", max_docs=2000)
test_docs = load_wikitext("test", max_docs=500)

train_encoded = pretokenize(train_docs, tokenizer, max_length=512)
test_encoded = pretokenize(test_docs, tokenizer, max_length=512)

train_batches = build_batches(train_encoded, batch_size=8)
test_batches = build_batches(test_encoded, batch_size=8)

# Collect activations at peak layer
print(f"Collecting activations at layer {LAYER}...")
train_data = collect_single_layer_fast(model, train_batches, LAYER, max_tokens=200000, device=DEVICE)
test_data = collect_single_layer_fast(model, test_batches, LAYER, max_tokens=50000, device=DEVICE)
print(f"  train: {len(train_data['losses'])} positions, test: {len(test_data['losses'])} positions")

# Real probe (baseline)
print("\nTraining real probe (seed 42)...")
real_head = train_linear_binary(train_data, seed=42)
_, real_pcorr, _ = evaluate_head(real_head, test_data)
print(f"  Real pcorr: {real_pcorr:+.3f}")

# Shuffle test: permute binary targets before training
print(f"\nRunning {N_PERMUTATIONS} shuffle permutations...")
shuffle_pcorrs = []

for i in range(N_PERMUTATIONS):
    rng = np.random.RandomState(SEED_BASE + i)

    # Compute real residuals, then permute the binary target
    residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    binary_targets = (residuals > 0).astype(np.float32)
    rng.shuffle(binary_targets)  # in-place permutation

    # Build a modified train_data with shuffled targets
    # We need to override the training so it uses our shuffled targets
    # instead of recomputing from residuals
    acts = train_data["activations"]
    targets_t = torch.from_numpy(binary_targets)

    torch.manual_seed(42)  # same init for fair comparison
    np.random.seed(42)
    head = torch.nn.Linear(acts.size(1), 1)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, targets_t)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    head.train()
    for _ in range(20):
        for bx, by in dl:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    head.eval()

    # Evaluate on real (unshuffled) test data
    with torch.inference_mode():
        scores = head(test_data["activations"]).squeeze(-1).numpy()
    rho, _ = partial_spearman(
        scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
    )
    shuffle_pcorrs.append(rho)
    print(f"  perm {i + 1:2d}: pcorr = {rho:+.4f}")

elapsed = time.time() - t0
mean_shuffle = np.mean(shuffle_pcorrs)
std_shuffle = np.std(shuffle_pcorrs)
print("\n=== Results ===")
print(f"  Real probe:     {real_pcorr:+.4f}")
print(f"  Shuffle mean:   {mean_shuffle:+.4f} +/- {std_shuffle:.4f}")
print(f"  Ratio:          {abs(real_pcorr / mean_shuffle) if abs(mean_shuffle) > 1e-6 else 'inf'}x")
print(f"  Time:           {elapsed:.0f}s")

# Save results
output = {
    "model": "openai-community/gpt2",
    "layer": LAYER,
    "n_permutations": N_PERMUTATIONS,
    "seed_base": SEED_BASE,
    "real_pcorr": real_pcorr,
    "shuffle_pcorrs": shuffle_pcorrs,
    "shuffle_mean": float(mean_shuffle),
    "shuffle_std": float(std_shuffle),
    "n_train_positions": len(train_data["losses"]),
    "n_test_positions": len(test_data["losses"]),
    "provenance": {
        "script": "scripts/shuffle_test.py",
        "device": str(DEVICE),
        "torch_version": torch.__version__,
    },
}

out_path = Path(__file__).resolve().parent.parent / "results" / "shuffle_test_gpt2.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {out_path}")
