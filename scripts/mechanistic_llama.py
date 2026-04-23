"""Mechanistic analysis: Llama 3.2 1B vs 3B via mean-ablation patching.

Compares the high-observability 1B model against the low-observability 3B
model within the same family. Tests whether the 3B signal is never
generated or generated and then suppressed by a specific component.
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch

# Add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformer_observe import (
    load_wikitext,
    run_mechanistic_general,
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RUN_START = time.time()


def elapsed():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


MODELS = [
    {
        "model_id": "meta-llama/Llama-3.2-1B",
        "peak_layer": 13,
        "label": "llama_1b",
    },
    {
        "model_id": "meta-llama/Llama-3.2-3B",
        "peak_layer": 14,
        "label": "llama_3b",
    },
]

print(f"Device: {DEVICE}")
print(f"Models: {[m['label'] for m in MODELS]}")

# Load WikiText once
print(f"\nLoading WikiText-103... [{elapsed()}]")
train_docs = load_wikitext("train", max_docs=2000)
test_docs = load_wikitext("test", max_docs=500)
print(f"  {len(train_docs)} train, {len(test_docs)} test docs")

results = {}

for spec in MODELS:
    model_id = spec["model_id"]
    peak_layer = spec["peak_layer"]
    label = spec["label"]

    print(f"\n{'=' * 60}")
    print(f"  Loading {model_id} [{elapsed()}]")
    print(f"{'=' * 60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(DEVICE)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.1f}B params, {n_layers} layers, {hidden_dim} dim")

    # Scale token budgets by hidden dim
    min_ex_per_dim = 150
    adj_train = max(min_ex_per_dim * hidden_dim, 200000)
    adj_test = max(min_ex_per_dim * hidden_dim // 2, 100000)

    print(f"  Token budget: train={adj_train}, eval_budget=15000")
    print(f"  Peak layer: {peak_layer} ({peak_layer / n_layers * 100:.0f}% depth)")

    mech = run_mechanistic_general(
        model,
        tokenizer,
        DEVICE,
        train_docs,
        test_docs,
        adj_train,
        adj_test,
        peak_layer=peak_layer,
        eval_budget=15000,
    )

    mech["model"] = model_id
    mech["label"] = label
    mech["n_params_b"] = round(n_params / 1e9, 1)
    mech["n_layers"] = n_layers
    mech["hidden_dim"] = hidden_dim
    results[label] = mech

    # Free memory
    del model, tokenizer
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    print(f"\n  {label} complete [{elapsed()}]")

# Save
out_path = Path(__file__).resolve().parent.parent / "results" / "mechanistic_llama_comparison.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out_path}")
print(f"Total time: {elapsed()}")

# Quick comparison summary
print(f"\n{'=' * 60}")
print("  COMPARISON SUMMARY")
print(f"{'=' * 60}")
for label, mech in results.items():
    print(f"\n  {label} (peak L{mech['peak_layer']}):")
    ablation = mech["ablation_results"]
    # Find top 3 components by absolute obs_resid_delta
    all_components = []
    for layer_str, comps in ablation.items():
        for comp_name, vals in comps.items():
            all_components.append((int(layer_str), comp_name, vals["obs_resid_delta"]))
    all_components.sort(key=lambda x: abs(x[2]), reverse=True)
    print("  Top 3 components by |obs_resid_delta|:")
    for layer, comp, delta in all_components[:3]:
        print(f"    L{layer} {comp}: {delta:+.4f}")

    if mech.get("composition_results"):
        print("  Composition (all_top):")
        at = mech["composition_results"].get("all_top", {})
        if at:
            print(
                f"    actual={at['obs_resid_delta']:+.4f}  expected={at['expected_additive']:+.4f}  interaction={at['interaction']:+.4f}"
            )
