"""Quick diagnostic for Llama 3.1 8B on latest code.

Verifies whether the weak signal (+0.04) from the Colab run was
stale code or a genuine architectural result. Runs three layers
at 33%, 50%, 67% depth with the latest float32 cast and ex/dim
scaling.

Usage: uv run --extra transformer scripts/llama8b_quick_diagnostic.py
"""

import sys

sys.path.insert(0, "src")

import json

import torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_observe import collect_layer_data, evaluate_head, load_wikitext, train_linear_binary

MODEL_ID = "meta-llama/Llama-3.2-3B"  # Use 3B locally (8B needs more RAM)

print(f"Loading {MODEL_ID}...")
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
model.eval()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

N_LAYERS = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
print(f"{N_LAYERS} layers, {HIDDEN_DIM} dim, device={device}")

wiki_train = load_wikitext("train", max_docs=2000)
wiki_test = load_wikitext("test", max_docs=500)

MAX_TRAIN = max(150 * HIDDEN_DIM, 200000)
MAX_TEST = MAX_TRAIN // 2
print(f"Token budget: {MAX_TRAIN} ({MAX_TRAIN / HIDDEN_DIM:.0f} ex/dim)")

results = {}
for frac in [0.33, 0.50, 0.67]:
    layer = int(frac * N_LAYERS)
    tr = collect_layer_data(model, tokenizer, wiki_train, layer, device, MAX_TRAIN)
    te = collect_layer_data(model, tokenizer, wiki_test, layer, device, MAX_TEST)

    # Verify float32
    assert tr["activations"].dtype == torch.float32, f"activations are {tr['activations'].dtype}"

    head = train_linear_binary(tr, seed=42)
    head.eval()
    with torch.inference_mode():
        sc = head(te["activations"]).squeeze(-1).numpy()
    raw, _ = spearmanr(sc, te["losses"])
    _, partial, _ = evaluate_head(head, te)
    results[layer] = {"raw": float(raw), "partial": float(partial), "depth": frac}
    print(f"  layer {layer} ({frac:.0%}): raw={raw:+.4f} partial={partial:+.4f}")

# Save
output = {
    "model": MODEL_ID,
    "n_layers": N_LAYERS,
    "hidden_dim": HIDDEN_DIM,
    "device": device,
    "results": results,
}
with open("results/llama3b_diagnostic.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved results/llama3b_diagnostic.json")
print(json.dumps(output, indent=2))
