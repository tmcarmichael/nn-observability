"""Mechanistic analysis: Mistral 7B via mean-ablation patching.

Compares Mistral 7B's component-level signal path against the Llama 1B/3B
mechanistic runs. Mistral uses GQA like Llama 3B but retains high
observability, so this isolates whether the 3B suppression pattern is
GQA-driven or family-specific.
"""

import datetime as _dt
import gc
import json
import sys
import time
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
RESULTS_DIR = _here.parent / "results"
sys.path.insert(0, str(_here.parent / "src"))


def _resolve_out(name_or_path):
    p = Path(name_or_path)
    if p.is_absolute():
        return p
    base = Path("/workspace") if Path("/workspace").exists() else _here.parent / "results"
    return base / p


def _revision_kwargs(model_id):
    manifest = _here.parent / "results" / "model_revisions.json"
    if not manifest.exists():
        return {}
    commit = json.loads(manifest.read_text()).get("models", {}).get(model_id, {}).get("commit")
    return {"revision": commit} if commit else {}


OUT_PATH = _resolve_out("mistral-7b-v0.3_mechanistic.json")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"Output: {OUT_PATH}")

from transformer_observe import (
    load_wikitext,
    run_mechanistic_general,
)

if not torch.cuda.is_available():
    import sys

    sys.exit(
        "mechanistic_mistral.py produces paper-quality results and requires CUDA. "
        "Run on a CUDA-enabled host (Colab GPU, runpod, local CUDA box)."
    )
DEVICE = "cuda"
RUN_START = time.time()


def elapsed():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


MODEL_ID = "mistralai/Mistral-7B-v0.3"
PEAK_LAYER = 22

# Fail-fast before model download.
if not RESULTS_DIR.is_dir():
    sys.exit(f"RESULTS_DIR not found: {RESULTS_DIR}")
manifest = RESULTS_DIR / "model_revisions.json"
if not manifest.is_file():
    sys.exit(f"Manifest missing: {manifest}")
if MODEL_ID not in json.loads(manifest.read_text()).get("models", {}):
    sys.exit(f"Model {MODEL_ID!r} not in manifest")

print(f"=== Mechanistic analysis: Mistral 7B [{elapsed()}] ===")
print(f"Device: {DEVICE}")

# Load data
print(f"\nLoading WikiText-103... [{elapsed()}]")
train_docs = load_wikitext("train", max_docs=2000)
test_docs = load_wikitext("test", max_docs=500)
print(f"  {len(train_docs)} train, {len(test_docs)} test docs")

# Load model
print(f"\nLoading {MODEL_ID}... [{elapsed()}]")

from transformers import AutoModelForCausalLM, AutoTokenizer

_rev_kw = _revision_kwargs(MODEL_ID)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **_rev_kw)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float16
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation="sdpa", **_rev_kw).to(
    DEVICE
)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
n_params = sum(p.numel() for p in model.parameters())
print(f"  {n_params / 1e9:.1f}B params, {n_layers} layers, {hidden_dim} dim")

# Scale token budgets
min_ex_per_dim = 150
adj_train = max(min_ex_per_dim * hidden_dim, 200000)
adj_test = max(min_ex_per_dim * hidden_dim // 2, 100000)

print(f"  Token budget: train={adj_train}, eval_budget=15000")
print(f"  Peak layer: {PEAK_LAYER} ({PEAK_LAYER / n_layers * 100:.0f}% depth)")

# Run mechanistic analysis
mech = run_mechanistic_general(
    model,
    tokenizer,
    DEVICE,
    train_docs,
    test_docs,
    adj_train,
    adj_test,
    peak_layer=PEAK_LAYER,
    eval_budget=15000,
)

mech["model"] = MODEL_ID
mech["label"] = "mistral_7b"
mech["n_params_b"] = round(n_params / 1e9, 1)
mech["n_layers"] = n_layers
mech["hidden_dim"] = hidden_dim
_revision = _rev_kw.get("revision") or getattr(model.config, "_commit_hash", None)
if not _revision:
    raise RuntimeError(
        f"Could not resolve model revision for {MODEL_ID}: pin via results/model_revisions.json "
        "or upgrade transformers (model.config._commit_hash unset)."
    )
mech["provenance"] = {
    "model_revision": _revision,
    "script": "scripts/mechanistic_mistral.py",
    "timestamp": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    "value_source": "runtime",
    "device": str(DEVICE),
}

# Save
with open(OUT_PATH, "w") as f:
    json.dump(mech, f, indent=2)
print(f"\nSaved {OUT_PATH}")

del model, tokenizer
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# Summary
print(f"\n{'=' * 60}")
print(f"  SUMMARY: Mistral 7B (peak L{PEAK_LAYER})")
print(f"{'=' * 60}")

ablation = mech["ablation_results"]
all_components = []
for layer_str, comps in ablation.items():
    for comp_name, vals in comps.items():
        all_components.append((int(layer_str), comp_name, vals["obs_resid_delta"]))
all_components.sort(key=lambda x: abs(x[2]), reverse=True)

print("\n  Top 10 components by |obs_resid_delta|:")
for layer, comp, delta in all_components[:10]:
    print(f"    L{layer} {comp}: {delta:+.4f}")

# Early-layer comparison block, matching Llama's collapsed-class peak region.
print("\n  L0/L1 comparison (vs Llama):")
for layer in [0, 1]:
    for comp in ["attn", "mlp"]:
        delta = ablation[str(layer)][comp]["obs_resid_delta"]
        loss_d = ablation[str(layer)][comp]["loss_delta"]
        print(f"    L{layer} {comp}: obs_resid={delta:+.4f}  loss={loss_d:+.4f}")

if mech.get("composition_results"):
    print("\n  Composition:")
    for gname, vals in mech["composition_results"].items():
        print(
            f"    {gname}: actual={vals['obs_resid_delta']:+.4f}  expected={vals['expected_additive']:+.4f}  interaction={vals['interaction']:+.4f}"
        )

print(f"\nTotal time: {elapsed()}")
