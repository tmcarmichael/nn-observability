"""Controlled training experiment: same data, different architecture, measure observability.

Trains two 150M-parameter models from scratch on identical OpenWebText data:
  - Config A (Qwen-style): MHA (all heads are query heads)
  - Config B (Llama-style): GQA (4 KV heads, rest are query heads)

Everything else is identical: layers, hidden dim, intermediate size, normalization,
activation function, tokenizer, data, optimizer, schedule, seeds.

If observability diverges, architecture is causal. If it doesn't, training data
was the confound all along.

GPU: H200 (144GB). Total time: ~12-16 hours (two training runs + evaluation).

Usage: pip install transformers datasets scipy scikit-learn accelerate && python controlled_training.py
"""

import gc
import json
import math
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


# ===========================================================================
# Probe functions (from run_model.py)
# ===========================================================================


def partial_spearman(x, y, covariates):
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates])
    rc = np.column_stack([rc, np.ones(len(rc))])
    coef_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    coef_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    r, p = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)
    return float(r), float(p)


def compute_loss_residuals(losses, max_softmax, activation_norm):
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    beta = np.linalg.lstsq(X, losses, rcond=None)[0]
    return losses - X @ beta


def train_probe(acts, losses, max_softmax, activation_norm, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts_dev = acts.to(DEVICE)
    residuals = compute_loss_residuals(losses, max_softmax, activation_norm)
    targets = torch.from_numpy((residuals > 0).astype(np.float32)).to(DEVICE)
    head = torch.nn.Linear(acts.size(1), 1).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts_dev, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.cpu()


# ===========================================================================
# Model configs: matched parameters, different architecture
# ===========================================================================

# ~150M parameters each
# Both use: RMSNorm, SwiGLU activation, rotary embeddings, same vocab
# Difference: Config A has full MHA, Config B has GQA (4 KV heads)

MODEL_CONFIGS = {
    "mha": {
        "name": "MHA (Qwen-style)",
        "num_hidden_layers": 16,
        "hidden_size": 768,
        "intermediate_size": 2048,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,  # MHA: all heads are KV heads
        "max_position_embeddings": 1024,
    },
    "gqa": {
        "name": "GQA (Llama-style)",
        "num_hidden_layers": 16,
        "hidden_size": 768,
        "intermediate_size": 2389,  # larger MLP compensates for fewer KV params (matches 151.9M total)
        "num_attention_heads": 12,
        "num_key_value_heads": 4,  # GQA: 4 KV heads shared across 12 query heads
        "max_position_embeddings": 1024,
    },
}

# Training config (identical for both)
TRAIN_CONFIG = {
    "total_tokens": 1_000_000_000,  # 1B tokens
    "batch_size": 32,
    "seq_length": 512,
    "lr": 3e-4,
    "warmup_steps": 500,
    "weight_decay": 0.1,
    "grad_accum": 4,
    "seed": 42,
    "eval_every": 500,
    "checkpoint_every": 2000,
}

# Probe eval config
PROBE_SEEDS = list(range(43, 50))
TARGET_EX_PER_DIM = 350


# ===========================================================================
# Data loading
# ===========================================================================


def load_openwebtext(tokenizer, seq_length, max_tokens):
    from datasets import load_dataset

    print(f"  Loading OpenWebText (target {max_tokens / 1e6:.0f}M tokens)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    all_ids = []
    total = 0
    for row in ds:
        ids = tokenizer.encode(row["text"])
        all_ids.extend(ids)
        total = len(all_ids)
        if total >= max_tokens + 100000:
            break

    # Chunk into sequences
    n_seqs = total // seq_length
    all_ids = all_ids[: n_seqs * seq_length]
    data = torch.tensor(all_ids, dtype=torch.long).view(n_seqs, seq_length)
    print(f"  {n_seqs} sequences, {total / 1e6:.1f}M tokens")
    return data


# ===========================================================================
# Training loop
# ===========================================================================


def train_model(config_name, config, train_data, val_data, tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM

    print(f"\n{'=' * 60}")
    print(f"Training {config['name']} [{elapsed_str()}]")
    print(f"{'=' * 60}")

    # Create model from config
    model_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        max_position_embeddings=config["max_position_embeddings"],
        rms_norm_eps=1e-6,
        hidden_act="silu",
        tie_word_embeddings=True,
    )

    torch.manual_seed(TRAIN_CONFIG["seed"])
    model = LlamaForCausalLM(model_config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {n_params:.1f}M parameters")

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        betas=(0.9, 0.95),
    )

    # LR schedule: linear warmup + cosine decay
    total_steps = len(train_data) // (TRAIN_CONFIG["batch_size"] * TRAIN_CONFIG["grad_accum"])
    warmup = TRAIN_CONFIG["warmup_steps"]

    def lr_schedule(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    # Training
    model.train()
    step = 0
    best_val_loss = float("inf")
    train_losses = []
    checkpoint_dir = Path(f"/workspace/checkpoint_{config_name}")
    checkpoint_dir.mkdir(exist_ok=True)

    indices = torch.randperm(len(train_data))
    pos = 0

    print(f"  Total steps: {total_steps}, warmup: {warmup}")

    for step in range(1, total_steps + 1):
        opt.zero_grad()
        accum_loss = 0.0

        for _ in range(TRAIN_CONFIG["grad_accum"]):
            if pos + TRAIN_CONFIG["batch_size"] > len(indices):
                indices = torch.randperm(len(train_data))
                pos = 0

            batch_idx = indices[pos : pos + TRAIN_CONFIG["batch_size"]]
            pos += TRAIN_CONFIG["batch_size"]

            input_ids = train_data[batch_idx].to(DEVICE)
            labels = input_ids.clone()

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / TRAIN_CONFIG["grad_accum"]
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        train_losses.append(accum_loss)

        if step % 100 == 0:
            avg = np.mean(train_losses[-100:])
            lr_now = scheduler.get_last_lr()[0]
            print(f"    step {step}/{total_steps}: loss={avg:.4f}, lr={lr_now:.2e} [{elapsed_str()}]")

        if step % TRAIN_CONFIG["eval_every"] == 0:
            model.eval()
            val_losses = []
            with torch.inference_mode():
                for i in range(0, min(len(val_data), 200), TRAIN_CONFIG["batch_size"]):
                    batch = val_data[i : i + TRAIN_CONFIG["batch_size"]].to(DEVICE)
                    out = model(input_ids=batch, labels=batch)
                    val_losses.append(out.loss.item())
            val_loss = np.mean(val_losses)
            ppl = math.exp(min(val_loss, 20))
            print(f"    [eval] val_loss={val_loss:.4f}, ppl={ppl:.1f} [{elapsed_str()}]")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            model.train()

        if step % TRAIN_CONFIG["checkpoint_every"] == 0:
            ckpt_path = checkpoint_dir / f"step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"    [checkpoint] {ckpt_path}")

    # Save final
    final_path = checkpoint_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    print(
        f"  Training complete. Final loss: {np.mean(train_losses[-100:]):.4f}, best val: {best_val_loss:.4f}"
    )

    return model, model_config, best_val_loss, train_losses


# ===========================================================================
# Probe evaluation
# ===========================================================================


def evaluate_observability(model, tokenizer, config_name, config):
    from datasets import load_dataset

    print(f"\n  === Evaluating observability: {config['name']} [{elapsed_str()}] ===")

    hidden_dim = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    max_tokens = TARGET_EX_PER_DIM * hidden_dim

    # Load WikiText eval data
    print("  Loading WikiText...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    docs = []
    current = []
    for row in ds:
        text = row["text"]
        if text.strip() == "" and current:
            docs.append("\n".join(current))
            current = []
            if len(docs) >= 500:
                break
        elif text.strip():
            current.append(text)

    # Collect activations at each layer (sweep)
    model.eval()
    layer_profile = {}

    for layer_idx in range(n_layers):
        captured = {}

        def make_hook(store=captured):  # noqa: B023
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if isinstance(h, tuple):
                    h = h[0]
                store[0] = h

            return hook_fn

        handle = model.model.layers[layer_idx].register_forward_hook(make_hook())

        all_acts, all_losses, all_sm, all_norms = [], [], [], []
        total = 0

        for doc in docs:
            if total >= max_tokens:
                break
            if not doc.strip():
                continue
            tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            input_ids = tokens["input_ids"].to(DEVICE)
            if input_ids.size(1) < 2:
                continue

            with torch.inference_mode():
                outputs = model(input_ids)

            h = captured[0][0, :-1, :].float().cpu()
            logits = outputs.logits[0, :-1, :]
            labels = input_ids[0, 1:]
            losses = F.cross_entropy(logits, labels, reduction="none").cpu()
            sm = F.softmax(logits, dim=-1).max(dim=-1).values.cpu()

            all_acts.append(h)
            all_losses.append(losses)
            all_sm.append(sm)
            all_norms.append(h.norm(dim=-1))
            total += h.size(0)

        handle.remove()

        if total < 1000:
            print(f"    L{layer_idx}: insufficient data ({total} tokens)")
            continue

        acts = torch.cat(all_acts)[:max_tokens]
        losses = torch.cat(all_losses).numpy()[:max_tokens]
        sm = torch.cat(all_sm).numpy()[:max_tokens]
        norms = torch.cat(all_norms).numpy()[:max_tokens]

        # Train probe with layer selection seed
        head = train_probe(acts, losses, sm, norms, seed=42)
        head.eval()
        with torch.inference_mode():
            scores = head(acts).squeeze(-1).numpy()
        rho, _ = partial_spearman(scores, losses, [sm, norms])
        layer_profile[layer_idx] = float(rho)
        print(f"    L{layer_idx}: pcorr={rho:+.4f}")

        del acts, all_acts, all_losses, all_sm, all_norms
        gc.collect()

    if not layer_profile:
        print("  ERROR: no layers evaluated")
        return {}

    # Multi-seed eval at peak layer
    peak_layer = max(layer_profile, key=layer_profile.get)
    print(f"\n  Peak: L{peak_layer} ({layer_profile[peak_layer]:+.4f})")
    print("  Multi-seed eval...")

    # Re-collect at peak
    captured = {}

    def make_hook():
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if isinstance(h, tuple):
                h = h[0]
            captured[0] = h

        return hook_fn

    handle = model.model.layers[peak_layer].register_forward_hook(make_hook())

    all_acts, all_losses, all_sm, all_norms = [], [], [], []
    total = 0
    for doc in docs:
        if total >= max_tokens:
            break
        if not doc.strip():
            continue
        tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
        input_ids = tokens["input_ids"].to(DEVICE)
        if input_ids.size(1) < 2:
            continue
        with torch.inference_mode():
            outputs = model(input_ids)
        h = captured[0][0, :-1, :].float().cpu()
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[0, 1:]
        losses = F.cross_entropy(logits, labels, reduction="none").cpu()
        sm = F.softmax(logits, dim=-1).max(dim=-1).values.cpu()
        all_acts.append(h)
        all_losses.append(losses)
        all_sm.append(sm)
        all_norms.append(h.norm(dim=-1))
        total += h.size(0)

    handle.remove()

    acts = torch.cat(all_acts)[:max_tokens]
    losses_np = torch.cat(all_losses).numpy()[:max_tokens]
    sm_np = torch.cat(all_sm).numpy()[:max_tokens]
    norms_np = torch.cat(all_norms).numpy()[:max_tokens]

    seed_rhos = []
    seed_scores_list = []
    for seed in PROBE_SEEDS:
        head = train_probe(acts, losses_np, sm_np, norms_np, seed=seed)
        head.eval()
        with torch.inference_mode():
            scores = head(acts).squeeze(-1).numpy()
        rho, _ = partial_spearman(scores, losses_np, [sm_np, norms_np])
        seed_rhos.append(float(rho))
        seed_scores_list.append(scores)

    # Seed agreement
    pw = [
        float(spearmanr(seed_scores_list[i], seed_scores_list[j])[0])
        for i in range(len(PROBE_SEEDS))
        for j in range(i + 1, len(PROBE_SEEDS))
    ]

    result = {
        "config_name": config_name,
        "config": config,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "layer_profile": {str(k): v for k, v in sorted(layer_profile.items())},
        "peak_layer": peak_layer,
        "peak_layer_frac": round(peak_layer / n_layers, 2),
        "partial_corr": {
            "mean": float(np.mean(seed_rhos)),
            "std": float(np.std(seed_rhos)),
            "per_seed": seed_rhos,
            "n_seeds": len(PROBE_SEEDS),
        },
        "seed_agreement": {"mean": float(np.mean(pw))},
    }

    print(f"  pcorr: {result['partial_corr']['mean']:+.4f} +/- {result['partial_corr']['std']:.4f}")
    print(f"  sagree: {result['seed_agreement']['mean']:.4f}")
    print(f"  peak: L{peak_layer} ({result['peak_layer_frac'] * 100:.0f}%)")

    return result


# ===========================================================================
# Main
# ===========================================================================

print(f"=== Controlled training experiment [{elapsed_str()}] ===")
print(f"Device: {DEVICE}")

from transformers import AutoTokenizer

# Shared tokenizer (GPT-2, both models use the same vocabulary)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer: {tokenizer.name_or_path}, vocab={tokenizer.vocab_size}")

# Load data once, share between both models
total_tokens = TRAIN_CONFIG["total_tokens"]
seq_length = TRAIN_CONFIG["seq_length"]
print(f"\n=== Loading data [{elapsed_str()}] ===")
all_data = load_openwebtext(tokenizer, seq_length, total_tokens + 500000)

# Split: 95% train, 5% val
n_val = max(200, len(all_data) // 20)
val_data = all_data[:n_val]
train_data = all_data[n_val:]
print(f"  Train: {len(train_data)} seqs, Val: {len(val_data)} seqs")
del all_data

# Train both models
results = {}

for config_name, config in MODEL_CONFIGS.items():
    model, model_config, best_val_loss, train_losses = train_model(
        config_name, config, train_data, val_data, tokenizer
    )

    # Evaluate observability
    obs_result = evaluate_observability(model, tokenizer, config_name, config)
    obs_result["best_val_loss"] = best_val_loss
    obs_result["final_train_loss"] = float(np.mean(train_losses[-100:]))
    results[config_name] = obs_result

    # Cleanup
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

# === Comparison ===
print(f"\n{'=' * 60}")
print(f"=== Results [{elapsed_str()}] ===")
print(f"{'=' * 60}")

for _name, r in results.items():
    pc = r["partial_corr"]
    print(f"\n{r['config']['name']}:")
    print(f"  Val loss: {r['best_val_loss']:.4f}, Train loss: {r['final_train_loss']:.4f}")
    print(f"  pcorr: {pc['mean']:+.4f} +/- {pc['std']:.4f}")
    print(f"  sagree: {r['seed_agreement']['mean']:.4f}")
    print(f"  Peak: L{r['peak_layer']} ({r['peak_layer_frac'] * 100:.0f}%)")

mha_pcorr = results["mha"]["partial_corr"]["mean"]
gqa_pcorr = results["gqa"]["partial_corr"]["mean"]
print(f"\n  MHA pcorr: {mha_pcorr:+.4f}")
print(f"  GQA pcorr: {gqa_pcorr:+.4f}")
print(f"  Delta: {mha_pcorr - gqa_pcorr:+.4f}")

if abs(mha_pcorr - gqa_pcorr) > 0.05:
    print("  DIVERGENCE: architecture causes observability difference on identical data")
else:
    print("  NO DIVERGENCE: both architectures produce similar observability")

# Save
output = {
    "experiment": "controlled_training",
    "description": "Same data, different architecture (MHA vs GQA), measure observability",
    "tokenizer": "openai-community/gpt2",
    "train_tokens": total_tokens,
    "train_config": TRAIN_CONFIG,
    "model_configs": MODEL_CONFIGS,
    "results": results,
    "comparison": {
        "mha_pcorr": mha_pcorr,
        "gqa_pcorr": gqa_pcorr,
        "delta": mha_pcorr - gqa_pcorr,
        "divergent": abs(mha_pcorr - gqa_pcorr) > 0.05,
    },
}

out_path = Path("/workspace/controlled_training_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {out_path}")
print(f"Total time: {elapsed_str()}")
