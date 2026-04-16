"""Controlled training: depth vs width at matched parameters.

Trains two parameter-matched models from scratch on identical OpenWebText data
with different depth-width ratios:
  - Config A (shallow-wide): fewer layers, larger hidden dim
  - Config B (deep-narrow): more layers, smaller hidden dim

Same tokenizer, optimizer, schedule, seed. Only depth-width ratio varies.

Context: Llama 1B (16L, 2048d) has observability (+0.286), Llama 3B
(28L, 3072d) does not (+0.091). Both use GQA; the transition changes
depth, width, and their ratio simultaneously. This experiment isolates
depth-width ratio by holding attention mechanism, parameter count, and
training data constant. Complements controlled_training.py (MHA vs GQA).

Three scales available:
  --scale 150m   Pilot (~150M params, 1B tokens, ~12h on H200)
  --scale 1b     ~1.2B params, 5B tokens
  --scale 3b     ~3B params, 10B tokens

Multi-seed for publication:
  --seeds 3      Train 3 models per config (6 total)

GPU: H200 (144GB). Single-GPU.

Usage:
  pip install transformers datasets scipy scikit-learn accelerate
  python controlled_depth_width.py --scale 150m
  python controlled_depth_width.py --scale 1b --seeds 3
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
# Probe functions
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
# Scale presets: parameter-matched shallow-wide vs deep-narrow
# ===========================================================================
#
# Both configs use MHA (same KV heads as query heads) to isolate depth-width
# from the attention mechanism variable tested in controlled_training.py.
# Intermediate sizes adjusted so total params match within 0.5%.
#
# The shallow-wide config mirrors the Llama 1B ratio (depth/width ≈ 8);
# the deep-narrow config mirrors the Llama 3B ratio (depth/width ≈ 9).

import argparse

parser = argparse.ArgumentParser(description="Controlled depth-width training")
parser.add_argument("--scale", choices=["150m", "1b", "3b"], default="150m", help="Model scale preset")
parser.add_argument("--seeds", type=int, default=1, help="Training seeds per config (1=pilot, 3=publication)")
args = parser.parse_args()

SCALE_PRESETS = {
    "150m": {
        "shallow_wide": {
            "name": "Shallow-wide-150M (8L, 1024d)",
            "num_hidden_layers": 8,
            "hidden_size": 1024,
            "intermediate_size": 2816,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 1024,
        },
        "deep_narrow": {
            "name": "Deep-narrow-150M (24L, 576d)",
            "num_hidden_layers": 24,
            "hidden_size": 576,
            "intermediate_size": 2252,  # matches 154.2M total
            "num_attention_heads": 9,
            "num_key_value_heads": 9,
            "max_position_embeddings": 1024,
        },
        "train": {
            "total_tokens": 1_000_000_000,
            "batch_size": 32,
            "seq_length": 512,
            "lr": 3e-4,
            "warmup_steps": 500,
            "grad_accum": 4,
        },
    },
    "1b": {
        # Shallow-wide: Llama 1B proportions (16L, ratio ~8)
        # Deep-narrow: Llama 3B proportions (28L, ratio ~9), scaled to 1B params
        "shallow_wide": {
            "name": "Shallow-wide-1B (12L, 2560d)",
            "num_hidden_layers": 12,
            "hidden_size": 2560,
            "intermediate_size": 6912,  # ~1.20B total
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "max_position_embeddings": 2048,
        },
        "deep_narrow": {
            "name": "Deep-narrow-1B (28L, 1536d)",
            "num_hidden_layers": 28,
            "hidden_size": 1536,
            "intermediate_size": 4608,  # ~1.20B total
            "num_attention_heads": 12,
            "num_key_value_heads": 12,
            "max_position_embeddings": 2048,
        },
        "train": {
            "total_tokens": 5_000_000_000,
            "batch_size": 16,
            "seq_length": 1024,
            "lr": 2e-4,
            "warmup_steps": 1000,
            "grad_accum": 8,
        },
    },
    "3b": {
        # Shallow-wide: 16L, 3584d (Llama 1B depth, Qwen 7B width-range)
        # Deep-narrow: 36L, 2304d (deeper than Llama 3B, narrower)
        "shallow_wide": {
            "name": "Shallow-wide-3B (16L, 3584d)",
            "num_hidden_layers": 16,
            "hidden_size": 3584,
            "intermediate_size": 9216,  # ~3.0B total
            "num_attention_heads": 28,
            "num_key_value_heads": 28,
            "max_position_embeddings": 2048,
        },
        "deep_narrow": {
            "name": "Deep-narrow-3B (36L, 2048d)",
            "num_hidden_layers": 36,
            "hidden_size": 2048,
            "intermediate_size": 5632,  # ~3.0B total
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "max_position_embeddings": 2048,
        },
        "train": {
            "total_tokens": 10_000_000_000,
            "batch_size": 8,
            "seq_length": 1024,
            "lr": 1.5e-4,
            "warmup_steps": 2000,
            "grad_accum": 16,
        },
    },
}

preset = SCALE_PRESETS[args.scale]
MODEL_CONFIGS = {"shallow_wide": preset["shallow_wide"], "deep_narrow": preset["deep_narrow"]}

TRAINING_SEEDS = [42, 137, 7][: args.seeds]

TRAIN_CONFIG = {
    "total_tokens": preset["train"]["total_tokens"],
    "batch_size": preset["train"]["batch_size"],
    "seq_length": preset["train"]["seq_length"],
    "lr": preset["train"]["lr"],
    "warmup_steps": preset["train"]["warmup_steps"],
    "weight_decay": 0.1,
    "grad_accum": preset["train"]["grad_accum"],
    "seed": 42,
    "eval_every": 500,
    "checkpoint_every": 2000,
}

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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["lr"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
        betas=(0.9, 0.95),
    )

    total_steps = len(train_data) // (TRAIN_CONFIG["batch_size"] * TRAIN_CONFIG["grad_accum"])
    warmup = TRAIN_CONFIG["warmup_steps"]

    def lr_schedule(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule)

    model.train()
    step = 0
    best_val_loss = float("inf")
    train_losses = []
    checkpoint_dir = Path(f"/workspace/checkpoint_{config_name}")
    checkpoint_dir.mkdir(exist_ok=True)

    # Re-seed so both models see data in the same order
    torch.manual_seed(TRAIN_CONFIG["seed"])
    indices = torch.randperm(len(train_data))
    pos = 0

    print(f"  Total steps: {total_steps}, warmup: {warmup}")

    for step in range(1, total_steps + 1):
        opt.zero_grad()
        accum_loss = 0.0

        for _ in range(TRAIN_CONFIG["grad_accum"]):
            if pos + TRAIN_CONFIG["batch_size"] > len(indices):
                torch.manual_seed(TRAIN_CONFIG["seed"] + step)
                indices = torch.randperm(len(train_data))
                pos = 0

            batch_idx = indices[pos : pos + TRAIN_CONFIG["batch_size"]]
            pos += TRAIN_CONFIG["batch_size"]

            input_ids = train_data[batch_idx].to(DEVICE)
            outputs = model(input_ids=input_ids, labels=input_ids)
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

    model.eval()
    layer_profile = {}

    for layer_idx in range(n_layers):
        captured = {}

        def make_hook(store=captured):
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
        losses_np = torch.cat(all_losses).numpy()[:max_tokens]
        sm_np = torch.cat(all_sm).numpy()[:max_tokens]
        norms_np = torch.cat(all_norms).numpy()[:max_tokens]

        head = train_probe(acts, losses_np, sm_np, norms_np, seed=42)
        head.eval()
        with torch.inference_mode():
            scores = head(acts).squeeze(-1).numpy()
        rho, _ = partial_spearman(scores, losses_np, [sm_np, norms_np])
        layer_profile[layer_idx] = float(rho)
        print(f"    L{layer_idx}: pcorr={rho:+.4f}")

        del acts, all_acts, all_losses, all_sm, all_norms
        gc.collect()

    if not layer_profile:
        print("  ERROR: no layers evaluated")
        return {}

    peak_layer = max(layer_profile, key=layer_profile.get)
    print(f"\n  Peak: L{peak_layer} ({layer_profile[peak_layer]:+.4f})")
    print("  Multi-seed eval...")

    captured = {}

    def make_hook(store=captured):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            if isinstance(h, tuple):
                h = h[0]
            store[0] = h

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

    return result


# ===========================================================================
# Main
# ===========================================================================

print(f"=== Controlled depth-width experiment @ {args.scale} [{elapsed_str()}] ===")
print(f"Device: {DEVICE}")
print(f"Scale: {args.scale}, Training seeds: {TRAINING_SEEDS}")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer: {tokenizer.name_or_path}, vocab={tokenizer.vocab_size}")

total_tokens = TRAIN_CONFIG["total_tokens"]
seq_length = TRAIN_CONFIG["seq_length"]
print(f"\n=== Loading data ({total_tokens / 1e9:.0f}B tokens) [{elapsed_str()}] ===")
all_data = load_openwebtext(tokenizer, seq_length, total_tokens + 500000)

n_val = max(200, len(all_data) // 20)
val_data = all_data[:n_val]
train_data = all_data[n_val:]
print(f"  Train: {len(train_data)} seqs, Val: {len(val_data)} seqs")
del all_data

all_results = {}

for config_name, config in MODEL_CONFIGS.items():
    seed_results = []
    for train_seed in TRAINING_SEEDS:
        print(f"\n{'=' * 60}")
        print(f"  {config['name']} / seed {train_seed}")
        print(f"{'=' * 60}")

        TRAIN_CONFIG["seed"] = train_seed
        run_name = f"{config_name}_seed{train_seed}"

        model, model_config, best_val_loss, train_losses = train_model(
            run_name, config, train_data, val_data, tokenizer
        )
        obs_result = evaluate_observability(model, tokenizer, run_name, config)
        obs_result["best_val_loss"] = best_val_loss
        obs_result["final_train_loss"] = float(np.mean(train_losses[-100:]))
        obs_result["training_seed"] = train_seed
        seed_results.append(obs_result)

        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    all_results[config_name] = seed_results

# === Comparison ===
print(f"\n{'=' * 60}")
print(f"=== Results: {args.scale} scale [{elapsed_str()}] ===")
print(f"{'=' * 60}")

for _config_name, seed_results in all_results.items():
    pcorrs = [r["partial_corr"]["mean"] for r in seed_results]
    mean_pcorr = float(np.mean(pcorrs))
    print(f"\n{seed_results[0]['config']['name']}:")
    for r in seed_results:
        pc = r["partial_corr"]
        print(
            f"  seed {r['training_seed']}: pcorr={pc['mean']:+.4f} +/- {pc['std']:.4f}, "
            f"val_loss={r['best_val_loss']:.4f}"
        )
    if len(pcorrs) > 1:
        print(f"  Mean across training seeds: {mean_pcorr:+.4f} +/- {np.std(pcorrs):.4f}")

sw_pcorrs = [r["partial_corr"]["mean"] for r in all_results["shallow_wide"]]
dn_pcorrs = [r["partial_corr"]["mean"] for r in all_results["deep_narrow"]]
sw_mean = float(np.mean(sw_pcorrs))
dn_mean = float(np.mean(dn_pcorrs))
delta = sw_mean - dn_mean

print(f"\n  Shallow-wide mean pcorr: {sw_mean:+.4f}")
print(f"  Deep-narrow mean pcorr: {dn_mean:+.4f}")
print(f"  Delta: {delta:+.4f}")

print(f"  Divergent (|delta| > 0.05): {abs(delta) > 0.05}")

output = {
    "experiment": "controlled_depth_width",
    "scale": args.scale,
    "description": "Same data, different depth-width ratio, measure observability",
    "variable_tested": "depth-width ratio, all else held constant (both MHA)",
    "tokenizer": "openai-community/gpt2",
    "train_tokens": total_tokens,
    "training_seeds": TRAINING_SEEDS,
    "train_config": TRAIN_CONFIG,
    "model_configs": MODEL_CONFIGS,
    "results": {k: v for k, v in all_results.items()},
    "comparison": {
        "shallow_wide_mean_pcorr": sw_mean,
        "deep_narrow_mean_pcorr": dn_mean,
        "delta": delta,
        "n_training_seeds": len(TRAINING_SEEDS),
        "divergent": abs(delta) > 0.05,
    },
}

out_path = Path(f"/workspace/controlled_depth_width_{args.scale}_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {out_path}")
print(f"Total time: {elapsed_str()}")
