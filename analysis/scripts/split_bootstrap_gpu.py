"""Split-level bootstrap: stability under document resampling (GPU-optimized).

Resamples WikiText train documents 30 times, retrains probe on each,
evaluates on fixed test set. Reports variance of partial correlation.
If tight, the signal is robust to document selection.

Optimized for H100: pre-tokenize once, batch_size=48, sorted length
packing, forward hooks, probe training on GPU.

Upload to RunPod /workspace/ and run:
  pip install transformers datasets scipy
  python split_bootstrap_gpu.py --model Qwen/Qwen2.5-7B --peak-layer 17

Usage: python split_bootstrap_gpu.py --model MODEL [--peak-layer N] [--n-boot 30]
"""

import argparse
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
elif shutil.which("rocm-smi"):
    subprocess.run(["rocm-smi"], check=False)
else:
    print("No GPU management tool found (nvidia-smi / rocm-smi)")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, rankdata
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 48
print(f"Device: {DEVICE}")

RUN_START = time.time()


def elapsed():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


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
    beta, _, _, _ = np.linalg.lstsq(X, losses, rcond=None)
    return losses - X @ beta


def load_wikitext(split="test", max_docs=None):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    docs, current = [], []
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


def pretokenize(docs, tokenizer, max_length=512):
    encoded = []
    for doc in docs:
        if not doc.strip():
            continue
        ids = tokenizer.encode(doc, truncation=True, max_length=max_length)
        if len(ids) >= 2:
            encoded.append(ids)
    encoded.sort(key=len)
    return encoded


def build_batches(encoded, batch_size):
    batches = []
    for i in range(0, len(encoded), batch_size):
        chunk = encoded[i : i + batch_size]
        max_len = len(chunk[-1])
        B = len(chunk)
        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attn_mask = torch.zeros(B, max_len, dtype=torch.long)
        for j, ids in enumerate(chunk):
            input_ids[j, : len(ids)] = torch.tensor(ids)
            attn_mask[j, : len(ids)] = 1
        batches.append((input_ids, attn_mask))
    return batches


def collect_from_batches(model, batches, layer, max_tokens, device):
    """Collect activations from pre-built batches using forward hook."""
    model.eval()
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured["h"] = h

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    all_acts, all_losses, all_sm, all_norms = [], [], [], []
    total = 0

    for input_ids_cpu, attn_mask_cpu in batches:
        if total >= max_tokens:
            break
        input_ids = input_ids_cpu.to(device)
        attn_mask = attn_mask_cpu.to(device)

        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attn_mask)

        h_all = captured["h"]
        shift_mask = attn_mask[:, 1:].bool()
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        V = shift_logits.size(-1)

        losses_2d = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none").view(
            input_ids.size(0), -1
        )

        # Chunked softmax to avoid materializing full (B, S, V) float32
        B_cur = input_ids.size(0)
        S_cur = shift_logits.size(1)
        sm_2d = torch.empty(B_cur, S_cur, device=device)
        SM_CHUNK = 8
        for ci in range(0, B_cur, SM_CHUNK):
            p = shift_logits[ci : ci + SM_CHUNK].float().softmax(dim=-1)
            sm_2d[ci : ci + SM_CHUNK] = p.max(dim=-1).values
            del p

        h_shift = h_all[:, :-1, :].float()

        all_acts.append(h_shift[shift_mask].cpu())
        all_losses.append(losses_2d[shift_mask].float().cpu())
        all_sm.append(sm_2d[shift_mask].float().cpu())
        all_norms.append(h_shift.norm(dim=-1)[shift_mask].cpu())
        total += shift_mask.sum().item()

        captured.pop("h", None)
        del outputs, input_ids, attn_mask, shift_logits, shift_labels, losses_2d, sm_2d, h_shift
        if device == "cuda":
            torch.cuda.empty_cache()

    handle.remove()

    n = min(total, max_tokens)
    return {
        "activations": torch.cat(all_acts)[:n],
        "losses": torch.cat(all_losses).numpy()[:n],
        "max_softmax": torch.cat(all_sm).numpy()[:n],
        "activation_norm": torch.cat(all_norms).numpy()[:n],
    }


def train_and_eval(train_data, test_data, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"].to(DEVICE)
    residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    targets = torch.from_numpy((residuals > 0).astype(np.float32)).to(DEVICE)
    head = torch.nn.Linear(acts.size(1), 1).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    head.cpu().eval()
    with torch.inference_mode():
        scores = head(test_data["activations"]).squeeze(-1).numpy()
    rho, _ = partial_spearman(
        scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
    )
    return rho


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--peak-layer", type=int, default=None)
    parser.add_argument("--ex-dim", type=int, default=350)
    parser.add_argument("--n-boot", type=int, default=30)
    parser.add_argument("--max-docs", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    max_tokens = args.ex_dim * hidden_dim
    peak = args.peak_layer if args.peak_layer is not None else n_layers * 2 // 3
    print(f"{hidden_dim} dim, {n_layers} layers, L{peak}")
    print(f"Token budget: {max_tokens} ({args.ex_dim} ex/dim)")

    # Load and pre-tokenize all docs once
    print("Loading WikiText...")
    all_train_docs = load_wikitext("train", max_docs=args.max_docs)
    test_docs = load_wikitext("test", max_docs=None)
    print(f"  {len(all_train_docs)} train docs, {len(test_docs)} test docs")

    print("Pre-tokenizing...")
    all_train_encoded = pretokenize(all_train_docs, tokenizer)
    test_encoded = pretokenize(test_docs, tokenizer)
    test_batches = build_batches(test_encoded, BATCH_SIZE)
    print(f"  {len(all_train_encoded)} train seqs, {len(test_encoded)} test seqs")

    # Collect test data once (fixed)
    print(f"Collecting test activations at L{peak}...")
    test_data = collect_from_batches(model, test_batches, peak, max_tokens, DEVICE)
    print(f"  {len(test_data['losses'])} test positions [{elapsed()}]")

    # Bootstrap: resample train documents, fix probe seed.
    # Fixed probe seed isolates split variance from seed variance.
    # This answers: "is the signal robust to which documents are in
    # training?" not "is the signal robust to probe initialization?"
    # Seed variance is already measured by the 7-seed protocol.
    PROBE_SEED = args.seed
    rng = np.random.RandomState(args.seed)
    boot_rhos = []

    print(f"\n=== {args.n_boot} bootstrap resamples (probe seed fixed at {PROBE_SEED}) ===")
    for i in range(args.n_boot):
        t0 = time.time()

        # Resample encoded docs (not raw docs, saves re-tokenization)
        idx = rng.choice(len(all_train_encoded), size=len(all_train_encoded), replace=True)
        boot_encoded = [all_train_encoded[j] for j in idx]
        boot_encoded.sort(key=len)
        boot_batches = build_batches(boot_encoded, BATCH_SIZE)

        train_data = collect_from_batches(model, boot_batches, peak, max_tokens, DEVICE)
        rho = train_and_eval(train_data, test_data, seed=PROBE_SEED)
        boot_rhos.append(rho)

        dt = time.time() - t0
        remaining = dt * (args.n_boot - i - 1)
        print(
            f"  boot {i + 1}/{args.n_boot}: pcorr={rho:+.4f}  "
            f"({dt:.0f}s, ~{remaining / 60:.0f}m remaining) [{elapsed()}]"
        )

        del train_data, boot_batches
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Checkpoint after each iteration
        _ckpt = {
            "model": args.model,
            "peak_layer": peak,
            "n_complete": i + 1,
            "boot_rhos": boot_rhos,
            "_elapsed": elapsed(),
        }
        _ckpt_path = Path("/workspace") / f"split_bootstrap_{args.model.split('/')[-1]}_checkpoint.json"
        with open(_ckpt_path, "w") as f:
            json.dump(_ckpt, f)

    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    boot_rhos = np.array(boot_rhos)
    ci_lo, ci_hi = np.percentile(boot_rhos, [2.5, 97.5])

    print("\n=== Split-level bootstrap results ===")
    print(f"  Model: {args.model}, Layer {peak}")
    print(f"  Bootstrap samples: {args.n_boot}")
    print(f"  Mean pcorr: {boot_rhos.mean():+.4f}")
    print(f"  Std:         {boot_rhos.std():.4f}")
    print(f"  95% CI:      [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Range:       [{boot_rhos.min():+.4f}, {boot_rhos.max():+.4f}]")
    print(f"  Total time:  {elapsed()}")

    out = {
        "model": args.model,
        "peak_layer": peak,
        "ex_dim": args.ex_dim,
        "n_boot": args.n_boot,
        "boot_rhos": boot_rhos.tolist(),
        "mean": float(boot_rhos.mean()),
        "std": float(boot_rhos.std()),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "design": {
            "probe_seed": PROBE_SEED,
            "note": "Probe seed fixed across all bootstrap iterations to isolate "
            "split variance from seed variance. Seed variance is measured "
            "separately by the 7-seed protocol in the main experiments.",
        },
    }
    out_path = Path("/workspace") / f"split_bootstrap_{args.model.split('/')[-1]}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
