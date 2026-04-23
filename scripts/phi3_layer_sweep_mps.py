"""Phi-3 Mini Instruct layer-sweep diagnostic.

Sweeps partial correlation across layers on WikiText validation to verify
whether the inherited peak layer is the actual per-device peak. Rules out
layer-selection drift as an explanation for downstream outliers.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
SWEEP_EX_PER_DIM = 100
VAL_EX_PER_DIM = 100
RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.1f}m" if m < 60 else f"{m / 60:.2f}h"


def load_wikitext(split, max_docs=None):
    from datasets import load_dataset

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


def _get_layer_list(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def collect_layer(model, batches, layer, max_tokens, device, sm_chunk=4):
    model.eval()
    layer_modules = _get_layer_list(model)
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured[0] = h

    handle = layer_modules[layer].register_forward_hook(hook_fn)
    all_acts, all_norms, all_losses, all_sm = [], [], [], []
    total = 0

    try:
        for _bi, (input_ids_cpu, attn_mask_cpu) in enumerate(batches):
            if total >= max_tokens:
                break
            input_ids = input_ids_cpu.to(device)
            attn_mask = attn_mask_cpu.to(device)
            B, S = input_ids.shape
            with torch.inference_mode():
                outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
            shift_mask = attn_mask[:, 1:].bool()
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            V = shift_logits.size(-1)
            losses_2d = F.cross_entropy(
                shift_logits.view(-1, V), shift_labels.view(-1), reduction="none"
            ).view(B, S - 1)
            sm_2d = torch.empty(B, S - 1, device=device)
            for ci in range(0, B, sm_chunk):
                p = shift_logits[ci : ci + sm_chunk].float().softmax(dim=-1)
                sm_2d[ci : ci + sm_chunk] = p.max(dim=-1).values
                del p
            h = captured[0][:, :-1, :]
            all_acts.append(h[shift_mask].cpu())  # keep fp16
            all_norms.append(h.float().norm(dim=-1)[shift_mask].cpu())
            all_losses.append(losses_2d[shift_mask].float().cpu())
            all_sm.append(sm_2d[shift_mask].float().cpu())
            total += shift_mask.sum().item()
            captured.pop(0, None)
            del outputs, input_ids, attn_mask, shift_logits, shift_labels
            del losses_2d, sm_2d, shift_mask, h
            if device == "mps":
                torch.mps.empty_cache()
    finally:
        handle.remove()

    n = min(total, max_tokens)
    return {
        "activations": torch.cat(all_acts)[:n],
        "losses": torch.cat(all_losses).numpy()[:n],
        "max_softmax": torch.cat(all_sm).numpy()[:n],
        "activation_norm": torch.cat(all_norms)[:n].numpy(),
    }


def compute_loss_residuals(losses, max_softmax, activation_norm):
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    beta = np.linalg.lstsq(X, losses, rcond=None)[0]
    return losses - X @ beta


def train_linear_binary(train_data, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"].float().to(TRAIN_DEVICE)
    residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    targets = torch.from_numpy((residuals > 0).astype(np.float32)).to(TRAIN_DEVICE)
    head = torch.nn.Linear(acts.size(1), 1).to(TRAIN_DEVICE)
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
    return head.cpu()


def partial_spearman(x, y, controls):
    from scipy.stats import rankdata

    ranked = [rankdata(v) for v in [x, y, *controls]]
    C = np.column_stack([ranked[i] for i in range(2, len(ranked))])
    C = np.column_stack([C, np.ones(len(x))])
    beta_x = np.linalg.lstsq(C, ranked[0], rcond=None)[0]
    beta_y = np.linalg.lstsq(C, ranked[1], rcond=None)[0]
    rx = ranked[0] - C @ beta_x
    ry = ranked[1] - C @ beta_y
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate(head, test_data):
    head.eval()
    with torch.inference_mode():
        scores = head(test_data["activations"].float()).squeeze(-1).numpy()
    return partial_spearman(
        scores,
        test_data["losses"],
        [test_data["max_softmax"], test_data["activation_norm"]],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=2, help="Sweep every Nth layer (default 2)")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated explicit layer list")
    parser.add_argument("--ex-per-dim", type=int, default=SWEEP_EX_PER_DIM)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"=== Phi-3 Mini layer sweep on {DEVICE} [{elapsed_str()}] ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if DEVICE == "mps" else torch.bfloat16
    attn = "eager" if DEVICE == "mps" else "sdpa"
    print(f"Loading dtype={dtype} attn={attn}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, attn_implementation=attn).to(DEVICE)
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"n_layers={n_layers}  hidden={hidden_dim}", flush=True)

    if args.layers:
        sweep = [int(x) for x in args.layers.split(",")]
    else:
        sweep = list(range(0, n_layers, args.step))
        if n_layers - 1 not in sweep:
            sweep.append(n_layers - 1)
    print(f"Sweeping layers: {sweep}  ex/dim={args.ex_per_dim}", flush=True)

    # --- Pre-tokenize ---
    print(f"\n=== Pre-tokenizing WikiText [{elapsed_str()}] ===", flush=True)
    train_docs = load_wikitext("train", max_docs=4000)
    val_docs = load_wikitext("validation")
    print(f"  train={len(train_docs)} docs  val={len(val_docs)} docs", flush=True)
    train_enc = pretokenize(train_docs, tokenizer)
    val_enc = pretokenize(val_docs, tokenizer)
    del train_docs, val_docs
    bs = 16 if DEVICE == "mps" else 48
    train_batches = build_batches(train_enc, bs)
    val_batches = build_batches(val_enc, bs)
    del train_enc, val_enc
    print(f"  batches: train={len(train_batches)}  val={len(val_batches)}", flush=True)

    # --- Sweep ---
    train_target = args.ex_per_dim * hidden_dim
    val_target = VAL_EX_PER_DIM * hidden_dim
    sweep_results = {}

    for layer in sweep:
        print(f"\n--- L{layer} [{elapsed_str()}] ---", flush=True)
        tr = collect_layer(model, train_batches, layer, train_target, DEVICE)
        print(f"  train collected: {tr['activations'].shape}", flush=True)
        head = train_linear_binary(tr, seed=42)
        del tr
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()
        va = collect_layer(model, val_batches, layer, val_target, DEVICE)
        print(f"  val collected: {va['activations'].shape}", flush=True)
        rho = evaluate(head, va)
        sweep_results[layer] = float(rho)
        print(f"  L{layer}: pcorr(val seed=42) = {rho:.4f}", flush=True)
        del va, head
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    peak_layer = max(sweep_results, key=sweep_results.get)
    print(f"\n=== Sweep summary [{elapsed_str()}] ===", flush=True)
    for layer in sorted(sweep_results):
        mark = "  <-- peak" if layer == peak_layer else ""
        mark += "  [inherited H100 peak]" if layer == 17 else ""
        print(f"  L{layer:>2}: {sweep_results[layer]:+.4f}{mark}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "model": MODEL_ID,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "device": DEVICE,
        "dtype": str(dtype),
        "attn_implementation": attn,
        "protocol": {
            "sweep_layers": sweep,
            "sweep_ex_per_dim": args.ex_per_dim,
            "val_ex_per_dim": VAL_EX_PER_DIM,
            "seed": 42,
            "batch_size": bs,
        },
        "inherited_peak_from_h100": 17,
        "inherited_peak_pcorr_from_h100_json": 0.3000711575550095,
        "mps_sweep_results": sweep_results,
        "mps_native_peak": int(peak_layer),
        "mps_native_peak_pcorr": sweep_results[peak_layer],
        "elapsed_total": elapsed_str(),
    }
    out_path = RESULTS_DIR / "phi3_mini_layer_sweep_mps.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}", flush=True)
    print(f"MPS-native peak: L{peak_layer} ({sweep_results[peak_layer]:.4f})", flush=True)
    print("H100 peak (inherited): L17 (0.3001)", flush=True)
    if peak_layer != 17:
        print(
            f"MISMATCH: MPS peak L{peak_layer} differs from H100 L17 by {abs(peak_layer - 17)} layers.",
            flush=True,
        )


if __name__ == "__main__":
    main()
