"""Dump per-token observer, confidence, norm, and target arrays.

Persists the per-token arrays needed for offline held-out-split evaluation
of the partial Spearman metric. Reuses the canonical peak-layer probe
protocol so the dumped tokens match the in-process v3 measurements.
"""

import argparse
import datetime as _dt
import gc
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENS_DIR = REPO_ROOT / "results" / "tokens"
TOKENS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


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


def compute_loss_residuals(losses, max_softmax, activation_norm):
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    beta = np.linalg.lstsq(X, losses, rcond=None)[0]
    return losses - X @ beta


def _get_layer_list(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def collect_single_layer(model, batches, layer, max_tokens, device, sm_chunk=8):
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
            h = captured[0][:, :-1, :].float()
            all_acts.append(h[shift_mask].cpu())
            all_norms.append(h.norm(dim=-1)[shift_mask].cpu())
            all_losses.append(losses_2d[shift_mask].float().cpu())
            all_sm.append(sm_2d[shift_mask].float().cpu())
            total += shift_mask.sum().item()
            captured.pop(0, None)
            del outputs, input_ids, attn_mask, shift_logits, shift_labels, losses_2d, sm_2d, shift_mask, h
            if device == "cuda":
                torch.cuda.empty_cache()
    finally:
        handle.remove()

    n = min(total, max_tokens)
    return {
        "activations": torch.cat(all_acts)[:n],
        "losses": torch.cat(all_losses).numpy()[:n],
        "max_softmax": torch.cat(all_sm).numpy()[:n],
        "activation_norm": torch.cat(all_norms)[:n].numpy(),
    }


def train_linear_binary(train_data, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"].to(TRAIN_DEVICE)
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


def score_probe(head, eval_data):
    head.eval()
    with torch.inference_mode():
        scores = head(eval_data["activations"]).squeeze(-1).numpy()
    return scores


def main():
    parser = argparse.ArgumentParser(description="Dump per-token arrays for held-out fit-split analysis.")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--peak-layer", type=int, required=True, help="Probe layer (0-indexed)")
    parser.add_argument("--ex-dim", type=int, default=350, help="Train budget per hidden dim (default 350)")
    parser.add_argument(
        "--val-ex-dim", type=int, default=350, help="Eval budget per hidden dim (default 350)"
    )
    parser.add_argument("--seeds", default="42,43,44", help="Probe seeds (comma-separated)")
    parser.add_argument("--max-train-docs", type=int, default=12000)
    parser.add_argument("--max-val-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True (default native HF impls)",
    )
    parser.add_argument(
        "--output", default=None, help="Output .npz path (default: results/tokens/<slug>_tokens.npz)"
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", args.model.split("/")[-1]).strip("_").lower()

    print(f"=== dump_tokens [{elapsed_str()}] ===")
    print(f"Model: {args.model} (slug={slug}), peak: L{args.peak_layer}, dtype: {args.dtype}, seeds: {seeds}")
    print(f"Device: {DEVICE}, train_device: {TRAIN_DEVICE}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=dtype,
        attn_implementation=args.attn_impl,
    ).to(DEVICE)
    model.eval()
    _model_revision = getattr(model.config, "_commit_hash", None) or "unknown"
    cfg = getattr(model.config, "text_config", model.config)
    hidden_dim = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    if args.peak_layer < 0 or args.peak_layer >= n_layers:
        print(f"ERROR: peak_layer {args.peak_layer} out of range [0, {n_layers - 1}]")
        sys.exit(1)
    max_train = args.ex_dim * hidden_dim
    max_val = args.val_ex_dim * hidden_dim
    print(f"Hidden dim: {hidden_dim}, layers: {n_layers}, train tokens: {max_train}, val tokens: {max_val}")

    print(f"\n=== Pre-tokenizing [{elapsed_str()}] ===")
    train_docs = load_wikitext("train", max_docs=args.max_train_docs)
    val_docs = load_wikitext("validation", max_docs=args.max_val_docs)
    print(f"  train docs: {len(train_docs)}, val docs: {len(val_docs)}")
    train_enc = pretokenize(train_docs, tokenizer)
    val_enc = pretokenize(val_docs, tokenizer)
    print(f"  train enc: {len(train_enc)}, val enc: {len(val_enc)}")

    train_batches = build_batches(train_enc, args.batch_size)
    val_batches = build_batches(val_enc, args.batch_size)
    del train_docs, val_docs, train_enc, val_enc
    gc.collect()

    print(f"\n=== Collecting train activations [{elapsed_str()}] ===")
    train_data = collect_single_layer(model, train_batches, args.peak_layer, max_train, DEVICE)
    del train_batches
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n=== Collecting val activations [{elapsed_str()}] ===")
    val_data = collect_single_layer(model, val_batches, args.peak_layer, max_val, DEVICE)
    del val_batches
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Free model GPU memory before probe training
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"\n=== Training probes ({len(seeds)} seeds) and scoring val [{elapsed_str()}] ===")
    observer_per_seed = {}
    for seed in seeds:
        head = train_linear_binary(train_data, seed=seed)
        scores = score_probe(head, val_data)
        observer_per_seed[seed] = scores
        print(f"  seed {seed} trained and scored, n_val_tokens={len(scores)}")

    out_path = Path(args.output) if args.output else TOKENS_DIR / f"{slug}_tokens.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs = {
        "max_softmax": val_data["max_softmax"].astype(np.float32),
        "activation_norm": val_data["activation_norm"].astype(np.float32),
        "target_surprise": val_data["losses"].astype(np.float32),
    }
    for seed, scores in observer_per_seed.items():
        save_kwargs[f"observer_seed{seed}"] = scores.astype(np.float32)

    np.savez_compressed(
        out_path,
        model=args.model,
        peak_layer=args.peak_layer,
        n_tokens=len(val_data["losses"]),
        ex_per_dim=args.ex_dim,
        val_ex_per_dim=args.val_ex_dim,
        seeds=np.array(seeds, dtype=np.int32),
        model_revision=_model_revision,
        timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        device=str(DEVICE),
        dtype=args.dtype,
        attn_impl=args.attn_impl,
        **save_kwargs,
    )

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {out_path} ({size_mb:.1f} MB)")
    print(f"Total time: {elapsed_str()}")


if __name__ == "__main__":
    main()
