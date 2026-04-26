"""Pythia checkpoint dynamics: track observability across training.

Probes Pythia models at intermediate training checkpoints to determine
when observability emerges (or fails to emerge) during training. If healthy
models develop the signal early and collapsed models never develop it,
the collapse is architectural from the start rather than a late-training
artifact.

At each checkpoint: full layer sweep, 7-seed eval at peak, output-controlled
residual (r_OC), and mean loss (perplexity proxy).

Usage:
    python scripts/pythia_checkpoint_dynamics.py --model pythia-410m
    python scripts/pythia_checkpoint_dynamics.py --model pythia-1.4b
    python scripts/pythia_checkpoint_dynamics.py --model pythia-1b
    python scripts/pythia_checkpoint_dynamics.py --model all
"""

import argparse
import datetime as _dt
import gc
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr

# ── Config ───────────────────────────────────────────────────────────

MODELS = {
    "pythia-160m": {
        "id": "EleutherAI/pythia-160m",
        "n_layers": 12,
        "hidden": 768,
        "heads": 12,
        "status": "healthy",
    },
    "pythia-410m": {
        "id": "EleutherAI/pythia-410m",
        "n_layers": 24,
        "hidden": 1024,
        "heads": 16,
        "status": "collapsed",
    },
    "pythia-1b": {
        "id": "EleutherAI/pythia-1b",
        "n_layers": 16,
        "hidden": 2048,
        "heads": 8,
        "status": "healthy",
    },
    "pythia-1.4b": {
        "id": "EleutherAI/pythia-1.4b",
        "n_layers": 24,
        "hidden": 2048,
        "heads": 16,
        "status": "collapsed",
    },
}

CHECKPOINTS = [
    ("step256", 256),
    ("step1000", 1000),
    ("step2000", 2000),
    ("step4000", 4000),
    ("step8000", 8000),
    ("step16000", 16000),
    ("step32000", 32000),
    ("step64000", 64000),
    ("step128000", 128000),
    ("step143000", 143000),
]

# Pythia: batch_size=1024, seq_len=2048 → 2097152 tokens/step
TOKENS_PER_STEP = 2097152

EX_DIM = 350
LAYER_SELECT_SEED = 42
EVAL_SEEDS = list(range(43, 50))  # 7 seeds
OC_SEEDS = list(range(43, 46))  # 3 seeds for r_OC (matches run_model.py)

_OUT_DIR = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parent.parent / "results"
)

# ── CLI ──────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Pythia checkpoint dynamics experiment.")
parser.add_argument(
    "--model",
    required=True,
    choices=list(MODELS.keys()) + ["all"],
    help="Which Pythia model to sweep (or 'all')",
)
parser.add_argument("--batch-size", type=int, default=48, help="Batch size (default: 48)")
parser.add_argument("--layers-per-pass", type=int, default=4, help="Layers per forward pass (default: 4)")
parser.add_argument(
    "--checkpoints",
    nargs="+",
    default=None,
    help="Subset of checkpoints to run (e.g. step1000 step32000). Default: all.",
)
args = parser.parse_args()

# ── Setup ────────────────────────────────────────────────────────────

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = args.batch_size
LAYERS_PER_PASS = args.layers_per_pass

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


# ── Core probe functions (bit-identical to run_model.py) ─────────────


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


def train_linear_binary(train_data, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"].to(TRAIN_DEVICE, dtype=torch.float32)
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
    out = head.cpu()
    del acts, targets
    if TRAIN_DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def evaluate_head(head, test_data):
    head.eval()
    with torch.inference_mode():
        acts_fp32 = test_data["activations"].to(dtype=torch.float32)
        scores = head(acts_fp32).squeeze(-1).numpy()
    rho, p = partial_spearman(
        scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
    )
    return scores, rho, p


# ── Data loading ─────────────────────────────────────────────────────


def load_wikitext(split, max_docs=None):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=bool(max_docs))
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


# ── Activation collection ────────────────────────────────────────────


def _get_layer_list(model):
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("expected GPT-NeoX architecture for Pythia")


def collect_multi_layer(model, batches, layers, max_tokens):
    model.eval()
    layer_modules = _get_layer_list(model)
    captured = {}
    handles = []
    for layer in layers:

        def make_hook(l):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                if isinstance(h, tuple):
                    h = h[0]
                captured[l] = h

            return hook_fn

        handles.append(layer_modules[layer].register_forward_hook(make_hook(layer)))

    per_layer_acts = {l: [] for l in layers}
    per_layer_norms = {l: [] for l in layers}
    all_losses, all_sm = [], []
    total = 0

    for _bi, (input_ids_cpu, attn_mask_cpu) in enumerate(batches):
        if total >= max_tokens:
            break
        input_ids = input_ids_cpu.to(DEVICE)
        attn_mask = attn_mask_cpu.to(DEVICE)
        B, S = input_ids.shape

        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)

        shift_mask = attn_mask[:, 1:].bool()
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        V = shift_logits.size(-1)

        losses_2d = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none").view(
            B, S - 1
        )

        sm_2d = torch.empty(B, S - 1, device=DEVICE)
        for ci in range(0, B, 8):
            p = shift_logits[ci : ci + 8].float().softmax(dim=-1)
            sm_2d[ci : ci + 8] = p.max(dim=-1).values
            del p

        all_losses.append(losses_2d[shift_mask].float().cpu())
        all_sm.append(sm_2d[shift_mask].float().cpu())

        for l in layers:
            h = captured[l][:, :-1, :]
            per_layer_norms[l].append(h.float().norm(dim=-1)[shift_mask].cpu())
            per_layer_acts[l].append(h[shift_mask].to(torch.bfloat16).cpu())

        total += shift_mask.sum().item()
        for l in layers:
            captured.pop(l, None)
        del outputs, input_ids, attn_mask, shift_logits, shift_labels, losses_2d, sm_2d, shift_mask
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    n = min(total, max_tokens)
    losses_cat = torch.cat(all_losses).numpy()[:n]
    sm_cat = torch.cat(all_sm).numpy()[:n]

    results = {}
    for l in layers:
        results[l] = {
            "activations": torch.cat(per_layer_acts[l])[:n],
            "losses": losses_cat,
            "max_softmax": sm_cat,
            "activation_norm": torch.cat(per_layer_norms[l])[:n].numpy(),
        }
    return results


def collect_single_layer(model, batches, layer, max_tokens):
    return collect_multi_layer(model, batches, [layer], max_tokens)[layer]


# ── Sweep + eval ─────────────────────────────────────────────────────


def sweep_all_layers(model, train_batches, val_batches, n_layers, max_tokens):
    layer_profile = {}
    for chunk_start in range(0, n_layers, LAYERS_PER_PASS):
        chunk = list(range(chunk_start, min(chunk_start + LAYERS_PER_PASS, n_layers)))
        tr = collect_multi_layer(model, train_batches, chunk, max_tokens)
        va = collect_multi_layer(model, val_batches, chunk, max_tokens)
        for layer in chunk:
            head = train_linear_binary(tr[layer], seed=LAYER_SELECT_SEED)
            _, rho, _ = evaluate_head(head, va[layer])
            layer_profile[layer] = float(rho)
        del tr, va
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return layer_profile


def eval_at_peak(model, train_batches, val_batches, peak_layer, output_layer, max_tokens):
    """Multi-seed eval at peak + r_OC via output layer. Returns eval dict."""
    # Collect peak + output in one pass
    layers = sorted(set([peak_layer, output_layer]))
    tr = collect_multi_layer(model, train_batches, layers, max_tokens)
    va = collect_multi_layer(model, val_batches, layers, max_tokens)

    tr_peak, va_peak = tr[peak_layer], va[peak_layer]
    tr_out, va_out = tr[output_layer], va[output_layer]

    # Mean loss (perplexity proxy)
    mean_loss = float(np.mean(va_peak["losses"]))
    perplexity = float(np.exp(min(mean_loss, 20.0)))  # cap to avoid overflow

    # Multi-seed ρ_partial
    seed_rhos, seed_scores = [], []
    for seed in EVAL_SEEDS:
        head = train_linear_binary(tr_peak, seed=seed)
        scores, rho, _ = evaluate_head(head, va_peak)
        seed_rhos.append(float(rho))
        seed_scores.append(scores)

    pw = [
        float(spearmanr(seed_scores[i], seed_scores[j])[0])
        for i in range(len(EVAL_SEEDS))
        for j in range(i + 1, len(EVAL_SEEDS))
    ]

    # r_OC: output-controlled residual (3 seeds, matches run_model.py)
    oc_train_acts = tr_out["activations"].to(TRAIN_DEVICE, dtype=torch.float32)
    oc_train_losses = torch.from_numpy(tr_out["losses"]).float().to(TRAIN_DEVICE)
    oc_tds = torch.utils.data.TensorDataset(oc_train_acts, oc_train_losses)
    oc_tdl = torch.utils.data.DataLoader(oc_tds, batch_size=1024, shuffle=True)

    oc_rhos = []
    for seed in OC_SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        pred = torch.nn.Sequential(
            torch.nn.Linear(oc_train_acts.size(1), 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ).to(TRAIN_DEVICE)
        opt = torch.optim.Adam(pred.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(20):
            for bx, by in oc_tdl:
                loss = F.mse_loss(pred(bx).squeeze(-1), by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        pred.eval().cpu()
        with torch.inference_mode():
            va_out_fp32 = va_out["activations"].to(dtype=torch.float32)
            ps = pred(va_out_fp32).squeeze(-1).numpy()
            del va_out_fp32

        obs = train_linear_binary(tr_peak, seed=seed)
        obs.eval()
        with torch.inference_mode():
            va_peak_fp32 = va_peak["activations"].to(dtype=torch.float32)
            os_ = obs(va_peak_fp32).squeeze(-1).numpy()
            del va_peak_fp32

        r, _ = partial_spearman(
            os_,
            va_peak["losses"],
            [va_peak["max_softmax"], va_peak["activation_norm"], ps],
        )
        oc_rhos.append(float(r))

    del oc_train_acts, oc_train_losses, oc_tds, tr, va
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "partial_corr": {
            "mean": float(np.mean(seed_rhos)),
            "std": float(np.std(seed_rhos)),
            "per_seed": seed_rhos,
            "seed_agreement": float(np.mean(pw)),
        },
        "output_controlled": {
            "mean": float(np.mean(oc_rhos)),
            "per_seed": oc_rhos,
        },
        "mean_loss": mean_loss,
        "perplexity": perplexity,
    }


# ── Main ─────────────────────────────────────────────────────────────


def run_model(model_key, model_cfg):
    model_id = model_cfg["id"]
    n_layers = model_cfg["n_layers"]
    hidden_dim = model_cfg["hidden"]
    max_tokens = EX_DIM * hidden_dim
    output_layer = n_layers - 1

    slug = model_key.replace("-", "_").replace(".", "")
    output_path = _OUT_DIR / f"{slug}_dynamics_results.json"

    # Resume: load existing results if present
    existing = {}
    if output_path.exists():
        existing = json.load(open(output_path))
        done = set(existing.get("checkpoints", {}).keys())
        print(f"Resuming {model_key}: {len(done)} checkpoints already done")
    else:
        print(f"Starting fresh: {model_key}")

    # Filter checkpoints
    ckpts = CHECKPOINTS
    if args.checkpoints:
        ckpts = [(rev, step) for rev, step in CHECKPOINTS if rev in args.checkpoints]
    ckpts = [(rev, step) for rev, step in ckpts if rev not in existing.get("checkpoints", {})]

    if not ckpts:
        print(f"All checkpoints done for {model_key}, skipping.")
        return

    print(f"\n{'=' * 60}")
    print(f"Model: {model_id} ({n_layers}L, {model_cfg['heads']}H, {hidden_dim}d)")
    print(f"Status: {model_cfg['status']}")
    print(f"Checkpoints remaining: {len(ckpts)}")
    print(f"Token budget: {max_tokens} ({EX_DIM} ex/dim)")
    print("Protocol: layer sweep + 7-seed peak + r_OC (3 seeds)")
    print(f"{'=' * 60}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize WikiText once (shared across checkpoints)
    print(f"\n=== Pre-tokenizing [{elapsed_str()}] ===")
    train_docs = load_wikitext("train", max_docs=12000)
    val_docs = load_wikitext("validation")
    train_enc = pretokenize(train_docs, tokenizer)
    val_enc = pretokenize(val_docs, tokenizer)
    train_batches = build_batches(train_enc, BATCH_SIZE)
    val_batches = build_batches(val_enc, BATCH_SIZE)
    print(f"  {len(train_enc)} train, {len(val_enc)} val sequences")
    del train_docs, val_docs, train_enc, val_enc
    gc.collect()

    checkpoint_results = existing.get("checkpoints", {})

    for rev, step in ckpts:
        tokens_seen = step * TOKENS_PER_STEP
        tokens_B = tokens_seen / 1e9
        print(f"\n--- {model_key} @ {rev} (step {step}, ~{tokens_B:.1f}B tokens) [{elapsed_str()}] ---")

        from transformers import AutoModelForCausalLM

        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=rev, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(DEVICE)
        model.eval()
        _revision = getattr(model.config, "_commit_hash", None) or "unknown"
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.0f}s")

        # Phase 1: layer sweep
        print(f"  Sweeping {n_layers} layers...")
        t0 = time.time()
        layer_profile = sweep_all_layers(model, train_batches, val_batches, n_layers, max_tokens)
        sweep_time = time.time() - t0
        print(f"  Sweep done in {sweep_time:.0f}s")

        for l in sorted(layer_profile):
            print(f"    L{l:>2}: {layer_profile[l]:+.4f}")

        peak = max(layer_profile, key=layer_profile.get)
        if peak >= n_layers - 2:
            mid = {l: r for l, r in layer_profile.items() if l <= int(0.8 * n_layers)}
            if mid:
                peak = max(mid, key=mid.get)
        print(f"  Peak: L{peak} = {layer_profile[peak]:+.4f}")

        # Phase 2: multi-seed eval at peak + r_OC + perplexity
        print(f"  Multi-seed eval + r_OC at L{peak} (output L{output_layer})...")
        t0 = time.time()
        eval_result = eval_at_peak(model, train_batches, val_batches, peak, output_layer, max_tokens)
        eval_time = time.time() - t0

        pc = eval_result["partial_corr"]
        oc = eval_result["output_controlled"]
        print(
            f"  rho = {pc['mean']:+.4f} ± {pc['std']:.4f}, "
            f"r_OC = {oc['mean']:+.4f}, "
            f"agree = {pc['seed_agreement']:.3f}, "
            f"ppl = {eval_result['perplexity']:.1f} "
            f"({eval_time:.0f}s)"
        )

        # Unload model
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        checkpoint_results[rev] = {
            "step": step,
            "tokens_seen": tokens_seen,
            "revision": _revision,
            "layer_profile": {str(k): v for k, v in sorted(layer_profile.items())},
            "peak_layer": peak,
            "peak_layer_frac": round(peak / n_layers, 2),
            "partial_corr": pc,
            "output_controlled": oc,
            "mean_loss": eval_result["mean_loss"],
            "perplexity": eval_result["perplexity"],
            "timing": {
                "load_s": round(load_time, 1),
                "sweep_s": round(sweep_time, 1),
                "eval_s": round(eval_time, 1),
            },
        }

        # Save incrementally after each checkpoint
        output = {
            "model": model_id,
            "experiment": "checkpoint_dynamics",
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "heads": model_cfg["heads"],
            "architecture_class": f"{n_layers}L_{model_cfg['heads']}H",
            "expected_status": model_cfg["status"],
            "provenance": {
                "script": "scripts/pythia_checkpoint_dynamics.py",
                "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
                "device": str(DEVICE),
                "torch_version": torch.__version__,
            },
            "protocol": {
                "layer_select_seed": LAYER_SELECT_SEED,
                "eval_seeds": EVAL_SEEDS,
                "oc_seeds": OC_SEEDS,
                "target_ex_per_dim": EX_DIM,
                "batch_size": BATCH_SIZE,
                "layers_per_pass": LAYERS_PER_PASS,
                "n_checkpoints": len(CHECKPOINTS),
                "tokens_per_step": TOKENS_PER_STEP,
            },
            "checkpoints": checkpoint_results,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved ({len(checkpoint_results)} checkpoints) -> {output_path.name}")

    print(f"\n=== {model_key} complete: {len(checkpoint_results)} checkpoints [{elapsed_str()}] ===")

    # Summary table
    print(f"\n{'step':>10}  {'tokens':>8}  {'peak':>4}  {'rho':>8}  {'r_OC':>8}  {'ppl':>8}  {'agree':>6}")
    print("-" * 62)
    for rev, step in CHECKPOINTS:
        if rev in checkpoint_results:
            cr = checkpoint_results[rev]
            pc = cr["partial_corr"]
            oc = cr["output_controlled"]
            tB = cr["tokens_seen"] / 1e9
            print(
                f"{step:>10}  {tB:>7.1f}B  L{cr['peak_layer']:>2}  "
                f"{pc['mean']:>+8.4f}  {oc['mean']:>+8.4f}  "
                f"{cr['perplexity']:>8.1f}  {pc['seed_agreement']:>6.3f}"
            )


def main():
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    for model_key in models_to_run:
        run_model(model_key, MODELS[model_key])
    print(f"\nAll done. Total time: {elapsed_str()}")


if __name__ == "__main__":
    main()
