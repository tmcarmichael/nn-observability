"""r_OC width sweep: does a wider output predictor absorb the mid-layer signal?

Trains output-layer MLP predictors of the target at several widths, then
measures the mid-layer observer's residual partial correlation after
controlling for each predictor. Tests whether the confidence-independent
signal is recoverable from the final layer given enough capacity.
"""

import shutil
import subprocess

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
elif shutil.which("rocm-smi"):
    subprocess.run(["rocm-smi"], check=False)
else:
    print("No GPU management tool found (nvidia-smi / rocm-smi)")

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, rankdata
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 48
SM_CHUNK = 8
print(f"Device: {DEVICE}")

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


# ---------------------------------------------------------------------------
# Data loading and pre-tokenization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Activation collection (single layer)
# ---------------------------------------------------------------------------


def collect_single_layer(model, batches, layer, max_tokens, device):
    model.eval()
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured[0] = h

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    all_acts, all_norms, all_losses, all_sm, all_ent = [], [], [], [], []
    total = 0

    for bi, (input_ids_cpu, attn_mask_cpu) in enumerate(batches):
        if total >= max_tokens:
            break
        input_ids = input_ids_cpu.to(device)
        attn_mask = attn_mask_cpu.to(device)
        B, S = input_ids.shape

        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attn_mask)

        shift_mask = attn_mask[:, 1:].bool()
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        V = shift_logits.size(-1)

        losses_2d = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none").view(
            B, S - 1
        )

        sm_2d = torch.empty(B, S - 1, device=device)
        ent_2d = torch.empty(B, S - 1, device=device)
        for ci in range(0, B, SM_CHUNK):
            p = shift_logits[ci : ci + SM_CHUNK].float().softmax(dim=-1)
            sm_2d[ci : ci + SM_CHUNK] = p.max(dim=-1).values
            ent_2d[ci : ci + SM_CHUNK] = -(p * (p + 1e-10).log()).sum(dim=-1)
            del p

        h = captured[0][:, :-1, :].float()
        all_acts.append(h[shift_mask].cpu())
        all_norms.append(h.norm(dim=-1)[shift_mask].cpu())
        all_losses.append(losses_2d[shift_mask].float().cpu())
        all_sm.append(sm_2d[shift_mask].float().cpu())
        all_ent.append(ent_2d[shift_mask].float().cpu())

        total += shift_mask.sum().item()
        captured.pop(0, None)
        del outputs, input_ids, attn_mask, shift_logits, shift_labels
        del losses_2d, sm_2d, ent_2d, shift_mask, h
        if device == "cuda":
            torch.cuda.empty_cache()

        if (bi + 1) % 10 == 0:
            print(f"      batch {bi + 1}/{len(batches)}, {total} positions")

    handle.remove()
    n = min(total, max_tokens)
    return {
        "activations": torch.cat(all_acts)[:n],
        "losses": torch.cat(all_losses).numpy()[:n],
        "max_softmax": torch.cat(all_sm).numpy()[:n],
        "logit_entropy": torch.cat(all_ent).numpy()[:n],
        "activation_norm": torch.cat(all_norms)[:n].numpy(),
    }


# ---------------------------------------------------------------------------
# Probe and OC predictor training
# ---------------------------------------------------------------------------


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


def train_oc_predictor(train_acts, train_losses, width, seed=42, epochs=20, lr=1e-3):
    """Train an MLP predictor from output-layer activations to loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_acts.to(TRAIN_DEVICE)
    targets = torch.from_numpy(train_losses).float().to(TRAIN_DEVICE)
    pred = torch.nn.Sequential(
        torch.nn.Linear(acts.size(1), width),
        torch.nn.ReLU(),
        torch.nn.Linear(width, 1),
    ).to(TRAIN_DEVICE)
    opt = torch.optim.Adam(pred.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    pred.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.mse_loss(pred(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return pred.cpu()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B"
PEAK_LAYER = 17
WIDTHS = [64, 128, 256, 512]
EVAL_SEEDS = [43, 44, 45]
TARGET_EX_PER_DIM = 350

print(f"=== r_OC width sweep: {MODEL_ID}, L{PEAK_LAYER}, widths={WIDTHS} ===")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
).to(DEVICE)
model.eval()

N_LAYERS = model.config.num_hidden_layers
HIDDEN_DIM = model.config.hidden_size
n_params = sum(p.numel() for p in model.parameters()) / 1e9
MAX_TRAIN = TARGET_EX_PER_DIM * HIDDEN_DIM
OUTPUT_LAYER = N_LAYERS - 1
print(f"{n_params:.1f}B, {N_LAYERS} layers, {HIDDEN_DIM} dim, {MAX_TRAIN} train tokens")

# --- Pre-tokenize ---
print(f"\n=== Pre-tokenizing [{elapsed_str()}] ===")
wiki_train_docs = load_wikitext("train", max_docs=12000)
wiki_val_docs = load_wikitext("validation", max_docs=None)
wiki_train_enc = pretokenize(wiki_train_docs, tokenizer)
wiki_val_enc = pretokenize(wiki_val_docs, tokenizer)
train_batches = build_batches(wiki_train_enc, BATCH_SIZE)
val_batches = build_batches(wiki_val_enc, BATCH_SIZE)
del wiki_train_docs, wiki_val_docs, wiki_train_enc, wiki_val_enc

# --- Collect peak layer and output layer ---
print(f"\n=== Collecting L{PEAK_LAYER} (peak) [{elapsed_str()}] ===")
train_peak = collect_single_layer(model, train_batches, PEAK_LAYER, MAX_TRAIN, DEVICE)
val_peak = collect_single_layer(model, val_batches, PEAK_LAYER, MAX_TRAIN, DEVICE)

print(f"\n=== Collecting L{OUTPUT_LAYER} (output) [{elapsed_str()}] ===")
train_output = collect_single_layer(model, train_batches, OUTPUT_LAYER, MAX_TRAIN, DEVICE)
val_output = collect_single_layer(model, val_batches, OUTPUT_LAYER, MAX_TRAIN, DEVICE)

del model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print(f"Model unloaded. [{elapsed_str()}]")

# --- Baseline: standard pcorr (no OC control) ---
print(f"\n=== Baseline pcorr [{elapsed_str()}] ===")
baseline_rhos = []
for seed in EVAL_SEEDS:
    head = train_linear_binary(train_peak, seed=seed)
    head.eval()
    with torch.inference_mode():
        scores = head(val_peak["activations"]).squeeze(-1).numpy()
    rho, _ = partial_spearman(
        scores, val_peak["losses"], [val_peak["max_softmax"], val_peak["activation_norm"]]
    )
    baseline_rhos.append(float(rho))
print(f"  Baseline pcorr (no OC): {np.mean(baseline_rhos):+.4f} +/- {np.std(baseline_rhos):.4f}")

# --- Width sweep ---
print(f"\n=== Width sweep [{elapsed_str()}] ===")
results = {"baseline_pcorr": {"mean": float(np.mean(baseline_rhos)), "per_seed": baseline_rhos}}

for width in WIDTHS:
    print(f"\n  Width {width}:")
    oc_rhos = []
    for seed in EVAL_SEEDS:
        # Train OC predictor at this width
        pred = train_oc_predictor(train_output["activations"], train_output["losses"], width=width, seed=seed)
        pred.eval()
        with torch.inference_mode():
            pred_scores = pred(val_output["activations"]).squeeze(-1).numpy()

        # Train observer and measure residual pcorr
        obs = train_linear_binary(train_peak, seed=seed)
        obs.eval()
        with torch.inference_mode():
            obs_scores = obs(val_peak["activations"]).squeeze(-1).numpy()

        rho, _ = partial_spearman(
            obs_scores,
            val_peak["losses"],
            [val_peak["max_softmax"], val_peak["activation_norm"], pred_scores],
        )
        oc_rhos.append(float(rho))
        print(f"    seed {seed}: r_OC = {rho:+.4f}")

    mean_oc = float(np.mean(oc_rhos))
    results[f"width_{width}"] = {
        "mean": mean_oc,
        "per_seed": oc_rhos,
        "delta_from_baseline": float(np.mean(baseline_rhos)) - mean_oc,
    }
    print(
        f"  Width {width} mean r_OC: {mean_oc:+.4f}  (absorbed: {float(np.mean(baseline_rhos)) - mean_oc:+.4f})"
    )

# --- Summary ---
print(f"\n=== Summary [{elapsed_str()}] ===")
print(f"  Baseline (no OC):  {results['baseline_pcorr']['mean']:+.4f}")
for width in WIDTHS:
    r = results[f"width_{width}"]
    print(f"  Width {width:>3}:         {r['mean']:+.4f}  (absorbed {r['delta_from_baseline']:+.4f})")

survives = results[f"width_{WIDTHS[-1]}"]["mean"] > 0.05
results["conclusion"] = "signal_survives" if survives else "signal_absorbed"
print(
    f"\n  Conclusion: {'Signal survives at max width' if survives else 'Signal absorbed by wider predictor'}"
)

# --- Save ---
output = {
    "model": MODEL_ID,
    "n_params_b": round(n_params, 2),
    "peak_layer": PEAK_LAYER,
    "output_layer": OUTPUT_LAYER,
    "hidden_dim": HIDDEN_DIM,
    "target_ex_per_dim": TARGET_EX_PER_DIM,
    "eval_seeds": EVAL_SEEDS,
    "widths": WIDTHS,
    "results": results,
}
_out_dir = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parent.parent / "results"
)
out_path = _out_dir / "roc_width_sweep_results.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved {out_path}")
print(f"Total time: {elapsed_str()}")
print(json.dumps(output, indent=2))
