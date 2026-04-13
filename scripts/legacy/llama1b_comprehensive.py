"""Llama 3.2 1B: full protocol rerun (confirms preliminary +0.250).

Model: meta-llama/Llama-3.2-1B | GPU: H100 SXM 80GB | 7 seeds (43-49) | 350 ex/dim
Protocol: layer selected on val with seed 42, evaluated on val with seeds 43-49
Optimizations: pre-tokenize once, vectorized GPU extraction, batch_size=48,
  sorted length packing, 3 layers per pass, TRAIN_DEVICE=cuda

Usage: pip install transformers datasets scipy && python scripts/llama1b_comprehensive.py
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
from scipy.stats import pearsonr, rankdata, spearmanr
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
    """Tokenize all docs once. Sort by length for efficient padding."""
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
    """Create padded batches on CPU from pre-tokenized docs."""
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
# Fast multi-layer collection with vectorized GPU extraction
# ---------------------------------------------------------------------------


def collect_multi_layer_fast(model, batches, layers, max_tokens, device):
    """Collect activations from multiple layers using pre-built batches.
    All extraction done on GPU with masked tensor ops, no Python per-sequence loop."""
    model.eval()
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

        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))

    per_layer_acts = {l: [] for l in layers}
    per_layer_norms = {l: [] for l in layers}
    all_losses, all_sm, all_ent = [], [], []
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

        all_losses.append(losses_2d[shift_mask].float().cpu())
        all_sm.append(sm_2d[shift_mask].float().cpu())
        all_ent.append(ent_2d[shift_mask].float().cpu())

        for l in layers:
            h = captured[l][:, :-1, :].float()
            per_layer_acts[l].append(h[shift_mask].cpu())
            per_layer_norms[l].append(h.norm(dim=-1)[shift_mask].cpu())

        total += shift_mask.sum().item()

        for l in layers:
            captured.pop(l, None)
        del outputs, input_ids, attn_mask, shift_logits, shift_labels
        del losses_2d, sm_2d, ent_2d, shift_mask
        if device == "cuda":
            torch.cuda.empty_cache()

        if (bi + 1) % 10 == 0:
            print(f"      batch {bi + 1}/{len(batches)}, {total} positions")

    for h in handles:
        h.remove()

    n = min(total, max_tokens)
    losses_cat = torch.cat(all_losses).numpy()[:n]
    sm_cat = torch.cat(all_sm).numpy()[:n]
    ent_cat = torch.cat(all_ent).numpy()[:n]

    results = {}
    for l in layers:
        results[l] = {
            "activations": torch.cat(per_layer_acts[l])[:n],
            "losses": losses_cat,
            "max_softmax": sm_cat,
            "logit_entropy": ent_cat,
            "activation_norm": torch.cat(per_layer_norms[l])[:n].numpy(),
        }

    print(f"    {n} positions ({len(layers)} layers)")
    return results


def collect_single_layer_fast(model, batches, layer, max_tokens, device):
    """Single-layer wrapper."""
    return collect_multi_layer_fast(model, batches, [layer], max_tokens, device)[layer]


# ---------------------------------------------------------------------------
# Probe training and evaluation
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


def evaluate_head(head, test_data):
    head.eval()
    with torch.inference_mode():
        scores = head(test_data["activations"]).squeeze(-1).numpy()
    rho, p = partial_spearman(
        scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
    )
    return scores, rho, p


def compute_hand_designed(data):
    acts = data["activations"]
    p = acts.abs() / (acts.abs().sum(dim=1, keepdim=True) + 1e-8)
    return {
        "ff_goodness": (acts**2).mean(dim=1).numpy(),
        "active_ratio": (acts.abs() > 0.01).float().mean(dim=1).numpy(),
        "act_entropy": -(p * (p + 1e-8).log()).sum(dim=1).numpy(),
        "activation_norm": data["activation_norm"],
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path("/workspace/llama1b_checkpoint.json")


def save_checkpoint(phase, **data):
    """Save partial results after each phase. Overwrites previous checkpoint."""
    ckpt = {"_checkpoint_phase": phase, "_elapsed": elapsed_str()}
    ckpt.update(data)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"  [checkpoint saved: phase {phase}, {CHECKPOINT_PATH}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B"
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
print(f"{n_params:.1f}B, {N_LAYERS} layers, {HIDDEN_DIM} dim")

TARGET_EX_PER_DIM = 350
MAX_TRAIN = TARGET_EX_PER_DIM * HIDDEN_DIM
LAYER_SELECT_SEED = 42
EVAL_SEEDS = list(range(43, 50))
LAYERS_PER_PASS = 3  # 1B fits multiple layers per pass comfortably
print(
    f"Token budget: {MAX_TRAIN} train ({TARGET_EX_PER_DIM} ex/dim), {LAYERS_PER_PASS} layers/pass, batch={BATCH_SIZE}"
)

# --- Pre-tokenize all datasets once ---
print(f"\n=== Pre-tokenizing [{elapsed_str()}] ===")
wiki_train_docs = load_wikitext("train", max_docs=12000)
wiki_val_docs = load_wikitext("validation", max_docs=None)
wiki_test_docs = load_wikitext("test", max_docs=None)
print(f"{len(wiki_train_docs)} train, {len(wiki_val_docs)} val, {len(wiki_test_docs)} test docs")

wiki_train_enc = pretokenize(wiki_train_docs, tokenizer)
wiki_val_enc = pretokenize(wiki_val_docs, tokenizer)
wiki_test_enc = pretokenize(wiki_test_docs, tokenizer)
print(f"Tokenized: {len(wiki_train_enc)} train, {len(wiki_val_enc)} val, {len(wiki_test_enc)} test seqs")

train_batches = build_batches(wiki_train_enc, BATCH_SIZE)
val_batches = build_batches(wiki_val_enc, BATCH_SIZE)
test_batches = build_batches(wiki_test_enc, BATCH_SIZE)
print(f"Batches: {len(train_batches)} train, {len(val_batches)} val, {len(test_batches)} test")
del wiki_train_docs, wiki_val_docs, wiki_test_docs, wiki_train_enc, wiki_val_enc, wiki_test_enc

# --- Phase 1: All-layer sweep ---
print(f"\n=== Phase 1: Sweeping {N_LAYERS} layers, {LAYERS_PER_PASS} per pass [{elapsed_str()}] ===")
layer_profile = {}
layer_chunks = [
    list(range(i, min(i + LAYERS_PER_PASS, N_LAYERS))) for i in range(0, N_LAYERS, LAYERS_PER_PASS)
]

for chunk in layer_chunks:
    t0 = time.time()
    print(f"  Collecting L{chunk[0]}-L{chunk[-1]} train...")
    tr_multi = collect_multi_layer_fast(model, train_batches, chunk, MAX_TRAIN, DEVICE)
    print(f"  Collecting L{chunk[0]}-L{chunk[-1]} val...")
    va_multi = collect_multi_layer_fast(model, val_batches, chunk, MAX_TRAIN, DEVICE)
    for layer in chunk:
        head = train_linear_binary(tr_multi[layer], seed=LAYER_SELECT_SEED)
        _, rho, _ = evaluate_head(head, va_multi[layer])
        layer_profile[layer] = float(rho)
        print(f"    L{layer:>2}: {rho:+.4f}")
    elapsed = time.time() - t0
    done = chunk[-1] + 1
    remaining = (N_LAYERS - done) / len(chunk) * elapsed
    print(f"  chunk done in {elapsed:.0f}s, ~{remaining / 60:.0f}m remaining [{elapsed_str()}]")
    del tr_multi, va_multi
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

peak_layer = max(layer_profile, key=layer_profile.get)
output_layer = N_LAYERS - 1
if peak_layer >= output_layer - 1:
    mid = {l: r for l, r in layer_profile.items() if l <= int(0.8 * N_LAYERS)}
    if mid:
        peak_layer = max(mid, key=mid.get)
candidates = sorted(
    [
        l
        for l, _ in sorted(layer_profile.items(), key=lambda x: x[1], reverse=True)[:4]
        if l <= int(0.8 * N_LAYERS)
    ]
)
if peak_layer not in candidates:
    candidates.append(peak_layer)
    candidates.sort()
print(f"\nPeak: L{peak_layer}, candidates: {candidates}")
save_checkpoint(
    "1_sweep",
    model=MODEL_ID,
    n_layers=N_LAYERS,
    hidden_dim=HIDDEN_DIM,
    layer_profile={str(k): v for k, v in sorted(layer_profile.items())},
    peak_layer=peak_layer,
    candidates=candidates,
)

# --- Phase 2: Collect at candidates + output ---
print(f"\n=== Phase 2: Collecting candidates + output [{elapsed_str()}] ===")
print(f"  Collecting candidates {candidates} train...")
tr_multi = collect_multi_layer_fast(model, train_batches, candidates, MAX_TRAIN, DEVICE)
print(f"  Collecting candidates {candidates} val...")
va_multi = collect_multi_layer_fast(model, val_batches, candidates, MAX_TRAIN, DEVICE)
train_cache = {l: tr_multi[l] for l in candidates}
val_cache = {l: va_multi[l] for l in candidates}
del tr_multi, va_multi
gc.collect()

print(f"  Collecting output layer L{output_layer}...")
wiki_train_output = collect_single_layer_fast(model, train_batches, output_layer, MAX_TRAIN, DEVICE)
wiki_val_output = collect_single_layer_fast(model, val_batches, output_layer, MAX_TRAIN, DEVICE)

# --- Phase 3: Multi-seed eval ---
print(f"\n=== Phase 3: Multi-seed eval [{elapsed_str()}] ===")
layer_eval = {}
for layer in candidates:
    seed_rhos, seed_scores = [], []
    for seed in EVAL_SEEDS:
        head = train_linear_binary(train_cache[layer], seed=seed)
        scores, rho, _ = evaluate_head(head, val_cache[layer])
        seed_rhos.append(float(rho))
        seed_scores.append(scores)
    pw = [
        float(spearmanr(seed_scores[i], seed_scores[j])[0])
        for i in range(len(EVAL_SEEDS))
        for j in range(i + 1, len(EVAL_SEEDS))
    ]
    layer_eval[layer] = {
        "mean": float(np.mean(seed_rhos)),
        "std": float(np.std(seed_rhos)),
        "per_seed": seed_rhos,
        "seed_agreement": float(np.mean(pw)),
    }
    print(f"  L{layer}: {np.mean(seed_rhos):+.4f} +/- {np.std(seed_rhos):.4f}  agree={np.mean(pw):.4f}")

FINAL = max(layer_eval, key=lambda l: layer_eval[l]["mean"])
ev = layer_eval[FINAL]
wiki_train_peak = train_cache[FINAL]
wiki_val_peak = val_cache[FINAL]
print(f"FINAL: L{FINAL} = {ev['mean']:+.4f} +/- {ev['std']:.4f}")
save_checkpoint(
    "3_multiseed",
    model=MODEL_ID,
    n_layers=N_LAYERS,
    hidden_dim=HIDDEN_DIM,
    layer_profile={str(k): v for k, v in sorted(layer_profile.items())},
    peak_layer_final=FINAL,
    peak_layer_frac=round(FINAL / N_LAYERS, 2),
    multi_layer_eval={str(l): d for l, d in layer_eval.items()},
    partial_corr={"mean": ev["mean"], "std": ev["std"], "per_seed": ev["per_seed"]},
    seed_agreement={"mean": ev["seed_agreement"]},
)

# --- Phase 4: Test + C4 at FINAL ---
print(f"\n=== Phase 4: Test + C4 at FINAL [{elapsed_str()}] ===")
wiki_test_peak = collect_single_layer_fast(model, test_batches, FINAL, MAX_TRAIN, DEVICE)

print("  Loading C4...")
c4_docs = []
ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
for i, row in enumerate(ds):
    if i < 50000:
        continue
    text = row["text"].strip()
    if len(text) > 100:
        c4_docs.append(text)
    if len(c4_docs) >= 500:
        break
c4_test_enc = pretokenize(c4_docs, tokenizer)
c4_test_batches = build_batches(c4_test_enc, BATCH_SIZE)
c4_test_peak = collect_single_layer_fast(model, c4_test_batches, FINAL, MAX_TRAIN // 2, DEVICE)

c4_train_docs = []
ds2 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
for row in ds2:
    text = row["text"].strip()
    if len(text) > 100:
        c4_train_docs.append(text)
    if len(c4_train_docs) >= 8000:
        break
c4_train_enc = pretokenize(c4_train_docs, tokenizer)
c4_train_batches = build_batches(c4_train_enc, BATCH_SIZE)
c4_train_peak = collect_single_layer_fast(model, c4_train_batches, FINAL, MAX_TRAIN, DEVICE)

del model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print(f"Model unloaded. [{elapsed_str()}]")

# --- Phase 5: Full battery ---
print(f"\n=== Phase 5: Full battery [{elapsed_str()}] ===")
test_rhos = [
    float(evaluate_head(train_linear_binary(wiki_train_peak, seed=s), wiki_test_peak)[1])
    for s in EVAL_SEEDS[:3]
]
print(f"  test mean: {np.mean(test_rhos):+.4f}")

# Baselines
print("\n  Baselines:")
bl = compute_hand_designed(wiki_val_peak)
baseline_results = {
    n: float(
        partial_spearman(
            s, wiki_val_peak["losses"], [wiki_val_peak["max_softmax"], wiki_val_peak["activation_norm"]]
        )[0]
    )
    for n, s in bl.items()
}
torch.manual_seed(99)
rh = torch.nn.Linear(HIDDEN_DIM, 1)
rh.eval()
with torch.inference_mode():
    baseline_results["random_head"] = float(
        partial_spearman(
            rh(wiki_val_peak["activations"]).squeeze(-1).numpy(),
            wiki_val_peak["losses"],
            [wiki_val_peak["max_softmax"], wiki_val_peak["activation_norm"]],
        )[0]
    )
for n, v in baseline_results.items():
    print(f"    {n}: {v:+.4f}")

# Output-controlled
print("\n  Output-controlled:")
ctrl_rhos = []
for seed in EVAL_SEEDS[:3]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    pred = torch.nn.Sequential(
        torch.nn.Linear(wiki_train_output["activations"].size(1), 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(pred.parameters(), lr=1e-3, weight_decay=1e-4)
    tds = torch.utils.data.TensorDataset(
        wiki_train_output["activations"], torch.from_numpy(wiki_train_output["losses"]).float()
    )
    tdl = torch.utils.data.DataLoader(tds, batch_size=1024, shuffle=True)
    for _ep in range(20):
        for bx, by in tdl:
            l = F.mse_loss(pred(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
    pred.eval()
    with torch.inference_mode():
        ps = pred(wiki_val_output["activations"]).squeeze(-1).numpy()
    obs = train_linear_binary(wiki_train_peak, seed=seed)
    obs.eval()
    with torch.inference_mode():
        os_ = obs(wiki_val_peak["activations"]).squeeze(-1).numpy()
    r, _ = partial_spearman(
        os_, wiki_val_peak["losses"], [wiki_val_peak["max_softmax"], wiki_val_peak["activation_norm"], ps]
    )
    ctrl_rhos.append(float(r))
    print(f"    seed {seed}: {r:+.4f}")

# Cross-domain
print("\n  Cross-domain:")
domain_results = {}
for dn, td in [("wikitext", wiki_val_peak), ("c4", c4_test_peak)]:
    domain_results[dn] = float(
        np.mean(
            [
                float(evaluate_head(train_linear_binary(wiki_train_peak, seed=s), td)[1])
                for s in EVAL_SEEDS[:3]
            ]
        )
    )
    print(f"    {dn}: {domain_results[dn]:+.4f}")
domain_results["c4_within"] = float(
    np.mean(
        [
            float(evaluate_head(train_linear_binary(c4_train_peak, seed=s), c4_test_peak)[1])
            for s in EVAL_SEEDS[:3]
        ]
    )
)
print(f"    c4_within: {domain_results['c4_within']:+.4f}")

# Control sensitivity
print("\n  Control sensitivity:")
torch.manual_seed(42)
conf_feats = torch.from_numpy(
    np.column_stack([wiki_train_peak["max_softmax"], wiki_train_peak["activation_norm"]])
).float()
loss_tgt = torch.from_numpy(wiki_train_peak["losses"]).float()
mlp_ctrl = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
opt = torch.optim.Adam(mlp_ctrl.parameters(), lr=1e-3, weight_decay=1e-4)
cs_ds = torch.utils.data.TensorDataset(conf_feats, loss_tgt)
cs_dl = torch.utils.data.DataLoader(cs_ds, batch_size=1024, shuffle=True)
for _ep in range(20):
    for bx, by in cs_dl:
        loss = F.mse_loss(mlp_ctrl(bx).squeeze(-1), by)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
mlp_ctrl.eval()
with torch.inference_mode():
    mlp_pred = (
        mlp_ctrl(
            torch.from_numpy(
                np.column_stack([wiki_val_peak["max_softmax"], wiki_val_peak["activation_norm"]])
            ).float()
        )
        .squeeze(-1)
        .numpy()
    )

cs_head = train_linear_binary(wiki_train_peak, seed=EVAL_SEEDS[0])
cs_head.eval()
with torch.inference_mode():
    cs_obs = cs_head(wiki_val_peak["activations"]).squeeze(-1).numpy()

td = wiki_val_peak
ctrl_results = {}
for name, covs in [
    ("none", None),
    ("softmax_only", [td["max_softmax"]]),
    ("norm_only", [td["activation_norm"]]),
    ("standard", [td["max_softmax"], td["activation_norm"]]),
    ("plus_entropy", [td["max_softmax"], td["activation_norm"], td["logit_entropy"]]),
    ("nonlinear", [mlp_pred]),
]:
    if covs is None:
        r, _ = spearmanr(cs_obs, td["losses"])
    else:
        r, _ = partial_spearman(cs_obs, td["losses"], covs)
    ctrl_results[name] = float(r)
    print(f"    {name:<16}: {r:+.4f}")

# Flagging
print("\n  Flagging:")
nf = min(len(wiki_val_peak["losses"]), len(wiki_val_output["losses"]))
fl = wiki_val_peak["losses"][:nf]
fsm = wiki_val_output["max_softmax"][:nf]
fa = wiki_val_peak["activations"][:nf]
ml = float(np.median(fl))
ihl = fl > ml
fr = [0.05, 0.10, 0.20, 0.30]
fres = []
for seed in EVAL_SEEDS[:3]:
    h = train_linear_binary(wiki_train_peak, seed=seed)
    h.eval()
    with torch.inference_mode():
        osc = h(fa).squeeze(-1).numpy()
    sr = {"observer": {}, "confidence": {}, "exclusive": {}}
    for rate in fr:
        k = int(nf * rate)
        of = osc >= np.sort(osc)[-k]
        cf = fsm <= np.sort(fsm)[k]
        sr["observer"][str(rate)] = float(ihl[of].mean()) if of.sum() > 0 else 0.0
        sr["confidence"][str(rate)] = float(ihl[cf].mean()) if cf.sum() > 0 else 0.0
        sr["exclusive"][str(rate)] = {"observer_only": int((of & ~cf & ihl).sum())}
    fres.append(sr)
fs = {
    str(r): {"observer_exclusive": float(np.mean([s["exclusive"][str(r)]["observer_only"] for s in fres]))}
    for r in fr
}
print(
    f"  Error coverage at 10%: {fs['0.1']['observer_exclusive']:.0f} tokens ({fs['0.1']['observer_exclusive'] / (nf / 2) * 100:.1f}%)"
)

# --- Save ---
print(f"\n=== Saving [{elapsed_str()}] ===")
output = {
    "model": MODEL_ID,
    "n_params_b": round(n_params, 2),
    "n_layers": N_LAYERS,
    "hidden_dim": HIDDEN_DIM,
    "protocol": {
        "layer_select_seed": LAYER_SELECT_SEED,
        "eval_seeds": EVAL_SEEDS,
        "target_ex_per_dim": TARGET_EX_PER_DIM,
        "batch_size": BATCH_SIZE,
        "layers_per_pass": LAYERS_PER_PASS,
    },
    "peak_layer_final": FINAL,
    "peak_layer_frac": round(FINAL / N_LAYERS, 2),
    "layer_profile": {str(k): v for k, v in sorted(layer_profile.items())},
    "multi_layer_eval": {str(l): {k: v for k, v in d.items()} for l, d in layer_eval.items()},
    "partial_corr": {
        "mean": ev["mean"],
        "std": ev["std"],
        "per_seed": ev["per_seed"],
        "n_seeds": len(EVAL_SEEDS),
        "split": "validation (held-out seeds)",
    },
    "test_split_comparison": {"mean": float(np.mean(test_rhos)), "per_seed": test_rhos},
    "seed_agreement": {"mean": ev["seed_agreement"]},
    "output_controlled": {"mean": float(np.mean(ctrl_rhos)), "per_seed": ctrl_rhos},
    "baselines": baseline_results,
    "cross_domain": domain_results,
    "control_sensitivity": ctrl_results,
    "flagging_6a": {"n_tokens": nf, "summary": fs},
}
out_path = Path("/workspace/llama1b_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved {out_path}")
print(f"FINAL: L{FINAL} = {ev['mean']:+.4f} +/- {ev['std']:.4f}")
print(f"Output-controlled: {np.mean(ctrl_rhos):+.4f}")
print(f"Cross-domain C4: {domain_results.get('c4', '?')}")
print(f"Total time: {elapsed_str()}")
print(json.dumps(output, indent=2))
