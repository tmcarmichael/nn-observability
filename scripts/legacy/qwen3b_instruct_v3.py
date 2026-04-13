"""Qwen 2.5 3B Instruct: v3 with full battery including control sensitivity.

Model: Qwen/Qwen2.5-3B-Instruct | GPU: H100 / A100 | 7 seeds (43-49) | 350 ex/dim
Protocol: layer selected on val with seed 42, evaluated on val with seeds 43-49
Full battery: baselines, output-controlled, cross-domain, control sensitivity, flagging

Usage: pip install transformers datasets scipy && python scripts/qwen3b_instruct_v3.py
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
print(f"Device: {DEVICE}")

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


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


def collect_layer_data(
    model, tokenizer, docs, layer, device, max_tokens=200000, max_length=512, batch_size=8
):
    model.eval()
    all_acts, all_losses, all_softmax, all_entropy, all_norms = [], [], [], [], []
    total = 0
    n_batches = 0
    batch_docs = []
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured["h"] = h

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    def process_batch(batch):
        nonlocal total, n_batches
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)
        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attn_mask)
        h_all = captured["h"]
        for b in range(input_ids.size(0)):
            if total >= max_tokens:
                break
            seq_mask = attn_mask[b].bool()
            valid_len = seq_mask.sum().item()
            if valid_len < 2:
                continue
            h = h_all[b, : valid_len - 1, :].float().cpu()
            logits = outputs.logits[b, : valid_len - 1, :]
            labels = input_ids[b, 1:valid_len]
            losses = F.cross_entropy(logits, labels, reduction="none").cpu()
            probs = F.softmax(logits, dim=-1)
            sm = probs.max(dim=-1).values.cpu()
            ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
            norms = h.norm(dim=-1)
            all_acts.append(h)
            all_losses.append(losses)
            all_softmax.append(sm)
            all_entropy.append(ent)
            all_norms.append(norms)
            total += h.size(0)
        del outputs, captured["h"]
        if device == "cuda":
            torch.cuda.empty_cache()
        n_batches += 1
        if n_batches % 50 == 0:
            print(f"      batch {n_batches}, {total} positions so far")

    for doc in docs:
        if total >= max_tokens:
            break
        if not doc.strip():
            continue
        batch_docs.append(doc)
        if len(batch_docs) >= batch_size:
            process_batch(batch_docs)
            batch_docs = []
    if batch_docs and total < max_tokens:
        process_batch(batch_docs)

    handle.remove()
    print(f"    {total} positions from {len(all_acts)} documents")
    return {
        "activations": torch.cat(all_acts).float(),
        "losses": torch.cat(all_losses).float().numpy(),
        "max_softmax": torch.cat(all_softmax).float().numpy(),
        "logit_entropy": torch.cat(all_entropy).float().numpy(),
        "activation_norm": torch.cat(all_norms).float().numpy(),
    }


def train_linear_binary(train_data, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"]
    residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    targets = torch.from_numpy((residuals > 0).astype(np.float32))
    head = torch.nn.Linear(acts.size(1), 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head


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

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
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
print(f"Token budget: {MAX_TRAIN} train ({TARGET_EX_PER_DIM} ex/dim)")

wiki_train = load_wikitext("train", max_docs=8000)
wiki_val = load_wikitext("validation", max_docs=None)
wiki_test = load_wikitext("test", max_docs=None)
print(f"{len(wiki_train)} train, {len(wiki_val)} val, {len(wiki_test)} test")

# --- Phase 1: All-layer sweep ---
print(f"\n=== Phase 1: Sweeping {N_LAYERS} layers [{elapsed_str()}] ===")
layer_profile = {}
layer_times = []
for layer in range(N_LAYERS):
    t0 = time.time()
    tr = collect_layer_data(model, tokenizer, wiki_train, layer, DEVICE, MAX_TRAIN)
    va = collect_layer_data(model, tokenizer, wiki_val, layer, DEVICE, MAX_TRAIN)
    head = train_linear_binary(tr, seed=LAYER_SELECT_SEED)
    _, rho, _ = evaluate_head(head, va)
    layer_profile[layer] = float(rho)
    dt = time.time() - t0
    layer_times.append(dt)
    remaining = (N_LAYERS - layer - 1) * np.mean(layer_times)
    print(f"  L{layer:>2}: {rho:+.4f}  ({dt:.0f}s, ~{remaining / 60:.0f}m left, {elapsed_str()} elapsed)")
    del tr, va
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

# --- Phase 2: Collect at candidates + output ---
print(f"\n=== Phase 2: Collecting candidates + output [{elapsed_str()}] ===")
train_cache, val_cache = {}, {}
for layer in candidates:
    print(f"  L{layer}...")
    train_cache[layer] = collect_layer_data(model, tokenizer, wiki_train, layer, DEVICE, MAX_TRAIN)
    val_cache[layer] = collect_layer_data(model, tokenizer, wiki_val, layer, DEVICE, MAX_TRAIN)

print(f"  output layer L{output_layer}...")
wiki_train_output = collect_layer_data(model, tokenizer, wiki_train, output_layer, DEVICE, MAX_TRAIN)
wiki_val_output = collect_layer_data(model, tokenizer, wiki_val, output_layer, DEVICE, MAX_TRAIN)

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

# --- Phase 4: Test + C4 at FINAL ---
print(f"\n=== Phase 4: Test + C4 at FINAL [{elapsed_str()}] ===")
wiki_test_peak = collect_layer_data(model, tokenizer, wiki_test, FINAL, DEVICE, MAX_TRAIN)

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
c4_test_peak = collect_layer_data(model, tokenizer, c4_docs, FINAL, DEVICE, MAX_TRAIN // 2)

c4_train_docs = []
ds2 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
for row in ds2:
    text = row["text"].strip()
    if len(text) > 100:
        c4_train_docs.append(text)
    if len(c4_train_docs) >= 8000:
        break
c4_train_peak = collect_layer_data(model, tokenizer, c4_train_docs, FINAL, DEVICE, MAX_TRAIN)

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
        "batch_size": 8,
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
out_path = Path("/workspace/qwen3b_instruct_v3_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved {out_path}")
print(f"FINAL: L{FINAL} = {ev['mean']:+.4f} +/- {ev['std']:.4f}")
print(f"Output-controlled: {np.mean(ctrl_rhos):+.4f}")
print(f"Cross-domain C4: {domain_results.get('c4', '?')}")
print(f"Total time: {elapsed_str()}")
print(json.dumps(output, indent=2))
