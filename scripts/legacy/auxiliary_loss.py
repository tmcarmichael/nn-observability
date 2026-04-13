"""Qwen 0.5B: auxiliary observability loss.

Model: Qwen/Qwen2.5-0.5B | HF: https://huggingface.co/Qwen/Qwen2.5-0.5B
GPU: MPS / any | Lambda sweep [0, 0.01, 0.05, 0.1, 0.5] | 300 steps | Date: 2026-04-08

Usage: uv run --extra transformer scripts/auxiliary_loss.py
"""

from __future__ import annotations

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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_wikitext(split="test", max_docs=None):
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


def collect_layer_data(model, tokenizer, docs, layer, device, max_tokens=200000, max_length=512):
    model.eval()
    all_acts, all_losses, all_softmax, all_norms = [], [], [], []
    total = 0
    for doc in docs:
        if total >= max_tokens:
            break
        if not doc.strip():
            continue
        tokens = tokenizer(doc, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = tokens["input_ids"].to(device)
        if input_ids.size(1) < 2:
            continue
        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True)
        h = outputs.hidden_states[layer + 1][0, :-1, :].cpu()
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[0, 1:]
        losses = F.cross_entropy(logits, labels, reduction="none").cpu()
        sm = F.softmax(logits, dim=-1).max(dim=-1).values.cpu()
        norms = h.norm(dim=-1)
        all_acts.append(h)
        all_losses.append(losses)
        all_softmax.append(sm)
        all_norms.append(norms)
        total += h.size(0)
    print(f"    {total} positions from {len(all_acts)} documents")
    return {
        "activations": torch.cat(all_acts).float(),
        "losses": torch.cat(all_losses).float().numpy(),
        "max_softmax": torch.cat(all_softmax).float().numpy(),
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


def measure_perplexity(model, tokenizer, docs, device, max_tokens=50000):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.inference_mode():
        for doc in docs:
            if total_tokens >= max_tokens:
                break
            if not doc.strip():
                continue
            ids = tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
            if ids.size(1) < 2:
                continue
            logits = model(ids).logits[0, :-1, :]
            loss = F.cross_entropy(logits, ids[0, 1:], reduction="sum").item()
            total_loss += loss
            total_tokens += ids.size(1) - 1
    return float(np.exp(total_loss / total_tokens))


def clear_cache():
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    MODEL_ID = "Qwen/Qwen2.5-0.5B"
    TARGET_LAYER = 12
    AUX_LAMBDA = 0.05
    LR = 2e-5
    N_STEPS = 300
    BATCH_SEQ_LEN = 256  # shorter seqs to fit in MPS memory
    LOG_EVERY = 50

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    N_LAYERS = model.config.num_hidden_layers
    HIDDEN_DIM = model.config.hidden_size
    print(
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, {N_LAYERS} layers, {HIDDEN_DIM} dim"
    )

    # Load data
    wiki_train = load_wikitext("train", max_docs=2000)
    wiki_test = load_wikitext("test", max_docs=None)
    MAX_TRAIN = 350 * HIDDEN_DIM
    MAX_TEST = MAX_TRAIN // 2
    print(f"Token budget: {MAX_TRAIN} train ({MAX_TRAIN // HIDDEN_DIM} ex/dim)")

    # -----------------------------------------------------------------------
    # Baseline
    # -----------------------------------------------------------------------
    print("\n=== BASELINE ===")
    baseline_profile = {}
    for layer in [0, 4, 8, 12, 16, 20, 23]:
        tr = collect_layer_data(model, tokenizer, wiki_train, layer, DEVICE, MAX_TRAIN)
        te = collect_layer_data(model, tokenizer, wiki_test, layer, DEVICE, MAX_TEST)
        head = train_linear_binary(tr, seed=42)
        _, rho, _ = evaluate_head(head, te)
        baseline_profile[layer] = float(rho)
        print(f"  layer {layer:>2}: {rho:+.4f}")
        del tr, te

    baseline_ppl = measure_perplexity(model, tokenizer, wiki_test, DEVICE)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")

    # -----------------------------------------------------------------------
    # Frozen probe + OLS
    # -----------------------------------------------------------------------
    print("\n=== FROZEN PROBE ===")
    probe_train = collect_layer_data(model, tokenizer, wiki_train, TARGET_LAYER, DEVICE, MAX_TRAIN)
    X_ols = np.column_stack(
        [probe_train["max_softmax"], probe_train["activation_norm"], np.ones(len(probe_train["losses"]))]
    )
    ols_beta = np.linalg.lstsq(X_ols, probe_train["losses"], rcond=None)[0]
    ols_beta_tensor = torch.tensor(ols_beta, dtype=torch.float32).to(DEVICE)

    frozen_probe = train_linear_binary(probe_train, seed=42)
    frozen_probe.eval()
    for p in frozen_probe.parameters():
        p.requires_grad_(False)
    frozen_probe.to(DEVICE)

    probe_test = collect_layer_data(model, tokenizer, wiki_test, TARGET_LAYER, DEVICE, MAX_TEST)
    _, baseline_rho_target, _ = evaluate_head(frozen_probe.cpu(), probe_test)
    frozen_probe.to(DEVICE)
    print(f"Frozen probe baseline at L{TARGET_LAYER}: {baseline_rho_target:+.4f}")
    del probe_train, probe_test
    clear_cache()
    import time

    time.sleep(1)  # let MPS release memory

    # -----------------------------------------------------------------------
    # Fine-tune
    # -----------------------------------------------------------------------
    print(f"\n=== FINE-TUNING (lambda={AUX_LAMBDA}, steps={N_STEPS}) ===")
    train_ids = []
    for doc in wiki_train:
        if not doc.strip():
            continue
        ids = tokenizer(doc, return_tensors="pt", truncation=True, max_length=BATCH_SEQ_LEN)["input_ids"]
        if ids.size(1) >= 64:
            train_ids.append(ids.squeeze(0))
    print(f"{len(train_ids)} training sequences")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    lm_losses, aux_losses = [], []
    rng = np.random.default_rng(42)

    for step in range(N_STEPS):
        idx = rng.integers(0, len(train_ids))
        input_ids = train_ids[idx].unsqueeze(0).to(DEVICE)
        outputs = model(input_ids, output_hidden_states=True)

        logits = outputs.logits[0, :-1, :]
        labels = input_ids[0, 1:]
        lm_loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=-1)
            sm = probs.max(dim=-1).values
            h_target = outputs.hidden_states[TARGET_LAYER + 1][0, :-1, :]
            norms = h_target.detach().norm(dim=-1)
            per_token_loss = F.cross_entropy(logits.detach(), labels, reduction="none")
            predicted = sm * ols_beta_tensor[0] + norms * ols_beta_tensor[1] + ols_beta_tensor[2]
            binary_target = (per_token_loss - predicted > 0).float()

        h_for_probe = outputs.hidden_states[TARGET_LAYER + 1][0, :-1, :].float()
        aux_loss = F.binary_cross_entropy_with_logits(frozen_probe(h_for_probe).squeeze(-1), binary_target)

        total_loss = lm_loss + AUX_LAMBDA * aux_loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        lm_losses.append(lm_loss.item())
        aux_losses.append(aux_loss.item())
        if (step + 1) % LOG_EVERY == 0:
            print(
                f"  step {step + 1}/{N_STEPS}: lm={np.mean(lm_losses[-LOG_EVERY:]):.4f}  "
                f"aux={np.mean(aux_losses[-LOG_EVERY:]):.4f}"
            )

    model.eval()
    print("Fine-tuning complete.")

    # -----------------------------------------------------------------------
    # Post-training
    # -----------------------------------------------------------------------
    print("\n=== POST-TRAINING ===")
    post_profile = {}
    for layer in [0, 4, 8, 12, 16, 20, 23]:
        tr = collect_layer_data(model, tokenizer, wiki_train, layer, DEVICE, MAX_TRAIN)
        te = collect_layer_data(model, tokenizer, wiki_test, layer, DEVICE, MAX_TEST)
        head = train_linear_binary(tr, seed=42)
        _, rho, _ = evaluate_head(head, te)
        post_profile[layer] = float(rho)
        delta = rho - baseline_profile[layer]
        print(f"  layer {layer:>2}: {rho:+.4f}  (was {baseline_profile[layer]:+.4f}, delta {delta:+.4f})")
        del tr, te

    post_test = collect_layer_data(model, tokenizer, wiki_test, TARGET_LAYER, DEVICE, MAX_TEST)
    _, post_rho_frozen, _ = evaluate_head(frozen_probe.cpu(), post_test)
    frozen_probe.to(DEVICE)
    print(f"\nFrozen probe at L{TARGET_LAYER}: {post_rho_frozen:+.4f} (was {baseline_rho_target:+.4f})")

    post_ppl = measure_perplexity(model, tokenizer, wiki_test, DEVICE)
    print(f"Perplexity: {baseline_ppl:.2f} -> {post_ppl:.2f} (delta {post_ppl - baseline_ppl:+.2f})")

    del model
    clear_cache()

    # -----------------------------------------------------------------------
    # Lambda sweep
    # -----------------------------------------------------------------------
    print("\n=== LAMBDA SWEEP ===")
    lambdas_to_test = [0.0, 0.01, 0.05, 0.1, 0.5]
    sweep_results = []

    for lam in lambdas_to_test:
        print(f"\n--- Lambda = {lam} ---")
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(DEVICE)

        if lam > 0:
            m.train()
            opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
            for _step in range(N_STEPS):
                idx = rng.integers(0, len(train_ids))
                ids = train_ids[idx].unsqueeze(0).to(DEVICE)
                out = m(ids, output_hidden_states=True)
                logits = out.logits[0, :-1, :]
                labs = ids[0, 1:]
                lm_l = F.cross_entropy(logits, labs)
                with torch.no_grad():
                    pr = F.softmax(logits.detach(), dim=-1)
                    sm = pr.max(dim=-1).values
                    h_t = out.hidden_states[TARGET_LAYER + 1][0, :-1, :]
                    nr = h_t.detach().norm(dim=-1)
                    ptl = F.cross_entropy(logits.detach(), labs, reduction="none")
                    pred = sm * ols_beta_tensor[0] + nr * ols_beta_tensor[1] + ols_beta_tensor[2]
                    bt = (ptl - pred > 0).float()
                h_fp = out.hidden_states[TARGET_LAYER + 1][0, :-1, :].float()
                al = F.binary_cross_entropy_with_logits(frozen_probe(h_fp).squeeze(-1), bt)
                total = lm_l + lam * al
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()

        m.eval()
        tr = collect_layer_data(m, tokenizer, wiki_train, TARGET_LAYER, DEVICE, MAX_TRAIN)
        te = collect_layer_data(m, tokenizer, wiki_test, TARGET_LAYER, DEVICE, MAX_TEST)
        head = train_linear_binary(tr, seed=42)
        _, rho, _ = evaluate_head(head, te)
        ppl = measure_perplexity(m, tokenizer, wiki_test, DEVICE)

        print(f"  Observability: {rho:+.4f}  Perplexity: {ppl:.2f}")
        sweep_results.append({"lambda": lam, "partial_corr": float(rho), "perplexity": float(ppl)})
        del m, tr, te
        clear_cache()

    print(f"\n{'Lambda':<10} {'Partial corr':>14} {'Perplexity':>12} {'Obs delta':>12} {'PPL delta':>12}")
    print("-" * 62)
    base_obs = sweep_results[0]["partial_corr"]
    base_ppl_sweep = sweep_results[0]["perplexity"]
    for r in sweep_results:
        print(
            f"{r['lambda']:<10} {r['partial_corr']:>+14.4f} {r['perplexity']:>12.2f} "
            f"{r['partial_corr'] - base_obs:>+12.4f} {r['perplexity'] - base_ppl_sweep:>+12.2f}"
        )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output = {
        "model": MODEL_ID,
        "target_layer": TARGET_LAYER,
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "fine_tuning": {"lr": LR, "n_steps": N_STEPS, "aux_lambda": AUX_LAMBDA},
        "baseline": {
            "layer_profile": {str(k): v for k, v in baseline_profile.items()},
            "perplexity": baseline_ppl,
            "frozen_probe_rho": baseline_rho_target,
        },
        "post_training": {
            "layer_profile": {str(k): v for k, v in post_profile.items()},
            "perplexity": post_ppl,
            "frozen_probe_rho": float(post_rho_frozen),
        },
        "lambda_sweep": sweep_results,
        "deltas": {
            "observability_at_target": float(post_profile[TARGET_LAYER] - baseline_profile[TARGET_LAYER]),
            "perplexity": float(post_ppl - baseline_ppl),
            "frozen_probe": float(post_rho_frozen - baseline_rho_target),
        },
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "auxiliary_loss_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {out_path}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
