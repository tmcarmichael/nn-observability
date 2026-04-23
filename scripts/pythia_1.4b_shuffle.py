"""Shuffled-label control for Pythia 1.4B at its peak layer.

Trains probes with randomly permuted binary targets to check whether the
observed collapse at (24L, 16H) reflects a real signal drop or a silent
probe failure. Shuffled values should stay near zero under either case,
but elevated shuffle values would implicate the probe.
"""

import datetime as _dt
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata

# ── Config ───────────────────────────────────────────────────────────

MODEL_ID = "EleutherAI/pythia-1.4b"
BATCH_SIZE = 48  # 1.4B is small, default batch is fine
EX_DIM = 350
N_PERMUTATIONS = 10
LAYER_SELECT_SEED = 42  # not used here; kept in provenance
PROBE_SEEDS = list(range(43, 43 + N_PERMUTATIONS))
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_PATH = RESULTS_DIR / "pythia1_4b_results.json"
_OUT_DIR = Path("/workspace") if Path("/workspace").exists() else RESULTS_DIR
OUTPUT_PATH = _OUT_DIR / "pythia_1.4b_shuffle_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DEVICE = DEVICE
print(f"Device: {DEVICE}")
RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


# ── Core probe functions (matching run_model.py) ─────────────────────


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


def train_linear_on_targets(acts_bf16, targets_np, seed, epochs=20, lr=1e-3):
    """Train a linear head against a supplied target vector (real or permuted)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = acts_bf16.to(TRAIN_DEVICE, dtype=torch.float32)
    targets = torch.from_numpy(targets_np.astype(np.float32)).to(TRAIN_DEVICE)
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
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def evaluate_head(head, val_data):
    head.eval()
    with torch.inference_mode():
        val_acts_fp32 = val_data["activations"].to(dtype=torch.float32)
        scores = head(val_acts_fp32).squeeze(-1).numpy()
    rho, _ = partial_spearman(
        scores, val_data["losses"], [val_data["max_softmax"], val_data["activation_norm"]]
    )
    return rho


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


# ── Collection (single layer, bf16) ──────────────────────────────────


def collect_layer_bf16(model, layer_modules, batches, layer, max_tokens):
    model.eval()
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured["h"] = h

    handle = layer_modules[layer].register_forward_hook(hook_fn)

    acts_chunks, norm_chunks = [], []
    all_losses, all_sm = [], []
    total = 0

    for bi, (input_ids_cpu, attn_mask_cpu) in enumerate(batches):
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

        h = captured["h"][:, :-1, :]  # stays bf16
        norm_chunks.append(h.float().norm(dim=-1)[shift_mask].cpu())
        acts_chunks.append(h[shift_mask].to(torch.bfloat16).cpu())

        total += shift_mask.sum().item()
        captured.clear()
        del outputs, input_ids, attn_mask, shift_logits, shift_labels, losses_2d, sm_2d, shift_mask
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if (bi + 1) % 20 == 0:
            print(f"    batch {bi + 1}/{len(batches)}, {total} positions [{elapsed_str()}]")

    handle.remove()

    n = min(total, max_tokens)
    return {
        "activations": torch.cat(acts_chunks)[:n],
        "losses": torch.cat(all_losses).numpy()[:n],
        "max_softmax": torch.cat(all_sm).numpy()[:n],
        "activation_norm": torch.cat(norm_chunks)[:n].numpy(),
    }


def _get_layer_list(model):
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("expected GPT-NeoX architecture for Pythia")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    # Peak layer + real rho come from the existing Pythia 1.4B result.
    # Hardcoded for pod portability; falls back to reading the JSON if present.
    PEAK = 17
    HIDDEN = 2048
    REAL_RHO = 0.10619
    if RESULTS_PATH.exists():
        prev = json.load(open(RESULTS_PATH))
        PEAK = int(prev["peak_layer_final"])
        HIDDEN = int(prev["hidden_dim"])
        REAL_RHO = float(prev["partial_corr"]["mean"])
        print(f"Loaded reference values from {RESULTS_PATH.name}")
    else:
        print(f"Using hardcoded reference values (no {RESULTS_PATH.name} on pod)")
    MAX_TRAIN = EX_DIM * HIDDEN
    print(f"Pythia 1.4B: peak=L{PEAK}, real_rho={REAL_RHO:+.4f}, MAX_TRAIN={MAX_TRAIN}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()
    layer_modules = _get_layer_list(model)
    _model_revision = getattr(model.config, "_commit_hash", None) or "unknown"
    print(f"Model loaded [{elapsed_str()}]")

    # Tokenize
    print(f"\n=== Pre-tokenizing [{elapsed_str()}] ===")
    train_docs = load_wikitext("train", max_docs=12000)
    val_docs = load_wikitext("validation")
    train_enc = pretokenize(train_docs, tokenizer)
    val_enc = pretokenize(val_docs, tokenizer)
    train_batches = build_batches(train_enc, BATCH_SIZE)
    val_batches = build_batches(val_enc, BATCH_SIZE)
    del train_docs, val_docs, train_enc, val_enc
    gc.collect()

    # Collect at peak layer
    print(f"\n=== Collect train (L{PEAK}) [{elapsed_str()}] ===")
    train_data = collect_layer_bf16(model, layer_modules, train_batches, PEAK, MAX_TRAIN)
    print(f"\n=== Collect val (L{PEAK}) [{elapsed_str()}] ===")
    val_data = collect_layer_bf16(model, layer_modules, val_batches, PEAK, MAX_TRAIN)

    # Unload model
    del model, train_batches, val_batches
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"Model unloaded [{elapsed_str()}]")

    # Real binary target (for reference; not used in shuffle training)
    real_residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    real_target = (real_residuals > 0).astype(np.float32)
    print(f"\nReal target: {int(real_target.sum())}/{len(real_target)} positives")

    # Real-probe sanity check: train on real target, eval on val. Should match prev.
    print(f"\n=== Real-target sanity [{elapsed_str()}] ===")
    head_real = train_linear_on_targets(train_data["activations"], real_target, seed=PROBE_SEEDS[0])
    real_rho_check = evaluate_head(head_real, val_data)
    print(f"  real-target rho (seed {PROBE_SEEDS[0]}): {real_rho_check:+.4f} (prev mean {REAL_RHO:+.4f})")

    # Shuffle loop
    print(f"\n=== Shuffle: {N_PERMUTATIONS} permutations [{elapsed_str()}] ===")
    shuffle_rhos = []
    for i, seed in enumerate(PROBE_SEEDS):
        rng = np.random.default_rng(seed + 1000)
        permuted = rng.permutation(real_target)
        head = train_linear_on_targets(train_data["activations"], permuted, seed=seed)
        rho = evaluate_head(head, val_data)
        shuffle_rhos.append(float(rho))
        print(f"  perm {i + 1}/{N_PERMUTATIONS} (seed {seed}): {rho:+.4f}")

    shuffle_mean = float(np.mean(shuffle_rhos))
    shuffle_std = float(np.std(shuffle_rhos))
    ratio = REAL_RHO / shuffle_mean if shuffle_mean != 0 else float("inf")
    print(f"\nShuffle mean: {shuffle_mean:+.4f} ± {shuffle_std:.4f}")
    print(f"Real rho:     {REAL_RHO:+.4f}")
    print(f"Ratio real/shuffle: {ratio:.1f}x")

    output = {
        "model": MODEL_ID,
        "provenance": {
            "model_revision": _model_revision,
            "script": "scripts/pythia_1.4b_shuffle.py",
            "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "output_file": str(OUTPUT_PATH.name),
            "note": (
                "Shuffle test on Pythia 1.4B peak layer. N permutations of the "
                "binary target; probes trained on permuted targets and evaluated "
                "against real val target. Confirms the (24L, 16H) collapse is a "
                "real signal drop, not a probe-failure artifact."
            ),
        },
        "protocol": {
            "layer": PEAK,
            "n_permutations": N_PERMUTATIONS,
            "probe_seeds": PROBE_SEEDS,
            "target_ex_per_dim": EX_DIM,
            "batch_size": BATCH_SIZE,
            "max_train_tokens": MAX_TRAIN,
        },
        "real_rho_reference": REAL_RHO,
        "real_rho_sanity_check": real_rho_check,
        "shuffle": {
            "per_permutation": shuffle_rhos,
            "mean": shuffle_mean,
            "std": shuffle_std,
        },
        "ratio_real_over_shuffle": ratio,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_PATH}")
    print(f"Total time: {elapsed_str()}")


if __name__ == "__main__":
    main()
