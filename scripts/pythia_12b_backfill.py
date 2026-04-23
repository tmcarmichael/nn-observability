"""Backfill the later probe phases for Pythia 12B from a saved checkpoint.

Reads the peak layer from the existing checkpoint, recollects only the
peak and output layer activations, and merges the results JSON to match
the schema produced by run_model.py.
"""

import datetime as _dt
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr

# ── Config ───────────────────────────────────────────────────────────

MODEL_ID = "EleutherAI/pythia-12b"
BATCH_SIZE = 24
EX_DIM = 350
EVAL_SEEDS = list(range(43, 50))  # 7 seeds; match run_model.py default
LAYER_SELECT_SEED = 42
_OUT_DIR = (
    Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parent.parent / "results"
)
CHECKPOINT_PATH = _OUT_DIR / "pythia_12b_checkpoint.json"
OUTPUT_PATH = _OUT_DIR / "pythia_12b_results.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DEVICE = DEVICE
print(f"Device: {DEVICE}")
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
    # Upcast bf16 → fp32 only at training time, on device
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
    if DEVICE == "cuda":
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


def compute_hand_designed(data):
    acts = data["activations"].to(dtype=torch.float32)
    p = acts.abs() / (acts.abs().sum(dim=1, keepdim=True) + 1e-8)
    return {
        "ff_goodness": (acts**2).mean(dim=1).numpy(),
        "active_ratio": (acts.abs() > 0.01).float().mean(dim=1).numpy(),
        "act_entropy": -(p * (p + 1e-8).log()).sum(dim=1).numpy(),
        "activation_norm": data["activation_norm"],
    }


# ── Data loading ─────────────────────────────────────────────────────


def load_wikitext(split="test", max_docs=None):
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


# ── Collection: only (peak, output), bf16 CPU storage ────────────────


def collect_two_layers_bf16(model, layer_modules, batches, layers, max_tokens):
    """Collect activations at exactly `layers` (expected length 2) as bf16 on CPU."""
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

        handles.append(layer_modules[layer].register_forward_hook(make_hook(layer)))

    per_layer_acts = {l: [] for l in layers}
    per_layer_norms = {l: [] for l in layers}
    all_losses, all_sm, all_ent = [], [], []
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
        ent_2d = torch.empty(B, S - 1, device=DEVICE)
        for ci in range(0, B, 8):
            p = shift_logits[ci : ci + 8].float().softmax(dim=-1)
            sm_2d[ci : ci + 8] = p.max(dim=-1).values
            ent_2d[ci : ci + 8] = -(p * (p + 1e-10).log()).sum(dim=-1)
            del p

        all_losses.append(losses_2d[shift_mask].float().cpu())
        all_sm.append(sm_2d[shift_mask].float().cpu())
        all_ent.append(ent_2d[shift_mask].float().cpu())

        for l in layers:
            h = captured[l][:, :-1, :]  # stays bf16
            # norm computed on bf16 tensor, float for numerical stability, then cpu
            per_layer_norms[l].append(h.float().norm(dim=-1)[shift_mask].cpu())
            # activations stored as bf16 on CPU (half the memory of fp32)
            per_layer_acts[l].append(h[shift_mask].to(torch.bfloat16).cpu())

        total += shift_mask.sum().item()
        for l in layers:
            captured.pop(l, None)
        del outputs, input_ids, attn_mask, shift_logits, shift_labels
        del losses_2d, sm_2d, ent_2d, shift_mask
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        if (bi + 1) % 20 == 0:
            print(f"    batch {bi + 1}/{len(batches)}, {total} positions [{elapsed_str()}]")

    for h in handles:
        h.remove()

    n = min(total, max_tokens)
    losses_cat = torch.cat(all_losses).numpy()[:n]
    sm_cat = torch.cat(all_sm).numpy()[:n]
    ent_cat = torch.cat(all_ent).numpy()[:n]

    results = {}
    for l in layers:
        results[l] = {
            "activations": torch.cat(per_layer_acts[l])[:n],  # bf16 CPU tensor
            "losses": losses_cat,
            "max_softmax": sm_cat,
            "logit_entropy": ent_cat,
            "activation_norm": torch.cat(per_layer_norms[l])[:n].numpy(),
        }
    print(f"    collected {n} positions × {len(layers)} layers [{elapsed_str()}]")
    return results


def _get_layer_list(model):
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("expected GPT-NeoX architecture for Pythia")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"checkpoint not found at {CHECKPOINT_PATH}")
    ckpt = json.load(open(CHECKPOINT_PATH))
    PEAK = int(ckpt["peak_layer_final"])
    N_LAYERS = int(ckpt["n_layers"])
    HIDDEN = int(ckpt["hidden_dim"])
    OUT_LAYER = N_LAYERS - 1
    MAX_TRAIN = EX_DIM * HIDDEN
    print(f"Checkpoint peak=L{PEAK} n_layers={N_LAYERS} hidden={HIDDEN}")
    print(f"Will collect L{PEAK} (peak) and L{OUT_LAYER} (output). MAX_TRAIN={MAX_TRAIN}")

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
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded: {n_params:.2f}B params [{elapsed_str()}]")

    # Tokenize
    print(f"\n=== Pre-tokenizing [{elapsed_str()}] ===")
    wiki_train_docs = load_wikitext("train", max_docs=12000)
    wiki_val_docs = load_wikitext("validation")
    wiki_test_docs = load_wikitext("test")
    print(f"Docs: {len(wiki_train_docs)} train, {len(wiki_val_docs)} val, {len(wiki_test_docs)} test")
    wiki_train_enc = pretokenize(wiki_train_docs, tokenizer)
    wiki_val_enc = pretokenize(wiki_val_docs, tokenizer)
    wiki_test_enc = pretokenize(wiki_test_docs, tokenizer)
    train_batches = build_batches(wiki_train_enc, BATCH_SIZE)
    val_batches = build_batches(wiki_val_enc, BATCH_SIZE)
    test_batches = build_batches(wiki_test_enc, BATCH_SIZE)
    del wiki_train_docs, wiki_val_docs, wiki_test_docs, wiki_train_enc, wiki_val_enc, wiki_test_enc
    gc.collect()

    # Collect peak + output on train, val, test
    layers = [PEAK, OUT_LAYER]
    print(f"\n=== Collect train (L{PEAK}, L{OUT_LAYER}) [{elapsed_str()}] ===")
    tr = collect_two_layers_bf16(model, layer_modules, train_batches, layers, MAX_TRAIN)
    print(f"\n=== Collect val (L{PEAK}, L{OUT_LAYER}) [{elapsed_str()}] ===")
    va = collect_two_layers_bf16(model, layer_modules, val_batches, layers, MAX_TRAIN)
    print(f"\n=== Collect test (L{PEAK}) [{elapsed_str()}] ===")
    te = collect_two_layers_bf16(model, layer_modules, test_batches, [PEAK], MAX_TRAIN)

    wiki_train_peak = tr[PEAK]
    wiki_val_peak = va[PEAK]
    wiki_test_peak = te[PEAK]
    wiki_train_output = tr[OUT_LAYER]
    wiki_val_output = va[OUT_LAYER]

    # Free wiki batch lists once collection is done (they held tokenized data)
    del tr, va, te, train_batches, val_batches, test_batches
    gc.collect()

    # C4 collection at peak layer only (matches run_model.py protocol)
    # Loaded streaming; matches run_model.py token budgets (MAX_TRAIN // 2 for test,
    # MAX_TRAIN for train)
    print(f"\n=== Loading C4 [{elapsed_str()}] ===")
    from datasets import load_dataset

    c4_test_docs = []
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for i, row in enumerate(ds):
        if i < 50000:
            continue
        text = row["text"].strip()
        if len(text) > 100:
            c4_test_docs.append(text)
        if len(c4_test_docs) >= 500:
            break
    c4_test_enc = pretokenize(c4_test_docs, tokenizer)
    c4_test_batches = build_batches(c4_test_enc, BATCH_SIZE)
    del c4_test_docs, c4_test_enc

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
    del c4_train_docs, c4_train_enc

    print(f"\n=== Collect C4 test (L{PEAK}) [{elapsed_str()}] ===")
    c4_te = collect_two_layers_bf16(model, layer_modules, c4_test_batches, [PEAK], MAX_TRAIN // 2)
    c4_test_peak = c4_te[PEAK]
    del c4_te, c4_test_batches
    gc.collect()

    print(f"\n=== Collect C4 train (L{PEAK}) [{elapsed_str()}] ===")
    c4_tr = collect_two_layers_bf16(model, layer_modules, c4_train_batches, [PEAK], MAX_TRAIN)
    c4_train_peak = c4_tr[PEAK]
    del c4_tr, c4_train_batches
    gc.collect()

    # Unload model: remaining work is all probe-level
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"Model unloaded [{elapsed_str()}]")

    # Test-split comparison (3 seeds, matches run_model.py)
    print(f"\n=== Test-split comparison [{elapsed_str()}] ===")
    test_rhos = []
    for s in EVAL_SEEDS[:3]:
        head = train_linear_binary(wiki_train_peak, seed=s)
        _, rho, _ = evaluate_head(head, wiki_test_peak)
        test_rhos.append(float(rho))
        print(f"  seed {s}: {rho:+.4f}")
    test_mean = float(np.mean(test_rhos))
    print(f"  mean: {test_mean:+.4f}")

    # Baselines on val_peak
    print(f"\n=== Baselines on val [{elapsed_str()}] ===")
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
    rh = torch.nn.Linear(HIDDEN, 1)
    rh.eval()
    with torch.inference_mode():
        val_acts_fp32 = wiki_val_peak["activations"].to(dtype=torch.float32)
        baseline_results["random_head"] = float(
            partial_spearman(
                rh(val_acts_fp32).squeeze(-1).numpy(),
                wiki_val_peak["losses"],
                [wiki_val_peak["max_softmax"], wiki_val_peak["activation_norm"]],
            )[0]
        )
        del val_acts_fp32
    for n, v in baseline_results.items():
        print(f"  {n}: {v:+.4f}")

    # Output-controlled (3 seeds)
    print(f"\n=== Output-controlled [{elapsed_str()}] ===")
    oc_train_acts = wiki_train_output["activations"].to(DEVICE, dtype=torch.float32)
    oc_train_losses = torch.from_numpy(wiki_train_output["losses"]).float().to(DEVICE)
    oc_tds = torch.utils.data.TensorDataset(oc_train_acts, oc_train_losses)
    oc_tdl = torch.utils.data.DataLoader(oc_tds, batch_size=1024, shuffle=True)
    ctrl_rhos = []
    for seed in EVAL_SEEDS[:3]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        pred = torch.nn.Sequential(
            torch.nn.Linear(oc_train_acts.size(1), 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        ).to(DEVICE)
        opt = torch.optim.Adam(pred.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(20):
            for bx, by in oc_tdl:
                loss = F.mse_loss(pred(bx).squeeze(-1), by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        pred.eval().cpu()
        with torch.inference_mode():
            val_out_fp32 = wiki_val_output["activations"].to(dtype=torch.float32)
            ps = pred(val_out_fp32).squeeze(-1).numpy()
            del val_out_fp32
        obs = train_linear_binary(wiki_train_peak, seed=seed)
        obs.eval()
        with torch.inference_mode():
            val_peak_fp32 = wiki_val_peak["activations"].to(dtype=torch.float32)
            os_ = obs(val_peak_fp32).squeeze(-1).numpy()
            del val_peak_fp32
        r, _ = partial_spearman(
            os_,
            wiki_val_peak["losses"],
            [wiki_val_peak["max_softmax"], wiki_val_peak["activation_norm"], ps],
        )
        ctrl_rhos.append(float(r))
        print(f"  seed {seed}: {r:+.4f}")
    del oc_train_acts, oc_train_losses, oc_tds
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Control sensitivity waterfall
    print(f"\n=== Control sensitivity [{elapsed_str()}] ===")
    torch.manual_seed(42)
    conf_feats = (
        torch.from_numpy(
            np.column_stack([wiki_train_peak["max_softmax"], wiki_train_peak["activation_norm"]])
        )
        .float()
        .to(DEVICE)
    )
    loss_tgt = torch.from_numpy(wiki_train_peak["losses"]).float().to(DEVICE)
    mlp_ctrl = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(DEVICE)
    opt = torch.optim.Adam(mlp_ctrl.parameters(), lr=1e-3, weight_decay=1e-4)
    cs_ds = torch.utils.data.TensorDataset(conf_feats, loss_tgt)
    cs_dl = torch.utils.data.DataLoader(cs_ds, batch_size=1024, shuffle=True)
    for _ in range(20):
        for bx, by in cs_dl:
            loss = F.mse_loss(mlp_ctrl(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    mlp_ctrl.eval().cpu()
    del conf_feats, loss_tgt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
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
        val_peak_fp32 = wiki_val_peak["activations"].to(dtype=torch.float32)
        cs_obs = cs_head(val_peak_fp32).squeeze(-1).numpy()
        del val_peak_fp32

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
        print(f"  {name:<16}: {r:+.4f}")

    # Cross-domain: wikitext, c4 (WikiText-trained -> C4), c4_within (C4-trained -> C4)
    print(f"\n=== Cross-domain [{elapsed_str()}] ===")
    domain_results = {}
    rhos = []
    for s in EVAL_SEEDS[:3]:
        head = train_linear_binary(wiki_train_peak, seed=s)
        _, rho, _ = evaluate_head(head, wiki_val_peak)
        rhos.append(float(rho))
    domain_results["wikitext"] = float(np.mean(rhos))
    print(f"  wikitext: {domain_results['wikitext']:+.4f}")

    rhos = []
    for s in EVAL_SEEDS[:3]:
        head = train_linear_binary(wiki_train_peak, seed=s)
        _, rho, _ = evaluate_head(head, c4_test_peak)
        rhos.append(float(rho))
    domain_results["c4"] = float(np.mean(rhos))
    print(f"  c4:       {domain_results['c4']:+.4f}")

    rhos = []
    for s in EVAL_SEEDS[:3]:
        head = train_linear_binary(c4_train_peak, seed=s)
        _, rho, _ = evaluate_head(head, c4_test_peak)
        rhos.append(float(rho))
    domain_results["c4_within"] = float(np.mean(rhos))
    print(f"  c4_within: {domain_results['c4_within']:+.4f}")

    # Flagging
    print(f"\n=== Flagging [{elapsed_str()}] ===")
    nf = min(len(wiki_val_peak["losses"]), len(wiki_val_output["losses"]))
    fl = wiki_val_peak["losses"][:nf]
    fsm = wiki_val_output["max_softmax"][:nf]
    fa_bf16 = wiki_val_peak["activations"][:nf]
    ml = float(np.median(fl))
    ihl = fl > ml
    fr = [0.05, 0.10, 0.20, 0.30]
    conf_thresholds = {}
    fsm_sorted = np.sort(fsm)
    for rate in fr:
        k = int(nf * rate)
        conf_thresholds[rate] = fsm_sorted[k]
    del fsm_sorted

    fres = []
    for seed in EVAL_SEEDS[:3]:
        h = train_linear_binary(wiki_train_peak, seed=seed)
        h.eval()
        with torch.inference_mode():
            fa_fp32 = fa_bf16.to(dtype=torch.float32)
            osc = h(fa_fp32).squeeze(-1).numpy()
            del fa_fp32
        osc_sorted = np.sort(osc)
        sr = {"observer": {}, "confidence": {}, "exclusive": {}}
        for rate in fr:
            k = int(nf * rate)
            of = osc >= osc_sorted[-k]
            cf = fsm <= conf_thresholds[rate]
            sr["observer"][str(rate)] = float(ihl[of].mean()) if of.sum() > 0 else 0.0
            sr["confidence"][str(rate)] = float(ihl[cf].mean()) if cf.sum() > 0 else 0.0
            sr["exclusive"][str(rate)] = {"observer_only": int((of & ~cf & ihl).sum())}
        fres.append(sr)
    fs = {
        str(r): {
            "observer_exclusive": float(np.mean([s["exclusive"][str(r)]["observer_only"] for s in fres]))
        }
        for r in fr
    }
    print(f"  10%: {fs['0.1']['observer_exclusive']:.0f} tokens")

    # Merge with checkpoint and save
    print(f"\n=== Saving [{elapsed_str()}] ===")
    pc_mean = ckpt.get("partial_corr", {}).get("mean")
    pc_std = ckpt.get("partial_corr", {}).get("std")
    pc_per_seed = ckpt.get("partial_corr", {}).get("per_seed")
    seed_agree_mean = ckpt.get("seed_agreement", {}).get("mean")
    peak_frac = ckpt.get("peak_layer_frac")
    layer_profile = ckpt.get("layer_profile", {})
    multi_layer_eval = ckpt.get("multi_layer_eval", {})

    output = {
        "model": MODEL_ID,
        "n_params_b": round(n_params, 2),
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN,
        "provenance": {
            "model_revision": _model_revision,
            "script": "scripts/pythia_12b_backfill.py",
            "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
            "device": str(DEVICE),
            "torch_version": torch.__version__,
            "output_file": str(OUTPUT_PATH.name),
            "note": (
                "Phase 1-3 (layer_profile, multi_layer_eval, partial_corr, seed_agreement) "
                "reproduced from pythia_12b_checkpoint.json (run_model.py, batch_size=24, "
                "bf16 forward, use_cache=False). Phase 4-5 plus C4 cross-domain collected "
                "by this backfill script with bf16 CPU activation storage to fit pod memory."
            ),
        },
        "protocol": {
            "layer_select_seed": LAYER_SELECT_SEED,
            "eval_seeds": EVAL_SEEDS,
            "target_ex_per_dim": EX_DIM,
            "batch_size": BATCH_SIZE,
            "layers_per_pass": 1,
        },
        "peak_layer_final": PEAK,
        "peak_layer_frac": peak_frac,
        "layer_profile": layer_profile,
        "multi_layer_eval": multi_layer_eval,
        "partial_corr": {
            "mean": pc_mean,
            "std": pc_std,
            "per_seed": pc_per_seed,
            "n_seeds": len(EVAL_SEEDS),
            "split": "validation (held-out seeds)",
        },
        "test_split_comparison": {"mean": test_mean, "per_seed": test_rhos},
        "seed_agreement": {"mean": seed_agree_mean},
        "output_controlled": {"mean": float(np.mean(ctrl_rhos)), "per_seed": ctrl_rhos},
        "baselines": baseline_results,
        "cross_domain": domain_results,
        "control_sensitivity": ctrl_results,
        "flagging_6a": {"n_tokens": nf, "summary": fs},
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {OUTPUT_PATH}")
    print(f"FINAL: L{PEAK} = {pc_mean:+.4f} +/- {pc_std:.4f}")
    print(f"Test-split: {test_mean:+.4f}")
    print(f"Output-controlled: {np.mean(ctrl_rhos):+.4f}")
    print(f"Random head: {baseline_results.get('random_head'):+.4f}")
    print(f"Total time: {elapsed_str()}")


if __name__ == "__main__":
    main()
