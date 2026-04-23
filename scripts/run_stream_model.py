"""Sharded / streaming variant of run_model.py for large models.

Drop-in alternative to run_model.py that stores activations on disk in
per-split bf16 shards so the probe protocol fits on machines where the
default in-memory path does not. The output JSON schema and probe
protocol are identical to run_model.py.
"""

import argparse
import datetime as _dt
import gc
import json
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr

# ── Core numerics (bit-identical to run_model.py) ────────────────────


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


def _get_layer_list(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


# ── Shard I/O (meta.pt + acts_NNNN.pt split) ────────────────────────


def shard_dir_for(root, split, layer):
    d = root / f"{split}_L{layer}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def meta_path(shard_dir):
    return shard_dir / "meta.pt"


def iter_acts_paths(shard_dir):
    return sorted(glob(str(shard_dir / "acts_*.pt")))


def shard_is_complete(shard_dir):
    """A (split, layer) is complete iff meta.pt exists."""
    return meta_path(shard_dir).exists()


def save_meta(
    shard_dir, *, losses, max_softmax, logit_entropy, activation_norm, per_shard_tokens, hidden_dim
):
    meta = {
        "losses": torch.from_numpy(np.ascontiguousarray(losses)),
        "max_softmax": torch.from_numpy(np.ascontiguousarray(max_softmax)),
        "logit_entropy": torch.from_numpy(np.ascontiguousarray(logit_entropy)),
        "activation_norm": torch.from_numpy(np.ascontiguousarray(activation_norm)),
        "per_shard_tokens": torch.tensor(per_shard_tokens, dtype=torch.long),
        "hidden_dim": int(hidden_dim),
        "n_tokens": int(len(losses)),
    }
    torch.save(meta, meta_path(shard_dir))


def load_meta(shard_dir):
    m = torch.load(meta_path(shard_dir), map_location="cpu", weights_only=True)
    return {
        "losses": m["losses"].numpy(),
        "max_softmax": m["max_softmax"].numpy(),
        "logit_entropy": m["logit_entropy"].numpy(),
        "activation_norm": m["activation_norm"].numpy(),
        "per_shard_tokens": m["per_shard_tokens"].tolist(),
        "hidden_dim": int(m["hidden_dim"]),
        "n_tokens": int(m["n_tokens"]),
    }


# ── Streaming collection ─────────────────────────────────────────────


def collect_multi_layer_stream(
    model, batches, layers, max_tokens, device, shard_root, split, sm_chunk=8, skip_if_complete=True
):
    """Forward-pass collection with meta.pt + per-batch acts_*.pt shards.

    Writes:
      shard_root/<split>_L<layer>/meta.pt
      shard_root/<split>_L<layer>/acts_NNNN.pt
    """
    layers = list(layers)
    # If all requested layers are already complete, short-circuit.
    if skip_if_complete and all(shard_is_complete(shard_dir_for(shard_root, split, l)) for l in layers):
        print(f"    [resume] {split} layers {layers} already complete, skipping collection")
        return

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

    # Fresh start: wipe any stale shard files for these layers
    shard_dirs = {l: shard_dir_for(shard_root, split, l) for l in layers}
    for d in shard_dirs.values():
        for p in glob(str(d / "acts_*.pt")):
            Path(p).unlink()
        mp = meta_path(d)
        if mp.exists():
            mp.unlink()

    per_layer_losses = {l: [] for l in layers}
    per_layer_sm = {l: [] for l in layers}
    per_layer_ent = {l: [] for l in layers}
    per_layer_norms = {l: [] for l in layers}
    per_layer_tokens = {l: [] for l in layers}

    total = 0
    batch_idx = 0
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

        losses_2d = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none").view(
            B, S - 1
        )

        sm_2d = torch.empty(B, S - 1, device=shift_logits.device)
        ent_2d = torch.empty(B, S - 1, device=shift_logits.device)
        for ci in range(0, B, sm_chunk):
            p = shift_logits[ci : ci + sm_chunk].float().softmax(dim=-1)
            sm_2d[ci : ci + sm_chunk] = p.max(dim=-1).values
            ent_2d[ci : ci + sm_chunk] = -(p * (p + 1e-10).log()).sum(dim=-1)
            del p

        batch_losses = losses_2d[shift_mask].float().cpu().numpy()
        batch_sm = sm_2d[shift_mask].float().cpu().numpy()
        batch_ent = ent_2d[shift_mask].float().cpu().numpy()

        for l in layers:
            h = captured[l][:, :-1, :]  # bf16
            acts = h[shift_mask].to(torch.bfloat16).contiguous().cpu()
            norms = h.float().norm(dim=-1)[shift_mask].cpu().numpy()
            # activations only per batch; scalars accumulated for meta.pt
            torch.save({"activations": acts}, shard_dirs[l] / f"acts_{batch_idx:06d}.pt")
            per_layer_losses[l].append(batch_losses)
            per_layer_sm[l].append(batch_sm)
            per_layer_ent[l].append(batch_ent)
            per_layer_norms[l].append(norms)
            per_layer_tokens[l].append(int(len(batch_losses)))
            del acts

        total += int(shift_mask.sum().item())
        batch_idx += 1
        for l in layers:
            captured.pop(l, None)
        del outputs, input_ids, attn_mask, shift_logits, shift_labels, losses_2d, sm_2d, ent_2d, shift_mask
        if device == "cuda":
            torch.cuda.empty_cache()

        if batch_idx % 10 == 0:
            print(f"      batch {batch_idx}/{len(batches)}, {total} positions")

    for h in handles:
        h.remove()

    n = min(total, max_tokens)
    # Write meta.pt per layer with concatenated scalars, truncated to n
    for l in layers:
        losses = np.concatenate(per_layer_losses[l])[:n]
        sm = np.concatenate(per_layer_sm[l])[:n]
        ent = np.concatenate(per_layer_ent[l])[:n]
        norms = np.concatenate(per_layer_norms[l])[:n]
        # per_shard_tokens should sum to len(losses); trim last shard record if truncation fell mid-shard
        tokens = list(per_layer_tokens[l])
        running = 0
        trimmed = []
        for t in tokens:
            if running + t <= n:
                trimmed.append(t)
                running += t
            else:
                if n - running > 0:
                    trimmed.append(n - running)
                    running = n
                break
        # Determine hidden dim from the first acts shard
        first_acts_path = iter_acts_paths(shard_dirs[l])[0]
        first_acts = torch.load(first_acts_path, map_location="cpu", weights_only=True)["activations"]
        hidden = first_acts.shape[1]
        del first_acts
        save_meta(
            shard_dirs[l],
            losses=losses,
            max_softmax=sm,
            logit_entropy=ent,
            activation_norm=norms,
            per_shard_tokens=trimmed,
            hidden_dim=hidden,
        )

    print(f"    {n} positions × {len(layers)} layers → {shard_root.name}")


# ── Probe training / eval from shards (uses meta.pt) ────────────────


def train_linear_binary_streaming(shard_dir, seed, epochs=20, lr=1e-3, mb_size=4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = load_meta(shard_dir)
    residuals = compute_loss_residuals(meta["losses"], meta["max_softmax"], meta["activation_norm"])
    all_targets = (residuals > 0).astype(np.float32)
    hidden = meta["hidden_dim"]
    per_shard = meta["per_shard_tokens"]
    n_total = meta["n_tokens"]
    assert sum(per_shard) == n_total, f"per_shard sum {sum(per_shard)} != n_tokens {n_total}"

    shard_paths = iter_acts_paths(shard_dir)
    assert len(shard_paths) == len(per_shard), f"{len(shard_paths)} shards vs {len(per_shard)} token records"

    offsets = [0]
    for t in per_shard:
        offsets.append(offsets[-1] + t)

    torch.manual_seed(seed)
    np.random.seed(seed)
    head = torch.nn.Linear(hidden, 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    head.train()
    for _epoch in range(epochs):
        order = np.random.permutation(len(shard_paths))
        for si in order:
            sp = shard_paths[si]
            s = torch.load(sp, map_location="cpu", weights_only=True)
            acts = s["activations"].to(device, dtype=torch.float32)
            start, end = offsets[si], offsets[si + 1]
            # Defensive: trim acts to match expected token count (handles last-shard truncation)
            expected = end - start
            if acts.size(0) != expected:
                acts = acts[:expected]
            tgt = torch.from_numpy(all_targets[start:end]).to(device)
            n = acts.size(0)
            perm = torch.randperm(n, device=device)
            for i in range(0, n, mb_size):
                idx = perm[i : i + mb_size]
                bx = acts[idx]
                by = tgt[idx]
                loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            del s, acts, tgt, perm
            if device == "cuda":
                torch.cuda.empty_cache()

    return head.cpu()


def _score_over_shards(head, shard_dir):
    head.eval()
    meta = load_meta(shard_dir)
    per_shard = meta["per_shard_tokens"]
    n_total = meta["n_tokens"]
    shard_paths = iter_acts_paths(shard_dir)
    scores = np.empty(n_total, dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for si, sp in enumerate(shard_paths):
            s = torch.load(sp, map_location="cpu", weights_only=True)
            acts = s["activations"].to(dtype=torch.float32)
            expected = per_shard[si]
            if acts.size(0) != expected:
                acts = acts[:expected]
            scores[offset : offset + expected] = head(acts).squeeze(-1).numpy()
            offset += expected
            del s, acts
    assert offset == n_total
    return scores, meta


def evaluate_head_streaming(head, shard_dir):
    scores, meta = _score_over_shards(head, shard_dir)
    rho, p = partial_spearman(scores, meta["losses"], [meta["max_softmax"], meta["activation_norm"]])
    return scores, float(rho), float(p)


def random_head_rho_streaming(shard_dir, hidden, seed=99):
    torch.manual_seed(seed)
    rh = torch.nn.Linear(hidden, 1)
    scores, meta = _score_over_shards(rh, shard_dir)
    rho, _ = partial_spearman(scores, meta["losses"], [meta["max_softmax"], meta["activation_norm"]])
    return float(rho)


def compute_hand_designed_streaming(shard_dir):
    meta = load_meta(shard_dir)
    ff_parts, ar_parts, ae_parts = [], [], []
    for sp in iter_acts_paths(shard_dir):
        s = torch.load(sp, map_location="cpu", weights_only=True)
        acts = s["activations"].to(dtype=torch.float32)
        p = acts.abs() / (acts.abs().sum(dim=1, keepdim=True) + 1e-8)
        ff_parts.append((acts**2).mean(dim=1).numpy())
        ar_parts.append((acts.abs() > 0.01).float().mean(dim=1).numpy())
        ae_parts.append(-(p * (p + 1e-8).log()).sum(dim=1).numpy())
        del s, acts, p
    ff = np.concatenate(ff_parts)[: meta["n_tokens"]]
    ar = np.concatenate(ar_parts)[: meta["n_tokens"]]
    ae = np.concatenate(ae_parts)[: meta["n_tokens"]]
    out = {}
    for name, s_arr in [("ff_goodness", ff), ("active_ratio", ar), ("act_entropy", ae)]:
        out[name], _ = partial_spearman(s_arr, meta["losses"], [meta["max_softmax"], meta["activation_norm"]])
    out["activation_norm"], _ = partial_spearman(
        meta["activation_norm"], meta["losses"], [meta["max_softmax"], meta["activation_norm"]]
    )
    return out, meta


def observer_scores_for_shard_dir(head, shard_dir):
    scores, _ = _score_over_shards(head, shard_dir)
    return scores


# ── GPU-resident full-battery helpers ───────────────────────────────


def load_peak_to_gpu(shard_dir, device):
    """Load all shards of one (split, layer) into one contiguous fp32 GPU tensor."""
    meta = load_meta(shard_dir)
    hidden = meta["hidden_dim"]
    n = meta["n_tokens"]
    acts = torch.empty(n, hidden, dtype=torch.float32, device=device)
    offset = 0
    for sp, tok in zip(iter_acts_paths(shard_dir), meta["per_shard_tokens"]):
        s = torch.load(sp, map_location="cpu", weights_only=True)
        chunk = s["activations"][:tok].to(device, dtype=torch.float32)
        acts[offset : offset + tok] = chunk
        offset += tok
        del s, chunk
    assert offset == n
    return acts, meta


def train_linear_binary_gpu(
    acts_fp32, losses, max_softmax, activation_norm, seed, epochs=20, lr=1e-3, mb_size=4096
):
    """Train a linear binary probe on a pre-loaded fp32 GPU tensor."""
    device = acts_fp32.device
    residuals = compute_loss_residuals(losses, max_softmax, activation_norm)
    targets = torch.from_numpy((residuals > 0).astype(np.float32)).to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = torch.nn.Linear(acts_fp32.size(1), 1).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts_fp32, targets)
    dl = torch.utils.data.DataLoader(ds, batch_size=mb_size, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.cpu()


def evaluate_head_gpu(head, acts_fp32, losses, max_softmax, activation_norm):
    head.eval()
    with torch.inference_mode():
        scores = head(acts_fp32.cpu()).squeeze(-1).numpy()
    rho, _ = partial_spearman(scores, losses, [max_softmax, activation_norm])
    return scores, float(rho)


# ── CLI ──────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser(description="Streaming observability protocol for large models.")
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--ex-dim", type=int, default=350)
parser.add_argument("--batch-size", type=int, default=16, help="default 16 for large models")
parser.add_argument("--layers-per-pass", type=int, default=1)
parser.add_argument("--trust-remote-code", action="store_true")
parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
parser.add_argument("--seeds", type=int, default=7)
parser.add_argument("--skip-c4", action="store_true")
parser.add_argument("--max-docs", type=int, default=None)
parser.add_argument("--shard-root", default="/workspace/shards")
parser.add_argument("--keep-shards", action="store_true")
parser.add_argument("--peak-layer", type=int, default=None, help="skip Phase 1; use this layer as peak")
parser.add_argument("--candidates-per-pass", type=int, default=None)
parser.add_argument(
    "--device-map",
    default="single",
    choices=["single", "auto"],
    help="single: .to('cuda') on one GPU. auto: device_map='auto' via Accelerate (multi-GPU / CPU-offload).",
)
parser.add_argument(
    "--max-gpu-mem",
    default=None,
    help="Per-GPU memory cap when device-map=auto (e.g. '130GiB'). Applied to every CUDA device found.",
)
parser.add_argument(
    "--cpu-offload-mem",
    default=None,
    help="Optional CPU-offload pool when device-map=auto (e.g. '300GiB'). Omit for no offload.",
)
parser.add_argument("--force-restart", action="store_true", help="ignore checkpoint, start from scratch")
parser.add_argument(
    "--gpu-resident-phase5",
    action="store_true",
    default=True,
    help="load peak-layer activations to GPU after model unload (default true)",
)
parser.add_argument("--no-gpu-resident-phase5", dest="gpu_resident_phase5", action="store_false")
args = parser.parse_args()


# ── Setup ────────────────────────────────────────────────────────────


if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DEVICE = DEVICE
SM_CHUNK = 8
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    n_gpus = torch.cuda.device_count()
    print(f"CUDA devices: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {props.name} {props.total_memory / 1e9:.1f} GB")

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


MODEL_ID = args.model
BATCH_SIZE = args.batch_size
LAYERS_PER_PASS = args.layers_per_pass
TARGET_EX_PER_DIM = args.ex_dim
model_slug = args.output.replace("_results.json", "").replace(".json", "")
SHARD_ROOT = Path(args.shard_root) / model_slug
SHARD_ROOT.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = Path(f"/workspace/{model_slug}_checkpoint.json")
print(f"Shard root: {SHARD_ROOT}")
print(f"Checkpoint: {CHECKPOINT_PATH}")


def save_checkpoint(phase, **data):
    ckpt = {"_checkpoint_phase": phase, "_elapsed": elapsed_str()}
    ckpt.update(data)
    try:
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(ckpt, f, indent=2)
        print(f"  [checkpoint saved: phase {phase}]")
    except OSError:
        pass


# Load prior checkpoint if resume is possible
prior_ckpt = None
if CHECKPOINT_PATH.exists() and not args.force_restart:
    try:
        prior_ckpt = json.load(open(CHECKPOINT_PATH))
        print(f"[resume] found checkpoint at phase {prior_ckpt.get('_checkpoint_phase')}")
    except (json.JSONDecodeError, OSError):
        prior_ckpt = None


# ── Model loading ────────────────────────────────────────────────────


from transformers import AutoModelForCausalLM, AutoTokenizer

load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": args.attn_impl}
if args.trust_remote_code:
    load_kwargs["trust_remote_code"] = True

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=args.trust_remote_code)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_max_memory():
    if args.max_gpu_mem is None and args.cpu_offload_mem is None:
        return None
    mm = {}
    if args.max_gpu_mem is not None:
        for i in range(torch.cuda.device_count()):
            mm[i] = args.max_gpu_mem
    if args.cpu_offload_mem is not None:
        mm["cpu"] = args.cpu_offload_mem
    return mm


if args.device_map == "auto":
    load_kwargs["device_map"] = "auto"
    load_kwargs["low_cpu_mem_usage"] = True
    mm = build_max_memory()
    if mm is not None:
        load_kwargs["max_memory"] = mm
        print(f"Loading with device_map=auto, max_memory={mm}")
    else:
        print("Loading with device_map=auto, no max_memory cap")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
else:
    print("Loading with .to('cuda') (single-GPU)")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs).to(DEVICE)
model.eval()

_model_revision = getattr(model.config, "_commit_hash", None) or "unknown"
_cfg = getattr(model.config, "text_config", model.config)
N_LAYERS = _cfg.num_hidden_layers
HIDDEN_DIM = _cfg.hidden_size
n_params = sum(p.numel() for p in model.parameters()) / 1e9
MAX_TRAIN = TARGET_EX_PER_DIM * HIDDEN_DIM
LAYER_SELECT_SEED = 42
EVAL_SEEDS = list(range(43, 43 + args.seeds))

print(f"{n_params:.2f}B, {N_LAYERS} layers, {HIDDEN_DIM} dim")
print(f"Token budget: {MAX_TRAIN} train ({TARGET_EX_PER_DIM} ex/dim), batch={BATCH_SIZE}")


# ── Tokenize ─────────────────────────────────────────────────────────


print(f"\n=== Pre-tokenizing [{elapsed_str()}] ===")
_train_max = args.max_docs or 12000
_other_max = args.max_docs
wiki_train_docs = load_wikitext("train", max_docs=_train_max)
wiki_val_docs = load_wikitext("validation", max_docs=_other_max)
wiki_test_docs = load_wikitext("test", max_docs=_other_max)
wiki_train_enc = pretokenize(wiki_train_docs, tokenizer)
wiki_val_enc = pretokenize(wiki_val_docs, tokenizer)
wiki_test_enc = pretokenize(wiki_test_docs, tokenizer)
train_batches = build_batches(wiki_train_enc, BATCH_SIZE)
val_batches = build_batches(wiki_val_enc, BATCH_SIZE)
test_batches = build_batches(wiki_test_enc, BATCH_SIZE)
print(f"Batches: {len(train_batches)} train, {len(val_batches)} val, {len(test_batches)} test")
del wiki_train_docs, wiki_val_docs, wiki_test_docs, wiki_train_enc, wiki_val_enc, wiki_test_enc
gc.collect()


# ── Phase 1: layer sweep ────────────────────────────────────────────


layer_profile = {}
peak_layer = None

if args.peak_layer is not None:
    print(f"\n=== Phase 1: skipped (peak layer {args.peak_layer} specified) [{elapsed_str()}] ===")
    peak_layer = args.peak_layer
elif prior_ckpt and prior_ckpt.get("_checkpoint_phase") in ("1_sweep", "3_multiseed"):
    print(f"\n=== Phase 1: skipped (resume from checkpoint) [{elapsed_str()}] ===")
    layer_profile = {int(k): v for k, v in prior_ckpt.get("layer_profile", {}).items()}
    peak_layer = int(prior_ckpt.get("peak_layer", prior_ckpt.get("peak_layer_final")))
else:
    print(f"\n=== Phase 1: sweeping {N_LAYERS} layers, {LAYERS_PER_PASS}/pass [{elapsed_str()}] ===")
    layer_chunks = [
        list(range(i, min(i + LAYERS_PER_PASS, N_LAYERS))) for i in range(0, N_LAYERS, LAYERS_PER_PASS)
    ]
    for chunk in layer_chunks:
        t0 = time.time()
        print(f"  Collecting L{chunk[0]}-L{chunk[-1]} train...")
        collect_multi_layer_stream(model, train_batches, chunk, MAX_TRAIN, DEVICE, SHARD_ROOT, "train")
        print(f"  Collecting L{chunk[0]}-L{chunk[-1]} val...")
        collect_multi_layer_stream(model, val_batches, chunk, MAX_TRAIN, DEVICE, SHARD_ROOT, "val")
        for layer in chunk:
            head = train_linear_binary_streaming(
                shard_dir_for(SHARD_ROOT, "train", layer), seed=LAYER_SELECT_SEED
            )
            _, rho, _ = evaluate_head_streaming(head, shard_dir_for(SHARD_ROOT, "val", layer))
            layer_profile[layer] = float(rho)
            print(f"    L{layer:>2}: {rho:+.4f}")
        # Delete Phase 1 chunk shards; Phase 2 will re-collect only candidates
        for layer in chunk:
            shutil.rmtree(shard_dir_for(SHARD_ROOT, "train", layer), ignore_errors=True)
            shutil.rmtree(shard_dir_for(SHARD_ROOT, "val", layer), ignore_errors=True)
        elapsed = time.time() - t0
        done = chunk[-1] + 1
        remaining = (N_LAYERS - done) / len(chunk) * elapsed
        print(f"  chunk done in {elapsed:.0f}s, ~{remaining / 60:.0f}m remaining [{elapsed_str()}]")

    peak_layer = max(layer_profile, key=layer_profile.get)
    output_layer_guess = N_LAYERS - 1
    if peak_layer >= output_layer_guess - 1:
        mid = {l: r for l, r in layer_profile.items() if l <= int(0.8 * N_LAYERS)}
        if mid:
            peak_layer = max(mid, key=mid.get)

output_layer = N_LAYERS - 1
if layer_profile:
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
else:
    candidates = [peak_layer]
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


# ── Phase 2: candidates + output ────────────────────────────────────


print(f"\n=== Phase 2: collecting candidates + output [{elapsed_str()}] ===")
cpp = args.candidates_per_pass or LAYERS_PER_PASS
for i in range(0, len(candidates), cpp):
    cand_chunk = candidates[i : i + cpp]
    print(f"  Candidates L{cand_chunk[0]}-L{cand_chunk[-1]} train...")
    collect_multi_layer_stream(model, train_batches, cand_chunk, MAX_TRAIN, DEVICE, SHARD_ROOT, "train")
    print(f"  Candidates L{cand_chunk[0]}-L{cand_chunk[-1]} val...")
    collect_multi_layer_stream(model, val_batches, cand_chunk, MAX_TRAIN, DEVICE, SHARD_ROOT, "val")

print(f"  Output layer L{output_layer} train...")
collect_multi_layer_stream(model, train_batches, [output_layer], MAX_TRAIN, DEVICE, SHARD_ROOT, "train")
print(f"  Output layer L{output_layer} val...")
collect_multi_layer_stream(model, val_batches, [output_layer], MAX_TRAIN, DEVICE, SHARD_ROOT, "val")


# ── Phase 3: multi-seed eval ────────────────────────────────────────


if prior_ckpt and prior_ckpt.get("_checkpoint_phase") == "3_multiseed":
    print(f"\n=== Phase 3: skipped (resume from checkpoint) [{elapsed_str()}] ===")
    layer_eval = {int(k): v for k, v in prior_ckpt.get("multi_layer_eval", {}).items()}
    FINAL = int(prior_ckpt.get("peak_layer_final"))
    ev = layer_eval[FINAL]
else:
    print(f"\n=== Phase 3: multi-seed eval [{elapsed_str()}] ===")
    layer_eval = {}
    for layer in candidates:
        seed_rhos, seed_scores = [], []
        for seed in EVAL_SEEDS:
            head = train_linear_binary_streaming(shard_dir_for(SHARD_ROOT, "train", layer), seed=seed)
            scores, rho, _ = evaluate_head_streaming(head, shard_dir_for(SHARD_ROOT, "val", layer))
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
    print(f"FINAL: L{FINAL} = {ev['mean']:+.4f} +/- {ev['std']:.4f}")

    # Drop non-peak candidate shards
    for layer in candidates:
        if layer != FINAL:
            shutil.rmtree(shard_dir_for(SHARD_ROOT, "train", layer), ignore_errors=True)
            shutil.rmtree(shard_dir_for(SHARD_ROOT, "val", layer), ignore_errors=True)

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


# ── Phase 4: test + C4 ──────────────────────────────────────────────


print(f"\n=== Phase 4: test + C4 at FINAL [{elapsed_str()}] ===")
collect_multi_layer_stream(model, test_batches, [FINAL], MAX_TRAIN, DEVICE, SHARD_ROOT, "test")

if not args.skip_c4:
    print("  Loading C4...")
    from datasets import load_dataset

    if not shard_is_complete(shard_dir_for(SHARD_ROOT, "c4_test", FINAL)):
        c4_docs_test = []
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for i, row in enumerate(ds):
            if i < 50000:
                continue
            text = row["text"].strip()
            if len(text) > 100:
                c4_docs_test.append(text)
            if len(c4_docs_test) >= 500:
                break
        c4_test_enc = pretokenize(c4_docs_test, tokenizer)
        c4_test_batches = build_batches(c4_test_enc, BATCH_SIZE)
        collect_multi_layer_stream(
            model, c4_test_batches, [FINAL], MAX_TRAIN // 2, DEVICE, SHARD_ROOT, "c4_test"
        )

    if not shard_is_complete(shard_dir_for(SHARD_ROOT, "c4_train", FINAL)):
        c4_docs_train = []
        ds2 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        for row in ds2:
            text = row["text"].strip()
            if len(text) > 100:
                c4_docs_train.append(text)
            if len(c4_docs_train) >= 8000:
                break
        c4_train_enc = pretokenize(c4_docs_train, tokenizer)
        c4_train_batches = build_batches(c4_train_enc, BATCH_SIZE)
        collect_multi_layer_stream(
            model, c4_train_batches, [FINAL], MAX_TRAIN, DEVICE, SHARD_ROOT, "c4_train"
        )
else:
    print("  Skipping C4 (--skip-c4)")

# Unload model
del model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print(f"Model unloaded. [{elapsed_str()}]")


# ── Phase 5: full battery ───────────────────────────────────────────


print(f"\n=== Phase 5: full battery [{elapsed_str()}] ===")

train_peak_dir = shard_dir_for(SHARD_ROOT, "train", FINAL)
val_peak_dir = shard_dir_for(SHARD_ROOT, "val", FINAL)
test_peak_dir = shard_dir_for(SHARD_ROOT, "test", FINAL)
train_output_dir = shard_dir_for(SHARD_ROOT, "train", output_layer)
val_output_dir = shard_dir_for(SHARD_ROOT, "val", output_layer)

if args.gpu_resident_phase5 and DEVICE == "cuda":
    print("  Loading peak-layer activations to GPU (fp32)...")
    train_peak_acts, train_peak_meta = load_peak_to_gpu(train_peak_dir, device="cuda:0")
    val_peak_acts, val_peak_meta = load_peak_to_gpu(val_peak_dir, device="cuda:0")
    print(
        f"    train acts: {train_peak_acts.shape} ({train_peak_acts.numel() * 4 / 1e9:.1f} GB), "
        f"val acts: {val_peak_acts.shape}"
    )
else:
    train_peak_acts = None
    val_peak_acts = None


def _train_peak(seed):
    if train_peak_acts is not None:
        return train_linear_binary_gpu(
            train_peak_acts,
            losses=train_peak_meta["losses"],
            max_softmax=train_peak_meta["max_softmax"],
            activation_norm=train_peak_meta["activation_norm"],
            seed=seed,
        )
    return train_linear_binary_streaming(train_peak_dir, seed=seed)


def _eval_on_val_peak(head):
    if val_peak_acts is not None:
        return evaluate_head_gpu(
            head,
            val_peak_acts,
            losses=val_peak_meta["losses"],
            max_softmax=val_peak_meta["max_softmax"],
            activation_norm=val_peak_meta["activation_norm"],
        )[1]
    _, rho, _ = evaluate_head_streaming(head, val_peak_dir)
    return rho


# Test-split comparison
print("\n  Test-split:")
test_rhos = []
for s in EVAL_SEEDS[:3]:
    head = _train_peak(seed=s)
    _, rho, _ = evaluate_head_streaming(head, test_peak_dir)
    test_rhos.append(float(rho))
    print(f"    seed {s}: {rho:+.4f}")
print(f"    mean: {np.mean(test_rhos):+.4f}")


# Baselines on val
print("\n  Baselines:")
baseline_results, val_meta = compute_hand_designed_streaming(val_peak_dir)
baseline_results["random_head"] = random_head_rho_streaming(val_peak_dir, HIDDEN_DIM)
for n, v in baseline_results.items():
    print(f"    {n}: {v:+.4f}")
val_losses = val_meta["losses"]
val_sm = val_meta["max_softmax"]
val_norms = val_meta["activation_norm"]
val_ent = val_meta["logit_entropy"]


# Output-controlled (3 seeds)
print("\n  Output-controlled:")
OC_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_output_mlp_streaming(seed, epochs=20, lr=1e-3, mb_size=1024):
    torch.manual_seed(seed)
    np.random.seed(seed)
    meta_out = load_meta(train_output_dir)
    hidden = meta_out["hidden_dim"]
    pred = torch.nn.Sequential(torch.nn.Linear(hidden, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(
        OC_DEVICE
    )
    opt = torch.optim.Adam(pred.parameters(), lr=lr, weight_decay=1e-4)
    per_shard = meta_out["per_shard_tokens"]
    offsets = [0]
    for t in per_shard:
        offsets.append(offsets[-1] + t)
    shard_paths = iter_acts_paths(train_output_dir)
    for _epoch in range(epochs):
        for si, sp in enumerate(shard_paths):
            s = torch.load(sp, map_location="cpu", weights_only=True)
            acts = s["activations"].to(OC_DEVICE, dtype=torch.float32)
            expected = offsets[si + 1] - offsets[si]
            if acts.size(0) != expected:
                acts = acts[:expected]
            tgt = torch.from_numpy(meta_out["losses"][offsets[si] : offsets[si + 1]]).float().to(OC_DEVICE)
            n = acts.size(0)
            perm = torch.randperm(n, device=OC_DEVICE)
            for i in range(0, n, mb_size):
                idx = perm[i : i + mb_size]
                bx = acts[idx]
                by = tgt[idx]
                loss = F.mse_loss(pred(bx).squeeze(-1), by)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            del s, acts, tgt
            if OC_DEVICE == "cuda":
                torch.cuda.empty_cache()
    return pred.cpu()


def predict_output_losses(pred, shard_dir):
    """Apply output-layer MLP to every token in shard_dir. Returns np.ndarray [n_tokens]."""
    pred.eval()
    meta = load_meta(shard_dir)
    out = np.empty(meta["n_tokens"], dtype=np.float32)
    offset = 0
    with torch.inference_mode():
        for si, sp in enumerate(iter_acts_paths(shard_dir)):
            s = torch.load(sp, map_location="cpu", weights_only=True)
            acts = s["activations"].to(dtype=torch.float32)
            expected = meta["per_shard_tokens"][si]
            if acts.size(0) != expected:
                acts = acts[:expected]
            out[offset : offset + expected] = pred(acts).squeeze(-1).numpy()
            offset += expected
            del s, acts
    return out


ctrl_rhos = []
for seed in EVAL_SEEDS[:3]:
    pred = train_output_mlp_streaming(seed)
    ps = predict_output_losses(pred, val_output_dir)
    obs = _train_peak(seed=seed)
    os_scores = observer_scores_for_shard_dir(obs, val_peak_dir)
    r, _ = partial_spearman(os_scores, val_losses, [val_sm, val_norms, ps])
    ctrl_rhos.append(float(r))
    print(f"    seed {seed}: {r:+.4f}")


# Control sensitivity waterfall (one probe seed, trained with confidence MLP)
print("\n  Control sensitivity:")
train_meta = load_meta(train_peak_dir)

torch.manual_seed(42)
conf_feats_train = (
    torch.from_numpy(np.column_stack([train_meta["max_softmax"], train_meta["activation_norm"]]))
    .float()
    .to(OC_DEVICE)
)
loss_tgt_train = torch.from_numpy(train_meta["losses"]).float().to(OC_DEVICE)
mlp_ctrl = torch.nn.Sequential(torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)).to(OC_DEVICE)
opt = torch.optim.Adam(mlp_ctrl.parameters(), lr=1e-3, weight_decay=1e-4)
cs_ds = torch.utils.data.TensorDataset(conf_feats_train, loss_tgt_train)
cs_dl = torch.utils.data.DataLoader(cs_ds, batch_size=1024, shuffle=True)
for _ in range(20):
    for bx, by in cs_dl:
        loss = F.mse_loss(mlp_ctrl(bx).squeeze(-1), by)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
mlp_ctrl.eval().cpu()
del conf_feats_train, loss_tgt_train
if OC_DEVICE == "cuda":
    torch.cuda.empty_cache()

with torch.inference_mode():
    mlp_pred_val = (
        mlp_ctrl(torch.from_numpy(np.column_stack([val_sm, val_norms])).float()).squeeze(-1).numpy()
    )

cs_head = _train_peak(seed=EVAL_SEEDS[0])
cs_obs = observer_scores_for_shard_dir(cs_head, val_peak_dir)

ctrl_results = {}
for name, covs in [
    ("none", None),
    ("softmax_only", [val_sm]),
    ("norm_only", [val_norms]),
    ("standard", [val_sm, val_norms]),
    ("plus_entropy", [val_sm, val_norms, val_ent]),
    ("nonlinear", [mlp_pred_val]),
]:
    if covs is None:
        r, _ = spearmanr(cs_obs, val_losses)
    else:
        r, _ = partial_spearman(cs_obs, val_losses, covs)
    ctrl_results[name] = float(r)
    print(f"    {name:<16}: {r:+.4f}")


# Cross-domain
print("\n  Cross-domain:")
domain_results = {}
domain_rhos = []
for s in EVAL_SEEDS[:3]:
    h = _train_peak(seed=s)
    _, rho, _ = evaluate_head_streaming(h, val_peak_dir)
    domain_rhos.append(float(rho))
domain_results["wikitext"] = float(np.mean(domain_rhos))
print(f"    wikitext: {domain_results['wikitext']:+.4f}")

if not args.skip_c4:
    c4_test_dir = shard_dir_for(SHARD_ROOT, "c4_test", FINAL)
    c4_train_dir = shard_dir_for(SHARD_ROOT, "c4_train", FINAL)
    rhos = []
    for s in EVAL_SEEDS[:3]:
        h = _train_peak(seed=s)
        _, rho, _ = evaluate_head_streaming(h, c4_test_dir)
        rhos.append(float(rho))
    domain_results["c4"] = float(np.mean(rhos))
    print(f"    c4: {domain_results['c4']:+.4f}")
    rhos = []
    for s in EVAL_SEEDS[:3]:
        h = train_linear_binary_streaming(c4_train_dir, seed=s)
        _, rho, _ = evaluate_head_streaming(h, c4_test_dir)
        rhos.append(float(rho))
    domain_results["c4_within"] = float(np.mean(rhos))
    print(f"    c4_within: {domain_results['c4_within']:+.4f}")


# Flagging
print("\n  Flagging:")
val_output_meta = load_meta(val_output_dir)
val_output_sm = val_output_meta["max_softmax"]
nf = min(len(val_losses), len(val_output_sm))
fl = val_losses[:nf]
fsm = val_output_sm[:nf]
ml = float(np.median(fl))
ihl = fl > ml
fr = [0.05, 0.10, 0.20, 0.30]
fsm_sorted = np.sort(fsm)
conf_thresholds = {rate: fsm_sorted[int(nf * rate)] for rate in fr}
del fsm_sorted

fres = []
for seed in EVAL_SEEDS[:3]:
    h = _train_peak(seed=seed)
    osc = observer_scores_for_shard_dir(h, val_peak_dir)[:nf]
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
    str(r): {"observer_exclusive": float(np.mean([s["exclusive"][str(r)]["observer_only"] for s in fres]))}
    for r in fr
}
print(f"  10%: {fs['0.1']['observer_exclusive']:.0f} tokens")


# ── Save ─────────────────────────────────────────────────────────────


print(f"\n=== Saving [{elapsed_str()}] ===")
output = {
    "model": MODEL_ID,
    "n_params_b": round(n_params, 2),
    "n_layers": N_LAYERS,
    "hidden_dim": HIDDEN_DIM,
    "provenance": {
        "model_revision": _model_revision,
        "script": "scripts/run_stream_model.py",
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "output_file": args.output,
        "note": (
            "Shard-based streaming collection (bf16 on disk). Protocol identical to "
            f"run_model.py; only storage and placement differ. device_map={args.device_map}, "
            f"gpu_resident_phase5={args.gpu_resident_phase5}."
        ),
    },
    "protocol": {
        "layer_select_seed": LAYER_SELECT_SEED,
        "eval_seeds": EVAL_SEEDS,
        "target_ex_per_dim": TARGET_EX_PER_DIM,
        "batch_size": BATCH_SIZE,
        "layers_per_pass": LAYERS_PER_PASS,
        "storage": "bf16_shards_on_disk",
        "device_map": args.device_map,
        "gpu_resident_phase5": bool(args.gpu_resident_phase5 and DEVICE == "cuda"),
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

if Path("/workspace").exists():
    out_path = Path(f"/workspace/{args.output}")
else:
    out_path = Path(__file__).resolve().parent.parent / "results" / args.output
    out_path.parent.mkdir(exist_ok=True)

with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved {out_path}")
print(f"FINAL: L{FINAL} = {ev['mean']:+.4f} +/- {ev['std']:.4f}")
print(f"Output-controlled: {np.mean(ctrl_rhos):+.4f}")
print(f"Total time: {elapsed_str()}")

# Clean up shards unless asked to keep
if not args.keep_shards:
    print(f"\nCleaning up shards at {SHARD_ROOT}...")
    shutil.rmtree(SHARD_ROOT, ignore_errors=True)
    print("  done")
