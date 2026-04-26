"""Core probing functions for the nn-observability project.

Shared across run_model.py, analysis scripts, and per-model GPU scripts.
All functions operate on frozen model activations with no model training.

Usage:
    from probe import (partial_spearman, compute_loss_residuals,
                       train_linear_binary, evaluate_head,
                       collect_multi_layer_fast, collect_single_layer_fast,
                       compute_hand_designed, load_wikitext, pretokenize,
                       build_batches)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: F401

# ---------------------------------------------------------------------------
# Data loading and pre-tokenization
# ---------------------------------------------------------------------------


def load_wikitext(split: str = "test", max_docs: int | None = None) -> list[str]:
    """Load WikiText-103 documents. Lazy-imports datasets."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    docs: list[str] = []
    current: list[str] = []
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


def pretokenize(docs: list[str], tokenizer, max_length: int = 512) -> list[list[int]]:
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


def build_batches(encoded: list[list[int]], batch_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
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


def partial_spearman(x, y, covariates) -> tuple[float, float]:
    """Spearman rank partial correlation controlling for covariates."""
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates])
    rc = np.column_stack([rc, np.ones(len(rc))])
    coef_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    coef_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    r, p = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)
    return float(r), float(p)


def compute_loss_residuals(losses, max_softmax, activation_norm) -> np.ndarray:
    """OLS residuals of loss ~ max_softmax + activation_norm."""
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    beta = np.linalg.lstsq(X, losses, rcond=None)[0]
    return np.asarray(losses - X @ beta)


# ---------------------------------------------------------------------------
# Activation collection
# ---------------------------------------------------------------------------


def _get_layer_list(model) -> torch.nn.ModuleList:
    """Return the nn.ModuleList of transformer layers, architecture-agnostic."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # type: ignore[no-any-return]  # Llama, Qwen, Mistral, Gemma, Phi
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # type: ignore[no-any-return]  # GPT-2
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def collect_multi_layer_fast(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    layers: list[int],
    max_tokens: int,
    device: str,
    sm_chunk: int = 8,
) -> dict[int, dict]:
    """Collect activations from multiple layers using pre-built batches.
    All extraction done on GPU with masked tensor ops."""
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

    per_layer_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    per_layer_norms: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
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
        for ci in range(0, B, sm_chunk):
            p = shift_logits[ci : ci + sm_chunk].float().softmax(dim=-1)
            sm_2d[ci : ci + sm_chunk] = p.max(dim=-1).values
            ent_2d[ci : ci + sm_chunk] = -(p * (p + 1e-10).log()).sum(dim=-1)
            del p

        all_losses.append(losses_2d[shift_mask].float().cpu())
        all_sm.append(sm_2d[shift_mask].float().cpu())
        all_ent.append(ent_2d[shift_mask].float().cpu())

        for l in layers:
            h = captured[l][:, :-1, :].float()
            per_layer_acts[l].append(h[shift_mask].cpu())
            per_layer_norms[l].append(h.norm(dim=-1)[shift_mask].cpu())

        total += int(shift_mask.sum().item())
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


def collect_single_layer_fast(
    model,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    layer: int,
    max_tokens: int,
    device: str,
    sm_chunk: int = 8,
) -> dict:
    """Single-layer wrapper."""
    return collect_multi_layer_fast(model, batches, [layer], max_tokens, device, sm_chunk)[layer]


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------


def train_linear_binary(
    train_data: dict,
    seed: int = 42,
    epochs: int = 20,
    lr: float = 1e-3,
    train_device: str = "cpu",
) -> torch.nn.Linear:
    """Train a linear binary probe on residualized targets."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"].to(train_device)
    residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    targets = torch.from_numpy((residuals > 0).astype(np.float32)).to(train_device)
    head = torch.nn.Linear(acts.size(1), 1).to(train_device)
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


def evaluate_head(head: torch.nn.Module, test_data: dict) -> tuple[np.ndarray, float, float]:
    """Evaluate a probe head, returning (scores, partial_corr, p_value)."""
    head.eval()
    with torch.inference_mode():
        scores = head(test_data["activations"]).squeeze(-1).numpy()
    rho, p = partial_spearman(
        scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
    )
    return scores, rho, p


def compute_hand_designed(data: dict) -> dict[str, np.ndarray]:
    """Compute hand-designed activation statistics for baseline comparison."""
    acts = data["activations"]
    p = acts.abs() / (acts.abs().sum(dim=1, keepdim=True) + 1e-8)
    return {
        "ff_goodness": (acts**2).mean(dim=1).numpy(),
        "active_ratio": (acts.abs() > 0.01).float().mean(dim=1).numpy(),
        "act_entropy": -(p * (p + 1e-8).log()).sum(dim=1).numpy(),
        "activation_norm": data["activation_norm"],
    }
