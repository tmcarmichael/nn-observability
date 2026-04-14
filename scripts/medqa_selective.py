"""MedQA selective prediction: does the observer catch medical errors confidence misses?

Model: Qwen/Qwen2.5-7B-Instruct | GPU: H100/H200 | Peak layer from v3 results
Protocol: train WikiText probe (standard), generate MedQA answers, compare
  observer vs confidence at catching wrong answers.

Tests whether the exclusive catch finding (7-11% at 10% flag rate) generalizes
from general-domain text to safety-critical medical QA. If it holds, the
monitoring value argument extends to the domain where it matters most.

Usage: pip install transformers datasets scipy scikit-learn && python medqa_selective.py
"""

import gc
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.0f}m" if m < 60 else f"{m / 60:.1f}h"


# ---------------------------------------------------------------------------
# Probe functions (from run_model.py)
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

    for _bi, (input_ids_cpu, attn_mask_cpu) in enumerate(batches):
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


# ---------------------------------------------------------------------------
# MedQA data loading
# ---------------------------------------------------------------------------


def load_medqa(max_questions=1000):
    """Load MedQA USMLE-style 4-option multiple choice questions."""
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    questions = []
    for row in ds:
        if len(questions) >= max_questions:
            break
        q = row["question"]
        options = row["options"]
        answer_idx = row["answer_idx"]
        answer = options[answer_idx] if isinstance(answer_idx, int) else row["answer"]

        option_text = (
            "\n".join(f"  {k}) {v}" for k, v in options.items())
            if isinstance(options, dict)
            else str(options)
        )

        questions.append(
            {
                "question": q,
                "options": option_text,
                "answer": answer,
                "answer_key": answer_idx if isinstance(answer_idx, str) else chr(65 + answer_idx),
            }
        )

    print(f"  Loaded {len(questions)} MedQA questions")
    return questions


# ---------------------------------------------------------------------------
# Generation with observer scoring
# ---------------------------------------------------------------------------


def format_medqa_prompt(question, options, tokenizer):
    messages = [
        {
            "role": "user",
            "content": (
                f"Answer the following medical question by selecting the correct option. "
                f"Reply with only the letter (A, B, C, or D).\n\n"
                f"{question}\n\n{options}\n\nAnswer:"
            ),
        }
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{question}\n\n{options}\n\nAnswer:"


def greedy_decode_with_scores(model, tokenizer, observer_head, peak_layer, prompt, device, max_new_tokens=16):
    model.eval()
    observer_head.eval()

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3500)["input_ids"].to(
        device
    )
    prompt_len = input_ids.size(1)
    observer_scores = []
    confidences = []
    past_key_values = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            input_chunk = input_ids if past_key_values is None else input_ids[:, -1:]
            outputs = model(
                input_chunk, past_key_values=past_key_values, output_hidden_states=True, use_cache=True
            )
            past_key_values = outputs.past_key_values

            h = outputs.hidden_states[peak_layer + 1][0, -1:, :].cpu().float()
            score = observer_head(h).squeeze().item()

            logits = outputs.logits[0, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            confidence = F.softmax(logits, dim=-1).max().item()

            observer_scores.append(score)
            confidences.append(confidence)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break
            decoded = tokenizer.decode(next_token.item())
            if "\n" in decoded and step > 0:
                break

    answer_ids = input_ids[0, prompt_len:].tolist()
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    obs_arr = np.array(observer_scores) if observer_scores else np.array([0.0])
    conf_arr = np.array(confidences) if confidences else np.array([0.0])

    return {
        "answer_text": answer_text,
        "mean_observer": float(obs_arr.mean()),
        "max_observer": float(obs_arr.max()),
        "mean_confidence": float(conf_arr.mean()),
        "min_confidence": float(conf_arr.min()),
        "n_tokens": len(observer_scores),
    }


def extract_answer_letter(text):
    """Extract answer letter (A/B/C/D) from generated text."""
    text = text.strip().upper()
    if text and text[0] in "ABCD":
        return text[0]
    match = re.search(r"[ABCD]", text)
    return match.group(0) if match else text[:1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PEAK_LAYER = 14
TARGET_EX_PER_DIM = 350
MAX_QUESTIONS = 1000
SEEDS = [42, 43, 44]

print(f"=== MedQA selective prediction [{elapsed_str()}] ===")
print(f"Model: {MODEL_ID}, peak layer: {PEAK_LAYER}")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa"
).to(DEVICE)
model.eval()

HIDDEN_DIM = model.config.hidden_size
MAX_TRAIN = TARGET_EX_PER_DIM * HIDDEN_DIM
print(f"Hidden dim: {HIDDEN_DIM}, train tokens: {MAX_TRAIN}")

# --- Train probe on WikiText ---
print(f"\n=== Training WikiText probe [{elapsed_str()}] ===")
wiki_train_docs = load_wikitext("train", max_docs=8000)
wiki_train_enc = pretokenize(wiki_train_docs, tokenizer)
train_batches = build_batches(wiki_train_enc, 48)
del wiki_train_docs, wiki_train_enc

train_data = collect_single_layer(model, train_batches, PEAK_LAYER, MAX_TRAIN, DEVICE)
del train_batches
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

probes = []
for seed in SEEDS:
    head = train_linear_binary(train_data, seed=seed)
    probes.append(head)
    print(f"  Probe seed {seed} trained")
del train_data

# --- Load MedQA ---
print(f"\n=== Loading MedQA [{elapsed_str()}] ===")
questions = load_medqa(max_questions=MAX_QUESTIONS)

# --- Generate and score ---
print(f"\n=== Generating MedQA answers [{elapsed_str()}] ===")
all_results = []
for qi, q in enumerate(questions):
    prompt = format_medqa_prompt(q["question"], q["options"], tokenizer)

    seed_obs, seed_conf = [], []
    for head in probes:
        gen = greedy_decode_with_scores(model, tokenizer, head, PEAK_LAYER, prompt, DEVICE)
        seed_obs.append(gen["mean_observer"])
        seed_conf.append(gen["mean_confidence"])

    answer_text = gen["answer_text"]
    pred_letter = extract_answer_letter(answer_text)
    is_correct = pred_letter == q["answer_key"]

    all_results.append(
        {
            "question": q["question"][:100],
            "answer": answer_text,
            "pred_letter": pred_letter,
            "gold_letter": q["answer_key"],
            "correct": is_correct,
            "mean_observer": float(np.mean(seed_obs)),
            "mean_confidence": float(np.mean(seed_conf)),
        }
    )

    if (qi + 1) % 50 == 0:
        n_correct = sum(r["correct"] for r in all_results)
        print(
            f"  {qi + 1}/{len(questions)}: {n_correct}/{qi + 1} correct ({n_correct / (qi + 1) * 100:.1f}%) [{elapsed_str()}]"
        )

del model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# --- Analysis ---
print(f"\n=== Analysis [{elapsed_str()}] ===")
n = len(all_results)
correct = np.array([r["correct"] for r in all_results])
obs = np.array([r["mean_observer"] for r in all_results])
conf = np.array([r["mean_confidence"] for r in all_results])

accuracy = float(correct.mean())
n_errors = int((~correct).sum())
print(f"  Accuracy: {accuracy:.3f} ({n_errors} errors out of {n})")

# Exclusive catches
print("\n  Exclusive catches:")
flag_rates = [0.05, 0.10, 0.20, 0.30]
catch_results = {}
for rate in flag_rates:
    k = max(1, int(n * rate))
    obs_flagged = obs >= np.sort(obs)[-k]
    conf_flagged = conf <= np.sort(conf)[k]

    obs_exclusive = int((obs_flagged & ~conf_flagged & ~correct).sum())
    conf_exclusive = int((conf_flagged & ~obs_flagged & ~correct).sum())
    both = int((obs_flagged & conf_flagged & ~correct).sum())

    pct = obs_exclusive / n_errors * 100 if n_errors > 0 else 0
    print(
        f"    {rate:.0%}: observer-exclusive={obs_exclusive} ({pct:.1f}% of errors), conf-exclusive={conf_exclusive}, both={both}"
    )

    catch_results[str(rate)] = {
        "observer_exclusive": obs_exclusive,
        "confidence_exclusive": conf_exclusive,
        "both": both,
        "pct_of_errors": round(pct, 1),
    }

# Coverage-accuracy curves
print("\n  Coverage-accuracy (AUACC):")
coverage_levels = list(np.arange(1.0, 0.49, -0.05))
for strategy_name, scores, ascending in [
    ("observer", obs, False),
    ("confidence", conf, True),
]:
    order = np.argsort(scores) if ascending else np.argsort(-scores)
    accs = []
    for cov in coverage_levels:
        k = max(1, int(n * cov))
        kept = order[:k]
        accs.append(float(correct[kept].mean()))
    auacc = float(np.trapz(accs, coverage_levels))
    print(f"    {strategy_name}: AUACC = {auacc:.3f}")
    catch_results[f"{strategy_name}_auacc"] = auacc

# --- Save ---
print(f"\n=== Saving [{elapsed_str()}] ===")
output = {
    "model": MODEL_ID,
    "task": "medqa_selective_prediction",
    "dataset": "GBaker/MedQA-USMLE-4-options",
    "peak_layer": PEAK_LAYER,
    "n_questions": n,
    "n_errors": n_errors,
    "accuracy": accuracy,
    "probe_seeds": SEEDS,
    "flag_rates": catch_results,
    "per_question": all_results[:50],
    "summary": {
        "accuracy": accuracy,
        "n_questions": n,
        "n_errors": n_errors,
        "exclusive_at_10pct": catch_results.get("0.1", {}),
        "observer_auacc": catch_results.get("observer_auacc"),
        "confidence_auacc": catch_results.get("confidence_auacc"),
    },
}

out_path = Path("/workspace/medqa_selective_results.json")
if not out_path.parent.exists():
    out_path = Path("results/medqa_selective_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {out_path}")
print(f"Accuracy: {accuracy:.3f}, errors: {n_errors}")
for rate in [0.05, 0.1, 0.2]:
    cr = catch_results[str(rate)]
    print(f"  @{rate:.0%}: observer-exclusive {cr['observer_exclusive']} ({cr['pct_of_errors']}% of errors)")
print(f"Total time: {elapsed_str()}")
