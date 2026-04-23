"""RAG hallucination detection: does the observer flag errors confidence misses?

Trains a WikiText probe, generates answers on SQuAD 2.0 passages, and
compares observer and confidence at flagging wrong answers. Tests whether
the probe transfers to a retrieval-augmented generation failure mode.
"""

import gc
import json
import re
import shutil
import string
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata

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
    all_acts, all_norms, all_losses, all_sm, all_ent = [], [], [], [], []
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
        ent_2d = torch.empty(B, S - 1, device=device)
        for ci in range(0, B, sm_chunk):
            p = shift_logits[ci : ci + sm_chunk].float().softmax(dim=-1)
            sm_2d[ci : ci + sm_chunk] = p.max(dim=-1).values
            ent_2d[ci : ci + sm_chunk] = -(p * (p + 1e-10).log()).sum(dim=-1)
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

    handle.remove()
    n = min(total, max_tokens)
    return {
        "activations": torch.cat(all_acts)[:n],
        "losses": torch.cat(all_losses).numpy()[:n],
        "max_softmax": torch.cat(all_sm).numpy()[:n],
        "logit_entropy": torch.cat(all_ent).numpy()[:n],
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
# NQ data loading
# ---------------------------------------------------------------------------


def load_nq(max_questions=1000):
    """Load SQuAD 2.0 answerable questions (paper baseline uses SQuAD, function name kept for caller compat)."""
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    questions = []
    for row in ds:
        if len(questions) >= max_questions:
            break
        answers = row["answers"]["text"]
        if not answers:
            continue  # skip SQuAD 2.0 unanswerable cases; baseline used answerable-only
        context = row["context"]
        if not context.strip():
            continue
        seen = set()
        unique = []
        for a in answers:
            norm = a.strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(a.strip())
        if not unique:
            continue
        questions.append({"question": row["question"], "context": context, "answers": unique})

    print(f"  Loaded {len(questions)} SQuAD 2.0 answerable questions")
    return questions


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match(prediction, references):
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(ref) == norm_pred for ref in references)


# ---------------------------------------------------------------------------
# Generation with observer scoring
# ---------------------------------------------------------------------------


def format_rag_prompt(question, context, tokenizer):
    messages = [
        {
            "role": "user",
            "content": (
                f"Answer the question based on the following context. "
                f"Be concise (1-5 words).\n\n"
                f"Context: {context[:2000]}\n\n"
                f"Question: {question}\nAnswer:"
            ),
        }
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"Context: {context[:2000]}\n\nQuestion: {question}\nAnswer:"


def greedy_decode_with_scores(model, tokenizer, observer_head, peak_layer, prompt, device, max_new_tokens=32):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import argparse
import datetime as _dt

parser = argparse.ArgumentParser(description="RAG hallucination detection with observer probe.")
parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="HuggingFace model ID")
parser.add_argument("--peak-layer", type=int, default=14, help="Peak layer index (0-indexed)")
parser.add_argument("--ex-dim", type=int, default=350, help="Probe training examples per hidden dim")
parser.add_argument("--max-questions", type=int, default=1000, help="RAG questions to evaluate")
parser.add_argument("--seeds", default="42,43,44", help="Comma-separated probe seeds")
parser.add_argument("--output", default=None, help="Output JSON path (default: auto from model slug)")
parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
parser.add_argument(
    "--trust-remote-code",
    action="store_true",
    help="Pass trust_remote_code=True to from_pretrained (native HF impls preferred otherwise)",
)
args = parser.parse_args()

MODEL_ID = args.model
PEAK_LAYER = args.peak_layer
TARGET_EX_PER_DIM = args.ex_dim
MAX_QUESTIONS = args.max_questions
SEEDS = [int(s) for s in args.seeds.split(",")]
DTYPE = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
MODEL_SLUG = re.sub(r"[^A-Za-z0-9]+", "_", MODEL_ID.split("/")[-1]).strip("_").lower()

print(f"=== RAG hallucination detection [{elapsed_str()}] ===")
print(f"Model: {MODEL_ID} (slug={MODEL_SLUG}), peak layer: {PEAK_LAYER}, dtype: {args.dtype}")

from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=args.trust_remote_code)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=args.trust_remote_code,
    dtype=DTYPE,
    attn_implementation=args.attn_impl,
).to(DEVICE)
model.eval()

_model_revision = getattr(model.config, "_commit_hash", None) or "unknown"

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

# --- Load NQ ---
print(f"\n=== Loading Natural Questions [{elapsed_str()}] ===")
questions = load_nq(max_questions=MAX_QUESTIONS)

# --- Generate and score ---
print(f"\n=== Generating RAG answers [{elapsed_str()}] ===")
all_results = []
for qi, q in enumerate(questions):
    prompt = format_rag_prompt(q["question"], q["context"], tokenizer)

    # Average scores across probe seeds
    seed_obs, seed_conf = [], []
    for head in probes:
        gen = greedy_decode_with_scores(model, tokenizer, head, PEAK_LAYER, prompt, DEVICE)
        seed_obs.append(gen["mean_observer"])
        seed_conf.append(gen["mean_confidence"])

    answer_text = gen["answer_text"]  # same greedy decode across seeds
    is_correct = exact_match(answer_text, q["answers"])

    all_results.append(
        {
            "question": q["question"],
            "answer": answer_text,
            "gold": list(q["answers"]),
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
em_correct = np.array([r["em_correct"] for r in all_results])
obs = np.array([r["mean_observer"] for r in all_results])
conf = np.array([r["mean_confidence"] for r in all_results])

accuracy = float(correct.mean())  # exact_match
n_errors = int((~correct).sum())
print(f"  Accuracy (exact_match): {accuracy:.3f} ({n_errors} errors out of {n})")

# Exclusive catches at various flag rates
print("\n  Exclusive catches (observer flags error, confidence doesn't):")
flag_rates = [0.05, 0.10, 0.20, 0.30]
catch_results = {}
for rate in flag_rates:
    k = max(1, int(n * rate))
    # Observer flags: highest observer score = most concerning
    obs_flagged = obs >= np.sort(obs)[-k]
    # Confidence flags: lowest confidence = most concerning
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

# --- Save ---
print(f"\n=== Saving [{elapsed_str()}] ===")
output = {
    "model": MODEL_ID,
    "task": "rag_hallucination",
    "dataset": "rajpurkar/squad_v2",
    "peak_layer": PEAK_LAYER,
    "provenance": {
        "model_revision": _model_revision,
        "script": "scripts/rag_hallucination.py",
        "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
        "device": str(DEVICE),
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "torch_version": torch.__version__,
        "args": vars(args),
    },
    "protocol": {
        "target_ex_per_dim": TARGET_EX_PER_DIM,
        "probe_seeds": SEEDS,
    },
    "n_questions": n,
    "n_errors": n_errors,
    "accuracy": accuracy,
    "probe_seeds": SEEDS,
    "flag_rates": catch_results,
    "grading": "exact_match",
    "per_question": all_results,  # full 1000 with predictions and exact-match labels
    "summary": {
        "accuracy": accuracy,
        "n_questions": n,
        "n_errors": n_errors,
        "exclusive_at_10pct": catch_results.get("0.1", {}),
    },
}

if args.output:
    out_path = Path(args.output)
else:
    filename = f"rag_hallucination_{MODEL_SLUG}_results.json"
    out_path = Path("/workspace") / filename
    if not out_path.parent.exists():
        out_path = Path("results") / filename
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved {out_path}")
print(f"Accuracy: {accuracy:.3f}, errors: {n_errors}")
for rate in [0.05, 0.1, 0.2]:
    cr = catch_results[str(rate)]
    print(f"  @{rate:.0%}: observer-exclusive {cr['observer_exclusive']} ({cr['pct_of_errors']}% of errors)")
print(f"Total time: {elapsed_str()}")
