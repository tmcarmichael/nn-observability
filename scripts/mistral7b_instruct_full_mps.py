"""Mistral 7B Instruct end-to-end replication.

Runs a sparse layer sweep, a seven-seed WikiText anchor at the selected
peak layer, and three downstream evaluations (MedQA, RAG, TruthfulQA) so
Mistral 7B Instruct can appear as its own row in the cross-family table.
"""

import argparse
import gc
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import trapezoid

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_EX_PER_DIM = 350
SWEEP_EX_PER_DIM = 100
ANCHOR_SEEDS = list(range(42, 49))  # 7 seeds for WikiText anchor
DOWNSTREAM_SEEDS = [42, 43, 44]  # 3 seeds matching Qwen protocol
SWEEP_CANDIDATES = [14, 18, 20, 22, 24, 26]  # around Mistral 7B base L22

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.1f}m" if m < 60 else f"{m / 60:.2f}h"


# ---------------------------------------------------------------------------
# WikiText loading, tokenization, activation collection, probe training
# (mirrors scripts/run_model.py and the Phi-3 downstream script).
# ---------------------------------------------------------------------------


def load_wikitext(split, max_docs=None):
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


def collect_single_layer(model, batches, layer, max_tokens, device, sm_chunk=4):
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

    try:
        for bi, (input_ids_cpu, attn_mask_cpu) in enumerate(batches):
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

            losses_2d = F.cross_entropy(
                shift_logits.view(-1, V), shift_labels.view(-1), reduction="none"
            ).view(B, S - 1)

            sm_2d = torch.empty(B, S - 1, device=device)
            for ci in range(0, B, sm_chunk):
                p = shift_logits[ci : ci + sm_chunk].float().softmax(dim=-1)
                sm_2d[ci : ci + sm_chunk] = p.max(dim=-1).values
                del p

            h = captured[0][:, :-1, :]
            all_acts.append(h[shift_mask].cpu())  # keep fp16 on CPU
            all_norms.append(h.float().norm(dim=-1)[shift_mask].cpu())
            all_losses.append(losses_2d[shift_mask].float().cpu())
            all_sm.append(sm_2d[shift_mask].float().cpu())
            total += shift_mask.sum().item()

            captured.pop(0, None)
            del outputs, input_ids, attn_mask, shift_logits, shift_labels
            del losses_2d, sm_2d, shift_mask, h
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()

            if (bi + 1) % 25 == 0:
                print(
                    f"    batch {bi + 1}/{len(batches)}  tokens={total}  [{elapsed_str()}]",
                    flush=True,
                )
    finally:
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
    acts = train_data["activations"].float().to(TRAIN_DEVICE)
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


def partial_spearman(x, y, controls):
    """Spearman partial correlation of x and y given controls. All numpy."""
    from scipy.stats import rankdata

    ranked = [rankdata(v) for v in [x, y, *controls]]
    C = np.column_stack([ranked[i] for i in range(2, len(ranked))])
    C = np.column_stack([C, np.ones(len(x))])
    # residualize rank(x) and rank(y) against rank(controls)
    beta_x = np.linalg.lstsq(C, ranked[0], rcond=None)[0]
    beta_y = np.linalg.lstsq(C, ranked[1], rcond=None)[0]
    rx = ranked[0] - C @ beta_x
    ry = ranked[1] - C @ beta_y
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def evaluate_probe(head, test_data):
    head.eval()
    with torch.inference_mode():
        scores = head(test_data["activations"].float()).squeeze(-1).numpy()
    rho = partial_spearman(
        scores,
        test_data["losses"],
        [test_data["max_softmax"], test_data["activation_norm"]],
    )
    return scores, rho


# ---------------------------------------------------------------------------
# Output-controlled residual: MLP on last-layer residual stream predicts loss,
# then partial correlation additionally controls for that prediction.
# ---------------------------------------------------------------------------


def train_output_mlp(train_data, seed=42, hidden=64, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts = train_data["activations"].float().to(TRAIN_DEVICE)
    losses = torch.from_numpy(train_data["losses"]).float().to(TRAIN_DEVICE)
    d = acts.size(1)
    mlp = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(
        TRAIN_DEVICE
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts, losses)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)
    mlp.train()
    for _ in range(epochs):
        for bx, by in dl:
            pred = mlp(bx).squeeze(-1)
            loss = F.mse_loss(pred, by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    mlp.eval()
    return mlp.cpu()


# ---------------------------------------------------------------------------
# Hook-based greedy decode with multi-probe scoring for downstream tasks.
# ---------------------------------------------------------------------------


def greedy_decode_with_observer(model, tokenizer, probes, peak_layer, prompt, device, max_new_tokens):
    model.eval()
    layer_modules = _get_layer_list(model)
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured[0] = h

    handle = layer_modules[peak_layer].register_forward_hook(hook_fn)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].to(
        device
    )
    prompt_len = input_ids.size(1)
    per_probe_scores = [[] for _ in probes]
    confidences = []
    past_key_values = None

    try:
        with torch.inference_mode():
            for step in range(max_new_tokens):
                input_chunk = input_ids if past_key_values is None else input_ids[:, -1:]
                outputs = model(input_chunk, past_key_values=past_key_values, use_cache=True)
                past_key_values = outputs.past_key_values

                h = captured[0][0, -1:, :].cpu().float()
                for i, head in enumerate(probes):
                    head.eval()
                    per_probe_scores[i].append(head(h).squeeze().item())

                logits = outputs.logits[0, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                confidence = F.softmax(logits.float(), dim=-1).max().item()
                confidences.append(confidence)

                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                captured.pop(0, None)

                if next_token.item() == tokenizer.eos_token_id:
                    break
                decoded = tokenizer.decode(next_token.item())
                if "\n" in decoded and step > 0:
                    break
    finally:
        handle.remove()

    answer_ids = input_ids[0, prompt_len:].tolist()
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    obs_means = [float(np.mean(s)) if s else 0.0 for s in per_probe_scores]
    conf_arr = np.array(confidences) if confidences else np.array([0.0])

    return {
        "answer_text": answer_text,
        "mean_observer": float(np.mean(obs_means)),
        "mean_confidence": float(conf_arr.mean()),
        "min_confidence": float(conf_arr.min()),
        "n_tokens": len(confidences),
    }


# ---------------------------------------------------------------------------
# Downstream task loaders and evaluators (copied from phi3_downstream_mps.py
# for self-containment).
# ---------------------------------------------------------------------------


def load_medqa(max_questions):
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    questions = []
    for row in ds:
        if len(questions) >= max_questions:
            break
        options = row["options"]
        answer_idx = row["answer_idx"]
        option_text = (
            "\n".join(f"  {k}) {v}" for k, v in options.items())
            if isinstance(options, dict)
            else str(options)
        )
        questions.append(
            {
                "question": row["question"],
                "options": option_text,
                "answer_key": answer_idx if isinstance(answer_idx, str) else chr(65 + answer_idx),
            }
        )
    return questions


def format_medqa_prompt(question, options, tokenizer):
    messages = [
        {
            "role": "user",
            "content": (
                "Answer the following medical question by selecting the correct option. "
                "Reply with only the letter (A, B, C, or D).\n\n"
                f"{question}\n\n{options}\n\nAnswer:"
            ),
        }
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{question}\n\n{options}\n\nAnswer:"


def extract_answer_letter(text):
    text = text.strip().upper()
    if text and text[0] in "ABCD":
        return text[0]
    match = re.search(r"[ABCD]", text)
    return match.group(0) if match else text[:1]


def load_nq(max_questions):
    from datasets import load_dataset

    ds = load_dataset(
        "google-research-datasets/natural_questions",
        "default",
        split="validation",
        streaming=True,
    )
    questions = []
    for row in ds:
        if len(questions) >= max_questions:
            break
        q_text = row["question"]["text"]
        doc_tokens = row["document"]["tokens"]["token"]
        doc_html = row["document"]["tokens"]["is_html"]
        plain = [t for t, h in zip(doc_tokens, doc_html) if not h]
        context = " ".join(plain[:500])

        annotations = row["annotations"]
        short_answers = []
        for ann in annotations["short_answers"]:
            for text in ann.get("text", []):
                answer = re.sub(r"<[^>]+>", "", text).strip()
                if answer:
                    short_answers.append(answer)

        if not short_answers or not context.strip():
            continue
        seen = set()
        unique = []
        for a in short_answers:
            norm = a.strip().lower()
            if norm not in seen:
                seen.add(norm)
                unique.append(a.strip())
        questions.append({"question": q_text, "context": context, "answers": unique})
    return questions


def format_rag_prompt(question, context, tokenizer):
    messages = [
        {
            "role": "user",
            "content": (
                "Answer the question based on the following context. "
                "Be concise (1-5 words).\n\n"
                f"Context: {context[:2000]}\n\n"
                f"Question: {question}\nAnswer:"
            ),
        }
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"Context: {context[:2000]}\n\nQuestion: {question}\nAnswer:"


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def rag_is_correct(prediction, references):
    norm_pred = normalize_answer(prediction)
    for ref in references:
        if normalize_answer(ref) in norm_pred or norm_pred in normalize_answer(ref):
            return True
    return False


def load_truthfulqa():
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    questions = []
    for row in ds:
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        questions.append(
            {
                "question": row["question"],
                "correct_answer": choices[correct_idx],
                "incorrect_answers": [c for c, l in zip(choices, labels) if l == 0],
                "category": row.get("category", "unknown"),
            }
        )
    return questions


def truthfulqa_is_correct(prediction, correct_answer, incorrect_answers):
    norm_pred = normalize_answer(prediction)
    norm_correct = normalize_answer(correct_answer)
    if norm_correct in norm_pred:
        return True
    for inc in incorrect_answers:
        if normalize_answer(inc) in norm_pred:
            return False
    pred_tokens = set(norm_pred.split())
    correct_overlap = len(pred_tokens & set(norm_correct.split()))
    best_incorrect_overlap = max(
        (len(pred_tokens & set(normalize_answer(inc).split())) for inc in incorrect_answers),
        default=0,
    )
    return correct_overlap > best_incorrect_overlap and correct_overlap > 0


# ---------------------------------------------------------------------------
# Downstream evaluation wrapper
# ---------------------------------------------------------------------------


def run_downstream_task(model, tokenizer, probes, peak_layer, task_name, dataset_name, max_questions):
    print(f"\n=== {task_name} [{elapsed_str()}] ===", flush=True)

    if task_name == "medqa_selective_prediction":
        questions = load_medqa(max_questions)

        def format_fn(q):
            return format_medqa_prompt(q["question"], q["options"], tokenizer)

        max_new = 16
    elif task_name == "rag_hallucination":
        questions = load_nq(max_questions)

        def format_fn(q):
            return format_rag_prompt(q["question"], q["context"], tokenizer)

        max_new = 32
    elif task_name == "truthfulqa_hallucination":
        questions = load_truthfulqa()[:max_questions] if max_questions else load_truthfulqa()

        def format_fn(q):
            return _format_truthfulqa(q, tokenizer)

        max_new = 64
    else:
        raise ValueError(task_name)

    print(f"  Loaded {len(questions)} questions", flush=True)

    results = []
    for qi, q in enumerate(questions):
        prompt = format_fn(q)
        gen = greedy_decode_with_observer(
            model, tokenizer, probes, peak_layer, prompt, DEVICE, max_new_tokens=max_new
        )
        correct = _check_correct(task_name, gen["answer_text"], q)

        entry = {
            "answer": gen["answer_text"],
            "correct": correct,
            "mean_observer": gen["mean_observer"],
            "mean_confidence": gen["mean_confidence"],
        }
        if task_name == "medqa_selective_prediction":
            entry["pred_letter"] = extract_answer_letter(gen["answer_text"])
            entry["gold_letter"] = q["answer_key"]
        elif task_name == "rag_hallucination":
            entry["gold_answers"] = q["answers"][:5]
        else:
            entry["correct_answer"] = q["correct_answer"]
            entry["category"] = q["category"]
            entry["min_confidence"] = gen["min_confidence"]
            entry["n_tokens"] = gen["n_tokens"]

        results.append(entry)

        if (qi + 1) % 50 == 0:
            n_c = sum(r["correct"] for r in results)
            print(
                f"  {qi + 1}/{len(questions)}: {n_c}/{qi + 1} "
                f"({n_c / (qi + 1) * 100:.1f}%) [{elapsed_str()}]",
                flush=True,
            )

    return _analyze_selective(results, task=task_name, dataset=dataset_name, peak_layer=peak_layer)


def _format_truthfulqa(q, tokenizer):
    messages = [
        {
            "role": "user",
            "content": f"Answer the following question truthfully and concisely.\n\nQuestion: {q['question']}\nAnswer:",
        }
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"Question: {q['question']}\nAnswer:"


def _check_correct(task_name, answer_text, q):
    if task_name == "medqa_selective_prediction":
        return extract_answer_letter(answer_text) == q["answer_key"]
    if task_name == "rag_hallucination":
        return rag_is_correct(answer_text, q["answers"])
    return truthfulqa_is_correct(answer_text, q["correct_answer"], q["incorrect_answers"])


def _analyze_selective(results, task, dataset, peak_layer):
    n = len(results)
    correct = np.array([r["correct"] for r in results])
    obs = np.array([r["mean_observer"] for r in results])
    conf = np.array([r["mean_confidence"] for r in results])

    accuracy = float(correct.mean())
    n_errors = int((~correct).sum())
    print(f"  Accuracy: {accuracy:.3f} ({n_errors} errors / {n} questions)", flush=True)

    catches = {}
    for rate in [0.05, 0.10, 0.20, 0.30]:
        k = max(1, int(n * rate))
        obs_flagged = obs >= np.sort(obs)[-k]
        conf_flagged = conf <= np.sort(conf)[k]
        obs_exclusive = int((obs_flagged & ~conf_flagged & ~correct).sum())
        conf_exclusive = int((conf_flagged & ~obs_flagged & ~correct).sum())
        both = int((obs_flagged & conf_flagged & ~correct).sum())
        pct = obs_exclusive / n_errors * 100 if n_errors > 0 else 0
        print(
            f"    @{rate:.0%}: obs-excl={obs_exclusive} ({pct:.1f}%), "
            f"conf-excl={conf_exclusive}, both={both}",
            flush=True,
        )
        catches[str(rate)] = {
            "observer_exclusive": obs_exclusive,
            "confidence_exclusive": conf_exclusive,
            "both": both,
            "pct_of_errors": round(pct, 1),
        }

    coverage_levels = list(np.arange(1.0, 0.49, -0.05))
    for name, scores, ascending in [("observer", obs, False), ("confidence", conf, True)]:
        order = np.argsort(scores) if ascending else np.argsort(-scores)
        accs = []
        for cov in coverage_levels:
            k = max(1, int(n * cov))
            accs.append(float(correct[order[:k]].mean()))
        catches[f"{name}_auacc"] = float(trapezoid(accs, coverage_levels))

    return {
        "model": MODEL_ID,
        "task": task,
        "dataset": dataset,
        "peak_layer": peak_layer,
        "n_questions": n,
        "n_errors": n_errors,
        "accuracy": accuracy,
        "probe_seeds": DOWNSTREAM_SEEDS,
        "flag_rates": catches,
        "per_question": results[:50],
        "summary": {
            "accuracy": accuracy,
            "n_questions": n,
            "n_errors": n_errors,
            "exclusive_at_10pct": catches.get("0.1", {}),
            "observer_auacc": catches.get("observer_auacc"),
            "confidence_auacc": catches.get("confidence_auacc"),
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="10 questions per downstream task, skip sweep")
    parser.add_argument("--skip-sweep", action="store_true", help="Use --peak-layer directly without sweep")
    parser.add_argument("--peak-layer", type=int, default=None, help="Override peak layer selection")
    parser.add_argument(
        "--tasks",
        default="anchor,medqa,rag,truthfulqa",
        help="Comma-separated subset of {anchor,medqa,rag,truthfulqa}",
    )
    parser.add_argument("--max-questions", type=int, default=1000)
    args = parser.parse_args()

    max_questions = 10 if args.smoke else args.max_questions
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"=== Mistral 7B Instruct full replication [{elapsed_str()}] ===", flush=True)
    print(f"Device: {DEVICE}  Train device: {TRAIN_DEVICE}", flush=True)
    print(f"Model: {MODEL_ID}", flush=True)
    print(f"Tasks: {tasks}  max_questions={max_questions}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float16 if DEVICE == "mps" else torch.bfloat16
    attn_impl = "eager" if DEVICE == "mps" else "sdpa"
    print(f"Loading model dtype={model_dtype} attn={attn_impl} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=model_dtype, attn_implementation=attn_impl
    ).to(DEVICE)
    model.eval()

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(
        f"Params {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B  n_layers={n_layers}  hidden={hidden_dim}",
        flush=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- WikiText loading ---
    print(f"\n=== Pre-tokenizing WikiText [{elapsed_str()}] ===", flush=True)
    train_docs = load_wikitext("train", max_docs=8000)
    val_docs = load_wikitext("validation")
    test_docs = load_wikitext("test")
    print(
        f"  train={len(train_docs)} docs  val={len(val_docs)} docs  test={len(test_docs)} docs",
        flush=True,
    )
    train_enc = pretokenize(train_docs, tokenizer)
    val_enc = pretokenize(val_docs, tokenizer)
    test_enc = pretokenize(test_docs, tokenizer)
    del train_docs, val_docs, test_docs
    batch_size = 8 if DEVICE == "mps" else 48
    train_batches = build_batches(train_enc, batch_size)
    val_batches = build_batches(val_enc, batch_size)
    test_batches = build_batches(test_enc, batch_size)
    del train_enc, val_enc, test_enc
    print(
        f"  batches: train={len(train_batches)}  val={len(val_batches)}  test={len(test_batches)}",
        flush=True,
    )

    # --- Layer sweep ---
    peak_layer = args.peak_layer
    sweep_results = None
    if peak_layer is None and not args.skip_sweep and not args.smoke:
        print(f"\n=== Layer sweep [{elapsed_str()}] ===", flush=True)
        print(f"  Candidates: {SWEEP_CANDIDATES}  ex/dim={SWEEP_EX_PER_DIM}  seed=42", flush=True)
        max_sweep = SWEEP_EX_PER_DIM * hidden_dim
        sweep_results = {}
        for layer in SWEEP_CANDIDATES:
            print(f"\n  -- L{layer} train collect --", flush=True)
            tr = collect_single_layer(model, train_batches, layer, max_sweep, DEVICE)
            head = train_linear_binary(tr, seed=42)
            print(f"  -- L{layer} val collect --", flush=True)
            va = collect_single_layer(model, val_batches, layer, 200 * hidden_dim, DEVICE)
            _, rho = evaluate_probe(head, va)
            sweep_results[layer] = float(rho)
            print(f"    L{layer}: pcorr = {rho:.4f}", flush=True)
            del tr, va, head
            gc.collect()

        peak_layer = max(sweep_results, key=sweep_results.get)
        print(
            f"\n  Sweep peak: L{peak_layer} (pcorr={sweep_results[peak_layer]:.4f})",
            flush=True,
        )
    else:
        if peak_layer is None:
            peak_layer = 22
        print(f"\n  Skipping sweep, using peak_layer=L{peak_layer}", flush=True)

    # --- Full WikiText anchor at peak layer ---
    print(f"\n=== WikiText anchor at L{peak_layer} [{elapsed_str()}] ===", flush=True)
    full_max = TARGET_EX_PER_DIM * hidden_dim
    print(f"  ex/dim={TARGET_EX_PER_DIM}  target tokens={full_max}", flush=True)

    print("  collecting train activations ...", flush=True)
    train_data = collect_single_layer(model, train_batches, peak_layer, full_max, DEVICE)
    print(f"    {train_data['activations'].shape}", flush=True)

    print("  collecting test activations ...", flush=True)
    test_data = collect_single_layer(model, test_batches, peak_layer, 100 * hidden_dim, DEVICE)
    print(f"    {test_data['activations'].shape}", flush=True)

    # Train 7 probes on train split, evaluate each on test split
    anchor_seeds = DOWNSTREAM_SEEDS if args.smoke else ANCHOR_SEEDS
    probe_rhos = []
    probes_list = []
    for seed in anchor_seeds:
        head = train_linear_binary(train_data, seed=seed)
        _, rho = evaluate_probe(head, test_data)
        probe_rhos.append(rho)
        probes_list.append(head)
        print(f"    seed {seed}: pcorr(test) = {rho:.4f}", flush=True)

    probe_rhos = np.array(probe_rhos, dtype=float)

    anchor_output = {
        "model": MODEL_ID,
        "n_params_b": round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "peak_layer_final": int(peak_layer),
        "peak_layer_frac": round(peak_layer / (n_layers - 1), 2),
        "protocol": {
            "sweep_candidates": SWEEP_CANDIDATES,
            "sweep_ex_per_dim": SWEEP_EX_PER_DIM,
            "target_ex_per_dim": TARGET_EX_PER_DIM,
            "anchor_seeds": anchor_seeds,
            "eval_split": "test",
            "batch_size": batch_size,
            "device": DEVICE,
            "dtype": str(model_dtype),
            "output_controlled": "not_computed_local_memory_budget",
        },
        "sweep_results": sweep_results,
        "partial_corr": {
            "mean": float(probe_rhos.mean()),
            "std": float(probe_rhos.std(ddof=1)) if len(probe_rhos) > 1 else 0.0,
            "per_seed": probe_rhos.tolist(),
        },
    }

    anchor_path = RESULTS_DIR / "mistral7b_instruct_anchor_results.json"
    if "anchor" in tasks:
        with open(anchor_path, "w") as f:
            json.dump(anchor_output, f, indent=2)
        print(f"  Saved {anchor_path}", flush=True)

    # Free anchor activations; keep probes_list (small) for downstream use.
    del train_data, test_data
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    # --- Downstream tasks ---
    # ANCHOR_SEEDS starts with DOWNSTREAM_SEEDS, so probes_list[:3] already
    # has probes trained on seeds [42, 43, 44] matching the Qwen protocol.
    probes_ds = probes_list[: len(DOWNSTREAM_SEEDS)]

    del train_batches, val_batches, test_batches
    gc.collect()

    task_map = {
        "medqa": ("medqa_selective_prediction", "GBaker/MedQA-USMLE-4-options"),
        "rag": ("rag_hallucination", "google-research-datasets/natural_questions"),
        "truthfulqa": ("truthfulqa_hallucination", "truthfulqa/truthful_qa"),
    }
    out_name = {
        "medqa": "mistral7b_instruct_medqa_selective_results.json",
        "rag": "mistral7b_instruct_rag_hallucination_results.json",
        "truthfulqa": "mistral7b_instruct_truthfulqa_hallucination_results.json",
    }

    for t in ["medqa", "rag", "truthfulqa"]:
        if t not in tasks:
            continue
        try:
            task_name, dataset_name = task_map[t]
            summary = run_downstream_task(
                model, tokenizer, probes_ds, peak_layer, task_name, dataset_name, max_questions
            )
            out_path = RESULTS_DIR / out_name[t]
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved {out_path}", flush=True)
        except Exception as e:
            print(f"  {t} failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            import traceback

            traceback.print_exc(file=sys.stderr)

    print(f"\n=== Done [{elapsed_str()}] ===", flush=True)


if __name__ == "__main__":
    main()
