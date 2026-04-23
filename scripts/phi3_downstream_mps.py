"""Phi-3 Mini downstream replication.

Runs MedQA, RAG, and TruthfulQA evaluations on Phi-3 Mini using the same
protocol as the Qwen 7B Instruct downstream battery. Produces results
JSONs with a matching schema for direct cross-family comparison.
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

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
PEAK_LAYER = 17
TARGET_EX_PER_DIM = 350
SEEDS = [42, 43, 44]

RUN_START = time.time()


def elapsed_str():
    m = (time.time() - RUN_START) / 60
    return f"{m:.1f}m" if m < 60 else f"{m / 60:.2f}h"


# ---------------------------------------------------------------------------
# WikiText probe training (mirrors scripts/run_model.py and the three
# downstream task scripts so results are directly comparable).
# ---------------------------------------------------------------------------


def load_wikitext(split="train", max_docs=None):
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

        losses_2d = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction="none").view(
            B, S - 1
        )

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
            print(f"    batch {bi + 1}/{len(batches)}  tokens={total}  [{elapsed_str()}]", flush=True)

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


# ---------------------------------------------------------------------------
# Hook-based greedy decode with multi-probe scoring. One decode per question;
# all probe heads score the captured activation at the peak layer.
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
# MedQA (USMLE 4-option) - mirror scripts/medqa_selective.py
# ---------------------------------------------------------------------------


def load_medqa(max_questions):
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


def run_medqa(model, tokenizer, probes, max_questions):
    print(f"\n=== MedQA [{elapsed_str()}] ===")
    questions = load_medqa(max_questions)
    print(f"  Loaded {len(questions)} questions")

    results = []
    for qi, q in enumerate(questions):
        prompt = format_medqa_prompt(q["question"], q["options"], tokenizer)
        gen = greedy_decode_with_observer(
            model, tokenizer, probes, PEAK_LAYER, prompt, DEVICE, max_new_tokens=16
        )
        pred_letter = extract_answer_letter(gen["answer_text"])
        is_correct_q = pred_letter == q["answer_key"]

        results.append(
            {
                "question": q["question"][:100],
                "answer": gen["answer_text"],
                "pred_letter": pred_letter,
                "gold_letter": q["answer_key"],
                "correct": is_correct_q,
                "mean_observer": gen["mean_observer"],
                "mean_confidence": gen["mean_confidence"],
            }
        )

        if (qi + 1) % 50 == 0:
            n_c = sum(r["correct"] for r in results)
            print(
                f"  {qi + 1}/{len(questions)}: {n_c}/{qi + 1} ({n_c / (qi + 1) * 100:.1f}%) [{elapsed_str()}]"
            )

    return _analyze_selective(
        results,
        task="medqa_selective_prediction",
        dataset="GBaker/MedQA-USMLE-4-options",
    )


# ---------------------------------------------------------------------------
# RAG on Natural Questions - mirror scripts/rag_hallucination.py
# ---------------------------------------------------------------------------


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


def run_rag(model, tokenizer, probes, max_questions):
    print(f"\n=== RAG (Natural Questions) [{elapsed_str()}] ===")
    questions = load_nq(max_questions)
    print(f"  Loaded {len(questions)} NQ questions with short answers")

    results = []
    for qi, q in enumerate(questions):
        prompt = format_rag_prompt(q["question"], q["context"], tokenizer)
        gen = greedy_decode_with_observer(
            model, tokenizer, probes, PEAK_LAYER, prompt, DEVICE, max_new_tokens=32
        )
        is_correct_q = rag_is_correct(gen["answer_text"], q["answers"])

        results.append(
            {
                "question": q["question"][:100],
                "answer": gen["answer_text"],
                "gold_answers": q["answers"][:5],
                "correct": is_correct_q,
                "mean_observer": gen["mean_observer"],
                "mean_confidence": gen["mean_confidence"],
            }
        )

        if (qi + 1) % 50 == 0:
            n_c = sum(r["correct"] for r in results)
            print(
                f"  {qi + 1}/{len(questions)}: {n_c}/{qi + 1} ({n_c / (qi + 1) * 100:.1f}%) [{elapsed_str()}]"
            )

    return _analyze_selective(
        results,
        task="rag_hallucination",
        dataset="google-research-datasets/natural_questions",
    )


# ---------------------------------------------------------------------------
# TruthfulQA - mirror scripts/truthfulqa_hallucination.py
# ---------------------------------------------------------------------------


def load_truthfulqa():
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    questions = []
    for row in ds:
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]
        correct_idx = labels.index(1)
        correct_answer = choices[correct_idx]
        incorrect_answers = [c for c, l in zip(choices, labels) if l == 0]
        questions.append(
            {
                "question": row["question"],
                "correct_answer": correct_answer,
                "incorrect_answers": incorrect_answers,
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


def run_truthfulqa(model, tokenizer, probes, max_questions):
    print(f"\n=== TruthfulQA [{elapsed_str()}] ===")
    questions = load_truthfulqa()
    if max_questions:
        questions = questions[:max_questions]
    print(f"  Loaded {len(questions)} TruthfulQA questions")

    results = []
    for qi, q in enumerate(questions):
        messages = [
            {
                "role": "user",
                "content": f"Answer the following question truthfully and concisely.\n\nQuestion: {q['question']}\nAnswer:",
            }
        ]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Question: {q['question']}\nAnswer:"

        gen = greedy_decode_with_observer(
            model, tokenizer, probes, PEAK_LAYER, prompt, DEVICE, max_new_tokens=64
        )
        is_correct_q = truthfulqa_is_correct(gen["answer_text"], q["correct_answer"], q["incorrect_answers"])

        results.append(
            {
                "question": q["question"],
                "answer": gen["answer_text"],
                "correct_answer": q["correct_answer"],
                "correct": is_correct_q,
                "category": q["category"],
                "mean_observer": gen["mean_observer"],
                "mean_confidence": gen["mean_confidence"],
                "min_confidence": gen["min_confidence"],
                "n_tokens": gen["n_tokens"],
            }
        )

        if (qi + 1) % 100 == 0:
            n_c = sum(r["correct"] for r in results)
            print(
                f"  {qi + 1}/{len(questions)}: {n_c}/{qi + 1} ({n_c / (qi + 1) * 100:.1f}%) [{elapsed_str()}]"
            )

    summary = _analyze_selective(results, task="truthfulqa_hallucination", dataset="truthfulqa/truthful_qa")

    conf_threshold = 0.9
    n = len(results)
    correct_arr = np.array([r["correct"] for r in results])
    obs = np.array([r["mean_observer"] for r in results])
    conf = np.array([r["mean_confidence"] for r in results])
    confident_wrong = (conf > conf_threshold) & (~correct_arr)
    confident_right = (conf > conf_threshold) & correct_arr
    uncertain_wrong = (conf <= conf_threshold) & (~correct_arr)
    uncertain_right = (conf <= conf_threshold) & correct_arr

    quadrant = {
        "confident_wrong": int(confident_wrong.sum()),
        "confident_right": int(confident_right.sum()),
        "uncertain_wrong": int(uncertain_wrong.sum()),
        "uncertain_right": int(uncertain_right.sum()),
    }
    confident_catches = {}
    n_cw = quadrant["confident_wrong"]
    if n_cw > 0:
        cw_obs = obs[confident_wrong]
        for pct in [10, 20, 30, 50]:
            threshold = np.percentile(obs, 100 - pct)
            caught = int((cw_obs >= threshold).sum())
            confident_catches[f"top_{pct}_pct"] = {
                "caught": caught,
                "total": n_cw,
                "fraction": round(caught / n_cw * 100, 1),
                "threshold": round(float(threshold), 4),
            }
        if n != quadrant["confident_wrong"] and quadrant["confident_right"] > 0:
            from sklearn.metrics import roc_auc_score

            confident_mask = conf > conf_threshold
            try:
                auc = roc_auc_score(~correct_arr[confident_mask], obs[confident_mask])
                confident_catches["auc_among_confident"] = round(float(auc), 3)
            except ValueError:
                pass

    summary["confidence_threshold"] = conf_threshold
    summary["quadrant"] = quadrant
    summary["confident_hallucination_catches"] = confident_catches
    return summary


# ---------------------------------------------------------------------------
# Shared analysis: exclusive catches + coverage-accuracy
# ---------------------------------------------------------------------------


def _analyze_selective(results, task, dataset):
    n = len(results)
    correct = np.array([r["correct"] for r in results])
    obs = np.array([r["mean_observer"] for r in results])
    conf = np.array([r["mean_confidence"] for r in results])

    accuracy = float(correct.mean())
    n_errors = int((~correct).sum())
    print(f"  Accuracy: {accuracy:.3f} ({n_errors} errors / {n} questions)")

    flag_rates = [0.05, 0.10, 0.20, 0.30]
    catches = {}
    for rate in flag_rates:
        k = max(1, int(n * rate))
        obs_flagged = obs >= np.sort(obs)[-k]
        conf_flagged = conf <= np.sort(conf)[k]
        obs_exclusive = int((obs_flagged & ~conf_flagged & ~correct).sum())
        conf_exclusive = int((conf_flagged & ~obs_flagged & ~correct).sum())
        both = int((obs_flagged & conf_flagged & ~correct).sum())
        pct = obs_exclusive / n_errors * 100 if n_errors > 0 else 0
        print(
            f"    @{rate:.0%}: obs-excl={obs_exclusive} ({pct:.1f}% of errors), "
            f"conf-excl={conf_exclusive}, both={both}"
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
        auacc = float(trapezoid(accs, coverage_levels))
        catches[f"{name}_auacc"] = auacc

    return {
        "model": MODEL_ID,
        "task": task,
        "dataset": dataset,
        "peak_layer": PEAK_LAYER,
        "n_questions": n,
        "n_errors": n_errors,
        "accuracy": accuracy,
        "probe_seeds": SEEDS,
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
    parser.add_argument("--smoke", action="store_true", help="10 questions per task, small probe budget")
    parser.add_argument(
        "--tasks",
        default="medqa,rag,truthfulqa",
        help="comma-separated subset of {medqa,rag,truthfulqa}",
    )
    parser.add_argument("--max-questions", type=int, default=1000)
    parser.add_argument(
        "--ex-per-dim",
        type=int,
        default=None,
        help=f"Training examples per hidden dimension (default: {TARGET_EX_PER_DIM}; --smoke sets 50)",
    )
    parser.add_argument(
        "--peak-layer",
        type=int,
        default=None,
        help=f"Override PEAK_LAYER constant (default {PEAK_LAYER})",
    )
    args = parser.parse_args()

    max_questions = 10 if args.smoke else args.max_questions
    ex_per_dim = args.ex_per_dim or (50 if args.smoke else TARGET_EX_PER_DIM)
    global PEAK_LAYER
    if args.peak_layer is not None:
        PEAK_LAYER = args.peak_layer
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"=== Phi-3 Mini downstream replication [{elapsed_str()}] ===")
    print(f"Device: {DEVICE}  Train device: {TRAIN_DEVICE}")
    print(f"Model: {MODEL_ID}  Peak layer: {PEAK_LAYER}")
    print(f"Tasks: {tasks}  max_questions={max_questions}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.float16 if DEVICE == "mps" else torch.bfloat16
    attn_impl = "eager" if DEVICE == "mps" else "sdpa"
    print(f"Loading model dtype={model_dtype} attn={attn_impl} ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=model_dtype,
        attn_implementation=attn_impl,
    ).to(DEVICE)
    model.eval()

    hidden_dim = model.config.hidden_size
    max_train = ex_per_dim * hidden_dim
    print(f"Hidden dim: {hidden_dim}  ex/dim: {ex_per_dim}  train tokens: {max_train}", flush=True)

    # --- Train WikiText probe ---
    print(f"\n=== Training WikiText probe [{elapsed_str()}] ===")
    train_docs = load_wikitext("train", max_docs=8000)
    print(f"  WikiText docs: {len(train_docs)}")
    train_enc = pretokenize(train_docs, tokenizer)
    del train_docs
    train_batches = build_batches(train_enc, 16 if DEVICE == "mps" else 48)
    del train_enc
    print(f"  Batches: {len(train_batches)}")

    train_data = collect_single_layer(model, train_batches, PEAK_LAYER, max_train, DEVICE)
    del train_batches
    gc.collect()
    print(f"  Collected activations: {train_data['activations'].shape}  [{elapsed_str()}]")

    probes = []
    for seed in SEEDS:
        head = train_linear_binary(train_data, seed=seed)
        probes.append(head)
        print(f"  Probe seed {seed} trained")
    del train_data
    gc.collect()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "medqa" in tasks:
        try:
            summary = run_medqa(model, tokenizer, probes, max_questions)
            out = RESULTS_DIR / "phi3_mini_medqa_selective_results.json"
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved {out}")
        except Exception as e:
            print(f"  MedQA failed: {type(e).__name__}: {e}", file=sys.stderr)
            raise

    if "rag" in tasks:
        try:
            summary = run_rag(model, tokenizer, probes, max_questions)
            out = RESULTS_DIR / "phi3_mini_rag_hallucination_results.json"
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved {out}")
        except Exception as e:
            print(f"  RAG failed: {type(e).__name__}: {e}", file=sys.stderr)

    if "truthfulqa" in tasks:
        try:
            summary = run_truthfulqa(model, tokenizer, probes, max_questions)
            out = RESULTS_DIR / "phi3_mini_truthfulqa_hallucination_results.json"
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved {out}")
        except Exception as e:
            print(f"  TruthfulQA failed: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"\n=== Done [{elapsed_str()}] ===")


if __name__ == "__main__":
    main()
