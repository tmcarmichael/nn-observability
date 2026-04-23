"""
Observer-guided selective prediction on TriviaQA.

Tests whether the observer signal translates to operational value on a
downstream QA task. Generates answers via greedy decoding on Qwen 7B
Instruct, collects per-token observer scores at peak layer, aggregates
per question, and builds coverage-accuracy curves comparing observer-guided,
confidence-guided, and combined abstention.

Usage:
    uv run --extra transformer src/selective_prediction.py --device cuda
    uv run --extra transformer src/selective_prediction.py --observer-source both --seeds 3
"""

from __future__ import annotations

import argparse
import gc
import re
import string
import time

import numpy as np
from scipy.integrate import trapezoid

from utils import _save_results, bootstrap_ci

# ---------------------------------------------------------------------------
# TriviaQA data loading
# ---------------------------------------------------------------------------


def load_triviaqa(split="validation", max_questions=2000):
    """Load TriviaQA questions with reference answers.

    Uses the 'rc.nocontext' config to avoid downloading full evidence documents.
    Returns list of dicts with 'question' and 'answers' keys.
    """
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    questions = []
    for row in ds:
        if len(questions) >= max_questions:
            break
        answers = row["answer"]["aliases"] + [row["answer"]["value"]]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for a in answers:
            norm = a.strip().lower()
            if norm not in seen:
                seen.add(norm)
                unique.append(a.strip())
        questions.append({"question": row["question"], "answers": unique})
    return questions


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------


def normalize_answer(s):
    """Standard TriviaQA normalization: lowercase, strip articles/punctuation/whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match(prediction, references):
    """Check if normalized prediction matches any normalized reference."""
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(ref) == norm_pred for ref in references)


# ---------------------------------------------------------------------------
# Generation with observer scoring
# ---------------------------------------------------------------------------


def format_qa_prompt(question, tokenizer):
    """Format a question as a QA prompt using the model's chat template."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": f"Answer the following question in as few words as possible.\n\nQuestion: {question}\nAnswer:",
            },
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback for models without chat template
    return f"Question: {question}\nAnswer:"


def greedy_decode_with_scores(model, tokenizer, observer_head, peak_layer, prompt, device, max_new_tokens=64):
    """Greedy decode with per-token observer scores and confidence, using KV caching.

    Returns dict with answer_text, per-token observer_scores and confidences,
    and aggregated mean/max values for both.
    """
    import torch
    import torch.nn.functional as F

    model.eval()
    observer_head.eval()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    prompt_len = input_ids.size(1)
    observer_scores = []
    confidences = []
    past_key_values = None

    with torch.inference_mode():
        for step in range(max_new_tokens):
            input_chunk = input_ids if past_key_values is None else input_ids[:, -1:]
            outputs = model(
                input_chunk,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

            # Hidden state at peak layer, last position
            h = outputs.hidden_states[peak_layer + 1][0, -1:, :].cpu().float()
            score = observer_head(h).squeeze().item()

            # Next token (greedy)
            logits = outputs.logits[0, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            confidence = F.softmax(logits, dim=-1).max().item()

            observer_scores.append(score)
            confidences.append(confidence)

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # Stop on EOS or newline (common stopping point for QA)
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
        "observer_scores": obs_arr,
        "confidences": conf_arr,
        "mean_observer": float(obs_arr.mean()),
        "max_observer": float(obs_arr.max()),
        "mean_confidence": float(conf_arr.mean()),
        "min_confidence": float(conf_arr.min()),
        "n_tokens": len(observer_scores),
    }


# ---------------------------------------------------------------------------
# Coverage-accuracy curves
# ---------------------------------------------------------------------------


def build_coverage_curves(per_question_results, coverage_levels=None):
    """Build coverage-accuracy curves for observer, confidence, and combined abstention.

    At each coverage level, keep the most "trustworthy" questions (lowest observer
    score or highest confidence) and compute accuracy on the kept set.

    Returns dict with per-strategy accuracy arrays and AUACC (area under
    accuracy-coverage curve).
    """
    if coverage_levels is None:
        coverage_levels = list(np.arange(1.0, 0.49, -0.05))

    n = len(per_question_results)
    correct = np.array([r["correct"] for r in per_question_results])
    base_accuracy = float(correct.mean())

    # Scoring: lower observer score = more trustworthy; higher confidence = more trustworthy
    obs_scores = np.array([r["mean_observer"] for r in per_question_results])
    obs_max_scores = np.array([r["max_observer"] for r in per_question_results])
    conf_scores = np.array([r["mean_confidence"] for r in per_question_results])
    conf_min_scores = np.array([r["min_confidence"] for r in per_question_results])

    # Sort indices for each strategy
    obs_order = np.argsort(obs_scores)  # ascending: keep low-score questions
    obs_max_order = np.argsort(obs_max_scores)
    conf_order = np.argsort(-conf_scores)  # descending: keep high-confidence questions
    conf_min_order = np.argsort(-conf_min_scores)

    strategies = {
        "observer_mean": obs_order,
        "observer_max": obs_max_order,
        "confidence_mean": conf_order,
        "confidence_min": conf_min_order,
    }

    result = {"coverage_levels": coverage_levels, "base_accuracy": base_accuracy, "n_questions": n}

    for name, order in strategies.items():
        accuracies = []
        for cov in coverage_levels:
            k = max(1, int(n * cov))
            kept = order[:k]
            acc = float(correct[kept].mean()) if len(kept) > 0 else 0.0
            accuracies.append(acc)
        # AUACC: trapezoidal integration of accuracy over coverage
        auacc = float(trapezoid(accuracies, coverage_levels))
        result[name] = {"accuracy": accuracies, "auacc": auacc}

    # Combined: flag if either observer or confidence flags
    # At each coverage, keep questions not flagged by either strategy
    obs_combined = []
    for cov in coverage_levels:
        # Each channel gets half the flag budget
        flag_budget = 1.0 - cov
        obs_flag_k = max(0, int(n * flag_budget))
        conf_flag_k = max(0, int(n * flag_budget))
        # Flag worst by observer (highest scores) and worst by confidence (lowest confidence)
        obs_flagged = set(np.argsort(-obs_scores)[:obs_flag_k])
        conf_flagged = set(np.argsort(conf_scores)[:conf_flag_k])
        combined_flagged = obs_flagged | conf_flagged
        kept = [i for i in range(n) if i not in combined_flagged]
        acc = float(correct[kept].mean()) if kept else 0.0
        obs_combined.append(acc)
    combined_auacc = float(trapezoid(obs_combined, coverage_levels))
    result["combined"] = {"accuracy": obs_combined, "auacc": combined_auacc}

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_selective_prediction(
    model,
    tokenizer,
    observer_head,
    peak_layer,
    questions,
    device,
    max_new_tokens=64,
    progress_interval=50,
):
    """Run generation + scoring on all questions. Returns per-question results."""
    results = []
    for i, q in enumerate(questions):
        prompt = format_qa_prompt(q["question"], tokenizer)
        gen = greedy_decode_with_scores(
            model,
            tokenizer,
            observer_head,
            peak_layer,
            prompt,
            device,
            max_new_tokens,
        )
        is_correct = exact_match(gen["answer_text"], q["answers"])
        results.append(
            {
                "question": q["question"],
                "answer": gen["answer_text"],
                "correct": is_correct,
                "mean_observer": gen["mean_observer"],
                "max_observer": gen["max_observer"],
                "mean_confidence": gen["mean_confidence"],
                "min_confidence": gen["min_confidence"],
                "n_tokens": gen["n_tokens"],
            }
        )
        if (i + 1) % progress_interval == 0:
            running_acc = np.mean([r["correct"] for r in results])
            print(f"    {i + 1}/{len(questions)}  running EM={running_acc:.3f}")

    return results


def train_observer_for_source(model, tokenizer, device, peak_layer, source, seed, questions=None):
    """Train an observer head from the specified source data."""
    import torch

    from probe import load_wikitext, train_linear_binary
    from transformer_observe import collect_layer_data

    if source == "wikitext":
        print(f"    Training observer on WikiText-103 (seed {seed})...")
        train_docs = load_wikitext("train", max_docs=2000)
        hidden_dim = model.config.hidden_size
        min_tokens = 150 * hidden_dim
        max_tokens = max(min_tokens, int(200000 * (768 / hidden_dim)))
        train_data = collect_layer_data(model, tokenizer, train_docs, peak_layer, device, max_tokens)
    elif source == "triviaqa":
        print(f"    Training observer on TriviaQA prompts (seed {seed})...")
        assert questions is not None, "Need questions for TriviaQA source"
        # Format questions as documents for collect_layer_data
        train_questions = load_triviaqa("train", max_questions=5000)
        docs = [format_qa_prompt(q["question"], tokenizer) for q in train_questions]
        hidden_dim = model.config.hidden_size
        min_tokens = 150 * hidden_dim
        max_tokens = max(min_tokens, int(200000 * (768 / hidden_dim)))
        train_data = collect_layer_data(model, tokenizer, docs, peak_layer, device, max_tokens)
    else:
        raise ValueError(f"Unknown observer source: {source}")

    head = train_linear_binary(train_data, seed=seed)
    head.eval()

    # Free training data
    del train_data
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return head


def main():
    import torch

    P = argparse.ArgumentParser(description="Observer-guided selective prediction on TriviaQA")
    P.add_argument("--device", default="auto")
    P.add_argument("--seeds", type=int, default=3, help="Number of observer head seeds")
    P.add_argument("--max-questions", type=int, default=2000)
    P.add_argument("--max-new-tokens", type=int, default=64)
    P.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    P.add_argument("--peak-layer", type=int, default=None, help="Override peak layer (default: from results)")
    P.add_argument(
        "--observer-source",
        default="both",
        choices=["wikitext", "triviaqa", "both"],
        help="Training data for observer head",
    )
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    seeds = list(range(42, 42 + a.seeds))

    print("=" * 60)
    print("  Selective prediction on TriviaQA")
    print(f"  Model: {a.model}")
    print(f"  Device: {a.device}  Seeds: {seeds}")
    print(f"  Observer source: {a.observer_source}")
    print(f"  Max questions: {a.max_questions}")
    print("=" * 60)

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading {a.model}...")
    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if a.device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True, torch_dtype=dtype).to(
        a.device
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.1f}B params, {n_layers} layers, hidden dim {hidden_dim}")

    # Determine peak layer
    if a.peak_layer is not None:
        peak_layer = a.peak_layer
    elif "Instruct" in a.model or "instruct" in a.model:
        peak_layer = 14  # Qwen 7B Instruct default
    else:
        peak_layer = 18  # Qwen 7B base default
    print(f"  Peak layer: {peak_layer}")

    # Load TriviaQA
    print(f"\nLoading TriviaQA validation ({a.max_questions} questions)...")
    questions = load_triviaqa("validation", max_questions=a.max_questions)
    print(f"  Loaded {len(questions)} questions")

    t0 = time.time()
    sources = ["wikitext", "triviaqa"] if a.observer_source == "both" else [a.observer_source]

    for source in sources:
        print(f"\n{'=' * 60}")
        print(f"  Observer source: {source}")
        print(f"{'=' * 60}")

        all_seed_results = []
        for seed in seeds:
            print(f"\n  --- Seed {seed} ---")

            # Train observer
            head = train_observer_for_source(
                model,
                tokenizer,
                a.device,
                peak_layer,
                source,
                seed,
                questions,
            )

            # Generate and score
            print(f"    Generating answers for {len(questions)} questions...")
            per_q = run_selective_prediction(
                model,
                tokenizer,
                head,
                peak_layer,
                questions,
                a.device,
                a.max_new_tokens,
            )

            # Build curves
            curves = build_coverage_curves(per_q)
            base_acc = curves["base_accuracy"]
            print(f"    Base EM accuracy: {base_acc:.3f}")
            print(
                f"    AUACC  observer_mean={curves['observer_mean']['auacc']:.4f}"
                f"  confidence_mean={curves['confidence_mean']['auacc']:.4f}"
                f"  combined={curves['combined']['auacc']:.4f}"
            )

            # Accuracy at 90% coverage
            cov_idx = curves["coverage_levels"].index(0.9) if 0.9 in curves["coverage_levels"] else None
            if cov_idx is not None:
                print(
                    f"    Acc@90% coverage:"
                    f"  obs={curves['observer_mean']['accuracy'][cov_idx]:.3f}"
                    f"  conf={curves['confidence_mean']['accuracy'][cov_idx]:.3f}"
                    f"  combined={curves['combined']['accuracy'][cov_idx]:.3f}"
                )

            all_seed_results.append(
                {
                    "seed": seed,
                    "base_accuracy": base_acc,
                    "n_questions": len(per_q),
                    "n_correct": int(sum(r["correct"] for r in per_q)),
                    "coverage_levels": curves["coverage_levels"],
                    "observer_mean": curves["observer_mean"],
                    "observer_max": curves["observer_max"],
                    "confidence_mean": curves["confidence_mean"],
                    "confidence_min": curves["confidence_min"],
                    "combined": curves["combined"],
                }
            )

            del head
            gc.collect()
            if a.device == "cuda":
                torch.cuda.empty_cache()

        # Aggregate across seeds
        summary = {}
        for strategy in ["observer_mean", "observer_max", "confidence_mean", "confidence_min", "combined"]:
            auaccs = [s[strategy]["auacc"] for s in all_seed_results]
            summary[f"{strategy}_auacc"] = {
                "mean": float(np.mean(auaccs)),
                "per_seed": auaccs,
                "ci_95": list(bootstrap_ci(auaccs)) if len(auaccs) > 1 else auaccs,
            }

        # Accuracy at 90% coverage
        acc_90 = {}
        for strategy in ["observer_mean", "confidence_mean", "combined"]:
            accs = []
            for s in all_seed_results:
                if 0.9 in s["coverage_levels"]:
                    idx = s["coverage_levels"].index(0.9)
                    accs.append(s[strategy]["accuracy"][idx])
            if accs:
                acc_90[strategy] = {"mean": float(np.mean(accs)), "per_seed": accs}
        summary["accuracy_at_90_coverage"] = acc_90

        result = {
            "model": a.model,
            "peak_layer": peak_layer,
            "n_questions": len(questions),
            "base_em_accuracy": float(np.mean([s["base_accuracy"] for s in all_seed_results])),
            "observer_source": source,
            "per_seed": all_seed_results,
            "summary": summary,
        }

        filename = (
            "selective_prediction.json" if source == "wikitext" else "selective_prediction_indomain.json"
        )
        _save_results({source: result}, filename=filename)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Cleanup
    del model, tokenizer
    gc.collect()
    if a.device == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
