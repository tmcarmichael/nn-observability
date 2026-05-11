"""Selective-prediction utilities.

Pure-function helpers for building coverage-accuracy curves and TriviaQA-style
answer normalization. Used by `tests/test_selective_prediction.py` and
available for downstream selective-prediction analysis. No model loading, no
dataset loading, no runtime side effects.
"""

from __future__ import annotations

import re
import string

import numpy as np
from scipy.integrate import trapezoid


def normalize_answer(s):
    """TriviaQA-style normalization: lowercase, strip articles/punctuation/whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match(prediction, references):
    """Check if normalized prediction matches any normalized reference."""
    norm_pred = normalize_answer(prediction)
    return any(normalize_answer(ref) == norm_pred for ref in references)


def build_coverage_curves(per_question_results, coverage_levels=None):
    """Build coverage-accuracy curves for observer, confidence, and combined abstention.

    At each coverage level, keep the most trustworthy questions (lowest observer
    score or highest confidence) and compute accuracy on the kept set.

    Returns dict with per-strategy accuracy arrays and AUACC (area under
    accuracy-coverage curve).
    """
    if coverage_levels is None:
        coverage_levels = list(np.arange(1.0, 0.49, -0.05))

    n = len(per_question_results)
    correct = np.array([r["correct"] for r in per_question_results])
    base_accuracy = float(correct.mean())

    obs_scores = np.array([r["mean_observer"] for r in per_question_results])
    obs_max_scores = np.array([r["max_observer"] for r in per_question_results])
    conf_scores = np.array([r["mean_confidence"] for r in per_question_results])
    conf_min_scores = np.array([r["min_confidence"] for r in per_question_results])

    obs_order = np.argsort(obs_scores)
    obs_max_order = np.argsort(obs_max_scores)
    conf_order = np.argsort(-conf_scores)
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
        auacc = float(trapezoid(accuracies, coverage_levels))
        result[name] = {"accuracy": accuracies, "auacc": auacc}

    obs_combined = []
    for cov in coverage_levels:
        flag_budget = 1.0 - cov
        obs_flag_k = max(0, int(n * flag_budget))
        conf_flag_k = max(0, int(n * flag_budget))
        obs_flagged = set(np.argsort(-obs_scores)[:obs_flag_k])
        conf_flagged = set(np.argsort(conf_scores)[:conf_flag_k])
        combined_flagged = obs_flagged | conf_flagged
        kept = [i for i in range(n) if i not in combined_flagged]
        acc = float(correct[kept].mean()) if kept else 0.0
        obs_combined.append(acc)
    combined_auacc = float(trapezoid(obs_combined, coverage_levels))
    result["combined"] = {"accuracy": obs_combined, "auacc": combined_auacc}

    return result
