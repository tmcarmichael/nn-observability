"""Figure 4: Downstream task ceiling -- four tasks, same catch rate, zero-shot.

Shows exclusive catch rate (% of errors) at 10% and 20% flag rates
across language modeling (5 families), SQuAD 2.0 RAG, MedQA-USMLE,
and TruthfulQA. The ceiling at 20% holds across all tasks.

All values read live from committed results JSONs.

Usage: cd nn-observability && uv run python figures/fig_downstream_ceiling.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from style import RESULTS_DIR, apply_style, save_fig

OUTPUT_NAME = "downstream_ceiling.pdf"


def _flagging_pct(fname, rate_key):
    """Extract exclusive catch % from a model's flagging_6a summary."""
    d = json.loads((RESULTS_DIR / fname).read_text())
    flag = d["flagging_6a"]
    summary = flag["summary"]
    n = flag["n_tokens"]
    raw = summary[rate_key]["observer_exclusive"]
    return raw / (n / 2) * 100


def main():
    apply_style()

    # WikiText: range across families with flagging data
    wikitext_models = [
        "mistral7b_results.json",
        "qwen7b_v3_results.json",
        "llama3b_v3_results.json",
    ]
    wt_10 = [_flagging_pct(f, "0.1") for f in wikitext_models]
    wt_20 = [_flagging_pct(f, "0.2") for f in wikitext_models]

    # Downstream tasks from committed JSONs
    squad = json.loads((RESULTS_DIR / "rag_hallucination_results.json").read_text())
    medqa = json.loads((RESULTS_DIR / "medqa_selective_results.json").read_text())
    tqa = json.loads((RESULTS_DIR / "truthfulqa_hallucination_results.json").read_text())

    tasks = [
        "WikiText\n(5 families)",
        "SQuAD 2.0\nRAG",
        "MedQA\nUSMLE",
        "TruthfulQA",
    ]
    x = np.arange(len(tasks))
    width = 0.35

    vals_10 = [
        np.mean(wt_10),
        squad["flag_rates"]["0.1"]["pct_of_errors"],
        medqa["flag_rates"]["0.1"]["pct_of_errors"],
        tqa["standard_catches"]["0.1"]["pct_of_errors"],
    ]
    vals_20 = [
        np.mean(wt_20),
        squad["flag_rates"]["0.2"]["pct_of_errors"],
        medqa["flag_rates"]["0.2"]["pct_of_errors"],
        tqa["standard_catches"]["0.2"]["pct_of_errors"],
    ]
    errs_10 = [(max(wt_10) - min(wt_10)) / 2, 0, 0, 0]
    errs_20 = [(max(wt_20) - min(wt_20)) / 2, 0, 0, 0]

    colors_10 = "#0072B2"  # Okabe-Ito blue
    colors_20 = "#D55E00"  # Okabe-Ito vermillion

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    ax.bar(
        x - width / 2,
        vals_10,
        width,
        yerr=errs_10,
        color=colors_10,
        alpha=0.85,
        label=r"10\% flag rate",
        capsize=3,
        error_kw={"linewidth": 0.8},
    )
    ax.bar(
        x + width / 2,
        vals_20,
        width,
        yerr=errs_20,
        color=colors_20,
        alpha=0.85,
        label=r"20\% flag rate",
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    # Ceiling band
    ceil_lo = min(vals_20)
    ceil_hi = max(vals_20)
    ax.axhspan(ceil_lo, ceil_hi, color="gray", alpha=0.08, zorder=0)
    ax.axhline(ceil_lo, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.axhline(ceil_hi, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.text(
        len(tasks) - 0.5, (ceil_lo + ceil_hi) / 2, "ceiling", fontsize=7, color="gray", alpha=0.6, ha="right"
    )

    ax.set_ylabel(r"Exclusive catches (\% of errors)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=8)
    ax.set_ylim(0, 18)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=8,
        frameon=False,
    )
    ax.grid(True, axis="y", alpha=0.2)
    fig.subplots_adjust(bottom=0.25)

    save_fig(fig, OUTPUT_NAME)


if __name__ == "__main__":
    main()
