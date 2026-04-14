"""Figure 4: Downstream task ceiling -- four tasks, same catch rate, zero-shot.

Shows exclusive catch rate (% of errors) at 10% and 20% flag rates
across language modeling (5 families), SQuAD 2.0 RAG, MedQA-USMLE,
and TruthfulQA. The 12-15% ceiling at 20% holds across all tasks.

Usage: cd nn-observability && uv run python figures/fig_downstream_ceiling.py
"""

import matplotlib.pyplot as plt
import numpy as np
from style import apply_style, save_fig

OUTPUT_NAME = "downstream_ceiling.pdf"

# Data from committed results
# Language modeling: range across 5 models at each flag rate
TASKS = {
    "WikiText\n(5 families)": {"10": (7.8, 11.4), "20": (12.0, 14.5)},  # range: min-max across families
    "SQuAD 2.0\nRAG": {"10": 5.9, "20": 11.8},
    "MedQA\nUSMLE": {"10": 8.8, "20": 11.6},
    "TruthfulQA": {"10": 8.8, "20": 13.5},
}


def main():
    apply_style()

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    tasks = list(TASKS.keys())
    x = np.arange(len(tasks))
    width = 0.35

    colors_10 = "#4878A8"
    colors_20 = "#C44E52"

    # 10% flag rate bars
    vals_10 = []
    errs_10 = []
    for t in tasks:
        v = TASKS[t]["10"]
        if isinstance(v, tuple):
            mid = (v[0] + v[1]) / 2
            err = (v[1] - v[0]) / 2
            vals_10.append(mid)
            errs_10.append(err)
        else:
            vals_10.append(v)
            errs_10.append(0)

    # 20% flag rate bars
    vals_20 = []
    errs_20 = []
    for t in tasks:
        v = TASKS[t]["20"]
        if isinstance(v, tuple):
            mid = (v[0] + v[1]) / 2
            err = (v[1] - v[0]) / 2
            vals_20.append(mid)
            errs_20.append(err)
        else:
            vals_20.append(v)
            errs_20.append(0)

    ax.bar(
        x - width / 2,
        vals_10,
        width,
        yerr=errs_10,
        color=colors_10,
        alpha=0.85,
        label="10\\% flag rate",
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
        label="20\\% flag rate",
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    # Ceiling band at 12-15%
    ax.axhspan(12, 15, color="gray", alpha=0.08, zorder=0)
    ax.axhline(12, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.axhline(15, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.text(len(tasks) - 0.5, 13.5, "ceiling", fontsize=7, color="gray", alpha=0.6, ha="right")

    ax.set_ylabel("Exclusive catches (\\% of errors)")
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
