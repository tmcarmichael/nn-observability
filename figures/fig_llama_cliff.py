"""Figure 3: The Llama cliff -- three models, one family, different architectures.

Shows layer-wise pcorr for Llama 1B (signal present, peaks at 81%),
Llama 3B (signal absent), and Llama 8B (signal absent). The cliff
from 1B to 3B is the within-family evidence that architectural
configuration, not family identity, determines observability.

Usage: cd nn-observability && uv run python figures/fig_llama_cliff.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from style import PALETTE, PCORR_YLIM, RESULTS_DIR, apply_style, save_fig

OUTPUT_NAME = "llama_cliff.pdf"

# Distinct colors: vermillion for 1B (signal present), brown and gold for 3B/8B (absent)
LLAMA_COLORS = {
    "1B": PALETTE["Llama"],  # vermillion
    "3B": "#8C564B",  # brown
    "8B": "#E69F00",  # gold
}


def main():
    apply_style()

    models = [
        ("llama1b_results.json", "Llama 1B (16L, 2048d)", "-o", LLAMA_COLORS["1B"]),
        ("llama3b_v3_results.json", "Llama 3B (28L, 3072d)", "--D", LLAMA_COLORS["3B"]),
        ("llama8b_results.json", "Llama 8B (32L, 4096d)", ":s", LLAMA_COLORS["8B"]),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for fname, label, linestyle, color in models:
        d = json.loads((RESULTS_DIR / fname).read_text())
        lp = d["layer_profile"]
        n_layers = d["n_layers"]
        layers = sorted(lp.keys(), key=int)
        depths = [int(l) / n_layers * 100 for l in layers]
        values = [lp[l] for l in layers]

        marker = linestyle[-1]
        style = linestyle[:-1] if len(linestyle) > 1 else "-"
        mark_every = max(1, n_layers // 8)

        ax.plot(
            depths,
            values,
            style,
            color=color,
            marker=marker,
            markersize=4,
            markevery=mark_every,
            label=label,
            linewidth=1.8,
        )

    # Error bar at 1B peak layer from 7-seed eval
    d1b = json.loads((RESULTS_DIR / "llama1b_results.json").read_text())
    peak = d1b["peak_layer_final"]
    peak_depth = peak / d1b["n_layers"] * 100
    per_seed = d1b["partial_corr"].get("per_seed", [])
    if len(per_seed) > 1:
        ax.errorbar(
            peak_depth,
            np.mean(per_seed),
            yerr=np.std(per_seed),
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            capthick=0.7,
            zorder=5,
        )

    ax.set_xlabel(r"Depth (\% of layers)")
    ax.set_ylabel(r"$\rho_{\mathrm{partial}}$")
    ax.set_xlim(-2, 102)
    ax.set_ylim(*PCORR_YLIM)

    # Detection threshold band
    ax.axhspan(0, 0.15, color="gray", alpha=0.06, zorder=0)
    ax.axhline(0.15, color="gray", linewidth=0.6, linestyle=":", alpha=0.4)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=7,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    ax.grid(True, alpha=0.2)
    fig.subplots_adjust(bottom=0.22)

    save_fig(fig, OUTPUT_NAME)


if __name__ == "__main__":
    main()
