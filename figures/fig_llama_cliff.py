"""Figure 3: The Llama cliff -- three models, one family, different architectures.

Shows layer-wise pcorr for Llama 1B (signal present, peaks at 81%),
Llama 3B (signal absent), and Llama 8B (signal absent). The cliff
from 1B to 3B is the within-family evidence that architectural
configuration, not family identity, determines observability.

Usage: cd nn-observability && uv run python figures/fig_llama_cliff.py
"""

import json

import matplotlib.pyplot as plt
from style import PALETTE, RESULTS_DIR, apply_style, save_fig

OUTPUT_NAME = "llama_cliff.pdf"


def main():
    apply_style()

    models = [
        ("results/llama1b_results.json", "Llama 1B (16L, 2048d)", "-o"),
        ("results/llama3b_v3_results.json", "Llama 3B (28L, 3072d)", "--D"),
        ("results/llama8b_results.json", "Llama 8B (32L, 4096d)", ":s"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    color = PALETTE["Llama"]

    for fpath, label, linestyle in models:
        d = json.loads((RESULTS_DIR.parent / fpath).read_text())
        lp = d["layer_profile"]
        n_layers = len(lp)
        layers = sorted(lp.keys(), key=int)
        depths = [int(l) / n_layers * 100 for l in layers]
        values = [lp[l] for l in layers]

        marker = linestyle[-1]
        style = linestyle[:-1] if len(linestyle) > 1 else "-"

        # Thin out markers to avoid crowding
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
            alpha=0.9,
        )

    ax.set_xlabel("Depth (\\% of layers)")
    ax.set_ylabel(r"$\rho_{\mathrm{partial}}$", fontsize=9)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.02, 0.32)

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
