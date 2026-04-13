"""Figure 2: Layer-wise partial correlation at matched 3B scale.

Qwen 3B vs Llama 3.2 3B, showing where signal forms (or doesn't)
through the forward pass.

Usage: cd nn-observability && uv run python figures/fig_layer_profiles.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from style import PALETTE, RESULTS_DIR, apply_style, save_fig

OUTPUT_NAME = "layer_profiles.pdf"

# Result files for the two 3B models
QWEN_FILE = "qwen3b_v3_results.json"
LLAMA_FILE = "llama3b_v2_results.json"


def load_profile(fname: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load layer profile, return (fractions, values, n_layers)."""
    d = json.loads((RESULTS_DIR / fname).read_text())
    n_layers = d["n_layers"]
    profile = d["layer_profile"]
    layers = sorted(int(k) for k in profile)
    fracs = np.array([l / (n_layers - 1) for l in layers])
    vals = np.array([profile[str(l)] for l in layers])
    return fracs, vals, n_layers


def main():
    apply_style()

    qwen_frac, qwen_val, qwen_n = load_profile(QWEN_FILE)
    llama_frac, llama_val, llama_n = load_profile(LLAMA_FILE)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Line through all points, markers every 4th layer to avoid crowding
    ax.plot(
        qwen_frac * 100, qwen_val, "-o", color=PALETTE["Qwen"], markersize=3, markevery=4, label="Qwen 2.5 3B"
    )
    ax.plot(
        llama_frac * 100,
        llama_val,
        "--D",
        color=PALETTE["Llama"],
        markersize=3,
        markevery=3,
        label="Llama 3.2 3B",
    )

    ax.set_xlabel(r"Layer depth (\%)", fontsize=9)
    ax.set_ylabel(r"$\rho_{\mathrm{partial}}$", fontsize=9)
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=8)
    ax.set_ylim(-0.02, 0.35)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.legend(
        loc="upper left", fontsize=8, framealpha=0.9, handlelength=1.5, handletextpad=0.4, borderpad=0.3
    )
    ax.grid(True, alpha=0.2)

    save_fig(fig, OUTPUT_NAME)


if __name__ == "__main__":
    main()
