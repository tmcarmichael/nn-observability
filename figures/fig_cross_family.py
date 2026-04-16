"""Figure 1: Cross-family scaling (pcorr vs parameters).

One point per model, colored by family. Shows Qwen flat at ~+0.25,
GPT-2 flat at ~+0.29, Gemma high, Llama cliff from 1B to 3B/8B.

Usage: cd nn-observability && uv run python figures/fig_cross_family.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from style import LEGEND_ORDER, MARKERS, PALETTE, PCORR_YLIM, RESULTS_DIR, apply_style, save_fig
from load_results import load_all_models

OUTPUT_NAME = "cross_family_scaling.pdf"

# Points not in load_results stat scope but in the figure.
EXTRA_SOURCES = {
    "Llama-1B": "llama1b_results.json",
    "Llama-8B": "llama8b_results.json",
    "Gemma-4B": "gemma4b_results.json",
}


def main():
    apply_style()
    models = load_all_models(verbose=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    points = []

    for _label, m in sorted(models.items(), key=lambda x: x[1]["params_b"]):
        family = m["family"]
        per_seed = m["partial_corr"].get("per_seed", [])
        yerr = np.std(per_seed) if len(per_seed) > 1 else 0
        points.append(
            {
                "x": m["params_b"],
                "y": m["partial_corr"]["mean"],
                "yerr": yerr,
                "family": family,
            }
        )

    for label, fname in EXTRA_SOURCES.items():
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        pc = d["partial_corr"]
        family = label.split("-")[0]
        points.append(
            {
                "x": d["n_params_b"],
                "y": pc["mean"],
                "yerr": pc.get("std", np.std(pc.get("per_seed", [0]))),
                "family": family,
            }
        )

    # Plot in legend order so legend matches visual hierarchy
    plotted_families = set()
    for fam in LEGEND_ORDER:
        fam_points = [p for p in points if p["family"] == fam]
        for p in fam_points:
            legend_label = fam if fam not in plotted_families else None
            ax.plot(
                p["x"],
                p["y"],
                MARKERS[fam],
                color=PALETTE[fam],
                markersize=5,
                zorder=3,
                label=legend_label,
            )
            plotted_families.add(fam)

    # Black error bars in front
    for p in points:
        if p["yerr"] > 0:
            ax.errorbar(
                p["x"],
                p["y"],
                yerr=p["yerr"],
                fmt="none",
                ecolor="black",
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.7,
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (B)")
    ax.set_ylabel(r"$\rho_{\mathrm{partial}}$ (confidence-controlled)")
    ax.set_ylim(*PCORR_YLIM)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.3)

    # Noise floor band
    ax.axhspan(0, 0.15, color="gray", alpha=0.06, zorder=0)
    ax.axhline(0.15, color="gray", linewidth=0.6, linestyle=":", alpha=0.4)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=6,
        fontsize=6.5,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.3,
        columnspacing=0.8,
        markerscale=0.9,
    )
    ax.grid(True, alpha=0.2)
    fig.subplots_adjust(bottom=0.22)

    save_fig(fig, OUTPUT_NAME)


if __name__ == "__main__":
    main()
