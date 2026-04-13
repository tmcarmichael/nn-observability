"""Figure 1: Cross-family scaling (pcorr vs parameters).

One point per model, colored by family. Shows Qwen flat at ~+0.25,
GPT-2 flat at ~+0.29, Gemma high, Llama declining from 1B to 3B/8B.
Preliminary points (3-seed, lower ex/dim) use open markers.
Black error bars drawn in front of markers for visibility.

Usage: cd nn-observability && uv run python figures/fig_cross_family.py
"""

import matplotlib.pyplot as plt
import numpy as np
from style import MARKERS, PALETTE, apply_style, save_fig  # must import first (sets up sys.path)
from load_results import load_all_models

OUTPUT_NAME = "cross_family_scaling.pdf"

# Preliminary Llama points (3-seed, lower ex/dim) not in load_results
PRELIMINARY = [
    {
        "label": "Llama-1B",
        "family": "Llama",
        "params_b": 1.236,
        "pcorr_mean": 0.250,
        "pcorr_std": 0.002,
        "source": "cross_family.json (3 seeds, 150 ex/dim)",
    },
    {
        "label": "Llama-8B",
        "family": "Llama",
        "params_b": 8.0,
        "pcorr_mean": 0.088,
        "pcorr_std": 0.004,
        "source": "llama8b_comprehensive.json (3 seeds)",
    },
]


def main():
    apply_style()
    models = load_all_models(verbose=True)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    # Collect all points for two-pass drawing: markers first, error bars on top
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
                "preliminary": False,
            }
        )

    for p in PRELIMINARY:
        points.append(
            {
                "x": p["params_b"],
                "y": p["pcorr_mean"],
                "yerr": p["pcorr_std"],
                "family": p["family"],
                "preliminary": True,
            }
        )

    # Pass 1: colored markers (behind)
    plotted_families = set()
    for p in points:
        fam = p["family"]
        legend_label = fam if fam not in plotted_families else None
        mkw = {}
        if p["preliminary"]:
            mkw = {"markerfacecolor": "white", "markeredgewidth": 1.0}
        ax.plot(
            p["x"],
            p["y"],
            MARKERS[fam],
            color=PALETTE[fam],
            markersize=5,
            zorder=3,
            label=legend_label,
            **mkw,
        )
        plotted_families.add(fam)

    # Pass 2: black error bars (in front)
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
    ax.set_ylabel(r"$\rho_{\mathrm{partial}}$")
    ax.set_ylim(-0.02, 0.45)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-", alpha=0.3)

    # Noise floor: below +0.15, signal is unreliable (ex/dim detection
    # threshold from Section 6)
    ax.axhspan(0, 0.15, color="gray", alpha=0.06, zorder=0)
    ax.axhline(0.15, color="gray", linewidth=0.6, linestyle=":", alpha=0.4)

    ax.legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        handlelength=1.2,
        handletextpad=0.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    ax.grid(True, alpha=0.2)

    save_fig(fig, OUTPUT_NAME)


if __name__ == "__main__":
    main()
