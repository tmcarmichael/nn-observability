"""Figure 1: Cross-family scaling (pcorr vs parameters).

One point per model, colored by family. Shows Qwen flat at ~+0.25,
GPT-2 flat at ~+0.29, Gemma high, Llama cliff from 1B to 3B/8B.
Preliminary points (3-seed, lower ex/dim) use open markers.
Black error bars drawn in front of markers for visibility.

Usage: cd nn-observability && uv run python figures/fig_cross_family.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from style import MARKERS, PALETTE, RESULTS_DIR, apply_style, save_fig
from load_results import load_all_models

OUTPUT_NAME = "cross_family_scaling.pdf"

# Points not in load_results scope but in the figure.
# Llama 1B: full protocol but excluded from family-level stats
# (different architecture from 3B, inflates within-family variance).
# Llama 8B: excluded from family-level stats (see load_results.py).
EXTRA_POINTS = []

# Llama 1B: load from committed full-protocol JSON if available
_llama1b_path = RESULTS_DIR / "llama1b_results.json"
if _llama1b_path.exists():
    _d = json.loads(_llama1b_path.read_text())
    EXTRA_POINTS.append(
        {
            "label": "Llama-1B",
            "family": "Llama",
            "params_b": _d["n_params_b"],
            "pcorr_mean": _d["partial_corr"]["mean"],
            "pcorr_std": _d["partial_corr"].get("std", 0),
            "preliminary": False,
        }
    )
else:
    EXTRA_POINTS.append(
        {
            "label": "Llama-1B",
            "family": "Llama",
            "params_b": 1.236,
            "pcorr_mean": 0.250,
            "pcorr_std": 0.002,
            "preliminary": True,
        }
    )

# Llama 8B: load from committed JSON if available
_llama8b_path = RESULTS_DIR / "llama8b_results.json"
if _llama8b_path.exists():
    _d = json.loads(_llama8b_path.read_text())
    EXTRA_POINTS.append(
        {
            "label": "Llama-8B",
            "family": "Llama",
            "params_b": _d["n_params_b"],
            "pcorr_mean": _d["partial_corr"]["mean"],
            "pcorr_std": _d["partial_corr"].get("std", 0),
            "preliminary": False,
        }
    )
else:
    EXTRA_POINTS.append(
        {
            "label": "Llama-8B",
            "family": "Llama",
            "params_b": 8.0,
            "pcorr_mean": 0.088,
            "pcorr_std": 0.004,
            "preliminary": True,
        }
    )


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
                "preliminary": False,
            }
        )

    for p in EXTRA_POINTS:
        points.append(
            {
                "x": p["params_b"],
                "y": p["pcorr_mean"],
                "yerr": p["pcorr_std"],
                "family": p["family"],
                "preliminary": p["preliminary"],
            }
        )

    # Pass 1: colored markers
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

    # Pass 2: black error bars in front
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
    ax.set_ylabel(r"$\rho_{\mathrm{partial}}$ (confidence-controlled)", fontsize=8)
    ax.set_ylim(-0.02, 0.45)
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
