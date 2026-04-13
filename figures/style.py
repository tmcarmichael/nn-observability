"""Shared figure style for the nn-observability paper.

All figure scripts import from here to keep visual consistency.
Follows the matplotlib template from nn-observability-paper/figures/README.

Usage:
    from style import apply_style, PALETTE, PAPER_FIGURES, save_fig
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Paths --
REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_FIGURES = REPO_ROOT.parent / "nn-observability-paper" / "figures"
RESULTS_DIR = REPO_ROOT / "results"

# Add analysis/ to import path so scripts can use load_results
sys.path.insert(0, str(REPO_ROOT / "analysis"))

# -- Colorblind-safe palette (Okabe-Ito derived) --
PALETTE = {
    "GPT-2": "#4878A8",
    "Qwen": "#E07B39",
    "Llama": "#C44E52",
    "Gemma": "#2CA02C",
    "Mistral": "#9467BD",
}

# Markers per family
MARKERS = {
    "GPT-2": "s",
    "Qwen": "o",
    "Llama": "D",
    "Gemma": "^",
    "Mistral": "P",
}


def apply_style():
    """Apply paper-consistent matplotlib rcParams."""
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
        }
    )


def save_fig(fig, name: str):
    """Save figure to paper figures/ directory as PDF."""
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIGURES / name
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  {out}")
