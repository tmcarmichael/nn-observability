"""Shared figure style for the nn-observability project.

Generic matplotlib styling and save helper. Okabe-Ito palette,
publication-quality rcParams, and a save_fig helper that writes
PDFs to a caller-provided output directory.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Colorblind-safe palette (Okabe-Ito)
# Source: https://jfly.uni-koeln.de/color/
PALETTE = {
    "GPT-2": "#0072B2",  # blue
    "Qwen": "#E69F00",  # orange
    "Llama": "#D55E00",  # vermillion
    "Gemma": "#009E73",  # bluish green
    "Mistral": "#CC79A7",  # reddish purple
    "Phi": "#56B4E9",  # sky blue
}

MARKERS = {
    "GPT-2": "s",
    "Qwen": "o",
    "Llama": "D",
    "Gemma": "^",
    "Mistral": "P",
    "Phi": "X",
}

LEGEND_ORDER = ["Gemma", "Mistral", "Phi", "GPT-2", "Qwen", "Llama"]

PCORR_YLIM = (-0.02, 0.45)

# Reference-line thresholds used across multiple figures.
# Detection floor: values at or below this are statistically indistinguishable
# from noise under the controlled-probe protocol.
# Healthy floor: the lower bound of the healthy-configuration cluster.
DETECTION_FLOOR = 0.15
HEALTHY_FLOOR = 0.21

# Shared styling for the detection-floor band. Open-coded across four
# generators before v3.0.0 centralization; keep in sync here.
_DETECTION_FLOOR_FILL = {"color": "#BBBBBB", "alpha": 0.14, "zorder": 0}
_DETECTION_FLOOR_LINE = {
    "color": "#888888",
    "linewidth": 0.6,
    "linestyle": (0, (2, 2)),
    "alpha": 0.7,
    "zorder": 1,
}

# Shared styling for the secondary healthy-floor reference line. Lower
# contrast than the detection floor so a reader can rank the two.
_HEALTHY_FLOOR_LINE = {
    "color": "#666666",
    "linewidth": 0.6,
    "linestyle": (0, (2, 3)),
    "alpha": 0.65,
    "zorder": 1,
}


def draw_detection_floor(ax, value: float = DETECTION_FLOOR) -> None:
    """Draw the shared detection-floor band (fill from 0 to value, dashed upper edge)."""
    ax.axhspan(0, value, **_DETECTION_FLOOR_FILL)
    ax.axhline(value, **_DETECTION_FLOOR_LINE)


def draw_healthy_floor(ax, value: float = HEALTHY_FLOOR) -> None:
    """Draw the shared healthy-floor reference line."""
    ax.axhline(value, **_HEALTHY_FLOOR_LINE)


def signed(val: float, dp: int = 3) -> str:
    """Format a float with APA/ML-paper sign convention.

    Positives render bare, negatives retain the minus sign. Used by
    figure generators that print numeric labels; pipeline generators
    (tables, data macros) mirror the same convention via f-strings.
    """
    return f"{val:.{dp}f}"


def apply_style():
    """Apply publication-quality matplotlib rcParams.

    Font family carries a fallback chain so the figures render reasonably
    in environments without TeX Live or Computer Modern installed, for
    example when arXiv HTML rerenders the figures or when a colleague
    regenerates outside the pinned environment.

    LaTeX preamble enables microtype for subliminal kerning and protrusion,
    and pins ams packages so math labels render identically across hosts.

    PDF-level knobs enable TrueType subsetting (fonttype 42), max stream
    compression (pdf.compression 9), and consistent axis-below-data order
    so grids always sit behind markers.
    """
    matplotlib.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"""
                \usepackage{amsmath,amssymb}
                \usepackage[protrusion=true,expansion=false,kerning=true]{microtype}
            """,
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "Latin Modern Roman",
                "STIX Two Text",
                "DejaVu Serif",
            ],
            "font.sans-serif": [
                "Latin Modern Sans",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "cm",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.titlepad": 6,
            "axes.labelpad": 4,
            "axes.axisbelow": True,  # grid / spans behind data markers
            "legend.fontsize": 9,
            "legend.frameon": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            "lines.markeredgewidth": 0.7,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "patch.linewidth": 0.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.3,
            "grid.color": "#888888",
            "figure.dpi": 150,
            # PDF output polish: Type 42 (TrueType) subset, max stream
            # compression. text.usetex=True paths still emit Type 1 for
            # math; these knobs harden the non-usetex fallback.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def despine(ax, top=True, right=True, left=False, bottom=False):
    """Hide axis spines (publication default: drop top + right).

    Matches the Tufte-style "remove non-data ink" practice by stripping
    the box around each axes. Remaining spines are thinned further for
    quieter frames.
    """
    for side, hide in [("top", top), ("right", right), ("left", left), ("bottom", bottom)]:
        ax.spines[side].set_visible(not hide)
    for side in ("left", "bottom"):
        if ax.spines[side].get_visible():
            ax.spines[side].set_linewidth(0.7)
            ax.spines[side].set_color("#333333")


def figure_rule(fig, y=0.985, color="#222222", linewidth=0.8, left=0.02, right=0.98, bottom=False):
    """Draw thin horizontal rule(s) across the figure.

    Default (bottom=False) draws a single rule above the suptitle,
    mimicking the Nature/Science practice of a header stroke separating
    the headline from the plot area. With bottom=True, also draws a
    lighter closing rule near the figure's bottom edge, giving the
    figure a symmetric two-rule frame that reads as a "pulled quote"
    without adding an explicit box.
    """
    import matplotlib.lines as mlines

    top_line = mlines.Line2D(
        [left, right],
        [y, y],
        transform=fig.transFigure,
        color=color,
        linewidth=linewidth,
        solid_capstyle="butt",
    )
    fig.add_artist(top_line)
    if bottom:
        bot_line = mlines.Line2D(
            [left, right],
            [0.012, 0.012],
            transform=fig.transFigure,
            color=color,
            linewidth=linewidth * 0.55,
            alpha=0.75,
            solid_capstyle="butt",
        )
        fig.add_artist(bot_line)


def panel_label(ax, letter, x=-0.02, y=1.04, fontsize=10):
    """Add a small bold letter label in the top-left of an axes."""
    ax.text(
        x,
        y,
        rf"\textbf{{{letter}}}",
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="right",
        va="bottom",
    )


def save_fig(fig, name: str, output_dir: Path) -> Path:
    """Save figure as PDF into output_dir. Caller owns the path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / name
    # Strip time-varying PDF metadata so byte-identical regeneration is possible.
    # Enables content-diff checks (figures/generate_all.py --check).
    fig.savefig(
        out,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"CreationDate": None, "ModDate": None, "Producer": None, "Creator": None},
    )
    plt.close(fig)
    print(f"  {out}")
    return out
