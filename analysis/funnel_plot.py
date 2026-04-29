"""Funnel plot and Egger's test for publication bias.

Uses SE = std/sqrt(n_seeds) as the precision metric. SE from probe-seed
variance understates true measurement uncertainty (seeds share data,
model, and layer), so treat it as a lower bound. Egger's test has low
power with fewer than ten studies, so the results are descriptive.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist

from analysis.load_results import load_all_models

fig_dir = Path(__file__).resolve().parent


def load_model_stats() -> list[dict]:
    """Return one row per model with mean, std, n_seeds, and SE for the funnel plot.

    Drops models with fewer than 2 seeds (no per-seed variance).
    """
    models_raw = load_all_models(verbose=True)
    models = []
    for label, m in models_raw.items():
        pc = m["partial_corr"]
        per_seed = pc.get("per_seed", [])
        mean = pc.get("mean")
        if mean is None:
            continue
        if len(per_seed) < 2:
            continue  # exclude models without proper per_seed data
        std = pc.get("std", np.std(per_seed, ddof=1))
        n_seeds = len(per_seed)
        se = std / np.sqrt(n_seeds) if std > 0 else 0
        models.append(
            {
                "name": label,
                "family": m["family"],
                "params_b": m["params_b"],
                "mean": float(mean),
                "std": float(std),
                "n_seeds": n_seeds,
                "se": float(se),
            }
        )
    return models


def eggers_test(
    means: list[float],
    ses: list[float],
) -> tuple[float, float, float]:
    precision = 1.0 / np.array(ses)
    standardized = np.array(means) / np.array(ses)
    X = np.column_stack([precision, np.ones(len(precision))])
    beta, _, _, _ = np.linalg.lstsq(X, standardized, rcond=None)
    predicted = X @ beta
    residuals = standardized - predicted
    n = len(means)
    if n <= 2:
        return 0, 0, 1.0
    mse = np.sum(residuals**2) / (n - 2)
    xbar = precision.mean()
    SSx = np.sum((precision - xbar) ** 2)
    se_intercept = np.sqrt(mse * (1.0 / n + xbar**2 / SSx)) if SSx > 0 else 0
    t_stat = beta[1] / se_intercept if se_intercept > 0 else 0
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2))
    return beta[1], t_stat, p_value


def run() -> None:
    """Print funnel-plot data, run Egger's test, save funnel_plot.pdf to analysis/."""
    print()
    models = load_model_stats()
    if len(models) < 3:
        print(f"Only {len(models)} models.")
        sys.exit(1)

    print(f"\n=== Funnel plot data ({len(models)} models) ===")
    print(f"  {'Model':<15} {'Family':<8} {'Mean':>8} {'Std':>8} {'Seeds':>6} {'SE':>8} {'1/SE':>8}")
    print(f"  {'-' * 65}")
    for m in models:
        prec = 1.0 / m["se"] if m["se"] > 0 else float("inf")
        print(
            f"  {m['name']:<15} {m['family']:<8} {m['mean']:+.4f} {m['std']:.4f} "
            f"{m['n_seeds']:>6} {m['se']:.4f} {prec:>8.1f}"
        )

    valid = [m for m in models if m["se"] > 0]
    means_valid = [m["mean"] for m in valid]
    ses_valid = [m["se"] for m in valid]

    if len(valid) >= 3:
        print("\n=== Egger's test (SE-based precision) ===")
        intercept, t_stat, p_value = eggers_test(means_valid, ses_valid)
        print(f"  Intercept: {intercept:.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print("  Significant asymmetry.")
            print("  Note: Egger's confounds publication bias with genuine")
            print("  heterogeneity. Asymmetry is expected when families")
            print("  differ in signal level and does not indicate selective")
            print("  reporting.")
        else:
            print("  No significant asymmetry.")
        if len(valid) < 10:
            print(f"  Caveat: Egger's has low power with {len(valid)} studies (recommend 10+).")

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        # Okabe-Ito palette.
        colors = {
            "GPT-2": "#0072B2",
            "Qwen": "#E69F00",
            "Llama": "#D55E00",
            "Gemma": "#009E73",
            "Mistral": "#CC79A7",
            "Phi": "#56B4E9",
            "Pythia": "#F0E442",
        }
        plotted_families = set()
        for m in valid:
            label = m["family"] if m["family"] not in plotted_families else None
            ax.scatter(
                m["mean"], 1.0 / m["se"], c=colors.get(m["family"], "gray"), s=50, zorder=3, label=label
            )
            plotted_families.add(m["family"])

        ax.legend(fontsize=9)
        ax.axvline(x=np.mean(means_valid), color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$\rho_{\mathrm{partial}}$")
        ax.set_ylabel("Precision (1 / SE)")
        plt.tight_layout()
        out_path = fig_dir / "funnel_plot.pdf"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"\nSaved {out_path}")
    except ImportError:
        print("\nmatplotlib not available, skipping plot.")


if __name__ == "__main__":
    run()
