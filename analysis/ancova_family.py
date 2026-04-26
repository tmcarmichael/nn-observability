"""ANCOVA: does architecture family predict observability after controlling for scale?

Supplementary analysis. The meta-regression (meta_regression.py) is the
primary test because it handles the nested data structure correctly.
"""

from __future__ import annotations

import sys

import numpy as np

from analysis.load_results import load_all_models, load_per_seed


def run_ancova() -> None:
    """Fit family + log10(params) ANCOVA on per-seed observations and print results.

    Reported p-values are anticonservative (per-seed observations are not
    independent; meta_regression.py is the primary test).
    """
    try:
        import pandas as pd
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        print("ERROR: needs pandas and statsmodels.")
        print("  uv pip install pandas statsmodels")
        sys.exit(1)

    rows = load_per_seed()
    if len(rows) < 5:
        print(f"Only {len(rows)} observations.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=["family", "model", "params_b", "seed_idx", "partial_corr"])
    df["log_params"] = np.log10(df["params_b"])

    print(f"Loaded {len(df)} observations from {df['family'].nunique()} families")
    # Report missing
    load_all_models(verbose=True)
    print()

    print("=== Data summary ===")
    for fam in sorted(df["family"].unique()):
        sub = df[df["family"] == fam]
        print(
            f"  {fam:<8}: {len(sub)} obs, {sub['model'].nunique()} models, "
            f"mean pcorr = {sub['partial_corr'].mean():+.4f}"
        )
    print()

    # Model comparison
    m1 = ols("partial_corr ~ 1", data=df).fit()
    m2 = ols("partial_corr ~ log_params", data=df).fit()
    m3 = ols("partial_corr ~ log_params + C(family)", data=df).fit()
    m4 = ols("partial_corr ~ log_params * C(family)", data=df).fit()

    print("=== Model comparison (BIC) ===")
    for label, model in [
        ("Intercept only", m1),
        ("+ log(params)", m2),
        ("+ family", m3),
        ("+ family x scale", m4),
    ]:
        print(f"  {label:<22}: BIC = {model.bic:.1f}, R² = {model.rsquared:.3f}")
    print()

    print("=== Type II ANOVA (scale + family) ===")
    print(anova_lm(m3, typ=2).to_string())
    print()

    print("=== Coefficients (scale + family + interaction) ===")
    print(m4.summary().tables[1].as_text())
    print()

    # Scale-matched contrast: Qwen 3B vs Llama 3B
    qwen_3b = df[(df["model"] == "Qwen-3B")]
    llama_3b = df[(df["model"] == "Llama-3B")]
    if len(qwen_3b) > 0 and len(llama_3b) > 0:
        from scipy.stats import mannwhitneyu

        u, p = mannwhitneyu(qwen_3b["partial_corr"], llama_3b["partial_corr"], alternative="greater")
        print("=== Scale-matched contrast: Qwen 3B vs Llama 3B ===")
        print(f"  Qwen 3B:  {qwen_3b['partial_corr'].mean():+.4f} (n={len(qwen_3b)})")
        print(f"  Llama 3B: {llama_3b['partial_corr'].mean():+.4f} (n={len(llama_3b)})")
        print(f"  Mann-Whitney U = {u:.0f}, p = {p:.6f} (one-sided)")
    else:
        missing = []
        if len(qwen_3b) == 0:
            missing.append("Qwen-3B")
        if len(llama_3b) == 0:
            missing.append("Llama-3B")
        print(f"  [Scale-matched contrast skipped: {', '.join(missing)} not loaded]")

    print()
    print("=== Note ===")
    print("  This ANCOVA is supplementary. Per-seed observations are not")
    print("  independent (shared data, model, layer). The meta-regression")
    print("  (meta_regression.py) with random effects per model is the")
    print("  primary analysis.")


if __name__ == "__main__":
    run_ancova()
