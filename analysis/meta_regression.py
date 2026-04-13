"""Three-level meta-analytic model: seeds within models within families.

Primary statistical analysis for the paper. Handles nested data
correctly via random effects per model. The ANCOVA (ancova_family.py)
is the supplementary accessible version.

Usage: cd nn-observability && uv run python analysis/meta_regression.py
"""

import sys

import numpy as np
from load_results import load_all_models, load_per_seed


def run_mixed_effects():
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        print("ERROR: needs pandas and statsmodels.")
        sys.exit(1)

    rows = load_per_seed()
    if len(rows) < 5:
        print(f"Only {len(rows)} observations.")
        sys.exit(1)

    df = pd.DataFrame(rows, columns=["family", "model", "params_b", "seed_idx", "partial_corr"])
    df["log_params"] = np.log10(df["params_b"])

    print(f"Loaded {len(df)} seed-level observations")
    load_all_models(verbose=True)
    print()

    # Flag models with n=1 (no seed variance estimable)
    model_counts = df.groupby("model").size()
    singletons = model_counts[model_counts == 1].index.tolist()
    if singletons:
        print(f"  Note: {singletons} have n=1 (no within-model variance).")
        print("  These enter the model but contribute no seed-level information.\n")

    print("=== Data summary ===")
    for fam in sorted(df["family"].unique()):
        sub = df[df["family"] == fam]
        print(
            f"  {fam:<8}: {len(sub)} obs across {sub['model'].nunique()} models, "
            f"mean = {sub['partial_corr'].mean():+.4f}"
        )
    print()

    # Mixed-effects: random intercept per model
    print("=== Mixed-effects: partial_corr ~ log(params) + family + (1|model) ===")
    try:
        md = smf.mixedlm("partial_corr ~ log_params + C(family)", df, groups=df["model"])
        mdf = md.fit(reml=True)
        print(mdf.summary())
        if mdf.cov_re.iloc[0, 0] < 1e-10:
            print("\n  Note: random-effects variance estimated at zero (boundary).")
            print("  The model collapsed to fixed-effects. This occurs when")
            print("  between-model variance is small relative to within-model")
            print("  noise. Adding families with different signal levels (Llama)")
            print("  should resolve this.")
    except Exception as e:
        print(f"  Mixed-effects failed: {e}")
        print("  Falling back to OLS with clustered standard errors.")
        md = smf.ols("partial_corr ~ log_params + C(family)", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df["model"]}
        )
        print(md.summary())

    # Variance decomposition (ddof=0 throughout for consistent partition)
    # Note: this is a descriptive decomposition, not formal REML variance components.
    # Percentages characterize the data, not inferential estimates.
    print("\n=== Variance decomposition (three-level, descriptive) ===")
    grand_mean = df["partial_corr"].mean()
    total_var = df["partial_corr"].var(ddof=0)

    family_means = df.groupby("family")["partial_corr"].mean()
    family_ns = df.groupby("family").size()
    var_between_family = sum(n * (m - grand_mean) ** 2 for m, n in zip(family_means, family_ns)) / len(df)

    model_means = df.groupby("model")["partial_corr"].mean()
    var_between_model = 0
    for model_name, model_mean in model_means.items():
        fam = df[df["model"] == model_name]["family"].iloc[0]
        fam_mean = family_means[fam]
        n = (df["model"] == model_name).sum()
        var_between_model += n * (model_mean - fam_mean) ** 2
    var_between_model /= len(df)

    var_within = 0
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        var_within += ((sub["partial_corr"] - sub["partial_corr"].mean()) ** 2).sum()
    var_within /= len(df)

    pct_fam = var_between_family / total_var * 100
    pct_model = var_between_model / total_var * 100
    pct_seed = var_within / total_var * 100

    print(f"  Total variance:          {total_var:.6f}")
    print(f"  Between families:        {var_between_family:.6f} ({pct_fam:.1f}%)")
    print(f"  Between models (w/in):   {var_between_model:.6f} ({pct_model:.1f}%)")
    print(f"  Within model (seeds):    {var_within:.6f} ({pct_seed:.1f}%)")
    print(f"  Sum check:               {pct_fam + pct_model + pct_seed:.1f}%")

    if pct_seed > 30:
        print("\n  Probe training introduces substantial measurement noise")
        print(f"  ({pct_seed:.0f}% of total variance), motivating the multi-seed")
        print("  protocol. The mixed-effects model absorbs this correctly.")


if __name__ == "__main__":
    run_mixed_effects()
