"""Permutation test for the family effect on observability.

Uses model means (one per model) to avoid pseudoreplication. Exact
enumeration when unique permutations < 100,000, Monte Carlo otherwise.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from itertools import permutations
from math import factorial

import numpy as np

from analysis.load_results import load_all_models, load_model_means


def family_f_stat(
    families: Sequence[str],
    log_params: np.ndarray,
    pcorrs: np.ndarray,
) -> float:
    """One-way F-statistic for family on pcorr, residualized against scale.

    Fits OLS pcorr ~ log10(params), takes residuals, then computes a
    standard one-way F-statistic for family on the residuals. This isolates
    the family effect from any residual scale signal.

    Args:
        families: per-model family labels, parallel to log_params and pcorrs.
        log_params: log10 of parameter count (in billions or absolute, scale
            does not affect the test) for each model.
        pcorrs: partial correlation values for each model.

    Returns:
        F-statistic (>= 0). Returns 0.0 when the test is undefined (one
        family, n <= number of families, or zero within-group variance).
    """
    X = np.column_stack([log_params, np.ones(len(log_params))])
    beta = np.linalg.lstsq(X, pcorrs, rcond=None)[0]
    resid = pcorrs - X @ beta

    unique_fam = list(set(families))
    group_means, group_ns = [], []
    for fam in unique_fam:
        mask = np.array([f == fam for f in families])
        group_means.append(resid[mask].mean())
        group_ns.append(mask.sum())

    grand_mean = resid.mean()
    ss_between = sum(n * (m - grand_mean) ** 2 for n, m in zip(group_ns, group_means))
    ss_within = 0
    for fam, m in zip(unique_fam, group_means):
        mask = np.array([f == fam for f in families])
        ss_within += ((resid[mask] - m) ** 2).sum()

    k = len(unique_fam)
    n = len(families)
    if k <= 1 or n <= k or ss_within == 0:
        return 0.0
    return float((ss_between / (k - 1)) / (ss_within / (n - k)))


def n_unique_permutations(families: Sequence[str]) -> int:
    """Multinomial coefficient for the number of distinct label permutations."""
    from collections import Counter

    counts = Counter(families)
    n = factorial(len(families))
    for c in counts.values():
        n //= factorial(c)
    return n


def run_permutation_test(
    mc_threshold: int = 100000,
    mc_n: int = 50000,
    seed: int = 42,
) -> None:
    """Run the family-effect permutation test and print a summary.

    Loads model means via `load_model_means()`, computes the observed F,
    builds the null distribution by permuting family labels, and prints
    the observed F, p-value, minimum achievable p, and per-model values.

    Args:
        mc_threshold: switch to Monte Carlo when unique permutations exceed
            this count; otherwise enumerate exactly.
        mc_n: Monte Carlo sample size when above threshold.
        seed: RNG seed for Monte Carlo sampling.
    """
    rng = np.random.RandomState(seed)

    models = load_model_means()
    if len(models) < 3:
        print(f"Only {len(models)} models.")
        sys.exit(1)

    print(f"Loaded {len(models)} models")
    load_all_models(verbose=True)
    print()

    families = [m[0] for m in models]
    log_params = np.array([m[1] for m in models])
    pcorrs = np.array([m[2] for m in models])

    observed_f = family_f_stat(families, log_params, pcorrs)
    n_unique = n_unique_permutations(families)

    if n_unique <= mc_threshold:
        seen: set[tuple[str, ...]] = set()
        null_fs_list: list[float] = []
        for perm in permutations(families):
            if perm in seen:
                continue
            seen.add(perm)
            null_fs_list.append(family_f_stat(list(perm), log_params, pcorrs))
        null_fs = np.array(null_fs_list)
        method = f"exact ({len(null_fs)} unique permutations)"
    else:
        null_fs = np.array(
            [family_f_stat(list(rng.permutation(families)), log_params, pcorrs) for _ in range(mc_n)]
        )
        method = f"Monte Carlo ({mc_n} samples of {n_unique} possible)"

    n_total = len(null_fs)
    p_value = float((null_fs >= observed_f).mean())
    min_p = 1.0 / n_total

    print("=== Permutation test for family effect ===")
    print(f"  Models: {len(models)}")
    print(f"  Families: {sorted(set(families))}")
    print(f"  Method: {method}")
    print(f"  Observed F-statistic: {observed_f:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Minimum achievable p: {min_p:.6f} (1/{n_total})")
    print(
        f"  Null F-stat: mean={null_fs.mean():.4f}, "
        f"95th={np.percentile(null_fs, 95):.4f}, "
        f"99th={np.percentile(null_fs, 99):.4f}"
    )

    n_families = len(set(families))
    if n_families < 3:
        print(f"\n  Note: only {n_families} families loaded. Test is underpowered")
        print("  by design. Llama/Gemma data will increase separation.")

    print("\n=== Per-model data ===")
    for fam, lp, pc in sorted(models, key=lambda x: (x[0], x[1])):
        print(f"  {fam:<8} {10**lp:.3f}B  pcorr={pc:+.4f}")


if __name__ == "__main__":
    run_permutation_test()
