"""Permutation test for the family effect on observability.

Uses model means (one per model) to avoid pseudoreplication. Exact
enumeration when unique permutations < 100,000, Monte Carlo otherwise.

Usage: cd nn-observability && uv run python analysis/permutation_test.py
"""

import sys
from itertools import permutations
from math import factorial

import numpy as np

from load_results import load_model_means, load_all_models


def family_f_stat(families, log_params, pcorrs):
    """F-statistic for family effect after residualizing against scale."""
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
    ss_between = sum(n * (m - grand_mean)**2 for n, m in zip(group_ns, group_means))
    ss_within = 0
    for fam, m in zip(unique_fam, group_means):
        mask = np.array([f == fam for f in families])
        ss_within += ((resid[mask] - m)**2).sum()

    k = len(unique_fam)
    n = len(families)
    if k <= 1 or n <= k or ss_within == 0:
        return 0.0
    return (ss_between / (k - 1)) / (ss_within / (n - k))


def n_unique_permutations(families):
    from collections import Counter
    counts = Counter(families)
    n = factorial(len(families))
    for c in counts.values():
        n //= factorial(c)
    return n


def run_permutation_test(mc_threshold=100000, mc_n=50000, seed=42):
    rng = np.random.RandomState(seed)

    models = load_model_means()
    if len(models) < 3:
        print(f'Only {len(models)} models.')
        sys.exit(1)

    print(f'Loaded {len(models)} models')
    load_all_models(verbose=True)
    print()

    families = [m[0] for m in models]
    log_params = np.array([m[1] for m in models])
    pcorrs = np.array([m[2] for m in models])

    observed_f = family_f_stat(families, log_params, pcorrs)
    n_unique = n_unique_permutations(families)

    if n_unique <= mc_threshold:
        seen = set()
        null_fs = []
        for perm in permutations(families):
            if perm in seen:
                continue
            seen.add(perm)
            null_fs.append(family_f_stat(list(perm), log_params, pcorrs))
        null_fs = np.array(null_fs)
        method = f'exact ({len(null_fs)} unique permutations)'
    else:
        null_fs = np.array([
            family_f_stat(list(rng.permutation(families)), log_params, pcorrs)
            for _ in range(mc_n)])
        method = f'Monte Carlo ({mc_n} samples of {n_unique} possible)'

    n_total = len(null_fs)
    p_value = (null_fs >= observed_f).mean()
    min_p = 1.0 / n_total

    print('=== Permutation test for family effect ===')
    print(f'  Models: {len(models)}')
    print(f'  Families: {sorted(set(families))}')
    print(f'  Method: {method}')
    print(f'  Observed F-statistic: {observed_f:.4f}')
    print(f'  p-value: {p_value:.6f}')
    print(f'  Minimum achievable p: {min_p:.6f} (1/{n_total})')
    print(f'  Null F-stat: mean={null_fs.mean():.4f}, '
          f'95th={np.percentile(null_fs, 95):.4f}, '
          f'99th={np.percentile(null_fs, 99):.4f}')

    n_families = len(set(families))
    if n_families < 3:
        print(f'\n  Note: only {n_families} families loaded. Test is underpowered')
        print(f'  by design. Llama/Gemma data will increase separation.')

    print('\n=== Per-model data ===')
    for fam, lp, pc in sorted(models, key=lambda x: (x[0], x[1])):
        print(f'  {fam:<8} {10**lp:.3f}B  pcorr={pc:+.4f}')


if __name__ == '__main__':
    run_permutation_test()
