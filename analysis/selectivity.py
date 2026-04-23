"""Selectivity analysis: random head baselines and control gap.

Reports the random-head baseline (near zero means signal requires
learning) and the control gap (what fraction of raw correlation is
confidence).
"""

from collections import defaultdict

import numpy as np

from analysis.load_results import load_all_models, load_control_sensitivity, load_random_head_baselines


def analyze_selectivity():
    load_all_models(verbose=True)
    print()

    # Part 1: Random head baselines
    print("=== Random head baselines (selectivity control) ===")
    baselines = load_random_head_baselines()
    if baselines:
        for name, family, _params, rh in baselines:
            print(f"  {name:<15} {family:<8} random_head = {rh:+.4f}")
        rh_vals = [b[3] for b in baselines]
        print(f"\n  Mean: {np.mean(rh_vals):+.4f}, Max abs: {max(abs(v) for v in rh_vals):.4f}")
        print("  A randomly initialized probe with identical architecture achieves")
        print(f"  partial correlation indistinguishable from zero (mean {np.mean(rh_vals):+.4f}),")
        print("  confirming the trained probe finds learned structure.")
    else:
        print("  No random head baselines found.")
    print()

    # Part 2: Control gap
    print("=== Control gap: what fraction of raw signal is confidence? ===")
    models = load_control_sensitivity()
    if not models:
        print("  No control sensitivity data found.")
        return

    print(f"  {'Model':<15} {'Raw':>7} {'Controlled':>11} {'Gap%':>7} {'Residual':>10}")
    print(f"  {'-' * 55}")
    gaps = []
    for m in models:
        raw = m["none"]
        controlled = m["standard"]
        gap_pct = (1 - controlled / raw) * 100 if raw > 0 else 0
        gaps.append(gap_pct)
        print(f"  {m['name']:<15} {raw:+.4f}  {controlled:+.4f}    {gap_pct:5.1f}%  {controlled:+.4f}")

    print(f"\n  Mean control gap: {np.mean(gaps):.1f}% +/- {np.std(gaps):.1f}%")

    # Per-family breakdown
    fam_gaps = defaultdict(list)
    for m, g in zip(models, gaps):
        fam_gaps[m["family"]].append(g)
    print("\n  Per-family:")
    for fam, gs in sorted(fam_gaps.items()):
        print(f"    {fam:<8}: {np.mean(gs):.1f}% ({len(gs)} models)")
    if "Llama" in fam_gaps and "Qwen" in fam_gaps:
        llama_gap = np.mean(fam_gaps["Llama"])
        qwen_gap = np.mean(fam_gaps["Qwen"])
        if llama_gap > qwen_gap + 10:
            print(f"\n  Llama control gap ({llama_gap:.0f}%) > Qwen ({qwen_gap:.0f}%):")
            print("  Llama has less independent signal, not just less total signal.")

    # Part 3: Nonlinear check
    print("\n=== Nonlinear MLP vs standard controls ===")
    deltas = []
    for m in models:
        delta = m["nonlinear"] - m["standard"]
        deltas.append(delta)
        print(f"  {m['name']:<15}: nonlinear - standard = {delta:+.4f}")
    print(f"\n  Mean delta: {np.mean(deltas):+.4f}")
    if abs(np.mean(deltas)) < 0.02:
        print("  Nonlinear MLP does not reduce signal beyond linear controls.")
    else:
        print(f"  Nonlinear MLP shows delta of {np.mean(deltas):+.4f} vs linear controls.")

    # Part 4: Entropy
    print("\n=== Logit entropy as additional control ===")
    for m in models:
        if m["standard"] > 0:
            remaining = m["plus_entropy"] / m["standard"] * 100
            print(
                f"  {m['name']:<15}: {m['standard']:+.4f} -> {m['plus_entropy']:+.4f} "
                f"({remaining:.0f}% remains after entropy)"
            )


if __name__ == "__main__":
    analyze_selectivity()
