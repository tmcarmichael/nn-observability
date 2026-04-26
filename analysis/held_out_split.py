"""Cross-validated partial Spearman residualization.

Reads per-token arrays dumped by scripts/dump_tokens.py and compares the
paper's in-sample partial Spearman against a held-out 50/50 split. Reports
the per-model delta to bound any in-sample fitting bias.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, rankdata

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENS_DIR = REPO_ROOT / "results" / "tokens"


def partial_spearman_in_sample(
    x: np.ndarray,
    y: np.ndarray,
    covariates: list[np.ndarray],
) -> float:
    """The paper's current partial-Spearman: fit and evaluate on the same set."""
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates])
    rc = np.column_stack([rc, np.ones(len(rc))])
    coef_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    coef_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    r, _ = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)
    return float(r)


def partial_spearman_held_out(
    x: np.ndarray,
    y: np.ndarray,
    covariates: list[np.ndarray],
    seed: int = 42,
) -> tuple[float, list[float]]:
    """Cross-validated partial-Spearman with two-fold split.

    Fit residualization on split A, evaluate on split B; swap; average.
    Within-split ranking. Both x (probe scores) and y (target loss) are
    residualized against the rank covariates (max softmax rank, activation
    norm rank). Returns (mean across folds, per-fold values).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    perm = rng.permutation(n)
    half = n // 2
    idx_a = perm[:half]
    idx_b = perm[half:]
    rs = []
    for fit_idx, eval_idx in [(idx_a, idx_b), (idx_b, idx_a)]:
        x_eval = x[eval_idx]
        y_eval = y[eval_idx]
        cov_eval = [c[eval_idx] for c in covariates]
        rx = rankdata(x_eval)
        ry = rankdata(y_eval)
        rc = np.column_stack([rankdata(c) for c in cov_eval])
        rc = np.column_stack([rc, np.ones(len(rc))])
        # Fit residualization on the *fit* split, applied to the *eval* split.
        x_fit = x[fit_idx]
        y_fit = y[fit_idx]
        cov_fit = [c[fit_idx] for c in covariates]
        rx_fit = rankdata(x_fit)
        ry_fit = rankdata(y_fit)
        rc_fit = np.column_stack([rankdata(c) for c in cov_fit])
        rc_fit = np.column_stack([rc_fit, np.ones(len(rc_fit))])
        coef_x = np.linalg.lstsq(rc_fit, rx_fit, rcond=None)[0]
        coef_y = np.linalg.lstsq(rc_fit, ry_fit, rcond=None)[0]
        r, _ = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)
        rs.append(float(r))
    return float(np.mean(rs)), rs


def analyze_token_file(path: Path) -> dict[str, Any]:
    """Load a per-token .npz dump and compute in-sample vs held-out pcorr."""
    d = np.load(path, allow_pickle=False)
    target = d["target_surprise"]
    sm = d["max_softmax"]
    norm = d["activation_norm"]
    seeds = d["seeds"].tolist()

    in_sample_per_seed = []
    held_out_per_seed = []
    for seed in seeds:
        scores = d[f"observer_seed{seed}"]
        in_s = partial_spearman_in_sample(scores, target, [sm, norm])
        ho_mean, ho_per_split = partial_spearman_held_out(scores, target, [sm, norm], seed=seed)
        in_sample_per_seed.append(in_s)
        held_out_per_seed.append(ho_mean)

    in_sample_mean = float(np.mean(in_sample_per_seed))
    held_out_mean = float(np.mean(held_out_per_seed))
    return {
        "model": str(d["model"]),
        "peak_layer": int(d["peak_layer"]),
        "n_tokens": int(d["n_tokens"]),
        "ex_per_dim": int(d.get("ex_per_dim", 0)) if "ex_per_dim" in d.files else None,
        "in_sample_per_seed": in_sample_per_seed,
        "held_out_per_seed": held_out_per_seed,
        "in_sample_mean": in_sample_mean,
        "held_out_mean": held_out_mean,
        "delta": held_out_mean - in_sample_mean,
    }


def main() -> None:
    """CLI entry point: scan tokens dir, run held-out vs in-sample, write JSON."""
    parser = argparse.ArgumentParser(description="Cross-validated held-out partial-Spearman analysis.")
    parser.add_argument("--tokens-dir", default=str(TOKENS_DIR), help="Directory of *_tokens.npz files")
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path (default: results/held_out_fit_split.json)",
    )
    args = parser.parse_args()

    tokens_dir = Path(args.tokens_dir)
    if not tokens_dir.exists():
        print(f"ERROR: tokens dir not found: {tokens_dir}")
        return

    files = sorted(tokens_dir.glob("*_tokens.npz"))
    if not files:
        print(f"No *_tokens.npz files in {tokens_dir}")
        return

    rows = []
    print(f"{'model':<40} {'L':>3} {'n':>9} {'in_sample':>10} {'held_out':>10} {'delta':>8}")
    print("-" * 90)
    for f in files:
        try:
            r = analyze_token_file(f)
        except Exception as exc:
            print(f"FAIL {f.name}: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        print(
            f"{r['model']:<40} {r['peak_layer']:>3} {r['n_tokens']:>9} "
            f"{r['in_sample_mean']:>10.4f} {r['held_out_mean']:>10.4f} {r['delta']:>+8.4f}"
        )

    if rows:
        deltas = [r["delta"] for r in rows]
        print()
        print(f"Held-out delta range: {min(deltas):+.4f} to {max(deltas):+.4f}")
        print(f"Max |delta|: {max(abs(d) for d in deltas):.4f}")

    out_path = Path(args.output) if args.output else REPO_ROOT / "results" / "held_out_fit_split.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"analysis": "held_out_50_50_partial_spearman", "rows": rows}, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
