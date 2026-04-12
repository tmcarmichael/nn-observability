"""
Scaling study: compare local vs global training objectives across 5 model sizes.
Tests whether representation structure differences hold, or change, as models scale.

Uses confounder-controlled metrics (label-masked probing, live-neuron pruning).

Usage:
    uv run scale.py                  # full sweep
    uv run scale.py --device cuda    # faster on GPU
    uv run scale.py --epochs 100     # longer training

Outputs results/scaling.json and generates scaling.png.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import (
    BPNet,
    FFNet,
    collect,
    eval_bp,
    eval_ff,
    eval_layer,
    get_data,
    pruning_curve,
    pruning_curve_live,
    train_bp,
    train_ff,
)

CONFIGS = [
    {"name": "XS", "layers": 2, "hidden": 256, "label": "2x256\n~200K"},
    {"name": "S", "layers": 4, "hidden": 500, "label": "4x500\n~1M"},
    {"name": "M", "layers": 4, "hidden": 1000, "label": "4x1000\n~4M"},
    {"name": "L", "layers": 6, "hidden": 1000, "label": "6x1000\n~6M"},
    {"name": "XL", "layers": 8, "hidden": 1000, "label": "8x1000\n~8M"},
]


def run_one(cfg, args):
    """Run a single local vs global training comparison at a given scale."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_dl, test_dl, in_dim, n_cls = get_data(args.dataset, args.batch, args.device)
    sizes = [in_dim] + [cfg["hidden"]] * cfg["layers"]
    n_params = sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
    print(f"\n{'=' * 60}")
    print(f"  {cfg['name']}:  {' -> '.join(map(str, sizes))}  ({n_params:,} params)")
    print(f"{'=' * 60}")

    # FF
    print("  Training FF...")
    ff = FFNet(sizes, args.threshold, args.ff_lr)
    t0 = time.time()
    train_ff(ff, train_dl, args.epochs, n_cls, args.device)
    ff_time = time.time() - t0
    ff_acc = eval_ff(ff, test_dl, n_cls, args.device)

    # BP
    print("  Training BP...")
    bp = BPNet(sizes, n_cls)
    t0 = time.time()
    train_bp(bp, train_dl, args.epochs, args.bp_lr, args.device)
    bp_time = time.time() - t0
    bp_acc = eval_bp(bp, test_dl, args.device)

    print(f"  FF acc: {ff_acc:.4f} ({ff_time:.0f}s)  BP acc: {bp_acc:.4f} ({bp_time:.0f}s)")

    # Evaluate (proper train/test split for probes)
    ff_tr, ff_tr_y = collect(ff, train_dl, "ff", n_cls, args.device)
    ff_te, ff_te_y = collect(ff, test_dl, "ff", n_cls, args.device)
    bp_tr, bp_tr_y = collect(bp, train_dl, "bp", n_cls, args.device)
    bp_te, bp_te_y = collect(bp, test_dl, "bp", n_cls, args.device)

    ff_m = [eval_layer(ff_tr[i], ff_tr_y, ff_te[i], ff_te_y, f"layer_{i}", n_cls) for i in range(len(ff_te))]
    bp_m = [eval_layer(bp_tr[i], bp_tr_y, bp_te[i], bp_te_y, f"layer_{i}", n_cls) for i in range(len(bp_te))]

    ff_prune = pruning_curve(ff_tr[-1], ff_tr_y, ff_te[-1], ff_te_y)
    bp_prune = pruning_curve(bp_tr[-1], bp_tr_y, bp_te[-1], bp_te_y)
    ff_prune_live = pruning_curve_live(ff_tr[-1], ff_tr_y, ff_te[-1], ff_te_y)
    bp_prune_live = pruning_curve_live(bp_tr[-1], bp_tr_y, bp_te[-1], bp_te_y)

    return {
        "config": cfg["name"],
        "n_layers": cfg["layers"],
        "hidden": cfg["hidden"],
        "n_params": n_params,
        "ff_accuracy": ff_acc,
        "bp_accuracy": bp_acc,
        "ff_time": ff_time,
        "bp_time": bp_time,
        "ff_metrics": ff_m,
        "bp_metrics": bp_m,
        "ff_pruning": {str(k): v for k, v in ff_prune.items()},
        "bp_pruning": {str(k): v for k, v in bp_prune.items()},
        "ff_pruning_live": {str(k): v for k, v in ff_prune_live.items()},
        "bp_pruning_live": {str(k): v for k, v in bp_prune_live.items()},
    }


def plot_scaling(all_results, out_path="scaling.png"):
    """Generate scaling study chart with confounder-controlled metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor="white")
    fig.suptitle("Scaling Study: Local vs Global Training Objectives", fontsize=16, fontweight="bold", y=0.97)

    FF_C, BP_C = "#2563eb", "#dc2626"
    labels = [c["label"] for c in CONFIGS[: len(all_results)]]
    x = np.arange(len(all_results))
    W = 0.35

    def get(metrics, key):
        if key in metrics[0]:
            return np.mean([m[key] for m in metrics])
        return 0.0

    # Panel 1: Task accuracy
    ax = axes[0, 0]
    ff_v = [r["ff_accuracy"] for r in all_results]
    bp_v = [r["bp_accuracy"] for r in all_results]
    ax.bar(x - W / 2, ff_v, W, color=FF_C, label="FF", zorder=3)
    ax.bar(x + W / 2, bp_v, W, color=BP_C, label="BP", zorder=3)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Task Accuracy", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    lo = min(ff_v + bp_v)
    ax.set_ylim(max(0, lo - 0.05), 1.005)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    # Panel 2: Probe accuracy MASKED (controlled for label overlay)
    ax = axes[0, 1]
    ff_v = [get(r["ff_metrics"], "probe_acc_masked") for r in all_results]
    bp_v = [get(r["bp_metrics"], "probe_acc_masked") for r in all_results]
    ax.bar(x - W / 2, ff_v, W, color=FF_C, label="FF", zorder=3)
    ax.bar(x + W / 2, bp_v, W, color=BP_C, label="BP", zorder=3)
    ax.set_ylabel("Probe Accuracy (avg)")
    ax.set_title("Probe Accuracy (label-masked)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    lo = min(ff_v + bp_v)
    ax.set_ylim(max(0, lo - 0.03), 1.005)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(
        0.02,
        0.02,
        "Label overlay dims zeroed out",
        transform=ax.transAxes,
        fontsize=8,
        color="#888",
        style="italic",
    )

    # Panel 3: Polysemanticity
    ax = axes[0, 2]
    ff_v = [get(r["ff_metrics"], "polysemanticity") for r in all_results]
    bp_v = [get(r["bp_metrics"], "polysemanticity") for r in all_results]
    ax.bar(x - W / 2, ff_v, W, color=FF_C, label="FF", zorder=3)
    ax.bar(x + W / 2, bp_v, W, color=BP_C, label="BP", zorder=3)
    ax.set_ylabel("Classes per neuron (avg)")
    ax.set_title("Polysemanticity", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(
        0.02,
        0.02,
        "Lower = more monosemantic",
        transform=ax.transAxes,
        fontsize=8,
        color="#888",
        style="italic",
    )

    # Panel 4: Pruning at 90% LIVE neurons (controlled for dead neurons)
    ax = axes[1, 0]
    ff_v = [r["ff_pruning_live"].get("0.9", 0) for r in all_results]
    bp_v = [r["bp_pruning_live"].get("0.9", 0) for r in all_results]
    ax.bar(x - W / 2, ff_v, W, color=FF_C, label="FF", zorder=3)
    ax.bar(x + W / 2, bp_v, W, color=BP_C, label="BP", zorder=3)
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Pruning@90% (live neurons only)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    lo = min(ff_v + bp_v)
    ax.set_ylim(max(0, lo - 0.05), 1.005)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(
        0.02,
        0.02,
        "Dead neurons excluded from pruning",
        transform=ax.transAxes,
        fontsize=8,
        color="#888",
        style="italic",
    )

    # Panel 5: Delta trends (controlled metrics)
    ax = axes[1, 1]
    probe_d = [
        get(r["ff_metrics"], "probe_acc_masked") - get(r["bp_metrics"], "probe_acc_masked")
        for r in all_results
    ]
    poly_d = [
        get(r["ff_metrics"], "polysemanticity") - get(r["bp_metrics"], "polysemanticity") for r in all_results
    ]
    prune_d = [r["ff_pruning_live"].get("0.9", 0) - r["bp_pruning_live"].get("0.9", 0) for r in all_results]

    ax.plot(x, probe_d, "o-", color="#16a34a", lw=2, ms=8, label="Probe (masked) d")
    ax.plot(x, poly_d, "s-", color="#9333ea", lw=2, ms=8, label="Polysemanticity d")
    ax.plot(x, prune_d, "^-", color="#ea580c", lw=2, ms=8, label="Pruning (live) d")
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.set_ylabel("FF - BP (positive = FF better)")
    ax.set_title("Controlled Metrics: Advantage vs Scale", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Panel 6: Summary table
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    for i, r in enumerate(all_results):
        table_data.append(
            [
                r["config"],
                f"{r['n_params'] / 1e6:.1f}M",
                f"{r['ff_accuracy'] - r['bp_accuracy']:+.1%}",
                f"{probe_d[i]:+.1%}",
                f"{prune_d[i]:+.1%}",
            ]
        )
    table = ax.table(
        cellText=table_data,
        colLabels=["Size", "Params", "Acc d", "Probe d\n(masked)", "Prune d\n(live)"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (r, _c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
    ax.set_title("Summary (confounder-controlled)", fontweight="bold", pad=20)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path}")


def avg_over_seeds(seed_results):
    """Average a list of single-seed results for one config into one summary."""
    r0 = seed_results[0]
    n_layers = len(r0["ff_metrics"])

    def mean_metric(key, metric):
        return np.mean([np.mean([m[metric] for m in r[key]]) for r in seed_results])

    def mean_pruning(key, level="0.9"):
        return np.mean([r[key].get(level, 0) for r in seed_results])

    # Average per-layer metrics across seeds
    ff_m, bp_m = [], []
    for l in range(n_layers):
        ff_layer, bp_layer = {}, {}
        for metric in r0["ff_metrics"][0]:
            if metric == "layer":
                ff_layer["layer"] = r0["ff_metrics"][l]["layer"]
                bp_layer["layer"] = r0["bp_metrics"][l]["layer"]
            else:
                ff_layer[metric] = np.mean([r["ff_metrics"][l][metric] for r in seed_results])
                bp_layer[metric] = np.mean([r["bp_metrics"][l][metric] for r in seed_results])
        ff_m.append(ff_layer)
        bp_m.append(bp_layer)

    def avg_pruning(key):
        return {s: np.mean([r[key].get(s, 0) for r in seed_results]) for s in r0[key]}

    return {
        "config": r0["config"],
        "n_layers": r0["n_layers"],
        "hidden": r0["hidden"],
        "n_params": r0["n_params"],
        "ff_accuracy": np.mean([r["ff_accuracy"] for r in seed_results]),
        "bp_accuracy": np.mean([r["bp_accuracy"] for r in seed_results]),
        "ff_time": np.mean([r["ff_time"] for r in seed_results]),
        "bp_time": np.mean([r["bp_time"] for r in seed_results]),
        "ff_metrics": ff_m,
        "bp_metrics": bp_m,
        "ff_pruning": avg_pruning("ff_pruning"),
        "bp_pruning": avg_pruning("bp_pruning"),
        "ff_pruning_live": avg_pruning("ff_pruning_live"),
        "bp_pruning_live": avg_pruning("bp_pruning_live"),
        "n_seeds": len(seed_results),
    }


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    P.add_argument("--epochs", type=int, default=50)
    P.add_argument("--batch", type=int, default=512)
    P.add_argument("--ff-lr", type=float, default=0.03)
    P.add_argument("--bp-lr", type=float, default=0.001)
    P.add_argument("--threshold", type=float, default=2.0)
    P.add_argument("--device", default="auto")
    P.add_argument("--seeds", type=int, default=3)
    args = P.parse_args()

    if args.device == "auto":
        args.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"Scaling study: {len(CONFIGS)} configs x {args.seeds} seed(s)")
    print(f"Dataset: {args.dataset}  Epochs: {args.epochs}  Device: {args.device}")

    all_results = []
    for cfg in CONFIGS:
        seed_runs = []
        for seed in range(42, 42 + args.seeds):
            args.seed = seed
            result = run_one(cfg, args)
            seed_runs.append(result)
        if args.seeds > 1:
            all_results.append(avg_over_seeds(seed_runs))
        else:
            all_results.append(seed_runs[0])

    # Save
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "scaling.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {out / 'scaling.json'}")

    # Plot
    chart_path = out.parent / "assets" / "scaling.png"
    chart_path.parent.mkdir(exist_ok=True)
    plot_scaling(all_results, str(chart_path))

    # Summary
    def get(metrics, key):
        if key in metrics[0]:
            return np.mean([m[key] for m in metrics])
        return 0.0

    print(f"\n{'=' * 70}")
    print(f"  SCALING SUMMARY (confounder-controlled, {args.seeds} seed(s))")
    print(f"{'=' * 70}")
    print(
        f"  {'Config':<6} {'Params':>8} {'FF Acc':>8} {'BP Acc':>8} "
        f"{'Probe d':>9} {'Poly d':>9} {'Prune d':>9}"
    )
    print(f"  {'':>6} {'':>8} {'':>8} {'':>8} {'(masked)':>9} {'':>9} {'(live)':>9}")
    print(f"  {'-' * 62}")
    for r in all_results:
        probe_d = get(r["ff_metrics"], "probe_acc_masked") - get(r["bp_metrics"], "probe_acc_masked")
        poly_d = get(r["ff_metrics"], "polysemanticity") - get(r["bp_metrics"], "polysemanticity")
        prune_d = r["ff_pruning_live"].get("0.9", 0) - r["bp_pruning_live"].get("0.9", 0)
        print(
            f"  {r['config']:<6} {r['n_params'] / 1e6:>7.1f}M {r['ff_accuracy']:>8.4f} "
            f"{r['bp_accuracy']:>8.4f} {probe_d:>+9.4f} {poly_d:>+9.4f} {prune_d:>+9.4f}"
        )


if __name__ == "__main__":
    main()
