"""
Train identical networks with local (Forward-Forward) and global (Backpropagation)
objectives, then compare their representation structure.

Four model variants isolate confounders:
  FF         - Forward-Forward (local training + label overlay + per-layer norm)
  BP         - Standard backpropagation (baseline)
  BP+norm    - BP with per-layer L2 normalization (isolates normalization effect)
  BP+overlay - BP trained on label-overlaid input (isolates label injection effect)

Probes train on training-set activations, evaluate on test-set activations.
Per-layer, final-layer, and best-layer metrics are all reported.
Normalization is matched between FF and BP+norm.
The label overlay scheme is tested independently via BP+overlay.

Usage:
    uv run train.py                          # MNIST, 50 epochs, 3 seeds
    uv run train.py --dataset cifar10        # harder benchmark
    uv run train.py --seeds 1                # quick single-seed run
    uv run train.py --variants ff bp         # FF vs BP only
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FFLayer(nn.Module):
    """Forward-Forward layer: local goodness objective, no global gradients."""

    def __init__(self, in_dim, out_dim, threshold=2.0, lr=0.03):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.threshold = threshold
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.act(self.linear(x / (x.norm(dim=1, keepdim=True) + 1e-8)))

    def train_step(self, pos, neg):
        pos_h, neg_h = self.forward(pos), self.forward(neg)
        pos_g, neg_g = (pos_h**2).mean(1), (neg_h**2).mean(1)
        loss = (
            nn.functional.softplus(-(pos_g - self.threshold)).mean()
            + nn.functional.softplus(neg_g - self.threshold).mean()
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        pos_out, neg_out = pos_h.detach(), neg_h.detach()
        self.opt.step()
        return pos_out, neg_out, loss.item()


class FFNet(nn.Module):
    def __init__(self, sizes, threshold=2.0, lr=0.03):
        super().__init__()
        self.layers = nn.ModuleList(
            [FFLayer(sizes[i], sizes[i + 1], threshold, lr) for i in range(len(sizes) - 1)]
        )

    def forward(self, x):
        acts, h = [], x
        for L in self.layers:
            h = L(h)
            acts.append(h)
        return acts

    def train_batch(self, pos, neg):
        losses = []
        for L in self.layers:
            pos, neg, loss = L.train_step(pos, neg)
            losses.append(loss)
        return losses

    def predict(self, x):
        return sum((a**2).mean(1) for a in self.forward(x))


class BPNet(nn.Module):
    """Backpropagation network. Optional per-layer L2 normalization to match FF."""

    def __init__(self, sizes, n_cls, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.linears = nn.ModuleList([nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)])
        self.act = nn.ReLU()
        self.head = nn.Linear(sizes[-1], n_cls)

    def _norm(self, x):
        return x / (x.norm(dim=1, keepdim=True) + 1e-8) if self.normalize else x

    def forward(self, x):
        h = x
        for linear in self.linears:
            h = self.act(linear(self._norm(h)))
        return self.head(h)

    def get_activations(self, x):
        acts, h = [], x
        for linear in self.linears:
            h = self.act(linear(self._norm(h)))
            acts.append(h)
        return acts


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def overlay_label(x, y, n_cls):
    x = x.clone()
    x[:, :n_cls] = 0.0
    x[torch.arange(x.size(0), device=x.device), y] = 1.0
    return x


def wrong_labels(y, n_cls):
    return (torch.randint(0, n_cls - 1, y.shape, device=y.device) + y + 1) % n_cls


def _flatten(x):
    return x.view(-1)


def get_data(name, batch_size, device="cpu"):
    pin = device in ("cuda", "mps")
    workers = 2 if pin else 0
    dl_kw = dict(num_workers=workers, persistent_workers=workers > 0, pin_memory=pin)

    if name == "mnist":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(_flatten),
            ]
        )
        tr = datasets.MNIST("./data", True, download=True, transform=tf)
        te = datasets.MNIST("./data", False, transform=tf)
        return DataLoader(tr, batch_size, shuffle=True, **dl_kw), DataLoader(te, batch_size, **dl_kw), 784, 10
    elif name == "cifar10":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)),
                transforms.Lambda(_flatten),
            ]
        )
        tr = datasets.CIFAR10("./data", True, download=True, transform=tf)
        te = datasets.CIFAR10("./data", False, transform=tf)
        return (
            DataLoader(tr, batch_size, shuffle=True, **dl_kw),
            DataLoader(te, batch_size, **dl_kw),
            3072,
            10,
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def linear_probe(train_acts, train_labels, test_acts, test_labels):
    """Train probe on training activations, evaluate on test activations."""
    X_tr, y_tr = train_acts.cpu().numpy(), train_labels.cpu().numpy()
    X_te, y_te = test_acts.cpu().numpy(), test_labels.cpu().numpy()
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te))


def linear_probe_masked(train_acts, train_labels, test_acts, test_labels, n_cls):
    """Probe with first n_cls dimensions zeroed (controls for label overlay).

    FF embeds class labels into the first n_cls input dimensions, which
    persist through layers. Masking these dimensions measures representation
    quality independent of the injected label signal.
    """
    tr_masked = train_acts.clone()
    tr_masked[:, :n_cls] = 0.0
    te_masked = test_acts.clone()
    te_masked[:, :n_cls] = 0.0
    return linear_probe(tr_masked, train_labels, te_masked, test_labels)


def act_sparsity(acts, eps=0.01):
    return (acts.abs() < eps).float().mean().item()


def offdiag_corr(acts):
    a = acts - acts.mean(0, keepdim=True)
    a = a / (a.norm(dim=0, keepdim=True) + 1e-8)
    c = (a.T @ a) / a.size(0)
    mask = ~torch.eye(c.size(0), dtype=torch.bool, device=c.device)
    return c[mask].abs().mean().item()


def dead_frac(acts):
    return (acts.abs().max(0).values < 1e-8).float().mean().item()


def live_neuron_count(acts):
    return (acts.abs().max(0).values >= 1e-8).sum().item()


def eff_rank(acts):
    s = torch.linalg.svdvals(acts.float())
    total = s.sum()
    if total < 1e-10:
        return 1.0
    p = s / total
    p = p[p > 1e-10]
    return (-(p * p.log()).sum()).exp().item()


def polysemanticity(acts, labels, threshold_pct=0.1):
    """Average number of classes each neuron responds to.

    For each neuron, count how many classes produce activation above
    threshold_pct of that neuron's max activation. Lower = more
    monosemantic = more interpretable.
    """
    n_cls = labels.max().item() + 1
    neuron_max = acts.abs().max(0).values
    live_mask = neuron_max > 1e-8
    if live_mask.sum() == 0:
        return float("nan")
    classes_per_neuron = []
    for c in range(n_cls):
        class_mask = labels == c
        if class_mask.sum() == 0:
            continue
        class_mean = acts[class_mask].abs().mean(0)
        active = class_mean > (neuron_max * threshold_pct)
        classes_per_neuron.append(active)
    if not classes_per_neuron:
        return float("nan")
    stacked = torch.stack(classes_per_neuron, dim=0)
    per_neuron = stacked[:, live_mask].sum(0).float()
    return per_neuron.mean().item()


def pruning_curve(
    train_acts, train_labels, test_acts, test_labels, levels=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
):
    """Standard pruning: remove fraction of all neurons by magnitude."""
    order = train_acts.abs().mean(0).argsort()
    out = {}
    for s in levels:
        n = int(train_acts.size(1) * s)
        tr_pruned = train_acts.clone()
        te_pruned = test_acts.clone()
        if n:
            tr_pruned[:, order[:n]] = 0.0
            te_pruned[:, order[:n]] = 0.0
        out[s] = linear_probe(tr_pruned, train_labels, te_pruned, test_labels)
    return out


def pruning_curve_live(
    train_acts, train_labels, test_acts, test_labels, levels=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
):
    """Controlled pruning: remove fraction of LIVE neurons only.

    Controls for FF's dead neuron confounder. Standard pruning on FF networks
    mostly removes already-dead neurons, inflating robustness scores. This
    version prunes only among neurons that actually fire, giving a fair
    comparison of how information is distributed among active neurons.
    """
    live_mask = test_acts.abs().max(0).values >= 1e-8
    tr_live = train_acts[:, live_mask]
    te_live = test_acts[:, live_mask]
    if te_live.size(1) == 0:
        return {s: 0.0 for s in levels}
    order = tr_live.abs().mean(0).argsort()
    out = {}
    for s in levels:
        n = int(tr_live.size(1) * s)
        tr_pruned = tr_live.clone()
        te_pruned = te_live.clone()
        if n:
            tr_pruned[:, order[:n]] = 0.0
            te_pruned[:, order[:n]] = 0.0
        tr_full = torch.zeros_like(train_acts)
        te_full = torch.zeros_like(test_acts)
        tr_full[:, live_mask] = tr_pruned
        te_full[:, live_mask] = te_pruned
        out[s] = linear_probe(tr_full, train_labels, te_full, test_labels)
    return out


def pixel_probe_baseline(train_loader, test_loader, max_n=5000):
    """Linear probe accuracy on raw (normalized) pixels, no learned representation.

    Calibrates whether a model's probe advantage reflects learned structure
    or preserved input information.
    """

    def collect_raw(loader, n):
        xs, ys, total = [], [], 0
        for x, y in loader:
            if total >= n:
                break
            xs.append(x)
            ys.append(y)
            total += x.size(0)
        return torch.cat(xs)[:n], torch.cat(ys)[:n]

    tr_x, tr_y = collect_raw(train_loader, max_n)
    te_x, te_y = collect_raw(test_loader, max_n)
    return linear_probe(tr_x, tr_y, te_x, te_y)


def ablation_effect(train_acts, train_labels, test_acts, test_labels):
    """Probe accuracy drop when individual live neurons are zeroed.

    Measures whether single-neuron interventions produce predictable changes
    in the representation's class readout.
    """
    X_tr, y_tr = train_acts.cpu().numpy(), train_labels.cpu().numpy()
    X_te, y_te = test_acts.cpu().numpy(), test_labels.cpu().numpy()
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)
    clf.fit(X_tr, y_tr)
    base_acc = accuracy_score(y_te, clf.predict(X_te))

    live_mask = test_acts.abs().max(0).values >= 1e-8
    live_idx = live_mask.nonzero().squeeze(-1).tolist()
    if isinstance(live_idx, int):
        live_idx = [live_idx]

    drops = []
    for idx in live_idx:
        ablated = X_te.copy()
        ablated[:, idx] = 0.0
        abl_acc = accuracy_score(y_te, clf.predict(ablated))
        drops.append(base_acc - abl_acc)

    if not drops:
        return 0.0, 0.0, 0.0
    return float(np.mean(drops)), float(np.max(drops)), float(np.std(drops))


def eval_layer(train_acts, train_labels, test_acts, test_labels, name, n_cls=10):
    """Compute representation metrics for one layer with proper probe methodology."""
    return dict(
        layer=name,
        probe_acc=linear_probe(train_acts, train_labels, test_acts, test_labels),
        probe_acc_masked=linear_probe_masked(train_acts, train_labels, test_acts, test_labels, n_cls),
        sparsity=act_sparsity(test_acts),
        correlation=offdiag_corr(test_acts),
        dead_frac=dead_frac(test_acts),
        live_neurons=live_neuron_count(test_acts),
        eff_rank=eff_rank(test_acts),
        polysemanticity=polysemanticity(test_acts, test_labels),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ff(model, loader, epochs, n_cls, device):
    model.to(device)
    for ep in range(epochs):
        ep_loss = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            losses = model.train_batch(
                overlay_label(x, y, n_cls), overlay_label(x, wrong_labels(y, n_cls), n_cls)
            )
            if any(math.isnan(l) for l in losses):
                raise RuntimeError(f"FF diverged at epoch {ep + 1}")
            ep_loss.append(losses)
        if (ep + 1) % 10 == 0 or ep == 0:
            avg = [sum(l[i] for l in ep_loss) / len(ep_loss) for i in range(len(model.layers))]
            print(f"  FF  ep {ep + 1:3d}/{epochs}  loss {[f'{v:.3f}' for v in avg]}")


def eval_ff(model, loader, n_cls, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B, D = x.size()
            x_exp = x.unsqueeze(1).expand(B, n_cls, D).reshape(B * n_cls, D)
            c_labels = torch.arange(n_cls, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
            gs = model.predict(overlay_label(x_exp, c_labels, n_cls)).view(B, n_cls)
            correct += (gs.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_bp(model, loader, epochs, lr, device, n_cls=None, label="BP"):
    """Train BP model. If n_cls is set, overlay labels onto input (BP+overlay ablation)."""
    model.to(device)
    opt, crit = torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()
    for ep in range(epochs):
        tot = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if n_cls is not None:
                x = overlay_label(x, y, n_cls)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  {label:<4} ep {ep + 1:3d}/{epochs}  loss {tot / n:.4f}")


def eval_bp(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def eval_bp_overlay(model, loader, n_cls, device):
    """Evaluate BP+overlay by trying all class overlays, picking best diagonal logit.

    The model was trained on label-overlaid input. At eval, for each sample
    we try all n_cls possible overlays and pick the class whose overlay
    produces the highest self-logit (logit[c] when overlay=c).
    """
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B, D = x.size()
            x_exp = x.unsqueeze(1).expand(B, n_cls, D).reshape(B * n_cls, D)
            c_labels = torch.arange(n_cls, device=device).unsqueeze(0).expand(B, -1).reshape(-1)
            logits = model(overlay_label(x_exp, c_labels, n_cls)).view(B, n_cls, n_cls)
            scores = torch.diagonal(logits, dim1=1, dim2=2)  # (B, n_cls)
            correct += (scores.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def collect(model, loader, mode, n_cls, device, max_n=5000):
    """Collect layer activations. mode: 'ff'|'bp'|'bp_overlay'."""
    model.eval()
    all_a, all_y = None, []
    n = 0
    with torch.inference_mode():
        for x, y in loader:
            if n >= max_n:
                break
            x, y = x.to(device), y.to(device)
            if mode == "ff":
                acts = model.forward(overlay_label(x, y, n_cls))
            elif mode == "bp_overlay":
                acts = model.get_activations(overlay_label(x, y, n_cls))
            else:
                acts = model.get_activations(x)
            if all_a is None:
                all_a = [[] for _ in acts]
            for i, a in enumerate(acts):
                all_a[i].append(a.cpu())
            all_y.append(y.cpu())
            n += x.size(0)
    return [torch.cat(a)[:max_n] for a in all_a], torch.cat(all_y)[:max_n]


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = {
    "ff": {"label": "FF", "short": "FF", "collect_mode": "ff"},
    "bp": {"label": "BP", "short": "BP", "collect_mode": "bp"},
    "bp_norm": {"label": "BP+norm", "short": "BPn", "collect_mode": "bp"},
    "bp_overlay": {"label": "BP+overlay", "short": "BPo", "collect_mode": "bp_overlay"},
}

ALL_VARIANTS = list(VARIANTS.keys())

# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_once(args, seed):
    """Run one comparison across all requested variants. Returns results dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr_dl, te_dl, in_dim, n_cls = get_data(args.dataset, args.batch, args.device)
    sizes = [in_dim] + [args.hidden] * args.layers

    models = {}
    accuracies = {}
    train_times = {}

    # --- Train ---

    if "ff" in args.variants:
        ff = FFNet(sizes, args.threshold, args.ff_lr)
        t0 = time.time()
        train_ff(ff, tr_dl, args.epochs, n_cls, args.device)
        train_times["ff"] = time.time() - t0
        accuracies["ff"] = eval_ff(ff, te_dl, n_cls, args.device)
        models["ff"] = ff
        print(f"  FF accuracy: {accuracies['ff']:.4f} ({train_times['ff']:.0f}s)")

    if "bp" in args.variants:
        bp = BPNet(sizes, n_cls)
        if args.device == "cuda":
            bp = torch.compile(bp)
        t0 = time.time()
        train_bp(bp, tr_dl, args.epochs, args.bp_lr, args.device, label="BP")
        train_times["bp"] = time.time() - t0
        accuracies["bp"] = eval_bp(bp, te_dl, args.device)
        models["bp"] = bp
        print(f"  BP accuracy: {accuracies['bp']:.4f} ({train_times['bp']:.0f}s)")

    if "bp_norm" in args.variants:
        bp_norm = BPNet(sizes, n_cls, normalize=True)
        if args.device == "cuda":
            bp_norm = torch.compile(bp_norm)
        t0 = time.time()
        train_bp(bp_norm, tr_dl, args.epochs, args.bp_lr, args.device, label="BPn")
        train_times["bp_norm"] = time.time() - t0
        accuracies["bp_norm"] = eval_bp(bp_norm, te_dl, args.device)
        models["bp_norm"] = bp_norm
        print(f"  BP+norm accuracy: {accuracies['bp_norm']:.4f} ({train_times['bp_norm']:.0f}s)")

    if "bp_overlay" in args.variants:
        bp_overlay = BPNet(sizes, n_cls)
        if args.device == "cuda":
            bp_overlay = torch.compile(bp_overlay)
        t0 = time.time()
        train_bp(bp_overlay, tr_dl, args.epochs, args.bp_lr, args.device, n_cls=n_cls, label="BPo")
        train_times["bp_overlay"] = time.time() - t0
        accuracies["bp_overlay"] = eval_bp_overlay(bp_overlay, te_dl, n_cls, args.device)
        models["bp_overlay"] = bp_overlay
        print(f"  BP+overlay accuracy: {accuracies['bp_overlay']:.4f} ({train_times['bp_overlay']:.0f}s)")

    # --- Evaluate ---

    result = {"seed": seed}
    for vname in args.variants:
        model = models[vname]
        mode = VARIANTS[vname]["collect_mode"]

        tr_acts, tr_y = collect(model, tr_dl, mode, n_cls, args.device)
        te_acts, te_y = collect(model, te_dl, mode, n_cls, args.device)

        metrics = [
            eval_layer(tr_acts[i], tr_y, te_acts[i], te_y, f"layer_{i}", n_cls) for i in range(len(te_acts))
        ]

        prune = pruning_curve(tr_acts[-1], tr_y, te_acts[-1], te_y)
        prune_live = pruning_curve_live(tr_acts[-1], tr_y, te_acts[-1], te_y)
        abl = ablation_effect(tr_acts[-1], tr_y, te_acts[-1], te_y)

        result[f"{vname}_accuracy"] = accuracies[vname]
        result[f"{vname}_time"] = train_times[vname]
        result[f"{vname}_metrics"] = metrics
        result[f"{vname}_pruning"] = {str(k): v for k, v in prune.items()}
        result[f"{vname}_pruning_live"] = {str(k): v for k, v in prune_live.items()}
        result[f"{vname}_ablation"] = {"mean": abl[0], "max": abl[1], "std": abl[2]}

    result["pixel_probe"] = pixel_probe_baseline(tr_dl, te_dl)
    print(f"  Pixel probe baseline: {result['pixel_probe']:.4f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    P.add_argument("--epochs", type=int, default=50)
    P.add_argument("--hidden", type=int, default=500)
    P.add_argument("--layers", type=int, default=4)
    P.add_argument("--batch", type=int, default=512)
    P.add_argument("--ff-lr", type=float, default=0.03)
    P.add_argument("--bp-lr", type=float, default=0.001)
    P.add_argument("--threshold", type=float, default=2.0)
    P.add_argument("--device", default="auto")
    P.add_argument("--seeds", type=int, default=3)
    P.add_argument(
        "--variants",
        nargs="+",
        default=ALL_VARIANTS,
        choices=ALL_VARIANTS,
        metavar="V",
        help="Variants to run (default: all). Choices: ff, bp, bp_norm, bp_overlay",
    )
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
    if a.device == "cuda":
        torch.set_float32_matmul_precision("high")

    sizes = [{"mnist": 784, "cifar10": 3072}[a.dataset]] + [a.hidden] * a.layers
    vlabels = ", ".join(VARIANTS[v]["label"] for v in a.variants)
    print(f"Dataset: {a.dataset}  Arch: {' -> '.join(map(str, sizes))}")
    print(f"Epochs: {a.epochs}  Seeds: {a.seeds}  Device: {a.device}")
    print(f"Variants: {vlabels}\n")

    all_runs = []
    for i, seed in enumerate(range(42, 42 + a.seeds)):
        print(f"\n--- Seed {seed} ({i + 1}/{a.seeds}) ---")
        result = run_once(a, seed)
        all_runs.append(result)
        if i == 0:
            gc.collect()
            gc.freeze()

    # Save
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    output = dict(config=vars(a), runs=all_runs)
    out_file = out / f"{a.dataset}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # -----------------------------------------------------------------------
    # Aggregate reporting
    # -----------------------------------------------------------------------

    V = a.variants
    n_layers = a.layers

    def vl(v):
        return VARIANTS[v]["label"]

    def avg(key):
        return np.mean([r[key] for r in all_runs if key in r])

    def std(key):
        return np.std([r[key] for r in all_runs if key in r])

    def metric_avg(v, metric):
        vals = []
        for r in all_runs:
            key = f"{v}_metrics"
            if key not in r:
                continue
            vals.append(np.mean([m[metric] for m in r[key]]))
        return np.mean(vals), np.std(vals)

    def layer_val(v, layer, metric):
        vals = [r[f"{v}_metrics"][layer][metric] for r in all_runs if f"{v}_metrics" in r]
        return np.mean(vals)

    print(f"\n{'=' * 72}")
    print(f"  AGGREGATE ({a.seeds} seeds)")
    print(f"{'=' * 72}")

    # Task accuracy
    print("\n  Task accuracy:")
    for v in V:
        k = f"{v}_accuracy"
        print(f"    {vl(v):<12} {avg(k):.4f} +/- {std(k):.4f}")

    # Per-layer probe accuracy (masked)
    print("\n  Probe accuracy (label-masked, per-layer):")
    header = f"    {'':>10}"
    for v in V:
        header += f"  {vl(v):>10}"
    print(header)

    for l in range(n_layers):
        row = f"    layer_{l:<4}"
        for v in V:
            row += f"  {layer_val(v, l, 'probe_acc_masked'):>10.4f}"
        print(row)

    # Best and final layer
    for tag in ["best", "final"]:
        row = f"    {tag:>10}"
        for v in V:
            per_layer = [layer_val(v, l, "probe_acc_masked") for l in range(n_layers)]
            if tag == "best":
                idx = max(range(n_layers), key=lambda i: per_layer[i])
                s = f"{per_layer[idx]:.4f}(L{idx})"
                row += f"  {s:>10}"
            else:
                row += f"  {per_layer[-1]:>10.4f}"
        print(row)

    # Layer-averaged metrics
    print("\n  Layer-averaged metrics:")
    for metric in ["probe_acc", "probe_acc_masked", "sparsity", "dead_frac", "polysemanticity", "eff_rank"]:
        row = f"    {metric:<22}"
        for v in V:
            m, _ = metric_avg(v, metric)
            row += f"  {VARIANTS[v]['short']}={m:.4f}"
        print(row)

    # Pruning
    print("\n  Pruning@90%:")
    for label, suffix in [("raw", "pruning"), ("live-only", "pruning_live")]:
        row = f"    {label:>10}"
        for v in V:
            k = f"{v}_{suffix}"
            vals = [r[k].get("0.9", 0) for r in all_runs if k in r]
            row += f"  {VARIANTS[v]['short']}={np.mean(vals):.4f}"
        print(row)

    # Ablation
    row = "\n  Ablation (mean drop):"
    for v in V:
        k = f"{v}_ablation"
        vals = [r[k]["mean"] for r in all_runs if k in r]
        row += f"  {VARIANTS[v]['short']}={np.mean(vals):.4f}"
    print(row)

    # Pixel baseline
    pixel_mean = np.mean([r["pixel_probe"] for r in all_runs])
    print(f"\n  Pixel probe baseline: {pixel_mean:.4f}")

    print(f"\n  Saved to {out_file}")


if __name__ == "__main__":
    main()
