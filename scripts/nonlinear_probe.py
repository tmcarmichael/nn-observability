"""Nonlinear probe comparison: 2-layer MLP versus linear head.

Trains both probe types on identical activations and compares partial
correlations. Tests whether the confidence-independent signal is
linearly readable or whether a nonlinear probe recovers more.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
elif shutil.which("rocm-smi"):
    subprocess.run(["rocm-smi"], check=False)
else:
    print("No GPU management tool found (nvidia-smi / rocm-smi)")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, rankdata


def partial_spearman(x, y, covariates):
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in covariates])
    rc = np.column_stack([rc, np.ones(len(rc))])
    coef_x = np.linalg.lstsq(rc, rx, rcond=None)[0]
    coef_y = np.linalg.lstsq(rc, ry, rcond=None)[0]
    r, p = pearsonr(rx - rc @ coef_x, ry - rc @ coef_y)
    return float(r), float(p)


def compute_loss_residuals(losses, max_softmax, activation_norm):
    X = np.column_stack([max_softmax, activation_norm, np.ones(len(losses))])
    beta, _, _, _ = np.linalg.lstsq(X, losses, rcond=None)
    return losses - X @ beta


def load_wikitext(split="test", max_docs=None):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    docs, current = [], []
    for row in ds:
        text = row["text"]
        if text.strip() == "" and current:
            docs.append("\n".join(current))
            current = []
            if max_docs and len(docs) >= max_docs:
                break
        elif text.strip():
            current.append(text)
    if current:
        docs.append("\n".join(current))
    return docs


def collect_layer_data(model, tokenizer, docs, layer, device, max_tokens, max_length=512, batch_size=16):
    model.eval()
    all_acts, all_losses, all_softmax, all_norms = [], [], [], []
    total = 0
    captured = {}

    def hook_fn(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        if isinstance(h, tuple):
            h = h[0]
        captured["h"] = h

    # Architecture-specific layer access
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        # GPT-2: model.transformer.h[layer]
        handle = model.transformer.h[layer].register_forward_hook(hook_fn)
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        # Pythia / GPTNeoX: model.gpt_neox.layers[layer]
        handle = model.gpt_neox.layers[layer].register_forward_hook(hook_fn)
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Qwen, Llama, Gemma, Mistral: model.model.layers[layer]
        handle = model.model.layers[layer].register_forward_hook(hook_fn)
    else:
        raise ValueError(f"Unknown model architecture: {type(model).__name__}")

    def process_batch(batch):
        nonlocal total
        tokens = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True)
        input_ids = tokens["input_ids"].to(device)
        attn_mask = tokens["attention_mask"].to(device)
        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attn_mask, use_cache=False)
        h_all = captured["h"]
        for b in range(input_ids.size(0)):
            if total >= max_tokens:
                break
            valid_len = attn_mask[b].bool().sum().item()
            if valid_len < 2:
                continue
            h = h_all[b, : valid_len - 1, :].float().cpu()
            logits = outputs.logits[b, : valid_len - 1, :]
            labels = input_ids[b, 1:valid_len]
            losses = F.cross_entropy(logits, labels, reduction="none").cpu()
            sm = F.softmax(logits, dim=-1).max(dim=-1).values.cpu()
            all_acts.append(h)
            all_losses.append(losses)
            all_softmax.append(sm)
            all_norms.append(h.norm(dim=-1))
            total += h.size(0)
        del outputs, captured["h"]
        if device == "cuda":
            torch.cuda.empty_cache()

    batch_docs = []
    for doc in docs:
        if total >= max_tokens:
            break
        if not doc.strip():
            continue
        batch_docs.append(doc)
        if len(batch_docs) >= batch_size:
            process_batch(batch_docs)
            batch_docs = []

    if batch_docs and total < max_tokens:
        process_batch(batch_docs)

    handle.remove()
    print(f"  Collected {total} positions")
    return {
        "activations": torch.cat(all_acts).float(),
        "losses": torch.cat(all_losses).float().numpy(),
        "max_softmax": torch.cat(all_softmax).float().numpy(),
        "activation_norm": torch.cat(all_norms).float().numpy(),
    }


TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_linear(acts, targets, seed=42, epochs=20, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts_d = acts.to(TRAIN_DEVICE)
    targets_d = targets.to(TRAIN_DEVICE)
    head = torch.nn.Linear(acts.size(1), 1).to(TRAIN_DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts_d, targets_d)
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.cpu()


def train_mlp(acts, targets, seed=42, epochs=50, lr=1e-3, hidden=64):
    torch.manual_seed(seed)
    np.random.seed(seed)
    acts_d = acts.to(TRAIN_DEVICE)
    targets_d = targets.to(TRAIN_DEVICE)
    head = torch.nn.Sequential(
        torch.nn.Linear(acts.size(1), hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    ).to(TRAIN_DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = torch.utils.data.TensorDataset(acts_d, targets_d)
    dl = torch.utils.data.DataLoader(ds, batch_size=4096, shuffle=True)
    head.train()
    for _ in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head.cpu()


def train_mlp_best(acts, targets, val_acts, val_losses, val_covs, seed=42, hidden=64):
    """Train MLP with hyperparameter sweep. Reports the best configuration.
    Sweeps lr x epochs and reports the best configuration."""
    best_rho = -1.0
    best_config = None
    best_head = None
    for lr in [1e-2, 1e-3, 1e-4]:
        for epochs in [20, 50]:
            head = train_mlp(acts, targets, seed=seed, epochs=epochs, lr=lr, hidden=hidden)
            head.eval()
            with torch.inference_mode():
                scores = head(val_acts).squeeze(-1).numpy()
            rho, _ = partial_spearman(scores, val_losses, val_covs)
            if rho > best_rho:
                best_rho = rho
                best_config = {"lr": lr, "epochs": epochs, "hidden": hidden}
                best_head = head
    return best_head, best_rho, best_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--peak-layer", type=int, default=None, help="Layer to probe (default: auto-detect from results)"
    )
    parser.add_argument("--ex-dim", type=int, default=350)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-docs", type=int, default=8000)
    parser.add_argument(
        "--output",
        default=None,
        help="Output filename (default: nonlinear_probe_<model>.json)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for activation collection")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained",
    )
    parser.add_argument(
        "--attn-impl",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention implementation",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {"dtype": torch.bfloat16, "attn_implementation": args.attn_impl}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device)
    model.eval()
    _model_revision = getattr(model.config, "_commit_hash", None) or "unknown"

    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    max_tokens = args.ex_dim * hidden_dim
    peak = args.peak_layer if args.peak_layer is not None else n_layers * 2 // 3
    print(f"{hidden_dim} dim, {n_layers} layers, probing L{peak}")
    print(f"Token budget: {max_tokens} ({args.ex_dim} ex/dim)")

    print("Loading data...")
    train_docs = load_wikitext("train", max_docs=args.max_docs)
    val_docs = load_wikitext("validation", max_docs=None)
    test_docs = load_wikitext("test", max_docs=None)

    print("Collecting train activations...")
    train_data = collect_layer_data(
        model, tokenizer, train_docs, peak, device, max_tokens, batch_size=args.batch_size
    )
    print("Collecting val activations...")
    val_data = collect_layer_data(
        model, tokenizer, val_docs, peak, device, max_tokens, batch_size=args.batch_size
    )
    print("Collecting test activations...")
    test_data = collect_layer_data(
        model, tokenizer, test_docs, peak, device, max_tokens, batch_size=args.batch_size
    )

    del model
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print("Model unloaded.\n")

    residuals = compute_loss_residuals(
        train_data["losses"], train_data["max_softmax"], train_data["activation_norm"]
    )
    targets = torch.from_numpy((residuals > 0).astype(np.float32))

    covs = [val_data["max_softmax"], val_data["activation_norm"]]

    # --- Fixed-hyperparameter comparison (apples to apples) ---
    print(f"=== Fixed-HP comparison at L{peak} (linear: 20ep 1e-3, MLP: 50ep 1e-3) ===")
    print(f"  {'Seed':<6} {'Linear':>10} {'MLP-64':>10} {'MLP-128':>10} {'Delta(64)':>10}")
    print(f"  {'-' * 48}")

    per_seed = []
    for seed in args.seeds:
        linear_head = train_linear(train_data["activations"], targets, seed=seed)
        mlp64_head = train_mlp(train_data["activations"], targets, seed=seed, hidden=64)
        mlp128_head = train_mlp(train_data["activations"], targets, seed=seed, hidden=128)

        linear_head.eval()
        mlp64_head.eval()
        mlp128_head.eval()
        with torch.inference_mode():
            lin_scores = linear_head(val_data["activations"]).squeeze(-1).numpy()
            mlp64_scores = mlp64_head(val_data["activations"]).squeeze(-1).numpy()
            mlp128_scores = mlp128_head(val_data["activations"]).squeeze(-1).numpy()

        lin_rho, _ = partial_spearman(lin_scores, val_data["losses"], covs)
        mlp64_rho, _ = partial_spearman(mlp64_scores, val_data["losses"], covs)
        mlp128_rho, _ = partial_spearman(mlp128_scores, val_data["losses"], covs)

        delta = mlp64_rho - lin_rho
        print(f"  {seed:<6} {lin_rho:+.4f}     {mlp64_rho:+.4f}     {mlp128_rho:+.4f}     {delta:+.4f}")
        per_seed.append(
            {
                "seed": seed,
                "linear": lin_rho,
                "mlp_64": mlp64_rho,
                "mlp_128": mlp128_rho,
                "delta_64": delta,
            }
        )

    # --- HP-swept MLP (best possible nonlinear probe) ---
    print("\n=== HP-swept MLP (best of lr x epochs grid) ===")
    print(f"  {'Seed':<6} {'Linear':>10} {'Best MLP-64':>12} {'Config':>25} {'Delta':>8}")
    print(f"  {'-' * 65}")

    swept_results = []
    for seed in args.seeds:
        linear_head = train_linear(train_data["activations"], targets, seed=seed)
        linear_head.eval()
        with torch.inference_mode():
            lin_scores = linear_head(val_data["activations"]).squeeze(-1).numpy()
        lin_rho, _ = partial_spearman(lin_scores, val_data["losses"], covs)

        _, best_rho, best_cfg = train_mlp_best(
            train_data["activations"],
            targets,
            val_data["activations"],
            val_data["losses"],
            covs,
            seed=seed,
            hidden=64,
        )

        _, best_rho_128, best_cfg_128 = train_mlp_best(
            train_data["activations"],
            targets,
            val_data["activations"],
            val_data["losses"],
            covs,
            seed=seed,
            hidden=128,
        )

        best_overall = max(best_rho, best_rho_128)
        delta = best_overall - lin_rho
        cfg_str = f"h={best_cfg['hidden']} lr={best_cfg['lr']} ep={best_cfg['epochs']}"
        if best_rho_128 > best_rho:
            cfg_str = f"h={best_cfg_128['hidden']} lr={best_cfg_128['lr']} ep={best_cfg_128['epochs']}"
        print(f"  {seed:<6} {lin_rho:+.4f}      {best_overall:+.4f}   {cfg_str:>25} {delta:+.4f}")
        swept_results.append(
            {
                "seed": seed,
                "linear": lin_rho,
                "best_mlp": best_overall,
                "best_config": cfg_str,
                "delta": delta,
            }
        )

    lin_mean = np.mean([s["linear"] for s in per_seed])
    mlp64_mean = np.mean([s["mlp_64"] for s in per_seed])
    delta_fixed = np.mean([s["delta_64"] for s in per_seed])
    delta_swept = np.mean([s["delta"] for s in swept_results])
    best_swept = np.mean([s["best_mlp"] for s in swept_results])

    # --- HP-swept MLP, held-out protocol: select on val, report on test ---
    print("\n=== HP-swept MLP, held-out protocol (select on val, report on test) ===")
    print(f"  {'Seed':<6} {'Lin(test)':>10} {'Val*':>10} {'MLP(test)':>10} {'Config':>25} {'Delta':>8}")
    print(f"  {'-' * 76}")

    test_covs = [test_data["max_softmax"], test_data["activation_norm"]]
    holdout_grid_hidden = [64, 128]
    holdout_grid_lr = [1e-2, 1e-3, 1e-4]
    holdout_grid_epochs = [20, 50]

    holdout_results = []
    for seed in args.seeds:
        linear_head = train_linear(train_data["activations"], targets, seed=seed)
        linear_head.eval()
        with torch.inference_mode():
            lin_scores_test = linear_head(test_data["activations"]).squeeze(-1).numpy()
        lin_rho_test, _ = partial_spearman(lin_scores_test, test_data["losses"], test_covs)

        best_val_rho = -np.inf
        best_test_rho = None
        best_cfg = None
        for hidden in holdout_grid_hidden:
            for lr in holdout_grid_lr:
                for epochs in holdout_grid_epochs:
                    head = train_mlp(
                        train_data["activations"],
                        targets,
                        seed=seed,
                        epochs=epochs,
                        lr=lr,
                        hidden=hidden,
                    )
                    head.eval()
                    with torch.inference_mode():
                        val_scores = head(val_data["activations"]).squeeze(-1).numpy()
                        test_scores = head(test_data["activations"]).squeeze(-1).numpy()
                    val_rho, _ = partial_spearman(val_scores, val_data["losses"], covs)
                    if val_rho > best_val_rho:
                        best_val_rho = val_rho
                        test_rho, _ = partial_spearman(test_scores, test_data["losses"], test_covs)
                        best_test_rho = test_rho
                        best_cfg = {"hidden": hidden, "lr": lr, "epochs": epochs}

        delta = best_test_rho - lin_rho_test
        cfg_str = f"h={best_cfg['hidden']} lr={best_cfg['lr']} ep={best_cfg['epochs']}"
        print(
            f"  {seed:<6} {lin_rho_test:+.4f}   {best_val_rho:+.4f}   "
            f"{best_test_rho:+.4f}   {cfg_str:>25} {delta:+.4f}"
        )
        holdout_results.append(
            {
                "seed": seed,
                "linear_test": lin_rho_test,
                "best_val_rho": best_val_rho,
                "best_mlp_test": best_test_rho,
                "best_config": cfg_str,
                "delta": delta,
            }
        )

    lin_test_mean = np.mean([s["linear_test"] for s in holdout_results])
    mlp_test_mean = np.mean([s["best_mlp_test"] for s in holdout_results])
    delta_holdout = np.mean([s["delta"] for s in holdout_results])

    print("\n=== Summary ===")
    print(f"  Linear mean (val):              {lin_mean:+.4f}")
    print(f"  MLP-64 (fixed HP, val) mean:    {mlp64_mean:+.4f}  (delta {delta_fixed:+.4f})")
    print(f"  Best MLP (swept HP, val) mean:  {best_swept:+.4f}  (delta {delta_swept:+.4f})")
    print(f"  Linear (held-out test) mean:    {lin_test_mean:+.4f}")
    print(f"  MLP (select val, eval test):    {mlp_test_mean:+.4f}  (delta {delta_holdout:+.4f})")
    print(
        f"  Delta as % of linear (test):    {delta_holdout / lin_test_mean * 100:+.1f}%"
        if lin_test_mean > 0
        else ""
    )
    print()
    print("  The held-out delta is the selection-bias-free estimate of how much")
    print("  a nonlinear probe improves over linear on this representation.")

    # Save results
    import datetime as _dt

    output_name = args.output or f"nonlinear_probe_{args.model.split('/')[-1]}.json"

    out = {
        "model": args.model,
        "peak_layer": peak,
        "ex_dim": args.ex_dim,
        "seeds": args.seeds,
        "hidden_dim": hidden_dim,
        "provenance": {
            "model_revision": _model_revision,
            "script": "scripts/nonlinear_probe.py",
            "timestamp": _dt.datetime.now(_dt.UTC).isoformat(),
            "device": str(device),
            "torch_version": torch.__version__,
            "output_file": output_name,
        },
        "protocol": {
            "peak_layer": peak,
            "ex_dim": args.ex_dim,
            "seeds": args.seeds,
            "batch_size": args.batch_size,
            "attn_impl": args.attn_impl,
        },
        "fixed_hp": {
            "per_seed": per_seed,
            "linear_mean": lin_mean,
            "mlp_64_mean": mlp64_mean,
            "delta_mean": delta_fixed,
        },
        "swept_hp": {"per_seed": swept_results, "best_mlp_mean": best_swept, "delta_mean": delta_swept},
        "swept_hp_holdout": {
            "protocol": "fit on train, select HP on wikitext validation, report on wikitext test",
            "grid": {
                "hidden": holdout_grid_hidden,
                "lr": holdout_grid_lr,
                "epochs": holdout_grid_epochs,
            },
            "per_seed": holdout_results,
            "linear_test_mean": lin_test_mean,
            "best_mlp_test_mean": mlp_test_mean,
            "delta_mean": delta_holdout,
        },
        "conclusion": "linear_sufficient" if abs(delta_holdout) < 0.02 else "nonlinear_advantage",
    }

    if Path("/workspace").exists():
        out_path = Path(f"/workspace/{output_name}")
    else:
        out_path = Path(__file__).resolve().parent.parent / "results" / output_name
        out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
