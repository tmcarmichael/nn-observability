"""
Phase 7: SAE comparison.

Does an SAE feature space recover the same decision-quality signal as the
direct linear observer? Apples-to-apples: same model, same hookpoint, same
train/test split, same binary target, same partial correlation evaluation.

Experiments:
  7a: SAE-code linear binary vs raw-residual linear binary
  7b: SAE probe under full-output control (same as 5e)
  7c: Rank overlap between SAE probe and linear observer
  7d: SAE probe in the Phase 6a flagging framework

Usage:
    uv run --extra transformer src/sae_compare.py
    uv run --extra transformer src/sae_compare.py --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from observe import compute_loss_residuals, partial_spearman
from transformer_observe import collect_layer_data, load_wikitext, train_linear_binary

# ---------------------------------------------------------------------------
# SAE encoding
# ---------------------------------------------------------------------------


def encode_with_sae(sae, activations):
    """Encode activations through a pretrained SAE, return sparse feature activations."""
    sae.eval()
    device = next(sae.parameters()).device
    all_codes = []
    batch_size = 4096

    with torch.inference_mode():
        for i in range(0, len(activations), batch_size):
            batch = activations[i : i + batch_size].to(device)
            codes = sae.encode(batch)
            all_codes.append(codes.cpu())

    return torch.cat(all_codes)


def train_sae_probe(sae_codes, losses, softmax, norms, seed=42, epochs=20, lr=1e-3):
    """Train linear binary probe on SAE feature codes."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    residuals = compute_loss_residuals(losses, softmax, norms)
    targets = torch.from_numpy((residuals > 0).astype(np.float32))

    n_features = sae_codes.size(1)
    head = torch.nn.Linear(n_features, 1)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(sae_codes, targets)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)

    head.train()
    for _ep in range(epochs):
        for bx, by in dl:
            loss = F.binary_cross_entropy_with_logits(head(bx).squeeze(-1), by)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return head


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def run_7a(sae, train_data, test_data, seeds):
    """7a: SAE-code linear binary vs raw-residual linear binary."""
    print(f"\n{'=' * 60}")
    print("  7a: SAE probe vs linear observer (layer 8)")
    print(f"{'=' * 60}")

    # Encode through SAE
    print("  Encoding train activations through SAE...")
    train_codes = encode_with_sae(sae, train_data["activations"])
    print(f"    shape: {train_codes.shape} (sparsity: {(train_codes == 0).float().mean():.1%})")
    print("  Encoding test activations through SAE...")
    test_codes = encode_with_sae(sae, test_data["activations"])

    raw_results = []
    sae_results = []

    for seed in seeds:
        # Raw residual linear binary (replication of Phase 5a at layer 8)
        raw_head = train_linear_binary(train_data, seed=seed)
        raw_head.eval()
        with torch.inference_mode():
            raw_scores = raw_head(test_data["activations"]).squeeze(-1).numpy()
        raw_rho, raw_p = partial_spearman(
            raw_scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
        )

        # SAE-code linear binary
        sae_head = train_sae_probe(
            train_codes,
            train_data["losses"],
            train_data["max_softmax"],
            train_data["activation_norm"],
            seed=seed,
        )
        sae_head.eval()
        with torch.inference_mode():
            sae_scores = sae_head(test_codes).squeeze(-1).numpy()
        sae_rho, sae_p = partial_spearman(
            sae_scores, test_data["losses"], [test_data["max_softmax"], test_data["activation_norm"]]
        )

        raw_results.append({"rho": raw_rho, "scores": raw_scores})
        sae_results.append({"rho": sae_rho, "scores": sae_scores})
        print(f"  Seed {seed}: raw={raw_rho:+.4f}  SAE={sae_rho:+.4f}")

    # Seed agreement
    for label, results in [("raw", raw_results), ("SAE", sae_results)]:
        pairs = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                r, _ = spearmanr(results[i]["scores"], results[j]["scores"])
                pairs.append(r)
        print(f"  {label} seed agreement: {np.mean(pairs):+.4f}")

    raw_mean = np.mean([r["rho"] for r in raw_results])
    sae_mean = np.mean([r["rho"] for r in sae_results])
    print("\n  7a RESULT:")
    print(f"    raw linear binary:  {raw_mean:+.4f}")
    print(f"    SAE linear binary:  {sae_mean:+.4f}")

    if abs(sae_mean - raw_mean) < 0.03:
        print("    --> Similar: signal lives in SAE feature space")
    elif sae_mean > raw_mean:
        print("    --> SAE outperforms: richer basis for decision quality")
    else:
        print("    --> Raw outperforms: SAE compression loses signal")

    return {
        "raw": {"mean": float(raw_mean), "per_seed": [r["rho"] for r in raw_results]},
        "sae": {"mean": float(sae_mean), "per_seed": [r["rho"] for r in sae_results]},
        "raw_scores": [r["scores"].tolist() for r in raw_results],
        "sae_scores": [r["scores"].tolist() for r in sae_results],
    }


def run_7c(results_7a, seeds):
    """7c: Rank overlap between SAE probe and linear observer."""
    print(f"\n{'=' * 60}")
    print("  7c: Rank overlap (SAE probe vs linear observer)")
    print(f"{'=' * 60}")

    for i, seed in enumerate(seeds):
        raw_scores = np.array(results_7a["raw_scores"][i])
        sae_scores = np.array(results_7a["sae_scores"][i])
        rho, _ = spearmanr(raw_scores, sae_scores)
        print(f"  Seed {seed}: rank corr = {rho:+.4f}")

    # Average
    all_rhos = []
    for i in range(len(seeds)):
        raw_scores = np.array(results_7a["raw_scores"][i])
        sae_scores = np.array(results_7a["sae_scores"][i])
        r, _ = spearmanr(raw_scores, sae_scores)
        all_rhos.append(r)
    mean_rho = np.mean(all_rhos)

    print(f"\n  7c RESULT: mean rank correlation = {mean_rho:+.4f}")
    if mean_rho > 0.7:
        print("    --> High overlap: reading the same signal through different decompositions")
    elif mean_rho > 0.3:
        print("    --> Moderate overlap: partially shared signal, combining may help")
    else:
        print("    --> Low overlap: reading different things, strong case for multi-channel")

    return {"mean_rank_correlation": float(mean_rho), "per_seed": all_rhos}


def run_7d(sae, test_data, test_11, train_data, seeds):
    """7d: SAE probe in the Phase 6a flagging framework."""
    print(f"\n{'=' * 60}")
    print("  7d: SAE probe flagging vs confidence")
    print(f"{'=' * 60}")

    train_codes = encode_with_sae(sae, train_data["activations"])
    test_codes = encode_with_sae(sae, test_data["activations"])

    n_test = min(len(test_data["losses"]), len(test_11["losses"]))
    test_losses = test_data["losses"][:n_test]
    output_softmax = test_11["max_softmax"][:n_test]
    test_codes_aligned = test_codes[:n_test]
    test_acts_aligned = test_data["activations"][:n_test]

    median_loss = np.median(test_losses)
    is_high_loss = test_losses > median_loss

    flag_rate = 0.10
    k = int(n_test * flag_rate)

    # Confidence flags
    conf_threshold = np.sort(output_softmax)[k]
    conf_flagged = output_softmax <= conf_threshold

    all_results = []
    for seed in seeds:
        # SAE probe
        sae_head = train_sae_probe(
            train_codes,
            train_data["losses"],
            train_data["max_softmax"],
            train_data["activation_norm"],
            seed=seed,
        )
        sae_head.eval()
        with torch.inference_mode():
            sae_scores = sae_head(test_codes_aligned).squeeze(-1).numpy()

        # Raw observer
        raw_head = train_linear_binary(train_data, seed=seed)
        raw_head.eval()
        with torch.inference_mode():
            raw_scores = raw_head(test_acts_aligned).squeeze(-1).numpy()

        sae_threshold = np.sort(sae_scores)[-k]
        sae_flagged = sae_scores >= sae_threshold
        raw_threshold = np.sort(raw_scores)[-k]
        raw_flagged = raw_scores >= raw_threshold

        sae_precision = is_high_loss[sae_flagged].mean()
        raw_precision = is_high_loss[raw_flagged].mean()
        conf_precision = is_high_loss[conf_flagged].mean()

        sae_exclusive = (sae_flagged & ~conf_flagged & is_high_loss).sum()
        raw_exclusive = (raw_flagged & ~conf_flagged & is_high_loss).sum()

        # Three-channel: flag if any of the three flags
        triple_flagged = sae_flagged | raw_flagged | conf_flagged
        triple_precision = is_high_loss[triple_flagged].mean() if triple_flagged.sum() > 0 else 0.0
        triple_catches = (triple_flagged & is_high_loss).sum()

        all_results.append(
            {
                "sae_precision": float(sae_precision),
                "raw_precision": float(raw_precision),
                "conf_precision": float(conf_precision),
                "sae_exclusive": int(sae_exclusive),
                "raw_exclusive": int(raw_exclusive),
                "triple_precision": float(triple_precision),
                "triple_catches": int(triple_catches),
            }
        )

        print(
            f"  Seed {seed}: SAE prec={sae_precision:.3f}  raw prec={raw_precision:.3f}  "
            f"conf prec={conf_precision:.3f}  SAE-exclusive={sae_exclusive}"
        )

    print(f"\n  7d RESULT (10% flag rate, {len(seeds)} seeds):")
    print(f"    SAE probe precision:  {np.mean([r['sae_precision'] for r in all_results]):.3f}")
    print(f"    Raw observer precision: {np.mean([r['raw_precision'] for r in all_results]):.3f}")
    print(f"    Confidence precision: {np.mean([r['conf_precision'] for r in all_results]):.3f}")
    print(f"    SAE-exclusive catches: {np.mean([r['sae_exclusive'] for r in all_results]):.0f}")
    print(f"    Raw-exclusive catches: {np.mean([r['raw_exclusive'] for r in all_results]):.0f}")
    triple_c = np.mean([r["triple_catches"] for r in all_results])
    triple_p = np.mean([r["triple_precision"] for r in all_results])
    print(f"    Three-channel: {triple_c:.0f} catches, {triple_p:.3f} precision")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--seeds", type=int, default=3)
    P.add_argument("--device", default="auto")
    P.add_argument("--max-tokens-train", type=int, default=200000)
    P.add_argument("--max-tokens-test", type=int, default=100000)
    P.add_argument("--all", action="store_true", help="Run 7a + 7c + 7d")
    a = P.parse_args()

    if a.device == "auto":
        a.device = (
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )

    seeds = list(range(42, 42 + a.seeds))
    print("Phase 7: SAE comparison")
    print(f"Device: {a.device}  Seeds: {seeds}")

    # Load model
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("\nLoading GPT-2 124M...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(a.device)
    model.eval()

    # Load SAE for layer 8 (= blocks.9.hook_resid_pre in SAE Lens convention)
    from sae_lens import SAE

    print("Loading pretrained SAE (blocks.9.hook_resid_pre)...")
    sae = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.9.hook_resid_pre")
    sae = sae.to(a.device)
    print(f"  SAE: {sae.cfg.d_in} -> {sae.cfg.d_sae} features")

    # Load data
    print("\nLoading WikiText-103...")
    train_docs = load_wikitext("train", max_docs=2000)
    test_docs = load_wikitext("test", max_docs=500)

    # Collect activations at layer 8 (same as Phase 5)
    print("\nCollecting layer 8 activations...")
    train_data = collect_layer_data(model, tokenizer, train_docs, 8, a.device, a.max_tokens_train)
    test_data = collect_layer_data(model, tokenizer, test_docs, 8, a.device, a.max_tokens_test)

    results = {}
    t0 = time.time()

    # 7a: core comparison
    results["7a"] = run_7a(sae, train_data, test_data, seeds)

    # 7c: rank overlap
    results["7c"] = run_7c(results["7a"], seeds)

    # 7d: flagging (needs layer 11 data for confidence baseline)
    if a.all:
        print("\nCollecting layer 11 test activations...")
        test_11 = collect_layer_data(model, tokenizer, test_docs, 11, a.device, a.max_tokens_test)
        results["7d"] = run_7d(sae, test_data, test_11, train_data, seeds)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")

    # Save
    out = Path(__file__).resolve().parent.parent / "results"
    out.mkdir(exist_ok=True)
    # Don't save raw scores (too large for JSON)
    save_results = {k: v for k, v in results.items() if k != "7a"}
    if "7a" in results:
        save_results["7a"] = {k: v for k, v in results["7a"].items() if k not in ("raw_scores", "sae_scores")}
    out_file = out / "sae_compare.json"
    with open(out_file, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
