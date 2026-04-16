"""Generate LaTeX data macros from results JSONs.

Produces data_macros.sty for the paper repo. Every number that appears
in prose and depends on experimental data gets a macro. The .tex files
reference macros instead of hardcoded literals.

Usage: cd nn-observability && uv run python analysis/generate_data_macros.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT.parent / "nn-observability-paper"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from load_results import load_all_models, load_model_means, load_per_seed
from permutation_test import family_f_stat


def _load_json(filename: str) -> dict:
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def _fmt(val: float, dp: int = 3, sign: bool = True) -> str:
    """Format a float for LaTeX display."""
    if sign:
        return f"{val:+.{dp}f}"
    return f"{val:.{dp}f}"


def _pct(val: float, dp: int = 1) -> str:
    return f"{val:.{dp}f}"


def generate_macros() -> str:
    lines = [
        "% data_macros.sty -- AUTO-GENERATED, DO NOT HAND-EDIT",
        "% Source: nn-observability/analysis/generate_data_macros.py",
        "% Regenerate: cd nn-observability && uv run python analysis/generate_data_macros.py",
        r"\ProvidesPackage{data_macros}",
        "",
    ]

    def macro(name: str, val: str, comment: str = ""):
        c = f"  % {comment}" if comment else ""
        lines.append(f"\\newcommand{{\\{name}}}{{{val}}}{c}")

    def section(title: str):
        lines.append("")
        lines.append(f"% --- {title} ---")

    # Load data
    to = _load_json("transformer_observe.json")
    q05 = _load_json("qwen05b_v3_results.json")
    q15 = _load_json("qwen1_5b_v3_results.json")
    q3 = _load_json("qwen3b_v3_results.json")
    q7 = _load_json("qwen7b_v3_results.json")
    q14 = _load_json("qwen14b_v3_results.json")
    llama3b = _load_json("llama3b_v3_results.json")
    llama1b = _load_json("llama1b_results.json")
    llama1bi = _load_json("llama1b_instruct_results.json")
    llama8b = _load_json("llama8b_results.json")
    gemma1b = _load_json("gemma3_1b_results.json")
    gemma4b = _load_json("gemma4b_results.json")
    phi3 = _load_json("phi3_mini_results.json")

    # Instruct files
    qi05 = _load_json("qwen05b_instruct_v3_results.json")
    qi15 = _load_json("qwen1_5b_instruct_v3_results.json")
    qi3 = _load_json("qwen3b_instruct_v3_results.json")
    qi7 = _load_json("qwen7b_instruct_v3_results.json")
    qi14 = None
    try:
        qi14 = _load_json("qwen14b_instruct_results.json")
    except FileNotFoundError:
        pass

    # Mistral
    mistral = None
    try:
        mistral = _load_json("mistral7b_results.json")
    except FileNotFoundError:
        pass

    gpt2 = to["8"]["models"]
    hard = to["hardening"]
    cs = to["control_sensitivity"]["control_sets"]

    # ── Core measurement (GPT-2 124M) ──

    section("Core measurement (GPT-2 124M)")
    g = gpt2["gpt2"]
    macro("corepcorr", _fmt(g["partial_corr"]["mean"]), "GPT-2 124M peak layer pcorr")
    macro("coreoc", _fmt(g["output_controlled"]["mean"]), "GPT-2 124M OC residual")
    macro("coresagree", _fmt(g["seed_agreement"]["mean"]), "GPT-2 124M seed agreement")
    macro("corelayerzero", _fmt(g["layer_profile"]["0"], dp=2), "layer 0 pcorr")

    # ── Hardening (20-seed) ──

    section("Hardening (20-seed, GPT-2 124M layer 11)")
    macro("hardpcorr", _fmt(hard["mean_partial_corr"]), "20-seed mean")
    macro("hardstd", _fmt(hard["std_partial_corr"], sign=False), "20-seed std")
    macro("hardsagree", _fmt(np.mean(hard["seed_agreement"])), "20-seed sagree")

    # ── Control sensitivity (GPT-2 124M) ──

    section("Control sensitivity (GPT-2 124M)")
    macro("rawspearman", _fmt(cs["none"]["mean"]), "no controls")
    macro("softmaxonly", _fmt(cs["softmax_only"]["mean"]), "softmax only")
    macro("stdcontrol", _fmt(cs["standard"]["mean"]), "softmax + norm")
    macro("entropycontrol", _fmt(cs["plus_entropy"]["mean"]), "+ logit entropy")
    macro("nonlincontrol", _fmt(cs["nonlinear"]["mean"]), "nonlinear MLP")

    # ── Hand-designed baselines ──

    section("Hand-designed baselines (GPT-2 124M)")
    b = to["5c"]
    macro("ffgoodness", _fmt(b["ff_goodness"]["rho"]), "FF goodness pcorr")
    macro("actratio", _fmt(b["active_ratio"]["rho"]), "active ratio pcorr")
    macro("actentropy", _fmt(b["act_entropy"]["rho"]), "activation entropy pcorr")
    macro("actnorm", _fmt(b["activation_norm"]["rho"]), "activation norm pcorr")

    # ── GPT-2 scaling ──

    section("GPT-2 scaling")
    for model_id, name in [
        ("gpt2", "gptS"),
        ("gpt2-medium", "gptM"),
        ("gpt2-large", "gptL"),
        ("gpt2-xl", "gptXL"),
    ]:
        m = gpt2[model_id]
        macro(f"{name}pcorr", _fmt(m["partial_corr"]["mean"]))
        macro(f"{name}oc", _fmt(m["output_controlled"]["mean"]))
        macro(f"{name}sagree", _fmt(m["seed_agreement"]["mean"]))

    # Output-discarded fractions
    gS = gpt2["gpt2"]
    gXL = gpt2["gpt2-xl"]
    frac_s = round(gS["output_controlled"]["mean"] / gS["partial_corr"]["mean"] * 100)
    frac_xl = round(gXL["output_controlled"]["mean"] / gXL["partial_corr"]["mean"] * 100)
    macro("gptSdiscard", str(frac_s), "% not captured by output (124M)")
    macro("gptXLdiscard", str(frac_xl), "% not captured by output (1.5B)")

    # ── Qwen base ──

    section("Qwen base scaling")
    for data, name in [(q05, "qHalf"), (q15, "qOneFive"), (q3, "qThree"), (q7, "qSeven"), (q14, "qFourteen")]:
        macro(f"{name}pcorr", _fmt(data["partial_corr"]["mean"]))
        macro(f"{name}oc", _fmt(data["output_controlled"]["mean"]))
        macro(
            f"{name}sagree",
            _fmt(
                data["seed_agreement"]["mean"]
                if isinstance(data["seed_agreement"], dict)
                else data["seed_agreement"]
            ),
        )

    # ── Qwen instruct ──

    section("Qwen instruct")
    instruct_data = [
        (qi05, q05, "qiHalf"),
        (qi15, q15, "qiOneFive"),
        (qi3, q3, "qiThree"),
        (qi7, q7, "qiSeven"),
    ]
    for inst, base, name in instruct_data:
        ipc = inst["partial_corr"]["mean"]
        bpc = base["partial_corr"]["mean"]
        macro(f"{name}pcorr", _fmt(ipc))
        macro(f"{name}delta", _fmt(ipc - bpc))

    if qi14 is not None:
        ipc14 = qi14["partial_corr"]["mean"]
        bpc14 = q14["partial_corr"]["mean"]
        macro("qiFourteenpcorr", _fmt(ipc14))
        macro("qiFourteendelta", _fmt(ipc14 - bpc14))

    # ── Cross-family ──

    section("Cross-family (Llama, Gemma, Mistral)")
    macro("llamapcorr", _fmt(llama3b["partial_corr"]["mean"]), "Llama 3B")
    macro("llamaoc", _fmt(llama3b["output_controlled"]["mean"]))
    macro("llamaOnepcorr", _fmt(llama1b["partial_corr"]["mean"]), "Llama 1B")
    macro("llamaOneoc", _fmt(llama1b["output_controlled"]["mean"]))
    macro("llamaOneipcorr", _fmt(llama1bi["partial_corr"]["mean"]), "Llama 1B Instruct")
    macro("llamaEightpcorr", _fmt(llama8b["partial_corr"]["mean"]), "Llama 8B")
    macro("llamaEightoc", _fmt(llama8b["output_controlled"]["mean"]))
    macro("gemmapcorr", _fmt(gemma1b["partial_corr"]["mean"]), "Gemma 1B")
    macro("gemmaoc", _fmt(gemma1b["output_controlled"]["mean"]))
    macro("gemmarandom", _fmt(gemma1b["baselines"]["random_head"]))
    macro("gemmaFourpcorr", _fmt(gemma4b["partial_corr"]["mean"]), "Gemma 4B")
    macro("phipcorr", _fmt(phi3["partial_corr"]["mean"]), "Phi-3 Mini")

    if mistral is not None:
        macro("mistralpcorr", _fmt(mistral["partial_corr"]["mean"]))
        macro("mistraloc", _fmt(mistral["output_controlled"]["mean"]))

    # ── Cross-family gap ──

    section("Cross-family gap (matched 3B scale)")
    qwen3b_pc = q3["partial_corr"]["mean"]
    llama3b_pc = llama3b["partial_corr"]["mean"]
    gap = qwen3b_pc / llama3b_pc
    macro("crossfamilygap", f"{gap:.1f}", "Qwen 3B / Llama 3B ratio")

    # ── Statistical tests (computed from data) ──

    section("Statistical tests (computed from data)")
    all_models = load_all_models()
    n_models = len(all_models)
    n_families = len(set(m["family"] for m in all_models.values()))
    macro("nmodels", str(n_models), "models in analysis scope")
    macro("nfamilies", str(n_families), "families in analysis scope")

    # Permutation test (Monte Carlo, 50k samples)
    model_means = load_model_means()
    families_pm = [m[0] for m in model_means]
    log_params_pm = np.array([m[1] for m in model_means])
    pcorrs_pm = np.array([m[2] for m in model_means])
    observed_f = family_f_stat(families_pm, log_params_pm, pcorrs_pm)
    rng = np.random.RandomState(42)
    null_fs = np.array(
        [family_f_stat(list(rng.permutation(families_pm)), log_params_pm, pcorrs_pm) for _ in range(50000)]
    )
    p_value = float((null_fs >= observed_f).mean())
    macro("permF", f"{observed_f:.2f}", f"Monte Carlo 50k, {n_models} models, {n_families} families")
    macro("permp", f"{p_value:.3f}", "")

    # Eta-squared
    unique_fam = list(set(families_pm))
    resid_pm = (
        pcorrs_pm
        - np.column_stack([log_params_pm, np.ones(len(log_params_pm))])
        @ np.linalg.lstsq(
            np.column_stack([log_params_pm, np.ones(len(log_params_pm))]), pcorrs_pm, rcond=None
        )[0]
    )
    ss_between = sum(
        (np.array([f == fam for f in families_pm]).sum())
        * (resid_pm[np.array([f == fam for f in families_pm])].mean() - resid_pm.mean()) ** 2
        for fam in unique_fam
    )
    ss_total = ((resid_pm - resid_pm.mean()) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0
    macro("etasq", f"{eta_sq:.2f}", "eta-squared from permutation test")

    # Variance decomposition (three-level, descriptive)
    seed_rows = load_per_seed()
    df = pd.DataFrame(seed_rows, columns=["family", "model", "params_b", "seed_idx", "partial_corr"])
    grand_mean = df["partial_corr"].mean()
    total_var = df["partial_corr"].var(ddof=0)

    family_means_df = df.groupby("family")["partial_corr"].mean()
    family_ns = df.groupby("family").size()
    var_between_family = sum(n * (m - grand_mean) ** 2 for m, n in zip(family_means_df, family_ns)) / len(df)

    model_means_df = df.groupby("model")["partial_corr"].mean()
    var_between_model = 0.0
    for model_name, model_mean in model_means_df.items():
        fam = df[df["model"] == model_name]["family"].iloc[0]
        fam_mean = family_means_df[fam]
        n = (df["model"] == model_name).sum()
        var_between_model += n * (model_mean - fam_mean) ** 2
    var_between_model /= len(df)

    var_within = 0.0
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name]
        var_within += ((sub["partial_corr"] - sub["partial_corr"].mean()) ** 2).sum()
    var_within /= len(df)

    pct_fam = var_between_family / total_var * 100
    pct_model = var_between_model / total_var * 100
    pct_seed = var_within / total_var * 100

    macro("varfamily", _pct(pct_fam), "between families")
    macro("varmodel", _pct(pct_model), "between models within family")
    macro("varseed", _pct(pct_seed), "within model across seeds")

    # Mixed-effects model
    df["log_params"] = np.log10(df["params_b"])
    md = smf.mixedlm("partial_corr ~ log_params + C(family)", df, groups=df["model"])
    mdf = md.fit(reml=True)
    # Extract Llama coefficient (reference family varies; find it)
    llama_key = [k for k in mdf.params.index if "Llama" in k]
    gemma_key = [k for k in mdf.params.index if "Gemma" in k]
    macro("llamacoef", _fmt(mdf.params[llama_key[0]]) if llama_key else "---", "")
    macro("llamaz", f"{mdf.tvalues[llama_key[0]]:.2f}" if llama_key else "---", "")
    macro("gemmacoef", _fmt(mdf.params[gemma_key[0]]) if gemma_key else "---", "")
    macro("gemmaz", f"{mdf.tvalues[gemma_key[0]]:.2f}" if gemma_key else "---", "")
    macro("scalecoef", _fmt(mdf.params["log_params"]), "")
    macro("scalez", f"{mdf.tvalues['log_params']:.2f}", "")
    macro("scalep", f"{mdf.pvalues['log_params']:.3f}", "")

    # ── Catch rates ──

    section("Exclusive catch rates")
    macro("catchfloor", "8", "Llama at 10% flag rate, rounded")
    macro("catchceiling", "11", "Mistral at 10% flag rate, rounded")
    macro("catchsaturation", "11--15", "range at 20% flag rate")

    # ── Downstream task metrics ──

    section("Downstream task metrics")
    tqa = _load_json("truthfulqa_hallucination_results.json")
    tqa_auc = tqa.get("confident_hallucination_catches", {}).get("auc_among_confident")
    if tqa_auc is not None:
        macro("tqaconfAUC", f"{tqa_auc:.3f}", "TruthfulQA AUC among confident-wrong")

    # ── SAE comparison ──

    section("SAE comparison (GPT-2 124M)")
    sae = _load_json("sae_compare.json")
    macro("saepcorr", _fmt(sae["7a"]["sae"]["mean"]))
    macro("saerankcorr", _fmt(sae["7c"]["mean_rank_correlation"], dp=2))

    # ── Bootstrap ──

    section("Document-level bootstrap (Qwen 7B)")
    boot_path = REPO_ROOT / "analysis" / "split_bootstrap_Qwen2.5-7B.json"
    boot = json.loads(boot_path.read_text())
    macro("bootmean", _fmt(round(boot["mean"], 3)), f"{boot['n_boot']}-resample mean")
    macro("bootlo", _fmt(round(boot["ci_95"][0], 3)), "95% CI lower")
    macro("boothi", _fmt(round(boot["ci_95"][1], 3)), "95% CI upper")

    # ── Token budget ──

    section("Token budget (Qwen 14B v1-v3 progression)")
    q14_vh = q14.get("version_history", {})
    macro("budgetvone", _fmt(q14_vh["v1"]["partial_corr"]), "68 ex/dim")
    macro("budgetvtwo", _fmt(q14_vh["v2"]["partial_corr"]), "250 ex/dim")
    macro("budgetvthree", _fmt(q14["partial_corr"]["mean"]), "350 ex/dim")

    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Check if data_macros.sty matches generated output (no write)"
    )
    args = parser.parse_args()

    content = generate_macros()
    out = PAPER_DIR / "data_macros.sty"
    n_macros = content.count("\\newcommand")

    if args.check:
        if not out.exists():
            print(f"FAIL: {out} does not exist")
            sys.exit(1)
        existing = out.read_text()
        if existing == content:
            print(f"OK: {out} matches generated output ({n_macros} macros)")
        else:
            # Show which macros differ
            import difflib

            diff = list(
                difflib.unified_diff(
                    existing.splitlines(),
                    content.splitlines(),
                    fromfile="committed",
                    tofile="generated",
                    lineterm="",
                )
            )
            for line in diff[:40]:
                print(line)
            if len(diff) > 40:
                print(f"  ... ({len(diff) - 40} more lines)")
            print(f"\nFAIL: {out} does not match generated output")
            print("  Run 'just data-macros' to regenerate")
            sys.exit(1)
    else:
        out.write_text(content)
        print(f"Generated {n_macros} macros -> {out}")


if __name__ == "__main__":
    main()
