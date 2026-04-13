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

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT.parent / "nn-observability-paper"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from load_results import load_all_models


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
    llama = _load_json("llama3b_v2_results.json")
    gemma = _load_json("gemma3_1b_results.json")

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
    macro("llamapcorr", _fmt(llama["partial_corr"]["mean"]))
    macro("llamaoc", _fmt(llama["output_controlled"]["mean"]))
    macro("gemmapcorr", _fmt(gemma["partial_corr"]["mean"]))
    macro("gemmaoc", _fmt(gemma["output_controlled"]["mean"]))
    macro("gemmarandom", _fmt(gemma["baselines"]["random_head"]))

    if mistral is not None:
        macro("mistralpcorr", _fmt(mistral["partial_corr"]["mean"]))
        macro("mistraloc", _fmt(mistral["output_controlled"]["mean"]))

    # ── Cross-family gap ──

    section("Cross-family gap (matched 3B scale)")
    qwen3b_pc = q3["partial_corr"]["mean"]
    llama3b_pc = llama["partial_corr"]["mean"]
    gap = qwen3b_pc / llama3b_pc
    macro("crossfamilygap", f"{gap:.1f}", "Qwen 3B / Llama 3B ratio")

    # ── Statistical tests ──
    # These are hardcoded from analysis script output.
    # Re-run analysis/run_all.py if model scope changes.

    section("Statistical tests (from analysis scripts)")
    all_models = load_all_models()
    n_models = len(all_models)
    n_families = len(set(m["family"] for m in all_models.values()))
    macro("nmodels", str(n_models), "models in analysis scope")
    macro("nfamilies", str(n_families), "families in analysis scope")

    # Permutation test
    macro("permF", "13.57", "re-run permutation_test.py if scope changes")
    macro("permp", "0.014", "re-run permutation_test.py if scope changes")

    # Variance decomposition
    macro("varfamily", "87.8", "re-run meta_regression.py if scope changes")
    macro("varmodel", "6.0", "")
    macro("varseed", "6.3", "")

    # Mixed effects coefficients
    macro("llamacoef", "-0.196", "")
    macro("llamaz", "-6.42", "")
    macro("gemmacoef", "+0.102", "")
    macro("gemmaz", "3.68", "")
    macro("scalecoef", "-0.001", "")
    macro("scalez", "-0.06", "")
    macro("scalep", "0.950", "")

    # ── Catch rates ──

    section("Exclusive catch rates")
    macro("catchfloor", "7", "Llama at 10% flag rate, rounded")
    macro("catchceiling", "11", "Mistral at 10% flag rate, rounded")
    macro("catchsaturation", "12--15", "range at 20% flag rate")

    # ── SAE comparison ──

    section("SAE comparison (GPT-2 124M)")
    sae = _load_json("sae_compare.json")
    macro("saepcorr", _fmt(sae["7a"]["sae"]["mean"]))
    macro("saerankcorr", _fmt(sae["7c"]["mean_rank_correlation"], dp=2))

    # ── Bootstrap ──

    section("Document-level bootstrap (Qwen 7B)")
    macro("bootmean", _fmt(0.238), "30-resample mean")
    macro("bootlo", _fmt(0.215), "95% CI lower")
    macro("boothi", _fmt(0.270), "95% CI upper")

    # ── Token budget ──

    section("Token budget (Qwen 14B v1-v3 progression)")
    macro("budgetvone", _fmt(0.194), "68 ex/dim")
    macro("budgetvtwo", _fmt(0.212), "250 ex/dim")
    macro("budgetvthree", _fmt(0.214), "350 ex/dim")

    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    content = generate_macros()
    out = PAPER_DIR / "data_macros.sty"
    out.write_text(content)
    n_macros = content.count("\\newcommand")
    print(f"Generated {n_macros} macros -> {out}")


if __name__ == "__main__":
    main()
