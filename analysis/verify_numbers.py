"""Verify inline numbers in .tex files against results JSONs.

Extracts numerical claims from the paper and cross-references them
against the authoritative JSON results. Reports mismatches without
changing any files.

Usage: cd nn-observability && uv run python analysis/verify_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT.parent / "nn-observability-paper"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from load_results import load_model_means
from permutation_test import family_f_stat


@dataclass
class Check:
    description: str
    tex_file: str
    expected: float
    tolerance: float


def _load_json(filename: str) -> dict:
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


def _load_macros() -> dict[str, str]:
    """Parse data_macros.sty into a name->value dict."""
    macros = {}
    path = PAPER_DIR / "data_macros.sty"
    if not path.exists():
        return macros
    for line in path.read_text().splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", line)
        if m:
            macros[m.group(1)] = m.group(2)
    return macros


_MACROS: dict[str, str] = {}


def _expand_macros(text: str) -> str:
    """Expand data macros in tex content for verification."""
    global _MACROS
    if not _MACROS:
        _MACROS = _load_macros()
    for name, val in _MACROS.items():
        text = text.replace(f"\\{name}", val)
    return text


def _tex_content(rel_path: str) -> str:
    raw = (PAPER_DIR / rel_path).read_text()
    return _expand_macros(raw)


def _find_number(text: str, pattern: str) -> float | None:
    """Find a number in tex content matching a regex pattern."""
    m = re.search(pattern, text)
    if m is None:
        return None
    val = m.group(1) if m.lastindex else m.group(0)
    val = val.replace("\\%", "").replace("%", "").replace(",", "")
    return float(val)


# ── Build checks from data ──────────────────────────────────────────


def build_checks() -> list[Check]:
    checks = []

    # Load data sources
    to = _load_json("transformer_observe.json")
    q7 = _load_json("qwen7b_v3_results.json")
    q14 = _load_json("qwen14b_v3_results.json")
    q3 = _load_json("qwen3b_v3_results.json")
    q15 = _load_json("qwen1_5b_v3_results.json")
    q05 = _load_json("qwen05b_v3_results.json")
    llama = _load_json("llama3b_v3_results.json")
    gemma = _load_json("gemma3_1b_results.json")

    # GPT-2 124M core
    gpt2 = to["8"]["models"]["gpt2"]
    checks.append(Check("GPT-2 124M pcorr", "tables/gpt2_scaling.tex", gpt2["partial_corr"]["mean"], 0.0005))
    checks.append(
        Check("GPT-2 124M OC", "tables/gpt2_scaling.tex", gpt2["output_controlled"]["mean"], 0.0005)
    )
    checks.append(
        Check("GPT-2 124M sagree", "tables/gpt2_scaling.tex", gpt2["seed_agreement"]["mean"], 0.0005)
    )

    # GPT-2 XL
    gpt2xl = to["8"]["models"]["gpt2-xl"]
    checks.append(Check("GPT-2 XL pcorr", "tables/gpt2_scaling.tex", gpt2xl["partial_corr"]["mean"], 0.0005))
    checks.append(
        Check("GPT-2 XL OC", "tables/gpt2_scaling.tex", gpt2xl["output_controlled"]["mean"], 0.0005)
    )

    # Hardening
    h = to["hardening"]
    checks.append(Check("Hardening pcorr mean", "sections/signal.tex", h["mean_partial_corr"], 0.0005))
    checks.append(Check("Hardening pcorr std", "sections/signal.tex", h["std_partial_corr"], 0.0005))

    # Control sensitivity
    cs = to["control_sensitivity"]["control_sets"]
    checks.append(Check("Raw Spearman", "sections/signal.tex", cs["none"]["mean"], 0.0005))
    checks.append(Check("Standard control", "tables/control_sensitivity.tex", cs["standard"]["mean"], 0.0005))
    checks.append(
        Check("Nonlinear control", "tables/control_sensitivity.tex", cs["nonlinear"]["mean"], 0.0005)
    )
    checks.append(
        Check("Entropy control", "tables/control_sensitivity.tex", cs["plus_entropy"]["mean"], 0.0005)
    )

    # Hand-designed baselines
    b = to["5c"]
    checks.append(Check("FF goodness", "tables/hand_designed_baselines.tex", b["ff_goodness"]["rho"], 0.0005))
    checks.append(
        Check(
            "Activation norm baseline",
            "tables/hand_designed_baselines.tex",
            b["activation_norm"]["rho"],
            0.0005,
        )
    )

    # Cross-family pcorr values
    checks.append(
        Check("Qwen 0.5B pcorr", "tables/cross_family_scaling.tex", q05["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Qwen 1.5B pcorr", "tables/cross_family_scaling.tex", q15["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Qwen 3B pcorr", "tables/cross_family_scaling.tex", q3["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Qwen 7B pcorr", "tables/cross_family_scaling.tex", q7["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Qwen 14B pcorr", "tables/cross_family_scaling.tex", q14["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Llama 3B pcorr", "tables/cross_family_scaling.tex", llama["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Gemma 1B pcorr", "tables/cross_family_scaling.tex", gemma["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check(
            "Gemma 1B random head",
            "tables/cross_family_scaling.tex",
            gemma["baselines"]["random_head"],
            0.0005,
        )
    )

    # OC residuals
    checks.append(
        Check("Qwen 7B OC", "tables/cross_family_scaling.tex", q7["output_controlled"]["mean"], 0.0005)
    )
    checks.append(
        Check("Llama 3B OC", "tables/cross_family_scaling.tex", llama["output_controlled"]["mean"], 0.0005)
    )

    # Downstream tasks (inline table in architecture.tex)
    rag = _load_json("rag_hallucination_results.json")
    rag_summary = rag.get("summary", {})
    checks.append(
        Check("SQuAD accuracy", "sections/architecture.tex", round(rag_summary["accuracy"] * 100), 1)
    )
    checks.append(
        Check("SQuAD @10%", "sections/architecture.tex", rag["flag_rates"]["0.1"]["pct_of_errors"], 0.1)
    )
    checks.append(
        Check("SQuAD @20%", "sections/architecture.tex", rag["flag_rates"]["0.2"]["pct_of_errors"], 0.1)
    )

    medqa = _load_json("medqa_selective_results.json")
    checks.append(Check("MedQA accuracy", "sections/architecture.tex", round(medqa["accuracy"] * 100), 1))
    checks.append(
        Check("MedQA @10%", "sections/architecture.tex", medqa["flag_rates"]["0.1"]["pct_of_errors"], 0.1)
    )
    checks.append(
        Check("MedQA @20%", "sections/architecture.tex", medqa["flag_rates"]["0.2"]["pct_of_errors"], 0.1)
    )

    tqa = _load_json("truthfulqa_hallucination_results.json")
    checks.append(Check("TruthfulQA accuracy", "sections/architecture.tex", round(tqa["accuracy"] * 100), 1))
    tqa_catches = tqa.get("standard_catches", {})
    checks.append(
        Check("TruthfulQA @10%", "sections/architecture.tex", tqa_catches["0.1"]["pct_of_errors"], 0.1)
    )
    checks.append(
        Check("TruthfulQA @20%", "sections/architecture.tex", tqa_catches["0.2"]["pct_of_errors"], 0.1)
    )

    # Statistical tests (computed from data, not hardcoded)
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
    checks.append(Check("Permutation F", "sections/architecture.tex", round(observed_f, 2), 0.05))
    checks.append(Check("Permutation p", "sections/architecture.tex", round(p_value, 3), 0.001))

    # Variance decomposition (eta-squared on model means, percentage form)
    unique_fam = list(set(families_pm))
    X = np.column_stack([log_params_pm, np.ones(len(log_params_pm))])
    resid_pm = pcorrs_pm - X @ np.linalg.lstsq(X, pcorrs_pm, rcond=None)[0]
    ss_between = sum(
        np.array([f == fam for f in families_pm]).sum()
        * (resid_pm[np.array([f == fam for f in families_pm])].mean() - resid_pm.mean()) ** 2
        for fam in unique_fam
    )
    ss_total = ((resid_pm - resid_pm.mean()) ** 2).sum()
    eta_pct = ss_between / ss_total * 100 if ss_total > 0 else 0
    checks.append(Check("Variance between families %", "sections/architecture.tex", round(eta_pct, 1), 0.5))

    # Extended cross-family table (models not in stat scope)
    llama1b = _load_json("llama1b_results.json")
    llama8b = _load_json("llama8b_results.json")
    gemma4b = _load_json("gemma4b_results.json")
    phi3 = _load_json("phi3_mini_results.json")
    mistral = _load_json("mistral7b_results.json")
    checks.append(
        Check("Llama 1B pcorr", "tables/cross_family_scaling.tex", llama1b["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Llama 8B pcorr", "tables/cross_family_scaling.tex", llama8b["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Llama 8B OC", "tables/cross_family_scaling.tex", llama8b["output_controlled"]["mean"], 0.001)
    )
    checks.append(
        Check("Gemma 4B pcorr", "tables/cross_family_scaling.tex", gemma4b["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Phi-3 pcorr", "tables/cross_family_scaling.tex", phi3["partial_corr"]["mean"], 0.0005)
    )
    checks.append(
        Check("Mistral 7B pcorr", "tables/cross_family_scaling.tex", mistral["partial_corr"]["mean"], 0.0005)
    )

    # Llama 1B instruct
    llama1bi = _load_json("llama1b_instruct_results.json")
    checks.append(
        Check(
            "Llama 1B Instruct pcorr", "sections/architecture.tex", llama1bi["partial_corr"]["mean"], 0.0005
        )
    )

    # Per-seed ranges (architecture.tex)
    l3_seeds = llama["partial_corr"]["per_seed"]
    q3_seeds = q3["partial_corr"]["per_seed"]
    checks.append(Check("Llama 3B seed min", "sections/architecture.tex", round(min(l3_seeds), 3), 0.001))
    checks.append(Check("Llama 3B seed max", "sections/architecture.tex", round(max(l3_seeds), 3), 0.001))
    checks.append(Check("Qwen 3B seed min", "sections/architecture.tex", round(min(q3_seeds), 3), 0.001))
    checks.append(Check("Qwen 3B seed max", "sections/architecture.tex", round(max(q3_seeds), 3), 0.001))

    # Shuffle test
    shuffle = _load_json("shuffle_test_gpt2.json")
    checks.append(Check("Shuffle mean", "sections/signal.tex", round(shuffle["shuffle_mean"], 3), 0.002))
    checks.append(Check("Shuffle std", "sections/signal.tex", round(shuffle["shuffle_std"], 3), 0.002))
    checks.append(Check("Shuffle real pcorr", "sections/signal.tex", round(shuffle["real_pcorr"], 3), 0.002))

    # Width sweep (signal.tex)
    roc = _load_json("roc_width_sweep_results.json")
    for w in ["64", "512"]:
        val = roc["results"][f"width_{w}"]["mean"]
        checks.append(Check(f"Width sweep {w}", "sections/signal.tex", round(val, 3), 0.002))

    # TruthfulQA AUC among confident-wrong
    tqa_auc = tqa.get("confident_hallucination_catches", {}).get("auc_among_confident")
    if tqa_auc is not None:
        checks.append(Check("TruthfulQA conf AUC", "sections/architecture.tex", tqa_auc, 0.001))

    # Cross-domain transfer
    gpt2_cd = to.get("cross_domain", {}).get("domains", {})
    if "code" in gpt2_cd:
        cd_val = gpt2_cd["code"].get("pcorr", gpt2_cd["code"].get("mean"))
        if cd_val is not None:
            checks.append(
                Check("GPT-2 CodeSearchNet", "appendix/appendix_cross_domain.tex", round(cd_val, 3), 0.002)
            )

    # Hardening seed agreement
    checks.append(
        Check(
            "Hardening sagree",
            "appendix/appendix_methodology.tex",
            to["hardening"]["mean_seed_agreement"],
            0.001,
        )
    )

    # GPT-2 flagging (computed from per_seed data)
    gpt2_6a = to.get("6a", {})
    if "per_seed" in gpt2_6a:
        n_tok = gpt2_6a.get("n_test_tokens", gpt2_6a.get("n_tokens", 0))
        total = n_tok * 0.5
        for rate in ["0.1", "0.2"]:
            obs = float(np.mean([s["exclusive"][rate]["observer_only"] for s in gpt2_6a["per_seed"]]))
            pct = round(obs / total * 100, 1)
            checks.append(Check(f"GPT-2 flagging @{rate}", "tables/flagging_cross_scale.tex", pct, 0.2))

    # Qwen 0.5B instruct delta
    qi05 = _load_json("qwen05b_instruct_v3_results.json")
    delta_05 = qi05["partial_corr"]["mean"] - q05["partial_corr"]["mean"]
    checks.append(
        Check("Qwen 0.5B instruct delta", "tables/instruct_comparison.tex", round(delta_05, 3), 0.002)
    )

    return checks


# ── Verification logic ───────────────────────────────────────────────


def verify_check(check: Check) -> tuple[bool, str]:
    """Verify a single check. Returns (passed, message)."""
    tex = _tex_content(check.tex_file)

    # Format the expected value as it would appear in the tex
    exp = check.expected
    abs_exp = abs(exp)

    found = False
    tex_val = None

    # Try to find the number in various formats
    if abs_exp >= 1:
        # Large numbers: look for the value with 1-2 decimal places
        for fmt in [f"{exp:.2f}", f"{exp:.1f}", f"{exp:.0f}"]:
            if fmt in tex:
                found = True
                tex_val = float(fmt)
                break
    else:
        # Small numbers: look for 3 decimal place representation
        for sign in [f"{exp:+.3f}", f"{abs_exp:.3f}", f"{exp:.3f}"]:
            if sign in tex:
                found = True
                tex_val = exp  # exact match
                break

    if not found:
        return False, f"NOT FOUND in {check.tex_file}"

    delta = abs(tex_val - exp) if tex_val is not None else 0
    if delta > check.tolerance:
        return False, f"MISMATCH tex={tex_val} json={exp:.6f} delta={delta:.6f}"

    return True, f"tex={tex_val} json={exp:.6f}"


def main():
    if not PAPER_DIR.exists():
        print(f"ERROR: paper directory not found: {PAPER_DIR}")
        sys.exit(1)

    checks = build_checks()
    print(f"Verifying {len(checks)} numbers across paper .tex files\n")

    passed = 0
    failed = 0
    not_found = 0

    for check in checks:
        ok, msg = verify_check(check)
        status = "PASS" if ok else "FAIL"
        symbol = " " if ok else "!"
        print(f"  [{status}]{symbol} {check.description:<32} {msg}")
        if ok:
            passed += 1
        elif "NOT FOUND" in msg:
            not_found += 1
            failed += 1
        else:
            failed += 1

    print(f"\nSummary: {passed}/{len(checks)} passed", end="")
    if failed:
        print(f", {failed} failed", end="")
        if not_found:
            print(f" ({not_found} not found in tex)", end="")
    print()

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
