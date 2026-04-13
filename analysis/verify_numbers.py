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

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_DIR = REPO_ROOT.parent / "nn-observability-paper"


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
    llama = _load_json("llama3b_v2_results.json")
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

    # Statistical tests (hardcoded expected values; re-run analysis scripts if these change)
    checks.append(Check("Permutation F", "sections/architecture.tex", 13.57, 0.005))
    checks.append(Check("Permutation p", "sections/architecture.tex", 0.014, 0.0005))
    checks.append(Check("Variance between families %", "sections/architecture.tex", 87.8, 0.5))

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
