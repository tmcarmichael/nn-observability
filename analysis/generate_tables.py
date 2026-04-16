"""Generate paper tables from results JSONs.

Produces the three data-dependent tables that change when new model
families are added. Outputs go to the paper repo tables/ directory.

Usage: cd nn-observability && uv run python analysis/generate_tables.py
       cd nn-observability && uv run python analysis/generate_tables.py --check
   or: cd nn-observability && just tables
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PAPER_TABLES = REPO_ROOT.parent / "nn-observability-paper" / "tables"

sys.path.insert(0, str(REPO_ROOT / "analysis"))
from load_results import load_all_models

# ── Formatting helpers ───────────────────────────────────────────────


def _fmt_pcorr(val: float) -> str:
    return f"${val:+.3f}$"


def _fmt_std(val: float) -> str:
    return f"${val:.3f}$"


def _fmt_oc(val: float) -> str:
    return f"${val:+.3f}$"


def _fmt_sagree(val: float) -> str:
    return f"${val:+.3f}$"


def _fmt_random_head(val: float | None) -> str:
    if val is None:
        return "---"
    return f"${val:+.3f}$"


def _fmt_peak_layer(peak: int, frac: float) -> str:
    return f"L{peak} ({frac * 100:.0f}\\%)"


def _fmt_catch_rate(pct: float) -> str:
    return f"{pct:.1f}\\%"


# ── Data loading (supplement load_results with full JSON fields) ─────


def _load_gpt2_full() -> dict:
    """Load GPT-2 models from transformer_observe.json phase 8."""
    path = RESULTS_DIR / "transformer_observe.json"
    with open(path) as f:
        data = json.load(f)
    return data["8"]["models"]


def _load_model_json(label: str, all_models: dict) -> dict | None:
    """Load the full JSON for a non-GPT-2 model."""
    info = all_models.get(label)
    fname = None
    if info is not None:
        fname = info.get("source_file")
    if fname is None:
        fname = TABLE_EXTRA_SOURCES.get(label)
    if fname is None:
        return None
    path = RESULTS_DIR / fname
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Table 1: cross_family_scaling ────────────────────────────────────

CROSS_FAMILY_ROWS = [
    {"label": "Qwen-0.5B", "display": r"\qwen{} 0.5B", "family": "Qwen", "params": "0.5B"},
    {"label": "Gemma-1B", "display": "Gemma~3 1B", "family": "Gemma", "params": "1B"},
    {"label": "Llama-1B", "display": r"\llama{} 3.2 1B", "family": "Llama", "params": "1.2B"},
    {"label": "Qwen-1.5B", "display": r"\qwen{} 1.5B", "family": "Qwen", "params": "1.5B"},
    {"label": "GPT2-1.5B", "display": r"\gpt{} XL", "family": "GPT-2", "params": "1.5B"},
    {"label": "Qwen-3B", "display": r"\qwen{} 3B", "family": "Qwen", "params": "3B"},
    {"label": "Llama-3B", "display": r"\llama{} 3.2 3B", "family": "Llama", "params": "3B"},
    {"label": "Phi-3-Mini", "display": "Phi-3 Mini", "family": "Phi", "params": "3.8B"},
    {"label": "Gemma-4B", "display": "Gemma~3 4B", "family": "Gemma", "params": "4.3B"},
    {"label": "Mistral-7B", "display": "Mistral 7B", "family": "Mistral", "params": "7B"},
    {"label": "Qwen-7B", "display": r"\qwen{} 7B", "family": "Qwen", "params": "7B"},
    {"label": "Llama-8B", "display": r"\llama{} 3.1 8B", "family": "Llama", "params": "8B"},
    {"label": "Qwen-14B", "display": r"\qwen{} 14B", "family": "Qwen", "params": "14B"},
]

# Models in the table but not in load_results.py's statistical scope.
# These need direct JSON paths for loading.
TABLE_EXTRA_SOURCES = {
    "Llama-1B": "llama1b_results.json",
    "Llama-8B": "llama8b_results.json",
    "Gemma-4B": "gemma4b_results.json",
    "Phi-3-Mini": "phi3_mini_results.json",
}

GPT2_ID_MAP = {
    "GPT2-124M": "gpt2",
    "GPT2-355M": "gpt2-medium",
    "GPT2-774M": "gpt2-large",
    "GPT2-1.5B": "gpt2-xl",
}


def _get_full_fields(label: str, all_models: dict, gpt2_data: dict) -> dict | None:
    """Extract peak_layer, peak_layer_frac, oc, sagree, random_head for a model."""
    gpt2_id = GPT2_ID_MAP.get(label)
    if gpt2_id:
        m = gpt2_data.get(gpt2_id)
        if m is None:
            return None
        return {
            "peak_layer": m.get("peak_layer", m.get("peak_layer_final", 0)),
            "peak_layer_frac": m.get("peak_layer_frac", 0),
            "oc": m.get("output_controlled", {}).get("mean"),
            "sagree": m.get("seed_agreement", {}).get("mean"),
            "random_head": m.get("baselines", {}).get("random_head"),
        }

    raw = _load_model_json(label, all_models)
    if raw is None:
        return None
    sa = raw.get("seed_agreement", {})
    sa_mean = sa.get("mean") if isinstance(sa, dict) else sa
    return {
        "peak_layer": raw.get("peak_layer_final", raw.get("peak_layer", 0)),
        "peak_layer_frac": raw.get("peak_layer_frac", 0),
        "oc": raw.get("output_controlled", {}).get("mean"),
        "sagree": sa_mean,
        "random_head": raw.get("baselines", {}).get("random_head"),
    }


def generate_cross_family_scaling() -> str:
    all_models = load_all_models()
    gpt2_data = _load_gpt2_full()

    rows = []
    for cfg in CROSS_FAMILY_ROWS:
        label = cfg["label"]
        info = all_models.get(label)
        if info is None and label in TABLE_EXTRA_SOURCES:
            raw = _load_model_json(label, all_models)
            if raw is not None:
                info = {"partial_corr": raw.get("partial_corr", {})}
        if info is None:
            print(f"  WARNING: {label} not found, skipping row")
            continue
        full = _get_full_fields(label, all_models, gpt2_data)
        if full is None:
            print(f"  WARNING: {label} full data not found, skipping row")
            continue

        pc = info["partial_corr"]
        pcorr = pc.get("mean", 0)
        std = pc.get("std", 0)
        rh = full["random_head"]
        # Suppress random_head for models where it wasn't in the original table
        if label in ("GPT2-1.5B", "Qwen-7B"):
            rh = None

        row = (
            f"{cfg['display']:<23} & {cfg['family']:<7} & {cfg['params']:<5} "
            f"& {_fmt_peak_layer(full['peak_layer'], full['peak_layer_frac']):<13} "
            f"& {_fmt_pcorr(pcorr)} & {_fmt_std(std)} "
            f"& {_fmt_oc(full['oc'])} & {_fmt_sagree(full['sagree'])} "
            f"& {_fmt_random_head(rh)} \\\\"
        )
        rows.append(row)

    body = "\n".join(rows)

    return (
        r"""\begin{table}[t]
\centering
\caption{Observability across six architecture families under identical
evaluation protocol. Token budgets scaled by hidden dimension (350 ex/dim,
600 for \qwen{} 0.5B). \llama{} 1B matches high-observability families;
3B and 8B drop to near the detection floor.}
\label{tab:cross_family_scaling}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llccccccc}
\toprule
Model & Family & Params & Peak layer & $\pcorr$\textsuperscript{a} & $\pm$ std & $\ocresid$ & $\sagree$ & Rand.\ head \\
\midrule
"""
        + body
        + r"""
\bottomrule
\multicolumn{9}{l}{\textsuperscript{a}\footnotesize Validation split (held-out seeds, $n = 6$--$7$). Test split confirms within 5\% (\cref{sec:appendix_methodology}).} \\
\multicolumn{9}{l}{\footnotesize Gemma 1B random head $+0.213$ reflects representation geometry (absent at 4B). Llama 8B OC from 2/3 seeds.} \\
\multicolumn{9}{l}{\footnotesize \llama{} 3B/8B peak at L0/L1: no layer exceeds $+0.12$; reported peak is argmax of a flat profile.} \\
\end{tabular}%
}
\end{table}
"""
    )


# ── Table 2: gpt2_scaling ───────────────────────────────────────────

GPT2_ROWS = [
    {"id": "gpt2", "display": r"\gpt{}", "params": "124M"},
    {"id": "gpt2-medium", "display": r"\gpt{} Medium", "params": "355M"},
    {"id": "gpt2-large", "display": r"\gpt{} Large", "params": "774M"},
    {"id": "gpt2-xl", "display": r"\gpt{} XL", "params": "1558M"},
]


def generate_gpt2_scaling() -> str:
    gpt2_data = _load_gpt2_full()

    rows = []
    for cfg in GPT2_ROWS:
        m = gpt2_data[cfg["id"]]
        pcorr = m["partial_corr"]["mean"]
        std = m["partial_corr"].get("std", 0)
        oc = m["output_controlled"]["mean"]
        sagree = m["seed_agreement"]["mean"]
        peak = m.get("peak_layer", m.get("peak_layer_final", 0))
        frac = m.get("peak_layer_frac", 0)
        ratio = round(oc / pcorr * 100)

        row = (
            f"{cfg['display']:<14} & {cfg['params']:<5} "
            f"& {_fmt_peak_layer(peak, frac):<11} "
            f"& {_fmt_pcorr(pcorr)} & {_fmt_std(std)} "
            f"& {_fmt_oc(oc)} & {ratio}\\% & {_fmt_sagree(sagree)} \\\\"
        )
        rows.append(row)

    body = "\n".join(rows)

    return (
        r"""\begin{table}[t]
\centering
\caption{\gpt{} scaling curve. Partial correlation is stable across
12$\times$ scale. The output-independent component ($\ocresid$) grows
monotonically; the fraction of signal not captured by the output layer
rises from \gptSdiscard\% at 124M to \gptXLdiscard\% at 1.5B.}
\label{tab:gpt2_scaling}
\begin{tabular}{lccccccc}
\toprule
Model & Params & Peak layer & $\pcorr$ & $\pm$ std & $\ocresid$ & $\ocresid / \pcorr$ & $\sagree$ \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )


# ── Table 3: flagging_cross_scale ────────────────────────────────────

FLAGGING_ROWS = [
    {"label": "Mistral-7B", "display": "Mistral 7B", "family": "Mistral"},
    {"label": "GPT2-124M", "display": r"\gpt{} 124M", "family": "GPT-2"},
    {"label": "Qwen-7B", "display": r"\qwen{} 7B", "family": "Qwen"},
    {"label": "Qwen-14B", "display": r"\qwen{} 14B", "family": "Qwen"},
    {"label": "Llama-3B", "display": r"\llama{} 3.2 3B", "family": "Llama"},
]

FLAGGING_SOURCES = {
    "GPT2-124M": ("transformer_observe.json", "6a"),
    "Qwen-7B": ("qwen7b_v3_results.json", "flagging_6a"),
    "Qwen-14B": ("qwen14b_v3_results.json", "flagging_6a"),
    "Llama-3B": ("llama3b_v3_results.json", "flagging_6a"),
    "Mistral-7B": ("mistral7b_results.json", "flagging_6a"),
}

FLAG_RATES = ["0.05", "0.1", "0.2", "0.3"]


def _compute_catch_rates(flagging: dict) -> dict[str, float]:
    """Compute exclusive catch rates at each flag rate.

    Returns dict mapping rate string to percentage of all errors.
    Handles both 'summary' and 'per_seed' data formats.
    """
    n_tokens = flagging.get("n_tokens") or flagging.get("n_test_tokens")
    total_errors = n_tokens * 0.5
    rates = {}

    if "per_seed" in flagging:
        for rate in FLAG_RATES:
            obs_only = np.mean([s["exclusive"][rate]["observer_only"] for s in flagging["per_seed"]])
            rates[rate] = obs_only / total_errors * 100
    elif "summary" in flagging:
        for rate in FLAG_RATES:
            entry = flagging["summary"].get(rate, {})
            if isinstance(entry, dict):
                obs_only = entry.get("observer_exclusive", 0)
            else:
                obs_only = entry
            rates[rate] = obs_only / total_errors * 100

    return rates


def generate_flagging_cross_scale() -> str:
    all_models = load_all_models()

    rows = []
    for cfg in FLAGGING_ROWS:
        label = cfg["label"]
        info = all_models.get(label)
        if info is None:
            print(f"  WARNING: {label} not found, skipping flagging row")
            continue

        source_file, source_key = FLAGGING_SOURCES[label]
        path = RESULTS_DIR / source_file
        with open(path) as f:
            data = json.load(f)

        flagging = data[source_key] if source_key != "6a" else data.get("6a", data)
        catch = _compute_catch_rates(flagging)
        pcorr = info["partial_corr"]["mean"]

        rate_cells = " & ".join(f"{catch[r]:5.1f}\\%" if r in catch else "---" for r in FLAG_RATES)
        # Right-align single-digit percentages at 10% column
        rate_cells = rate_cells.replace("  9.8\\%", " 9.8\\%")
        rate_cells = rate_cells.replace("  7.8\\%", " 7.8\\%")

        row = f"{cfg['display']:<17} & {cfg['family']:<7} & {_fmt_pcorr(pcorr)} & {rate_cells} \\\\"
        rows.append(row)

    body = "\n".join(rows)

    return (
        r"""\begin{table}[t]
\centering
\caption{Exclusive error catches (observer finds, confidence misses) as a
percentage of all errors, across four architecture families at four flag
rates. The catch rate increases with $\pcorr$ at low flag rates but
converges to \catchsaturation\% at 20\%, indicating a ceiling set by the error
structure rather than by observability.}
\label{tab:flagging_cross_scale}
\begin{tabular}{llccccc}
\toprule
Model & Family & $\pcorr$ & 5\% & 10\% & 20\% & 30\% \\
\midrule
"""
        + body
        + r"""
\bottomrule
\multicolumn{7}{l}{\footnotesize All values are 3-seed means. Flag rate is the fraction of tokens flagged by each monitor.} \\
\end{tabular}
\end{table}
"""
    )


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check tables match generated output (no write)")
    args = parser.parse_args()

    if not PAPER_TABLES.exists():
        print(f"ERROR: paper tables directory not found: {PAPER_TABLES}")
        sys.exit(1)

    generators = [
        ("cross_family_scaling.tex", generate_cross_family_scaling),
        ("gpt2_scaling.tex", generate_gpt2_scaling),
        ("flagging_cross_scale.tex", generate_flagging_cross_scale),
    ]

    mismatches = 0
    for filename, gen_fn in generators:
        try:
            content = gen_fn()
            path = PAPER_TABLES / filename

            if args.check:
                if not path.exists():
                    print(f"  FAIL: {filename} does not exist")
                    mismatches += 1
                    continue
                existing = path.read_text()
                if existing == content:
                    print(f"  OK: {filename}")
                else:
                    diff = list(
                        difflib.unified_diff(
                            existing.splitlines(),
                            content.splitlines(),
                            fromfile=f"committed/{filename}",
                            tofile=f"generated/{filename}",
                            lineterm="",
                        )
                    )
                    for line in diff[:30]:
                        print(line)
                    if len(diff) > 30:
                        print(f"  ... ({len(diff) - 30} more lines)")
                    print(f"  FAIL: {filename} does not match generated output")
                    mismatches += 1
            else:
                path.write_text(content)
                print(f"  wrote {filename}")
        except Exception as e:
            print(f"  {filename} FAILED: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    if args.check:
        if mismatches:
            print(f"\nFAIL: {mismatches} table(s) do not match generated output")
            print("  Run 'just tables' to regenerate")
            sys.exit(1)
        else:
            print(f"\nOK: all {len(generators)} tables match generated output")
    else:
        print(f"Generated {len(generators)} tables -> {PAPER_TABLES}")


if __name__ == "__main__":
    main()
