# Architecture Determines Observability of Transformers
# Run `just` to see all available recipes

set dotenv-load := false
set fallback

default_device := "auto"
default_seeds  := "3"
default_epochs := "50"

# List available recipes
default:
    @just --list

# Run MNIST comparison (3 seeds, 50 epochs)
train dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/train.py --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Run CIFAR-10 comparison
cifar10 seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/train.py --dataset cifar10 --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Run scaling study (5 model sizes)
scale dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/scale.py --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Run observer faithfulness test (pure observer)
observe dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/observe.py --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Run auxiliary loss observer test (auxiliary-overlay training)
observe-aux dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/observe.py --mode auxiliary --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Run denoising contrast observer test (same-domain co-training)
observe-denoise dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/observe.py --mode denoise --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Observer head variant sweep (linear vs MLP, regression vs binary)
observer-variants:
    uv run src/observer_variants.py

# Cross-seed ranking agreement test
seed-agreement:
    uv run src/seed_agreement.py

# Weight vector analysis for linear binary heads
inspect-weights:
    uv run src/inspect_weights.py

# GPT-2 124M observer heads (direct replication)
transformer seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --seeds {{seeds}} --device {{device}}

# ── Historical phase-taxonomy recipes (v1/v2 provenance, not part of v3+ scope) ──
# These run src/transformer_observe.py with phase-specific flags. They are the
# audit trail behind results/transformer_observe.json and the GPT-2 era of the
# project. Kept on purpose; do not delete as dead code.

# Layer sweep across all 12 GPT-2 layers
transformer-sweep seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --layer-sweep --seeds {{seeds}} --device {{device}}

# Hand-designed baselines on GPT-2
transformer-baselines seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --baselines --seeds {{seeds}} --device {{device}}

# Neuron-ablation intervention on GPT-2
transformer-intervention seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --intervention --seeds {{seeds}} --device {{device}}

# Full-output control (layer 8 vs layer 11 predictor)
transformer-output-control seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --output-control --seeds {{seeds}} --device {{device}}

# Directional ablation (residual stream projection)
transformer-directional seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --directional-ablation --seeds {{seeds}} --device {{device}}

# Early flagging (layer 8 observer vs output confidence)
transformer-flagging seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --flagging --seeds {{seeds}} --device {{device}}

# All transformer experiments on GPT-2 124M
transformer-all seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --all --seeds {{seeds}} --device {{device}}

# SAE comparison (SAE probe, rank overlap, flagging framework)
sae-compare seeds=default_seeds device=default_device:
    uv run --extra transformer src/sae_compare.py --all --seeds {{seeds}} --device {{device}}

# Three-channel causal decomposition for SAE
causal seeds=default_seeds device=default_device:
    uv run --extra transformer src/sae_compare.py --causal --seeds {{seeds}} --device {{device}}

# 20-seed statistical hardening with bootstrap CIs
hardening device=default_device:
    uv run --extra transformer src/transformer_observe.py --statistical-hardening --device {{device}}

# Control sensitivity analysis (6 control specifications)
control-sensitivity seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --control-sensitivity --seeds {{seeds}} --device {{device}}

# Cross-domain transfer (WikiText to OpenWebText, code)
cross-domain seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --cross-domain --seeds {{seeds}} --device {{device}}

# Scale characterization across GPT-2 family (124M to 1.5B)
gpt2-scale seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --scale --seeds {{seeds}} --device {{device}}

# Per-family rerun handles for cross-family-all (run individually after a partial failure)
# Cross-family replication: Llama 3.2 1B
cross-family-llama seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --cross-family-llama --seeds {{seeds}} --device {{device}}

# Cross-family replication: Qwen 2.5 0.5B + 1.5B
cross-family-qwen seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --cross-family-qwen --seeds {{seeds}} --device {{device}}

# Cross-family v1/v2 historical scope (Llama 1B + Qwen 0.5B + Qwen 1.5B). For v3+ 14-model scope, use individual cross-family-* recipes.
cross-family-all seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --cross-family-all --seeds {{seeds}} --device {{device}}

# Mechanistic analysis on Qwen 7B (mean-ablation patching at scale)
mechanistic-7b device=default_device:
    uv run --extra transformer src/transformer_observe.py --mechanistic-7b --device {{device}}

# Run all experiments (alias for reproduce)
all device=default_device:
    just reproduce {{device}}

# Quick smoke test (1 seed, 5 epochs, verifies pipeline works)
smoke device=default_device:
    uv run src/train.py --dataset mnist --epochs 5 --seeds 1 --device {{device}}

# Smoke test for run_model.py pipeline (GPT-2 124M, CPU)
smoke-pipeline:
    uv run pytest tests/test_smoke_run_model.py -v

# CUDA preflight gate: run the canonical protocol on the smallest paper-scope model
# end-to-end on CUDA before tag-cut. Catches CUDA-specific regressions that the
# CPU-only CI cannot. Not part of CI; run manually on the GPU host used for
# paper-quality runs (Colab, runpod, local CUDA box). Skips C4 cross-domain
# to save time. Output written to results/_preflight_temp.json and removed
# on completion. Expected runtime: a few minutes on a small GPU.

# Pre-tag CUDA gate: run canonical protocol on pythia-70m end-to-end (GPU host only)
preflight-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! uv run --extra transformer python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "FAIL: CUDA not available. Run on a CUDA-enabled host." >&2
        echo "preflight-cuda is intentionally not part of CI; run manually before tagging." >&2
        exit 1
    fi
    OUT="_preflight_temp.json"
    trap 'rm -f results/$OUT' EXIT
    echo "=== CUDA preflight: pythia-70m canonical protocol, no C4 ==="
    uv run --extra transformer python scripts/run_model.py \
        --model EleutherAI/pythia-70m \
        --output "$OUT" \
        --skip-c4
    echo ""
    echo "=== Validating output ==="
    uv run python -c "
    import json
    from pathlib import Path
    d = json.loads(Path('results/$OUT').read_text())
    assert d['provenance']['device'] == 'cuda', f\"device={d['provenance']['device']}\"
    assert 'partial_corr' in d, 'missing partial_corr'
    assert 'control_sensitivity' in d, 'missing control_sensitivity'
    print(f\"OK: device=cuda, peak_layer={d.get('peak_layer_final')}, pcorr={d['partial_corr']['mean']:.3f}\")
    "
    echo ""
    echo "=== CUDA preflight passed ==="

# Run metric tests
test:
    uv run pytest tests/ -v

# Validate configured results JSONs against schema
validate-results:
    uv run python analysis/load_results.py

# Validate configured results JSONs (strict: includes provenance)
validate-results-strict:
    uv run python analysis/load_results.py --strict

# Export named scopes from analysis/load_results.py to reports/scopes.json
export-scopes:
    uv run python scripts/export_scopes.py

# Check reports/scopes.json matches generated output (content diff)
check-scopes:
    uv run python scripts/export_scopes.py --check

# Validate every results/*_main.json and *_dynamics.json against schema/
validate-schemas:
    uv run python scripts/validate_schemas.py --strict

# Regenerate croissant.json from results/ and schema/
croissant:
    uv run python scripts/generate_croissant.py

# Verify croissant.json matches generator and validates against the Croissant 1.1 spec
check-croissant:
    uv run python scripts/generate_croissant.py --check
    uvx --quiet --from "mlcroissant==1.1.0" mlcroissant validate --jsonld croissant.json

# Export all release-time reports artifacts
export-reports: export-scopes

# Verify all reports/ artifacts match generated output (content diff, no writes)
check-reports: check-scopes

# Reproduce predecessor MLP work, GPT-2 scaling, and the cross-family sample
# (Llama 1B + Qwen 0.5B + Qwen 1.5B). Full paper scope runs via
# scripts/run_model.py; see `just pythia-suite` and `just downstream-all`.
# Committed results/*.json are the source of truth.

# Historical v1/v2 reproduction pipeline (MLP baselines + GPT-2 + 3-model cross-family)
reproduce device=default_device:
    just train mnist 3 50 {{device}}
    just cifar10 3 50 {{device}}
    just scale mnist 3 50 {{device}}
    just observe mnist 3 50 {{device}}
    just observe-aux mnist 3 50 {{device}}
    just observe-denoise mnist 3 50 {{device}}
    just observer-variants
    just seed-agreement
    just inspect-weights
    just transformer-all 3 {{device}}
    just sae-compare 3 {{device}}
    just gpt2-scale 3 {{device}}
    just cross-family-all 3 {{device}}

# Runs sequentially. On failure, rerun individual models with
# `just pythia EleutherAI/pythia-410m` (positional args, not key=value).

# Pythia controlled suite (9 configurations, 70M to 12B plus 1.4B-deduped)
pythia-suite:
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-70m --output pythia-70m_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-160m --output pythia-160m_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-410m --output pythia-410m_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-1b --output pythia-1b_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-1.4b --output pythia-1.4b_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-1.4b-deduped --output pythia-1.4b-deduped_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-2.8b --output pythia-2.8b_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-6.9b --output pythia-6.9b_main.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-12b --output pythia-12b_main.json

# Run a single Pythia model (for reruns after partial failure)
pythia model output="":
    #!/usr/bin/env bash
    out="{{output}}"
    if [ -z "$out" ]; then
        slug=$(echo "{{model}}" | sed 's|.*/||' | tr '[:upper:]' '[:lower:]')
        out="${slug}_main.json"
    fi
    uv run --extra transformer python scripts/run_model.py --model "{{model}}" --output "$out"

# Downstream QA tasks (3 tasks x 3 instruct models = 9 evaluations).
# Each trains a WikiText probe then evaluates on the downstream task.
# SQuAD 2.0: exact-match scoring on 1000 answerable questions (validation split).
# MedQA-USMLE: letter-match on 1000 4-option questions (test split).
# TruthfulQA: substring/overlap heuristic on ~817 questions (validation split).
# Exclusive catch = errors flagged by observer but not confidence, as % of total errors.

# Run all 3 downstream tasks (RAG + MedQA + TruthfulQA) for one instruct model
downstream model:
    uv run --extra transformer python scripts/rag_hallucination.py --model {{model}}
    uv run --extra transformer python scripts/medqa_selective.py --model {{model}}
    uv run --extra transformer python scripts/truthfulqa_hallucination.py --model {{model}}

downstream-all:
    just downstream Qwen/Qwen2.5-7B-Instruct
    just downstream mistralai/Mistral-7B-Instruct-v0.3
    just downstream microsoft/Phi-3-mini-4k-instruct

# Per-token dump for offline held-out fit-split analysis (CUDA required).
# Produces results/tokens/{slug}_tokens.npz with target, max_softmax, norm,
# and per-seed observer scores. Pair with `just held-out-fit-split` (CPU).
# peak_layer is optional; when omitted, dump_tokens.py reads peak_layer_final
# from results/<slug>_main.json (matching the producer auto-resolve pattern).

# Dump per-token observer/confidence/norm arrays (CUDA only)
dump-tokens model peak_layer="" ex_dim="350" seeds="43,44,45,46,47,48,49":
    #!/usr/bin/env bash
    set -euo pipefail
    extra=""
    if [ -n "{{peak_layer}}" ]; then
        extra="--peak-layer {{peak_layer}}"
    fi
    uv run --extra transformer python scripts/dump_tokens.py \
        --model {{model}} $extra \
        --ex-dim {{ex_dim}} --seeds {{seeds}}

# Representative subset spanning healthy and collapsed configurations under
# two recipe families plus the Pythia controlled suite. 8 models. Run on a
# single H100; expect ~1.5-2h end-to-end. Peak layers auto-resolve from each
# model's <slug>_main.json (no hardcoded layers per the v5.0.0 producer pattern).

# Dump per-token arrays for the 8-model representative suite (CUDA, single H100)
dump-tokens-suite:
    just dump-tokens openai-community/gpt2
    just dump-tokens meta-llama/Llama-3.2-1B
    just dump-tokens meta-llama/Llama-3.2-3B
    just dump-tokens Qwen/Qwen2.5-7B
    just dump-tokens mistralai/Mistral-7B-v0.3
    just dump-tokens microsoft/Phi-3-mini-4k-instruct
    just dump-tokens EleutherAI/pythia-1b
    just dump-tokens EleutherAI/pythia-1.4b

# Two-fold cross-validated held-out partial-Spearman analysis on dumped
# tokens (CPU only). Reports in-sample vs held-out delta per model;
# writes results/held_out_fit_split.json. Pair with `just dump-tokens-suite`.

# Run held-out partial-Spearman analysis on dumped tokens (CPU)
held-out-fit-split:
    uv run python analysis/held_out_split.py

# Install pre-commit hooks (ruff on commit, version check on push)
install-hooks:
    uv run pre-commit install
    uv run pre-commit install --hook-type pre-push

# Lint all Python (matches CI scope)
lint: lint-scripts
    uv run ruff check src/ scripts/ analysis/ tests/

# Auto-format all Python
fmt:
    uv run ruff format src/ scripts/ analysis/ tests/

# Reject f-string interpolation of /workspace/ in producer scripts.
# Output paths must use the canonical _resolve_out helper from
# scripts/run_model.py (Path("/workspace") when mounted, else repo's
# results/), so output location adapts to the runtime environment.

# Reject hardcoded /workspace/ paths in producer scripts
lint-scripts:
    #!/usr/bin/env bash
    set -euo pipefail
    matches=$(grep -n -e 'f"/workspace/' -e "f'/workspace/" scripts/*.py 2>/dev/null || true)
    if [ -n "$matches" ]; then
        echo "FAIL: f-string interpolation of /workspace/ detected. Use the canonical"
        echo "_resolve_out helper from scripts/run_model.py instead."
        echo ""
        echo "$matches"
        exit 1
    fi
    echo "OK: no path-bug interpolation patterns in scripts/"

# Type check (analysis API + core library + paper-scope GPU entry points)
typecheck:
    uv run mypy analysis/ src/observe.py src/probe.py src/utils.py scripts/run_model.py scripts/verify_manifest_revisions.py --ignore-missing-imports

# Dead code check
deadcode:
    uv run vulture src/ scripts/ analysis/ scripts/vulture_whitelist.py --min-confidence 90

# Run all checks (lint + format + types + dead code + version + schema + reports parity)
check:
    @just lint
    uv run ruff format --check src/ scripts/ analysis/
    @just typecheck
    @just deadcode
    @just test
    @just validate-results-strict
    @just validate-schemas
    @just check-croissant
    @just check-reports
    @just check-version

# Verify pyproject.toml version matches latest git tag
check-version:
    #!/usr/bin/env bash
    toml_version=$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
    latest_tag=$(git tag -l 'v*' --sort=-v:refname | head -1 | sed 's/^v//')
    if [ -z "$latest_tag" ]; then
        echo "  version: $toml_version (no tags yet)"
        exit 0
    fi
    if [ "$toml_version" != "$latest_tag" ]; then
        echo "  WARNING: pyproject.toml version ($toml_version) != latest tag (v$latest_tag)"
    else
        echo "  version: $toml_version (matches v$latest_tag)"
    fi

# CHANGELOG.md is the source of truth; release notes are derived.
# Pair with `gh release create vX.Y.Z --notes-file <(just release-notes X.Y.Z) ...`
# so the GitHub release body is byte-identical to the in-repo changelog entry.

# Extract a CHANGELOG.md version block as GitHub release notes body
release-notes version:
    #!/usr/bin/env bash
    set -euo pipefail
    out=$(awk '/^## v{{version}}[ (]/{flag=1; next} /^## v[0-9]/{flag=0} flag' CHANGELOG.md)
    if [ -z "$out" ]; then
        echo "ERROR: no entry for v{{version}} in CHANGELOG.md" >&2
        exit 1
    fi
    echo "$out"

# Remove build artifacts and caches
clean:
    rm -rf src/__pycache__ scripts/__pycache__ analysis/__pycache__ tests/__pycache__
    rm -rf .pytest_cache .ruff_cache .mypy_cache
    rm -f .coverage coverage.json
