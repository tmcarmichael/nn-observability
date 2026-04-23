# Architecture Determines Observability in Transformers
# Run `just` to see all available recipes

set dotenv-load := false

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
observer-variants device=default_device:
    uv run src/observer_variants.py

# Cross-seed ranking agreement test
seed-agreement device=default_device:
    uv run src/seed_agreement.py

# Weight vector analysis for linear binary heads
inspect-weights device=default_device:
    uv run src/inspect_weights.py

# GPT-2 124M observer heads (direct replication)
transformer seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --seeds {{seeds}} --device {{device}}

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

# Cross-domain transfer (WikiText → OpenWebText, code)
cross-domain seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --cross-domain --seeds {{seeds}} --device {{device}}

# Scale characterization across GPT-2 family (124M → 1.5B)
phase8 seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --scale --seeds {{seeds}} --device {{device}}

# Cross-family replication: Llama 3.2 1B
phase9a seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --phase9a --seeds {{seeds}} --device {{device}}

# Cross-family replication: Qwen 2.5 0.5B + 1.5B
phase9b seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --phase9b --seeds {{seeds}} --device {{device}}

# All cross-family experiments (Llama 1B + Qwen 0.5B + Qwen 1.5B)
phase9 seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --phase9 --seeds {{seeds}} --device {{device}}

# Mechanistic analysis on Qwen 7B (mean-ablation patching at scale)
mechanistic-7b device=default_device:
    uv run --extra transformer src/transformer_observe.py --mechanistic-7b --device {{device}}

# Selective prediction on TriviaQA (Qwen 7B Instruct)
selective-prediction seeds=default_seeds device=default_device:
    uv run --extra transformer src/selective_prediction.py --seeds {{seeds}} --device {{device}}

# Run all experiments (alias for reproduce)
all device=default_device:
    just reproduce {{device}}

# Quick smoke test (1 seed, 5 epochs, verifies pipeline works)
smoke device=default_device:
    uv run src/train.py --dataset mnist --epochs 5 --seeds 1 --device {{device}}

# Smoke test for run_model.py (GPT-2 124M, CPU)
smoke-gpu:
    uv run pytest tests/test_smoke_run_model.py -v

# Run metric tests
test:
    uv run pytest tests/ -v

# Validate configured results JSONs against schema
validate-results:
    uv run python analysis/load_results.py

# Validate configured results JSONs (strict: includes provenance)
validate-results-strict:
    uv run python analysis/load_results.py --strict

# Reproduce predecessor MLP work, GPT-2 scaling, and the cross-family sample
# (Llama 1B + Qwen 0.5B + Qwen 1.5B). Full v3 paper scope (13 cross-family models,
# 9 Pythia configurations, 3 downstream tasks) runs via scripts/run_model.py; see
# `just pythia-suite` below and the per-model commands in notebooks/README.md.
# Committed results/*.json are the source of truth.
reproduce device=default_device:
    just train mnist 3 50 {{device}}
    just cifar10 3 50 {{device}}
    just scale mnist 3 50 {{device}}
    just observe mnist 3 50 {{device}}
    just observe-aux mnist 3 50 {{device}}
    just observe-denoise mnist 3 50 {{device}}
    just observer-variants {{device}}
    just seed-agreement {{device}}
    just inspect-weights {{device}}
    just transformer-all 3 {{device}}
    just sae-compare 3 {{device}}
    just phase8 3 {{device}}
    just phase9 3 {{device}}

# Pythia controlled suite (9 configurations, 70M to 12B plus 1.4B-deduped)
pythia-suite:
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-70m --output pythia_70m_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-160m --output pythia_160m_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-410m --output pythia_410m_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-1b --output pythia1b_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-1.4b --output pythia1_4b_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-1.4b-deduped --output pythia_1.4b_deduped_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-2.8b --output pythia_2.8b_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-6.9b --output pythia_6.9b_results.json
    uv run --extra transformer python scripts/run_model.py --model EleutherAI/pythia-12b --output pythia_12b_results.json

# Install pre-commit hooks (ruff on commit, version check on push)
install-hooks:
    uv run pre-commit install
    uv run pre-commit install --hook-type pre-push

# Lint all Python (matches CI scope)
lint:
    uv run ruff check src/ scripts/ figures/ analysis/

# Auto-format all Python
fmt:
    uv run ruff format src/ scripts/ figures/ analysis/

# Run all checks (lint + format check + version consistency + README freshness)
check:
    uv run ruff check src/ scripts/ figures/ analysis/
    uv run ruff format --check src/ scripts/ figures/ analysis/
    @just check-version
    @just check-readme-freshness

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

# Verify non-root READMEs reference the current pyproject.toml version
check-readme-freshness:
    #!/usr/bin/env bash
    toml_version=$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
    failed=0
    for readme in analysis/README.md scripts/README.md results/README.md \
                  notebooks/README.md assets/README.md assets/share/README.md; do
        if [ ! -f "$readme" ]; then
            echo "  MISSING: $readme"
            failed=1
            continue
        fi
        if ! grep -q "for repo v${toml_version}" "$readme"; then
            current=$(grep -oE "for repo v[0-9]+\.[0-9]+\.[0-9]+" "$readme" | head -1 || echo "(no version line)")
            echo "  FAIL: $readme says '$current', pyproject.toml is v${toml_version}"
            failed=1
        fi
    done
    if [ $failed -eq 0 ]; then
        echo "  READMEs: all pinned to v${toml_version}"
    else
        exit 1
    fi

# Bump version and date in every non-root README (run after reviewing each)
bump-readme-versions:
    #!/usr/bin/env bash
    toml_version=$(grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')
    today=$(date +%Y-%m-%d)
    for readme in analysis/README.md scripts/README.md results/README.md \
                  notebooks/README.md assets/README.md assets/share/README.md; do
        [ -f "$readme" ] || continue
        sed -i.bak -E "s/^_Updated [0-9]{4}-[0-9]{2}-[0-9]{2} for repo v[0-9]+\.[0-9]+\.[0-9]+\._$/_Updated ${today} for repo v${toml_version}._/" "$readme"
        rm -f "$readme.bak"
    done
    echo "  READMEs: bumped to ${today} / v${toml_version}"

# Remove generated results and charts
clean:
    rm -f results/*.json assets/*.png
    rm -rf src/__pycache__
