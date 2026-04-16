# Architecture Predicts Linear Readability of Decision Quality in Transformers
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

# Run observer faithfulness test (Phase 2, pure observer)
observe dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/observe.py --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Run auxiliary loss observer test (Phase 2, overlay auxiliary)
observe-aux dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/observe.py --mode auxiliary --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Run denoising contrast observer test (Phase 2, same-domain co-training)
observe-denoise dataset="mnist" seeds=default_seeds epochs=default_epochs device=default_device:
    uv run src/observe.py --mode denoise --dataset {{dataset}} --epochs {{epochs}} --seeds {{seeds}} --device {{device}}

# Phase 4: observer head variant sweep (linear/MLP, regression/binary)
observer-variants device=default_device:
    uv run src/observer_variants.py

# Phase 4: cross-seed ranking agreement test
seed-agreement device=default_device:
    uv run src/seed_agreement.py

# Phase 4: weight vector analysis for linear binary heads
inspect-weights device=default_device:
    uv run src/inspect_weights.py

# Phase 5a: GPT-2 124M observer heads (direct replication)
transformer seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --seeds {{seeds}} --device {{device}}

# Phase 5b: layer sweep across all 12 GPT-2 layers
transformer-sweep seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --layer-sweep --seeds {{seeds}} --device {{device}}

# Phase 5c: hand-designed baselines on GPT-2
transformer-baselines seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --baselines --seeds {{seeds}} --device {{device}}

# Phase 5d: intervention (neuron ablation) on GPT-2
transformer-intervention seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --intervention --seeds {{seeds}} --device {{device}}

# Phase 5e: full-output control (layer 8 vs layer 11 predictor)
transformer-output-control seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --output-control --seeds {{seeds}} --device {{device}}

# Phase 5f: directional ablation (residual stream projection)
transformer-directional seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --directional-ablation --seeds {{seeds}} --device {{device}}

# Phase 6a: early flagging (layer 8 observer vs output confidence)
transformer-flagging seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --flagging --seeds {{seeds}} --device {{device}}

# All transformer experiments (5a-6a)
transformer-all seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --all --seeds {{seeds}} --device {{device}}

# Phase 7: SAE comparison (7a + 7c + 7d)
sae-compare seeds=default_seeds device=default_device:
    uv run --extra transformer src/sae_compare.py --all --seeds {{seeds}} --device {{device}}

# 7b: three-channel causal decomposition
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

# Phase 8: scale characterization across GPT-2 family (124M → 1.5B)
phase8 seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --scale --seeds {{seeds}} --device {{device}}

# Phase 9a: cross-family replication (Llama 3.2 1B)
phase9a seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --phase9a --seeds {{seeds}} --device {{device}}

# Phase 9b: second family replication (Qwen 2.5 0.5B + 1.5B)
phase9b seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --phase9b --seeds {{seeds}} --device {{device}}

# Phase 9: all cross-family experiments (9a + 9b)
phase9 seeds=default_seeds device=default_device:
    uv run --extra transformer src/transformer_observe.py --phase9 --seeds {{seeds}} --device {{device}}

# Mechanistic analysis on Qwen 7B (mean-ablation patching at scale)
mechanistic-7b device=default_device:
    uv run --extra transformer src/transformer_observe.py --mechanistic-7b --device {{device}}

# Phase 11: Selective prediction on TriviaQA (Qwen 7B Instruct)
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

# Validate paper-scope results JSONs against schema
validate-results:
    uv run python analysis/load_results.py

# Validate paper-scope results JSONs (strict: includes provenance)
validate-results-strict:
    uv run python analysis/load_results.py --strict

# Reproduce published results (Phases 1-9)
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

# Generate all paper figures (outputs to ../nn-observability-paper/figures/)
figures:
    uv run python figures/generate_all.py

# Generate paper tables (outputs to ../nn-observability-paper/tables/)
tables:
    uv run python analysis/generate_tables.py

# Generate data_macros.sty for the paper
data-macros:
    uv run python analysis/generate_data_macros.py

# Generate a single figure (e.g., just figure fig_cross_family)
figure name:
    uv run python figures/{{name}}.py

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

# Run all checks (lint + format check + version consistency)
check:
    uv run ruff check src/ scripts/ figures/ analysis/
    uv run ruff format --check src/ scripts/ figures/ analysis/
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

# Remove generated results and charts
clean:
    rm -f results/*.json assets/*.png
    rm -rf src/__pycache__
