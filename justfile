# Learned Observers Recover Decision-Quality Signal from Frozen Activations
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

# Run all experiments (alias for reproduce)
all device=default_device:
    just reproduce {{device}}

# Quick smoke test (1 seed, 5 epochs, verifies pipeline works)
smoke device=default_device:
    uv run src/train.py --dataset mnist --epochs 5 --seeds 1 --device {{device}}

# Run metric tests
test:
    uv run pytest tests/ -v

# Reproduce published results exactly
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

# Lint source files
lint:
    uv run ruff check src/

# Auto-format source files
fmt:
    uv run ruff format src/

# Run all checks (lint + format check)
check:
    uv run ruff check src/
    uv run ruff format --check src/

# Remove generated results and charts
clean:
    rm -f results/*.json assets/*.png
    rm -rf src/__pycache__
