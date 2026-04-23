#!/usr/bin/env bash
# Held-out nonlinear probe on all three collapsed configurations,
# 7 seeds, unified protocol. Designed for a single-GPU pod (H100 or A100).
#
# Models:
#   Pythia 410M  @ L2
#   Pythia 1.4B  @ L17
#   Llama 3.2 3B @ L0
#
# Common flags: --seeds 42..48, --batch-size 48, --attn-impl sdpa, --ex-dim 350.
#
# Expected wall time on a single H100 80GB: 45-55 minutes total.
#
# Usage on pod:
#   export HF_TOKEN=hf_xxx
#   bash runpod_all_nonlinear_7seed.sh

set -euo pipefail

ROOT="${ROOT:-/workspace}"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_XET_HIGH_PERFORMANCE=1

mkdir -p "$ROOT/scripts" "$ROOT/logs" "$HF_HOME"

if [ ! -f "$ROOT/scripts/nonlinear_probe.py" ] && [ -f "$ROOT/nonlinear_probe.py" ]; then
    cp "$ROOT/nonlinear_probe.py" "$ROOT/scripts/nonlinear_probe.py"
fi

# --- Pre-flight ---
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set (required for Llama 3.2 3B)" >&2
    exit 1
fi
if [ ! -f "$ROOT/scripts/nonlinear_probe.py" ]; then
    echo "ERROR: $ROOT/scripts/nonlinear_probe.py not found" >&2
    exit 1
fi
if ! grep -q "swept_hp_holdout" "$ROOT/scripts/nonlinear_probe.py"; then
    echo "ERROR: swept_hp_holdout protocol missing; upload updated version" >&2
    exit 1
fi
if ! grep -q "gpt_neox" "$ROOT/scripts/nonlinear_probe.py"; then
    echo "ERROR: gpt_neox branch missing; upload updated version" >&2
    exit 1
fi

nvidia-smi

# --- Install deps ---
pip install -q hf_xet transformers datasets scipy huggingface_hub

# --- HF login ---
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# --- Pre-download all three models ---
python - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc
for mid in [
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1.4b",
    "meta-llama/Llama-3.2-3B",
]:
    print(f"Downloading {mid}...")
    AutoTokenizer.from_pretrained(mid)
    AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16)
    gc.collect()
print("All three models cached.")
PY

cd "$ROOT"

COMMON_ARGS=(
    --seeds 42 43 44 45 46 47 48
    --batch-size 48
    --attn-impl sdpa
    --ex-dim 350
)

# --- Pythia 410M @ L2 ---
echo
echo "=============================="
echo "  Pythia 410M @ L2"
echo "=============================="
python -u scripts/nonlinear_probe.py \
    --model EleutherAI/pythia-410m \
    --peak-layer 2 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$ROOT/logs/pythia-410m.log"

# --- Pythia 1.4B @ L17 ---
echo
echo "=============================="
echo "  Pythia 1.4B @ L17"
echo "=============================="
python -u scripts/nonlinear_probe.py \
    --model EleutherAI/pythia-1.4b \
    --peak-layer 17 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$ROOT/logs/pythia-1.4b.log"

# --- Llama 3.2 3B @ L0 ---
echo
echo "=============================="
echo "  Llama 3.2 3B @ L0"
echo "=============================="
python -u scripts/nonlinear_probe.py \
    --model meta-llama/Llama-3.2-3B \
    --peak-layer 0 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$ROOT/logs/llama-3b.log"

# --- Summary ---
echo
echo "=============================="
echo "  Summary: held-out deltas"
echo "=============================="
python - <<'PY'
import json
from pathlib import Path
for stem, peak in [
    ("pythia-410m", 2),
    ("pythia-1.4b", 17),
    ("Llama-3.2-3B", 0),
]:
    p = Path(f"/workspace/nonlinear_probe_{stem}.json")
    if not p.exists():
        print(f"  {stem:20s}  MISSING")
        continue
    d = json.load(open(p))
    ho = d["swept_hp_holdout"]
    print(
        f"  {stem:20s}  L{peak:2d}  "
        f"linear {ho['linear_test_mean']:+.4f}  "
        f"mlp {ho['best_mlp_test_mean']:+.4f}  "
        f"delta {ho['delta_mean']:+.4f}"
    )
PY
