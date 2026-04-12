#!/bin/bash
# Run nonlinear probe on all v1 paper models + Llama multi-layer sweep
# Upload to /workspace/ on RunPod and run: bash /workspace/run_all_nonlinear.sh

set -e
export HF_HOME=/workspace/hf_cache

echo "=== Nonlinear probe: 7 models ==="

python /workspace/nonlinear_probe.py --model Qwen/Qwen2.5-0.5B --peak-layer 19 --ex-dim 600 --max-docs 6000 2>&1 | tee /workspace/nl_qwen05b.log
python /workspace/nonlinear_probe.py --model Qwen/Qwen2.5-1.5B --peak-layer 18 --ex-dim 350 --max-docs 8000 2>&1 | tee /workspace/nl_qwen1_5b.log
python /workspace/nonlinear_probe.py --model Qwen/Qwen2.5-3B --peak-layer 25 --ex-dim 350 --max-docs 8000 2>&1 | tee /workspace/nl_qwen3b.log
python /workspace/nonlinear_probe.py --model Qwen/Qwen2.5-7B --peak-layer 17 --ex-dim 350 --max-docs 8000 2>&1 | tee /workspace/nl_qwen7b.log
python /workspace/nonlinear_probe.py --model Qwen/Qwen2.5-14B --peak-layer 30 --ex-dim 350 --max-docs 8000 2>&1 | tee /workspace/nl_qwen14b.log
python /workspace/nonlinear_probe.py --model meta-llama/Llama-3.2-3B --peak-layer 0 --ex-dim 200 --max-docs 8000 2>&1 | tee /workspace/nl_llama3b.log
python /workspace/nonlinear_probe.py --model google/gemma-3-1b-pt --peak-layer 1 --ex-dim 150 --max-docs 6000 2>&1 | tee /workspace/nl_gemma1b.log

echo "=== Llama multi-layer sweep ==="

for layer in 0 7 14 21 27; do
    python /workspace/nonlinear_probe.py --model meta-llama/Llama-3.2-3B --peak-layer $layer --ex-dim 200 --max-docs 8000 2>&1 | tee /workspace/nl_llama3b_L${layer}.log
done

echo "=== All done ==="
