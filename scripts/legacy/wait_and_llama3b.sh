#!/bin/bash
# Usage: nohup bash /workspace/wait_and_llama3b.sh > /workspace/wait_and_llama3b.log 2>&1 &

set -e

# Environment (nohup spawns a clean shell)
export HF_HOME=/workspace/hf_cache

# Verify setup before waiting
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before running this script."
    exit 1
fi
if ! python -c "import transformers" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install transformers datasets scipy scikit-learn accelerate
fi
if [ ! -f /workspace/run_model.py ]; then
    echo "ERROR: /workspace/run_model.py not found"
    exit 1
fi

echo "$(date): environment OK, waiting for current job to finish..."

while pgrep -f "run_model.py" > /dev/null 2>&1; do
    echo "$(date): run_model.py still running"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || true
    sleep 300
done

echo "$(date): GPU free, pausing 30s for cleanup..."
sleep 30

echo "$(date): starting Llama 3.2 3B rerun"
python /workspace/run_model.py --model meta-llama/Llama-3.2-3B --output llama3b_v3_results.json --trust-remote-code
echo "$(date): Llama 3.2 3B complete"
echo "$(date): results at /workspace/llama3b_v3_results.json"
