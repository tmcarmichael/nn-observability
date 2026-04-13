#!/bin/bash
# Wait for Mistral to finish, then run Phi-3 Mini.
# Upload to /workspace/ and run: bash /workspace/wait_and_phi3.sh
# Requires HF_HOME and HF_TOKEN set in the terminal before running.

set -e

pip install transformers datasets scipy scikit-learn accelerate 2>/dev/null

echo "$(date): waiting for Mistral to finish..."

while pgrep -f mistral7b_comprehensive > /dev/null; do
  echo "$(date): Mistral still running"
  ls -lt /workspace/mistral7b_results.json 2>/dev/null | head -1
  sleep 300
done

echo "$(date): Mistral done, starting Phi-3 Mini"
python /workspace/phi3_mini_comprehensive.py 2>&1 | tee /workspace/phi3_mini.log
echo "$(date): Phi-3 Mini complete"
