# Scripts

_Updated 2026-04-28 for repo v3.4.0._

GPU experiment scripts. Results write to `results/`.

## Primary

- `run_model.py` -- unified launcher for any HuggingFace model (full battery, provenance tracking)

## Probing and evaluation

- `nonlinear_probe.py` -- linear vs MLP probe comparison (per model)
- `split_bootstrap_gpu.py` -- document-level bootstrap (per model)
- `roc_width_sweep.py` -- output predictor bottleneck sweep (64-512 units)
- `gpt2_shuffle_test.py` -- shuffled-label probe on GPT-2 124M (target validity test)
- `pythia_1.4b_shuffle.py` -- shuffled-label probe on Pythia 1.4B (collapse validity test)
- `phi3_layer_sweep_mps.py` -- Phi-3 Mini layer-sweep diagnostic on local MPS

## Downstream tasks

- `medqa_selective.py` -- MedQA-USMLE selective prediction
- `rag_hallucination.py` -- SQuAD 2.0 RAG error detection
- `truthfulqa_hallucination.py` -- TruthfulQA error detection

## Per-device replications

MPS-compatible scripts that run the full battery on Apple Silicon without pod infrastructure.

- `mistral7b_instruct_full_mps.py` -- Mistral 7B Instruct: layer sweep + seven-seed anchor + three downstream evaluations
- `phi3_downstream_mps.py` -- Phi-3 Mini downstream: MedQA, RAG, TruthfulQA with the same schema as the Qwen 7B Instruct battery

## Mechanistic analysis

- `mechanistic_llama.py` -- mean-ablation patching on Llama 1B vs 3B
- `mechanistic_mistral.py` -- mean-ablation patching on Mistral 7B

## Controlled training

- `controlled_depth_width.py` -- controlled training: depth vs width at matched parameters
- `controlled_training.py` -- controlled training: MHA vs GQA

## Infrastructure

- `dump_tokens.py` -- per-token observer/confidence/norm/target dump for held-out fit-split analysis
- `pythia_12b_backfill.py` -- resumable partial-result backfill for large runs
- `run_stream_model.py` -- streaming variant of run_model.py for memory-constrained settings
- `runpod_all_nonlinear_7seed.sh` -- pod-side shell driver: caches the three collapsed-config models, runs `nonlinear_probe.py` on each, prints a held-out delta summary
