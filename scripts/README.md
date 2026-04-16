# Scripts

GPU experiment scripts. Results write to `results/`.

## Primary

- `run_model.py` -- unified launcher for any HuggingFace model (full battery, provenance tracking)

## Probing and evaluation

- `nonlinear_probe.py` -- linear vs MLP probe comparison (per model)
- `split_bootstrap_gpu.py` -- document-level bootstrap (per model)
- `roc_width_sweep.py` -- output predictor bottleneck sweep (64-512 units)
- `shuffle_test.py` -- shuffled-label probe (target validity test)

## Downstream tasks

- `medqa_selective.py` -- MedQA-USMLE selective prediction
- `rag_hallucination.py` -- SQuAD 2.0 RAG error detection
- `truthfulqa_hallucination.py` -- TruthfulQA error detection

## Mechanistic analysis

- `mechanistic_llama.py` -- mean-ablation patching on Llama 1B vs 3B
- `mechanistic_mistral.py` -- mean-ablation patching on Mistral 7B

## Experimental (not in paper)

- `controlled_depth_width.py` -- controlled training: depth vs width at matched parameters (paper 2)
- `controlled_training.py` -- controlled training: MHA vs GQA (paper 2)

## Legacy

- `legacy/` -- per-model scripts from v1/v2 data collection (provenance, not for new models)
