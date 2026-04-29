# Results directory

_Updated 2026-04-28 for repo v3.4.0._

Every JSON file here is a committed experimental result. Analysis scripts read from these files via `analysis/load_results.py`.

## Cross-family scaling (v3 protocol)

| File | Model | Protocol |
|---|---|---|
| `transformer_observe.json` | GPT-2 124M-1.5B | 3-20 seeds, full battery |
| `qwen05b_v3_results.json` | Qwen 2.5 0.5B | 7-seed, 600 ex/dim |
| `qwen05b_instruct_v3_results.json` | Qwen 2.5 0.5B Instruct | 7-seed, 600 ex/dim |
| `qwen1_5b_v3_results.json` | Qwen 2.5 1.5B | 7-seed, 350 ex/dim |
| `qwen1_5b_instruct_v3_results.json` | Qwen 2.5 1.5B Instruct | 7-seed, 350 ex/dim |
| `qwen3b_v3_results.json` | Qwen 2.5 3B | 7-seed, 350 ex/dim |
| `qwen3b_instruct_v3_results.json` | Qwen 2.5 3B Instruct | 7-seed, 350 ex/dim |
| `qwen7b_v3_results.json` | Qwen 2.5 7B | 7-seed, 350 ex/dim |
| `qwen7b_instruct_v3_results.json` | Qwen 2.5 7B Instruct | 7-seed, 350 ex/dim |
| `qwen14b_v3_results.json` | Qwen 2.5 14B | 7-seed, 350 ex/dim |
| `qwen14b_instruct_results.json` | Qwen 2.5 14B Instruct | 7-seed, 350 ex/dim |
| `gemma3_1b_results.json` | Gemma 3 1B | 7-seed, 150 ex/dim |
| `gemma4b_v3_results.json` | Gemma 3 4B | 7-seed, 350 ex/dim |
| `llama1b_results.json` | Llama 3.2 1B | 7-seed, 350 ex/dim |
| `llama1b_instruct_results.json` | Llama 3.2 1B Instruct | 7-seed, 350 ex/dim |
| `llama3b_v3_results.json` | Llama 3.2 3B | 7-seed, 350 ex/dim |
| `llama8b_v3_results.json` | Llama 3.1 8B | 7-seed, 350 ex/dim |
| `mistral7b_results.json` | Mistral 7B v0.3 | 7-seed, 350 ex/dim |
| `mistral7b_instruct_v3_results.json` | Mistral 7B Instruct v0.3 | 7-seed, 350 ex/dim |
| `phi3_mini_results.json` | Phi-3 Mini 4K Instruct | 7-seed, 350 ex/dim |

## Pythia suite (within-recipe controlled)

| File | Model | Notes |
|---|---|---|
| `pythia_70m_results.json` | Pythia 70M | Includes random-probe baseline |
| `pythia_160m_results.json` | Pythia 160M | 7-seed |
| `pythia_410m_results.json` | Pythia 410M | Collapse configuration (24L, 16H) |
| `pythia1b_results.json` | Pythia 1B | 7-seed |
| `pythia1_4b_results.json` | Pythia 1.4B | Collapse configuration (24L, 16H) |
| `pythia_1.4b_deduped_results.json` | Pythia 1.4B deduped Pile | Collapse replication across corpora |
| `pythia_1.4b_shuffle_results.json` | Pythia 1.4B | Shuffled-label null distribution |
| `pythia_2.8b_results.json` | Pythia 2.8B | 7-seed |
| `pythia_6.9b_results.json` | Pythia 6.9B | 7-seed |
| `pythia_12b_results.json` | Pythia 12B | 7-seed |

## Checkpoint dynamics (within-recipe controlled)

| File | Model | Notes |
|---|---|---|
| `pythia_1b_dynamics_results.json` | Pythia 1B (16L/8H, d=2048) | 10 checkpoints, step 256 to 143000; healthy trajectory |
| `pythia_14b_dynamics_results.json` | Pythia 1.4B (24L/16H, d=2048) | 10 checkpoints, step 256 to 143000; collapse trajectory |

Matched hidden dimension ($d = 2048$). Both start healthy at step 256; the 1B recovers after a mid-training dip, the 1.4B converges collapsed. Per-checkpoint fields: partial_corr (7-seed), output_controlled (3-seed), perplexity, peak layer, and HuggingFace revision hash. These files use a different schema from the single-model results and are validated separately by `validate_dynamics_json` in `analysis/load_results.py`.

## Downstream tasks

| File | Task | Model |
|---|---|---|
| `rag_hallucination_qwen7b_instruct_L17_results.json` | SQuAD 2.0 RAG | Qwen 7B Instruct |
| `rag_hallucination_phi3_mini_instruct_results.json` | SQuAD 2.0 RAG | Phi-3 Mini Instruct |
| `rag_hallucination_mistral7b_instruct_results.json` | SQuAD 2.0 RAG | Mistral 7B Instruct |
| `medqa_selective_qwen7b_instruct_L17_results.json` | MedQA-USMLE | Qwen 7B Instruct (L17) |
| `medqa_selective_qwen7b_instruct_L18_results.json` | MedQA-USMLE | Qwen 7B Instruct (L18 sensitivity) |
| `medqa_selective_qwen7b_instruct_L19_results.json` | MedQA-USMLE | Qwen 7B Instruct (L19 sensitivity) |
| `medqa_selective_phi3_mini_instruct_results.json` | MedQA-USMLE | Phi-3 Mini Instruct |
| `medqa_selective_mistral7b_instruct_results.json` | MedQA-USMLE | Mistral 7B Instruct |
| `truthfulqa_hallucination_qwen7b_instruct_L17_results.json` | TruthfulQA | Qwen 7B Instruct |
| `truthfulqa_hallucination_phi3_mini_instruct_results.json` | TruthfulQA | Phi-3 Mini Instruct |
| `truthfulqa_hallucination_mistral7b_instruct_results.json` | TruthfulQA | Mistral 7B Instruct |

## Probe-validity controls

| File | Model | Test |
|---|---|---|
| `shuffle_test_gpt2.json` | GPT-2 124M | 10 permutations, shuffled labels |
| `roc_width_sweep_results.json` | Qwen 2.5 7B | Output predictor 64-512 units |
| `qwen05b_exdim_sweep.json` | Qwen 2.5 0.5B | Token budget sensitivity 150-1000 ex/dim |
| `split_bootstrap_Qwen2.5-7B.json` | Qwen 2.5 7B | 30-resample document-level bootstrap |
| `nonlinear_probe_gpt2.json` | GPT-2 124M | Linear vs MLP |
| `nonlinear_probe_Qwen2.5-0.5B.json` | Qwen 2.5 0.5B | Linear vs MLP |
| `nonlinear_probe_Qwen2.5-1.5B.json` | Qwen 2.5 1.5B | Linear vs MLP |
| `nonlinear_probe_Qwen2.5-3B.json` | Qwen 2.5 3B | Linear vs MLP |
| `nonlinear_probe_Qwen2.5-7B.json` | Qwen 2.5 7B | Linear vs MLP |
| `nonlinear_probe_Qwen2.5-14B.json` | Qwen 2.5 14B | Linear vs MLP |
| `nonlinear_probe_gemma-3-1b-pt.json` | Gemma 3 1B | Linear vs MLP |
| `nonlinear_probe_Llama-3.2-3B.json` | Llama 3.2 3B | Linear vs MLP, held-out HP selection |
| `nonlinear_probe_Llama_Multi-3.2-3B.json` | Llama 3.2 3B | 5-layer sweep |
| `nonlinear_probe_pythia-410m.json` | Pythia 410M | Collapse-point MLP comparison |
| `nonlinear_probe_pythia-1.4b.json` | Pythia 1.4B | Collapse-point MLP comparison |

## Mechanistic analysis

| File | Model | Analysis |
|---|---|---|
| `mechanistic_7b.json` | Qwen 2.5 7B base+instruct | Mean-ablation patching |
| `mechanistic_llama_comparison.json` | Llama 3.2 1B + 3B | Mean-ablation patching |
| `mechanistic_mistral.json` | Mistral 7B | Mean-ablation patching |

## Held for future work

| File | Model | Notes |
|---|---|---|
| `qwen32b_results.json` | Qwen 2.5 32B | Partial save: pcorr mean 0.213 and std 0.0067, peak L44, no per-seed array, no baselines, no cross-domain; clean rerun queued for v3.1.0 |

## Preliminary and superseded

| File | Model | Notes |
|---|---|---|
| `cross_family.json` | Qwen 0.5B/1.5B, Llama 1B | Preliminary, 3-seed |
| `llama8b_results.json` | Llama 3.1 8B | Intermediate, superseded by v3 |
| `gemma4b_results.json` | Gemma 3 4B | Intermediate, superseded by v3 |
| `qwen7b_comprehensive.json` | Qwen 2.5 7B | Pre-v3, fallback |
| `qwen7b_instruct_results.json` | Qwen 2.5 7B Instruct | Pre-v3 |
| `qwen05b_v2_results.json` | Qwen 2.5 0.5B | v2 protocol, fallback |
| `qwen1_5b_v2_results.json` | Qwen 2.5 1.5B | v2 protocol, fallback |
| `qwen3b_v2_results.json` | Qwen 2.5 3B | v2 protocol, fallback |
| `qwen7b_flagging_results.json` | Qwen 2.5 7B | Legacy flagging variant |

## Predecessor MLP observability work

| File | Task | Notes |
|---|---|---|
| `scaling.json` | MLP scaling (5 sizes) | Width scaling |
| `bottleneck_scaling.json` | Multiple | Output predictor bottleneck sizes |
| `sae_compare.json` | GPT-2 124M | SAE vs raw probe, 3 seeds |

## Infrastructure

| File | Purpose |
|---|---|
| `model_revisions.json` | HuggingFace commit hashes for all models |
| `smoke_fixture_gpt2.json` | CI smoke test fixture |

## Versioning

Files named `*_v3_*` use the final protocol (matched ex/dim, 7-seed, full control battery). Files named `*_v2_*` are intermediate. Fallback logic is in `analysis/load_results.py`.

## JSON schema

Every full-protocol result contains: `model`, `n_layers`, `hidden_dim`, `protocol`, `peak_layer_final`, `layer_profile`, `partial_corr` (with `per_seed`), `test_split_comparison`, `seed_agreement`, `output_controlled`, `baselines`, `cross_domain`, `control_sensitivity`, `flagging_6a`. See `analysis/load_results.py` for the schema validator.
