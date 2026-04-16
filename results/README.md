# Results directory

Every JSON file here is a committed experimental result. The analysis scripts and paper figures read from these files via `analysis/load_results.py`.

## Primary results (v3 protocol, used in paper)

| File | Model | Protocol | Paper reference |
|---|---|---|---|
| `transformer_observe.json` | GPT-2 124M-1.5B | Phases 5-8, 3-20 seeds | `tab:gpt2_scaling`, `fig:waterfall`, `tab:control_sensitivity`, `tab:hand_designed_baselines`, `tab:sae_comparison` |
| `sae_compare.json` | GPT-2 124M | SAE vs raw probe, 3 seeds | `tab:sae_comparison` |
| `qwen05b_v3_results.json` | Qwen 2.5 0.5B | 7-seed, 600 ex/dim | `tab:cross_family_scaling` |
| `qwen05b_instruct_v3_results.json` | Qwen 2.5 0.5B Instruct | 7-seed, 600 ex/dim | `tab:instruct` |
| `qwen1_5b_v3_results.json` | Qwen 2.5 1.5B | 7-seed, 350 ex/dim | `tab:cross_family_scaling` |
| `qwen1_5b_instruct_v3_results.json` | Qwen 2.5 1.5B Instruct | 7-seed, 350 ex/dim | `tab:instruct` |
| `qwen3b_v3_results.json` | Qwen 2.5 3B | 7-seed, 350 ex/dim | `tab:cross_family_scaling`, `fig:layer_profiles` |
| `qwen3b_instruct_v3_results.json` | Qwen 2.5 3B Instruct | 7-seed, 350 ex/dim | `tab:instruct` |
| `qwen7b_v3_results.json` | Qwen 2.5 7B | 7-seed, 350 ex/dim | `tab:cross_family_scaling`, `tab:flagging_cross_scale` |
| `qwen7b_instruct_v3_results.json` | Qwen 2.5 7B Instruct | 7-seed, 350 ex/dim | `tab:instruct` |
| `qwen14b_v3_results.json` | Qwen 2.5 14B | 7-seed, 350 ex/dim | `tab:cross_family_scaling`, `tab:flagging_cross_scale` |
| `qwen14b_instruct_results.json` | Qwen 2.5 14B Instruct | 7-seed, 350 ex/dim | `tab:instruct` |
| `gemma3_1b_results.json` | Gemma 3 1B | 7-seed, 150 ex/dim | `tab:cross_family_scaling` |
| `gemma4b_results.json` | Gemma 3 4B | 7-seed, 350 ex/dim | `tab:cross_family_scaling` (flagging incomplete, rerun queued) |
| `llama1b_results.json` | Llama 3.2 1B | 7-seed, 350 ex/dim | `tab:cross_family_scaling` |
| `llama1b_instruct_results.json` | Llama 3.2 1B Instruct | 7-seed, 350 ex/dim | Architecture section (instruction tuning) |
| `llama3b_v3_results.json` | Llama 3.2 3B | 7-seed, 350 ex/dim | `tab:cross_family_scaling`, `fig:llama_cliff`, `tab:flagging_cross_scale` |
| `llama8b_results.json` | Llama 3.1 8B | 7-seed, 350 ex/dim | `tab:cross_family_scaling`, `fig:llama_cliff` (missing ctrl_sens/flagging/cross_domain, rerun in progress) |
| `mistral7b_results.json` | Mistral 7B v0.3 | 7-seed, 350 ex/dim | `tab:cross_family_scaling`, `fig:cross_family`, `tab:flagging_cross_scale` |
| `phi3_mini_results.json` | Phi-3 Mini 4K Instruct | 7-seed, 350 ex/dim | `tab:cross_family_scaling` |
| `qwen05b_exdim_sweep.json` | Qwen 2.5 0.5B | 7-seed, 150-1000 ex/dim | `fig:exdim` |
| `qwen7b_flagging_results.json` | Qwen 2.5 7B base+instruct | 3-seed, flagging only | `tab:flagging_cross_scale` (Qwen 7B base row) |
| `mechanistic_7b.json` | Qwen 2.5 7B base+instruct | Mean-ablation patching | Appendix mechanistic |
| `mechanistic_llama_comparison.json` | Llama 1B + 3B | Mean-ablation patching | Appendix mechanistic (Llama cliff localization) |
| `selective_prediction_v2_results.json` | Qwen 2.5 7B Instruct | TriviaQA, 3 seeds | Architecture section (AUACC) |
| `shuffle_test_gpt2.json` | GPT-2 124M | 10 permutations, shuffled labels | Method section (target validity) |
| `roc_width_sweep_results.json` | Qwen 2.5 7B | Output predictor 64-512 units | Method section (width sweep) |
| `model_revisions.json` | All models | HuggingFace commit hashes | Reproducibility section |

## Downstream task results

| File | Task | Model |
|---|---|---|
| `rag_hallucination_results.json` | SQuAD 2.0 RAG | Qwen 7B |
| `medqa_selective_results.json` | MedQA-USMLE | Qwen 7B |
| `truthfulqa_hallucination_results.json` | TruthfulQA | Qwen 7B |

## Supplementary results (used in analysis scripts or appendix)

| File | Model | Notes |
|---|---|---|
| `mnist.json` | MLP (MNIST) | Phase 1, BP vs FF comparison |
| `cifar10.json` | MLP (CIFAR-10) | Phase 1, BP vs FF comparison |
| `observe_mnist.json` | MLP observer (MNIST) | Phase 2, pure observer |
| `observe_mnist_auxiliary.json` | MLP observer (MNIST) | Phase 2, auxiliary loss |
| `observe_mnist_denoise.json` | MLP observer (MNIST) | Phase 2, denoising |
| `observe_mnist_observer_head.json` | MLP observer head (MNIST) | Phase 4, learned head variants |
| `scaling.json` | MLP scaling (5 sizes) | Phase 3, width scaling |
| `auxiliary_loss_results.json` | Qwen 0.5B | Auxiliary observability loss experiment |
| `bottleneck_scaling.json` | Multiple | Output predictor bottleneck sizes |
| `nonlinear_probe_gpt2.json` | GPT-2 124M | 3-seed, linear vs MLP comparison |
| `nonlinear_probe_Qwen2.5-0.5B.json` | Qwen 2.5 0.5B | 3-seed, linear vs MLP comparison |
| `nonlinear_probe_Qwen2.5-1.5B.json` | Qwen 2.5 1.5B | 3-seed, linear vs MLP comparison |
| `nonlinear_probe_Qwen2.5-3B.json` | Qwen 2.5 3B | 3-seed, linear vs MLP comparison |
| `nonlinear_probe_Qwen2.5-7B.json` | Qwen 2.5 7B | 3-seed, linear vs MLP comparison |
| `nonlinear_probe_Qwen2.5-14B.json` | Qwen 2.5 14B | 3-seed, linear vs MLP comparison |
| `nonlinear_probe_gemma-3-1b-pt.json` | Gemma 3 1B | 3-seed, linear vs MLP comparison |
| `nonlinear_probe_Llama-3.2-3B.json` | Llama 3.2 3B | 3-seed, linear vs MLP + HP sweep |
| `nonlinear_probe_Llama_Multi-3.2-3B.json` | Llama 3.2 3B | 5-layer sweep, linear + swept MLP |
| `smoke_fixture_gpt2.json` | GPT-2 124M | CI smoke test fixture |

## Preliminary and superseded results

| File | Model | Notes |
|---|---|---|
| `cross_family.json` | Qwen 0.5B/1.5B, Llama 1B | Phase 9 preliminary, 3-seed |
| `llama3b_v2_results.json` | Llama 3.2 3B | v2 protocol, superseded by `llama3b_v3_results.json` |
| `llama3b_diagnostic.json` | Llama 3.2 3B | Early diagnostic run |
| `llama8b_comprehensive.json` | Llama 3.1 8B | 3-seed preliminary, superseded by `llama8b_results.json` |
| `qwen7b_comprehensive.json` | Qwen 2.5 7B | Pre-v3, fallback for `qwen7b_v3_results.json` |
| `qwen7b_instruct_results.json` | Qwen 2.5 7B Instruct | Pre-v3 |
| `qwen05b_instruct_results.json` | Qwen 2.5 0.5B Instruct | Pre-v3 |
| `qwen1_5b_instruct_results.json` | Qwen 2.5 1.5B Instruct | Pre-v3 |
| `qwen05b_v2_results.json` | Qwen 2.5 0.5B | v2 protocol, fallback |
| `qwen1_5b_v2_results.json` | Qwen 2.5 1.5B | v2 protocol, fallback |
| `qwen3b_v2_results.json` | Qwen 2.5 3B | v2 protocol, fallback |
| `qwen3b_instruct_v2_results.json` | Qwen 2.5 3B Instruct | v2 protocol |
| `qwen14b_results.json` | Qwen 2.5 14B | v1 protocol (68 ex/dim) |
| `qwen14b_v2_results.json` | Qwen 2.5 14B | v2 protocol (250 ex/dim) |
| `qwen32b_results.json` | Qwen 2.5 32B | 7-seed, 350 ex/dim (disk quota failure, rerun queued) |
| `comprehensive_v3_results.json` | Multiple | Aggregated v3 snapshot |
| `selective_prediction_results.json` | Qwen 2.5 7B Instruct | v1 selective prediction, superseded by v2 |

## Versioning

Files named `*_v3_*` use the final protocol (matched ex/dim, 7-seed, full control battery). Files named `*_v2_*` are intermediate. The `v3` files are the paper's primary data source. Fallback logic is in `analysis/load_results.py`.

## JSON schema

Every full-protocol result contains: `model`, `n_layers`, `hidden_dim`, `protocol`, `peak_layer_final`, `layer_profile`, `partial_corr` (with `per_seed`), `test_split_comparison`, `seed_agreement`, `output_controlled`, `baselines`, `cross_domain`, `control_sensitivity`, `flagging_6a`. See `analysis/README.md` for the full schema specification.
