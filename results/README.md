# Results directory

Every JSON file here is a committed experimental result. The analysis scripts and paper figures read from these files via `analysis/load_results.py`.

## File map

| File | Model | Protocol | Paper reference |
|---|---|---|---|
| `transformer_observe.json` | GPT-2 124M-1.5B | Phases 5-8, 3-20 seeds | Table 2, Figure 2 |
| `qwen05b_v3_results.json` | Qwen 2.5 0.5B | 7-seed, 600 ex/dim | Table 3 |
| `qwen05b_instruct_v3_results.json` | Qwen 2.5 0.5B Instruct | 7-seed, 600 ex/dim | Table 4 |
| `qwen1_5b_v3_results.json` | Qwen 2.5 1.5B | 7-seed, 350 ex/dim | Table 3 |
| `qwen1_5b_instruct_v3_results.json` | Qwen 2.5 1.5B Instruct | 7-seed, 350 ex/dim | Table 4 |
| `qwen3b_v3_results.json` | Qwen 2.5 3B | 7-seed, 350 ex/dim | Table 3, Figure 3 |
| `qwen3b_instruct_v3_results.json` | Qwen 2.5 3B Instruct | 7-seed, 350 ex/dim | Table 4 |
| `qwen7b_v3_results.json` | Qwen 2.5 7B | 7-seed, 350 ex/dim | Table 3, Table 5, Table 6 |
| `qwen7b_instruct_v3_results.json` | Qwen 2.5 7B Instruct | 7-seed, 350 ex/dim | Table 4, Table 5, Table 6 |
| `qwen14b_v3_results.json` | Qwen 2.5 14B | 7-seed, 350 ex/dim | Table 3 |
| `qwen14b_instruct_results.json` | Qwen 2.5 14B Instruct | 7-seed, 350 ex/dim | Table 4 |
| `gemma3_1b_results.json` | Gemma 3 1B | 7-seed, 150 ex/dim | Table 3 |
| `llama3b_v2_results.json` | Llama 3.2 3B | 7-seed, 200 ex/dim | Table 3, Figure 3 |
| `llama8b_comprehensive.json` | Llama 3.1 8B | 3-seed, preliminary | Figure 1 (open marker) |
| `mistral7b_results.json` | Mistral 7B v0.3 | 7-seed, 350 ex/dim | Figure 1 |
| `qwen05b_exdim_sweep.json` | Qwen 2.5 0.5B | 7-seed, 150-1000 ex/dim | Figure 4 |
| `cross_family.json` | Qwen 0.5B/1.5B, Llama 1B, Qwen 3B | Phase 9, 3-seed | Figure 1 (Llama 1B open marker) |
| `qwen32b_results.json` | Qwen 2.5 32B | 7-seed, 350 ex/dim | Excluded from v1 |

## Versioning

Files named `*_v3_*` use the final protocol (matched ex/dim, 7-seed, full control battery). Files named `*_v2_*` are intermediate. The `v3` files are the paper's primary data source.

## JSON schema

Every full-protocol result contains: `model`, `n_layers`, `hidden_dim`, `protocol`, `peak_layer_final`, `layer_profile`, `partial_corr` (with `per_seed`), `test_split_comparison`, `seed_agreement`, `output_controlled`, `baselines`, `cross_domain`, `control_sensitivity`, `flagging_6a`.
