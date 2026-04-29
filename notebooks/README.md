# Notebooks

_Updated 2026-04-28 for repo v3.4.0._

These notebooks ran the GPU experiments and wrote the committed JSONs in `results/`. To reproduce the results, read the JSONs directly; re-execution is not required. The committed JSONs are the source of truth.

| Notebook | Model | What it does |
|---|---|---|
| `walkthrough_analysis` | (CPU-only) | Analysis pipeline walkthrough: cross-family table, permutation test, partial correlation, Pythia collapse |

| `qwen05b_base_instruct_v3` | Qwen 0.5B base+instruct | v3 protocol at 600 ex/dim, full battery |
| `qwen3b_base_instruct_v3` | Qwen 3B base+instruct | v3 protocol, adds control sensitivity |
| `qwen7b_base_instruct_v2` | Qwen 7B base+instruct | Back-to-back run, two JSONs |
| `qwen7b_mechanistic` | Qwen 7B | Mean-ablation patching, C4 flagging, temperature scaling |
| `qwen7b_selective_prediction3` | Qwen 7B Instruct | TriviaQA selective prediction + instruct mechanistic |
| `qwen14b_comprehensive_v4` | Qwen 14B | Val-seed protocol, dense layer sweep L16-40 |
| `qwen14b_hardening` | Qwen 14B | 20-seed statistical hardening |
| `qwen14b_instruct` | Qwen 14B Instruct | RLHF invariance at 14B |
| `qwen05b_auxiliary_loss` | Qwen 0.5B | Auxiliary observability loss sweep |
| `llama3b_comprehensive_v2` | Llama 3.2 3B | Cross-family divergence validation, 7-seed |
| `cross_domain_qwen` | Qwen 1.5B | WikiText to C4/code transfer |
| `colab_nonlinear_probe_llama3b` | Llama 3.2 3B | Linear vs MLP probe comparison, held-out HP selection |
| `colab_nonlinear_probe_pythia_1_4b` | Pythia 1.4B | Linear vs MLP probe at the (24L, 16H) collapse configuration |

Models added after the notebook workflow (Llama 1B, Llama 8B, Llama 1B Instruct, Mistral 7B, Phi-3 Mini, Gemma 4B, and the Pythia suite) were run via `scripts/run_model.py`.

Notebooks are designed for Google Colab execution on A100/L4 GPUs. Cell outputs are not committed because they depend on the runtime environment; the committed JSONs in `results/` are the source of truth for all paper numbers. To view outputs, open a notebook in Colab and run it, or read the corresponding result JSON directly.
