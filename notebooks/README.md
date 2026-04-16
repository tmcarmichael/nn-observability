# Notebooks

Read-only consumers of `results/*.json`. These ran the GPU experiments and saved results; they do not need to be re-executed. The committed JSONs are the source of truth.

| Notebook | Model | What it does |
|---|---|---|
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

Models added after the notebook workflow (Llama 1B, Llama 8B, Llama 1B Instruct, Mistral 7B, Phi-3 Mini, Gemma 4B) were run via `scripts/run_model.py` or archived per-model scripts in `docs/internal/notes/archive/scripts/`.
