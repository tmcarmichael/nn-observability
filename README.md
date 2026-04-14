# Architecture Predicts Linear Readability of Decision Quality in Transformers

**DOI:** [10.5281/zenodo.19435674](https://doi.org/10.5281/zenodo.19435674) | **License:** [MIT](LICENSE) | **Python:** 3.12+

#### 12% of confident LLM errors are invisible to every output-based monitoring method. The model says it's right. Standard safety checks agree. Both are wrong.

But the hidden layers already know. One dot product per token catches these failures zero-shot on medical licensing questions and RAG hallucinations. No fine-tuning. No medical data.

Which architecture you deploy determines whether these errors are detectable. Same evaluation, same data, 3x gap across six model families. **The signals exist in production models today. Nobody is reading them.**

<p align="center">
<img src="assets/cross_family_scaling.png" width="85%" alt="Cross-family scaling: observability signal across six architecture families at matched scale. Qwen, GPT-2, and Mistral maintain stable signal. Llama drops sharply above 1B.">
</p>

_Decision quality signal independent of output confidence, measured across six architecture families. Same protocol, same data, same controls. Qwen, GPT-2, and Mistral hold a stable signal across scale. Llama drops sharply above 1B. The difference is architectural._

## Key findings

**The output layer hides what the model knows.** Raw probe-loss correlation is +0.55 on GPT-2 124M. After controlling for max softmax probability and activation norm: +0.28 survives. A nonlinear MLP confirms the residual is genuine (+0.289). Half to two-thirds of what standard probes measure is confidence in disguise (49-64% across nine of ten models). The information that remains is a signal the output distribution cannot express.

**One direction, not a subspace.** Twenty probe initializations on frozen GPT-2 124M converge to the same direction (+0.282 +/- 0.001, seed agreement +0.993). A nonlinear MLP does not exceed the linear probe on any of eight models tested. If a dot product finds it, the signal was already linearly encoded.

**Architecture determines readability; scale does not.** Mistral 7B: +0.313. Phi-3 Mini: +0.300. GPT-2 (124M-1.5B): +0.290. Qwen 2.5 (0.5B-14B): +0.25. Gemma 3 1B: +0.388 (high random baseline). Llama 3.2 3B: +0.091. Six families, 88% of variance between families, parameter count contributes nothing (p = 0.950). Within Llama, a 1B model produces +0.286 before the signal vanishes at 3B: the variable is architectural configuration, not family identity.

**Exclusive catch rate saturates.** At 20% flag rate, all six families converge to 12-15% exclusive catches. A 3.5x range in observability compresses to 1.2x in catch rate. The ceiling is set by error structure, not architecture.

**Zero-shot transfer.** WikiText-trained probes transfer to medical QA, retrieval-augmented generation, and factual QA. No domain-specific training. The signal lives in the representations regardless of domain.

## Cross-family results

| Model         | Family  | Params | rho_partial | +/- std | r_OC   | tau_seed |
| ------------- | ------- | ------ | ----------- | ------- | ------ | -------- |
| Qwen 2.5 0.5B | Qwen    | 0.5B   | +0.215      | 0.020   | +0.059 | +0.959   |
| Gemma 3 1B    | Gemma   | 1B     | +0.388      | 0.004   | +0.307 | +0.980   |
| Llama 3.2 1B  | Llama   | 1.2B   | +0.286      | 0.006   | +0.120 | +0.995   |
| Qwen 2.5 1.5B | Qwen    | 1.5B   | +0.275      | 0.032   | +0.127 | +0.953   |
| GPT-2 XL      | GPT-2   | 1.5B   | +0.290      | 0.004   | +0.174 | +0.952   |
| Qwen 2.5 3B   | Qwen    | 3B     | +0.263      | 0.021   | +0.144 | +0.925   |
| Llama 3.2 3B  | Llama   | 3B     | +0.091      | 0.006   | +0.031 | +0.998   |
| Qwen 2.5 7B   | Qwen    | 7B     | +0.255      | 0.019   | +0.137 | +0.964   |
| Mistral 7B    | Mistral | 7.2B   | +0.313      | 0.001   | +0.156 | +0.995   |
| Llama 3.1 8B  | Llama   | 8B     | +0.093      | 0.012   | -0.007 | +0.994   |
| Qwen 2.5 14B  | Qwen    | 14B    | +0.214      | 0.032   | +0.096 | +0.851   |

All models use the 7-seed protocol with 350 ex/dim. rho_partial is Spearman partial correlation after controlling for max softmax and activation norm. r_OC is the residual after controlling for a 64-unit output-layer predictor. tau_seed is mean pairwise Spearman across probe initializations.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Analysis runs on CPU; model evaluation requires a GPU with 16GB+ VRAM.

```bash
git clone https://github.com/tmcarmichael/nn-observability
cd nn-observability
uv sync
just install-hooks                 # ruff on commit, version check on push

just test                          # run tests (242 pass, CPU only)
just check                         # lint + format + version
just smoke-gpu                     # end-to-end smoke test (GPT-2 124M, CPU)
```

## Reproducing the analysis

The CPU analysis scripts run on committed result JSONs without any GPU:

```bash
uv run python analysis/run_all.py
```

This produces the permutation test (p = 0.006), mixed-effects model (88% family variance), ANCOVA, selectivity analysis, exclusive catch rates, and funnel plot. All scripts import from `load_results.py`, the single source of truth for which result files are in scope.

## Repository structure

```
src/                         Core library
  probe.py                   Shared probing functions (partial correlation,
                               activation collection, probe training)
  observe.py                 MLP observer (phases 1-3)
  transformer_observe.py     Transformer observer (phases 5-9)
  selective_prediction.py    TriviaQA selective prediction

scripts/                     GPU data collection
  run_model.py               Single entry point for any HuggingFace model
  roc_width_sweep.py         Output-controlled residual width sweep

analysis/                    Statistical analysis (all CPU, no GPU needed)
  load_results.py            Single source of truth for result loading
  run_all.py                 Run all analysis scripts
  permutation_test.py        Exact permutation test for family effect
  meta_regression.py         Mixed-effects model + variance decomposition
  exclusive_catch_rates.py   Multi-rate exclusive catch analysis
  ancova_family.py           ANCOVA supplement
  selectivity.py             Random head baselines + control gap
  funnel_plot.py             Publication bias diagnostic

figures/                     Paper figure generation
  generate_all.py            Regenerate all figures

results/                     All result JSONs (committed, reproducible)
tests/                       Pytest suite (242 tests)
```

## Citation

```bibtex
@article{carmichael2026architecture,
  title={Architecture Predicts Linear Readability of Decision Quality in Transformers},
  author={Carmichael, Thomas},
  year={2026},
  doi={10.5281/zenodo.19435674},
  url={https://github.com/tmcarmichael/nn-observability}
}
```

## License

[MIT License](LICENSE)
