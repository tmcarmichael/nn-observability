# Architecture Predicts Linear Readability of Decision Quality in Transformers

[DOI: 10.5281/zenodo.19435674](https://doi.org/10.5281/zenodo.19435674)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Half the signal in standard activation probes is output confidence in disguise. After controlling for it, a stable linear signal remains, and its strength depends on architecture family, not model scale.

## Key findings

**Confidence controls eliminate most of what probes measure.**
Raw Spearman correlation between a linear probe and per-token loss is +0.55 on GPT-2 124M. After controlling for max softmax probability and activation norm, only +0.28 survives. Four hand-designed activation statistics that show strong raw correlation all collapse to near zero under the same controls. Published probing results without confidence controls are difficult to interpret.

**The surviving signal is stable, linear, and reproducible.**
Twenty independent probe initializations on frozen GPT-2 124M converge to the same direction (rho_partial = +0.282 +/- 0.001, seed agreement +0.993). A nonlinear MLP does not exceed the linear probe at matched hyperparameters on any of eight models tested. The signal peaks at two-thirds depth across GPT-2 (124M--1.5B) and Qwen 2.5 (0.5B--14B).

**Architecture family predicts observability; scale does not.**
Under identical evaluation, Qwen 2.5 maintains rho_partial ~ +0.25 from 0.5B to 14B (28x range). GPT-2 is stable at +0.28 from 124M to 1.5B. Llama 3.2 3B produces +0.089, a 2.8x gap with Qwen at matched scale (permutation test p = 0.014, 88% of variance between families). Instruction tuning preserves the signal at every Qwen scale tested.

**9--10% of model errors are invisible to confidence.**
At 10% flag rate, the observer catches tokens where the model is confident but wrong. This fraction is stable from GPT-2 124M through Qwen 2.5 14B, across base and instruct variants, and across WikiText-to-C4 domain transfer. Confidence precision is 1.000 on all Qwen variants: no confidence threshold would flag these tokens.

![Cross-family scaling](assets/cross_family_scaling.png)

## Cross-family results

| Model | Family | Params | Peak layer | rho_partial | r_OC | tau_seed |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen 2.5 0.5B | Qwen | 0.5B | L19 (79%) | +0.215 | +0.059 | +0.959 |
| Gemma 3 1B | Gemma | 1B | L1 (4%) | +0.388 | +0.307 | +0.980 |
| Llama 3.2 1B\* | Llama | 1.2B | L6 (38%) | +0.250 | +0.126 | +0.999 |
| Qwen 2.5 1.5B | Qwen | 1.5B | L18 (64%) | +0.275 | +0.127 | +0.953 |
| GPT-2 XL | GPT-2 | 1.5B | L34 (71%) | +0.290 | +0.174 | +0.952 |
| Qwen 2.5 3B | Qwen | 3B | L25 (69%) | +0.263 | +0.144 | +0.925 |
| Llama 3.2 3B | Llama | 3B | L0 (0%) | +0.089 | +0.033 | +0.999 |
| Qwen 2.5 7B | Qwen | 7B | L17 (61%) | +0.255 | +0.137 | +0.964 |
| Llama 3.1 8B\* | Llama | 8B | L0 (0%) | +0.088 | +0.054 | --- |
| Qwen 2.5 14B | Qwen | 14B | L30 (62%) | +0.214 | +0.096 | +0.851 |

\*Llama 1B and 8B are preliminary (3 seeds, lower ex/dim). All other models use the full 7-seed protocol with matched token budgets per hidden dimension. Gemma's random untrained probe baseline is +0.213, indicating representational geometry rather than learned signal accounts for part of its high value.

## Quick start

```bash
git clone https://github.com/tmcarmichael/nn-observability
cd nn-observability
uv sync

just test                          # run tests
just check                         # lint + format check
just reproduce                     # reproduce MLP + GPT-2 results

uv sync --extra transformer        # transformer dependencies
just transformer model=gpt2        # run a single model
just phase9                        # cross-family evaluation
```

## Reproducing the analysis

The CPU analysis scripts run on the committed result JSONs without any GPU:

```bash
cd analysis && python run_all.py
```

This produces the permutation test (p = 0.014), mixed-effects model (88% family variance), ANCOVA, selectivity analysis, and funnel plot. All scripts import from `load_results.py`, the single source of truth for which result files are in scope.

## Repository structure

```
src/                       Core library
  observe.py                 MLP observer
  transformer_observe.py     Transformer observer
  selective_prediction.py    TriviaQA selective prediction

scripts/                   Data collection scripts (one per model)
notebooks/                 Colab/Jupyter notebooks for GPU collection

analysis/                  Statistical analysis (all CPU)
  load_results.py            Single source of truth for result loading
  run_all.py                 Run all analysis scripts
  nonlinear_probe.py         Linear vs MLP probe comparison
  meta_regression.py         Mixed-effects model + variance decomposition
  permutation_test.py        Exact permutation test for family effect
  ancova_family.py           ANCOVA supplement
  selectivity.py             Random head baselines + control gap
  pearson_vs_spearman.py     Rank correlation methodology check
  loocv_scaling.py           Leave-one-out CV on Qwen scaling
  funnel_plot.py             Publication bias diagnostic

results/                   All result JSONs (committed)
assets/                    Figures and images
tests/                     Pytest suite
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
