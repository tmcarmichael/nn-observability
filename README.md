# Architecture Predicts Linear Readability of Decision Quality in Transformers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19435674.svg)](https://doi.org/10.5281/zenodo.19435674)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

> *Transformers carry internal error signals long before output. Architecture determines whether those signals are linearly monitorable or effectively hidden.*

AI models can detect their own mistakes internally, but whether anyone can read that signal depends on architecture choices made before deployment. Some architectures make it readable, others don't. Current interpretability methods can't recover what the architecture didn't produce. Which architecture you deploy determines whether a 13-15% blind spot in confidence monitoring is visible or permanently opaque.

## Key findings

**Confidence controls eliminate most of what probes measure.**
Raw Spearman correlation between a linear probe and per-token loss is +0.55 on GPT-2 124M. After controlling for max softmax probability and activation norm, only +0.28 survives. Four hand-designed activation statistics that show strong raw correlation all collapse to near zero under the same controls. The probe is standard; the measurement is not.

**The surviving signal is stable, linear, and reproducible.**
Twenty independent probe initializations on frozen GPT-2 124M converge to the same direction (rho_partial = +0.282 +/- 0.001, seed agreement +0.993). A nonlinear MLP does not exceed the linear probe at matched hyperparameters on any of eight models tested. A five-layer sweep across Llama 3.2 3B confirms the signal is absent at every depth under both linear and nonlinear probing.

**Architecture family predicts observability; scale does not.**
Under identical evaluation, Qwen 2.5 maintains rho_partial ~ +0.25 from 0.5B to 14B (28x range). GPT-2 is stable at +0.28 from 124M to 1.5B. Llama 3.2 3B produces +0.089, a 3.0x gap with Qwen at matched scale (permutation test p = 0.014, 88% of variance between families). Instruction tuning preserves the signal at every Qwen scale and improves its operational stability across monitoring budgets (at 30% flag rate, Qwen 7B instruct holds at 13.9% exclusive catches while base drops to 12.0%).

**The observer catches errors confidence can never see.**
At every flag rate tested, the observer catches errors confidence misses: 6-7% of all errors at 5% flag rate, 9-10% at 10%, saturating near 13-15% at 20%. This fraction is stable from GPT-2 124M through Qwen 2.5 14B, across base and instruct variants, and across WikiText-to-C4 domain transfer. Confidence precision is 1.000 on all Qwen variants: no confidence threshold would flag these tokens.

<p align="center">
<img src="assets/cross_family_scaling.png" width="85%" alt="Cross-family scaling">
</p>

*Confidence-independent decision quality signal across five transformer families. Same evaluation protocol, same data, same controls. Qwen, GPT-2, and Mistral maintain a stable signal. Llama drops sharply above 1B. The difference is architectural. Shaded band marks the detection threshold. Open markers are preliminary (3 seeds).*

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
| Mistral 7B | Mistral | 7.2B | L22 (69%) | +0.313 | +0.156 | +0.995 |
| Llama 3.1 8B\* | Llama | 8B | L0 (0%) | +0.088 | +0.054 | --- |
| Qwen 2.5 14B | Qwen | 14B | L30 (62%) | +0.214 | +0.096 | +0.851 |

\*Llama 1B and 8B are preliminary (3 seeds, lower ex/dim). All other models use the full 7-seed protocol with matched token budgets per hidden dimension. Gemma's random untrained probe baseline is +0.213, indicating representational geometry rather than learned signal accounts for part of its high value.

## Quick start

```bash
git clone https://github.com/tmcarmichael/nn-observability
cd nn-observability
uv sync
just install-hooks                 # ruff on commit, version check on push

just test                          # run tests
just check                         # lint + format + version
just smoke-gpu                     # end-to-end smoke test (GPT-2 124M, ~3 min CPU)
```

## Adding a new model

The most valuable contribution is data from a new architecture family. On a GPU (RunPod, Colab, or local):

```bash
# Option A: script (recommended for RunPod / SSH)
uv sync --extra transformer
python scripts/run_model.py --model <hf-model-id> --output <name>_results.json

# Option B: notebook (recommended for Colab)
# Copy notebooks/qwen7b_base_instruct_v2.ipynb, change MODEL_ID
```

Then integrate:

```bash
cp <name>_results.json results/                 # 1. add the JSON
# 2. add one entry to analysis/load_results.py
# 3. add color + marker to figures/style.py
uv run python analysis/run_all.py               # 4. updated stats
just figures                                     # 5. new point in Figure 1
```

A 7B model takes ~2 hours on an H100. Token budgets must be at least 200 ex/dim (target 350 for models above 3B). See `results/README.md` for the output schema and `mistral7b_results.json` for a complete example.

## Reproducing the analysis

The CPU analysis scripts run on the committed result JSONs without any GPU:

```bash
uv run python analysis/run_all.py
```

This produces the permutation test (p = 0.004 with 5 families), mixed-effects model (88% family variance), ANCOVA, selectivity analysis, exclusive catch rates across flag rates, and funnel plot. All scripts import from `load_results.py`, the single source of truth for which result files are in scope.

## Repository structure

```
src/                       Core library
  probe.py                   Shared probing functions (partial correlation,
                               activation collection, probe training)
  observe.py                 MLP observer (phases 1-3)
  transformer_observe.py     Transformer observer (phases 5-9)
  selective_prediction.py    TriviaQA selective prediction

scripts/                   GPU data collection
  run_model.py               Parameterized entry point for any HF model
  *.py                       Per-model scripts (legacy, use run_model.py)

notebooks/                 Colab/Jupyter notebooks for GPU collection

analysis/                  Statistical analysis (all CPU, no GPU needed)
  load_results.py            Single source of truth for result loading
  run_all.py                 Run all analysis scripts
  meta_regression.py         Mixed-effects model + variance decomposition
  permutation_test.py        Exact permutation test for family effect
  exclusive_catch_rates.py   Multi-rate exclusive catch analysis
  nonlinear_probe.py         Linear vs MLP probe comparison
  ancova_family.py           ANCOVA supplement
  selectivity.py             Random head baselines + control gap
  pearson_vs_spearman.py     Rank correlation methodology check
  loocv_scaling.py           Leave-one-out CV on Qwen scaling
  funnel_plot.py             Publication bias diagnostic

figures/                   Paper figure generation scripts
  style.py                   Shared matplotlib style and palette
  fig_cross_family.py        Figure 1: cross-family scaling
  fig_layer_profiles.py      Figure 2: Qwen vs Llama layer sweep
  fig_waterfall.py           Figure 3: control sensitivity cascade
  fig_exdim.py               Figure 4: ex/dim sensitivity
  generate_all.py            Regenerate all figures

results/                   All result JSONs (committed, reproducible)
assets/                    Figures for README
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
