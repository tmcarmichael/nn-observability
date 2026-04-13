# Architecture Predicts Linear Readability of Decision Quality in Transformers

**DOI:** [10.5281/zenodo.19435674](https://doi.org/10.5281/zenodo.19435674) | **License:** [MIT](LICENSE) | **Python:** 3.12+ | **Paper:** [PDF](https://doi.org/10.5281/zenodo.19435674)

We measure whether transformer activations contain error signals beyond output confidence, across 12 models in 5 architecture families. They do, but only in some architectures, and the difference is 3x at matched scale. Half of what standard probes measure is confidence in disguise. The residual signal catches 7-15% of errors that no output-based method can see.

<p align="center">
<img src="assets/cross_family_scaling.png" width="85%" alt="Cross-family scaling: Mistral, GPT-2, and Qwen maintain stable observability signal across scale. Llama drops sharply above 1B.">
</p>

*Confidence-independent decision quality signal across five architecture families. Same evaluation protocol, same data, same controls. Qwen, GPT-2, and Mistral maintain a stable signal. Llama drops sharply above 1B. The difference is architectural. Open markers are preliminary (3 seeds).*

## Why this matters

For **probing research**: standard probe evaluations overstate what the probe finds. Raw correlation between a probe and per-token loss is +0.55 on GPT-2 124M. After controlling for max softmax and activation norm, +0.28 survives. Every probing result that reports raw correlation without confidence controls may be measuring confidence, not the claimed property.

For **deployment monitoring**: output confidence has a blind spot. At 10% flag rate, a linear probe catches 7-11% of errors confidence misses, across five families. At 20% flag rate all architectures converge to 12-15% exclusive catches. This floor holds even on Llama (pcorr +0.089), where the signal is barely above noise. Choosing the right architecture determines whether this blind spot is visible.

## Key findings

**Half the raw signal is confidence.**
Raw probe-loss correlation is +0.55. After controlling for max softmax and activation norm: +0.28. A nonlinear MLP control confirms the residual is genuine (+0.289), not an artifact of linear deconfounding. Adding logit entropy absorbs another 30%. What remains is a signal the output distribution cannot express.

**The residual is one direction, not a subspace.**
Twenty probe initializations on frozen GPT-2 124M converge to the same direction (+0.282 +/- 0.001, seed agreement +0.993). A nonlinear MLP does not exceed the linear probe on any of eight models tested. A five-layer sweep on Llama 3.2 3B confirms the signal is absent at every depth under both linear and nonlinear probing.

**Architecture determines readability; scale does not.**
Mistral 7B: +0.313. GPT-2 (124M-1.5B): +0.290. Qwen 2.5 (0.5B-14B): +0.25. Gemma 3 1B: +0.388 raw, +0.175 net of random baseline. Llama 3.2 3B: +0.089. The 88% of variance falls between families; parameter count contributes nothing (p = 0.950). Instruction tuning preserves the signal at every Qwen scale.

**Exclusive catch rate saturates across architectures.**
At 10% flag rate: 7.8% (Llama) to 11.4% (Mistral). At 20%: all five families converge to 12-15%. A 3.5x range in observability compresses to 1.2x in catch rate. The ceiling is set by error structure, not architecture.

**The signal transfers; the calibration data doesn't.**
WikiText-trained probes transfer to C4 on all four non-Gemma families tested. Training directly on C4 fails on all four. The signal lives in the representations regardless of domain. The binary target breaks on noisy web text where high loss reflects formatting, not model uncertainty.

## Cross-family results

| Model | Family | Params | rho_partial | +/- std | r_OC | tau_seed | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen 2.5 0.5B | Qwen | 0.5B | +0.215 | 0.020 | +0.059 | +0.959 | |
| Gemma 3 1B | Gemma | 1B | +0.388 | 0.004 | +0.307 | +0.980 | Random head +0.213 |
| Llama 3.2 1B | Llama | 1.2B | +0.250 | --- | +0.126 | +0.999 | Preliminary (3 seeds) |
| Qwen 2.5 1.5B | Qwen | 1.5B | +0.275 | 0.032 | +0.127 | +0.953 | |
| GPT-2 XL | GPT-2 | 1.5B | +0.290 | 0.004 | +0.174 | +0.952 | |
| Qwen 2.5 3B | Qwen | 3B | +0.263 | 0.021 | +0.144 | +0.925 | |
| Llama 3.2 3B | Llama | 3B | +0.089 | 0.003 | +0.033 | +0.999 | |
| Qwen 2.5 7B | Qwen | 7B | +0.255 | 0.019 | +0.137 | +0.964 | |
| Mistral 7B | Mistral | 7.2B | +0.313 | 0.001 | +0.156 | +0.995 | |
| Llama 3.1 8B | Llama | 8B | +0.088 | --- | +0.054 | --- | Preliminary (3 seeds) |
| Qwen 2.5 14B | Qwen | 14B | +0.214 | 0.032 | +0.096 | +0.851 | |

All models use the 7-seed protocol with 350 ex/dim unless noted. rho_partial is Spearman partial correlation after controlling for max softmax and activation norm. r_OC is the residual after controlling for a 64-unit output-layer predictor. tau_seed is mean pairwise Spearman across probe initializations.

## Status

Phi-3 Mini full protocol running (sixth family). Llama 1B and 8B full-protocol reruns in progress. r_OC width sweep (128/256/512-unit output predictor on Qwen 7B) queued. Six families expected by end of April 2026.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Analysis runs on CPU; model evaluation requires a GPU with 16GB+ VRAM.

```bash
git clone https://github.com/tmcarmichael/nn-observability
cd nn-observability
uv sync
just install-hooks                 # ruff on commit, version check on push

just test                          # run tests (54 pass, CPU only)
just check                         # lint + format + version
just smoke-gpu                     # end-to-end smoke test (GPT-2 124M, ~3 min CPU)
```

## Adding a new model

The most valuable contribution is data from a new architecture family. One file, one command:

```bash
# On GPU (RunPod, Colab, or local):
pip install transformers datasets scipy scikit-learn
python run_model.py --model <hf-model-id> --output <name>_results.json
```

A 7B model takes ~2 hours on an H100. The script is self-contained (no local imports). Token budgets should be at least 200 ex/dim (target 350 for models above 3B).

Then integrate:

```bash
cp <name>_results.json results/                 # 1. commit the JSON
# 2. add one tuple to analysis/load_results.py:
#    ('name_results.json', 7.0, 'Name-7B'),
# 3. add color + marker to figures/style.py
uv run python analysis/run_all.py               # 4. updated stats
just figures                                     # 5. new point in Figure 1
```

See `results/README.md` for the output schema and `mistral7b_results.json` for a complete example.

## Reproducing the analysis

The CPU analysis scripts run on committed result JSONs without any GPU:

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
  run_model.py               Single entry point for any HuggingFace model
  roc_width_sweep.py         Output-controlled residual width sweep

analysis/                  Statistical analysis (all CPU, no GPU needed)
  load_results.py            Single source of truth for result loading
  run_all.py                 Run all analysis scripts
  permutation_test.py        Exact permutation test for family effect
  meta_regression.py         Mixed-effects model + variance decomposition
  exclusive_catch_rates.py   Multi-rate exclusive catch analysis
  nonlinear_probe.py         Linear vs MLP probe comparison
  ancova_family.py           ANCOVA supplement (see pseudoreplication note)
  selectivity.py             Random head baselines + control gap
  funnel_plot.py             Publication bias diagnostic

figures/                   Paper figure generation
  generate_all.py            Regenerate all figures
  fig_cross_family.py        Cross-family scaling (Figure 1)
  fig_layer_profiles.py      Qwen vs Llama layer sweep (Figure 2)
  fig_waterfall.py           Control sensitivity cascade (Figure 3)
  fig_exdim.py               Token budget sensitivity (Figure 4)

results/                   All result JSONs (committed, reproducible)
tests/                     Pytest suite (54 tests)
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
