[![Paper](https://img.shields.io/badge/paper-preprint-B31B1B.svg)](https://doi.org/10.5281/zenodo.19435674)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19435674.svg)](https://doi.org/10.5281/zenodo.19435674)
[![CI](https://github.com/tmcarmichael/nn-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/tmcarmichael/nn-observability/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

# Architecture Determines Observability in Transformers

**[Read the paper](https://doi.org/10.5281/zenodo.19435674)** (preprint)

A linear probe on frozen mid-layer activations detects transformer errors that output confidence misses. Whether this signal exists depends on architecture, not scale or training data.

#### 5 to 9% of confident model errors at 10% flag rate are invisible to the output distribution. Confidence thresholds miss them. Calibrated probabilities miss them. A trained predictor on the full output representation misses them. They reach users undetected.

A single dot product on frozen mid-layer activations catches them. No fine-tuning, no task-specific data. A probe trained on Wikipedia reads the same failure signal zero-shot on medical licensing questions and retrieval-augmented QA.

Which model you deploy determines whether this signal exists at all. Some architectures undergo **observability collapse**: the mid-layer readable signal falls from +0.21 to +0.10 and stays there. No layer recovers it. A nonlinear probe does not recover it. The information is not preserved in linearly readable form.

<p align="center">
<img src="assets/share/within_family_cliff.png" width="95%" alt="Two panels showing observability collapse in two training recipes. Left, Llama: 1B rises to +0.28, while 3B and 8B stay flat near +0.05 to +0.10 across all layers. Right, Pythia: six sizes peak between +0.20 and +0.38, while 410M and 1.4B (both 24 layers, 16 heads) stay flat near +0.10.">
</p>

Both panels are the same protocol, the same token budget per hidden dimension, and the same shaded detection band. Left panel, Llama 3.2 under a cross-recipe split: 1B preserves the signal, 3B and 8B do not. Right panel, Pythia under held-recipe training: three of eight configurations collapse, and all three are (24 layers, 16 heads). The replication is across a 3.5x parameter gap, two Pile variants (original and deduplicated), and two hidden dimensions. Six other Pythia depths are healthy. No intermediate values appear.

## What this repo contains

The code, data, and analysis behind [the paper](https://doi.org/10.5281/zenodo.19435674). Every number in the PDF traces to a committed JSON in `results/` through an automated verification pipeline.

```bash
git clone https://github.com/tmcarmichael/nn-observability
cd nn-observability
uv sync                             # or: pip install -e .

uv run pytest tests/ -q             # 410 tests, CPU only, schema + property + smoke
uv run python analysis/run_all.py   # permutation test, mixed-effects, variance decomposition
```

## The finding

Half to two-thirds of what standard probes measure is confidence in disguise. Raw probe-loss correlation on GPT-2 124M is +0.55. After controlling for max softmax and activation norm: +0.28 survives. Four hand-designed activation statistics that show strong raw correlation all collapse to near zero under the same controls.

The signal that survives is real, linear, and output-independent. Twenty probe initializations converge to the same direction (+/- 0.001). A nonlinear MLP is statistically equivalent. A 512-unit output predictor absorbs no more than a 64-unit bottleneck. The information exists in the model's hidden layers, and the output layer discards it. Output-independence grows with scale: the residual signal is 34% at GPT-2 124M and 60% at GPT-2 XL.

Scale does not predict whether the signal is present. Configuration does. At matched 3B scale, Qwen produces +0.263 and Llama produces +0.091, a 2.9x gap with non-overlapping per-seed distributions. Within Llama 3.2, the signal is present at 1B (+0.286) and absent at 3B (+0.091) and 8B (+0.093). Under Pythia's held-recipe training, both (24 layers, 16 heads) configurations collapse to ~+0.10, with a third replication on the deduplicated Pile variant. Across 13 cross-family models, family membership explains 92% of the variance (permutation p = 0.006).

## Reproduce a paper number

Every number in the paper traces to a committed JSON through an automated verification pipeline. Pick a claim and verify it:

| Paper claim                                | Value     | Command                                           | Source                                       |
| ------------------------------------------ | --------- | ------------------------------------------------- | -------------------------------------------- |
| Cross-family permutation F (family effect) | p = 0.006 | `uv run python analysis/permutation_test.py`      | 13-model scope in `analysis/load_results.py` |
| Llama 1B partial correlation               | +0.286    | `uv run python analysis/load_results.py`          | `results/llama1b_v3_results.json`            |
| Exclusive catch rate at 20% flag rate      | 12-15%    | `uv run python analysis/exclusive_catch_rates.py` | `results/transformer_observe.json` key `6a`  |

## The cross-family comparison

| Model        | Family    | Params | pcorr      | OC residual |
| ------------ | --------- | ------ | ---------- | ----------- |
| Gemma 3 1B\* | Gemma     | 1B     | +0.388\*   | +0.307      |
| Mistral 7B   | Mistral   | 7B     | +0.313     | +0.156      |
| Phi-3 Mini   | Phi       | 3.8B   | +0.300     | +0.144      |
| GPT-2 XL     | GPT-2     | 1.5B   | +0.290     | +0.174      |
| Llama 1B     | Llama     | 1.2B   | +0.286     | +0.120      |
| Qwen 7B      | Qwen      | 7B     | +0.255     | +0.137      |
| **Llama 3B** | **Llama** | **3B** | **+0.091** | **+0.031**  |
| **Llama 8B** | **Llama** | **8B** | **+0.093** | **-0.007**  |

Sorted by signal strength. Every row except the bold Llama entries produces observability above +0.19. Gemma 3 1B\* has anomalous representation geometry (random untrained probe achieves +0.213); its high score reflects this artifact rather than stronger observability. Within Llama, the signal is present at 1B (+0.286) and absent at 3B (+0.091). Same lab, same training pipeline, different architectural configuration.

**pcorr**: partial Spearman correlation between probe scores and per-token loss, controlling for max softmax probability and activation norm. **OC residual**: the additional partial correlation after also controlling for a trained MLP on the last-layer activations. All values are 7-seed means (3-seed for GPT-2 family) on WikiText-103, evaluated at each model's peak layer with matched token budget per hidden dimension. The full 13-model table with standard deviations, seed agreement, and random head baselines is in the [paper](https://doi.org/10.5281/zenodo.19435674).

<p align="center">
<img src="assets/share/oc_vs_pcorr.png" width="80%" alt="Scatter of 22 models showing output-controlled residual on the y-axis against confidence-controlled partial correlation on the x-axis. Bootstrap linear fit slope 0.88, with collapse points near the origin for Llama 3B and 8B and three Pythia (24L, 16H) configurations.">
</p>

Across 22 models spanning seven families, the output-controlled residual tracks partial correlation with slope 0.88. Collapse points sit near the origin. A monitoring tool that reads the mid-layer signal exposes information not recovered by the tested output-side predictors, and this surplus vanishes at exactly the configurations where the partial correlation collapses.

## What the observer does not catch

Three documented boundaries that any deployment should respect.

**Fluent factual errors.** TruthfulQA isolates the subset of confidently wrong answers where the model asserts a smooth falsehood. The observer scores at chance on this subset (AUC 0.499 to 0.568 across three production instruct models). Activation monitoring catches token-level prediction failures, not learned falsehoods.

**Architectures where the signal collapsed.** Llama 3B and 8B, and Pythia (24 layers, 16 heads). On these, the linear probe scores at the detection floor and a held-out-tuned nonlinear probe does not cross it. Whether a deployed model is observable in this sense is a property of the architecture, not a property of better tooling.

**Adversarial evasion.** McGuinness et al. (2025) show that activation monitors can be evaded under training pressure. The observer has not been tested against adaptive attacks. PC1 cosine of 0.002 indicates the observer direction is not on a dominant variance axis, but the threat model still applies.

The observer's value is the complementary catch: errors confidence marks correct. Confidence has higher single-signal precision at every flag rate. Use both.

## Run it on your model

```bash
pip install -e ".[transformer]"   # or: uv sync --extra transformer

python scripts/run_model.py \
  --model Qwen/Qwen2.5-7B \
  --output qwen7b_results.json
```

This runs the full protocol: layer sweep, 7-seed evaluation, output-controlled residual, cross-domain transfer, control sensitivity, and flagging analysis. Output is a self-contained JSON with provenance metadata.

To add the result to the analysis scope, validate the JSON and add one line to `analysis/load_results.py`:

```bash
just validate-results                          # check required fields
```

```python
# In analysis/load_results.py, add to the appropriate family list:
QWEN_MODELS = [
    ...
    ("qwen7b_v3_results.json", 7.0, "Qwen 7B"),   # existing
    ("your_model_results.json", 7.0, "Your 7B"),   # new entry
]
```

Then `uv run python analysis/run_all.py` includes the new model in every statistical test. See `analysis/README.md` for the full schema and checklist.

## Repository structure

```
src/                  Core library (probe, observer, experiment engine)
scripts/              GPU experiment launchers (run_model.py is the entry point)
analysis/             CPU statistical analysis (permutation test, mixed-effects, schema validation)
results/              All result JSONs (committed, reproducible, schema-validated)
figures/              Shared matplotlib style and save helper
tests/                Schema, metrics, analysis smoke, probe-sync drift guards
```

Full directory map and script descriptions in `analysis/README.md` and `results/README.md`.

## Using the analysis library

The `analysis` package is the stable public API (v3.x). Install the repo as a package (`uv sync` or `pip install -e .`) and import directly:

```python
from analysis import load_all_models, load_model_means, family_f_stat, validate_all
```

Nine exported functions cover data loading, statistical primitives, and schema validation. See `analysis/__init__.py` for the full list with descriptions.

## Where to find what

| I want to                                  | Start here                                                      |
| ------------------------------------------ | --------------------------------------------------------------- |
| Read the paper                             | [Zenodo pre-print](https://doi.org/10.5281/zenodo.19435674)     |
| Run the tests                              | `uv run pytest tests/ -q`                                       |
| Run the full analysis pipeline             | `uv run python analysis/run_all.py`                             |
| Reproduce a specific paper number          | "Reproduce a paper number" table above                          |
| See the raw experimental data              | `results/*.json` (every paper number traces here)               |
| Walk through the analysis pipeline         | `notebooks/walkthrough_analysis.ipynb` (CPU-only, no GPU)       |
| Use the analysis library in your own code  | `analysis/__init__.py` (public API, stable across v3.x)         |
| Add my own model to the cross-family scope | "Run it on your model" section above, then `analysis/README.md` |
| Understand the result-JSON schema          | `analysis/load_results.py` and `results/README.md`              |
| Look at how a specific number was produced | `notebooks/README.md` (per-model run history)                   |

## Citation

Cite the paper and the code separately. Both share a Zenodo concept DOI that resolves to the latest version; pin to a specific version DOI from the [Zenodo record](https://doi.org/10.5281/zenodo.19435674) for reproducibility.

```bibtex
@article{carmichael2026observability,
  title={Architecture Determines Observability in Transformers},
  author={Carmichael, Thomas},
  year={2026},
  journal={Zenodo pre-print},
  doi={10.5281/zenodo.19435674},
  url={https://doi.org/10.5281/zenodo.19435674},
  note={v3.1.0}
}

@software{carmichael2026code,
  title={nn-observability: code for ``Architecture Determines Observability in Transformers''},
  author={Carmichael, Thomas},
  year={2026},
  version={3.1.0},
  doi={10.5281/zenodo.19435674},
  url={https://github.com/tmcarmichael/nn-observability}
}
```

## License

[MIT License](LICENSE)
