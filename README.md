[![arXiv](https://img.shields.io/badge/arXiv-2604.24801-b31b1b.svg)](https://arxiv.org/abs/2604.24801)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19435674-blue)](https://doi.org/10.5281/zenodo.19435674)
[![CI](https://github.com/tmcarmichael/nn-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/tmcarmichael/nn-observability/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

# Architecture Determines Observability in Transformers

**[Read the paper](https://arxiv.org/abs/2604.24801)** (arXiv)

Transformer activations carry information about which tokens will be wrong that output confidence does not expose. Whether this signal exists at all depends on which model you deploy. Training can erase it while the model keeps getting better at its task.

A frozen linear probe, one dot product per token, reads this signal with no fine-tuning and no task-specific data. A probe trained on Wikipedia catches the same errors zero-shot on medical licensing questions and reading comprehension. Standard probing methodology overstates the signal by a factor of two to three: confidence controls absorb 57.7% of the raw probe signal across 13 models in 6 families. After controlling for confidence, the surviving signal is stable across 20 seeds and output-independent. A trained MLP on last-layer activations does not recover it.

## Observability collapse

**Whether this signal exists in a given model is determined before deployment.** Under Pythia's controlled training, both matched-width configurations form the signal at the earliest measured checkpoint. Training then erases it in the (24L, 16H) class while perplexity improves monotonically in both configurations through the collapse. Architecture determines observability not by preventing the signal from appearing, but by determining whether training preserves or erases it.

The result is **observability collapse**: the decision-quality signal that neither confidence nor output-layer predictors recover falls to the detection floor at every measured layer. The collapse survives the standard escape hatches: it is not layer choice, probe nonlinearity, underpowered training, or final-layer predictor capacity. Six other Pythia configurations stay healthy across a 170x parameter range.

The pattern replicates across families and training recipes. At matched 3B scale, Qwen and Llama differ by 2.9x with non-overlapping seed distributions. Mistral 7B preserves the signal where Llama 3.1 8B collapses despite similar architecture. The collapse map changes across recipes, but the phenomenon persists. Family membership explains 92% of variance at p = 0.006.

## Implications

**Monitorability has a ceiling set during training.** A probe trained on Wikipedia, with no task-specific data, transfers zero-shot to SQuAD, MedQA, and TruthfulQA. It exclusively catches 10.9-13.4% of all errors at 20% flag rate, errors that confidence marks correct, across seven of nine downstream model-task cells. When observability collapses, no post-hoc probe design recovers healthy-range signal. Architecture selection is a monitoring decision.

**This ceiling is invisible to standard evaluation.** Raw probes can confuse confidence with decision quality. Output confidence is a lossy interface: it exposes a prediction and a score, but discards internal evidence about whether that prediction is fragile. Access to activations is not the same as access to useful internal evidence; a white-box model can still be unobservable if training failed to preserve the relevant signal. **Predictive capability can improve while monitorability is destroyed.** Model selection must evaluate not only what a model can do, but what internal evidence it preserves for oversight. Observability becomes an evaluation dimension alongside accuracy, latency, cost, and calibration.

## Representation geometry as a design target

The observable signal occupies a low-variance direction in representation space, nearly orthogonal to the dominant variance axes. The erasure is selective: some architecture-recipe configurations systematically push representation geometry toward structures where that direction cannot survive. These results turn representation geometry from a passive diagnostic object into an upstream design target that mediates tradeoffs between capability, interpretability, and monitorability. Architecture sets a geometric prior, training optimizes it, probing measures it, monitoring reads it out. Internal representation geometry becomes a first-class design variable, alongside loss, architecture, and data.

<p align="center">
<img src="assets/share/within_family_cliff.png" width="95%" alt="Two panels showing observability collapse in two training recipes. Left, Llama: 1B rises to +0.28, while 3B and 8B stay flat near +0.05 to +0.10 across all layers. Right, Pythia: six sizes peak between +0.20 and +0.38, while 410M and 1.4B (both 24 layers, 16 heads) stay flat near +0.10.">
</p>

Both panels use the same protocol, the same token budget per hidden dimension, and the same shaded detection band. Left panel: Llama 3.2 under a cross-recipe split, where 1B preserves the signal and 3B and 8B do not. Right panel: Pythia under held-recipe training, where three of nine configurations collapse, all sharing 24 layers and 16 heads. The replication spans a 3.5x parameter gap, two Pile variants, and two hidden dimensions. No intermediate values appear.

<p align="center">
<img src="assets/share/oc_vs_pcorr.png" width="80%" alt="Scatter of 25 models showing output-controlled residual on the y-axis against confidence-controlled partial correlation on the x-axis. Bootstrap linear fit slope 0.88, with collapse points near the origin for Llama 3B and 8B and three Pythia (24L, 16H) configurations.">
</p>

25 models, seven families. The x-axis is pcorr (partial Spearman correlation between probe scores and per-token loss, controlling for confidence and activation norm). The y-axis is the output-controlled residual: what remains after also controlling for a trained last-layer predictor. Collapse points cluster near the origin. Where pcorr collapses, the surplus over output-side prediction vanishes with it.

## What this repo contains

The code, data, and analysis behind the paper. Every number in the PDF traces to a committed JSON in `results/` through an automated verification pipeline.

```bash
git clone https://github.com/tmcarmichael/nn-observability
cd nn-observability
uv sync                             # or: pip install -e . (requires Python 3.12+)

uv run pytest tests/ -q             # CPU only, schema + property + smoke
uv run python analysis/run_all.py   # permutation test, mixed-effects, variance decomposition
```

## Reproduce a paper number

Pick a claim and verify it:

| Paper claim                                | Value     | Command                                           | Source                                       |
| ------------------------------------------ | --------- | ------------------------------------------------- | -------------------------------------------- |
| Cross-family permutation F (family effect) | p = 0.006 | `uv run python analysis/permutation_test.py`      | 13-model scope in `analysis/load_results.py` |
| Llama 1B partial correlation               | +0.286    | `uv run python analysis/load_results.py`          | `results/llama1b_results.json`               |
| Exclusive catch rate at 20% flag rate (LM) | 12-15%    | `uv run python analysis/exclusive_catch_rates.py` | `results/transformer_observe.json` key `6a`  |

## Run it on your model

```bash
uv sync --extra transformer       # or: pip install -e ".[transformer]"

uv run python scripts/run_model.py \
  --model Qwen/Qwen2.5-7B \
  --output qwen7b_results.json
```

This runs the full protocol: layer sweep, 7-seed evaluation, output-controlled residual, cross-domain transfer, control sensitivity, and flagging analysis. Output is a self-contained JSON with provenance metadata.

To add a new model to the analysis scope, see `analysis/README.md`.

## Repository structure

```
src/                  Core library (probe, observer, experiment engine)
scripts/              GPU experiment launchers (run_model.py is the entry point)
analysis/             CPU statistical analysis (permutation test, mixed-effects, schema validation)
results/              All result JSONs (committed, reproducible, schema-validated)
tests/                Schema, metrics, analysis smoke, probe-sync drift guards
notebooks/            Walkthrough and per-model run history
assets/               Paper figures and share-ready PNGs
```

## Citation

Cite the [paper](https://arxiv.org/abs/2604.24801) and the [code](https://doi.org/10.5281/zenodo.19435674) separately:

```bibtex
@misc{carmichael2026observability,
  title={Architecture Determines Observability in Transformers},
  author={Carmichael, Thomas},
  year={2026},
  eprint={2604.24801},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2604.24801},
  url={https://arxiv.org/abs/2604.24801}
}

@software{carmichael2026code,
  title={nn-observability: code for ``Architecture Determines Observability in Transformers''},
  author={Carmichael, Thomas},
  year={2026},
  version={3.4.0},
  doi={10.5281/zenodo.19435674},
  url={https://github.com/tmcarmichael/nn-observability}
}
```

## License

[MIT License](LICENSE)
