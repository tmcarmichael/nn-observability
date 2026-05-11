[![arXiv](https://img.shields.io/badge/arXiv-2604.24801-b31b1b.svg)](https://arxiv.org/abs/2604.24801)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19435674-blue)](https://doi.org/10.5281/zenodo.19435674)
[![CI](https://github.com/tmcarmichael/nn-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/tmcarmichael/nn-observability/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

# Architecture Determines Observability of Transformers

**[Read the paper](https://arxiv.org/abs/2604.24801)** (arXiv)

Transformer activations carry information about which tokens will be wrong that output confidence does not expose. Whether this signal exists at all depends on which model you deploy. Training can erase it while the model keeps getting better at its task.

A frozen linear probe, one dot product per token, reads this signal with no fine-tuning and no task-specific data. A probe trained on Wikipedia catches the same errors zero-shot on medical licensing questions and reading comprehension. Standard probing methodology overstates the signal by a factor of two to three: confidence controls absorb 60.3% of the raw probe signal across 14 models in 6 families. After controlling for confidence, the surviving signal is stable across 20 seeds and output-independent. A trained MLP on last-layer activations does not recover it.

## Observability collapse

**Whether this signal exists in a given model is determined before deployment.** Under Pythia's controlled training, both matched-width configurations form the signal at the earliest measured checkpoint. Training then erases it in the (24L, 16H) class while perplexity improves monotonically in both configurations through the collapse. Architecture does not prevent the signal from appearing. It determines whether training preserves or erases it.

The result is **observability collapse**: the decision-quality signal that neither confidence nor output-layer predictors recover falls to the detection floor at every measured layer. The collapse survives the standard escape hatches: it is not layer choice, probe nonlinearity, underpowered training, or final-layer predictor capacity. Six other Pythia configurations stay healthy across a 170x parameter range.

The pattern replicates across families and training recipes. At matched 3B scale, Qwen and Llama differ by 2.9x with non-overlapping seed distributions. Mistral 7B preserves the signal where Llama 3.1 8B collapses despite similar architecture. The collapse map changes across recipes, but the phenomenon persists. Family membership explains 91% of variance at p = 0.003.

## Implications

**Signal engineering: architecture and training recipe determine which internal signals persist in frozen trained models.** The mechanism that erases decision-quality signal pushes representation geometry toward structures where a low-variance direction nearly orthogonal to the dominant variance axes cannot persist. Decision quality is the first demonstrated target. The same configuration-class collapse plausibly affects other oversight-relevant signals (factuality structure, refusal appropriateness, retrieval faithfulness, reward-model disagreement, tool-use reliability), though this paper does not establish it. The protocol carries over: define the target, residualize against confidence and output-side predictors, train a frozen mid-layer observer, and test the same evidence hierarchy.

**Predictive capability and monitorability are not inherently in tension.** In Pythia's controlled training, the 16-layer, 8-head configuration preserves both through convergence, at comparable final perplexity to the 24-layer, 16-head class that loses observability. Decision-quality preservation is not a capability tax. The natural next problem is signal engineering: training models not only to predict well, but to retain readable internal evidence for oversight.

**Monitorability has a ceiling set during training.** A probe trained on Wikipedia, with no task-specific data, transfers zero-shot to SQuAD, MedQA, and TruthfulQA. It exclusively catches 10.9-13.4% of all errors at 20% flag rate, errors that confidence marks correct, across seven of nine downstream model-task cells. When observability collapses, no post-hoc probe design recovers healthy-range signal. Architecture selection is a monitoring decision.

This ceiling is invisible to standard evaluation. Raw probes can confuse confidence with decision quality. Output confidence is a lossy interface: it exposes a prediction and a score, but discards internal evidence about whether that prediction is fragile. Access to activations is not the same as access to useful internal evidence. A white-box model can still be unobservable if training failed to preserve the relevant signal. Predictive capability can improve while monitorability is destroyed. Model selection must evaluate what a model can do and what internal evidence it preserves for oversight. Observability becomes an evaluation dimension alongside accuracy, latency, cost, and calibration.

## Representation geometry as a design target

The observable signal occupies a low-variance direction in representation space, nearly orthogonal to the dominant variance axes. The erasure is selective: some architecture-recipe configurations systematically push representation geometry toward structures where that direction cannot survive. These results turn representation geometry from a passive diagnostic object into an upstream design target that mediates tradeoffs between capability, interpretability, and monitorability. Architecture sets a geometric prior, training optimizes it, probing measures it, monitoring reads it out. Internal representation geometry becomes a first-class design variable, alongside loss, architecture, and data.

<p align="center">
<img src="assets/share/within_family_cliff.png" width="95%" alt="Two panels showing observability collapse in two training recipes. Left, Llama: 1B rises to +0.28, while 3B and 8B stay flat near +0.05 to +0.10 across all layers. Right, Pythia: six sizes peak between +0.20 and +0.38, while 410M and 1.4B (both 24 layers, 16 heads) stay flat near +0.10.">
</p>

Both panels use the same protocol, the same token budget per hidden dimension, and the same shaded detection band. Left panel: Llama 3.2 under a cross-recipe split, where 1B preserves the signal and 3B and 8B do not. Right panel: Pythia under held-recipe training, where three of nine configurations collapse, all sharing 24 layers and 16 heads. The replication spans a 3.5x parameter gap, two Pile variants, and two hidden dimensions. No intermediate values appear.

<p align="center">
<img src="assets/share/oc_vs_pcorr.png" width="80%" alt="Scatter of 26 models showing output-controlled residual on the y-axis against confidence-controlled partial correlation on the x-axis. Bootstrap linear fit slope ≈0.48, with collapse points near the origin for Llama 3B and 8B and three Pythia (24L, 16H) configurations.">
</p>

26 models, seven families. The x-axis is pcorr (partial Spearman correlation between probe scores and per-token loss, controlling for confidence and activation norm). The y-axis is the output-controlled residual: what remains after also controlling for a trained last-layer predictor. Collapse points cluster near the origin. Where pcorr collapses, the surplus over output-side prediction vanishes with it.

## What this repo contains

The code, data, and analysis behind the paper. Every paper-cited number is derived from committed JSONs. `reports/paper_values.json` enumerates every macro the paper text uses. The directly-readable subset carries full `source_files` + `key_paths` + `formula` annotations; the live count cannot regress across releases. Every result JSON validates against a formal Draft 2020-12 schema in `schema/`. Every model revision in `results/model_revisions.json` is SHA-verified against the Hugging Face API. The full result-file inventory is published as a Croissant 1.1 metadata descriptor at `croissant.json` (validated against the official MLCommons spec).

```bash
git clone https://github.com/tmcarmichael/nn-observability
cd nn-observability
uv sync

uv run pytest tests/ -q
uv run python analysis/run_all.py
```

## Verify a paper claim

Three independent paths, in increasing depth:

**Path A: structured claim provenance.** `reports/paper_values.json` carries every macro the paper cites. Every annotated entry includes `source_files`, `key_paths`, `formula`, and `scope`. The live counts (`n_macros`, `n_macros_with_provenance`) are at the top of the JSON, and the count cannot regress. Pick any annotated macro and walk the chain by hand:

```python
import json
import numpy as np

pv = json.load(open("reports/paper_values.json"))
macro = next(m for m in pv["macros"] if m["name"] == "confabsorbmean")

deltas = []
for fname in macro["source_files"]:
    cs = json.load(open(f"results/{fname}"))["control_sensitivity"]
    deltas.append((cs["none"] - cs["standard"]) / cs["none"] * 100)

print(np.mean(deltas))
```

`reports/scopes.json` carries the named scopes (`cross_family_14`, `pythia_controlled_9`, etc.; membership is mirrored from `analysis/load_results.py:SCOPES` and locked by a drift test). `reports/figure_sources.json` maps each PDF figure to its source JSONs. `schema/` holds Draft 2020-12 JSON Schemas for every result type, dispatched by filename pattern in `scripts/validate_schemas.py:DISPATCH`. `tests/test_paper_values.py` enforces these guarantees end-to-end: every `source_files` entry exists, every `key_paths` resolves, every direct-read macro matches its JSON cell at the formatted precision, every result JSON validates against its dispatched schema, exporters are idempotent, `paper_version` matches `main.tex`, and macro coverage cannot regress below the minimum threshold.

**Path B: targeted CLI verification.** Pick a claim and run an analysis script:

| Paper claim                                | Value     | Command                                           | Source                                                |
| ------------------------------------------ | --------- | ------------------------------------------------- | ----------------------------------------------------- |
| Cross-family permutation F (family effect) | p = 0.003 | `uv run python analysis/permutation_test.py`      | `cross_family_14` scope in `analysis/load_results.py` |
| Llama 1B partial correlation               | +0.286    | `uv run python analysis/load_results.py`          | `results/llama-3.2-1b_main.json`                      |
| Exclusive catch rate at 20% flag rate (LM) | 12-15%    | `uv run python analysis/exclusive_catch_rates.py` | `results/transformer_observe.json` key `6a`           |

**Path C: full pipeline.** `uv run pytest tests/ -q` runs the test suite (formal schema validation across every result type, scope membership, `paper_values.json` integrity, direct-read auto-verification, exporter idempotency, figure-source existence, manifest revision pinning across every model-loading script, and a Python compile gate for the script set). `uv run python scripts/validate_schemas.py --strict` validates every result JSON against its dispatched schema and exits non-zero on any unmatched file. The paper-side `just check` layers content diffs against every generated artifact on top. CI runs both on every push.

Running `uv run pytest tests/ --cov-report=json` produces `coverage.json` at the repo root: line coverage from that run, reported by category rather than as a single headline percentage. The analysis library and statistical primitives that the paper's claim chain depends on (`src/probe.py`, `src/observe.py`, `src/utils.py`, `analysis/`, `scripts/validate_schemas.py`) carry the bulk of the test surface. The GPU experiment harness in `scripts/run_model.py` is exercised end-to-end by `just preflight-cuda` on the GPU host and pinned by `tests/test_smoke_run_model.py` against a committed fixture.

## Reproducibility environment

Paper-cited results were produced on the [RunPod PyTorch image](https://hub.docker.com/r/runpod/pytorch) `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. This image provides the system substrate (Ubuntu 24.04, CUDA 12.8.1, cuDNN, Python 3.12, PyTorch 2.8.0 on x86-64 Linux). The Python-package graph on top resolves from `uv.lock` under `--frozen`, so the install either reproduces the locked graph exactly or fails.

```bash
docker pull runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Inside the container:
git clone https://github.com/tmcarmichael/nn-observability && cd nn-observability
uv sync --frozen --extra transformer
just preflight-cuda       # end-to-end smoke on EleutherAI/pythia-70m
```

Host requirements: x86-64 Linux with NVIDIA GPU, CUDA driver >= 535, `nvidia-container-toolkit`, and VRAM sized for the target model. CUDA availability and the partial-correlation pipeline are verified end-to-end by `just preflight-cuda`.

## Reproduce an experiment

1. Pull the substrate per the section above and clone: `docker pull runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, then `git clone https://github.com/tmcarmichael/nn-observability && cd nn-observability && uv sync --frozen --extra transformer`.
2. Verify the environment: `just preflight-cuda` runs the canonical 350 ex/dim 7-seed protocol on `EleutherAI/pythia-70m` end-to-end, asserts `provenance.device == "cuda"`, and confirms the partial-correlation pipeline produces a finite value. Catches CUDA-specific regressions (kernel changes, dtype handling, device transfer bugs) before any larger run.
3. Run the full protocol on any Hugging Face model:

   ```bash
   uv run python scripts/run_model.py --model Qwen/Qwen2.5-7B --output qwen2.5-7b_main.json
   ```

   For the full Pythia controlled suite, run `just pythia-suite` (all configurations, sequential). Output is a self-contained JSON per model with full provenance: layer sweep, 7-seed evaluation, output-controlled residual, cross-domain transfer, control sensitivity, and flagging analysis. The manifests `results/model_revisions.json` and `results/dataset_revisions.json` pin Hugging Face model and eval-dataset SHAs and must be present. Every entry in `model_revisions.json` is programmatically verified against the Hugging Face API; the latest report under `results/manifest_verification/` records the verification timestamp and exact-SHA-match status per entry, and is regeneratable via `uv run --extra transformer python scripts/verify_manifest_revisions.py`. For strict reproduction, run with `HF_HOME=$(mktemp -d)` so cached versions cannot override `revision=`; this isolates the run from your personal Hugging Face cache.

   Pile is the upstream training corpus for Pythia and Pythia-deduped and is not a reproduction dependency. Reproducing this work does not require acquiring Pile. Probing data is WikiText, pinned in `dataset_revisions.json`. See [DATA.md](DATA.md) for per-dataset license, role in the paper, subset and transforms, and known limitations. See [MODELS.md](MODELS.md) for per-family model documentation: license, creator, training corpus, role in the paper, and family-level observability findings across the 33 evaluated checkpoints.

4. Schema-validate the new JSON against the recorded protocol: `just validate-results-strict`.

To add a new model to the analysis scope, see `analysis/README.md`. Local development without a GPU is sufficient for `uv run pytest tests/` and CPU analysis (`analysis/run_all.py`, `analysis/exclusive_catch_rates.py`).

## Pre-tag CUDA preflight

CI runs lint, typecheck, tests, and Croissant validation on CPU. Before cutting a release tag, run the CUDA preflight gate on the GPU host used for paper-quality runs:

```bash
just preflight-cuda
```

This runs the canonical 350 ex/dim 7-seed protocol on `EleutherAI/pythia-70m` (the smallest paper-scope model) end-to-end on CUDA, skipping the C4 cross-domain step to save time. It validates that `provenance.device == "cuda"`, that the result schema is intact, and that the partial-correlation pipeline produces a finite value. Expected runtime is a few minutes on a small GPU.

## Croissant metadata

A Croissant 1.1 metadata descriptor at `croissant.json` covers the full results inventory and the upstream provenance graph. The descriptor declares `conformsTo` for both Croissant 1.1 and the RAI 1.0 extension. `cr:FileObject` entries cover the source archive, the model and dataset revision manifests, and the latest Hugging Face verification report. `cr:FileSet` entries cover one glob per result file type. `cr:RecordSet` entries cover one record set per result file type with fields derived from `schema/*.schema.json`, plus two first-class record sets `hugging-face-models` (33 records) and `hugging-face-datasets` (7 records) exposing model id, pinned commit SHA, license identifier, and license URL as queryable fields. Nine `rai:*` properties cover data collection, preprocessing, PII posture, use cases, limitations, and maintenance plan. The creator block carries the author's ORCID. `usageInfo` and `documentation` link to [MODELS.md](MODELS.md), [DATA.md](DATA.md), and [NOTICE](NOTICE).

Regenerated by `just croissant` and gated by `just check-croissant`, which runs the official MLCommons validator. Spec: <https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html>. The parent FileObject's `sha256` is a deterministic merkle hash over `(filename, sha256)` pairs of every distribution file. To verify integrity, clone at the cited tag and rerun the generator: the regenerated `croissant.json` must match the committed file byte-for-byte.

## Repository structure

```
src/        Core library (probe, observer, experiment engine).
scripts/    GPU experiment launchers, schema validator, exporters.
analysis/   CPU statistical analysis.
schema/     Draft 2020-12 JSON Schemas, one per result type.
results/    Result JSONs, manifests, verification reports.
reports/    Cross-repo claim provenance (paper_values, scopes, figure_sources).
tests/      Schema, drift, integrity, and contract gates.
assets/     Paper figures and share-ready PNGs.
```

## Citation

Cite the [paper](https://arxiv.org/abs/2604.24801) and the [code](https://doi.org/10.5281/zenodo.19435674) separately:

```bibtex
@misc{carmichael2026observability,
  title={Architecture Determines Observability of Transformers},
  author={Carmichael, Thomas},
  year={2026},
  eprint={2604.24801},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  doi={10.48550/arXiv.2604.24801},
  url={https://arxiv.org/abs/2604.24801}
}

@software{carmichael2026code,
  title={nn-observability: code for ``Architecture Determines Observability of Transformers''},
  author={Carmichael, Thomas},
  year={2026},
  doi={10.5281/zenodo.19435674},
  url={https://github.com/tmcarmichael/nn-observability}
}
```

## License

[MIT License](LICENSE) covers the source code, result schemas, and human-readable documentation in this repository. Result JSON values are derived from upstream model activations; downstream use must respect the upstream terms recorded in [NOTICE](NOTICE). Eight evaluated model checkpoints are under restrictive licenses: four Llama 3.2 / 3.1 variants under the Llama Community License, two Gemma 3 sizes under the Gemma Terms of Use, and Qwen 2.5-3B base and 3B-Instruct under the Qwen Research License (non-commercial). The other 25 evaluated checkpoints are under permissive licenses (MIT or Apache 2.0). Per-model and per-dataset license URLs are recorded in `results/model_revisions.json` and `results/dataset_revisions.json`.
