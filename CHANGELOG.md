# Changelog

All notable changes to this repository.

## v5.1.0 (2026-05-11)

No experiments rerun. No committed result JSONs change. Paper PDF unchanged. Zenodo concept DOI unchanged.

### Added

- `schema/README.md`, `reports/README.md`.
- `results/manifest_verification/2026-05-12.json`. 33/33 verified.

### Changed

- CI: `astral-sh/setup-uv@v4` with `enable-cache: true` and `cache-dependency-glob: "uv.lock"` across `lint`, `typecheck`, `test`, `manifest-verify`.
- Pre-commit: `ruff-pre-commit` v0.15.8 to v0.15.12, `pre-commit-hooks` v5.0.0 to v6.0.0.
- `CITATION.cff` abstract: rewritten to track v5.1.0 paper abstract.
- `pyproject.toml` description: rewritten to track v5.1.0 paper abstract.
- README `Observability collapse`: `(24L, 16H)` to `(24-layer, 16-head)`.

### Fixed

- README family-membership statistic: "91% of variance at p = 0.003" to "56.8% of variance, p = 0.053". Matches `reports/paper_values.json:varfamily`.

## v5.0.0 (2026-05-10)

No paper-cited result numbers change. Producer pipeline preserved bit-for-bit.

### Added

- `NOTICE`. Built-with-Llama and per-creator attribution for Llama, Gemma, Qwen, and transitive Apache 2.0 deps.
- `license` and `license_url` per entry in `results/model_revisions.json` (33) and `results/dataset_revisions.json` (7). Verified against HF card metadata.
- ORCID on creator block in `CITATION.cff` and `croissant.json`.
- Numerical-pinning test for `partial_spearman` and `compute_loss_residuals` at 1e-12 tolerance.
- Mypy coverage extended to `scripts/run_model.py` and `scripts/verify_manifest_revisions.py`.
- CUDA OOM handler in `run_model.py` model-load path. Re-raises with free/total VRAM and dtype context.
- CI job `manifest-verify` on every push.
- Pre-commit hook `validate-schemas` on result-JSON or schema changes.
- `croissant.json`: `conformsTo` declares Croissant 1.1 and RAI 1.0. Nine `rai:*` fields. RecordSets `hugging-face-models` (33) and `hugging-face-datasets` (7) with model id, commit SHA, license, license URL. Top-level `usageInfo` and `documentation` linking `MODELS.md`, `DATA.md`, `NOTICE`.
- Description coverage on `schema/*.schema.json`: 107/107.
- `DATA.md` sections: Datasheet alignment, per-dataset Composition, Pretraining-corpus overlap audit, PII and sensitive content, License compatibility, Recommended/out-of-scope uses.
- `DATA.md` entries: OpenWebText, CodeSearchNet (Python).
- `MODELS.md` sections: Model Card alignment, License compatibility, Required-attribution paragraph for four Llama checkpoints under Llama Community License Section 1(b)(i).
- README annotation on `coverage.json` scope.

### Changed

- `pyproject.toml` and `croissant.json` description: 33 checkpoints across 7 families (26 primary + 7 instruct).
- License URLs: SPDX for Apache 2.0 and MIT, HF blob for Llama Community License, Google AI for Gemma Terms of Use.
- `scripts/controlled_depth_width.py` and `scripts/controlled_training.py` WikiText loads pass `revision=` from `dataset_revisions.json`.
- Dataset-pinning guarantee scoped to `tests/test_script_preflight.py:SCRIPTS_REQUIRING_PREFLIGHT`. Local-dev MPS scripts and from-scratch trainer scripts exempt.
- `tests/test_paper_values.py` macro provenance thresholds: minimum with provenance 118 to 135 (43.7% to 50.0% of 271). Maximum orphan macros 153 to 136.

### Fixed

- Qwen 2.5-3B and Qwen 2.5-3B-Instruct license: Apache 2.0 to Qwen Research License (non-commercial).
- WikiText-103 license: CC BY-SA 4.0 to CC BY-SA 3.0 + GFDL.
- CodeSearchNet license: MIT to "other". MIT covers the upstream curation tool only.
- `reports/figure_sources.json` `oc_vs_pcorr.pdf`: 25 to 26 files (missing `pythia-1.4b-deduped_main.json`).
- `assets/share/README.md` slope: ~0.80 to ~0.48.
- Downstream producers (`rag_hallucination.py`, `medqa_selective.py`, `truthfulqa_hallucination.py`) auto-resolve `--peak-layer` from `<slug>_main.json:peak_layer_final`. Fail-fast on missing main JSON.
- New test `tests/test_downstream_protocol.py::test_downstream_peak_matches_main`. Parametrized across 9 committed downstream JSONs.
- `rag_hallucination.py` saves `per_question: all_results[:50]`. Prior path saved 1000 records.
- Source-text fields stripped from `per_question` in 9 committed downstream JSONs: `question`, `answer`, `gold`/`correct_answer`/`category`, `gold_letter`. Numeric scores, correctness flags, `pred_letter` preserved.
- `analysis/load_results.py`: `LLAMA_MODELS` adds Llama-3.2-1B and Llama-3.1-8B; `GEMMA_MODELS` adds Gemma-3-4B. Registered model count 26. New scope `absorption_cohort_14`.
- `scripts/pythia_1.4b_shuffle.py` reference values load from `pythia-1.4b_main.json`. Fail-fast on missing file. Removes hardcoded fallback.
- Stale numeric values removed from comments/docstrings: `analysis/selectivity.py`, `analysis/load_results.py`, `scripts/pythia_1.4b_shuffle.py`, `scripts/mistral7b_instruct_full_mps.py`, `scripts/mechanistic_mistral.py`.

### Removed

- Docker artifacts: `Dockerfile`, `.dockerignore`, `.hadolint.yaml`, four `docker-*` justfile recipes, Hadolint CI step. Past tags retain Dockerfile in Zenodo snapshots.
- TriviaQA experiment helpers in `src/selective_prediction.py`. `selective-prediction` justfile recipe. Utility functions retained for `tests/test_selective_prediction.py`.

### Notes

- `src/transformer_observe.py` held as single 3,767-line module. Refactor would shift committed result-JSON values.

## v4.0.0 (2026-05-05, arXiv v2)

Companion code release for arXiv v2. Paper retitled "Architectural Observability Collapse in Transformers" (was "Architecture Determines Observability in Transformers" in arXiv v1).

### New experiments

- `scripts/run_residualizer_split.py`: OLS residualizer fit on disjoint document pool R, applied without refit to probe-training and evaluation pools. Four regimes: GPT-2 124M, Pythia 1B, Pythia 1.4B, Llama 3.2 3B. All four pass pre-specified regime-preservation criterion; max same-pool $|\Delta\rho_{\rm partial}| = 0.017$. Results in `results/*_residualizer-split.json`.
- Pythia 1B and 1.4B checkpoint dynamics at matched hidden dimension (d=2048), 10 checkpoints from 0.5B to 300B tokens. Both form signal at earliest measured checkpoint. Training erases in 1.4B (24L/16H); 1B (16L/8H) recovers. Results in `results/pythia-{1b,1.4b}_dynamics.json`.
- Gemma 3 1B canonical-protocol rerun at 350 ex/dim. Replaces 150 ex/dim under-saturated measurement. $\rho_{\rm partial} = 0.216$, mid-layer peak L11.
- Qwen 2.5 32B added. Within-Qwen range 0.5B to 32B (six base sizes). `results/qwen2.5-32b_main.json`.
- GPT-2 family reruns at 350 ex/dim, 7-seed (124M, 355M, 774M, 1.5B). Replaces non-uniform per-size protocol from v3.3.x. `results/gpt2-{124m,medium,large,xl}_main.json`.
- 10-permutation shuffle of binary targets on GPT-2 124M. Backs five-sigma shuffle null. `results/gpt-2-124m_shuffle-control.json`.

### Reproducibility infrastructure

- `reports/paper_values.json`: 271 macros; 118 with `source_files` + `key_paths` + `formula` provenance. Coverage floor enforced.
- `reports/scopes.json`: `cross_family_14`, `pythia_controlled_9`. Drift test enforced.
- `reports/figure_sources.json`: PDF figure to generator script and source JSONs.
- `schema/` JSON Schemas (Draft 2020-12) for main, dynamics, residualizer-split, nonlinear-probe, mechanistic, downstream, shuffle-control, bootstrap, width-sweep, legacy. Dispatched by filename pattern in `scripts/validate_schemas.py:DISPATCH`. CI validates every committed result JSON.
- Manifest verification against HF API for every entry in `results/model_revisions.json`. Report at `results/manifest_verification/2026-05-03.json`.
- Croissant 1.1 manifest at `croissant.json`. Validated against `mlcroissant` reference spec via `just check-croissant`.
- Schema enforces `provenance.device == "cuda"` on every committed `*_main.json`, `*_dynamics.json`, `*_residualizer-split.json`. MPS for local dev only.
- Dataset revision pinning via `results/dataset_revisions.json`. Every `load_dataset(...)` in `tests/test_script_preflight.py:SCRIPTS_REQUIRING_PREFLIGHT` passes pinned `revision=`.
- `tests/test_paper_values.py`: 97 tests, 2 module-level skips. Schema validation, provenance integrity, direct-read auto-verification, exporter idempotency, scope membership, figure source-file existence, `paper_version` consistency with `main.tex` `\paperversion`.

### Methodology

- 350 ex/dim protocol standardized across paper-cited result JSONs. Cross-family scope: 14 models in 6 families.
- 20-seed hardening on GPT-2 124M (L11): $\rho_{\rm partial} = 0.282 \pm 0.001$, 95% CI [0.282, 0.283], per-seed range [0.279, 0.284], seed agreement 0.993.
- Three-pool protocol (R/T/V) for residualizer-fit split: disjoint WikiText-103 documents at canonical token budget.

### Removed

- Mechanistic ablation appendix (Appendix G) from paper. Directional ablation, mean-ablation patching, layer formation of output-independent component. Result JSONs (`*_mechanistic.json`) and `transformer_observe.json` mechanistic blocks retained as historical artifacts; not paper-cited.
- Layer-1 ablation pattern subsection (A.12) from Methodology hardening appendix. Llama 3.2 1B vs 3B and Mistral 7B sign-reversal observation.

### Documentation

- Three-path verification structure in `README.md`: structured claim provenance, targeted CLI verification, full pipeline. Worked example: `confabsorbmean` from `paper_values.json` through 14 source JSONs.

### Numerical updates

- Confidence absorption at canonical protocol: $60.3\% \pm 7.0\%$ across 14 models in 6 families (was 58.2% in v3.3.x).

### v3.3.0 (arXiv v1) baseline

Tagged `arxiv-v1`.
