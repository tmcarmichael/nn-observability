# Changelog

All notable changes to this repository.

## v5.0.0 (2026-05-10)

Major release. No paper-cited result numbers change. The producer pipeline is preserved bit-for-bit. Surface improvements span legal attribution, metadata, testing, CI, and documentation.

### Added

- `NOTICE` file at repo root with Built-with-Llama and per-creator attribution covering Llama, Gemma, Qwen Research License, and transitive Apache 2.0 dependencies.
- Per-record license attribution in `results/model_revisions.json` (33 entries) and `results/dataset_revisions.json` (7 entries). Each entry carries `license` and `license_url` fields verified against Hugging Face card metadata.
- ORCID on the creator block in `CITATION.cff` and `croissant.json`.
- Numerical-pinning test for `partial_spearman` and `compute_loss_residuals` at 1e-12 tolerance.
- Mypy coverage extended to `scripts/run_model.py` and `scripts/verify_manifest_revisions.py`.
- Narrow CUDA OOM handler in the `run_model.py` model-load path. Re-raises with free / total VRAM and dtype context.
- CI job `manifest-verify` runs `verify_manifest_revisions.py` on every push.
- Pre-commit hook `validate-schemas` runs the strict schema validator on result-JSON or schema changes.
- Croissant descriptor now declares `conformsTo` against both Croissant 1.1 and RAI 1.0.
- Nine `rai:*` fields in `croissant.json` covering data collection, preprocessing, PII posture, use cases, limitations, and maintenance plan.
- `hugging-face-models` and `hugging-face-datasets` RecordSets in `croissant.json` with 33 and 7 records respectively, exposing model id, pinned commit SHA, license, and license URL as queryable fields.
- Top-level `usageInfo` and `documentation` fields in `croissant.json` linking `MODELS.md`, `DATA.md`, and `NOTICE`.
- 100 percent description coverage on top-level fields across `schema/*.schema.json` (107 of 107).
- `DATA.md` sections for Datasheet alignment (Gebru et al. 2021), per-dataset Composition, Pretraining-corpus overlap audit, PII and sensitive content, License compatibility, and Recommended / out-of-scope uses.
- `DATA.md` entries for OpenWebText and CodeSearchNet (Python).
- `MODELS.md` sections for Model Card alignment (Mitchell 2019), License compatibility, and the Required-attribution paragraph naming the four evaluated Llama checkpoints under Llama Community License Section 1(b)(i).
- README annotation on the scope of `coverage.json`.

### Changed

- `pyproject.toml` and `croissant.json` description state 33 evaluated transformer checkpoints across 7 families (26 primary plus 7 instruct variants).
- License URLs unified to canonical sources. SPDX URLs for Apache 2.0 and MIT, Hugging Face blob URLs for Llama Community License, Google AI URL for Gemma Terms of Use.
- WikiText loads in `scripts/controlled_depth_width.py` and `scripts/controlled_training.py` now read `dataset_revisions.json` and pass `revision=` to `load_dataset`.
- Dataset-pinning prose in `DATA.md`, `dataset_revisions.json`, and this changelog scopes the guarantee explicitly to `tests/test_script_preflight.py:SCRIPTS_REQUIRING_PREFLIGHT` rather than "repo-wide". Local-dev MPS scripts and from-scratch trainer scripts are intentionally exempt.
- `tests/test_paper_values.py` macro provenance thresholds tightened. Minimum macros with provenance raised from 118 to 135 (43.7 to 50.0 percent of 271 total). Maximum allowed orphan macros lowered from 153 to 136.

### Fixed

- Qwen 2.5-3B and Qwen 2.5-3B-Instruct license corrected from Apache 2.0 to the Qwen Research License (custom, non-commercial). The other Qwen 2.5 sizes remain Apache 2.0.
- WikiText-103 license corrected from CC BY-SA 4.0 to dual CC BY-SA 3.0 plus GFDL. The Hugging Face card declares both, reflecting Wikipedia's pre-June-2023 source-content license.
- CodeSearchNet license corrected from MIT to "other". MIT covers the upstream curation tool. Constituent function bodies retain heterogeneous original licenses from their GitHub source repositories.
- `reports/figure_sources.json` `oc_vs_pcorr.pdf` source list now lists 26 files (was 25, missing `pythia-1.4b-deduped_main.json`).
- Slope value in `assets/share/README.md` corrected from approximately 0.80 to approximately 0.48 to match the current README alt text.
- Downstream producer scripts (`rag_hallucination.py`, `medqa_selective.py`, `truthfulqa_hallucination.py`) auto-resolve `--peak-layer` from `<slug>_main.json:peak_layer_final` when the flag is not passed. The prior L14 default could silently produce wrong-layer regen output. Fail-fast on missing main JSON.
- New regression test `tests/test_downstream_protocol.py::test_downstream_peak_matches_main` verifies every committed downstream JSON's `peak_layer` matches its corresponding `_main.json:peak_layer_final`. Parametrized across all 9 committed downstream JSONs.
- SQuAD producer `rag_hallucination.py` now saves `per_question: all_results[:50]` matching the committed JSON shape and the MedQA truncation pattern. Prior code path saved the full 1000 records.
- Source-text fields stripped from `per_question` records in all 9 committed downstream JSONs (`question`, `answer`, `gold`/`correct_answer`/`category`, and `gold_letter` for MedQA). Producer scripts updated to no longer emit these fields. Numeric scores, correctness flags, and model-output labels (`pred_letter` A/B/C/D) are preserved. The repository's NOTICE statement ("source-text content is never redistributed") is now categorically accurate for committed result JSONs.
- `analysis/load_results.py` `LLAMA_MODELS` expanded to include Llama-3.2-1B and Llama-3.1-8B; `GEMMA_MODELS` expanded to include Gemma-3-4B. Total registered model count is 26. New scope `absorption_cohort_14` registered in `SCOPES`. The README's confidence-absorption headline is now reproducible via `analysis/selectivity.py` (default scope updated to match). The CLI and Python-API entry points both produce the same headline value.
- `scripts/pythia_1.4b_shuffle.py` reference values (`peak_layer_final`, `hidden_dim`, `partial_corr.mean`) are now loaded from the committed `pythia-1.4b_main.json` with fail-fast on missing file. Removes the prior hardcoded fallback that mirrored paper-cited data and could silently drift.
- Stale numeric values removed from comments and docstrings across `analysis/selectivity.py`, `analysis/load_results.py`, `scripts/pythia_1.4b_shuffle.py`, `scripts/mistral7b_instruct_full_mps.py`, and `scripts/mechanistic_mistral.py`. Specific paper-cited values replaced with qualitative descriptions that point at the canonical sources (`paper_values.json`, committed JSONs).

### Removed

- Docker artifacts and recipes (`Dockerfile`, `.dockerignore`, `.hadolint.yaml`, four `docker-*` justfile recipes, Hadolint CI step). Paper-cited results were produced on the RunPod PyTorch image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Past tags retain the Dockerfile in their Zenodo snapshots.
- Orphaned TriviaQA experiment helpers in `src/selective_prediction.py` and the `selective-prediction` recipe in `justfile`. Utility functions are retained for `tests/test_selective_prediction.py`.

### Notes

- `src/transformer_observe.py` is held as a single 3,767-line module to preserve numerical reproducibility. Refactoring would shift committed result-JSON values and require re-running CUDA experiments to verify bit-equivalence.

## v4.0.0 (2026-05-05, arXiv v2)

Companion code release for arXiv v2 of the paper, retitled
**"Architectural Observability Collapse in Transformers"** (was
"Architecture Determines Observability in Transformers" in arXiv v1).

### New experiments

- **Residualizer-fit split robustness** (`scripts/run_residualizer_split.py`).
  Tests whether the OLS residualizer overfits to the probe-training pool
  by fitting on a disjoint document pool R and applying without refit
  to the probe-training and evaluation pools. Run on four
  representative regimes: GPT-2 124M (healthy anchor), Pythia 1B
  (healthy controlled), Pythia 1.4B (controlled collapse), Llama 3.2
  3B (observational collapse). All four satisfy the pre-specified
  regime-preservation criterion; max same-pool $|\Delta\rho_{\rm partial}| = 0.017$.
  Results in `results/*_residualizer-split.json`.

- **Pythia 1B and 1.4B checkpoint dynamics** at matched hidden
  dimension (d=2048) across 10 checkpoints from 0.5B to 300B tokens.
  Both configurations form the signal at the earliest measured
  checkpoint; training erases it in 1.4B (24L/16H) while 1B (16L/8H)
  recovers. Results in `results/pythia-1b_dynamics.json` and
  `results/pythia-1.4b_dynamics.json`.

- **Gemma 3 1B canonical-protocol rerun** at 350 ex/dim. The earlier
  150 ex/dim measurement was under-saturated; the canonical-protocol
  rerun produces $\rho_{\rm partial} = 0.216$ with normal random-baseline
  behavior and a mid-layer peak at L11. Replaces previous results.

- **Qwen 2.5 32B** added to the cross-family cohort. Within-Qwen
  observability now characterized across a 64x parameter range
  (0.5B through 32B, six base sizes). Result in
  `results/qwen2.5-32b_main.json`.

- **GPT-2 family canonical-protocol reruns** (124M, 355M, 774M, 1.5B)
  at 350 ex/dim, 7-seed protocol with output-controlled residual.
  Replaces the earlier non-uniform per-size protocol used in v3.3.x.
  Results in `results/gpt2-{124m,medium,large,xl}_main.json`.

- **Shuffle-control replication** (`results/gpt-2-124m_shuffle-control.json`).
  10-permutation shuffle of binary targets on GPT-2 124M; backs the
  five-sigma shuffle null cited in the validity argument.

### Reproducibility infrastructure

- **Cross-repo claim provenance** in `reports/`:
  - `reports/paper_values.json`: every paper-cited macro with its
    value, description, section, source files, key paths, formula,
    and named scope. 271 macros total; 118 with full source-file +
    key-path + formula provenance, including all paper headline
    values. Coverage floor enforced by tests; cannot regress without
    explicit consent.
  - `reports/scopes.json`: named cohort definitions
    (`cross_family_14`, `pythia_controlled_9`) mirrored from
    `analysis/load_results.py:SCOPES`, with drift test.
  - `reports/figure_sources.json`: every committed PDF figure mapped
    to its generator script and source JSONs.

- **Formal JSON Schemas** in `schema/` (Draft 2020-12) for every
  result type: main, dynamics, residualizer-split, nonlinear-probe,
  mechanistic, downstream, shuffle-control, bootstrap, width-sweep,
  and legacy. Dispatched by filename pattern in
  `scripts/validate_schemas.py:DISPATCH`. CI validates every
  committed result JSON against its dispatched schema.

- **Manifest verification**: every entry in
  `results/model_revisions.json` is programmatically verified against
  the Hugging Face API. Latest report at
  `results/manifest_verification/2026-05-03.json`. Regeneratable via
  `scripts/verify_manifest_revisions.py`.

- **Croissant 1.1 manifest** at `croissant.json` exposes the
  result-file dataset to ML benchmark indexers
  ([mlcommons.org/croissant](https://mlcommons.org/croissant)).
  Validated against the `mlcroissant` reference spec on every CI run
  via `just check-croissant`.

- **CUDA enforcement** for paper-cited result JSONs. The schema
  enforces `provenance.device == "cuda"` on every committed
  `*_main.json`, `*_dynamics.json`, and `*_residualizer-split.json`.
  MPS is allowed for local development only.

- **Dataset revision pinning** via `results/dataset_revisions.json`.
  Every `load_dataset(...)` call in canonical paper-result CUDA producer
  scripts (the `SCRIPTS_REQUIRING_PREFLIGHT` list in
  `tests/test_script_preflight.py`) passes a pinned revision read from
  the manifest; the test enforces this for that scoped set. Local-dev
  MPS scripts and from-scratch trainer scripts are intentionally exempt
  from preflight (see `SCRIPTS_EXEMPT_FROM_PREFLIGHT`) and are not
  covered by the pinning guarantee.

- **Test suite expansion** in `tests/test_paper_values.py` (97 tests,
  2 module-level skips for missing artifacts):
  - Schema validation against `schema/main.schema.json` and
    `schema/dynamics.schema.json`
  - Provenance integrity (every `source_files` entry resolves; every
    `key_paths` walks; every named scope exists)
  - Direct-read auto-verification (every macro tagged
    `formula="direct read"` matches its JSON cell at the formatted
    precision)
  - Idempotency of all three exporters
  - Scope membership matches `analysis/load_results.py:SCOPES`
  - Figure source-file existence
  - `paper_version` consistency between
    `reports/paper_values.json` and the paper repo's
    `main.tex` `\paperversion`

### Methodology

- **Canonical 350 ex/dim protocol** standardized across all
  paper-cited result JSONs. Cross-family scope is 14 models in 6
  families.

- **20-seed statistical hardening** on GPT-2 124M (layer 11):
  $\rho_{\rm partial} = 0.282 \pm 0.001$, 95% CI [0.282, 0.283],
  per-seed range [0.279, 0.284], seed agreement 0.993.

- **Three-pool protocol** (R/T/V) for the residualizer-fit split:
  disjoint WikiText-103 documents at the same hidden-dim token
  budget as the canonical protocol.

### Removed

- **Mechanistic ablation appendix (Appendix G)** from the paper:
  directional ablation, mean-ablation patching, and layer formation
  of the output-independent component were exploratory and not
  load-bearing for any Paper 1 claim. Mechanism work is the scope of
  Paper 2 (`nn-mechanistic`). The corresponding result JSONs
  (`*_mechanistic.json`) and `transformer_observe.json` mechanistic
  blocks remain in this repository as historical artifacts; they are
  no longer cited by the paper.

- **Layer-1 ablation pattern subsection (A.12)** from the
  Methodology hardening appendix: the Llama 3.2 1B vs 3B and Mistral
  7B sign-reversal observation was exploratory and removed for the
  same reason as Appendix G.

### Documentation

- **Three-path verification structure** in `README.md`:
  structured claim provenance, targeted CLI verification, and full
  pipeline, with a worked example walking `confabsorbmean` from
  `paper_values.json` through the 14 source JSONs to recompute.

### Numerical updates

- Confidence absorption recomputed under canonical protocol: $60.3\%
  \pm 7.0\%$ across 14 models in 6 families (was 58.2% in v3.3.x;
  drift due to the cohort change).

### v3.3.0 (arXiv v1) baseline

The arXiv v1 release. Tagged `arxiv-v1`. Every change above is
relative to that baseline.

---

For full provenance of every paper-cited number, see
`reports/paper_values.json` (regenerated with each release). For
schema-validated reproducibility of every result file, see `schema/`.
