# Analysis scripts

_Updated 2026-04-25 for repo v3.1.0._

CPU-only statistical analysis. All scripts read from `results/` and import model scope from `load_results.py`.

## Running

```bash
uv run python analysis/run_all.py          # all analysis scripts
uv run python analysis/load_results.py     # validate results JSON schema
uv run python analysis/load_results.py --strict  # also check provenance fields
```

## Scripts

| Script | Purpose | Output |
|---|---|---|
| `load_results.py` | Model scope definition and JSON schema validation | Imported by all others |
| `run_all.py` | Run all analysis scripts in sequence | Combined summary |
| `permutation_test.py` | Monte Carlo permutation test for family effect | F-statistic, p-value |
| `meta_regression.py` | Mixed-effects model + three-level variance decomposition | Coefficients, variance % |
| `ancova_family.py` | Supplementary ANCOVA (anticonservative, mixed-effects is primary) | F-stats, p-values |
| `exclusive_catch_rates.py` | Multi-rate exclusive catch analysis across families | Catch rates at 5/10/20/30% |
| `selectivity.py` | Random head baselines and control gap analysis | Per-model selectivity |
| `funnel_plot.py` | Publication bias diagnostic | Funnel plot data |
| `pearson_vs_spearman.py` | Pearson vs Spearman correlation comparison | Correlation table |
| `loocv_scaling.py` | Leave-one-out cross-validation for scaling | Prediction errors |
| `held_out_split.py` | Cross-validated partial-Spearman from per-token dumps | In-sample vs held-out delta per model |

## Adding a new model

1. Run the experiment:
   ```bash
   python scripts/run_model.py --model org/Model-Name --output modelname_results.json
   ```

2. Validate the output:
   ```bash
   just validate-results
   ```
   The schema checks for required fields: `model`, `partial_corr.mean`, `partial_corr.per_seed` (minimum 3 seeds; v3 protocol uses 7), `output_controlled.mean`, `peak_layer_final` (or `peak_layer`), `peak_layer_frac`, `seed_agreement`, `baselines`. Run with `--strict` to also check provenance fields.

3. Add to analysis scope in `load_results.py`:
   - Add an entry to the appropriate family list (e.g., `QWEN_MODELS`) or create a new family list
   - The entry is a tuple: `("filename.json", params_in_billions, "Display-Label")`

## Required JSON schema

Every full-protocol results file must contain:

```json
{
  "model": "org/Model-Name",
  "n_params_b": 7.6,
  "n_layers": 32,
  "hidden_dim": 3584,
  "provenance": {
    "model_revision": "abc123...",
    "script": "scripts/run_model.py",
    "timestamp": "2026-04-15T12:00:00+00:00",
    "device": "cuda",
    "torch_version": "2.2.0"
  },
  "protocol": {
    "layer_select_seed": 42,
    "eval_seeds": [43, 44, 45, 46, 47, 48, 49],
    "target_ex_per_dim": 350,
    "batch_size": 4
  },
  "peak_layer_final": 17,
  "peak_layer_frac": 0.61,
  "partial_corr": {
    "mean": 0.255,
    "std": 0.019,
    "per_seed": [0.241, 0.258, "..."]
  },
  "output_controlled": { "mean": 0.137 },
  "seed_agreement": { "mean": 0.964 },
  "baselines": { "random_head": 0.014 },
  "flagging_6a": { "..." : "..." },
  "control_sensitivity": { "..." : "..." }
}
```

Fields marked `provenance` are written automatically by `run_model.py`. Older results files may lack them; `--strict` validation flags this.
