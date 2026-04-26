# Paper figures

_Updated 2026-04-25 for repo v3.1.0._

PNG copies of the seven figures used in the paper and referenced from the repo README. Captions are condensed from the paper PDF; see the paper repo for the full typeset versions.

## Figures

### `within_family_cliff.png`

Within-family observability collapse in two training recipes. Left: Llama 3.2/3.1 (1B preserves signal at +0.28; 3B and 8B stay flat near +0.05-0.10). Right: Pythia configurations at (24 layers, 16 heads) collapse while the other six sizes peak between +0.20 and +0.38. Paper introduction, Figure 1.

### `cross_family_scaling.png`

Peak-layer partial correlation across six families at their default layer. Llama drops from +0.286 at 1B to +0.091 at 3B; the other five families stay flat around +0.29. Shaded band marks the detection threshold. Paper architecture section.

### `control_sensitivity_waterfall.png`

Control sensitivity on GPT-2 124M. Cumulative waterfall of raw Spearman plus softmax, norm, and logit-entropy controls, with an independent nonlinear MLP control as the fifth bar. Paper signal section.

### `oc_vs_pcorr.png`

Output-controlled residual versus confidence-controlled partial correlation across 22 models (13 cross-family plus 9 Pythia). Healthy configurations fall on a linear trend (slope 0.88); five collapsed configurations (Llama 3B/8B and three Pythia 24L/16H replications) land near the origin on both axes. Paper Pythia section.

### `pythia_layers.png`

Pythia layer profiles under held-recipe training. Six healthy configurations peak at mid-to-late depth; three (24L, 16H) configurations stay flat across all layers. No layer choice rescues the collapse. Paper Pythia section.

### `downstream_three_model.png`

Exclusive catch rates across three tasks (RAG, MedQA, TruthfulQA) and three production instruct models (Qwen 7B, Phi-3 Mini, Mistral 7B) at 5, 10, and 20 percent flag rates. At 20 percent flag rate, seven of nine cells land in the 11-15 percent ceiling band carried over from the WikiText flagging table. Paper architecture section.

### `exdim_sensitivity.png`

Ex/dim sensitivity at Qwen 2.5 0.5B across seven token budgets (150-1000 ex/dim) at the peak layer. Signal stays below the detection threshold between 150 and 450 ex/dim and rises sharply between 450 and 600. Paper token-budget appendix.
