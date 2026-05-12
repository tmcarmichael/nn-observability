# MODELS.md

## Scope

This file documents the public transformer models evaluated in the paper [Architecture Determines Observability of Transformers](https://arxiv.org/abs/2604.24801) and their role in this repository. The repo is a *consumer* of upstream public models, not a creator. Each model is loaded from Hugging Face at a pinned commit hash; this file states the family creator, license, role in the paper, family-level observability finding, and limitations that bound the paper's claims. Architecture details, training data, and creator-side responsible-AI documentation live with the upstream creators and are linked from each entry.

## Provenance

This card documents the model state at `nn-observability` v5.1.0 / paper-adot v5.0.0 (Zenodo concept DOI [10.5281/zenodo.19435674](https://doi.org/10.5281/zenodo.19435674)).

Three artifacts already pin and verify the model references:

- `results/model_revisions.json` carries the Hugging Face model id, source URL, and pinned commit hash for every evaluated model.
- `results/manifest_verification/` carries dated reports of the model-revision verification pass against the Hugging Face API. Every entry is verified to exist at the pinned commit. The latest report at the v5.1.0 cutoff is dated 2026-05-12.
- `DATA.md` is the companion documentation for the seven datasets plus The Pile (license, role in paper, subset and transforms, known limitations).

For strict reproduction, `HF_HOME=$(mktemp -d)` isolates the run from any local cache so the pinned revision is the only one used.

To spot-check any claim in this file, see the README's [three verification paths](README.md#verify-a-paper-claim) which walk from a paper-cited number to its source JSON and producer script.

## Model Card alignment

This file follows Mitchell et al. (2019) *Model Cards for Model Reporting* in spirit. Because this repo is a consumer of upstream public models, the Mitchell sections map onto the per-family entries below as follows:

- **Model Details** (creator, release date, version, license): covered by per-family "License", "Creator", and "Models pinned" fields.
- **Intended Use** and **Out-of-Scope Use Cases**: covered by "Role in this paper" and "Limitations and out-of-scope uses".
- **Training Data**: pointed to via "Training corpus", with details documented by the upstream creators.
- **Evaluation Data, Metrics, Quantitative Analyses**: covered by "Family-level finding" and the result JSONs at `results/<model>_main.json`; the paper itself carries the metric definitions and statistical analyses. Mitchell's "Quantitative Analyses" section expects disaggregated subgroup performance; the evaluation corpora used in this paper (WikiText-103, C4, SQuAD v2, MedQA-USMLE, TruthfulQA, OpenWebText, CodeSearchNet) carry no demographic subgroup axis, so the disaggregation reported here is by model family and architecture configuration rather than by population subgroup.
- **Ethical Considerations** and **Caveats and Recommendations**: covered jointly by per-family "Limitations and out-of-scope uses" and "Cross-cutting limitations and scope".

The per-family headings below are kept in their existing order rather than renamed to the Mitchell vocabulary, both for backward-compatible cross-references from the paper and to keep the consumer-vs-creator distinction explicit.

## Models by family

Seven families, 33 model checkpoints in total, are pinned in `model_revisions.json`. The cross-family aggregate scope (`cross_family_14` in `analysis/load_results.py`) covers 14 base models across 6 families; the Pythia controlled suite (`pythia_controlled_9`) covers 8 base models plus the deduped 1.4B variant. Instruct variants are evaluated separately and not part of the aggregate scopes. Headline numbers per model live in `results/<model>_main.json` and in `paper-adot/data_macros.sty`.

### GPT-2

OpenAI's transformer family from 2019, four sizes from 124M to 1.5B parameters.

- **License.** MIT per the [OpenAI Community HF cards](https://huggingface.co/openai-community/gpt2).
- **Creator.** OpenAI (released as community-maintained mirrors under `openai-community/`).
- **Training corpus.** WebText (Reddit-linked web pages with karma threshold). Not loaded by this repo.
- **Role in this paper.** Cross-family scaling baseline. The 124M variant is the paper's hardening anchor (20-seed protocol at layer 11). The four sizes evaluate scaling within a single family at fixed pre-training recipe.
- **Models pinned.** `openai-community/gpt2` (124M, 12L, hidden 768), `gpt2-medium` (355M, 24L, 1024), `gpt2-large` (774M, 36L, 1280), `gpt2-xl` (1.56B, 48L, 1600). Result files `results/gpt2-{124m,medium,large,xl}_main.json`.
- **Family-level finding.** All four sizes preserve observability. Pcorr 0.288 to 0.296 across the four sizes; OC residual 0.084 to 0.151. Output-controlled residual decreases at 1.56B even as pcorr stays high (the discard-rate trend reported in the GPT-2 scaling section).
- **Limitations and out-of-scope uses.** GPT-2 was trained on WebText, a relatively small and English-skewed web corpus by 2026 standards. The scaling claim within GPT-2 is bounded to this single training distribution at four sizes. WebText predates instruction tuning and modern alignment; GPT-2 does not represent the post-2023 deployed-model regime.
- **Citation.** Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners. OpenAI technical report.

### Pythia

EleutherAI's controlled training suite, all sizes pretrained on identical data (The Pile) with matched optimizer, tokenizer, and training schedule. The controlled suite is the paper's headline causal evidence: when data is held constant, architecture configuration is the only varying factor.

- **License.** Apache 2.0 per the [EleutherAI HF cards](https://huggingface.co/EleutherAI/pythia-70m).
- **Creator.** EleutherAI.
- **Training corpus.** The Pile (and the deduplicated variant for `pythia-1.4b-deduped`). Documented separately in DATA.md as upstream-only.
- **Role in this paper.** Within-recipe controlled causation. Eight base sizes plus one deduped variant give nine architecture configurations under identical training. The (24-layer, 16-head) class collapses across three replications (`pythia-410m`, `pythia-1.4b`, `pythia-1.4b-deduped`); the other six configurations preserve observability.
- **Models pinned.** `pythia-70m`, `pythia-160m`, `pythia-410m`, `pythia-1b`, `pythia-1.4b`, `pythia-1.4b-deduped`, `pythia-2.8b`, `pythia-6.9b`, `pythia-12b`. Result files `results/pythia-*_main.json`. Checkpoint dynamics for 1B and 1.4B in `pythia-1b_dynamics.json` and `pythia-1.4b_dynamics.json` (10 checkpoints from step 256 to step 143000).
- **Family-level finding.** Three of nine configurations collapse to pcorr ~0.10; the other six preserve a healthy band from 0.21 to 0.38. The collapse class shares (24L, 16H) architecture across a 3.5x parameter gap and two Pile variants. Checkpoint dynamics show the collapse is emergent during training: both 1B and 1.4B form the signal at the earliest measured checkpoint (step 256), but 1.4B erases it through training while 1B preserves it.
- **Limitations and out-of-scope uses.** Pythia is a research suite, not a deployed model family. The collapse class within Pythia identifies an architectural risk factor; whether the same factor predicts collapse outside of Pile-trained models is an open question addressed observationally via the cross-family evidence below. Pythia-deduped is not a separate family; it tests data-side variance with an otherwise-identical training recipe.
- **Citation.** Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., and van der Wal, O. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. arXiv:2304.01373.

### Qwen 2.5

Alibaba's Qwen 2.5 family, evaluated at six base sizes (0.5B through 32B) and five instruction-tuned variants, the broadest in-family scaling in the paper.

- **License.** Mixed within Qwen 2.5. Sizes 0.5B, 1.5B, 7B, 14B, and 32B base plus their Instruct variants are Apache 2.0 per the [Qwen HF cards](https://huggingface.co/Qwen/Qwen2.5-7B). The 3B base and 3B Instruct variants are released under the [Qwen Research License](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE) (custom, non-commercial research use); both are evaluated in this paper. The 72B variant is on a separate Qwen License (custom, commercial threshold) and is not evaluated here. Per-checkpoint license attribution is recorded in `results/model_revisions.json`.
- **Creator.** Alibaba Cloud (Qwen team).
- **Training corpus.** Proprietary. Qwen reports include English, Chinese, and code; details vary by Qwen technical report version.
- **Role in this paper.** Cross-family observational evidence and within-family scaling at six sizes (0.5B to 32B) plus five instruct variants. Qwen is the family with the highest observed observability values; instruct variants are documented to compare RLHF effects on observability.
- **Models pinned.** Base: `Qwen2.5-{0.5B, 1.5B, 3B, 7B, 14B, 32B}`. Instruct: `Qwen2.5-{0.5B, 1.5B, 3B, 7B, 14B}-Instruct`. Result files `results/qwen2.5-*_main.json`.
- **Family-level finding.** Within-Qwen, observability persists from 0.5B through 32B (pcorr 0.214 to 0.291). Instruct variants score similar or slightly higher than their base counterparts (e.g., 7B Instruct pcorr 0.291 vs 7B base 0.255). Qwen and Llama differ by 2.9x at matched 3B scale (Qwen 3B pcorr 0.263; Llama 3B pcorr 0.091) with non-overlapping seed distributions.
- **Limitations and out-of-scope uses.** Qwen training data is not public, so cross-recipe causal claims through Qwen are observational. The within-family Qwen scaling result holds within Alibaba's training pipeline; extension to other large-model training pipelines is not established.
- **Citation.** Qwen Team. (2024). Qwen2.5 Technical Report. arXiv:2412.15115.

### Llama 3

Meta's Llama 3.1 and 3.2 families. The paper evaluates 1B, 3B, and 8B base variants plus the 1B Instruct variant.

- **License.** Llama 3.2 Community License (custom, commercial) per the [Meta HF cards](https://huggingface.co/meta-llama/Llama-3.2-1B). Not Apache or MIT. Use governed by Meta's [Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy). Commercial-scale users (>700M MAU) require separate licensing from Meta.
- **Creator.** Meta.
- **Training corpus.** Proprietary. Llama 3 technical report mentions ~15T tokens of largely-English text data; precise composition is not public.
- **Role in this paper.** Cross-family observational evidence at three matched sizes against Qwen, Pythia, Mistral, Gemma, and Phi. The Llama collapse cliff (3B and 8B at pcorr ~0.09 vs 1B at pcorr 0.286) is the headline cross-family contrast. The 1B Instruct variant tests RLHF invariance.
- **Models pinned.** `Llama-3.2-1B`, `Llama-3.2-3B`, `Llama-3.1-8B`, `Llama-3.2-1B-Instruct`. Result files `results/llama-3.{2-1b,2-3b,1-8b,2-1b-instruct}_main.json`.
- **Family-level finding.** 1B preserves observability (pcorr 0.286 at peak L13). 3B and 8B collapse (pcorr 0.091 at L0; pcorr 0.093 at L1). The collapse persists across both Llama 3.2 (3B) and Llama 3.1 (8B), suggesting the collapse is recipe-driven rather than version-specific. 1B Instruct stays close to 1B base (pcorr 0.285), indicating RLHF does not erase observability where the base model preserves it.
- **Limitations and out-of-scope uses.** Llama license restrictions affect downstream redistribution and commercial-scale use; users should consult the Llama 3.2 Community License directly. The cross-family contrast against Llama is observational because Meta does not release training-data composition.
- **Required attribution.** Built with Llama. This repository persists statistical summaries derived from the frozen activations and outputs of `Llama-3.2-1B`, `Llama-3.2-3B`, `Llama-3.2-1B-Instruct`, and `Llama-3.1-8B`. Downstream redistributors of artifacts derived from these models include the "Built with Llama" attribution as required by the Llama Community License (Section 1(b)(i)) and remain bound by Meta's Acceptable Use Policy. The repository itself does not redistribute Llama weights. The corresponding NOTICE file at the repo root carries this attribution.
- **Citation.** Grattafiori, A., Dubey, A., Jauhri, A., et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.

### Mistral 7B v0.3

Mistral AI's 7B model in base and instruct variants.

- **License.** Apache 2.0 per the [Mistral HF cards](https://huggingface.co/mistralai/Mistral-7B-v0.3).
- **Creator.** Mistral AI.
- **Training corpus.** Proprietary. Mistral 7B paper mentions a mix of public web data and synthetic generations.
- **Role in this paper.** Cross-family observational evidence at 7B scale, instruct comparison. Mistral 7B preserves observability where Llama 3.1 8B collapses despite similar broad architecture (32 layers, 4096 hidden), establishing that architecture alone does not predict collapse.
- **Models pinned.** `Mistral-7B-v0.3`, `Mistral-7B-Instruct-v0.3`. Result files `results/mistral-7b-{v0.3, instruct-v0.3}_main.json`. Mistral 7B Instruct is also evaluated on the three downstream tasks (`mistral-7b-instruct-v0.3_{medqa, squad-rag, truthfulqa}.json`).
- **Family-level finding.** Mistral 7B base preserves observability (pcorr 0.313 at peak L22, OC residual 0.156). The instruct variant scores slightly higher (pcorr 0.339 at L29, OC residual 0.178). Both substantially exceed the Llama 3.1 8B collapse at similar parameter scale.
- **Limitations and out-of-scope uses.** Mistral training data is proprietary; cross-recipe causal claims through Mistral are observational. The Mistral-vs-Llama-at-matched-architecture contrast is the load-bearing finding from this family.
- **Citation.** Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L. R., Lachaux, M.-A., Stock, P., Le Scao, T., Lavril, T., Wang, T., Lacroix, T., and El Sayed, W. (2023). Mistral 7B. arXiv:2310.06825.

### Gemma 3

Google's Gemma 3 family, evaluated at 1B and 4B pretrained sizes.

- **License.** Gemma license (custom) per the [Google HF cards](https://huggingface.co/google/gemma-3-1b-pt). Use governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- **Creator.** Google.
- **Training corpus.** Proprietary. Gemma 3 technical report mentions multilingual web text; details vary by Gemma version.
- **Role in this paper.** Cross-family observational evidence, mid-2026 model recency anchor. Both sizes evaluated under the canonical 350 ex/dim, 7-seed protocol.
- **Models pinned.** `gemma-3-1b-pt`, `gemma-3-4b-pt`. Result files `results/gemma-3-{1b, 4b}_main.json`.
- **Family-level finding.** Gemma 3 1B preserves observability with pcorr 0.216 at L11 (mid-layer peak), OC residual 0.161. Gemma 3 4B is at the lower edge of the healthy band (pcorr 0.191 at L6). Both are evaluated under the canonical 350 ex/dim, 7-seed protocol.
- **Limitations and out-of-scope uses.** Gemma license restricts redistribution and certain commercial uses; users should consult the Gemma Terms of Use directly. Gemma training data is proprietary; cross-recipe claims through Gemma are observational.
- **Citation.** Gemma Team. (2025). Gemma 3 Technical Report. Google DeepMind.

### Phi-3 Mini

Microsoft's Phi-3 Mini Instruct, included as a specialist on the small-instruct end of the family axis.

- **License.** MIT per the [Microsoft HF card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).
- **Creator.** Microsoft.
- **Training corpus.** Proprietary curated mix. The Phi series uses heavily filtered "textbook-quality" data per Microsoft's reports.
- **Role in this paper.** Cross-family observational evidence at the small-instruct end. Phi-3 Mini is evaluated on the cross-family observability protocol and on the three downstream tasks.
- **Models pinned.** `Phi-3-mini-4k-instruct`. Result files `results/phi-3-mini_main.json` and `phi-3-mini_{medqa, squad-rag, truthfulqa}.json`.
- **Family-level finding.** Phi-3 Mini Instruct preserves observability (pcorr 0.265 at L25). Downstream observer transfer holds for MedQA and SQuAD; TruthfulQA is the boundary case (AUC 0.556).
- **Limitations and out-of-scope uses.** The Phi training-data curation is proprietary and atypical (heavily filtered textbook-style data). Generalizing Phi-3 results to broader-distribution models is not warranted; Phi-3 represents one specific data-curation regime.
- **Citation.** Abdin, M., Aneja, J., Awadalla, H., et al. (2024). Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arXiv:2404.14219.

## Cross-cutting limitations and scope

All 33 evaluated checkpoints come from public, upstream-creator-maintained Hugging Face repositories. This repository is a *consumer* of these models, not a creator: no models are trained, fine-tuned, or quantized here, and no human-subjects data is collected by this repo. Models are loaded at evaluation time from `load_pretrained` calls pinned via `results/model_revisions.json`.

The evaluated models span English-only and English-dominant training mixes. The paper's observability and cross-family claims are scoped to this English-text setting; multilingual generalization is not measured. Within each family, training-data composition is upstream-creator-determined and largely proprietary outside Pythia. This is why the within-Pythia results carry causal weight (data held constant, architecture varied) while cross-family results are framed observationally.

The paper's evaluations are scientific benchmarks for a frozen-probe research method. They do not constitute endorsement for production or regulated deployment of activation monitoring on the underlying models. Architecture selection on the basis of observability is appropriate as a research input; downstream production decisions weigh capability, latency, cost, calibration, and safety in addition to observability.

## License compatibility for released artifacts

This repository is licensed under the MIT License. Released artifacts are: source code (MIT), result JSONs (MIT), and the human-readable documentation in this file (MIT). Model weights from any upstream creator are never redistributed; the persisted artifacts are statistical summaries derived from frozen activations (per-token loss values, observer scores, partial correlations, exclusive-catch counts).

| Family | License | Constraints inherited by derivatives |
|---|---|---|
| GPT-2 | MIT | Permissive. |
| Pythia | Apache 2.0 | Permissive; attribution required. |
| Qwen 2.5 | Mixed: Apache 2.0 for 0.5B / 1.5B / 7B / 14B / 32B (base + Instruct); Qwen Research License for 3B base and 3B Instruct | Apache 2.0 entries are permissive with attribution. The 3B base and 3B Instruct entries are under the Qwen Research License, which restricts use to non-commercial research; downstream redistributors of derivatives that depend on these two checkpoints inherit those terms. The 72B variant is on a separate Qwen License and is not evaluated here. |
| Llama 3 | Llama 3.2 Community License (custom, commercial) | Restrictive. Use governed by Meta's Acceptable Use Policy; commercial-scale users (>700M MAU) require separate licensing. Derivative artifacts inherit Llama license terms. |
| Mistral 7B v0.3 | Apache 2.0 | Permissive; attribution required. |
| Gemma 3 | Gemma license (custom) | Restrictive. Use governed by the Gemma Terms of Use, including a Prohibited Use Policy. Derivative artifacts inherit Gemma terms. |
| Phi-3 Mini | MIT | Permissive. |

Per-token loss values, observer scores, and other statistical summaries derived from frozen model activations are not redistributed model weights. Llama and Gemma derivatives are governed by their respective custom licenses; the Qwen 2.5 3B base and 3B Instruct derivatives are governed by the Qwen Research License (non-commercial). Consumers redistributing derived artifacts that depend on these checkpoints consult the upstream license directly. The other 25 evaluated checkpoints ship under permissive licenses (MIT or Apache 2.0).

## Distribution and maintenance

The 33 checkpoints are distributed by their upstream creators via Hugging Face. This repository does not redistribute any model weights; consumed copies are pulled at evaluation time from `from_pretrained` calls pinned via `results/model_revisions.json`. Maintenance of the upstream models, including future updates, errata, and deprecations, is the responsibility of the upstream creators. The pinned commit hashes in this repository are immutable references to the model state at the paper's evaluation cutoff; subsequent upstream changes are not reflected here.

The MLCommons Croissant 1.1 metadata descriptor in `croissant.json` carries the machine-readable record-set inventory and cross-references the model and dataset manifests; see `croissant.json` for the structured form alongside this human-readable card.

## Updating this file

`results/model_revisions.json` is the source of truth for pinned commits; this file is the human-readable companion. When a model entry in `model_revisions.json` changes, update the matching family section here in the same change.
