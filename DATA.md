# DATA.md

## Scope

This file documents the public datasets used in the paper [Architecture Determines Observability of Transformers](https://arxiv.org/abs/2604.24801) and their role in this repository. The repo is a *consumer* of upstream public datasets, not a creator. Each dataset is pulled from Hugging Face at a pinned commit hash; this file states how each one is used in the paper, the license attribution, and the limitations that bound the paper's claims. Construction details, raw collection methods, and creator-side responsible-AI documentation live with the upstream creators and are linked from each entry.

## Provenance

This card documents the data state at `nn-observability` v5.0.0 / paper-adot v5.0.0 (Zenodo concept DOI [10.5281/zenodo.19435674](https://doi.org/10.5281/zenodo.19435674)). It regenerates alongside the dataset manifest in future versions; if you are reading a downstream copy, the in-tree version above is authoritative for the `nn-observability` tag you have checked out.

Four artifacts already pin and verify the dataset references:

- `results/dataset_revisions.json` carries the Hugging Face dataset id, config name, source URL, pinned commit hash, and license attribution for every paper-cited evaluation dataset.
- `croissant.json` is the MLCommons Croissant 1.1 metadata descriptor for the result-file inventory; it cross-references the dataset manifest and conforms to `http://mlcommons.org/croissant/1.1`.
- `results/manifest_verification/` carries dated reports of the model-revision verification pass against the Hugging Face API. The dataset side is pinned the same way; canonical paper-result CUDA producer scripts read the manifest and pass `revision=DATASET_REVISIONS[id]["commit"]` to `load_dataset`. Pinning is enforced for the paper-result producer set by `tests/test_script_preflight.py` (the `SCRIPTS_REQUIRING_PREFLIGHT` list); local-dev MPS scripts and from-scratch trainer scripts exempt from preflight are not covered by that test.
- `MODELS.md` is the companion documentation for the 33 evaluated transformer checkpoints (license, family, role in paper, observability findings).

For strict reproduction, `HF_HOME=$(mktemp -d)` isolates the run from any local cache so the pinned revision is the only one used.

To spot-check any claim in this file, see the README's [three verification paths](README.md#verify-a-paper-claim) which walk from a paper-cited number to its source JSON and producer script.

## Datasheet alignment

This file follows Gebru et al. (2021) *Datasheets for Datasets* in spirit. Because this repo is a consumer of upstream public datasets, the seven Gebru sections map onto the per-dataset entries below as follows:

- **Motivation** and **Collection process**: belong to the upstream creator. Each per-dataset entry links the upstream HF card and the source publication; consult those for the canonical answers.
- **Composition**: covered inline within "Subset and transforms" (n_examples, eval split, token budget, evaluated-model accuracy where applicable).
- **Preprocessing/cleaning/labeling**: covered by "Subset and transforms" (tokenization, truncation, sampling).
- **Uses (this paper)**: covered by "Role in this paper" plus "Limitations and out-of-scope uses."
- **Distribution** and **Maintenance**: covered by the per-dataset license note and the consolidated "Distribution and maintenance" section at the end of this file.

The per-dataset headings below are kept in their existing order rather than renamed to the Gebru vocabulary, both for backward-compatible cross-references from the paper and to keep the consumer-vs-creator distinction explicit.

## Datasets

Seven datasets are pinned in `dataset_revisions.json`. WikiText-103 is the probe training corpus. C4, OpenWebText, and CodeSearchNet are cross-domain transfer evaluations (the latter two appear in `appendix/appendix_cross_domain.tex`). SQuAD v2, MedQA-USMLE, and TruthfulQA are the three downstream tasks. All seven entries are pinned end-to-end across the canonical paper-result CUDA producer scripts that generated the committed result JSONs: those scripts pass the recorded `commit` via the `revision=` argument to `load_dataset`, and `tests/test_script_preflight.py` enforces this for the `SCRIPTS_REQUIRING_PREFLIGHT` list. Local-dev MPS scripts and from-scratch trainer scripts are intentionally exempt from that check (see the script's `SCRIPTS_EXEMPT_FROM_PREFLIGHT` list); they retain working-but-not-paper-cited dataset loads. All seven datasets are English. None are constructed by this paper or this repo; we evaluate on splits and subsets as documented below.

### WikiText-103

`Salesforce/wikitext` config `wikitext-103-raw-v1` revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

- **License.** Dual CC BY-SA 3.0 and GFDL per the [Salesforce HF card](https://huggingface.co/datasets/Salesforce/wikitext) (which declares `cc-by-sa-3.0` and `gfdl` in its YAML license tags). The source content was scraped from Wikipedia featured and good articles prior to Wikipedia's June 2023 migration from CC BY-SA 3.0 to CC BY-SA 4.0, so WikiText-103 inherits the historical 3.0 + GFDL dual license. Verbatim text from Wikipedia featured and good articles.
- **Role in this paper.** Probe training corpus and language-modeling-domain reference. Every cross-family probe in the paper is fit on WikiText activations. The cross-domain transfer claim ("the probe catches errors confidence misses on MedQA, SQuAD, and TruthfulQA without retraining") rests on this single training distribution.
- **Subset and transforms.** `train` split is sampled at 350 examples per probe dimension across 7 evaluation seeds (the canonical protocol). Default cap is 12,000 documents on the training split (a small slice of the full WikiText-103 training corpus); validation and test loads are unlimited (see the `--max-docs` argument default in `scripts/run_model.py`). `validation` is the held-out fit pool; `test` is used for the cross-domain transfer baseline. WikiText rows are grouped into documents using blank-line separators; each document is tokenized with the target model's tokenizer and truncated to 512 tokens.
- **Composition.** Upstream WikiText-103 is roughly 1.8M training documents, ~3,760 validation documents, ~4,358 test documents. The per-experiment probe-fit budget is 350 ex/dim × hidden_dim × 7 seeds: for GPT-2 124M (hidden_dim 768) that is ~268,800 tokens per seed-fit; for Qwen 2.5 7B (hidden_dim 3,584) that is ~1.25M tokens per seed-fit. Per-token loss values are persisted in each `results/<model>_main.json` under `partial_corr.per_seed`; the result JSONs are the source of truth for distributional summaries.
- **Limitations and out-of-scope uses.** WikiText is encyclopedic English with formal register, no dialog, no code, no non-English content. The downstream transfer claim is bounded by this. Performance on conversational, code-heavy, multilingual, or untested specialist domains is not measured here. The probe's success on MedQA and SQuAD demonstrates *some* generality, but extension beyond the tested settings is observational, not established.
- **Citation.** Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). Pointer Sentinel Mixture Models. arXiv:1609.07843.

### C4 (English)

`allenai/c4` config `en` revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.

- **License.** ODC-BY 1.0 per the [AllenAI HF card](https://huggingface.co/datasets/allenai/c4), with the additional binding that users comply with the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use/) because the dataset is sourced from Common Crawl. Derived from Common Crawl with the C4 cleaning rules from Raffel et al. (2020).
- **Role in this paper.** Cross-domain transfer evaluation. Probe is trained on WikiText, scored on C4, used to test whether observability degrades when the evaluation distribution shifts away from the probe training corpus.
- **Subset and transforms.** Loaded by `scripts/run_model.py` (the `allenai/c4` `load_dataset` calls in the C4-transfer block). Same tokenization and context-window construction as WikiText. Sample sizes are matched to the WikiText evaluation budget; this is a tiny subset of the C4 English corpus, which exceeds 300 million documents at upstream scale.
- **Composition.** Per-experiment subset matches the WikiText probe-fit budget (350 ex/dim × hidden_dim × 7 seeds). Upstream C4 English is ~365M documents; the slice loaded here is approximately 0.0001% of that and is not retained.
- **Limitations and out-of-scope uses.** C4 is web-scraped English with heavy genre and quality variance. The cross-domain delta we report measures probe stability under a within-language distribution shift; it does not generalize to multilingual, low-resource, or non-text modalities. C4 has documented quality and bias issues from the upstream Common Crawl source; the cross-domain finding does not depend on C4 being clean, and the C4 result is not a generic "real-world text" benchmark.
- **Citation.** Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., and Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. arXiv:1910.10683.

### OpenWebText

`Skylion007/openwebtext` revision `b4325f019c648b1641a1784748667e8b74e5e064`. Legacy alias `openwebtext` is a redirect to this canonical id and resolves to the same SHA.

- **License.** CC0 1.0 Universal per the [Skylion007 HF card](https://huggingface.co/datasets/Skylion007/openwebtext). Open replication of OpenAI's WebText corpus, scraped from URLs cited on Reddit submissions with karma greater than two.
- **Role in this paper.** Cross-domain transfer evaluation cited in `appendix/appendix_cross_domain.tex`. The probe is fit on WikiText activations and scored on OpenWebText, testing whether the WikiText-trained signal generalizes to a broader web register. The appendix reports the OpenWebText transfer as weak (`\owttransfer` in `data_macros.sty`), consistent with the cross-domain interpretation.
- **Subset and transforms.** Loaded by `src/transformer_observe.py` (the `load_openwebtext` function uses the legacy `openwebtext` id, streaming) and by `scripts/controlled_depth_width.py` and `scripts/controlled_training.py` (the `Skylion007/openwebtext` `load_dataset` calls, streaming). Same per-document tokenize-and-truncate-to-512 transform as WikiText. Producer call sites use mixed dataset ids; both resolve to the same content.
- **Composition.** Upstream OpenWebText is approximately 8 million documents (around 38 GB of plaintext). The per-experiment subset is bounded by the controlled-training token budget (`controlled_training.py` requests roughly `total_tokens + 500000` from the streaming loader); the slice loaded here is a small fraction of upstream and is not retained.
- **Limitations and out-of-scope uses.** OpenWebText is a curated web corpus from a 2019-era Reddit-link snapshot; it is not a full Common Crawl. The cross-domain transfer claim measures probe stability under a register shift toward general web text but does not generalize to multilingual, low-resource, or non-text modalities.
- **Citation.** Gokaslan, A. and Cohen, V. (2019). OpenWebText Corpus. <http://Skylion007.github.io/OpenWebTextCorpus>.

### CodeSearchNet (Python)

`code_search_net` config `python` revision `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555`.

- **License.** Mixed; the [HF card](https://huggingface.co/datasets/code_search_net) declares `license: other` because the dataset content has heterogeneous origins. The MIT license at the [upstream GitHub repository](https://github.com/github/CodeSearchNet) covers the curation tool and dataset wrapper; constituent function bodies retain the licenses of their original GitHub source repositories (predominantly MIT, Apache 2.0, BSD, with smaller fractions of other licenses). Function-level open-source code mined from GitHub, paired with natural-language docstrings. Six languages upstream; this repo uses the Python config only. This repo does not redistribute source-text content; persisted artifacts are statistical summaries of model behavior on the corpus.
- **Role in this paper.** Cross-domain transfer evaluation cited in `appendix/appendix_cross_domain.tex`. The probe is fit on WikiText activations and scored on Python source code; the appendix reports core-pcorr-comparable transfer (`\corepcorr`) on this domain, consistent with the protocol-generalization interpretation that signal survives across natural-language and code domains.
- **Subset and transforms.** Loaded by `src/transformer_observe.py` (the `code_search_net` `load_dataset` call: `python` config, streaming, `trust_remote_code=True`). Same per-document tokenize-and-truncate-to-512 transform. Test split.
- **Composition.** Upstream CodeSearchNet Python test split is approximately 22,000 functions (paired with docstrings); the per-experiment subset matches the cross-domain transfer evaluation budget. Not retained after evaluation.
- **Limitations and out-of-scope uses.** CodeSearchNet is a 2019 snapshot of public GitHub Python; it is not representative of all code (no other languages, no proprietary code, no recent code). The "transfers to code" finding is a within-method robustness check, not a downstream code-task benchmark. The loader script passes `trust_remote_code=True` because the upstream dataset requires it; revision pinning covers the loader script as well as the data.
- **Citation.** Husain, H., Wu, H.-H., Gazit, T., Allamanis, M., and Brockschmidt, M. (2019). CodeSearchNet Challenge: Evaluating the State of Semantic Code Search. arXiv:1909.09436.

### SQuAD v2

`rajpurkar/squad_v2` revision `3ffb306f725f7d2ce8394bc1873b24868140c412`.

- **License.** CC BY-SA 4.0 per the [Stanford HF card](https://huggingface.co/datasets/rajpurkar/squad_v2). Crowdsourced extractive QA over Wikipedia passages with adversarially constructed unanswerable questions.
- **Role in this paper.** Downstream RAG-style hallucination detection. Producer is `scripts/rag_hallucination.py`. The model is given a passage plus a question; the probe (trained on WikiText) is evaluated on whether it flags wrong answers that confidence marks correct.
- **Subset and transforms.** Validation split, approximately 12,000 examples upstream (the SQuAD v2 test split is held out for the leaderboard and not publicly labeled). The committed evaluation samples 1,000 questions per evaluated model with a fixed seed. Each example is formatted as a passage-plus-question prompt; the model generates an answer and the probe scores activations at the WikiText-trained peak layer. Exclusive-catch and shared-catch flag rates are computed.
- **Composition.** Per-evaluated-model on the n=1,000 sample: Qwen 2.5 7B Instruct accuracy 0.779 (221 errors), Mistral 7B Instruct v0.3 accuracy 0.484 (516 errors), Phi-3 Mini Instruct accuracy 0.637 (363 errors). Errors form the positive class for the binary observability flagging target; the class balance varies materially across evaluated models.
- **Limitations and out-of-scope uses.** SQuAD v2 questions are extractive and crowdsourced; passages are Wikipedia. The RAG-failure claim is bounded by this format. Real-world retrieval failures may involve longer passages, conflicting evidence across multiple retrieved documents, or domain text outside Wikipedia. SQuAD's adversarial unanswerable design is what makes it useful for hallucination detection; the paper's flag-rate numbers reflect performance under that design, not under arbitrary RAG distributions.
- **Citation.** Rajpurkar, P., Jia, R., and Liang, P. (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. arXiv:1806.03822.

### MedQA-USMLE (4-options)

`GBaker/MedQA-USMLE-4-options` revision `0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18`.

- **License.** CC BY 4.0 per the [GBaker HF card](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options). Derived from the MedQA dataset (Jin et al. 2020) with the answer set normalized to four options.
- **Role in this paper.** Downstream selective-prediction evaluation in a narrow specialist domain. Producer is `scripts/medqa_selective.py`. Tests whether the WikiText-trained probe transfers to medical multiple-choice reasoning.
- **Subset and transforms.** Test split, approximately 1,300 questions upstream. The committed evaluation samples 1,000 questions per evaluated model with a fixed seed. Each example is formatted as a clinical vignette plus four answer options; the model generates an answer letter; the probe scores activations at the WikiText-trained peak layer. Selective-prediction curves and exclusive-catch rates are computed.
- **Composition.** Per-evaluated-model on the n=1,000 sample: Qwen 2.5 7B Instruct accuracy 0.605 (395 errors), Mistral 7B Instruct v0.3 accuracy 0.511 (489 errors), Phi-3 Mini Instruct accuracy 0.582 (418 errors). The positive (error) class is roughly balanced across evaluated models on this benchmark, in contrast to SQuAD where it is not.
- **Limitations and out-of-scope uses.** MedQA-USMLE is multiple-choice English-language US medical board content. The "transfers to MedQA" claim is bounded by this domain and format. Real medical applications include clinical notes, patient histories, multilingual settings, and free-form generation, none of which are tested here. The probe's performance on MedQA is observational evidence that observability is not destroyed by domain shift to medical text; it is not an endorsement for clinical deployment.
- **Citation.** Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., and Szolovits, P. (2020). What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams. arXiv:2009.13081.

### TruthfulQA

`truthfulqa/truthful_qa` config `multiple_choice` revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`.

- **License.** Apache 2.0 per the [TruthfulQA HF card](https://huggingface.co/datasets/truthfulqa/truthful_qa). Adversarially constructed by Lin, Hilton, and Evans (2022) to elicit imitative falsehoods from base language models.
- **Role in this paper.** Boundary case for the observability claim. Producer is `scripts/truthfulqa_hallucination.py`. By design, the dataset isolates the failure mode where confidence cannot help: the model is fluently and confidently wrong because it has imitated a falsehood from training data.
- **Subset and transforms.** Validation split, `multiple_choice` config (817 questions, the full multiple_choice set). Same probe-and-confidence flagging protocol as the other two downstream tasks.
- **Composition.** Per-evaluated-model on the full n=817 set: Qwen 2.5 7B Instruct accuracy 0.345 (535 errors), Mistral 7B Instruct v0.3 accuracy 0.338 (541 errors), Phi-3 Mini Instruct accuracy 0.289 (581 errors). The positive (error) class dominates by a roughly 2-to-1 margin on every evaluated model, reflecting the dataset's adversarial design. This skew is one reason TruthfulQA is reported as a boundary case rather than a positive transfer result.
- **Limitations and out-of-scope uses.** TruthfulQA is adversarially designed; performance numbers on TruthfulQA are not the same as performance on natural confident-wrong distributions. The paper reports this as a *negative* finding: the WikiText-trained observer scores at chance on the TruthfulQA confident-among-confident subset, with AUC 0.499 on Qwen 2.5 7B Instruct, 0.556 on Phi-3 Mini Instruct, and 0.568 on Mistral 7B Instruct v0.3. This is the boundary the paper names: activation monitoring catches token-level prediction failures, not learned falsehoods. TruthfulQA contributes 3 in-band cells to the 7-of-9 aggregate downstream claim. The *negative* near-chance-AUC finding is specific to the fluent-falsehood subset.
- **Citation.** Lin, S., Hilton, J., and Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL 2022. arXiv:2109.07958.

## The Pile (upstream training corpus, not loaded)

The Pythia and Pythia-deduped base models in `results/pythia-*_main.json` are pretrained by EleutherAI on The Pile (and its deduplicated variant). The Pile is the upstream training corpus for those base models; it is not loaded or processed by this repository. Reproducing this paper's results does not require acquiring The Pile. Probing data for every Pythia model in the paper is WikiText-103, pinned above. The Pile's documentation lives with EleutherAI: [Gao et al. (2020). The Pile: An 800GB Dataset of Diverse Text for Language Modeling. arXiv:2101.00027](https://arxiv.org/abs/2101.00027).

## Pretraining-corpus overlap audit

Probe training and probe evaluation use disjoint document pools at the WikiText level (`scripts/run_residualizer_split.py`). Separate from that within-protocol leakage check, the evaluated *language models* themselves were pretrained on corpora that may overlap with the evaluation datasets. The table below is a qualitative audit per evaluation set; per-model corpus details live in `MODELS.md`. Where a corpus is proprietary, expected overlap is inferred from public training reports.

| Evaluation set | Source population | Likely overlap with model pretraining |
|---|---|---|
| WikiText-103 | Wikipedia featured/good articles | **High** for Pythia (The Pile contains Wikipedia), **expected high** for Qwen 2.5, Llama 3.x, Gemma 3, Mistral 7B (large undisclosed mixes that almost certainly include Wikipedia), **low/none** for GPT-2 (WebText explicitly excluded Wikipedia per Radford et al. 2019). |
| C4 (English) | Common Crawl with C4 cleaning rules | **High** for Pythia (The Pile contains Common Crawl), **expected high** for Qwen, Llama, Gemma, Mistral (web-derived corpora overlap heavily with Common Crawl), **low** for GPT-2 (WebText is a separate Reddit-curated subset of the web). |
| SQuAD v2 | Wikipedia passages plus crowdsourced QA annotations | **Passages**: same overlap pattern as WikiText (high for Pythia/Qwen/Llama/Gemma/Mistral, low for GPT-2). **QA annotations**: unlikely in any pretraining corpus before SQuAD v2 release date but conditional on whether the upstream creator scraped HuggingFace datasets pages or academic mirrors during pretraining. Conservatively flag as **possible but unverified**. |
| MedQA-USMLE (4-options) | US medical board examination questions | **Medical text content**: present at low density in any large web-crawled corpus. **Specific question/answer pairs**: unlikely as labeled dataset, possibly present in raw form on study-aid websites. Conservatively flag as **possible but unverified** for the proprietary-mix models; **low** for GPT-2 (WebText is general web). |
| TruthfulQA | Adversarially constructed false-belief patterns | **By construction**, this dataset targets falsehoods that LMs are likely to imitate from training data. Imitative falsehoods imply the *content* is in pretraining for almost every modern LM. The *labeled question/answer pairs* are unlikely to be in any pretraining corpus before TruthfulQA's 2021 release; conditional on dataset-page scraping for later releases. The dataset's design assumes high content overlap; that is the point. |
| OpenWebText | 2019 Reddit-link snapshot of web URLs (open replication of OpenAI's WebText) | **By construction high** for GPT-2 (WebText is the closed OpenAI source that OpenWebText replicates from the same Reddit-link methodology). **Expected high** for Pythia (The Pile contains web data including OpenWebText itself), and for Qwen, Llama, Gemma, Mistral (web-derived pretraining corpora overlap heavily with this distribution). |
| CodeSearchNet (Python) | 2019 GitHub Python functions paired with docstrings | **Expected high** for Pythia (The Pile contains GitHub code), Qwen 2.5, Llama 3.x, Gemma 3 (recent LMs include code in pretraining mixes). **Variable** for Mistral 7B (smaller code share publicly disclosed). **Low** for GPT-2 (WebText is general web, not code-heavy). |

This audit is qualitative because several evaluated models pretrain on undisclosed corpora. The probe-side evidence is more directly bounded: the WikiText probe-fit pool is doc-level disjoint from the WikiText evaluation pool (`scripts/run_residualizer_split.py`), and per-token-loss values are computed on held-out documents. Pretraining-side overlap is a property of the *evaluated language model*, not of the probe; it conditions how to interpret cross-domain transfer numbers but does not invalidate the within-protocol probe-leakage controls.

Cross-corpus passage overlap is a separate concern at the WikiText/SQuAD interface: SQuAD v2 passages are extracted from Wikipedia articles, and the WikiText-103 probe-fit pool draws from Wikipedia featured/good articles. Document-level overlap between the two pools is possible. This does not translate into probe-level leakage. The probe is a fitted direction in activation space, not a text lookup, and downstream catch rates are computed on per-question scores from held-out questions. Direction-level transfer is what the paper claims; document overlap at the underlying corpus does not undermine that claim.

## PII and sensitive content

The seven datasets vary in PII exposure, with the consumer-side posture below; canonical discussion lives with each upstream dataset card.

| Dataset | PII risk | Upstream posture |
|---|---|---|
| WikiText-103 | Low | Wikipedia featured/good articles; public encyclopedic content with editorial review. |
| C4 (English) | Web-scrape residual | Common Crawl source filtered with the C4 cleaning rules of Raffel et al. (2020); incidental personal information from public web text may persist. |
| OpenWebText | Web-scrape residual | 2019 Reddit-link snapshot of public URLs; the Skylion007 release inherits the upstream WebText posture (no targeted PII removal). |
| CodeSearchNet (Python) | Author-identifier residual | Open-source GitHub Python; comments and string literals may contain author handles, emails, or contact information from the source repositories. |
| SQuAD v2 | Low | Wikipedia passages plus crowdsourced QA annotations on those passages; no subjects beyond the upstream Wikipedia content. |
| MedQA-USMLE (4-options) | Low | Synthetic clinical vignettes from US medical board examinations; no real patient data. |
| TruthfulQA | Low | Adversarially constructed prompts; no human subjects. |

This repo does not collect PII, does not redistribute source-text content, and persists only per-token loss values, observer scores, and aggregate flagging counts. The probe is a fitted direction in activation space; activations and derived statistics are not text-recoverable. Each dataset's upstream card and source publication carries the canonical PII discussion.

## Cross-cutting limitations and scope

All seven datasets and The Pile are public, English-language, and pulled from upstream creators (Hugging Face for the seven datasets; EleutherAI for The Pile, not loaded by this repo). This repository is a *consumer* of these datasets, not a creator: no annotations, splits, or transformations originate here, and no human-subjects data is collected by this repo. No institutional ethics review was required of the authors because no data collection occurred; upstream datasets carry their own consent and ethics processes, documented in their HF cards and source publications.

The seven datasets represent English-language text from distinct populations: Wikipedia editors (WikiText-103), web users at internet scale (C4 and OpenWebText), Wikipedia plus crowdworkers (SQuAD v2), US medical board examiners (MedQA-USMLE), open-source Python developers (CodeSearchNet), and adversarially-selected fact patterns drawn from an array of internet-distilled sources (TruthfulQA). Cross-language and cross-demographic generalization of the paper's claims is not measured. The observability and downstream-transfer findings are scoped to English text and the populations represented above; extension to other languages, registers, or population groups is observational and not established here.

The downstream-task evaluations in the paper are scientific benchmarks for a frozen-probe research method. They do not constitute endorsement for production or regulated deployment of activation monitoring on the underlying tasks. For safety-sensitive contexts (clinical decisions, legal or financial advice, hiring, lending, content moderation), the upstream creators' dataset cards and the relevant domain literature take precedence over the role descriptions in this file.

## License compatibility for released artifacts

This repository is licensed under the MIT License. Released artifacts are: source code (MIT), result JSONs and committed metadata (MIT), and the human-readable documentation in this file (MIT). Source-text content from any upstream dataset is never redistributed; the persisted artifacts are statistical summaries (per-token loss values, observer scores, exclusive-catch counts, partial correlations).

| Dataset | License | Constraints inherited by derivatives |
|---|---|---|
| WikiText-103 | CC BY-SA 3.0 + GFDL (dual) | Share-alike applies to substantial reuse of source text under either license. Per-token statistics and activation summaries persisted in result JSONs are not substantial source-text derivatives. The HF card declares both `cc-by-sa-3.0` and `gfdl`; the source content predates Wikipedia's June 2023 migration to CC BY-SA 4.0. |
| C4 (English) | ODC-BY 1.0 | Attribution required for substantial database reuse. Per-experiment subsets are non-substantial slices and statistics; no database is redistributed. |
| OpenWebText | CC0 1.0 | No constraints. |
| CodeSearchNet (Python) | Mixed (HF declares "other"); MIT covers curation tool only | Constituent function bodies retain the licenses of their original GitHub source repositories (heterogeneous). The MIT license at github.com/github/CodeSearchNet covers the upstream curation tool and dataset wrapper. Per-experiment statistics persisted here are derived measurements of model behavior, not redistributed source-text. |
| SQuAD v2 | CC BY-SA 4.0 | Same as WikiText. Per-question scoring outputs and aggregate flag-rate counts are non-substantial. |
| MedQA-USMLE (4-options) | CC BY 4.0 | Attribution required. |
| TruthfulQA | Apache 2.0 | Attribution required; permissive otherwise. |

Attribution to upstream creators is provided through citations in the paper, this file, and the per-dataset entries above. Downstream consumers reproducing or extending this work inherit the per-dataset attribution obligations directly from the upstream cards.

## Recommended and out-of-scope uses

**Recommended uses** for the paper's derived artifacts (probe directions, observer scores, residual targets):

- Reproducing the observability measurements on the same models with the same protocol.
- Extending the protocol to additional autoregressive transformer models or architectures under the same probe class.
- Using the WikiText-trained observer as a starting point for further methodological hardening (alternative target definitions, additional confidence covariates, larger seed counts, alternative readouts).
- Probing-validity research that builds on the confidence-controlled / output-controlled framework introduced here.

**Out-of-scope uses:**

- Production or regulated deployment of activation monitors in clinical, legal, financial, hiring, lending, content moderation, or other safety-sensitive contexts. The paper provides scientific evidence for a research method, not an operational defense.
- Generalization to multilingual settings, low-resource languages, dialog systems, or non-text modalities without re-validation under the protocol described in the paper.
- Detection of fluent confident falsehoods (TruthfulQA-style imitative falsehoods); the paper explicitly identifies this as a method boundary and reports near-chance AUC on this subset across three production instruct models.
- Adversarial-robustness claims: the protocol has not been tested under adaptive attacks. McGuinness et al. (2025) demonstrate that activation monitors can be evaded; the artifacts here are research instruments, not deployed defenses.

The artifacts here are research instruments. Endorsement for any operational use requires task-specific validation that this paper does not provide.

## Distribution and maintenance

The seven datasets are distributed by their upstream creators via Hugging Face. This repository does not redistribute any dataset content; consumed copies are pulled at evaluation time from `load_dataset` calls pinned via `results/dataset_revisions.json`. No significant data preprocessing compute is performed by this repo: the per-document tokenize-and-truncate-to-512 transform runs at evaluation time, and evaluation-side compute is reported in the paper's reproducibility section.

Maintenance of the datasets, including future updates, errata, and deprecations, is the responsibility of the upstream creators. The pinned commit hashes in this repository are immutable references to the dataset state at the paper's evaluation cutoff; subsequent upstream changes are not reflected here. The MLCommons Croissant 1.1 metadata descriptor in `croissant.json` carries the machine-readable record-set inventory and is regenerated by `scripts/generate_croissant.py`; see `croissant.json` for the structured form alongside this human-readable card.

## Updating this file

`results/dataset_revisions.json` is the source of truth for pinned commits; this file is the human-readable companion. When a dataset entry in `dataset_revisions.json` changes, update the matching section here in the same change.
