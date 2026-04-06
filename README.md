# Learned Observers Recover Decision-Quality Signal from Frozen Activations

Can a neural network's internal activations tell you something about its decisions that output confidence does not?

The short answer is yes, but every obvious approach fails. Hand-designed activation statistics (energy, sparsity, entropy, prototype similarity) all collapse to near-zero independent signal once you control for output confidence. The finding that works is specific: a learned linear projection trained with binary supervision on frozen activations recovers a stable direction that confidence cannot access. On GPT-2 124M, three independent initializations converge to the same token ranking (seed agreement +0.99), and a third of the signal survives after controlling for the full output distribution.

This project builds to that conclusion through seven phases, each motivated by the previous result's failure or limitation. Phases 1-3 systematically close off the easy paths. Phases 4-7 show what does work, why, and what it buys you in practice.

## At a glance

**Core question:** Do frozen activations contain decision-quality signal independent of output confidence?

**Answer:** Yes. A learned linear projection with binary supervision recovers it; hand-designed statistics cannot. On GPT-2 124M, the signal is near-deterministic (seed agreement +0.99 across initializations).

| Phase | Question | Result | Takeaway |
|---|---|---|---|
| **Phase 1** | Does training objective change representation structure? | **Yes** | FF induces sparser, lower-rank, more concentrated representations than BP, independent of overlay and normalization confounders. |
| **Phase 2a** | Does FF goodness faithfully read BP activations? | **No** | `sum(h²)` collapses into a confidence proxy after controlling for logit margin and activation norm. |
| **Phase 2b** | Can co-training rescue the observer? | **Weakly** | Denoising produced a small positive partial correlation (+0.066), but with much weaker raw predictive utility. |
| **Phase 3** | Do alternative hand-designed observers work? | **No** | All passive structural observers collapse to near-zero partial correlation under proper controls. |
| **Phase 4** | Can a learned observer head recover signal? | **Yes** | Binary-trained linear heads on frozen BP activations: partial corr +0.28, seed agreement +0.36. |
| **Phase 5** | Does this transfer to transformers? | **Yes** | On frozen GPT-2 124M: partial corr +0.282 +/- 0.001, seed agreement +0.99. Signal peaks at layer 8 of 12. Layer 8 retains +0.099 after controlling for the full output distribution. |
| **Phase 6** | Does the signal catch errors confidence misses? | **Yes** | At 10% flag rate, the layer 8 observer catches 4,368 high-loss tokens (5.2% of test set) that output confidence does not flag. |
| **Phase 7** | How does this compare to SAE-based probes? | **Raw observer wins** | A 768-dim linear observer outperforms a 24,576-feature SAE probe (+0.290 vs +0.255 partial corr). Combining all three channels catches 1.8x more errors. |

**Key findings:**

- **The bottleneck is the training target, not the architecture.** Binary supervision (predict whether loss residual is positive) produces both stronger signal and stable convergence. Regression on continuous residuals finds signal but disagrees across seeds. Linear heads recover ~91% of what MLP heads find. Phase 3's four hand-designed directions all fail; the informative direction is a different learned combination entirely.
- **The signal is partially independent of output information.** On GPT-2 124M, the signal peaks at layer 8 of 12. After controlling for a trained MLP on the full layer-11 representation, layer 8 retains +0.099 partial correlation (+/- 0.008 across seeds). This is not early access to what the model will output. It is information the output does not carry.
- **Multiple channels catch different errors.** At 10% flag rate, the observer catches 4,368 high-loss tokens confidence misses. An SAE probe catches a different 4,527. Combined with confidence, the three channels catch 1.8x more errors than any single channel. No one monitoring signal is sufficient.
- FF induces real structural differences (sparser, lower-rank representations) independent of confounders, but these structural properties do not translate to per-example observability.

## Why this matters

Most deployed activation monitors predict what the model will output, using activations as a cheaper feature space. Whether they capture anything about the decision process that output confidence doesn't already reveal is untested. This project tests the harder question directly: after controlling for output confidence and activation norm, does any activation-derived signal carry independent information about decision quality?

The partial correlation methodology is the key distinction. On MLPs, the confidence control is logit margin; on transformers, max softmax probability. Every phase applies this control, which is why the headline numbers are small. They measure the independent component, not the total correlation. Most published probing results report total correlation without this control, which means their claimed signal may be entirely redundant with confidence.

The results split "observability" into two problems that behave differently.

- **Per-example monitoring** (does the observer flag likely errors on individual inputs?) fails under passive hand-designed readouts. But learned observer heads on frozen activations recover stable signal (Phases 4-5). The problem was not absence of information but absence of the right projection and the right training target.
- **Neuron-level causal targeting** (does the observer identify neurons whose removal disproportionately harms performance?) works with simple statistics. FF-derived signals and magnitude rankings pick out causally important neurons, even though they fail as per-example monitors.

Two practical implications follow. First, per-example observability requires learning the right projection from activations, not computing hand-designed statistics. This holds across architectures; the learned projections that work on MLPs transfer to GPT-2 124M with no loss in signal strength. Second, no single monitoring channel is sufficient. Confidence, raw activation probes, and SAE-based probes each flag different subsets of errors (Phase 7). Production monitoring systems that rely on confidence alone, or on any one activation-derived signal, leave a measurable gap.

A downstream concern: unfaithful probes are evadable. Models can maintain identical input-output behavior while rearranging activations into subspaces that defeat monitors. An observer that reads signal independent of the output distribution should be harder to evade, because the model can't rearrange that signal without changing its internal computation. The +0.099 output-controlled residual is a first measurement of that independence.

### The faithfulness bar

Observability was evaluated against three tests. This project passes two cleanly and leaves the third open.

- **Correlation.** Does the observer signal track decision-relevant metrics beyond what cheap baselines capture? *Passed.* Partial correlation +0.282 after controlling for confidence and activation norm (Phases 4-5).
- **Prediction.** Can the observer rank likely failures in a way that complements output confidence? *Passed.* 4,368 exclusive high-loss catches at 10% flag rate (Phase 6). Three-channel monitoring catches 1.8x more errors than any single channel (Phase 7).
- **Intervention.** Does observer-guided neuron ablation degrade performance faster than random? *Passed on MLPs* (Phase 2 intervention). *Inconclusive on transformers* due to residual stream buffering (Phase 5d). A more sensitive causal design (activation patching, path patching) is needed.

## Phase 1: structural comparison (complete)

Phase 1 establishes what local (Forward-Forward) vs. global (backpropagation) learning objectives do to representation structure. This is setup for the observer experiments: if FF produces structurally different representations, can those differences be read as decision-quality signals?

4x500 MLPs, 50 epochs, 3 seeds. Two confound controls: BP+norm (adds layer normalization matching FF) and BP+overlay (trains BP on label-overlaid input, same scheme as FF).

|                       | FF (MNIST) | BP (MNIST) | FF (CIFAR-10) | BP (CIFAR-10) |
|-----------------------|-----------:|-----------:|---------------:|--------------:|
| Test accuracy         |     94.57% |     98.32% |         47.49% |        54.15% |
| Probe acc (label-masked) | 99.55% |     97.65% |         86.83% |        49.74% |
| Sparsity              |      87.6% |      81.0% |          86.0% |         76.8% |
| Dead neuron fraction  |      23.9% |       6.9% |          20.5% |          4.3% |
| Effective rank        |       44.7 |      164.2 |          140.0 |         336.7 |

**Label overlay is the dominant confounder.** BP+overlay matches or exceeds FF on probe accuracy and pruning robustness. The probe advantage originally attributed to FF comes from the input conditioning scheme, not from local learning.

What FF genuinely produces, independent of overlay: higher activation sparsity, lower effective rank, and more concentrated information in fewer neurons. These structural effects persist across all five model sizes (200K to 8M parameters) in the scaling study. But as Phases 2-3 will show, structural legibility does not translate to per-example observability.

Full per-variant tables in `results/mnist.json` and `results/cifar10.json`. Scaling data in `results/scaling.json`. Analysis and figures in `analyze.ipynb`.

## Phase 2: observer faithfulness (complete)

### Phase 2a: passive observer test (negative)

FF goodness on vanilla BP activations (4x500 MLP, MNIST, 50 epochs, 3 seeds). Raw correlations look strong:

| Observer          | Spearman vs loss | AUC (error detection) |
| ----------------- | ---------------: | --------------------: |
| ff_goodness       |           -0.725 |                 0.923 |
| max_softmax       |           -0.998 |                 0.959 |

But partial correlation of ff_goodness with loss, controlling for logit margin and activation norm: **-0.056** (+/- 0.039 across seeds). The independent component vanishes. `sum(h²)` collapses into activation energy, which is a confidence proxy. Full baseline table in `results/observe_mnist.json`.

### Phase 2b: co-training search

Two co-training formulations tested, both using `sum(h²)` as the observer:

- **Overlay auxiliary** (BP + FF contrastive loss with label overlay). ff_goodness partial correlation: +0.015, inconsistent across seeds. The overlay creates a train/eval domain mismatch that makes the result uninterpretable.

- **Denoising auxiliary** (BP + FF contrastive loss with noise corruption, no overlay). ff_goodness partial correlation: **+0.066** (p < 0.001). AUC dropped from 0.923 to 0.624: denoising decoupled goodness from confidence without replacing the lost predictive utility.

Denoising co-training produced the first positive significant partial correlation in the project (+0.066). The cost: raw error-detection AUC dropped from 0.923 to 0.624, while max softmax on the same model maintained 0.961. The denoising objective decoupled goodness from confidence but did not replace the lost information. This was the foothold for Phase 4: explicit shaping moved the partial correlation from negative to positive, suggesting that observability is trainable even though it is not passively readable.

### Intervention

Even when FF-based signals fail as independent per-example quality estimates, they still identify neurons whose removal disproportionately harms performance. This reveals a second axis of observability: per-neuron causal salience is distinct from per-example faithfulness.

At 70% ablation of the last layer (3 seeds, 50 epochs):

| Strategy | Accuracy |
|---|---|
| FF-targeted | 0.845 |
| Magnitude | 0.836 |
| Class-disc | 0.839 |
| Sparsity-guided | 0.893 |
| Random | 0.983 |
| Anti-targeted | 0.983 |

FF-targeted and magnitude-guided ablation are far more destructive than random, confirming these signals identify causally important neurons. Sparsity-guided ablation is less destructive: the most frequently active neurons are background, not decision-makers. The causally important neurons are concentrated among the high-magnitude, selectively firing ones.

## Phase 3: alternative structural observers (negative)

Phase 2 showed FF goodness fails as a passive observer. The natural follow-up: maybe `sum(h²)` is the wrong readout, and simpler structural metrics on the same activations would work. Three alternatives were tested on vanilla BP activations with no retraining or co-training.

| Observer | What it reads | Partial corr | AUC |
|---|---|---|---|
| ff_goodness | Activation energy | -0.056 | 0.923 |
| active_ratio | Per-example neuron sparsity | -0.035 | 0.502 |
| act_entropy | Activation concentration per layer | -0.039 | 0.429 |
| class_similarity | Cosine similarity to class prototype | -0.141 | 0.951 |

All partial correlations are near zero or negative after controlling for logit margin and activation norm. No passive structural observer on vanilla BP activations recovers meaningful independent signal. The raw Spearman correlations are strong, but the independent component vanishes under proper controls. Structural legibility (Phase 1) is real, but per-example observability beyond confidence has not been achieved through passive readout.

## Phase 4: learned observer heads (complete)

Phase 3 showed hand-designed observers fail. Phase 4 asks: can a trained function extract what passive statistics miss?

A small observer head is trained on frozen BP activations. Four variants cross two axes: architecture (linear vs. MLP) and target (regression on continuous loss residuals vs. binary classification of residual sign). Three seeds each:

| Variant | Partial corr | Seed agreement | Architecture |
|---|---|---|---|
| MLP regression | +0.139 +/- 0.090 | -0.06 | 500→64→1, MSE on residuals |
| Linear regression | +0.177 +/- 0.068 | +0.12 | 500→1, MSE on residuals |
| **MLP binary** | **+0.302 +/- 0.078** | **+0.35** | 500→64→1, BCE on residual sign |
| **Linear binary** | **+0.276 +/- 0.070** | **+0.36** | 500→1, BCE on residual sign |
| Random MLP (baseline) | +0.046 +/- 0.115 | - | Untrained 500→64→1 |

Binary supervision materially improves both partial correlation and seed agreement over regression targets. Regression heads find signal but disagree across seeds (agreement near zero). Binary heads converge on a similar decision boundary instead of carving up a noisy continuous residual landscape differently per seed.

Most of the signal is linearly accessible. Linear binary (+0.276) captures ~91% of what MLP binary (+0.302) finds. Phase 3 tested four specific linear directions (energy, sparsity, entropy, prototype similarity) and the informative direction is a different learned linear combination. The random MLP baseline (+0.046) confirms the learned component is real.

### Weight vector inspection

The three linear binary heads (one per seed) have orthogonal weight vectors (pairwise cosine similarity ~0.01) and zero top-neuron overlap (0/20 shared between seeds 42 and 43). Yet their example rankings correlate moderately (+0.30 to +0.45). Different projections through activation space produce correlated scores because the informative property is not a single direction. It is a distributed geometric property of the activation manifold that multiple linear projections can partially recover.

Weight mass is spread across 250+ of 500 neurons (not concentrated in a few), and the learned directions are orthogonal to the uniform vector (cosine < 0.1), ruling out activation energy as the underlying signal. On MLPs, where each seed trains a different BP model, the observer signal is subspace-like: each seed finds a different functionally useful projection of different underlying geometry. Phase 5a reveals that this instability was a property of varying models, not of the signal itself. On a fixed pretrained transformer, three initializations converge to the same projection (seed agreement +0.99).

## Phase 5: transformer transfer (complete)

Does the Phase 4 finding survive the MLP-to-transformer jump? Pretrained GPT-2 124M (frozen), WikiText-103, linear binary observer heads at every layer.

### Phase 5a: direct replication (positive)

Linear binary observer heads on frozen GPT-2 124M residual streams, 3 seeds, 84,650 token positions:

|                | MLP (Phase 4)    | GPT-2 124M (Phase 5a) |
|----------------|------------------|-----------------------|
| Partial corr   | +0.276 +/- 0.070 | +0.282 +/- 0.001      |
| Seed agreement | +0.36            | +0.99                 |

The partial correlation is nearly identical. The seed agreement jumped from +0.36 to +0.99. On a fixed pretrained model, different observer head initializations converge to essentially the same ranking. The signal is not a per-seed artifact. It is one stable direction in the residual stream.

The MLP instability (Phase 4, +0.36 agreement) was from comparing across different trained models. Different seeds produced different BP models with different activation geometry. On GPT-2, the activations are fixed and the learned projection is near-deterministic.

### Phase 5b: layer sweep

The observer signal exists at every layer, starting at +0.19 (layer 0) and peaking at layer 8 (+0.290). The profile increases monotonically through layer 8, then declines slightly through layers 9-11 (0.285, 0.278, 0.282). Full per-layer data in `results/transformer_observe.json`.

The peak at layer 8, not layer 11, means the observer is reading compositional structure formed during the middle-to-late layers, not the output distribution taking shape at the final layer. The decision-quality information is fully formed three layers before the model commits to a prediction.

### Phase 5c: hand-designed baselines (negative)

The Phase 3 negative result replicates on transformers. All hand-designed statistics collapse under partial correlation controls:

| Observer              | Partial corr (GPT-2, layer 11) |
|-----------------------|---------------------------------|
| ff_goodness           | -0.010                          |
| active_ratio          | -0.057                          |
| act_entropy           | -0.110                          |
| activation_norm       | -0.002                          |
| Learned linear binary | **+0.282**                      |

The gap between hand-designed and learned observers is not an MLP quirk. It is an architecture-general property: the decision-quality signal in frozen activations is invisible to standard statistics and recoverable only by a learned projection with the right training target.

### Phase 5d: intervention (inconclusive)

Observer-guided ablation of MLP intermediate neurons at layer 8, compared against magnitude-guided and random ablation:

| Fraction ablated | Observer | Magnitude | Random |
|---|---|---|---|
| 0% | 4.97 | 4.97 | 4.97 |
| 10% | 5.03 | 4.94 | 4.97 |
| 30% | 4.98 | 5.01 | 4.99 |
| 50% | 5.05 | 4.94 | 4.94 |

No strategy produces meaningful loss increase. Layer 8's MLP is robust to ablation of up to 50% of its 3072 intermediate neurons regardless of which neurons are removed. The residual stream architecture buffers MLP damage through skip connections. The causal question remains open, not answered negatively.

### Phase 5e: full-output control (positive)

The critical test: is the layer 8 signal early access to output information, or a different signal that the output doesn't carry? A small MLP trained on the layer 11 residual stream serves as the strongest possible output-derived baseline. The layer 8 observer is then evaluated after partialling out this predictor.

|         | Standard controls | + Layer 11 predictor |
|---------|-------------------|----------------------|
| Seed 42 | +0.294            | +0.111               |
| Seed 43 | +0.287            | +0.093               |
| Seed 44 | +0.287            | +0.094               |
| Mean    | +0.290            | **+0.099 +/- 0.008** |

The layer 11 predictor absorbs about two-thirds of the signal. But +0.099 survives, consistent across all three seeds. Layer 8 contains decision-quality information that is not recoverable from the model's output distribution. This is not early access to the same signal. It is a different signal, one that is lost or transformed by the time the model produces logits three layers later.

This changes the framing from monitoring (read internal state for signals the output already carries) to observability (read internal state for signals the output does not carry at all).

## Phase 6: practical application (complete)

### Phase 6a: early flagging

Does the layer 8 signal catch errors that output confidence misses? A token is defined as high-loss if its per-position cross-entropy exceeds the median cross-entropy on the test set. Train the observer at layer 8, flag the top-k% of test tokens by observer score, and compare against flagging by low max-softmax.

| Flag rate | Observer precision | Confidence precision | Observer-exclusive catches |
|---|---|---|---|
| 5% | 0.915 | 1.000 | 2,798 |
| 10% | 0.869 | 0.968 | 4,368 |
| 20% | 0.808 | 0.879 | 6,074 |
| 30% | 0.761 | 0.816 | 6,740 |

Confidence has higher standalone precision at every flag rate. But the observer catches a large, non-overlapping set of errors. At 10% flag rate, 4,368 high-loss tokens (5.2% of the test set) are flagged by the observer but not by confidence. These are tokens where the model is confident but wrong, or where the layer 8 representation signals fragility that the output distribution masks.

Combining both methods (flag if either flags) gives 0.904 precision on a wider net at 10% flag rate. The observer is not a replacement for confidence monitoring. It is a complementary channel that reads different information, available before the model finishes its forward pass.

## Phase 7: SAE comparison (complete)

Does a sparse autoencoder feature basis recover the same signal? Using Joseph Bloom's pretrained SAE for GPT-2 small (24,576 features, 97.4% sparsity), same hookpoint, same train/test split, same binary target, same partial correlation evaluation.

### 7a: SAE probe vs raw linear observer

| Method | Partial corr | Seed agreement | Input dims |
|---|---|---|---|
| Raw linear binary | +0.290 | +0.92 | 768 |
| SAE linear binary | +0.255 | +0.94 | 24,576 |

The raw residual stream outperforms the SAE decomposition by 12%. SAE compression loses signal. The interpretable feature basis is not aligned with decision quality. A 768-dim learned projection reads the residual stream geometry more effectively than a linear probe on 24,576 sparse features.

### 7c: rank overlap

Mean rank correlation between SAE probe and raw observer: +0.70. They share about 70% of their ranking information but diverge on 30%. The two methods read partially overlapping aspects of decision quality through different decompositions.

### 7d: three-channel flagging

At 10% flag rate, combining raw observer, SAE probe, and output confidence:

| Channel | Precision | Exclusive catches |
|---|---|---|
| Confidence | 0.968 | - |
| Raw observer | 0.869 | 4,368 |
| SAE probe | 0.842 | 4,527 |
| **All three combined** | **0.864** | **14,661** |

Three complementary channels, each reading different aspects of decision quality from the same activations, together catch 1.8x more high-loss tokens than the best single channel (confidence). The SAE probe catches a slightly different set of errors than the raw observer (4,527 vs 4,368 exclusive catches), confirming the 30% rank divergence from 7c translates to operationally distinct coverage.

## What this means

The experimental arc supports two claims, one methodological and one practical.

**For evaluation:** published activation-monitoring results that report total correlation without controlling for output confidence are not measuring what they claim. The gap between raw Spearman (-0.725 in Phase 2a) and partial correlation (-0.056) is the difference between "this probe tracks loss" and "this probe tells you something confidence doesn't." Any system claiming to read internal state should report the independent component, not the total.

**For deployment:** single-channel monitoring leaves errors on the table. Confidence is the strongest standalone signal (0.968 precision at 10% flag rate), but it misses a large class of errors that activation-based observers catch. The raw observer and SAE probe each flag different subsets of those misses. A production monitoring system that combines learned activation probes with confidence-based flagging has measurably better coverage than either alone.

## Limitations

- Tested on GPT-2 124M. Whether the signal persists, strengthens, or vanishes at billion-parameter scale is unknown.
- No circuit discovery or feature visualization. Statistical proxies only.
- Hyperparameters not swept (FF lr=0.03, BP lr=0.001, auxiliary weight=0.1 based on convention).
- Intervention on GPT-2 is inconclusive due to MLP robustness at layer 8. The causal link between observer-weighted neurons and model decisions is established on MLPs but not on transformers.

## Open questions

Each of these is a separate project with different compute requirements and baselines.

- **Scale.** Does the signal persist, strengthen, or vanish at billion-parameter scale? GPT-2 124M is a testbed. The answer at Llama 8B or larger would determine whether the finding is practically relevant for production monitoring.
- **Causal intervention on transformers.** Phase 5d was inconclusive because residual stream skip connections buffer MLP ablation. A more sensitive design (attention head ablation, path patching, or activation patching) could confirm or deny the causal link on transformers.
- **Cross-task transfer.** Does a single observer head trained on one domain (Wikipedia) flag errors on a different domain (code, dialogue, reasoning)? If so, the signal is a general property of the residual stream geometry, not task-specific.

## How to run

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just). Runs on CPU, MPS, or CUDA.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install just  # or: cargo install just

just test          # run tests
just smoke         # pipeline smoke test (~1 min)
just reproduce     # full reproduction (~60 min)
```

Individual experiments:

```bash
just train                  # Phase 1: MNIST, 3 seeds, 50 epochs
just cifar10                # Phase 1: CIFAR-10, 3 seeds, 50 epochs
just scale                  # Phase 1: scaling study, 5 model sizes
just observe                # Phase 2: observer faithfulness test
just observe-aux            # Phase 2b: auxiliary co-training variant
just observe-denoise        # Phase 2b: denoising co-training variant
just observer-variants      # Phase 4: head variant sweep
just seed-agreement         # Phase 4: cross-seed agreement test
just inspect-weights        # Phase 4: weight vector analysis
just transformer            # Phase 5a: GPT-2 124M observer heads
just transformer-sweep      # Phase 5b: layer sweep (all 12 layers)
just transformer-baselines  # Phase 5c: hand-designed baselines on GPT-2
just transformer-intervention # Phase 5d: neuron ablation intervention
just transformer-output-control # Phase 5e: full-output control
just transformer-flagging   # Phase 6a: early flagging experiment
just transformer-all        # All transformer experiments (5a-6a)
just sae-compare            # Phase 7: SAE comparison (7a + 7c + 7d)
```

Results go to `results/`. Phase 1 charts are generated by `analyze.ipynb`. Phase 2 generates intervention dose-response plots in `assets/`. Phase 5 requires the `transformer` dependency group (installed automatically by `uv run --extra transformer`).

## Repo structure

- `src/train.py` Phase 1: trains FF, BP, and ablation variants, computes confounder-controlled metrics
- `src/scale.py` Phase 1: scaling study across 5 model sizes
- `src/observe.py` Phases 2-3: observer faithfulness testing, co-training variants
- `src/observer_variants.py` Phase 4: observer head variant sweep (linear/MLP, regression/binary)
- `src/seed_agreement.py` Phase 4: cross-seed ranking agreement test
- `src/inspect_weights.py` Phase 4: weight vector analysis for linear binary heads
- `src/transformer_observe.py` Phases 5-6: GPT-2 124M observer heads, layer sweep, baselines, flagging
- `src/sae_compare.py` Phase 7: SAE comparison (probe, rank overlap, three-channel flagging)
- `analyze.ipynb` generates Phase 1 figures and analysis from result JSON files
- `results/` result data (JSON, committed)
- `assets/` generated charts (committed for README)

## References

| Paper                                                                                 | Relevance                                                                       |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [Production-Ready Probes for Gemini (Kramár et al., 2026)](https://arxiv.org/abs/2601.11516) | Google deployed activation probes at scale; found fragility to distribution shifts |
| [Neural Chameleons (McGuinness et al., 2025)](https://arxiv.org/abs/2512.11949)       | Models can learn to evade activation monitors while preserving behavior         |
| [GAVEL: Rule-Based Activation Monitoring (Rozenfeld et al., 2026)](https://arxiv.org/abs/2601.19768) | Composable predicate rules over activation-derived features (ICLR 2026)        |
| [Beyond Linear Probes / TPCs (Oldfield et al., 2026)](https://arxiv.org/abs/2509.26238) | Adjustable-overhead runtime probes with adaptive cascades (ICLR 2026)          |
| [The Forward-Forward Algorithm (Hinton, 2022)](https://arxiv.org/abs/2212.13345)      | Local, layer-wise training without backpropagation; starting point for Phase 1  |
| [Fractured Entangled Representations (2025)](https://arxiv.org/abs/2505.11581)        | BP+SGD produces entangled representations; alternative optimization does not    |
| [Limits of AI Explainability (2025)](https://arxiv.org/abs/2504.20676)                | Proves global interpretability is impossible; local explanations can be simpler |
| [Infomorphic Networks (PNAS, 2025)](https://www.pnas.org/doi/10.1073/pnas.2408125122) | Local learning rules produce inherently interpretable representations           |
| [Inference-Time Intervention (2023)](https://arxiv.org/abs/2306.03341)                | Precedent for inference-time activation monitoring and steering                 |

## License

[MIT License](LICENSE)
