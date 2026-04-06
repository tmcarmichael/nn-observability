# Learned Observers Recover Output-Independent Signal from Frozen Transformer Activations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19435674.svg)](https://doi.org/10.5281/zenodo.19435674)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Can a neural network's internal activations tell you something about its decisions that output confidence does not?

The short answer is yes, but only with the right readout. Hand-designed activation statistics (energy, sparsity, entropy, prototype similarity) collapse to near-zero independent signal once you control for output confidence. A learned linear projection trained with binary supervision on frozen activations recovers a stable direction that output confidence does not fully capture. Across GPT-2 124M to 1.5B, independent initializations produce highly consistent token rankings (seed agreement 0.88-0.95), and the component of signal that survives after controlling for a strong output-side predictor increases across this scaling curve (+0.099 at 124M, +0.174 at 1.5B). The result also replicates across architecture families: Qwen 2.5 1.5B (+0.284) and Llama 3.2 1B (+0.250) show the same pattern under the same evaluation protocol.

This project builds to that conclusion through nine phases, each motivated by the previous result's failure or limitation. Phases 1-3 systematically close off the easy paths. Phases 4-7 show what does work and what it buys in practice. Phase 8 tests the signal across the GPT-2 family. Phase 9 tests cross-family replication.

## At a glance

**Core question:** Do frozen activations contain decision-quality signal independent of output confidence?

**Answer:** Yes. A learned linear projection with binary supervision recovers it under the same controls that cause hand-designed statistics to collapse. The signal is stable across GPT-2 124M to 1.5B (partial corr +0.279 to +0.290) and replicates across architecture families: Qwen 2.5 1.5B (+0.284), Llama 3.2 1B (+0.250). Output-controlled residuals are positive in all three families.

| Phase | Question | Result | Takeaway |
|---|---|---|---|
| **Phase 1** | Does training objective change representation structure? | **Yes** | FF induces sparser, lower-rank, more concentrated representations than BP, independent of overlay and normalization confounders. |
| **Phase 2a** | Does FF goodness faithfully read BP activations? | **No** | `sum(h²)` collapses into a confidence proxy after controlling for logit margin and activation norm. |
| **Phase 2b** | Can co-training rescue the observer? | **Weakly** | Denoising produced a small positive partial correlation (+0.066), but with much weaker raw predictive utility. |
| **Phase 3** | Do alternative hand-designed observers work? | **No** | All passive structural observers collapse to near-zero partial correlation under proper controls. |
| **Phase 4** | Can a learned observer head recover signal? | **Yes** | Binary-trained linear heads on frozen BP activations: partial corr +0.28, seed agreement +0.36. |
| **Phase 5** | Does this transfer to transformers? | **Yes** | On frozen GPT-2 124M: partial corr +0.282 +/- 0.001, seed agreement +0.99. Signal peaks at layer 8 of 12. Layer 8 retains +0.099 after a strong output-side control (MLP on last-layer activations). |
| **Phase 6** | Does the signal catch errors confidence misses? | **Yes** | At 10% flag rate, the layer 8 observer catches 4,368 high-loss tokens (5.2% of test set) that output confidence does not flag. |
| **Phase 7** | How does this compare to SAE-based probes? | **Raw observer wins** | A 768-dim linear observer outperforms a 24,576-feature SAE probe (+0.290 vs +0.255 partial corr). Combining all three channels catches substantially more errors than any single channel. |
| **Phase 8** | Does the signal persist across model scale? | **Yes** | Partial corr +0.279 to +0.290 across GPT-2 124M to 1.5B. Output-independent component increases from +0.099 to +0.174 across this scaling curve. Seed agreement 0.88-0.95. Peak at roughly two-thirds depth throughout. |
| **Phase 9** | Does the signal replicate outside GPT-2? | **Yes** | Qwen 2.5 1.5B (+0.284) and Llama 3.2 1B (+0.250) both replicate. Output-controlled residuals positive in both. Hand-designed baselines collapse across all families. |

**Key findings:**

- **The bottleneck is the training target, not the architecture.** Binary supervision (predict whether loss residual is positive) produces both stronger signal and stable convergence. Regression on continuous residuals finds signal but disagrees across seeds. Linear heads recover ~91% of what MLP heads find. Phase 3's four hand-designed directions all fail; the informative direction is a different learned combination entirely.
- **The signal is partially independent of output information.** On GPT-2 124M, the signal peaks at layer 8 of 12. After controlling for a trained MLP on last-layer activations (a strong output-side predictor), the peak layer retains +0.099 partial correlation. This is not early access to what the model will output. It is information that the output-side predictor does not capture. Across the GPT-2 scaling curve, the output-independent component increases: +0.099 (124M), +0.103 (355M), +0.164 (774M), +0.174 (1.5B). Total signal strength stays flat.
- **Multiple channels catch different errors.** At 10% flag rate, the observer catches 4,368 high-loss tokens confidence misses. An SAE probe catches a different 4,527. Each channel flags thousands of errors the others miss entirely. No single monitoring signal is sufficient.
- FF induces real structural differences (sparser, lower-rank representations) independent of confounders, but these structural properties do not translate to per-example observability.

## Why this matters

Most deployed activation monitors predict what the model will output, using activations as a cheaper feature space. Whether they capture anything about the decision process that output confidence doesn't already reveal is untested. This project tests the harder question directly: after controlling for output confidence and activation norm, does any activation-derived signal carry independent information about decision quality?

The partial correlation methodology is the key distinction. On MLPs, the confidence control is logit margin; on transformers, max softmax probability. Every phase applies this control, which is why the headline numbers are small. They measure the independent component, not the total correlation. Most published probing results report total correlation without this control, which means their claimed signal may be entirely redundant with confidence.

The results split "observability" into two problems that behave differently.

- **Per-example monitoring** (does the observer flag likely errors on individual inputs?) fails under passive hand-designed readouts. Learned observer heads on frozen activations recover stable signal (Phases 4-5). The bottleneck was the projection and the training target, not the information content of activations.
- **Neuron-level causal targeting** (does the observer identify neurons whose removal disproportionately harms performance?) works with simple statistics. FF-derived signals and magnitude rankings pick out causally important neurons, even though they fail as per-example monitors.

Per-example observability requires learning the right projection from activations, not computing hand-designed statistics. This holds across architectures and scales; the learned projections that work on MLPs transfer to GPT-2 124M through 1.5B with no loss in signal strength. No single monitoring channel is sufficient. Confidence, raw activation probes, and SAE-based probes each flag different subsets of errors (Phase 7). Production monitoring systems that rely on confidence alone, or on any one activation-derived signal, leave a measurable gap.

Even systems with strong output-based evaluation operate on the end result of computation. The observer signal is available earlier in the forward pass, before the final mapping into the output distribution, so in principle it could support failure detection during inference rather than only after generation. Under a strong output-side control, Phase 8 suggests that logits capture a shrinking fraction of the internally encoded decision-quality signal across the GPT-2 scaling curve. If that pattern generalizes, an internal monitoring channel that reads signal not fully captured by the output could become more useful at larger scale, not less.

A downstream concern is probe evasion: models can in principle preserve input-output behavior while rearranging activations into subspaces that defeat monitors. An observer that reads signal independent of the output distribution may be harder to evade in this way, because the model would have less freedom to preserve that signal while changing only surface behavior. The output-controlled residual gives one operational measure of that independence: +0.099 at 124M, increasing to +0.174 at 1.5B in Phase 8.

### The faithfulness bar

Observability was evaluated against three tests.

- **Correlation.** Does the observer signal track decision-relevant metrics beyond what cheap baselines capture? *Passed across scale and families.* Partial correlation +0.250 to +0.290 across GPT-2, Qwen, and Llama, after controlling for confidence and activation norm (Phases 4-5, 8-9).
- **Prediction.** Can the observer rank likely failures in a way that complements output confidence? *Passed on GPT-2 124M.* 4,368 exclusive high-loss catches at 10% flag rate (Phase 6). Three-channel monitoring catches substantially more errors than any single channel (Phase 7). Not yet tested at larger model sizes.
- **Intervention.** Does observer-guided ablation degrade performance faster than random? *Passed on MLPs* (Phase 2 intervention). *Partial on transformers* (Phase 5f). Neuron ablation was inconclusive due to residual stream buffering (Phase 5d). Directional ablation shows weak but bidirectional causal evidence: removing the observer direction causes monotonic loss increase, and amplifying it preferentially helps observer-flagged tokens (3x targeting ratio). The observer direction is functionally relevant but not a dominant causal axis of output formation.

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

No strategy produces meaningful loss increase. Layer 8's MLP is robust to ablation of up to 50% of its 3072 intermediate neurons regardless of which neurons are removed. The residual stream architecture buffers MLP damage through skip connections.

### Phase 5f: directional ablation (partial causal)

Phase 5d failed because neuron ablation targets individual basis vectors, but the observer signal is distributed across 250+ neurons (Phase 4). Directional ablation intervenes on the residual stream directly, projecting out the learned observer direction: h' = h - alpha * (h . d) * d. Three baselines (random directions, confidence direction), dose-response sweep (0-100%), and bidirectional steering (removal and amplification).

| Alpha | Observer | Random | Confidence | Obs flagged | Obs unflagged |
|---|---|---|---|---|---|
| 0% | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| 25% | +0.002 | +0.000 | +0.029 | +0.002 | +0.002 |
| 50% | +0.004 | +0.001 | +0.134 | +0.004 | +0.005 |
| 75% | +0.007 | +0.002 | +0.306 | +0.006 | +0.007 |
| 100% | +0.010 | +0.004 | +0.549 | +0.007 | +0.010 |

Removing the observer direction causes monotonic loss increase, roughly 2x the random baseline. The confidence direction is 57x more destructive, which is expected: confidence is directly output-relevant, while the observer reads something more diagnostic than decisive.

The stronger evidence comes from amplification. Adding the observer direction back reduces loss, and the effect is 3x larger on observer-flagged tokens (-0.013) than unflagged tokens (-0.004). This is direction-specific, sign-specific, and target-specific, making it harder to dismiss as generic residual perturbation.

Destructive removal does not selectively harm flagged tokens (ratio 0.63, unflagged degrade slightly more). The observer direction is functionally relevant but not a dominant causal axis. The signal it reads is diagnostic (predicts which tokens will have high loss) rather than decisive (directly determines the output).

### Phase 5e: full-output control (positive)

The critical test: is the layer 8 signal early access to output information, or something the output doesn't carry? A small MLP trained on the layer 11 residual stream serves as a strong output-side control. It can learn any function of the last-layer representation, making it a stronger baseline than raw confidence. The layer 8 observer is then evaluated after partialling out this predictor.

|         | Standard controls | + Layer 11 predictor |
|---------|-------------------|----------------------|
| Seed 42 | +0.294            | +0.111               |
| Seed 43 | +0.287            | +0.093               |
| Seed 44 | +0.287            | +0.094               |
| Mean    | +0.290            | **+0.099 +/- 0.008** |

The layer 11 predictor absorbs about two-thirds of the signal. But +0.099 survives, consistent across all three seeds. Layer 8 contains decision-quality information that remains after accounting for what the output-side predictor captures. This is not well explained as early access to what the model will output. It is a different signal, one that is lost or transformed by the time the model produces logits three layers later.

This shifts the framing from monitoring (read internal state for signals the output already carries) toward observability (read internal state for signals the output does not carry).

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

Combining both methods (flag if either flags) gives 0.904 precision at an effective flag rate of ~18% (the union of two 10% sets with partial overlap). The observer is not a replacement for confidence monitoring. It is a complementary channel that catches errors confidence misses, available before the model finishes its forward pass.

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

Each channel flags 10% of test tokens independently. The three-channel union has a larger effective flag budget (~25% of tokens after overlap), so the 1.8x catch improvement reflects both complementary coverage and expanded flagging volume. The comparison that controls for budget is the per-channel exclusive catches: the SAE probe catches 4,527 high-loss tokens that neither confidence nor the raw observer flags, and the raw observer catches 4,368 that neither of the others flags. The 30% rank divergence from 7c translates to operationally distinct error coverage.

### 7b: three-channel causal decomposition

Directional ablation applied to each channel's direction independently, measuring loss delta on each channel's exclusive token catches. The 3x3 matrix of (direction removed) x (token subset affected) tests whether the 7d correlational structure is causally real. Effect sizes are small and the expected diagonal dominance pattern (each direction disproportionately hurting its own exclusive catches) does not emerge clearly. The residual stream's redundancy absorbs single-direction removal without localized damage, consistent with the Phase 5f finding that the observer direction is functional but not output-dominant.

## Phase 8: Scale characterization (complete)

Phases 5-7 established the finding on GPT-2 124M. Phase 8 tests whether the signal persists across model scale. The GPT-2 family (124M, 355M, 774M, 1.5B) provides a four-point scaling curve with no confounders: same architecture, tokenizer, and training distribution. Model size is the only variable.

Three diagnostics tracked per model, each with a specific failure mode:

| Diagnostic | What it tests | Failure mode |
|---|---|---|
| Partial correlation at peak layer | Does signal strength hold? | Collapses as model capacity grows |
| Output-controlled residual | Does the output-independent component persist? | Absorbed by larger output head |
| Seed agreement | Does a stable linear readout emerge? | Fragments into subspace at scale |

### Results

| Model | Params | Peak layer | Partial corr | Output-controlled | Seed agreement |
|---|---|---|---|---|---|
| GPT-2 | 124M | L8 (67%) | +0.290 | +0.099 | +0.918 |
| GPT-2 Medium | 355M | L16 (67%) | +0.279 | +0.103 | +0.877 |
| GPT-2 Large | 774M | L24 (67%) | +0.286 | +0.164 | +0.901 |
| GPT-2 XL | 1558M | L34 (71%) | +0.290 | +0.174 | +0.952 |

**Partial correlation is stable across 12x scale.** +0.279 to +0.290 across all four model sizes (bootstrap 95% CIs overlap). A learned linear projection recovers nearly the same amount of decision-quality information regardless of model capacity.

**The output-independent component increases across this scaling curve.** After controlling for a strong output-side predictor (MLP on last-layer activations), the surviving signal increases from +0.099 at 124M to +0.103 at 355M, +0.164 at 774M, and +0.174 at 1.5B. Under this output-side control, the output-side predictor captures a shrinking fraction of the residual-stream decision-quality signal at larger model sizes. The trend is monotonic across this four-point curve, though confirming it beyond the GPT-2 family requires further scaling experiments.

**Seed agreement stays high.** 0.88-0.95 across all sizes at the peak (two-thirds depth) layer. Phase 5a reported +0.99 on GPT-2 124M at layer 11 (the last layer); Phase 8 reports +0.918 on the same model at layer 8 (the peak layer). The difference reflects the layer, not a regression: last-layer representations are closer to the output distribution, which is more constrained, so probes trained there agree more tightly. At the peak layer, agreement is slightly lower but still high, indicating a near-canonical linear readout rather than a fragmented subspace.

**Peak layer is consistently at roughly two-thirds depth.** The peak partial correlation occurs at layers 8/12, 16/24, 24/36, and 34/48. The observer signal is fully formed well before the model commits to a prediction, and this relative position is stable across model sizes.

**Note on GPT-2 Medium peak selection.** The global partial correlation peak for Medium lands at layer 23 of 24 (96% depth), essentially the output layer. At that position, the output-control test becomes degenerate (comparing a layer's signal against itself). The table reports layer 16 (67% depth), which is the highest-signal layer with clean separation from the output representation. The partial correlation at layer 16 is +0.279 (3 seeds, versus +0.286 from the single-seed coarse sweep). Layer 16 is consistent with the two-thirds-depth pattern observed in all other models.

Full per-layer profiles and bootstrap CIs in `results/transformer_observe.json` under key `"8"`.

Run: `just phase8`

### Methodology hardening

Three supplementary experiments tighten the GPT-2 124M results without changing the research narrative. These are infrastructure improvements, not a numbered phase.

**20-seed statistical hardening** (`just hardening`). Trains 20 independent observer heads at layer 11 (seeds 42-61) and reports bootstrap 95% confidence intervals. The 3-seed CIs from Phases 5-7 are tight but underpowered for formal inference.

**Control sensitivity** (`just control-sensitivity`). Tests partial correlation stability across six control specifications: no controls (raw Spearman), softmax only, norm only, the standard pair, standard plus logit entropy, and a nonlinear MLP predictor trained on the same two covariates. If the nonlinear control absorbs the signal, the linear residualization was hiding a confound.

**Cross-domain transfer** (`just cross-domain`). Trains the observer on WikiText-103, evaluates on OpenWebText (web text) and CodeSearchNet (Python functions). If partial correlation transfers, the signal is a general property of the residual stream, not a Wikipedia-specific artifact.

## Phase 9: Cross-family replication (complete)

Phase 8 established the signal across the GPT-2 family. Phase 9 tests whether it is a GPT-2-specific artifact or a broader property of pretrained decoder-only transformers. Same evaluation protocol: layer sweep, three-seed battery, output-controlled residual, and negative baselines (hand-designed observers, random head).

### 9a: Llama 3.2 1B

| Model | Params | Peak layer | Partial corr | Output-controlled | Seed agreement |
|---|---|---|---|---|---|
| Llama 3.2 1B | 1236M | L6 (38%) | +0.250 | +0.126 | +0.999 |

The signal replicates in a third architecture family (Meta Llama, full attention, RoPE, GQA). Partial correlation is +0.250, slightly below the GPT-2/Qwen band (+0.279 to +0.290), likely reflecting the smaller model size (1.2B vs 1.5B). Output-controlled residual is +0.126, consistent with the GPT-2 range. Seed agreement is +0.999, the highest in the project. All hand-designed baselines collapse (ff_goodness -0.013, active_ratio +0.018, act_entropy -0.015, activation_norm -0.000, random head +0.028).

The peak at layer 6 of 16 (38% depth) is earlier than the two-thirds pattern observed in GPT-2 and Qwen. Whether this reflects Llama's architecture (grouped-query attention, different layer normalization) or the smaller model size is an open question.

### 9b: Qwen 2.5 (0.5B, 1.5B)

| Model | Params | Peak layer | Partial corr | Output-controlled | Seed agreement |
|---|---|---|---|---|---|
| Qwen 2.5 0.5B | 495M | L0 (0%) | +0.134 | +0.055 | +0.998 |
| Qwen 2.5 1.5B | 1544M | L19 (68%) | +0.284 | +0.207 | +0.982 |

**Qwen 2.5 1.5B replicates the full finding.** Partial correlation (+0.284) is in the GPT-2 band (+0.279 to +0.290). The output-controlled residual (+0.207) is the highest measured in the project, exceeding GPT-2 XL (+0.174). Seed agreement is +0.982. The peak layer falls at 68% depth, consistent with the two-thirds-depth pattern observed across the GPT-2 family. All hand-designed baselines collapse to near zero (ff_goodness -0.000, active_ratio +0.017, act_entropy -0.027, activation_norm +0.001). The random head baseline is -0.035. The full failure-then-recovery pattern replicates.

**Qwen 2.5 0.5B shows weaker signal.** Partial correlation is +0.134 with a positive but small output-controlled residual (+0.055). The layer profile does not form a clean mid-depth peak; the signal is present but does not concentrate at two-thirds depth in this 24-layer model. This is consistent with the pattern from Phase 8: signal strength and geometric stability increase with model capacity.

### Cross-family summary

| Model | Family | Params | Partial corr | Output-controlled | Seed agreement | Peak depth |
|---|---|---|---|---|---|---|
| GPT-2 XL | GPT-2 | 1558M | +0.290 | +0.174 | +0.952 | 71% |
| Qwen 2.5 1.5B | Qwen | 1544M | +0.284 | +0.207 | +0.982 | 68% |
| Llama 3.2 1B | Meta | 1236M | +0.250 | +0.126 | +0.999 | 38% |

Three architecture families, three different training corpora, three different tokenizers. All show positive partial correlation (+0.250 to +0.290), positive output-controlled residuals (+0.126 to +0.207), and high seed agreement (+0.952 to +0.999). All hand-designed baselines collapse in every family. The learned observer signal does not appear to be a GPT-2-specific artifact.

The consistency of this signal across independently trained model families suggests that it reflects a recurring property of how autoregressive transformers encode computation quality, rather than a result specific to any one architecture, training corpus, or tokenizer.

Run: `just phase9a` (Llama), `just phase9b` (Qwen), `just phase9` (both)

## What this means

Across three independent architecture families (GPT-2, Qwen, Llama), the residual stream contains a stable decision-quality signal with a positive output-independent component in every case tested. This result has two complementary readings depending on the objective.

**For observability,** this is a strong positive result. The signal persists across model sizes and architecture families, stays linearly accessible, and becomes more cleanly separable from output-derived information at larger scale. A learned linear projection recovers partial correlation of +0.250 to +0.290 across GPT-2, Qwen, and Llama, and the component that survives after controlling for a strong output-side predictor increases from +0.099 to +0.207 across the models tested. The residual-stream decision-quality signal appears stable across the models tested and becomes more distinguishable from output-derived information under this control setting.

**For deployment and safety,** the same result implies that output-based monitoring captures a shrinking fraction of the model's internally encoded decision-quality signal as models scale. Confidence-only oversight is not just incomplete (Phase 6 already showed that); it becomes progressively more incomplete. At Qwen 1.5B, 73% of the observer's signal is not captured by the output-side predictor, compared to 34% at GPT-2 124M. This pattern holds across three architecture families (Phase 9).

**For evaluation methodology,** published activation-monitoring results that report total correlation without controlling for output confidence are not measuring what they claim. The gap between raw Spearman (-0.725 in Phase 2a) and partial correlation (-0.056) is the difference between "this probe tracks loss" and "this probe tells you something confidence doesn't." Any system claiming to read internal state should report the independent component, not the total.

**For multi-channel monitoring,** no single signal is sufficient. Confidence, raw activation probes, and SAE-based probes each flag different subsets of errors (Phase 7). The three channels together catch substantially more errors than any single channel, though the 1.8x figure reflects both complementary coverage and expanded flag budget (three independent 10% thresholds produce a union larger than 10%). The budget-controlled comparison is the exclusive catches: each channel flags thousands of high-loss tokens that neither of the other two catches.

## Limitations

- Tested on GPT-2 (124M to 1.5B), Qwen 2.5 (0.5B, 1.5B), and Llama 3.2 (1B). All base models up to 1.5B. Whether the signal persists at frontier scale (8B+) or in instruction-tuned models is unknown.
- No circuit discovery or feature visualization. Statistical proxies only.
- Hyperparameters not swept (FF lr=0.03, BP lr=0.001, auxiliary weight=0.1 based on convention).
- Causal evidence on transformers is partial. Directional ablation (Phase 5f) shows weak but bidirectional functional relevance; the observer direction is not a dominant causal axis of output formation.
- Methodology hardening experiments (20-seed CIs, control sensitivity, cross-domain transfer) are implemented but not yet run.

## Open questions

Each of these is a separate project with different compute requirements and baselines.

- **Frontier scale.** Phases 8-9 cover GPT-2, Qwen, and Llama up to 1.5B. Whether the signal persists at 8B+ would determine whether the finding is practically relevant for production-scale monitoring.
- **Stronger causal evidence.** Phase 5f provides partial causal support (monotonic dose-response, amplification preferentially helps flagged tokens). Stronger designs (multi-direction projection, activation patching, path patching) could establish whether the observer direction is causally necessary for specific decisions, not just functionally correlated.
- **Cross-domain transfer.** The cross-domain experiment tests WikiText-to-OpenWebText and WikiText-to-code transfer. Broader coverage (dialogue, reasoning, multilingual) would further characterize the signal's generality.
- **Mechanism.** What is the learned direction actually encoding? Circuit-level analysis or feature visualization could connect the statistical finding to a mechanistic account.

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
just causal                 # 7b: three-channel causal decomposition
just phase8                 # Phase 8: GPT-2 scaling curve (124M → 1.5B)
just phase9a                # Phase 9a: Llama 3.2 1B cross-family test
just phase9b                # Phase 9b: Qwen 2.5 0.5B + 1.5B replication
just phase9                 # Phase 9: all cross-family experiments
just hardening              # 20-seed statistical hardening
just control-sensitivity    # Control sensitivity analysis
just cross-domain           # Cross-domain transfer test
```

Results go to `results/`. Phase 1 charts are generated by `analyze.ipynb`. Phase 2 generates intervention dose-response plots in `assets/`. Phase 5 requires the `transformer` dependency group (installed automatically by `uv run --extra transformer`).

## Repo structure

- `src/train.py` Phase 1: trains FF, BP, and ablation variants, computes confounder-controlled metrics
- `src/scale.py` Phase 1: scaling study across 5 model sizes
- `src/observe.py` Phases 2-3: observer faithfulness testing, co-training variants
- `src/observer_variants.py` Phase 4: observer head variant sweep (linear/MLP, regression/binary)
- `src/seed_agreement.py` Phase 4: cross-seed ranking agreement test
- `src/inspect_weights.py` Phase 4: weight vector analysis for linear binary heads
- `src/transformer_observe.py` Phases 5-6, 8-9: GPT-2 and cross-family observer heads, layer sweep, baselines, flagging, scaling, cross-family replication
- `src/sae_compare.py` Phase 7: SAE comparison (probe, rank overlap, three-channel flagging, causal decomposition)
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
