# Learned Observers Recover Decision-Quality Signal from Frozen Activations

Can a neural network's internal activations tell you something about its decisions that output confidence does not? This project tests that question through a controlled experimental arc: establishing that hand-designed activation statistics fail under proper controls, showing that learned linear projections with binary supervision succeed, demonstrating that the finding transfers from MLPs to transformers, and proving the signal catches errors that output monitoring cannot.

**Thesis:** Frozen neural network activations contain decision-quality signal independent of output confidence, but no hand-designed statistic recovers it. Learned linear projections with binary supervision do. On GPT-2 124M, the signal peaks at layer 8, retains +0.099 partial correlation after controlling for the full output distribution, and catches 4,368 high-loss tokens (5.2% of test data) that output confidence does not flag. The bottleneck was the observer learning objective, not the absence of readable structure.

## At a glance

**Core question:** Can frozen neural network activations be read in a way that adds information beyond output confidence?

**Current answer:** Yes. Frozen activations contain independent signal beyond confidence, but hand-designed statistics can't find it. A learned linear projection with binary supervision recovers it. This transfers from MLPs to transformers: on GPT-2 124M, three independent observer initializations converge to the same ranking of 84,650 token positions (seed agreement +0.99).

| Phase | Question | Result | Takeaway |
|---|---|---|---|
| **Phase 1** | Does training objective change representation structure? | **Yes** | FF induces sparser, lower-rank, more concentrated representations than BP, independent of overlay and normalization confounders. |
| **Phase 2a** | Does FF goodness faithfully read BP activations? | **No** | `sum(h²)` collapses into a confidence proxy after controlling for logit margin and activation norm. |
| **Phase 2b** | Can co-training rescue the observer? | **Weakly** | Denoising produced a small positive partial correlation (+0.07), but with much weaker raw predictive utility. |
| **Phase 3** | Do alternative hand-designed observers work? | **No** | All passive structural observers collapse to near-zero partial correlation under proper controls. |
| **Phase 4** | Can a learned observer head recover signal? | **Yes** | Binary-trained linear heads on frozen BP activations: partial corr +0.28, seed agreement +0.36. |
| **Phase 5** | Does this transfer to transformers? | **Yes** | On frozen GPT-2 124M: partial corr +0.283 +/- 0.001, seed agreement +0.99. Signal peaks at layer 8 of 12. Layer 8 retains +0.099 after controlling for the full output distribution. |
| **Phase 6** | Does the signal catch errors confidence misses? | **Yes** | At 10% flag rate, the layer 8 observer catches 4,368 high-loss tokens (5.2% of test set) that output confidence does not flag. |

**Key findings:**

- Hand-designed activation statistics (energy, sparsity, entropy, prototype similarity) all collapse to near-zero partial correlation after controlling for output confidence. This holds on both MLPs and transformers.
- Binary supervision is the unlock. Regression-trained observer heads find signal but disagree across seeds. Binary heads (predict residual sign) produce both stronger partial correlation and stable convergence.
- The signal is linearly accessible. A learned linear projection recovers ~91% of what an MLP head finds. Phase 3 tested four specific linear directions and the informative one is a different combination entirely.
- On GPT-2 124M, the signal peaks at layer 8 of 12. After controlling for the best possible output-derived prediction (a learned layer 11 predictor), layer 8 retains +0.099 partial correlation. This is not early access to output information. It is a different signal that the output does not carry.
- At 10% flag rate on GPT-2, the layer 8 observer catches 4,368 high-loss tokens that output confidence does not flag (5.2% of test set, 87% precision). The observer is complementary to confidence, not redundant.
- FF induces real structural differences (sparser, lower-rank representations) independent of confounders, but these structural properties do not translate to per-example observability.

**Bottom line:** A single linear projection at layer 8 of GPT-2, computed before the model finishes its forward pass, identifies thousands of error-prone tokens invisible to output-confidence monitoring. The signal is stable (+0.99 seed agreement), independent of output (+0.099 after full-output control), and practically useful (4,368 exclusive catches at 10% flag rate).

## Why this matters

Most deployed "observability" systems train binary classifiers on activations to predict output categories (misuse, PII, deception). These achieve high accuracy but measure something different from internal observation: they predict what the model will output, using activations as a cheaper feature space. Whether they capture anything about the decision process that output confidence doesn't already reveal is untested.

This matters because unfaithful probes are evadable. Models can maintain identical input-output behavior while rearranging activations into subspaces that defeat monitors. A faithful observer that reads causal structure should be harder to evade, because the model can't rearrange its causal computation without changing its outputs.

This project tests the harder question: can you learn anything from activations that output confidence doesn't already tell you? The partial correlation methodology (controlling for output confidence and activation norm) is the key distinction. On MLPs, the confidence control is logit margin; on transformers, max softmax probability. Every phase applies this control, which is why the headline numbers are small. They measure the independent component, not the total correlation.

The results split "observability" into two problems that behave differently.

- **Per-example monitoring** (does the observer flag likely errors on individual inputs?) fails under passive hand-designed readouts. But learned observer heads on frozen activations recover stable signal (Phases 4-5). The problem was not absence of information but absence of the right projection and the right training target.
- **Neuron-level causal targeting** (does the observer identify neurons whose removal disproportionately harms performance?) works with simple statistics. FF-derived signals and magnitude rankings pick out causally important neurons, even though they fail as per-example monitors.

The practical implication: per-example observability requires learning the right projection from activations, not computing hand-designed statistics. This holds across architectures. The learned projections that work on MLPs transfer to GPT-2 124M with no loss in signal strength and near-perfect stability across initializations.

### The faithfulness bar

Any observability system must pass three tests:

- **Correlation.** Does the observer signal track decision-relevant metrics (per-example loss, logit margin) beyond what cheap baselines already capture?
- **Intervention.** When neurons are ablated, does observer-guided targeting degrade performance faster than random, in a way that diverges from simple magnitude ranking?
- **Prediction.** Can the observer rank likely failures better than max softmax, entropy, or a linear probe on the same activations?

## Phase 1: structural comparison (complete)

### MNIST (4x500 MLP, 50 epochs, 3 seeds)

|                                 | Local (FF) |    Global (BP) |    BP+norm | BP+overlay |
| ------------------------------- | ---------: | -------------: | ---------: | ---------: |
| Test accuracy                   |     94.57% |     **98.32%** |     98.29% |     95.09% |
| Probe accuracy (label-masked)   |     99.55% |         97.65% |     97.20% | **99.85%** |
| Pruning@90% (live neurons only) |     99.20% |         97.75% |     97.82% | **99.91%** |
| Polysemanticity (classes/neuron) | 1.78 | **1.63** | 2.55 | 4.05 |
| Dead neuron fraction            |      23.9% |       **6.9%** |      13.1% |       9.1% |
| Effective rank (repr. dimensions) |       44.7 |      **164.2** |      145.3 |      112.0 |
| Sparsity                        |  **87.6%** |          81.0% |      75.3% |      55.4% |

BP+norm: same architecture with per-layer L2 normalization matching FF. Normalization is not the confounder; BP+norm performs identically to BP.

BP+overlay: same architecture trained on label-overlaid input (same scheme as FF). **Label overlay is the dominant confounder.** BP+overlay matches or exceeds FF on probe accuracy and pruning robustness. The probe advantage originally attributed to FF was from the input conditioning scheme, not from local learning.

What FF genuinely produces, independent of label overlay: higher activation sparsity, lower effective rank, and more concentrated information in fewer neurons. These are real structural effects of local learning.

### CIFAR-10 (4x500 MLP, 50 epochs, 3 seeds)

|                                 | Local (FF) |    Global (BP) |    BP+norm | BP+overlay |
| ------------------------------- | ---------: | -------------: | ---------: | ---------: |
| Test accuracy                   |     47.49% |     **54.15%** |     53.84% |     28.92% |
| Probe accuracy (label-masked)   | **86.83%** |         49.74% |     53.79% |     99.91% |
| Sparsity                        |  **86.0%** |         76.8% |      71.7% |      75.9% |
| Dead neuron fraction            |      20.5% |       **4.3%** |      11.1% |      20.7% |
| Effective rank                  |      140.0 |      **336.7** |      283.5 |      121.6 |

CIFAR-10 amplifies the structural gaps. FF probe accuracy (86.8%) far exceeds BP (49.7%). BP+overlay collapses to 28.9% task accuracy, confirming label overlay is a severe confounder on harder tasks.

Probes are trained on training-set activations and evaluated on test-set activations (no test-set contamination). Label-masked probing zeros the first n_cls dimensions. Pruning curves use live neurons only. Full analysis in `analyze.ipynb`.

### Scaling study (MNIST, 5 sizes, 3 seeds each)

Do these structural differences hold as models grow? Five configurations from 200K to 8M parameters.

| Size | Params | Acc (FF-BP) | Dead frac (FF-BP) | Eff rank (FF-BP) | Sparsity (FF-BP) |
|---|---|---|---|---|---|
| XS (2x256) | 0.3M | -3.4% | +4.0% | -54 | +49.6% |
| S (4x500) | 1.1M | -3.7% | +17.0% | -120 | +6.5% |
| M (4x1000) | 3.8M | -3.0% | +19.8% | -183 | -0.6% |
| L (6x1000) | 5.8M | -4.1% | +13.0% | -115 | -1.7% |
| XL (8x1000) | 7.8M | -5.4% | +10.4% | -70 | -1.2% |

FF consistently trades accuracy for more concentrated representations. Two patterns persist across all five sizes: higher dead neuron fraction and lower effective rank. The accuracy gap is stable at 3-5%, widening slightly at XL. Sparsity, the most visually striking difference at small scale (FF 89% vs BP 40% at XS), converges as BP models grow deeper and is negligible by M.

Full scaling data in `results/scaling.json` and `assets/scaling.png`.

## Phase 2: observer faithfulness (complete)

### Phase 2a: passive observer test (negative)

FF goodness on vanilla BP activations against baselines (4x500 MLP, MNIST, 50 epochs, 3 seeds):

| Observer          | Spearman vs loss | AUC (error detection) | Within-class rho |
| ----------------- | ---------------: | --------------------: | ---------------: |
| ff_goodness       |           -0.725 |                 0.923 |           +0.887 |
| max_softmax       |           -0.998 |                 0.959 |           +0.801 |
| logit_margin      |           -0.811 |                 0.965 |           +1.000 |
| entropy           |           +0.811 |                 0.964 |           -0.999 |
| activation_norm   |           -0.706 |                 0.917 |           +0.865 |
| probe_confidence  |           -0.629 |                 0.970 |           +0.738 |

Partial correlation of ff_goodness with loss, controlling for logit margin and activation norm: **-0.056** (+/- 0.039 across seeds). The effect is small and inconsistent in sign across seeds. The observer is not tracking decision structure beyond what confidence already captures. `sum(h²)` collapses into activation energy, which is a confidence proxy. Alternative structural observers are tested in Phase 3.

### Phase 2b: co-training search

Two co-training formulations tested, both using `sum(h²)` as the observer:

- **Overlay auxiliary** (BP + FF contrastive loss with label overlay). ff_goodness partial correlation: +0.015, inconsistent across seeds. The overlay creates a train/eval domain mismatch that makes the result uninterpretable.

- **Denoising auxiliary** (BP + FF contrastive loss with noise corruption, no overlay). ff_goodness partial correlation: **+0.070** (p < 0.001). AUC dropped from 0.923 to 0.688: denoising decoupled goodness from confidence without replacing the lost predictive utility.

Denoising co-training produced the first positive significant partial correlation in the project (+0.070). The cost: raw error-detection AUC dropped from 0.923 to 0.688, while max softmax on the same model maintained 0.947. The denoising objective decoupled goodness from confidence but did not replace the lost information. This was the foothold for Phase 4: explicit shaping moved the partial correlation from negative to positive, suggesting that observability is trainable even though it is not passively readable.

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
| Partial corr   | +0.276 +/- 0.070 | +0.283 +/- 0.001      |
| Seed agreement | +0.36            | +0.99                 |

The partial correlation is nearly identical. The seed agreement jumped from +0.36 to +0.99. On a fixed pretrained model, different observer head initializations converge to essentially the same ranking. The signal is not a per-seed artifact. It is one stable direction in the residual stream.

The MLP instability (Phase 4, +0.36 agreement) was from comparing across different trained models. Different seeds produced different BP models with different activation geometry. On GPT-2, the activations are fixed and the learned projection is near-deterministic.

### Phase 5b: layer sweep

The observer signal exists at every layer, starting at +0.19 (layer 0) and peaking at layer 8 (+0.290). The profile is monotonically increasing through layer 8, then plateaus through layer 11 (+0.283). Full per-layer data in `results/transformer_observe.json`.

The peak at layer 8, not layer 11, means the observer is reading compositional structure formed during the middle-to-late layers, not the output distribution taking shape at the final layer. The decision-quality information is fully formed three layers before the model commits to a prediction.

### Phase 5c: hand-designed baselines (negative)

The Phase 3 negative result replicates on transformers. All hand-designed statistics collapse under partial correlation controls:

| Observer              | Partial corr (GPT-2, layer 11) |
|-----------------------|---------------------------------|
| ff_goodness           | -0.010                          |
| active_ratio          | -0.057                          |
| act_entropy           | -0.110                          |
| activation_norm       | -0.002                          |
| Learned linear binary | **+0.283**                      |

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

## Limitations

- Tested on GPT-2 124M. Whether the signal persists, strengthens, or vanishes at billion-parameter scale is unknown.
- No SAE comparison. The most important missing baseline.
- No circuit discovery or feature visualization. Statistical proxies only.
- Hyperparameters not swept (FF lr=0.03, BP lr=0.001, auxiliary weight=0.1 based on convention).
- Intervention on GPT-2 is inconclusive due to MLP robustness at layer 8. The causal link between observer-weighted neurons and model decisions is established on MLPs but not on transformers.

## What this is not

This is not a claim that FF is better than BP, or that FF should replace BP, or that FF is the right observability objective. FF is one instance of a local, layer-wise training signal that served as the starting point for a controlled investigation of what makes internal representations readable. The main finding is that learned observer heads recover signal that hand-designed statistics miss, and that this transfers from MLPs to transformers. On MLPs with varying trained models, the informative structure appears distributed across multiple directions. On a fixed pretrained transformer, it converges to a single stable projection. The remaining questions are whether the signal persists at larger scale and whether observer-guided interventions on transformers confirm the causal link established on MLPs.

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
