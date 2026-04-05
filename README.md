# Toward Dual-Path Architectures for Neural Network Observability

Can a neural network's internal structure tell you something about its decisions that output confidence does not? This project tests that question systematically: compare representations under different training rules, then try to read decision-quality signals from BP activations using passive observers and co-training.

**Thesis:** Passive hand-designed observers fail to read decision-quality signal from BP activations beyond confidence. But supervised observer heads on frozen activations succeed: binary-trained linear projections recover stable independent signal (partial correlation +0.28, seed agreement +0.36). The bottleneck was the observer learning objective, not the absence of readable structure.

## At a glance

**Core question:** Can internal structure in standard BP models be read in a way that adds information beyond output confidence?

**Current answer:** Yes. Frozen BP activations contain independent signal beyond confidence, but hand-designed statistics can't find it. A learned linear projection with binary supervision recovers a stable direction in activation space correlated with residual error.

| Phase | Question | Result | Takeaway |
|---|---|---|---|
| **Phase 1** | Does training objective change representation structure? | **Yes** | FF induces sparser, lower-rank, more concentrated representations than BP, independent of overlay and normalization confounders. |
| **Phase 2a** | Does FF goodness faithfully read BP activations? | **No** | `sum(h²)` collapses into a confidence proxy after controlling for logit margin and activation norm. |
| **Phase 2b** | Can co-training rescue the observer? | **Weakly** | Denoising produced a small positive partial correlation (+0.07), but with much weaker raw predictive utility. |
| **Phase 3** | Do alternative hand-designed observers work? | **No** | All passive structural observers collapse to near-zero partial correlation under proper controls. |
| **Phase 4** | Can a learned observer head recover signal? | **Yes** | Binary-trained linear heads on frozen BP activations: partial corr +0.28, seed agreement +0.36. |

**Key findings:**

- FF changes representation structure in real, confounder-controlled ways.
- Hand-designed passive observers (energy, sparsity, entropy, prototype similarity) all fail under partial correlation controls.
- Learned observer heads on frozen BP activations recover stable independent signal. Binary supervision materially improves both partial correlation and seed agreement over regression targets.
- Most of the signal is linearly accessible. The missed direction was a learnable projection, not nonlinear structure.

**Bottom line:** The signal is present in frozen activations, but the hand-designed observers tested in Phases 2-3 failed to recover it. Learned binary observer heads find stable independent projections. The next test is whether this transfers to transformers.

## Why this matters

Most deployed "observability" systems train binary classifiers on activations to predict output categories (misuse, PII, deception). These achieve high accuracy but measure something different from internal observation: they predict what the model will output, using activations as a cheaper feature space. Whether they capture anything about the decision process that output confidence doesn't already reveal is untested.

This matters because unfaithful probes are evadable. Models can maintain identical input-output behavior while rearranging activations into subspaces that defeat monitors. A faithful observer that reads causal structure should be harder to evade, because the model can't rearrange its causal computation without changing its outputs.

This project tests the harder question: can you learn anything from activations that output confidence doesn't already tell you? The partial correlation methodology (controlling for logit margin and activation norm) is the key distinction. Every phase applies this control, which is why the headline numbers are small. They measure the independent component, not the total correlation.

The results split "observability" into two problems that behave differently.

- **Per-example monitoring** (does the observer flag likely errors on individual inputs?) fails under passive hand-designed readouts. But learned observer heads on frozen activations recover stable signal (Phase 4). The problem was not absence of information but absence of the right projection.
- **Neuron-level causal targeting** (does the observer identify neurons whose removal disproportionately harms performance?) works with simple statistics. FF-derived signals and magnitude rankings pick out causally important neurons, even though they fail as per-example monitors.

The practical implication: per-example observability requires learning the right projection from activations, not just computing hand-designed statistics. The next question is whether the learned projections that work on MLPs have analogs in transformer residual streams.

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

Denoising co-training produced the only positive significant partial correlation in the project (+0.070). The cost: raw error-detection AUC dropped from 0.923 to 0.688, while max softmax on the same model maintained 0.947. The denoising objective successfully decoupled goodness from confidence but did not replace the lost information. This is the foothold for Phase 4: explicit shaping moved the partial correlation from negative to positive, suggesting that observability may be trainable even though it is not passively readable.

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

A small observer head is trained on frozen BP activations to predict per-example loss residuals (the component of loss not explained by logit margin and activation norm). Four variants tested across 3 seeds:

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

Weight mass is spread across 250+ of 500 neurons (not concentrated in a few), and the learned directions are orthogonal to the uniform vector (cosine < 0.1), ruling out activation energy as the underlying signal. The observer signal is subspace-like: each seed finds a different functionally useful projection of the same underlying activation geometry.

### Phase 5: transformer transfer (planned)

Linear binary observer heads are the primary candidate for GPT-2 124M validation. The methodology (partial correlations after controlling for output-derived baselines) transfers directly. The question is whether the geometric property discovered on MLP activations has an analog in transformer residual streams.

## Limitations

- Small scale (MLPs on MNIST/CIFAR-10). Results may not transfer to transformers or billion-parameter models.
- No SAE comparison. The most important missing baseline.
- No circuit discovery or feature visualization. Statistical proxies only.
- Hyperparameters not swept (FF lr=0.03, BP lr=0.001, auxiliary weight=0.1 based on convention).
- Intervention tests identify causally important neurons but do not provide independent per-example monitoring.

## What this is not

This is not a claim that FF is better than BP, or that FF should replace BP, or that FF is the right observability objective. FF is one instance of a local, layer-wise training signal that served as the starting point for a controlled investigation of what makes internal representations readable. The main finding is that learned observer heads recover signal that hand-designed statistics miss, and that the informative structure is a distributed geometric property rather than a single interpretable feature. The remaining question is whether this transfers to transformer architectures.

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
```

Results go to `results/`. Phase 1 charts are generated by `analyze.ipynb`. Phase 2 generates intervention dose-response plots in `assets/`.

## Repo structure

- `src/train.py` Phase 1: trains FF, BP, and ablation variants, computes confounder-controlled metrics
- `src/scale.py` Phase 1: scaling study across 5 model sizes
- `src/observe.py` Phases 2-4: observer faithfulness testing, learned observer heads
- `src/observer_variants.py` Phase 4: observer head variant sweep (linear/MLP, regression/binary)
- `src/seed_agreement.py` Phase 4: cross-seed ranking agreement test
- `src/inspect_weights.py` Phase 4: weight vector analysis for linear binary heads
- `analyze.ipynb` generates Phase 1 figures and analysis from result JSON files
- `results/` result data (JSON, committed)
- `assets/` generated charts (committed for README)

## References

| Paper                                                                                 | Relevance                                                                       |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [The Forward-Forward Algorithm (Hinton, 2022)](https://arxiv.org/abs/2212.13345)      | Local, layer-wise training without backpropagation                              |
| [Fractured Entangled Representations (2025)](https://arxiv.org/abs/2505.11581)        | BP+SGD produces entangled representations; alternative optimization does not    |
| [Limits of AI Explainability (2025)](https://arxiv.org/abs/2504.20676)                | Proves global interpretability is impossible; local explanations can be simpler |
| [Infomorphic Networks (PNAS, 2025)](https://www.pnas.org/doi/10.1073/pnas.2408125122) | Local learning rules produce inherently interpretable representations           |
| [Scalable FF (ICML 2025)](https://arxiv.org/abs/2501.03176)                           | Block-local hybrids outperform pure BP                                          |
| [Contrastive FF for ViT (2025)](https://arxiv.org/abs/2502.00571)                     | FF applied to transformers, small performance gap vs BP                         |
| [Deep-CBN Hybrid (2025)](https://www.nature.com/articles/s41598-025-92218-y)          | FF+BP hybrid exceeds prior baselines in molecular prediction                    |
| [Inference-Time Intervention (2023)](https://arxiv.org/abs/2306.03341)                | Precedent for inference-time activation monitoring and steering                 |

## License

[MIT License](LICENSE)
