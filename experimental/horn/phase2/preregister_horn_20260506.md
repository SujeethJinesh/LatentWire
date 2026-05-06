# HORN Preregistered Mac Gates

- date: 2026-05-06
- status: preregistered before measurement
- branch: directional outlier propagation through hybrid boundaries

## H1a: Single-Model Activation Screen

The first live hybrid packet is only a **single-model screen**. It must target a
live hybrid model from
`experimental/shared/results/hybrid_model_eligibility_20260506/`, use the shared
config-derived boundary IDs, include at least 12 fixed reasoning prompts or an
explicit blocker if no live hybrid weights are available, and pass the real
packet checker. A passing H1a packet may promote the branch to H1/H2 follow-up,
but it cannot by itself support a paper claim.

## H1: Activation Magnitude Characterization

Measure max-channel magnitude and kurtosis immediately before and after hybrid
boundaries. Group boundaries into `attention->ssm` and `ssm->attention`.

Pass if max magnitude differs at least 3x or kurtosis differs at least 2x
between directions on at least 60% of layer pairs in at least two hybrid
models, with a cluster bootstrap 95% lower bound above 1.0 for the selected
directional ratio. Required H1 controls: explicit boundary rows, non-boundary
adjacent rows, permuted-direction rows, and matched normalization-side labels.

## H2: Quantization-Noise Propagation

Inject FP4-equivalent noise on one side of each boundary direction and measure
downstream perplexity or NLL drift.

Pass if one direction is at least 1.5x more sensitive than the other and the
paired bootstrap 95% lower bound on the drift ratio exceeds 1.0. The
perturbation-off hook control must leave logits/NLL unchanged within `1e-5`
relative tolerance before any noisy row is admissible.

## H3: Cross-Model And Architecture Controls

Repeat H1/H2 on additional hybrid models. Pure-attention and pure-Mamba controls
should not show the same directional asymmetry.

Pass if asymmetry direction is consistent across hybrid models and absent or
substantially weaker in controls. "Substantially weaker" means the pure-control
ratio is below 1.2 or its 95% interval overlaps 1.0.

## Kill Rule

Kill if boundary directions do not differ, if noise propagation is symmetric, or
if pure-architecture controls show the same effect size.
