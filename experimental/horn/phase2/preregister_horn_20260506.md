# HORN Preregistered Mac Gates

- date: 2026-05-06
- status: preregistered before measurement
- branch: directional outlier propagation through hybrid boundaries

## H1: Activation Magnitude Characterization

Measure max-channel magnitude and kurtosis immediately before and after hybrid
boundaries. Group boundaries into `attention->ssm` and `ssm->attention`.

Pass if max magnitude differs at least 3x or kurtosis differs at least 2x
between directions on at least 60% of layer pairs in at least two hybrid
models.

## H2: Quantization-Noise Propagation

Inject FP4-equivalent noise on one side of each boundary direction and measure
downstream perplexity or NLL drift.

Pass if one direction is at least 1.5x more sensitive than the other.

## H3: Cross-Model And Architecture Controls

Repeat H1/H2 on additional hybrid models. Pure-attention and pure-Mamba controls
should not show the same directional asymmetry.

Pass if asymmetry direction is consistent across hybrid models and absent or
substantially weaker in controls.

## Kill Rule

Kill if boundary directions do not differ, if noise propagation is symmetric, or
if pure-architecture controls show the same effect size.
