# SSQ-LR Nemotron State-Cache Baseline Notes

Date: 2026-05-07

Purpose: ground SSQ-LR's novelty boundary against current hybrid
Mamba-Transformer deployment recipes for recurrent SSM cache precision.

## Primary Sources

- NVIDIA Research. "Nemotron 3 Super: Open, Efficient Mixture-of-Experts
  Hybrid Mamba-Transformer Model for Agentic Reasoning." Technical report PDF:
  <https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf>
- NVIDIA Nemotron deployment docs, "Stage 3: Quantization." It states that the
  Mamba SSM cache is a distinct quantization challenge and names FP16 with
  stochastic rounding as the selected deployed recipe:
  <https://docs.nvidia.com/nemotron/latest/nemotron/super3/quantization.html>
- NVIDIA Nemotron DGX Spark deployment guide. It exposes deployment flags
  `mamba_ssm_cache_dtype: float16` and `mamba_ssm_stochastic_rounding: true`:
  <https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide/README.html>

## Implications For SSQ-LR

- SSQ-LR cannot claim novelty for identifying SSM cache precision as important.
  The novelty boundary must be narrower: sub-FP16 recurrent state recipes for
  hybrid reasoners, with explicit quality and byte accounting.
- A credible SSQ-LR S2 gate must compare against FP16 stochastic rounding and
  INT16/block-scaled cache baselines, not only BF16, INT8, FP8, and MXFP4/INT3
  Mac simulations.
- Argmax fidelity is insufficient as a quality proxy. The gate should track
  paired continuation NLL and verbosity/length drift before any revival, because
  cache precision can preserve the next-token argmax while changing decode
  trajectory under longer generation.

## Current Use

This memo supports the 2026-05-07 SSQ-LR status update: S1b heterogeneity is
alive, but S2 remains blocked until a frozen sub-4-bit/native-packed recipe beats
the current FP16-stochastic-rounding deployment baseline under paired quality
metrics.
