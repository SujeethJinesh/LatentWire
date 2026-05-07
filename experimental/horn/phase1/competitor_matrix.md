# HORN Competitor Matrix

Status: historical novelty guardrail; current branch evidence is demoted by
H1a/H2 failures as of 2026-05-07. Use the README, reviewer pack, and
preregistration closure before treating this matrix as evidence for any live or
GPU-ready claim.

## Claim Boundary

HORN is only novel if directional attention-to-SSM versus SSM-to-attention
boundary behavior is robust in hybrid models and absent or much weaker in
pure-attention and pure-SSM controls.

## Competitor Classes

| Class | Examples to cite/check | Reviewer risk | Required separation |
|---|---|---|---|
| Activation outlier studies | Transformer outlier-channel work | "This is generic outliers." | Compare boundary directions, non-boundary pairs, and pure architectures. |
| Quantization smoothing | SmoothQuant/AWQ-style salience | "Smoothing already handles this." | Show direction-specific sensitivity after matched normalization. |
| Rotation methods | QuaRot, SpinQuant, HIGGS | "A rotation baseline solves it." | Include identity, random orthogonal, Hadamard, and bounded scaling controls if H2 passes. |
| Hybrid model characterization | Hybrid-LLM outlier or sensitivity audits | "They already found hybrid outliers." | Report directional propagation and noise amplification, not only magnitude. |
| HBSM overlap | Layer-sensitivity predictors | "This is the same result." | Fold into HBSM if directional boundaries explain the same variance. |

## Before A Paper Claim

- Verify module names with manual architecture maps, not substring detection alone.
- Include direction-label permutation and non-boundary adjacent-pair controls.
- Normalize for RMSNorm/LayerNorm placement and matched activation scale.
- Require paired CIs for H1 and H2, not only median ratios.

## Executable Baseline Columns

Every real H1/H2 result table must include:

| Column | Meaning |
|---|---|
| `boundary_direction` | `attention->ssm` or `ssm->attention`, from an explicit architecture map. |
| `non_boundary_control` | Adjacent non-boundary transition with matched layer distance. |
| `permuted_direction` | Direction label permutation; should erase the effect. |
| `norm_position` | Pre/post RMSNorm or LayerNorm position for scale matching. |
| `matched_scale_noise` | Same-magnitude noise applied to both directions. |
| `pure_arch_control` | Pure-attention or pure-SSM control where directional hybrid asymmetry should weaken. |

## Source Anchors Checked

- SmoothQuant: `https://arxiv.org/abs/2211.10438`
- QuaRot: `https://arxiv.org/abs/2404.00456`
- KL Lens: `https://arxiv.org/abs/2604.13440`
- vLLM hybrid model support overview: `https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/`
