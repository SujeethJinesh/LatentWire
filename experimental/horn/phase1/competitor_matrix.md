# HORN Competitor Matrix

Status: local novelty guardrail before boundary measurements.

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
