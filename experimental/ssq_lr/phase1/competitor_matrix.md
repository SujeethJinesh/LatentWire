# SSQ-LR Competitor Matrix

Status: local novelty guardrail before model measurements.

## Claim Boundary

SSQ-LR is only novel if it isolates recurrent SSM state as the precision target.
It must not be framed as generic weight quantization, activation quantization,
KV-cache quantization, or a serving-speed result before native validation.

## Competitor Classes

| Class | Examples to cite/check | Reviewer risk | Required separation |
|---|---|---|---|
| Weight-only PTQ | GPTQ, AWQ | "This is just another low-bit row." | Freeze weights; perturb/replay state only. |
| Activation smoothing | SmoothQuant-style methods | "Outliers can be moved into weights." | Show recurrent state remains a separate runtime object. |
| Rotation/outlier handling | QuaRot, SpinQuant, HIGGS-style rotations | "State quantization should use known rotations." | Include rotation/protected-outlier controls before claiming a recipe. |
| KV-cache quantization | KIVI, KVQuant, PM-KVQ | "This is KV quantization, not SSM state." | Report SSM state bytes separately from KV bytes. |
| Hybrid sensitivity tools | KL/sensitivity sweeps, AutoQuantize-style allocation | "Sensitivity already identifies layers." | Show a frozen state-only recipe transfers without per-model retuning. |

## Before A Paper Claim

- Define exact state tensor(s), scale metadata, and byte accounting.
- Include BF16 no-op replay, INT8, simulated MXFP4, random same-L2 noise,
  shuffled scales, per-token scales, per-channel scales, and same-byte controls.
- Require paired uncertainty on quality drift, not only reconstruction metrics.
