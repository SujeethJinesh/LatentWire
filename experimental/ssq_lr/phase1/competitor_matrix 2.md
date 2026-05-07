# SSQ-LR Competitor Matrix

Status: historical novelty guardrail; current branch evidence is stopped by
S3 transfer failure as of 2026-05-07. Use the README, reviewer pack, and
preregistration closure before treating this matrix as evidence for any live or
GPU-ready claim.

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
| Nemotron-style recurrent-cache deployment | FP16 stochastic rounding; INT16 block-scaled SSM cache controls | "Production hybrid SSM cache precision already has a recipe." | Beat FP16 stochastic rounding with sub-FP16 state bytes under paired NLL and verbosity/length drift; cite `references/754_ssq_lr_nemotron_state_cache_refs_20260507.md`. |

## Before A Paper Claim

- Define exact state tensor(s), scale metadata, and byte accounting.
- Include BF16 no-op replay, INT8, simulated MXFP4, random same-L2 noise,
  shuffled scales, per-token scales, per-channel scales, and same-byte controls.
- Require paired uncertainty on quality drift, not only reconstruction metrics.

## Executable Baseline Columns

Every real S2/S3 result table must include:

| Column | Meaning |
|---|---|
| `bf16_noop` | Replay/dump path with no quantization; catches hook or serialization bugs. |
| `int8_state` | Symmetric INT8 state-only simulation. |
| `fp8_state` | E4M3/E5M2-style state-only simulation if available. |
| `mxfp4_state` | MXFP4-style state-only simulation with stated block size. |
| `random_same_l2` | Same-norm random perturbation to separate quantization structure from noise magnitude. |
| `shuffled_scales` | Same quantized values with mismatched scale assignment. |
| `byte_accounting` | State bytes plus scale/metadata overhead, separated from KV and weights. |

## Source Anchors Checked

- SmoothQuant: `https://arxiv.org/abs/2211.10438`
- KIVI: `https://arxiv.org/abs/2402.02750`
- QuaRot: `https://arxiv.org/abs/2404.00456`
- KL Lens: `https://arxiv.org/abs/2604.13440`
- vLLM hybrid SSM state update docs: `https://docs.vllm.ai/en/v0.12.0/api/vllm/model_executor/layers/mamba/ops/mamba_ssm/`
