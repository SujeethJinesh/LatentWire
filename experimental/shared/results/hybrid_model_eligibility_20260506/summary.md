# Hybrid Model Eligibility

Metadata-only preflight for HybridKernel, SSQ-LR, HORN, and HBSM.
No model weights are downloaded by this packet.

| Model | Type tag | Architecture hash | Safetensors GB | Local weights | Decision |
|---|---|---|---:|---|---|
| ibm-granite/granite-4.0-h-tiny | granitemoehybrid | `bda8fd574ace` | 12.93 | no | `BLOCKED_NOT_CACHED` |
| ibm-granite/granite-4.0-h-small | granitemoehybrid | `8616e9f0b30e` | 59.99 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |
| ibm-granite/granite-4.0-h-small-FP8 | granitemoehybrid | `8616e9f0b30e` | 31.19 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |
| Qwen/Qwen3-Next-80B-A3B-Instruct | qwen3_next | `2d483c7cabad` | 151.49 | no | `GPU_RECOMMENDED_SIZE_NOT_CACHED` |

Decision: `GPU_RECOMMENDED_FOR_LARGE_MODELS`.

A real SSQ-LR/HORN/HBSM trace packet still requires loaded model weights,
hooked recurrent state or boundary activations, and the strict real-packet checker.
