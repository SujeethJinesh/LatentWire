# HybridKernel Pre-GPU Threshold Model

Status: **WEAKLY ALIVE; needs unusually high avoidable boundary overhead before GPU work is justified.**

This model asks what fraction of the layer-boundary activation stream must be actual avoidable overhead before a fused boundary kernel could plausibly clear a 3% end-to-end gate.
It is not a GPU benchmark and not a latency result.

## Required Avoidable Boundary Fraction

Assuming 60% recovery of truly avoidable boundary overhead:

| Config | Required avoidable boundary fraction for 3% proxy gain |
|---|---:|
| ibm-granite-4.0-h-small.config.json | 25.0% |
| ibm-granite-4.0-h-tiny.config.json | 25.0% |
| qwen3-next-80b-a3b-instruct.config.json | 10.4% |

## Sensitivity

| Config | Boundary stream fraction | Avoidable overhead | Recovery | Proxy gain | Clears 3%? |
|---|---:|---:|---:|---:|---|
| ibm-granite-4.0-h-small.config.json | 20.0% | 5% | 60% | 0.6% | no |
| ibm-granite-4.0-h-small.config.json | 20.0% | 10% | 60% | 1.2% | no |
| ibm-granite-4.0-h-small.config.json | 20.0% | 20% | 60% | 2.4% | no |
| ibm-granite-4.0-h-small.config.json | 20.0% | 30% | 60% | 3.6% | yes |
| ibm-granite-4.0-h-small.config.json | 20.0% | 50% | 60% | 6.0% | yes |
| ibm-granite-4.0-h-tiny.config.json | 20.0% | 5% | 60% | 0.6% | no |
| ibm-granite-4.0-h-tiny.config.json | 20.0% | 10% | 60% | 1.2% | no |
| ibm-granite-4.0-h-tiny.config.json | 20.0% | 20% | 60% | 2.4% | no |
| ibm-granite-4.0-h-tiny.config.json | 20.0% | 30% | 60% | 3.6% | yes |
| ibm-granite-4.0-h-tiny.config.json | 20.0% | 50% | 60% | 6.0% | yes |
| qwen3-next-80b-a3b-instruct.config.json | 47.9% | 5% | 60% | 1.4% | no |
| qwen3-next-80b-a3b-instruct.config.json | 47.9% | 10% | 60% | 2.9% | no |
| qwen3-next-80b-a3b-instruct.config.json | 47.9% | 20% | 60% | 5.8% | yes |
| qwen3-next-80b-a3b-instruct.config.json | 47.9% | 30% | 60% | 8.6% | yes |
| qwen3-next-80b-a3b-instruct.config.json | 47.9% | 50% | 60% | 14.4% | yes |

## Decision

This weakens Mac-only implementation. Granite requires roughly 25% of boundary traffic to be genuinely avoidable at 60% recovery to clear a 3% proxy gain.
Qwen3-Next is closer, requiring roughly 10.4%, but its linear-attention/Gated-DeltaNet boundary is less directly matched to the Granite Mamba2 fusion idea.
Before NVIDIA GPU work, the only useful local action is to prepare the profiler runbook and exact counters to verify avoidable boundary overhead.
