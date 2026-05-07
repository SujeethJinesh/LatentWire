# ThoughtFlow-FP8 Stop/Pivot Decision

Status: **STOP current policy-family tuning on the available saved traces; pivot only with a pre-registered new utility signal.**

## Trigger

The frozen sparse-cache gate committed as `37241e39` evaluated the two current
ThoughtFlow-family candidates without retuning:

- `thoughtflow_saliency_recent`
- `tf_sparse_r0.55_p0.05_m0.12_a2`

The 74-trace CPU sparse-cache slice weakens the family. ThinKV-like is the best
compressed row under fixed nominal-budget accounting.

| Row | NLL | Paired delta vs R-KV-like | Paired delta vs ThinKV-like |
|---|---:|---:|---:|
| thin_kv_like | 3.900 | -0.039 [-0.100,+0.015] | 0.000 |
| tf_sparse_r0.55_p0.05_m0.12_a2 | 3.908 | -0.031 [-0.078,+0.020] | +0.008 [-0.060,+0.085] |
| thoughtflow_saliency_recent | 3.920 | -0.019 [-0.048,+0.006] | +0.020 [-0.030,+0.074] |
| rkv_like | 3.939 | 0.000 | +0.039 [-0.015,+0.099] |

Promotion required a frozen ThoughtFlow row to beat both R-KV-like and
ThinKV-like by at least 0.03 NLL with paired CIs below zero. No row cleared
that rule.

## Decision

Do not tune anchor/recent/phase/math weights further on the current saved trace
set. The current policy family is ruled out as a robust positive method on the
available Mac-local distilgpt2 sparse-cache surface.

This does not kill the sparse-cache probe infrastructure. It kills further
parameter search over the current policy family on these traces.

## Saturated

- Synthetic marker-retention evidence.
- Text-prefix-only retained-context tuning.
- Anchor/recent/phase/math sparse-cache weight sweeps on the current saved
  traces.
- Claims based on phase-marker preservation without continuation-quality
  improvement.

## Still Alive

- CPU sparse-cache pruning as a diagnostic falsification harness.
- Hidden/KV telemetry for explaining eviction bias.
- A future one-shot successor only if its utility signal is specified before
  looking at frozen sparse-cache outcomes.

## Pivot Rule

A future ThoughtFlow-FP8 attempt must first write a pre-registration artifact
that defines one new utility signal, the exact policy transformation, the frozen
evaluation command, and the promotion threshold. It may then be evaluated once
on the frozen sparse-cache probe.

Acceptable signal families must be genuinely new relative to the stopped policy
family. Examples include recurrence-aware future-use proxies, prefix-to-
continuation influence estimates, or cache-state contribution scores. They must
not be another sweep over anchor/recent/phase/math weights on the same traces.

## Next Exact Gate

Either:

1. leave ThoughtFlow-FP8 as a negative/mixed workshop artifact, or
2. pre-register one new utility signal and run exactly one frozen sparse-cache
   evaluation later.

No NVIDIA or FP8 kernel work is justified until a pre-registered successor beats
ThinKV-like and R-KV-like under matched sparse-cache quality with paired
uncertainty.
