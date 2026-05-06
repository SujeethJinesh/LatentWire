# SSQ-LR

SSQ-LR tests whether recurrent SSM state in hybrid reasoners can be quantized
below FP16 without quality loss during long reasoning.

## Current Readiness

Status: **NEW / Mac gates pending**.

Estimated completion:

- **10%** as a positive-method paper: hypothesis and gates are scaffolded.
- **0%** as a systems-result paper: no native GPU state-cache integration or
  benchmark exists.

## Paper Story

Published systems often leave recurrent SSM state at FP16/FP32 while quantizing
weights, activations, or KV cache. SSQ-LR asks whether a sub-FP16 state recipe
can preserve reasoning quality and reduce state memory/bandwidth in hybrid
Mamba-Transformer models.

## Preregistered Gates

Primary preregistration:

- `phase2/preregister_ssq_lr_20260506.md`
- `phase1/competitor_matrix.md`

Gate S1 tests state distribution heterogeneity. Gate S2 tests simulated state
quantization sensitivity. Gate S3 tests cross-model transfer without retuning.

## Current Mac Packet

Synthetic-only packet:

- `phase2/results/ssq_lr_synthetic_s1/`
- decision: `SYNTHETIC_PASS_REAL_STATE_DUMPS_NEXT`

This validates artifact mechanics only. It is not model evidence.

Validate packet shape with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1 \
  --expected-decision-prefix SYNTHETIC
```

Real trace packet requirements are in
`../shared/hybrid_trace_packet_runbook.md`.

Use the explicit architecture hashes in
`../shared/results/hybrid_architecture_maps_20260506/` for packet provenance.
Model-size/cache eligibility is recorded in
`../shared/results/hybrid_model_eligibility_20260506/`.

Validate the first real S1 packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_gate_s1_<YYYYMMDD>_<model_slug> \
  --mode real --project ssq_lr
```

The real checker requires `prefill_end`, `2k_or_end`, `8k_or_end`, and
`final_minus_128` buckets plus at least 12 prompt IDs, unless `config.json`
records a resource-limit note.

## Output Paths

Use:

```text
experimental/ssq_lr/phase2/results/ssq_lr_gate_<gate>_<YYYYMMDD>_<model_slug>/
```

## Local Setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

Reproduce the current synthetic packet:

```bash
./venv_arm64/bin/python -m experimental.ssq_lr.phase2.ssq_lr_synthetic_s1_gate
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1 \
  --expected-decision-prefix SYNTHETIC
jq '.decision, .max_abs_ratio_late_vs_early, .std_ratio_late_vs_early' \
  experimental/ssq_lr/phase2/results/ssq_lr_synthetic_s1/summary.json
```

Expected decision: `SYNTHETIC_PASS_REAL_STATE_DUMPS_NEXT`.

## GPU Rule

No 5090 work until S1--S3 pass and the exact state quantization recipe is
frozen.
