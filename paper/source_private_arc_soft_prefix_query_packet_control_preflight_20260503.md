# ARC Soft-Prefix Query Packet-Control Preflight

Date: 2026-05-03

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: fixed-byte source-private packets, public ARC/OpenBookQA
  benchmark gates, destructive controls, and systems byte/exposure accounting
  are the defensible core.
- Exact gap: a positive source-necessary learned connector is still missing.
  The stricter ARC n32 target-loss soft-prefix/query preflight below fails.

## Contributions Being Protected

1. Fixed-byte source-private packet protocol with source-destroying controls.
2. Public-basis benchmark gates plus a falsification ladder that prevents
   overclaiming latent communication.
3. Systems byte/exposure accounting against KV/cache transfer baselines.

The candidate fourth contribution, a positive learned soft-prefix/query
connector, is still not evidence-backed.

## Gate

Updated script:
`scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`

New strict controls:

- `source_free_prefix`
- `packet_only_source_index`
- `qwen_substituted_packet`
- `--qwen-source-cache-path`

Artifact:
`results/source_private_arc_openbookqa_soft_prefix_preflight_20260503_arc_hf_hidden_score_public_innovation_candidate_pool_residual_n32_cpu_label_choice_packet_controls/`

Command:

```bash
PYTHONUNBUFFERED=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 ./venv_arm64/bin/python scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py \
  --output-dir results/source_private_arc_openbookqa_soft_prefix_preflight_20260503_arc_hf_hidden_score_public_innovation_candidate_pool_residual_n32_cpu_label_choice_packet_controls \
  --row-limit 32 \
  --fit-fraction 0.5 \
  --source-feature-mode hf_choice_hidden_score_public_innovation_candidate_pool_residual \
  --source-token-pool-size 8 \
  --prefix-len 2 \
  --hidden-dim 8 \
  --epochs 1 \
  --lr 0.003 \
  --weight-decay 0.001 \
  --seed 23 \
  --bootstrap-samples 500 \
  --continuation-mode label_and_choice \
  --source-device auto_cpu \
  --target-device auto_cpu \
  --train-device auto_cpu \
  --dtype float32 \
  --target-attn-implementation eager \
  --local-files-only true \
  --same-byte-budget 12 \
  --target-max-length 192
```

## Result

ARC validation n32 CPU preflight, `label_and_choice` continuation:

- pass gate: `False`
- fit/eval rows: `16 / 16`
- matched soft-prefix accuracy: `0.125`
- target-only accuracy: `0.375`
- source-free prefix accuracy: `0.375`
- zero-source accuracy: `0.5625`
- same-byte visible text accuracy: `0.500`
- packet-only source-index accuracy: `0.375`
- Qwen-substituted packet accuracy: `0.375`
- label-shuffled accuracy: `0.1875`
- matched minus best-control accuracy: `-0.4375`
- matched minus best-control margin: `-0.9514`
- paired matched-vs-zero-source mean/CI95: `-0.4375 [-0.6875, -0.1875]`
- paired matched-vs-packet-only mean/CI95: `-0.25 [-0.5625, 0.0625]`
- runtime: `1507.6s`
- peak RSS: `18.25 GiB`

The source feature was a rank-3 candidate pool:
`hf_choice_hidden_score_public_innovation_candidate_pool_residual`, pool size
`8`, using Qwen2.5-0.5B-Instruct hidden candidate features plus a cached source
selection score. The train-only public ridge explained `0.9963` of fit hidden
variance before the residual pool was passed to the query connector.

## Decision

This exact target-loss soft-prefix/query connector is weakened enough to stop
widening on Mac. It does not show source necessity. More prefix capacity or a
larger hidden pool is lower priority than changing the communicated object.

Promote as next method branch:

1. conditional innovation packets/features: source evidence minus target-side
   public/candidate/logit side information;
2. sparse/common-feature packet bus with feature-ID permutation controls;
3. quantized attention-resonance sketches only after the conditional-innovation
   packet has a positive small-slice signal.

Do not claim a positive learned connector in the COLM/ICLR story from this
gate.

## Lay Explanation

This experiment tried to teach a tiny translator to turn hidden clues from the
source model into soft tokens for the target model. It failed because the target
did better when the source clue was removed or replaced by simpler controls.
That means this soft-prefix translator is not yet a real communication channel;
it is mostly adding noise on this slice.
