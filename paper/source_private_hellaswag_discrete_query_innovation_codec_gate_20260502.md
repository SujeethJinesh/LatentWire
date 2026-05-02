# HellaSwag Discrete Query Innovation Codec Gate

- artifact: `results/source_private_hellaswag_discrete_query_innovation_codec_gate_20260502/`
- script: `scripts/build_source_private_hellaswag_discrete_query_innovation_codec_gate.py`
- test: `tests/test_build_source_private_hellaswag_discrete_query_innovation_codec_gate.py`
- references: `references/625_hellaswag_discrete_query_innovation_codec_refs_20260502.md`

## Question

Can a source-private, decoder-conditioned fixed-query encoder convert
HellaSwag TinyLlama source-score complementarity into a positive learned
one-byte packet?

In plain terms: the source model privately looks at the four answer options,
compresses what it thinks the target model is missing into one byte, and the
target model tries to use that byte to fix its own answer.

## Protocol

The gate keeps the compact `1B` raw / `4B` framed packet contract. During
calibration, the source encoder sees official train labels and a target-side
residual objective. At validation time it receives only TinyLlama source score
features, summarizes the four candidate tokens with fixed query vectors,
predicts a four-way residual, vector-quantizes that residual, and transmits
`cluster_id * 4 + source_candidate`.

The target-side decoder receives only Qwen candidate-score features and the
discrete source code. No source text, KV/cache, raw hidden vectors, raw scores,
or query embeddings cross the boundary.

## Result

The gate fails promotion.

- official-train calibration rows: `1487`
- fit/dev split: `1115/372`
- validation rows: `10042`
- packet-only accuracy: `0.619199`
- default accuracy: `0.605158`
- default delta vs packet-only: `-0.014041`
- default CI95 low vs packet-only: `-0.018124`
- best eval-scout accuracy: `0.619399`
- best eval-scout delta vs packet-only: `+0.000199`
- best eval-scout CI95 low vs packet-only: `-0.000398`
- packet-or-Qwen-target oracle delta: `+0.058355`
- default oracle capture fraction: `-0.240614`
- pass gate: `false`

The official-dev-selected default hurts all five contiguous validation blocks.
The best validation scout is effectively tied with packet-only and does not
survive uncertainty.

## Controls

The destructive controls mostly collapse far below packet-only, so the problem
is not a missing negative control; the real method simply does not help.

- row-shuffle source code: `-0.350329`
- source-feature shuffle before encoding: `-0.363175`
- codebook permutation mismatch: `-0.117307`
- random same-byte code: `-0.330910`
- random cluster preserving packet: `-0.085043`
- Qwen-derived code: `-0.140609`
- candidate-only code: `-0.002689`
- label-permutation decoder: `-0.356602`
- compact candidate-only decoder: `0.000000`

## Systems Readout

The systems side remains strong as byte accounting, but not as an end-to-end
quality win for this learned codec.

- cached batch-1 encode/decode p50: `36.21us` / `63.92us`
- cached batch-256 encode/decode p50: `2.62us` / `0.82us`
- FP16 one-token KV byte floor vs 4B framed packet: `3072x`
- KVComm 30% FP16 byte floor vs framed packet: `921.6x`
- QJL 1-bit byte floor vs framed packet: `192x`
- TurboQuant 3.5-bit byte floor vs framed packet: `672x`

These are conservative byte/exposure floors only. They are not native TTFT,
TPOT, HBM, goodput, or quality measurements.

## Decision

This branch is weakened. The result says that target-conditioned residual
training plus a fixed-query discrete code is still not enough when it is built
only from the current TinyLlama source-score surface. Combined with the
switch-observability gate, this should stop further HellaSwag source-score
selector or source-score codec tuning.

For ICLR, the next positive-method branch must change source information rather
than retune this one. The highest-value options are:

1. train a true hidden/activation connector or query bottleneck on NVIDIA,
2. run a strict cached hidden/PQ residual code only if hidden features are
   available across the full validation surface,
3. otherwise cut HellaSwag receiver-improvement and use HellaSwag as
   complementarity/headroom, negative-ablation, and systems-rate evidence.
