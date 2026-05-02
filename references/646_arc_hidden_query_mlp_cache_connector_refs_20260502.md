# ARC Hidden/Query MLP Cache Connector References

Date: 2026-05-02

## Purpose

This memo supports the negative ARC hidden/query MLP cache-to-packet connector
gate and tightens the novelty boundary for the next learned connector branch.

## Primary Sources Checked

- BLIP-2 / Q-Former: Li et al. propose frozen vision and language backbones
  bridged by a lightweight Querying Transformer, which is the closest positive
  precedent for a learned query bottleneck between heterogeneous
  representation spaces. Source: https://arxiv.org/abs/2301.12597
- Flamingo / Perceiver Resampler: Alayrac et al. use a Perceiver-style
  bottleneck plus gated cross-attention to condition a frozen language model on
  non-text features. Source: https://arxiv.org/abs/2204.14198
- Perceiver IO: Jaegle et al. give the general latent-query architecture for
  mapping high-dimensional inputs through a small set of latent queries.
  Source: https://arxiv.org/abs/2107.14795
- Prefix-Tuning: Li and Liang optimize continuous prefixes while keeping the
  language model frozen. This is the obvious baseline and claim-boundary for
  any soft-token connector. Source: https://arxiv.org/abs/2101.00190
- Cache-to-Cache (C2C): direct semantic communication through projected/fused
  KV-cache state is the closest multi-LLM competitor and must remain distinct
  from LatentWire's fixed-byte packet/exposure claim. Source:
  https://openreview.net/forum?id=LeatkxrBCi
- KVComm and KVCOMM: selective KV sharing and cross-context KV reuse are
  systems-side cache communication baselines, not source-private fixed-byte
  packet methods. Sources: https://arxiv.org/abs/2510.03346 and
  https://arxiv.org/abs/2510.12872
- QJL: quantized Johnson-Lindenstrauss KV-cache compression is a mandatory
  compression/systems baseline for any native systems claim involving cache
  traffic. Source: https://arxiv.org/abs/2406.03482
- TurboQuant: online vector quantization/KV-cache compression is a required
  byte/quality baseline for future systems tables, especially once NVIDIA
  measurements exist. Source: https://arxiv.org/abs/2504.19874
- SAE universal feature spaces: sparse autoencoders may provide a better
  common-language representation than dense mean hidden/query features, but
  they are not tested by the current mean-cache gate. Source:
  https://arxiv.org/abs/2410.06981

## Boundary For This Gate

The new ARC gate is not equivalent to prefix tuning because it does not learn a
static task prompt. It learns a per-example source-conditioned map from cached
TinyLlama hidden/query evidence into a fixed-byte packet. However, it is also
not a full Q-Former, Flamingo, C2C, or soft-prefix method because it only has
per-choice mean hidden/query caches and never runs a target LM with learned
soft tokens or fused KV state.

The result should therefore be written as a negative low-data cache-to-packet
proxy:

- it rules out the current TinyLlama mean hidden/query cache family;
- it does not rule out tokenwise query bottlenecks, soft-prefix transfer,
  sparse-crosscoder feature spaces, or KV/cache fusers;
- it reinforces that the next positive-method attempt needs stronger training
  signal and stricter baselines rather than another shallow map.

## Reviewer-Facing Comparison

Against prefix/prompt tuning, LatentWire must show that the connector is
source-conditioned per example and rate-limited, not merely a learned static
target prompt.

Against C2C/KVComm/KVCOMM, LatentWire must show a different exposure contract:
tiny packets cross the model boundary, not raw or compressed KV-cache state.
The current native systems claim remains accounting-only until matched GPU
serving rows exist.

Against QJL/TurboQuant, LatentWire must compare the transmitted bytes and
source-state exposure at the same accuracy/SLO. Quantization methods compress
model memory; LatentWire is trying to send a task-specific source-private
message.

Against SAE/common-feature-space work, LatentWire should not claim a universal
latent language from the current result. The negative MLP cache gate suggests
mean hidden/query vectors are insufficient; sparse/tokenwise feature spaces
remain a plausible future branch.
