# ARC-Challenge Seed-Stability References, 2026-05-01

## Local Finding

Artifact:
`results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_validation/`
and
`results/source_private_arc_challenge_seed_stability_20260501_qwen05_bge_test/`.

The fixed `12B` ARC-Challenge packet result is stable across five packet
projection/random-control seeds (`47/53/59/61/67`) when the source-choice cache
is held fixed from the answer-key-forbidden Qwen2.5-0.5B source run.

Validation passes `5/5` seeds. The matched packet accuracy is `0.381-0.385`
versus target-only `0.244`, same-byte structured text `0.348`, and maximum
candidate-derangement accuracy `0.197`.

Test passes `5/5` seeds. The matched packet accuracy is `0.341-0.346` versus
target-only `0.265`, same-byte structured text `0.311`, and maximum
candidate-derangement accuracy `0.216`. The minimum lift over target is
`0.076`, and the minimum paired CI95 lower bound versus target is `0.038`.

## Why This Matters

The prior ARC row could be criticized as one lucky random projection. This
follow-up varies the projection and random packet controls while keeping the
source choices fixed. It therefore isolates the packet-code robustness of the
public ARC bridge.

This does not remove the main ICLR caveat: the source scorer is still a local
Qwen choice log-likelihood bridge, not a native cross-model endpoint.

## Primary Sources

- Clark et al., 2018. "Think you have Solved Question Answering? Try ARC, the
  AI2 Reasoning Challenge." https://arxiv.org/abs/1803.05457
- Hugging Face dataset card for `allenai/ai2_arc`.
  https://huggingface.co/datasets/allenai/ai2_arc
- C2C / Cache-to-Cache communication. https://openreview.net/forum?id=LeatkxrBCi
- KVComm selective KV sharing. https://openreview.net/forum?id=F7rUng23nw
- KVCOMM online cross-context KV-cache communication.
  https://openreview.net/forum?id=yGOytgjurF
- DroidSpeak KV cache sharing for cross-LLM communication.
  https://arxiv.org/abs/2411.02820
- Communicating Activations Between Language Model Agents.
  https://arxiv.org/abs/2501.14082
- vLLM serving benchmark metrics for TTFT, TPOT, ITL, end-to-end latency, and
  goodput. https://docs.vllm.ai/en/v0.9.1/api/vllm/benchmarks/serve.html
- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  https://arxiv.org/abs/2504.19874

## Related-Work Scan

The closest communication systems transmit or fuse source internals:
C2C projects and merges source KV cache; KVComm/KVCOMM selectively share KV
pairs or cross-context cache state; DroidSpeak reuses embeddings/KV across
related LLMs; activation communication injects another LM's hidden activations.
These are important competitors, but they do not use a fixed `12B`
source-private packet decoded against public multiple-choice candidates.

The closest systems/quantization work compresses vectors or KV state:
TurboQuant and QJL-style methods motivate randomized projections and residual
correction, but they optimize high-dimensional vector/KV compression rather
than source-private candidate disambiguation under shuffle/random/text and
derangement controls.

## Safe Claim Boundary

Safe: the ARC public transfer is projection-seed-stable under the fixed-byte
source-private packet protocol and destructive controls.

Unsafe: claiming general latent transfer, native GPU systems superiority, or
that this replaces C2C/KV-cache communication. Those claims still require a
cleaner endpoint and native systems rows.
