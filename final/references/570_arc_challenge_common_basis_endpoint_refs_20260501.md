# ARC-Challenge Shared-Basis Endpoint References, 2026-05-01

## Local Finding

Artifacts:

- `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/`
- `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_validation/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_hashed_test/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_validation/`
- `results/source_private_arc_challenge_seed_stability_20260501_qwen05_anchor_relative_test/`
- `results/source_private_arc_challenge_source_latent_endpoint_gate_20260501_qwen05_bge_validation/`

The shared-basis ARC endpoint uses Qwen2.5-0.5B choice log-likelihood as the
source decision surface and a public hashed text basis as the packet basis. This
is source-computable: both sender and receiver can compute the same basis from
question and candidate strings, so the packet is no longer constructed in a
receiver-only BGE embedding space.

Validation: matched/target/same-byte text is `0.388/0.244/0.348`, with CI95
lower bound versus target `0.070`.

Test: matched/target/same-byte text is `0.344/0.265/0.311`, with CI95 lower
bound versus target `0.044`.

Projection-seed stability passes `5/5` validation seeds and `5/5` test seeds.

The anchor-relative follow-up replaces direct hashed coordinates with
similarities to a deterministic public set of train-split question/candidate
anchors. It also passes `5/5` validation seeds and `5/5` test seeds. Test
matched mean/min/max is `0.344/0.344/0.345`, with minimum same-byte-text lift
`0.033` and minimum CI95 lower bound versus target `0.039`.

## Negative Diagnostic

The direct Qwen-hidden-to-BGE residual endpoint does not pass validation:
matched/target/same-byte text is `0.281/0.244/0.348`, and the CI95 lower bound
versus target is negative. This suggests a naive ridge map from source hidden
summaries into target residual space is not yet a publishable hidden-state
endpoint.

## Primary Sources

- ARC / AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457
- Hugging Face `allenai/ai2_arc`. https://huggingface.co/datasets/allenai/ai2_arc
- Cache-to-Cache communication. https://openreview.net/forum?id=LeatkxrBCi
- KVComm selective KV sharing. https://openreview.net/forum?id=F7rUng23nw
- KVCOMM online cross-context KV-cache communication.
  https://openreview.net/forum?id=yGOytgjurF
- DroidSpeak KV-cache sharing. https://arxiv.org/abs/2411.02820
- Communicating Activations Between Language Model Agents.
  https://arxiv.org/abs/2501.14082
- Relative Representations. https://arxiv.org/abs/2209.15430
- LSTIRP. https://arxiv.org/abs/2406.15057
- QJL. https://arxiv.org/abs/2406.03482
- TurboQuant. https://arxiv.org/abs/2504.19874
- Diffusion Transformers. https://arxiv.org/abs/2212.09748
- LaDiR latent diffusion reasoner. https://arxiv.org/abs/2510.04573
- vLLM serving metrics. https://docs.vllm.ai/en/stable/design/metrics/

## Safe Claim Boundary

Safe: an agreed public basis, including a public anchor-relative coordinate
chart, can make the ARC fixed-byte packet source-computable while preserving the
positive public benchmark result and destructive-control behavior.

Unsafe: claiming that naive hidden-state alignment solves cross-model latent
communication. The local Qwen-hidden-to-BGE endpoint failed this turn.
