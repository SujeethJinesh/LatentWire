# References: ICLR/COLM Readiness Update

This memo supports
`paper/source_private_iclr_colm_readiness_update_20260502.md` and the dated
evidence bundle at `results/source_private_iclr_evidence_bundle_20260502/`.

## Primary Sources Checked

1. HellaSwag
   - https://arxiv.org/abs/1905.07830
   - Relevance: hard adversarial commonsense continuation benchmark used for
     the latest packet, headroom, and hidden-code gates.

2. Cache-to-Cache (C2C)
   - https://arxiv.org/abs/2510.03215
   - Relevance: closest direct inter-LLM communication competitor; it projects
     and fuses source KV-cache with the target model. Boundary: LatentWire's
     current packet transmits a task-level fixed-byte record, not source KV.

3. KVComm
   - https://openreview.net/forum?id=F7rUng23nw
   - Relevance: selective KV-pair/layer sharing baseline for inter-LLM
     communication. Boundary: LatentWire must compare bytes/exposure and
     native systems behavior, not only accuracy.

4. TurboQuant and QJL
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2406.03482
   - Relevance: vector/KV quantization baselines and byte-floor comparators.
     Boundary: these optimize vector/cache distortion and attention inner
     products, while the PQ HellaSwag gate optimized downstream candidate
     decision accuracy.

5. Sparse autoencoders and crosscoders
   - https://arxiv.org/abs/2309.08600
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Relevance: plausible shared-dictionary/common-basis route for a future
     learned connector. Boundary: current PQ codebook is not an SAE/crosscoder
     dictionary and should not be framed as interpretability evidence.

6. Prefix tuning and query bottlenecks
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2301.12597
   - Relevance: continuous virtual-token and learned query baselines for future
     connector methods. Boundary: the current source-private packet does not
     insert continuous prefix/query embeddings into the receiver.

7. vLLM/PagedAttention
   - https://arxiv.org/abs/2309.06180
   - Relevance: native serving substrate for the needed NVIDIA systems rows.

## Uniqueness Boundary

Safe novelty angle: a source-private fixed-byte task packet with explicit
source-destroy controls and systems byte/exposure accounting.

Unsafe novelty angle: claiming robust cross-model latent reasoning from the
current HellaSwag hidden-code branch. The latest PQ gate fails versus
packet-only, and prior branch-kill gates already weakened source-score,
selector, and anchor-relative common-basis variants.

## Decision

For COLM, frame the paper as a protocol/evaluation/systems contribution with
honest negative latent-code evidence. For ICLR, require a positive learned
receiver/common-basis connector before claiming cross-model latent reasoning.
