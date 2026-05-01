# Candidate-Conditioned Packet Builder LOO Hybrid References

Date: 2026-05-01

## Local Result

The promoted method is a source-prioritized innovation packet:

```text
packet_vector = source_to_candidate_ridge(source_atoms) + 0.75 * source_atoms
```

It is evaluated with `leave_one_family_out_public` packet-builder calibration,
12-byte packets, n512 frozen slices, seeds 47/53/59, and the same strict
source-destroying controls used by the live candidate-local receiver. It passes
`9/9` rows, with candidate accuracy `0.625`, live source packet `0.500`, target
`0.250`, and max best destructive control `0.258`.

## Primary Sources And Novelty Boundaries

- Slepian-Wolf and Wyner-Ziv side-information coding motivate the setup where a
  sender communicates a compressed source to a decoder that already has side
  information. Our packet is not a theorem-level syndrome code, but the
  source-prioritized residual plays the same practical role: preserve exact
  source bits that the decoder side information may not recover from the
  learned map alone.
  - https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
  - https://ieeexplore.ieee.org/document/1055638

- Error-correcting output codes motivate candidate-code robustness: small
  symbolic packets can be decoded against a candidate codebook, and wrong-code
  controls are necessary to prove the channel is not just a label lookup.
  - https://arxiv.org/abs/cs/9501101
  - https://www.jmlr.org/papers/v1/allwein00a.html

- DomainBed and WILDS motivate leave-one-domain/family-out evaluation. This
  local LOO gate is stricter than public eval-disjoint calibration for the
  packet builder, but still weaker than train-only real-domain generalization.
  - https://arxiv.org/abs/2007.01434
  - https://arxiv.org/abs/2012.07421

- Cross-model and multi-agent communication baselines mostly transmit internal
  activations or KV cache material. That is a different threat model from this
  source-private packet: C2C/KVComm/KVCOMM should be compared as high-rate
  internal-state baselines, not as identical protocols.
  - https://arxiv.org/abs/2510.03215
  - https://arxiv.org/abs/2510.03346
  - https://arxiv.org/abs/2510.12872
  - https://arxiv.org/abs/2501.14082

- TurboQuant, KIVI, KVQuant, and QJL are systems/byte-floor comparators for
  cache/vector transport. They compress large internal vectors; LatentWire sends
  tiny source-private sparse records. Native NVIDIA/vLLM rows are still needed
  before making serving-speed claims.
  - https://arxiv.org/abs/2504.19874
  - https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
  - https://arxiv.org/abs/2402.02750
  - https://arxiv.org/abs/2401.18079
  - https://arxiv.org/abs/2406.03482
  - https://arxiv.org/abs/2309.06180

## Reviewer Risk

The LOO hybrid is meaningfully stronger than the public-disjoint learned packet
because the packet builder does not see the evaluation family while fitting.
However, the candidate dictionary still uses public eval-disjoint calibration,
and the benchmark is synthetic. The paper should claim a positive source-private
candidate-side-information packet method, not universal model-to-model latent
reasoning.
