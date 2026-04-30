# Anti-Lookup Label-Blind Scale-Up References

Purpose: support the `n=160` label-blind anti-lookup scale-up and keep the
paper framing honest: the live method is source-private communication with
decoder side information, not protocol-free latent semantic transfer.

## Sources

### Slepian and Wolf, 1973

- source: https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
- blocker helped: prevents overclaiming a tiny coded packet as a new coding
  principle.
- mechanism/design idea: a source can send a compact syndrome/bin when the
  decoder has correlated side information.
- next experiment change: keep exact-ID, label-remap, code-remap,
  order-permutation, and label-blind destructive controls in every promoted
  packet result.
- role: theory support and ablation framing.

### Wyner and Ziv, 1976

- source: https://www.itsoc.org/publications/papers/the-rate-distortion-function-for-source-coding-with-side-information-at-the-decoder
- blocker helped: clarifies that target-side public candidate state is part of
  the communication channel, not a nuisance detail.
- mechanism/design idea: report task distortion versus packet rate when the
  decoder has side information unavailable to the encoder.
- next experiment change: emphasize byte/accuracy frontiers and label-blind
  collapse, not just a single positive 2-byte row.
- role: theory support and paper framing.

### Tang, Yang, and Gunduz, 2024

- source: https://arxiv.org/abs/2405.13483
- blocker helped: connects private-evidence communication to recovering a
  latent task variable rather than reconstructing the full source trace.
- mechanism/design idea: optimize packet utility for target posterior or
  candidate margin distortion.
- next experiment change: learned receivers should optimize candidate decision
  utility under destructive controls, not source-log reconstruction.
- role: theory support and learned-objective inspiration.

### Whang et al., 2021

- source: https://arxiv.org/abs/2106.02797
- blocker helped: learned source coding with decoder side information is prior
  work, so novelty must come from the LLM/agent side-information setting and
  controls.
- mechanism/design idea: conditional VQ-style codes as a learned syndrome
  baseline.
- next experiment change: a learned receiver branch should include
  shuffled-code and codebook-remap controls.
- role: baseline and ablation inspiration.

### DeepSC, 2021

- source: https://arxiv.org/abs/2006.10685
- blocker helped: semantic communication is established terminology.
- mechanism/design idea: evaluate task utility rather than exact message
  reconstruction.
- next experiment change: no immediate gate change; use for wording and metric
  framing.
- role: framing.

### C2C, 2025

- source: https://arxiv.org/abs/2510.03215
- blocker helped: non-text LLM-to-LLM communication has close prior art at a
  much higher-rate internal-state interface.
- mechanism/design idea: project/fuse source KV cache into target KV cache with
  learned gates.
- next experiment change: keep C2C as a quality/latency baseline or adjacent
  high-rate comparator, not a same-byte control.
- role: baseline and novelty boundary.

### KVComm, 2025

- source: https://arxiv.org/abs/2510.03346
- blocker helped: selective KV sharing is a natural systems comparator for
  model-to-model communication.
- mechanism/design idea: transmit selected KV layers/pairs instead of compact
  source-private evidence packets.
- next experiment change: systems tables should include KV byte-floor and
  selective-KV accounting.
- role: baseline and systems framing.

## Takeaway

The label-blind scale-up does not claim a new coding theorem. Its contribution
is an LLM/agent communication protocol and benchmark control suite showing that
source-private packets help only when target-side public side information is
available, and that the gain disappears under label-blind removal of the public
repair-key table.
