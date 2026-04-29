# Source-Private Novelty Stance After Learned Receiver Holdout Failure

Date: `2026-04-29`

## Current Paper Status

The current package is a scoped positive-method candidate, not yet a broad
cross-family latent-transfer paper. The strongest story is source-private,
extreme-rate evidence packets decoded by an endpoint label-strict or
target-preserving receiver under source-destroying controls. The exact blocker
for broader submission is heldout-family generalization: the learned
target-preserving receiver is stable on same-distribution repeats but fails the
core-to-holdout family gate.

## Current Novelty Stance

The method should be positioned as decoder-side-information communication for
private evidence, not as first general LLM latent communication. C2C,
activation communication, KVComm/KVCOMM, LatentMAS, vector translation, and
prompt compression already occupy adjacent or direct spaces. The defensible
novelty is the combination of:

- source-private evidence observed by the sender but unavailable to the target;
- a byte-level extreme-rate packet rather than high-dimensional activation or
  KV transfer;
- an endpoint label-strict / target-preserving receiver contract;
- destructive source controls, matched-byte text controls, and rate frontier
  accounting;
- an explicit negative boundary: same-distribution learned receivers pass,
  heldout-family learned receivers currently fail.

## Sources And Impact

### 1. C2C: Cache-to-Cache

Primary source: https://arxiv.org/abs/2510.03215

- Blocker helped: prevents any claim of first direct semantic LLM
  communication.
- Mechanism idea: project and fuse source KV cache into target KV cache with
  learned gating over useful target layers.
- Next experiment impact: compare only as a higher-rate internal-state
  communication baseline; report our method at the far-left byte point.
- Role: direct competitor / novelty threat.

### 2. Communicating Activations Between Language Model Agents

Primary source: https://openreview.net/forum?id=W6RPXUUFic

- Blocker helped: activation-level inter-agent communication is already a
  published baseline.
- Mechanism idea: pause receiver computation, combine sender activation with
  receiver activation, then continue the receiver forward pass.
- Next experiment impact: emphasize endpoint/API-compatible packet decoding
  and no activation injection in current claims.
- Role: direct activation-communication competitor.

### 3. KVComm: Selective KV Sharing

Primary source: https://openreview.net/forum?id=F7rUng23nw

- Blocker helped: KV pairs are already proposed as an efficient inter-LLM
  communication medium.
- Mechanism idea: layer-wise KV selection by attention importance and a
  Gaussian prior; transmit only informative KV pairs.
- Next experiment impact: if cache baselines are added, byte accounting must
  separate selected KV tensors from source-private packets.
- Role: direct KV-communication competitor.

### 4. KVCOMM: Online Cross-context KV-cache Communication

Primary source: https://arxiv.org/abs/2510.12872

- Blocker helped: systems reviewers may view packet handoff as another KV reuse
  system unless the threat model is clear.
- Mechanism idea: reuse and align KV caches for overlapping contexts using an
  online anchor pool that estimates prefix-induced cache deviations.
- Next experiment impact: future systems table should separate TTFT/prefill
  reuse from private-evidence selection accuracy.
- Role: systems neighbor / serving baseline.

### 5. LatentMAS

Primary source: https://arxiv.org/abs/2511.20639

- Blocker helped: latent multi-agent collaboration already claims text-free
  latent collaboration.
- Mechanism idea: autoregressive latent thoughts and shared latent working
  memory transfer internal representations among agents.
- Next experiment impact: do not claim broad latent collaboration; keep the
  live claim source-private, rate-capped, and control-heavy.
- Role: adjacent latent-agent communication competitor.

### 6. Direct Semantic Communication via Vector Translation

Primary source: https://arxiv.org/abs/2511.03945

- Blocker helped: learned vector translation between LLM representation spaces
  is close prior art for latent transfer.
- Mechanism idea: train dual-encoder translators between model representation
  spaces, then inject translated vectors at controlled blending strength.
- Next experiment impact: heldout-family failure should be reported as a real
  boundary; next receiver should use family-invariant or anchor-relative packet
  features, not raw candidate coordinates alone.
- Role: direct latent-transfer comparison / mechanism caution.

### 7. LLMLingua

Primary source: https://aclanthology.org/2023.emnlp-main.825/

- Blocker helped: compressed text can beat naive text relay at larger budgets.
- Mechanism idea: budget-controlled coarse-to-fine prompt compression with
  token-level iterative compression and distribution alignment.
- Next experiment impact: keep matched-byte structured/free-text controls and
  report the full rate curve where text becomes oracle at larger budgets.
- Role: prompt-compression baseline.

### 8. Slepian-Wolf Coding

Primary source: https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources

- Blocker helped: source-private packet decoding with target context has a
  classical distributed-source-coding analogue.
- Mechanism idea: transmit a syndrome/bin index that the decoder resolves using
  correlated side information.
- Next experiment impact: train/evaluate packets as syndromes over target
  candidate pools with codebook-remap and source-destroying controls.
- Role: theory support.

### 9. Wyner-Ziv Source Coding

Primary source: https://doi.org/10.1109/TIT.1976.1055508

- Blocker helped: decoder side information is not itself novel; our novelty
  must be the LLM evidence-packet instantiation and evaluation contract.
- Mechanism idea: lossy source coding when side information is available at the
  decoder but not the encoder.
- Next experiment impact: use accuracy/log-loss versus bytes as a
  rate-distortion curve rather than a single accuracy row.
- Role: theory support / framing constraint.

### 10. Distributed Indirect Source Coding With Decoder Side Information

Primary source: https://arxiv.org/abs/2405.13483

- Blocker helped: task-oriented source coding with decoder side information
  closely matches "recover the useful latent/candidate" rather than reconstruct
  the full source trace.
- Mechanism idea: independent encoders send compact messages; decoder combines
  messages and side information to recover a correlated latent variable under a
  distortion constraint.
- Next experiment impact: learned receiver should optimize candidate posterior
  distortion/margin and cross-family calibration, not raw packet
  reconstruction.
- Role: theory and learned-receiver objective.

## Decision

Current deterministic packet evidence is strong enough for a scoped method
claim. The learned target-preserving receiver should be described as a promising
successor branch that passes same-distribution stability but is not yet
paper-ready because heldout-family transfer fails. The highest-priority next
gate is a family-invariant learned syndrome/anchor-relative receiver that keeps
the label-strict endpoint contract and passes heldout-family with paired
uncertainty.

## Concise Novelty Claim

LatentWire contributes a source-private, extreme-rate evidence-packet protocol:
the source sends only a few bytes of hidden diagnostic evidence, while a
label-strict or target-preserving receiver uses its public candidate/context
side information to decode the packet. Unlike C2C, activation communication,
KV sharing, LatentMAS, or prompt compression, the claim is not broad latent
state transfer; it is controlled decoder-side-information communication at the
far-left rate point, with destructive source controls and a documented
heldout-family failure boundary for learned receivers.

## Top Reviewer Risks

1. Coded-label risk: the packet may look like a task-specific codebook rather
   than semantic communication.
2. Related-work risk: C2C/KVComm/activation/LatentMAS already own broad
   non-text communication claims.
3. Generalization risk: the learned receiver is same-distribution stable but
   currently fails heldout-family transfer.
