# Source-Private ICLR Strengthening Deep Dive

- date: `2026-04-30`
- status: strengthening cycle completed with a positive direct Qwen
  target-decoder `n=160` all-control CPU gate
- live branch: source-private product-codebook / semantic-anchor packets with
  model-mediated receiver hardening
- scale rung: medium confirmation on Mac-only hardware

## Cycle Start

1. Current ICLR readiness and distance: stronger scoped positive-method paper,
   but not comfortably ICLR-full. Estimated distance is one clean
   model-mediated receiver gate plus one larger/cross-family systems-aware
   confirmation, or a narrower paper framed explicitly around source-private
   packet communication.
2. Current paper story: a source agent has hidden evidence, sends a tiny
   source-private packet, and a target uses public candidate side information
   to recover the useful residual. Gains must vanish when the source packet is
   zeroed, shuffled, randomized, answer-only, answer-masked, or replaced by
   target/text controls.
3. Exact blocker to submission: reviewers can still argue that the strongest
   results are protocol-shaped side-channel decoding rather than a robust
   model-consumable communication method.
4. Current live branch: learned product-codebook packets plus
   semantic-anchor/scalar Wyner-Ziv packet evidence.
5. Highest-priority gate this cycle: direct Qwen3-0.6B target decoder at
   `n=160` with all source-destroying and same-byte text controls.
6. Scale-up rung: medium Mac CPU receiver confirmation.

## Technical Contributions We Can Defend Today

1. **Source-private packet communication protocol.** The paper defines a
   setting where the target has public context/candidates and the source has
   private evidence. The source sends a rate-capped packet, and the target must
   improve only when that packet carries real source information.

2. **Semantic-anchor and scalar Wyner-Ziv packets.** These are positive packet
   methods under strict controls. The strongest medium row passes `18/18` seed
   and direction rows on the held-out semantic-anchor surface with minimum
   learned-target lift `+0.500` and clean destructive controls.

3. **Learned product-codebook packets.** Product-codebook packets are a learned
   discrete codec over source-private evidence. On the `n=256` remapped
   surface, they pass `8/9` rows, cover all remaps, reach max accuracy `0.598`,
   and have paired CI95 lower bounds of at least `+0.191` vs target and
   `+0.152` vs the strongest destructive control on passing rows.

4. **Systems/hardware traffic accounting.** The paper has a deterministic
   memory-traffic ledger and packet-ISA trace card. It reports raw payload
   bytes, cache-line/DMA rounding, batch packing, source-text exposure, KV/cache
   exposure, TTFT proxies, and a serving SLO envelope. This avoids the
   overclaim that a 2-byte payload is always a 2-byte hardware transaction or
   that Mac proxy latency is production goodput.

5. **Negative-boundary and pruning evidence.** Several plausible latent
   receiver families failed under controls: public anchor-relative receivers,
   simple frozen embeddings, low-rank BGE bottlenecks, trainable JEPA-Q, and
   whole-pool contrastive JEPA-Q. These failures sharpen the claim boundary and
   prevent us from overselling broad latent transfer.

## What Still Needs Work

- **Model-mediated receiver.** The target must consume packets with less
  hand-coded logic. This cycle upgrades direct Qwen target-decoder evidence from
  n64 to n160 on both core and held-out surfaces; product-codebook-specific
  model-mediated decoding remains open.
- **Cross-family robustness.** The paper is strongest for scoped
  source-private packet communication, not broad bidirectional cross-family
  latent transfer.
- **Native systems telemetry.** Mac-local bytes/latency and deterministic
  traffic/SLO ledgers are useful, but NVIDIA/server TTFT, TPOT, goodput, and
  memory traffic remain future gates.
- **Competitor baselines.** We must position against C2C/KVCOMM/KV cache
  compression, prompt compression, gist/soft prompt compression, Q-Former-style
  connectors, and product quantization without claiming wins on their native
  tasks.

## Layman Explanation Of The Current Experiments

Imagine four possible fixes are on the table. The target model can see those
four fixes, but it does not see the hidden test result that tells which fix is
right. The source model sees the hidden test result and is allowed to send only
a tiny message, like two bytes or a few codebook indices. If the target picks
the right fix only when the real tiny message is present, and not when the
message is scrambled or replaced, then the tiny message genuinely carried
useful private evidence.

The product-codebook experiment asks whether the tiny message can be a learned
compressed code rather than a hand-written diagnostic. The target-decoder
experiment asks whether a frozen Qwen model can read the packet and choose the
right candidate, rather than relying only on a deterministic lookup function.
The systems experiments ask how much information actually crosses the boundary
and whether the savings still look meaningful after accounting for cache lines,
DMA bursts, and batching.

## Subagent / Literature Synthesis

The current scout memo is
`references/516_receiver_systems_novelty_scout_refs_20260430.md`.

Key conclusions:

- C2C, DroidSpeak, KVCOMM, and activation-communication work are close to broad
  cross-model latent/cache communication. LatentWire should not claim to beat
  those on their native tasks.
- TurboQuant, KIVI, QJL, and product quantization make it unsafe to claim that
  our codec primitives are inherently novel. The novelty is the source-private
  task-communication setting plus destructive controls and byte/traffic
  accounting.
- BLIP-2/Q-Former and Flamingo motivate the next learned receiver architecture:
  query bottlenecks and gated source injection that preserve the target prior
  when source information is destroyed.
- DiT, flow matching, and consistency models motivate a bounded next method
  branch: a packet-conditioned candidate-score denoiser that refines target
  candidate logits for `1/2/4` steps under packet corruptions and
  source-destroying negatives.
- FlashAttention, vLLM/PagedAttention, DistServe, and Tambe-lab systems themes
  support reporting memory movement, transfer granularity, TTFT/TPOT/goodput,
  and per-query adaptive cost rather than only raw byte counts.

The serving SLO envelope artifact is
`results/source_private_serving_slo_envelope_20260430/`; reference memo:
`references/517_serving_slo_envelope_refs_20260430.md`. It reports `10` rows,
`4` TTFT proxy rows, `0` production-goodput claim rows, packet batch-64 traffic
of `5.0` line bytes/request and `6.0` DMA bytes/request, and explicitly marks
all rows as requiring GPU counters before production serving claims.

## Reviewer-Risk Ranking

1. **Hand-coded receiver objection.** Partly addressed this cycle. The direct
   Qwen3-0.6B target decoder reaches `0.694` on core `n=160` and `0.719` on
   held-out `n=160`; target-only is `0.250`, best control is at most `0.263`,
   and the combined paired CI95 lower bound is `+0.369` versus target and best
   control.
2. **Coded benchmark objection.** Response: keep codebook remaps, learned
   product-codebook packets, semantic-anchor held-out surface, answer controls,
   and matched text baselines visible.
3. **Systems overclaim objection.** Response: emphasize payload/privacy/traffic
   accounting, not production accelerator throughput until NVIDIA runs exist.
4. **Novelty objection.** Response: cite PQ, WZ/Slepian-Wolf, C2C/KVCOMM,
   prompt compression, and connector work as prior ingredients; claim only the
   controlled source-private packet instantiation.
5. **Cross-family latent-transfer objection.** Response: do not claim universal
   latent transfer unless a new receiver branch passes strict cross-family
   controls.

## Next Method Branch If Receiver Gate Is Insufficient

`packet_consistency_denoiser`:

- input: public candidate features, target-prior score, packet code embeddings,
  packet mask/noise level;
- corruptions: zero packet, shuffled packet, random same-byte packet,
  permuted-code packet, answer-masked packet, target-derived packet;
- objective: listwise candidate ranking plus consistency between noisy and
  clean packet states;
- inference: `1`, `2`, and `4` refinement steps;
- pass bar on `n=256`: matched accuracy preserves at least `80%` of the
  deterministic product-codebook lift, destructive controls stay within
  target `+0.03`, and additional refinement steps do not inflate controls.

This is the highest-yield creative branch because it stacks on the packet
format that already works, attacks the hand-coded receiver critique, and gives
a systems-aware compute/accuracy knob.

## What Is Enough For COLM Workshop

The current controlled packet protocol plus semantic-anchor/product-codebook
evidence is likely enough for a scoped workshop paper if written honestly:

- packet method improves over target-only;
- destructive controls stay near target;
- product-codebook packets add a learned discrete codec;
- systems ledger shows payload/private-state savings and caveats;
- negative receiver attempts are documented rather than hidden.

## What Is Needed For Comfortable ICLR Full Paper

- direct model-mediated receiver at `n=160` or `n=256` with paired uncertainty;
- product-codebook-specific model-mediated receiver or a learned receiver with
  equivalent controls;
- final table that includes target-only, scalar WZ, product-codebook,
  same-byte/query-aware text, full-log relay, QJL/TurboQuant/KV byte floors, and
  C2C/KVCOMM comparison notes plus the serving SLO envelope;
- one cross-family or ontology-stress row that does not depend on exact
  phrase overlap, or a very explicit claim boundary;
- native GPU/server systems run when available, reporting TTFT/TPOT/goodput and
  transfer assumptions.

## Blockers Requiring User Help Later

- NVIDIA/server access for production-serving systems telemetry and larger
  model runs.
- Permission and endpoint credentials if we want Qwen3.6 MoE/FP8 or other
  hosted latest-model rows.
- Author decision on the final claim boundary: scoped source-private packet
  communication versus broader latent/model-to-model communication.

## Current Decision

Promote direct Qwen target decoding from smoke to medium Mac-local supporting
evidence with held-out replication. Do not claim fast receiver serving from
this row: p50 CPU latency is about `2.45-2.67 s` per matched condition. The
next exact gate is product-codebook-specific model-mediated target decoding or
the `packet_consistency_denoiser` branch because both attack the remaining
"deterministic receiver" critique while preserving the packet interface that
already works.
