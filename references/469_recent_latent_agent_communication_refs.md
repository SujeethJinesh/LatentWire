# Recent Latent Agent Communication References

Date: 2026-04-27

## Status

This memo updates the local reference base with recent primary-source work that
is directly adjacent to the current blocker: converting a real source signal
into target-preserving gains with a systems story. The durable source-surface
ranking still selects `svamp70_live` as the next primary gate, so these papers
do not change the immediate surface. They do change the next method design:
another shallow receiver/router is lower value than a target-preserving latent
injection or side-information decoder with explicit byte/latency accounting.

## Sources and Experiment Implications

### Latent Collaboration in Multi-Agent Systems

- Primary source: `https://arxiv.org/abs/2511.20639`
- Problem it helps with: text relay and natural-language multi-agent baselines
  are now too weak as comparison points for a latent communication paper.
- Mechanism/design idea: shared latent working memory and training-free
  hidden-state transfer among agents.
- Does it change the next experiment? Yes. The next learned branch must report
  token/byte, latency, and end-to-end compute against text relay, because this
  work claims large efficiency gains from latent rather than text exchange.
- Role: baseline, systems framing, and ablation pressure.

Bounded local translation:

- Add a matched-budget baseline where the target receives a fixed-size latent
  sidecar or candidate code, not unlimited source text.
- Record generated tokens, bytes, latency, TTFT when practical, and receiver
  calls. A positive method without a systems tradeoff will look incremental.

### Enabling Agents to Communicate Entirely in Latent Space / Interlat

- Primary sources:
  - `https://arxiv.org/abs/2511.09149`
  - `https://openreview.net/forum?id=rmYbgsehTd`
- Problem it helps with: target preservation and compression of source latent
  messages.
- Mechanism/design idea: transmit last hidden states directly, then compress
  the latent communication while preserving task utility.
- Does it change the next experiment? Yes. If MPS clears and a stronger surface
  exists, the next learned connector should start from a zero-init gate and
  fixed latent budget, then add compression only after it survives source
  controls.
- Role: baseline, inspiration, and reviewer threat model.

Bounded local translation:

- Implement the smallest target-preserving latent injection: source hidden
  summary -> fixed K slots -> zero-init target gate. Compare to target-only
  slots, shuffled-source slots, zero-source slots, and text relay.
- Do not claim latent communication unless shuffled/zero-source controls lose
  the recovered clean IDs and target-correct harm is bounded.

### Communicating Activations Between Language Model Agents

- Primary source: `https://openreview.net/forum?id=W6RPXUUFic`
- Problem it helps with: direct activation combination is a strong competitor
  to any learned bridge or sidecar.
- Mechanism/design idea: pause the receiver model at an intermediate layer,
  combine activations from another model with a function `f`, then continue the
  receiver forward pass.
- Does it change the next experiment? Yes. The next latent branch should include
  a simple activation-combination baseline, even if only same-family first:
  add/mean/linear combine at a matched layer and matched byte budget.
- Role: fair baseline and implementation inspiration.

Bounded local translation:

- For same-family smoke only, test source activation mean/add/linear residual
  injection at one or two layers with source-destroying controls.
- For cross-family, do not direct-add raw coordinates; use projected or
  anchor-relative slots to avoid treating gauge mismatch as a failure of the
  broader idea.

### Thought Communication in Multiagent Collaboration

- Primary sources:
  - `https://openreview.net/forum?id=d671ljgwfY`
  - `https://arxiv.org/abs/2510.20733`
- Problem it helps with: paper framing and interpretability. It argues that
  useful communication can be about shared/private latent factors rather than
  decoded natural language.
- Mechanism/design idea: separate shared and private latent thoughts, then
  route only relevant shared factors.
- Does it change the next experiment? Lightly. It strengthens the case for
  source-difference controls and per-ID diagnostics, but does not override the
  immediate need for a clean source surface.
- Role: theory support, framing, and interpretability design.

Bounded local translation:

- Track which recovered IDs require shared source information, which are target
  priors, and which are target-correct harms.
- For any revived RotAlign/latent-bridge method, add a control that zeros only
  the source-difference lane while preserving shared/target lanes.

## Revised Next-Branch Requirements

The next method branch should satisfy these before promotion:

1. Use the durable primary surface `svamp70_live` from
   `results/durable_source_surface_ranking_20260427/source_surface_ranking.json`.
2. Preserve target behavior by construction, preferably zero-init gated.
3. Carry a fixed latent/sidecar budget and log bytes plus latency.
4. Include activation/latent communication baselines, not only text relay.
5. Include zero-source, shuffled-source, target-only slots, random sidecar, and
   slots-only controls.
6. Replay immediately on canonical `svamp70_holdout` if the live surface passes,
   even though holdout has only two clean source-only IDs.

## Decision

Promote zero-init gated latent side-information as the highest-value learned
method once MPS clears and a source surface is available. Keep candidate
syndrome decoding as the CPU-side conceptual branch only if it moves from
numeric hash artifacts to learned source predicates. Do not spend another cycle
on shallow receiver thresholds without a new source-derived feature.
