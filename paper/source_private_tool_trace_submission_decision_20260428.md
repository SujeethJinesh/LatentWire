# Source-Private Tool-Trace Submission Decision

- date: `2026-04-28`
- status: scoped-submit decision reached
- live branch: explicit source-private tool-trace packet handoff
- scale rung: large frozen slice plus submission-polish gate
- estimated distance: final human PDF/source read and conference-form checks

## Starting Status

Current ICLR readiness: close to a scoped positive-method submission, but not a
broad latent-transfer paper. The evidence supports a narrow protocol-method
claim.

Current paper story: source-private tool traces can be compressed into explicit
rate-capped diagnostic packets that transfer hidden execution evidence to a
target-side candidate decoder with decoder side information.

Exact blocker: final claim-boundary and provenance decision. The remaining
question is whether to submit the scoped protocol-method paper now or spend a
cycle on the optional `n=160` model-mediated target-decoder scale-up.

## Decision

Proceed with the scoped protocol-method submission path. Do not spend the next
cycle on the optional target-decoder `n=160` scale-up unless the paper claim is
expanded to make an LLM target receiver a main result.

## Rationale

The main claim is not that LatentWire learns an unconstrained target-side latent
bridge. The supported claim is narrower and stronger: source-private tool traces
can be compressed into explicit rate-capped diagnostic packets that transfer
hidden execution evidence to a target-side candidate decoder.

This claim is backed by:

- four frozen 500-example source-model surfaces
- cross-source-model packet emission from Qwen3-0.6B and Phi-3-mini
- strict zero-source, shuffled-source, random same-byte, answer-only,
  answer-masked, target-derived, helper/no-log, and diagnostic-masked controls
- matched-byte text, structured JSON, free-text, full diagnostic, and full-log
  relay rows
- raw-log/no-trace falsification
- byte/token accounting and rate-curve artifacts
- candidate-pool versus selector separation

The target-decoder result remains useful as a smoke ablation showing that the
deterministic decoder is not essential in principle. It is not large enough to
carry the paper's main claim and should not be described as learned latent
transfer or a fully learned target bridge.

## Claim Boundary

This paper demonstrates explicit source-private diagnostic-code communication
for candidate selection under rate limits. It does not claim general raw-log
repair, unconstrained program synthesis, universal cross-model communication,
learned latent alignment, or cache transfer.

## Residual Risk

Reviewers may view the method as synthetic or protocol-shaped. The rebuttal is
that this is deliberate: the benchmark isolates private source evidence, exact
ID parity, byte-rate tradeoffs, source-destroying controls, and selector-vs-pool
accounting. The result should be presented as a clean positive method and
evaluation harness for source-private agent communication, not as a broad
repair benchmark.

## Optional Target-Decoder Scale-Up

If the claim is later expanded to include an LLM target receiver, run the
`n=160` target-decoder scale-up before submission. Promotion should require:

- exact ID parity
- valid prediction rate `1.000`
- matched-target accuracy at least `0.55`
- target-only remains near `0.250`
- matched-best-control delta at least `0.25`
- best source-destroying control no higher than `target + 0.05`
- no promotion unless both core and held-out slices pass

## Next Exact Gate

`source_private_tool_trace_human_pdf_read_20260428`:

- read the compiled PDF/source linearly
- check anonymity/style/conference constraints
- verify table, figure, and artifact references
- decide whether to submit the scoped protocol-method paper or request the
  optional target-decoder scale-up as a separate strengthening run
