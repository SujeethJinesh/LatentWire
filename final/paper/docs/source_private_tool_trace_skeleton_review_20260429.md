# Source-Private Tool-Trace Skeleton Review

- date: `2026-04-29`
- status: skeptical review completed
- live branch: explicit source-private tool-trace packet handoff
- scale rung: paper-skeleton review

## Review Verdict

The paper is draftable as a narrow source-private evidence-communication paper.
It is not ready as a broader latent-transfer, raw-log reasoning, or universal
cross-model communication paper.

## Top Rejection Risks

1. Overclaiming beyond the evidence. The title, abstract, and introduction must
   avoid inviting a latent-transfer or general cross-model-communication read.
2. Structured text relay. The paper must foreground the rate curve: same-byte
   JSON/free-text relays fail at `2` bytes but become oracles at `32` bytes.
3. Benchmark construction artifact. Candidate metadata exposes the diagnostic
   mapping, so the contribution must be framed as candidate selection with
   decoder side information.
4. Systems value. Bytes/token savings are strong, but parsing validity and
   protocol-design cost must be reported, especially Qwen3 validity
   `0.776-0.864`.
5. Novelty. A deterministic protocol decoder may look like coded-label lookup;
   one learned or LLM-mediated target-decoder row would reduce this risk.

## Paper-Structure Decision

Use the following outline:

1. Introduction: source-private evidence communication, not latent stitching.
2. Problem formulation: `X, T, S, M, D`, rate, candidate-pool recall, controls.
3. Benchmark: hidden-repair candidate selection.
4. Method: source-private tool-trace packets and protocol decoder.
5. Main results: model-produced packet rows across four `500`-example surfaces.
6. Controls and threat model.
7. Rate and systems analysis.
8. Interpretability.
9. Limitations.
10. Conclusion.

Core figures/tables:

- main model-mediated evidence table
- threat-model control table
- rate curve over packet/text/full-log budgets

## Wording Changes Applied

- Changed the working title from cross-agent communication to rate-capped
  evidence communication.
- Replaced broad “hidden execution evidence” language in the one-sentence claim
  with the explicit private `REPAIR_DIAG` trace field.
- Replaced “far beyond target priors” with frozen-surface target/control
  comparisons.
- Made the next gate a target-decoder smoke rather than more paper polishing.

## Next Gate

`source_private_tool_trace_target_decoder_smoke_20260429`:

- replace deterministic lookup with an LLM-mediated or learned target-side
  selector on a small frozen slice
- preserve the same source-destroying controls
- promote only if the row recovers a large fraction of deterministic gain
  without giving the target enough text to trivially parse the diagnostic
