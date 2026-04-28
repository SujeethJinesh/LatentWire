# Source-Private Tool-Trace Draft Review

- date: `2026-04-30`
- status: skeptical review completed
- draft reviewed: `paper/source_private_tool_trace_paper_draft_20260430.md`

## Verdict

The paper is scoped-draft-ready, but not submission-ready. The main evidence
supports a source-private evidence-communication claim. The draft still needs
figure/table assets, concrete citations, and either a scaled target-decoder row
or explicit framing that the target-decoder row is only a smoke ablation.

## Top Rejection Risks

1. Coded-label lookup / benchmark artifact: candidate metadata exposes
   `handles_repair_diag=<code>`, and the private trace exposes `REPAIR_DIAG`.
2. Target decoder does too much work: main decoder is deterministic exact-match
   protocol logic; Qwen3 target-decoder evidence is only `N=16/32`.
3. Synthetic-only external validity: no real tool/retrieval workflow yet.
4. Structured text relay can win with enough bytes: JSON/free-text relays are
   oracles at `32` bytes, so the rate curve must be visible.
5. Related-work citations are not yet concrete in the draft.

## Wording Patches Applied

- softened “often need” to “can benefit”
- replaced “useful hidden source evidence” with “the benchmark's hidden
  diagnostic evidence”
- replaced “strict controls” with “the controls tested here”
- changed “The result survives...” to “On these frozen synthetic surfaces...”
- changed “interpretable by construction” to “directly auditable in this
  benchmark”
- changed “robust across large frozen slices” to “stable across the tested
  frozen `500`-example surfaces”

## Missing Before Submission

- setup diagram
- rate curve figure
- appendix table with per-seed exact IDs and artifact hashes
- Table 1 counts as well as rates
- concrete bibliography entries for distributed source coding, rate-distortion,
  semantic communication, tool-use agents, VLM connectors, JEPA, C2C/cache
  transfer, and KV compression/communication

## Highest-Priority Evidence Patch

Scale or optimize the Qwen3 target-decoder smoke if reviewer strategy requires
it. This directly addresses the hand-coded decoder novelty risk. If compute is
too expensive, leave the row as a smoke ablation and make the deterministic
protocol decoder a clear scope limitation.

## Next Gate

`source_private_tool_trace_latex_or_figures_20260430`:

- create setup/rate-curve figure assets or convert the draft to ICLR LaTeX
- add citation placeholders and counts
- keep target-decoder scale-up as a separate optional evidence gate
