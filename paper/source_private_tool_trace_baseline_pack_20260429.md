# Source-Private Tool-Trace Baseline Pack

- date: `2026-04-29`
- status: reviewer-facing evidence package, not final paper draft
- live branch: explicit source-private tool-trace packet handoff
- scale rung: reviewer evidence packaging

## Claim

Explicit source-private tool-trace packets communicate hidden execution evidence
to a target-side candidate decoder.

Boundary: the method is not raw-log repair inference and not unstructured latent
transfer. The explicit private `REPAIR_DIAG` trace field is the communication
interface.

## Evidence Summary

Across four frozen `500`-example surfaces:

- core seed `29`
- core seed `31`
- held-out seed `30`
- held-out seed `32`

Primary `trace_no_hint` rows pass for Qwen3 and Phi-3 on every surface. Qwen3
`raw_log_no_trace` fails on every surface with `0` valid packets.

Aggregate:

- primary rows passing: `8/8`
- destruction rows failing as intended: `4/4`
- minimum primary lower bound over target-only: `0.516`
- minimum primary lower bound over best control: `0.506`
- maximum destruction matched accuracy: `0.250`

## Systems Summary

- primary packet mean bytes range: `1.55-2.00`
- primary packet validity range: `0.776-1.000`
- deterministic matched-byte hidden-log text accuracy: `0.250`
- deterministic full hidden-log relay accuracy: `1.000`
- full hidden-log relay costs roughly `366-374` bytes and `34` tokens per
  example on the representative `500`-example core/held-out surfaces

## Controls Covered

| Risk | Control | Result |
|---|---|---|
| Target prior or wrapper solves the task | target-only and target-wrapper/no-source | target rows stay at `0.250` |
| Source is not needed | zero-source, shuffled-source, random same-byte | controls remain `0.252-0.258` |
| Answer leakage explains the gain | answer-only and answer-masked sidecars | controls stay at target-only |
| Target metadata explains the gain | target-derived sidecar | stays at target-only |
| Matched-byte text relay explains the gain | truncated matched-byte hidden-log text | stays at `0.250` |
| Raw logs are enough without protocol | `raw_log_no_trace` | Qwen3 returns to target-only with `0` valid packets |
| Template leakage | disjoint held-out repair families | Qwen3 reaches `0.922/0.924`, Phi-3 reaches `1.000` |
| Seed instability | four frozen 500-example surfaces | all primary rows pass |

## Remaining Reviewer Gaps

| Gap | Status |
|---|---|
| Matched-byte structured JSON/free-text relay | truncated matched-byte text is covered; JSON relay still needed |
| Target helper-only/no-log oracle | target-only and target-wrapper are covered; stronger no-log baselines remain |
| Masked trace component ablations | raw trace removal is covered; expected/actual, line-number, test-name masking remain |
| Candidate/selector separation | candidate pool recall is deterministic `1.0`; paper table should still separate pool and selector |
| Second target-family pair | source emitters are cross-family; target decoder is deterministic protocol decoder |

## Decision

Promote the evidence package as the reviewer-facing baseline pack for the live
method. The next work should turn this into paper-claim text and close the
remaining reviewer gaps only if they threaten the exact scoped claim.

## Next Gate

`source_private_tool_trace_paper_claim_draft_20260429`:

- draft exact method, benchmark, baseline, systems, and limitations language
- keep the claim scoped to explicit private tool-trace packets
- include the remaining-gap table as planned ablations, not as solved evidence
- avoid claiming raw-log repair inference, latent transfer, or universal
  cross-model communication
