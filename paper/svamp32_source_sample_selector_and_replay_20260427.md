# SVAMP32 Source Sample Selector And Replay

Date: `2026-04-27`

## Cycle Start

1. Current ICLR readiness: not ready; still missing a source-derived positive
   method.
2. Current paper story: source sampling exposed two C2C-clean residual IDs, but
   the result needed attribution before connector training.
3. Exact blocker to submission: matched source must select or generate one of
   the two IDs beyond target-only, prompt-wrapper, and source-destroying
   controls.
4. Current live branches: source-sampled selector surface; JEPA-style
   source-innovation connector only if the surface survives controls.
5. Highest-priority gate: strict selector over the new two-ID candidate pool,
   followed by a source-vs-target prompt-wrapper replay.
6. Scale-up rung: smoke.

## Gate 1: Source-Sample Candidate Selector

Added `scripts/build_candidate_pool_decision_surface.py` to append the four
source-sampling rows as distinct `source_sample_s*` candidate labels while
retaining their raw `target_sample_s*` methods for provenance. Added a
`zero_source` condition to `scripts/analyze_candidate_score_sidecar_top_select.py`.

Results:

| Profile | Matched Correct | Matched Clean Correct | Control Clean Union | Accepted Harm |
|---|---:|---:|---:|---:|
| `full` | `6/32` | `0` | `0` | `5` |
| `answer_only` | `6/32` | `0` | `0` | `5` |
| `answer_masked` | `2/32` | `0` | `0` | `6` |

Decision: fail. The deterministic source-candidate sidecar cannot select either
new clean candidate, including with answer-masked source profiles.

## Gate 2: Two-ID S16 Replay

The planner subagent recommended checking whether the source-sampled surface is
stable before training a connector. The replay used 16 samples per condition on
only `6e9745b37ab6fc45` and `de1bf4d142544e5b`.

| Condition | Model | Prompt Mode | Oracle |
|---|---|---|---:|
| `source_sample` | `Qwen/Qwen2.5-Math-1.5B` | `source_reasoning` | `2/2` |
| `target_direct_sample` | `Qwen/Qwen3-0.6B` | `direct` | `0/2` |
| `target_brief_sample` | `Qwen/Qwen3-0.6B` | `source_reasoning` | `2/2` |
| `source_direct_sample` | `Qwen/Qwen2.5-Math-1.5B` | `direct` | `1/2` |

Decision: hard fail for source-specific attribution. The source replay is
stable, but the target model with the same brief-analysis wrapper also reaches
`2/2`. The two-ID surface is prompt-wrapper reachable, not source-specific.

## Subagent Integration

- Reviewer subagent: a two-ID pass would still be only an attribution clue, not
  a benchmark or method claim; prompt-wrapper controls are mandatory.
- Planner subagent: run the S16 replay before any learned connector.
- JEPA subagent: keep Query-JEPA/source-innovation as a future design, but only
  after a surface survives prompt and source-destroying controls.
- Cross-field scout: future compact sidecars should use answer-masked
  conditional innovation, syndrome, or one-bit fingerprint ideas with explicit
  source destruction.

## Decision

Prune this two-ID source-sampled branch as a connector training surface. The
prompt wrapper itself is a powerful target-side candidate generator and must be
promoted to a baseline/control in future gates.

## Next Exact Gate

Switch to prompt-wrapper surface discovery with source attribution controls:
run SVAMP32 or SVAMP70 target brief-wrapper sampling as a target-prior baseline,
then search for source-conditioned gains beyond that wrapper baseline. A method
branch cannot proceed unless matched source adds clean residual IDs not reached
by target direct, target brief-wrapper, zero-source, shuffled-source,
answer-only, answer-masked, or random same-byte controls.
