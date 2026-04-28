# Source-Private Hidden-Repair Seed-Repeat Gate

- date: `2026-04-29`
- status: promoted seed-stable positive method candidate
- live branch: explicit source-private tool-trace packet handoff
- scale rung: seed stability

## Question

Does the private tool-trace packet result remain stable across additional
frozen seeds for both core and held-out repair-family surfaces?

## Setup

The seed-repeat gate aggregates four frozen 500-example surfaces:

- core seed `29`
- core seed `31`
- held-out seed `30`
- held-out seed `32`

Each surface uses the same protocol:

- source prompt: `trace_no_hint`
- source emitters: Qwen3-0.6B and Phi-3-mini
- source-signal destruction row: Qwen3 `raw_log_no_trace`
- target/control rows: target-only, zero-source, shuffled-source,
  random-same-byte, answer-only, answer-masked, and target-derived sidecar

## Results

| Surface | Family set | Seed | Model | Mode | Matched | Target | Best control | Valid | Delta target 95% CI | Delta control 95% CI |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.252 | 0.776 | [0.516, 0.600] | [0.514, 0.602] |
| core_seed29 | core | 29 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | [0.714, 0.788] | [0.708, 0.786] |
| core_seed29 | core | 29 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | [0.000, 0.000] | [-0.006, 0.000] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | trace_no_hint | 0.808 | 0.250 | 0.256 | 0.776 | [0.516, 0.602] | [0.506, 0.594] |
| core_seed31 | core | 31 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.256 | 1.000 | [0.710, 0.786] | [0.704, 0.780] |
| core_seed31 | core | 31 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.256 | 0.000 | [0.000, 0.000] | [-0.014, 0.000] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | trace_no_hint | 0.922 | 0.250 | 0.258 | 0.864 | [0.632, 0.712] | [0.622, 0.706] |
| holdout_seed30 | holdout | 30 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.258 | 1.000 | [0.710, 0.788] | [0.702, 0.778] |
| holdout_seed30 | holdout | 30 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.258 | 0.000 | [0.000, 0.000] | [-0.016, -0.002] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | trace_no_hint | 0.924 | 0.250 | 0.252 | 0.860 | [0.634, 0.716] | [0.632, 0.712] |
| holdout_seed32 | holdout | 32 | microsoft/Phi-3-mini-4k-instruct | trace_no_hint | 1.000 | 0.250 | 0.252 | 1.000 | [0.710, 0.786] | [0.710, 0.786] |
| holdout_seed32 | holdout | 32 | Qwen/Qwen3-0.6B | raw_log_no_trace | 0.250 | 0.250 | 0.252 | 0.000 | [0.000, 0.000] | [-0.006, 0.000] |

Aggregate:

- primary rows: `8/8` pass
- destruction rows: `4/4` fail as intended
- minimum primary lower bound over target-only: `0.516`
- minimum primary lower bound over best control: `0.506`
- maximum destruction matched accuracy: `0.250`
- Qwen3 minimum matched accuracy: `0.808`
- Phi-3 minimum matched accuracy: `1.000`

## Interpretation

The method is now stable across seeds, source model families, core repair
families, and held-out repair families. The trace-removed destruction row stays
at target-only with zero valid packets on every seed. This is the strongest
evidence so far that the gain is a source-private trace communication effect,
not a target prior, selector artifact, or shuffled/random sidecar artifact.

The claim remains scoped to explicit private tool-trace communication. The
method is not raw-log inference and should not be framed as unstructured latent
transfer.

## Decision

Promote to seed-stable positive method candidate. The next work should be
reviewer-facing: baseline consolidation, systems table, threat model, and paper
story.

## Next Gate

`source_private_tool_trace_baseline_pack_20260429`:

- consolidate target-only, matched packet, zero/shuffled/random/answer-only/
  answer-masked/target-derived, matched-byte text, full hidden-log relay, and
  full diagnostic oracle rows
- report accuracy, bytes, generated tokens, p50/p95 source latency, and paired
  uncertainty
- explicitly frame the method as explicit source-private tool-trace packet
  communication
- produce a reviewer threat-model table before any paper-claim draft
