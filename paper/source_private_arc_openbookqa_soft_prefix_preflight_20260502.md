# ARC/OpenBookQA Soft-Prefix Preflight

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop is plausible; ICLR full paper remains
  blocked.
- Current story: fixed-byte source-private packets, public-basis benchmark
  gates, and systems byte/exposure accounting are the defensible core.
- Exact gap: a target-loss tokenwise/soft-prefix connector still has not
  produced a positive source-necessary result.

## Gate

New script:
`scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`

Target artifact:
`results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_qwen_hidden_n8_cpu_label_choice/`

This is the first isolated ARC/OpenBookQA target-loss soft-prefix preflight.
It trains only a small connector from answer-key-forbidden source summaries to
target embedding-prefix tokens. The source and target LMs remain frozen. The
readout compares the matched source prefix against target-only, static prefix,
target-cache-only prefix, zero-source, shuffled-source, same-norm noise,
train-mean source, label-shuffled, candidate-deranged, and same-byte visible
text controls.

## Mac Hardware Finding

The MPS target `inputs_embeds` path failed with an Apple attention-shape
`mps.matmul` error on the cheapest hashed-source smoke. The CPU path completed
with `attn_implementation=eager`, so the MPS failure is an environment blocker,
not a scientific result. The promoted artifact is CPU-only.

## Result

Qwen-source-hidden ARC n8 CPU smoke, `label_and_choice` continuation:

- pass gate: `False`
- fit/eval rows: `4 / 4`
- source features: Qwen2.5-0.5B selected-choice hidden summaries
- target: Qwen3-0.6B frozen target loss through `inputs_embeds`
- matched soft-prefix accuracy: `0.000`
- target-only accuracy: `0.250`
- slots-only/static prefix accuracy: `0.500`
- same-norm noise accuracy: `0.500`
- same-byte visible text accuracy: `0.250`
- source-label-copy audit upper bound: `0.750`
- matched mean margin: `-0.931525`
- best pass-control margin: `-0.180762`
- matched minus best-control margin: `-0.750763`
- runtime: about `75.5s`
- peak RSS: about `6.9 GiB`

## Decision

This does not kill the soft-prefix/query-connector branch. It kills only the
tiny Mac-local Qwen-hidden n8 CPU smoke as a positive result. The important
engineering win is that the target-loss connector path now exists and emits
the destructive controls reviewers will demand.

The next live gate should be a larger validation run, but only after improving
the method surface:

- use at least `n=64` and three seeds;
- use a better continuation/scoring policy selected on validation;
- add query-token or cross-attention style source pooling instead of a single
  selected-choice mean hidden vector;
- run on NVIDIA if available, because CPU is too slow and MPS is unstable for
  this target-forward path.

## Lay Explanation

This experiment tried to teach a tiny translator to turn Qwen's hidden clue
about a science question into a few soft tokens that Qwen3 reads before
answering. On this very small smoke slice, the learned hint did not help. A
static learned prefix and random same-norm source noise did better, which means
the current tiny connector is not yet using the source model in a reliable way.
