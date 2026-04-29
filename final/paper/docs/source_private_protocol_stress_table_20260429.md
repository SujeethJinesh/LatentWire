# Source-Private Protocol Stress Table

- date: `2026-04-29`
- artifact: `results/source_private_protocol_stress_table_20260429/`
- script: `scripts/build_source_private_protocol_stress_table.py`
- test: `tests/test_build_source_private_protocol_stress_table.py`
- references: `references/486_protocol_stress_and_uniqueness_refs.md`
- scale rung: reviewer-risk ablation aggregation

## Purpose

This gate addresses the strongest protocol novelty criticism: the method could
be dismissed as a fixed diagnostic-label lookup. The table collects the existing
stress evidence into one appendix-ready artifact and marks the missing stress
row explicitly.

## Result

The aggregate has `22` rows:

- `12` deterministic diagnostic-codebook remap rows over `500` examples, three
  remapped codebooks, and budgets `2/4/8/16` bytes. All pass with matched
  accuracy `1.000`, no-source `0.250`, and controls at `0.250-0.256`.
- `3` learned slot-feature remap rows over `512` examples at `6` bytes. All
  pass, but with weaker margins: `0.463-0.508` accuracy versus target-only
  `0.250`.
- `7` canonical candidate-order remap rows over `512` examples at `4` bytes.
  These are positive versus target-only but marked as near-miss rows because the
  aggregate seven-remap bootstrap previously missed the strict CI threshold
  (`+0.146` vs `+0.150`).

Headline: `15` pass rows and `7` near-miss rows; minimum passing delta versus
target-only is `+0.213`.

## Interpretation

This strengthens the paper’s scope-limited claim:

> Source-private evidence packets are robust to deterministic codebook remaps,
> learned slot-feature remaps, and canonical candidate-order remaps, but the
> strongest cross-codebook/candidate-order evidence is still a scoped same-family
> result, not a broad cross-family latent-transfer claim.

The table also keeps the core limitation visible: learned target-decoder prompt
paraphrase stress is not yet measured. That is the next cheapest reviewer-facing
gate after the local MPS target-decoder crash.

## Literature Positioning

The new reference memo concludes that our contribution should not be sold as
generic model-to-model communication. C2C, KVComm, cache reuse, prompt
compression, and multimodal connector papers already occupy that broader space.
The defensible novelty is narrower: extreme-rate source-private evidence
handoff under decoder side information, exact-ID parity, source-destroying
controls, and byte/rate systems accounting.

## Next Gate

Add query-aware compressed-text controls to the rate frontier or run a learned
target-decoder prompt-paraphrase stress table. The target-decoder `n=256` gate
is still valuable, but local MPS remains blocked by the known Apple MPS matmul
shape failure and CPU would be slow.
