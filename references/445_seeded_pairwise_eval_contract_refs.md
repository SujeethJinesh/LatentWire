# Seeded Pairwise Eval Contract References

Date: `2026-04-22`

## Why This Memo Exists

Reviewer feedback is now clear on one point: GSM8K32 is a smoke gate, not a
decision surface for nearby variants. The next evaluation phase needs larger
frozen slices, paired uncertainty, and explicit oracle-style diagnostics.

## Strongest References

- [Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores](https://arxiv.org/abs/2410.03492)
  Best direct citation for seed repeats, confidence intervals, and uncertainty
  reporting.
- [tinyBenchmarks: evaluating LLMs with fewer examples](https://arxiv.org/abs/2402.14992)
  Useful for understanding what a carefully chosen small slice can and cannot
  support.
- [RULER](https://arxiv.org/abs/2404.06654)
  Controlled long-context benchmark for the first widening step after the
  frozen same-pair slice.
- [SCBench](https://arxiv.org/abs/2412.10319)
  Best direct benchmark for KV/cache-centric methods.
- [LongBench v2](https://arxiv.org/abs/2412.15204)
  Broader long-context reasoning benchmark for the next realism step.

## Exact Testing Contract To Use Next

1. Treat `32` examples only as a smoke/regression gate.
2. Move the main same-pair decision surface to at least `500` frozen examples.
3. Run `5` seeds minimum on exactly the same example IDs.
4. Report paired bootstrap intervals and McNemar-style paired comparison, not
   only point accuracy.
5. Always log:
   - `source_alone`
   - `target_alone`
   - text relay
   - communicated row
   - oracle bound
   - wins / ties / losses on the same IDs

## Smallest High-Signal Eval Upgrades

- Larger frozen same-pair campaign using the new campaign runner.
- One strict matched cross-family falsification pair after that.
- Anti-leakage controls:
  - latent shuffle
  - residual swap
  - structured-text relay at matched budget
