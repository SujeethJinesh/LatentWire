# Source-Private OpenBookQA Second-Benchmark Gate, 2026-05-01

## Status

- bridge contract:
  `results/source_private_openbookqa_bridge_contract_20260501/`
- validation source-cache run:
  `results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_validation/`
- test source-cache run:
  `results/source_private_openbookqa_fixed_packet_gate_20260501_qwen05_hashed_test_4b/`
- promoted `3B` seed-stability artifacts:
  `results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_validation_3b/`
  and
  `results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_3b/`
- supporting rate probes:
  `results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_validation_4b/`
  and
  `results/source_private_openbookqa_seed_stability_20260501_qwen05_hashed_test_4b/`
- SciQ diagnostic:
  `results/source_private_sciq_bridge_contract_20260501/`
  and
  `results/source_private_sciq_fixed_packet_gate_20260501_qwen05_hashed_validation/`
- references:
  `references/574_second_benchmark_openbookqa_sciq_refs_20260501.md`

## Contract

OpenBookQA `main` was materialized as train/validation/test
`4957/500/500`. The canonical packet rows expose only question and candidate
choices to the source packet builder. `answerKey`, `fact1`, and auxiliary human
metadata are forbidden for this gate. There is no cross-split content overlap.
The upstream dataset has five duplicate content IDs and three duplicate-choice
rows in train; these are logged as dataset warnings, not removed.

SciQ was also materialized as train/validation/test `11679/1000/1000` with
deterministically shuffled answer positions. SciQ is kept as a diagnostic
because short answer strings make same-byte text nearly saturate the source
signal.

## Results

| Benchmark | Split | Budget | Seeds | Pass | Matched mean/min/max | Target | Same-byte text | Min lift vs target | Min lift vs text | Min CI95 low |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenBookQA | validation | 3B | 5 | 5/5 | 0.356 / 0.354 / 0.358 | 0.252 | 0.326 | 0.102 | 0.028 | 0.045 |
| OpenBookQA | test | 3B | 5 | 5/5 | 0.378 / 0.378 / 0.380 | 0.276 | 0.350 | 0.102 | 0.028 | 0.038 |
| OpenBookQA | validation | 4B | 5 | 5/5 | 0.356 / 0.356 / 0.356 | 0.252 | 0.328 | 0.104 | 0.028 | 0.046 |
| OpenBookQA | test | 4B | 5 | 5/5 | 0.378 / 0.378 / 0.378 | 0.276 | 0.360 | 0.102 | 0.018 | 0.038 |
| SciQ | validation | 12B | 1 | pass | 0.712 / 0.712 / 0.712 | 0.246 | 0.706 | 0.466 | 0.006 | 0.425 |

## Interpretation

OpenBookQA promotes the second-benchmark branch. The fixed-byte packet preserves
the Qwen source-choice signal on a public benchmark beyond ARC, passes
validation and test, and improves over same-byte text at `3B` with `5/5`
projection seeds. This gives a stronger systems story than ARC alone: the
positive packet can be as small as one signed projected residual coordinate
(`3B` payload, `6B` framed record under the current record accounting).

The result is still not a full hidden-state endpoint. The source decision is a
local Qwen choice log-likelihood cache computed from question and candidates,
and the packet transmits that source choice through a public hashed basis. The
safe claim is benchmark-general source-private common-basis packet transfer, not
universal latent reasoning or semantic-anchor superiority.

SciQ is useful as a reviewer-facing limitation. It passes target/control gates,
but same-byte text is nearly identical because many answer strings are short
enough for a 12-byte text snippet. This supports including a text-catch-up
discussion rather than claiming every benchmark has a text-baseline win.

## Next Gate

The next highest-value ICLR gate is to add a native hidden-state/common-basis
endpoint or a non-science public benchmark. The systems blocker remains
NVIDIA/vLLM or SGLang TTFT, TPOT, goodput, HBM traffic, and KV/cache comparator
rows.
