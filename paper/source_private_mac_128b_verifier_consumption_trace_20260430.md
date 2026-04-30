# Mac 128B Verifier Consumption Trace Correction

- date: `2026-04-30`
- artifact:
  `results/source_private_verifier_consumption_trace_20260430/qwen3_seed31_core_holdout_n160_binary_logprob_combined_cpu/`
- status: pass; corrects the systems accounting for this Mac

## Cycle Start

1. Current ICLR readiness and distance: strong scoped/COLM-ready evidence, but
   not comfortable full ICLR until the receiver is less protocol-shaped and the
   systems story has native serving telemetry.
2. Current paper story: source-private packets communicate residual evidence to
   a target that has public candidate side information.
3. Exact blocker: reviewers can accept the packet/receiver row only if the
   systems accounting is hardware-observed and the method claim stays scoped.
4. Current live branch: frozen Qwen3-0.6B binary verifier over 2-byte packets.
5. Highest-priority gate this cycle: correct Mac-local cache-line accounting
   before using the systems trace as a contribution.
6. Scale-up rung: systems reproducibility guard, not a new accuracy rung.

## Layman Version

The earlier packet result said the source sends only two useful bytes, but the
hardware still moves data in minimum chunks. We checked the Mac's chunk size
instead of assuming it. This Mac reports `128` bytes, so one isolated packet is
charged as `128B`. If packets are packed into batches, the cost drops to
`6B/request` at batch 64 and `5B/request` at batch 256.

## What Changed

`scripts/build_source_private_verifier_consumption_trace.py` now:

- auto-detects cache-line size with `sysctl hw.cachelinesize`;
- records the line-size source in `headline.cache_line_size_source`;
- keeps `--line-size` as an explicit override for non-Mac or controlled
  comparisons;
- reports batch floors for `1, 4, 16, 64, 256`;
- still rejects partial prediction files unless
  `--allow-partial-predictions` is passed.

The focused test suite now includes a deterministic `128B` cache-line case.

## Corrected Seed31 n160 Trace

| Metric | Value |
|---|---:|
| matched accuracy, min | `1.000` |
| target-only accuracy, max | `0.250` |
| best control accuracy, max | `0.250` |
| matched minus best control, min | `+0.750` |
| source payload bytes | `2` |
| packet record bytes | `5` |
| observed cache-line floor | `128B` |
| DMA floor | `128B` |
| batch-1 line bytes/request | `128.0` |
| batch-4 line bytes/request | `32.0` |
| batch-16 line bytes/request | `8.0` |
| batch-64 line bytes/request | `6.0` |
| batch-256 line bytes/request | `5.0` |
| target verifier forwards/example | `4.0` |

## Interpretation

This makes the systems contribution more defensible because the artifact now
uses the observed local hardware floor. It slightly weakens the batch-64 line
claim from `5.0B/request` to `6.0B/request`, but the broader systems story
survives: source-boundary traffic remains byte-scale, and the dominant current
cost is the target-side verifier compute, not packet transfer.

The non-claim remains important: this is Mac CPU receiver telemetry and
deterministic boundary accounting, not production GPU/vLLM throughput.

## Next Exact Gate

`source_private_anchor_relative_crosscoder_receiver_n256`

Implement the learned anchor-relative receiver gate with public-only,
shuffled-source, random same-byte, matched-byte text, feature-ID permutation,
top-feature knockout, exact-ID parity, paired uncertainty, and the same
hardware-observed byte accounting.
