# Stage 1 E4 Preregistration: Format-Axis Triangulation

**Date:** 2026-05-26

## Goal

Test whether the W4A16 channel-drift findings are specific to the chosen
format axis by comparing four inference formats on Granite-4.0-H-Small:

1. FP8
2. W4A16
3. NVFP4-W4A16
4. NVFP4-W4A4

The experiment reports task accuracy, tokens per second, and peak VRAM on
AIME-2025 and MATH-500. It is descriptive rather than PASS/KILL gated.

## Revised Scope

The revised throughput plan narrows E4 to Granite-Small only. Nemotron-3 is
excluded because observed throughput would consume disproportionate GPU budget.
The benchmarks are AIME-2025 and MATH-500 only; GPQA-Diamond is out of scope
for this format-axis check.

## Kernel Discipline

The runner must not silently substitute simulated formats for unavailable
runtime kernels. Missing or immature NVFP4 kernels must be reported as
`SKIPPED_INFRA` with a concrete reason. FP8, W4A16, NVFP4-W4A16, and
NVFP4-W4A4 rows must remain distinguishable in the output packet even if some
are skipped.

The local W4A16 baseline may use the established dequantized INT4 weight-only
implementation from the outlier-migrate intervention stack. FP8 and NVFP4 rows
require explicit runtime support probes before any benchmark metrics are
accepted.

## Measurement

For each completed format and benchmark:

- accuracy: exact-match task accuracy under the benchmark's answer format
- tokens_per_second: generated output tokens divided by measured wall-clock time
- peak_vram_gb: maximum GPU memory observed during the run
- prompt_count: number of prompts evaluated

The runner may ingest metrics from a benchmark adapter file, but the checker
must validate that each row is tied to one of the preregistered formats and
benchmarks.

## Budget and Stop Rule

Cap: 15 GPU hours.

If the cap is reached before all formats complete, commit the partial packet.
Rows without valid runtime metrics are either `PENDING_GPU_RUN` or
`SKIPPED_INFRA`; no row may be backfilled from another format.

## Reporting

The final packet reports all four formats in a single table. If at least one
NVFP4 format is skipped, the paper limitation should state that NVFP4 runtime
kernel maturity blocked the comparison on this hardware/software stack.
