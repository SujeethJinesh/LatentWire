# V1 ParoQuant-on-Nemotron Baseline Vetting Preregistration

**Frozen on**: 2026-05-27
**Frozen by**: Codex, after Stage 1 integration, citation additions, and math
formalization, before any ParoQuant-on-Nemotron score cache was inspected.
**Status**: Frozen baseline-vetting preregistration.

## Purpose

Granite-Small ParoQuant recovered a large fraction of the W4A16 static-top-1%
gap, while Nemotron-3 M11b top-10 recovered strongly after the static baseline
salvage. This run tests whether ParoQuant is also strong on Nemotron-3. The
outcome decides whether the paper should frame the positive evidence as
architecture-dependent method selection or as rotation-dominant across the two
hybrid models.

## Model

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- HuggingFace snapshot commit:
  `cbd3fa9f933d55ef16a84236559f4ee2a0526848`

## Trace Set

- Source: AIME-2025
- Count: 12 deterministic prompts, indices `0-11`
- Prompt file: `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- The canonical prompt hash must match the existing Phase 9 AIME slice.

## Reused Artifacts

This run may reuse the validated Nemotron salvage packet:

`experimental/outlier_migrate/phase9/results/om_phase9_m11b_nemotron_static1pct_salvage_20260524T2130Z`

Allowed reuse:

- BF16 traces;
- static top-1% protected sets;
- BF16 score cache;
- static top-1% score cache.

The run must write reused artifact paths and hashes to `source_artifacts.json`.
No ParoQuant score cache may be reused because no Nemotron ParoQuant packet has
been run.

## Quantization and Scoring

- Baseline formula matches the Granite-Small ParoQuant baseline.
- ParoQuant implementation mode is the audited local algorithmic reproduction:
  scaled pairwise rotations plus groupwise affine INT4 folded back to
  dequantized weights.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap samples: `1000`.
- Bootstrap seed: `20260621`.

## Regimes

1. BF16 baseline.
2. Static top-1% protected W4A16 baseline from position `100`.
3. ParoQuant W4A16 algorithmic baseline.

## Metric

For each trace:

`recovery = 1 - (perplexity_ParoQuant - perplexity_BF16) / (perplexity_static_top1 - perplexity_BF16)`

Only traces with a positive recoverable static top-1% gap are included in the
primary recovery median. The no-gap count and fraction remain reported.

## Decision Criteria

Let Nemotron M11b top-10 median recovery be the validated salvage result
`0.8147397989034302`.

- `PASS_V1_PAROQUANT_NEMOTRON_ROTATION_DOMINATES` if ParoQuant median recovery
  is greater than `0.85`.
- `PASS_V1_PAROQUANT_BEATS_M11B_NEMOTRON` if ParoQuant beats M11b top-10 by at
  least `0.05` median recovery.
- `SUPPORTS_V1_ARCHITECTURAL_DEPENDENCE` if ParoQuant median recovery is in
  `[0.50, 0.8147397989034302]`.
- `KILL_V1_PAROQUANT_NEMOTRON_WEAK` if ParoQuant median recovery is below
  `0.50`.
- `AMBIGUOUS_V1_PAROQUANT_NEMOTRON` otherwise.
- `FAIL_INFRA_V1_PAROQUANT_NEMOTRON` for incomplete packets or runner/checker
  failures.

Headline-changing outcomes are the two `PASS_*` decisions above and must be
surfaced before paper reframing.

## Forbidden Actions

- Changing trace indices after seeing any ParoQuant-on-Nemotron score.
- Reusing any ParoQuant-on-Nemotron score cache.
- Changing ParoQuant group size, rotation count, or scale clipping after
  observing the result.
- Reporting this as a full upstream ParoQuant reproduction.
- Modifying model source code.
