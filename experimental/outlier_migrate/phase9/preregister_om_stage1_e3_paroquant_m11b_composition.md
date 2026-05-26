# Stage 1 E3 Preregistration: ParoQuant and M11b Composition

Date: 2026-05-26

## Purpose

Test whether rotation-based W4A16 quantization and budget-tuned channel
protection compose. ParoQuant is the strongest Granite-Small rotation baseline
in the current paper, while M11b top-10 is the strongest budget-tuned
channel-set arm on Nemotron and a partial signal on Granite.

## Primary Scope

Run Granite-Small first under the revised throughput plan. Attempt Nemotron
only if the 15 GPU-hour E3 cap leaves room. The prior Stage 1 E1 throughput
probe projected Nemotron dense/manual decode to dominate the remaining budget,
so Granite-only E3 is a preregistered scope reduction if Nemotron would exceed
the cap.

## Regimes

1. BF16 baseline
2. Static top-1% baseline for recovery denominator
3. ParoQuant alone
4. M11b top-10 alone
5. ParoQuant plus M11b top-10 protected channels
6. ParoQuant plus random top-10 matched-budget protected channels

The composition is implemented by applying ParoQuant-style folded rotation and
groupwise INT4 quantization while restoring protected input/output hidden-axis
channels to their BF16 weights after quantization.

## Decision Rule

For each completed model, compute positive-static-gap median recovery and a
95% bootstrap CI using 30 fixed bootstrap seeds. Let `composition` be
ParoQuant+M11b top-10 and `best_individual` be the larger of ParoQuant alone
and M11b top-10.

- `PASS_E3_SUPER_ADDITIVE`: composition exceeds `best_individual` by at least
  0.10 median recovery and composition CI lower bound is above 0.10.
- `PASS_E3_COMPLEMENTARY`: composition is within 0.05 of `best_individual` and
  beats at least one individual method by at least 0.15.
- `KILL_E3_SUB_ADDITIVE`: composition is at least 0.05 below
  `best_individual`.
- `AMBIGUOUS_E3`: none of the above.
- `FAIL_INFRA_E3`: artifacts are missing or internally inconsistent.

If the Granite-first result is `PASS_E3_SUPER_ADDITIVE`, pause for human
framing review because the paper headline changes.

## Budget

Cap: 15 GPU hours. Stop E3 at the cap, commit completed model results, and
continue to E2. Cumulative GPU work stops at 340 hours and never exceeds 355.
