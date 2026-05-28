# V1 ParoQuant DeepSeek/Falcon Rotation Smoke Preregistration

**Frozen on**: 2026-05-28
**Status**: Frozen rotation-first smoke preregistration.

## Purpose

V1 ParoQuant-on-Nemotron showed that ParoQuant-style rotation dominates the
previous Nemotron M11b top-10 positive result. This smoke packet tests whether
the same static rotation baseline also resolves the two remaining non-positive
model regimes:

- DeepSeek-R1-Distill-Qwen-1.5B, where static top-10 is the honest bar.
- Falcon-H1-0.5B-Instruct, where existing channel policies are near zero.

ParoQuant is a baseline, not this project's proposed method. These runs decide
whether DriftRot should focus on improving a strong rotation baseline or
whether Falcon/DeepSeek still need architecture-local rescue.

## Models

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
  - snapshot `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- `tiiuae/Falcon-H1-0.5B-Instruct`
  - snapshot `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368`

## Trace Set

Use the same 12 deterministic AIME-2025 traces, indices `0-11`, as the V2
M11b DeepSeek/Falcon packets. The runners may reuse each model's existing BF16
traces, BF16 score cache, static top-1% score cache, and protected sets from
the corresponding V2 M11b packet.

## Quantization and Scoring

Use the same local algorithmic ParoQuant reproduction as the Granite and
Nemotron ParoQuant packets:

- group size `128`;
- `8` deterministic pairwise rotations;
- scale clip `[0.25, 4.0]`;
- W4A16 weight-only scoring;
- scoring position `10000`;
- scoring window `512` tokens.

## Decision Criteria

DeepSeek:

- `PASS_V1_PAROQUANT_DEEPSEEK_ROTATION_DOMINATES` if ParoQuant median recovery
  is greater than static top-10's validated median recovery `0.376752614594403`.
- `KILL_V1_PAROQUANT_DEEPSEEK_WEAK` if ParoQuant median recovery is below
  `0.30`.
- `AMBIGUOUS_V1_PAROQUANT_DEEPSEEK` otherwise.

Falcon:

- `PASS_V1_PAROQUANT_FALCON_ROTATION_RESCUE` if ParoQuant median recovery is
  at least `0.20` or beats Falcon M11b top-10 median recovery
  `0.04393973019116826` by at least `0.10`.
- `KILL_V1_PAROQUANT_FALCON_WEAK` if ParoQuant median recovery is no better
  than Falcon M11b top-10.
- `AMBIGUOUS_V1_PAROQUANT_FALCON` otherwise.

Any PASS changes the next GPU queue by lowering priority of channel-set rescue
on that model.

## Forbidden Actions

- Do not claim ParoQuant as this project's method.
- Do not change trace indices after observing ParoQuant scores.
- Do not change ParoQuant group size, rotation count, or clipping after
  observing scores.
- Do not run killed WJAC, M-PRED, hard-switch, or naive ParoQuant+M11b
  composition branches as part of this smoke.
