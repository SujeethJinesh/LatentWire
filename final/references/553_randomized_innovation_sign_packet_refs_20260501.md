# Randomized Innovation Sign Packet References, 2026-05-01

## Purpose

This memo supports the Mac-bounded randomized candidate-local sign-sketch
probe added on 2026-05-01. The probe tests whether the live candidate-local
residual packet can be reframed as a public randomized measurement interface:
rotate the target-local residual chart with deterministic public randomness,
quantize to signs, and score by sign agreement. It is a diagnostic for a
possible third technical contribution, not yet a promoted method.

## Primary Sources

- TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
  arXiv:2504.19874. https://arxiv.org/abs/2504.19874
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. arXiv:2404.00456.
  https://arxiv.org/abs/2404.00456
- SpinQuant: LLM quantization with learned rotations. arXiv:2405.16406.
  https://arxiv.org/abs/2405.16406
- Similarity Estimation Techniques from Rounding Algorithms. STOC 2002.
  https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p380-charikar.pdf
- Relative representations enable zero-shot latent space communication.
  arXiv:2209.15430. https://arxiv.org/abs/2209.15430
- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  arXiv:2510.03215. https://arxiv.org/abs/2510.03215

## Local Probe Added

Code:
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`

Tests:
`tests/test_run_source_private_learned_synonym_dictionary_packet_gate.py`

New decoder modes:

- `candidate_local_random_rotation_sign_residual_norm`
- `candidate_local_random_rotation_rank_sign_residual_norm`

Both modes leave packet encoding unchanged. They transform only the
candidate-local residual decision surface by subtracting the candidate-local
mean, normalizing, applying a deterministic public orthogonal matrix, optionally
rank-normalizing, quantizing to signs, and scoring by normalized sign
agreement.

## Mac Smoke Evidence

Artifacts:

- `results/source_private_candidate_local_random_rotation_sign_20260501_seed47_n128_evaldisjoint/`
- `results/source_private_candidate_local_random_rotation_rank_sign_20260501_seed47_n128_evaldisjoint/`
- `results/source_private_candidate_local_random_rotation_sign_20260501_seed47_n128_evaldisjoint_tau0/`
- `results/source_private_candidate_local_random_rotation_sign_20260501_seed47_n128_evaldisjoint_tau0.10/`
- `results/source_private_candidate_local_random_rotation_sign_20260501_seed47_n128_evaldisjoint_tau0.20/`
- `results/source_private_candidate_local_random_rotation_sign_20260501_seed47_n128_evaldisjoint_tau0.30/`

At the live cosine threshold `min_decision_score=0.48`, both sign and
rank-sign variants pass only `2/3` n128 smoke directions:

- `core_to_holdout`: matched `0.625`, best control `0.250`, pass.
- `holdout_to_core`: matched `0.250`, best control `0.250`, fail.
- `same_family_all`: matched `0.4375`, best control `0.250`, pass.

At `min_decision_score=0.0`, matched accuracy rises, but destructive controls
rise with it:

- `core_to_holdout`: matched `0.875`, best control `0.500`.
- `holdout_to_core`: matched `0.625`, best control `0.500`.
- `same_family_all`: matched `0.750`, best control `0.4375`.

Threshold and margin sweeps did not find a clean operating point: low
thresholds leak controls, while high thresholds collapse holdout-to-core.

## Interpretation

The randomized sign-sketch branch is weakened, not promoted. It reveals
recoverable source signal at low threshold, but the same low-threshold regime
also admits destructive control packets. At the safe threshold, it becomes a
clean two-direction partial method and fails the strict reverse cross-family
row. This is evidence that the live candidate-local residual packet's strength
is not explained by generic random measurement alone.

## Next Gate

Do not spend n512 cycles on this branch unless a new mechanism blocks
destructive controls. The next high-value method branch should be a true
candidate-conditioned residual code with destructive-control regularization,
or a native systems run that strengthens the already-clean live row.
