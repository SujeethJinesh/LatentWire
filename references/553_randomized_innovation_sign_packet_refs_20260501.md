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
  - Relevant claim boundary: TurboQuant uses random rotation, scalar
    quantization, and a 1-bit QJL residual correction for vector/KV-cache
    quantization and inner-product estimation. LatentWire must not claim
    novelty for random rotations or sign sketches.
- QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead. arXiv:2406.03482. https://arxiv.org/abs/2406.03482
  - Relevant claim boundary: JL transforms plus sign-bit quantization for
    KV-cache compression and unbiased asymmetric inner-product estimation
    are prior work.
- QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. arXiv:2404.00456.
  https://arxiv.org/abs/2404.00456
  - Relevant claim boundary: rotation-based hidden-state/activation/KV-cache
    quantization is prior work.
- SpinQuant: LLM quantization with learned rotations. arXiv:2405.16406.
  https://arxiv.org/abs/2405.16406
  - Relevant claim boundary: learned rotations improve over random rotation
    in some LLM quantization settings, so random rotation itself is a weak
    novelty axis.
- Similarity Estimation Techniques from Rounding Algorithms. STOC 2002.
  https://www.cs.princeton.edu/courses/archive/spr05/cos598E/bib/p380-charikar.pdf
  - Relevant claim boundary: random hyperplane signs for angular/cosine
    similarity estimation are classical sketching machinery.
- Relative representations enable zero-shot latent space communication.
  arXiv:2209.15430. https://arxiv.org/abs/2209.15430
  - Relevant contrast: anchor-relative coordinates are a principled way to
    avoid absolute latent-basis mismatch, but the live row is candidate-local
    and source-private rather than a shared anchor representation.
- Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
  arXiv:2510.03215. https://arxiv.org/abs/2510.03215
  - Relevant contrast: C2C projects/fuses source KV cache into target KV cache.
    LatentWire's live claim remains no source text and no source KV exposure.

## Local Probe Added

Code:
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`

Tests:
`tests/test_run_source_private_learned_synonym_dictionary_packet_gate.py`

New decoder modes:

- `candidate_local_random_rotation_sign_residual_norm`
- `candidate_local_random_rotation_rank_sign_residual_norm`

Both modes leave packet encoding unchanged. They transform only the
candidate-local residual decision surface:

1. subtract the target candidate-local mean;
2. L2-normalize candidate residual rows and the decoded payload vector;
3. apply a deterministic public orthogonal matrix seeded by a namespace;
4. optionally rank-normalize rotated coordinates;
5. quantize to signs and score by normalized sign agreement.

This is a scorer-level diagnostic, not a true packed sign-packet encoder.

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

Threshold sweep for the sign variant:

- `tau=0.10`: same leakage pattern as `tau=0.0`.
- `tau=0.20`: matched remains high but controls stay too high
  (`0.375-0.500`).
- `tau=0.30`: controls improve, but holdout-to-core collapses to target.
- `tau=0.48`: controls are clean, but holdout-to-core stays at target.

Margin-gating replay on the `tau=0.0` predictions did not find a clean
operating point. At margin `0.15`, holdout-to-core matched reaches `0.500`,
but controls remain up to `0.289`; at margin `0.20`, controls rise again.

## Interpretation

The randomized sign-sketch branch is weakened, not promoted. It reveals
recoverable source signal at low threshold, but the same low-threshold regime
also admits destructive control packets. At the safe threshold, it becomes a
clean two-direction partial method and fails the strict reverse cross-family
row. This behavior is too similar to previous unsafe common-basis probes to
claim as a positive source-private method.

## Safe Paper Framing

Do not claim:

- random rotation is novel;
- sign sketches are novel;
- TurboQuant/QJL is being improved;
- the current diagnostic is a real compressed sign-packet encoder.

Safe claim:

LatentWire tested a public randomized sign-sketch interface motivated by
rotation/JL quantization and random-hyperplane sketching. The probe shows that
public randomized bases can expose source signal, but also create control
leakage unless aggressively thresholded. This is evidence that the live
candidate-local residual packet's strength is not explained by generic random
measurement alone; source-private control separation remains the key
contribution.

## Next Gate

Do not spend n512 cycles on this branch unless a new mechanism blocks
destructive controls. The next high-value method branch should be a true
candidate-conditioned residual code with destructive-control regularization,
or a native systems run that strengthens the already-clean live row.
