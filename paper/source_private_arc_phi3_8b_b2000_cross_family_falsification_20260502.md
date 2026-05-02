# ARC Phi-3 8B b2000 Cross-Family Falsification

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains defensible; ICLR full paper is
  still blocked.
- Current story: the best positive row is still the ARC `8B` payload / `11B`
  framed Fourier/anchor-syndrome packet over a Qwen-family source cache, with
  10 seed stability, b2000 paired uncertainty, and destructive basis controls.
- Exact gap: the same packet does not survive the strict Mac-local Phi-3
  cross-family source split, so the ICLR method still needs either a stronger
  non-Qwen source endpoint or a richer common-feature connector.

## Commands

Plain same-protocol Fourier gate with Phi-3 source caches:

```bash
./venv_arm64/bin/python scripts/build_source_private_arc_challenge_fourier_anchor_syndrome_gate.py \
  --output-dir results/source_private_arc_challenge_fourier_anchor_syndrome_cross_family_phi3_gate_20260502_budget8_10seed_b2000 \
  --validation-source-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_validation/source_prediction_cache.jsonl \
  --test-source-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_test/source_prediction_cache.jsonl \
  --seeds 47,53,59,61,67,71,73,79,83,89 \
  --budget-bytes 8 \
  --bootstrap-samples 2000
```

Strict source-family falsification wrapper:

```bash
PYTHONDONTWRITEBYTECODE=1 ./venv_arm64/bin/python \
  scripts/build_source_private_arc_challenge_source_family_cache_falsification.py \
  --output-dir results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu_budget8_10seed_b2000 \
  --skip-cache-materialization \
  --alt-validation-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_validation/source_prediction_cache.jsonl \
  --alt-test-cache results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu/phi3_test/source_prediction_cache.jsonl \
  --alternate-source-family phi3_mini_4k \
  --alternate-source-model /Users/sujeethjinesh/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct/snapshots/f39ac1d28e925b323eae81227eaba4464caced4e \
  --budget-bytes 8 \
  --seeds 47,53,59,61,67,71,73,79,83,89 \
  --bootstrap-samples 2000
```

## Lay Explanation

The previous positive result used a Qwen-family source cache. This test asks a
different model family, Phi-3, to privately choose ARC answers, then compresses
those choices into the exact same tiny `8B` packet. If the packet still helped
the Qwen receiver, the method would look less like a same-family artifact. It
did not help.

## Result

Plain Fourier cross-family artifact:
`results/source_private_arc_challenge_fourier_anchor_syndrome_cross_family_phi3_gate_20260502_budget8_10seed_b2000/`

- pass gate: `False`;
- source family: `phi3_mini_4k`;
- payload/framed bytes: `8B/11B`;
- seeds/bootstrap samples: `10/2000`;
- validation matched pass: `0/10`;
- validation matched mean/min: `0.277/0.274`;
- validation target/text: `0.244/0.254`;
- validation min CI95 low vs target: `-0.043`;
- test matched pass: `0/10`;
- test matched mean/min: `0.244/0.242`;
- test target/text: `0.265/0.232`;
- test min CI95 low vs target: `-0.061`;
- anchor-ID shuffle, anchor-value shuffle, and spectral-bin permutation all
  fail at `0/10`.

Strict wrapper artifact:
`results/source_private_arc_challenge_source_family_cache_falsification_20260502_phi3_cpu_budget8_10seed_b2000/`

- overall pass gate: `False`;
- full validation/test pass: `0/10` and `0/10`;
- Qwen-disagreement validation/test pass: `0/10` and `0/10`;
- test Qwen-disagreement rows: `833`;
- Phi/Qwen agreement rate on test: `0.289`;
- Phi source-choice accuracy before packet: validation `0.274`, test `0.246`;
- Qwen source-choice accuracy before packet: validation `0.388`, test `0.346`;
- test Qwen-disagreement matched/Qwen-substituted/text/target:
  `0.200/0.340/0.203/0.273`;
- test Qwen-disagreement matched-minus-Qwen-substituted min: `-0.143`;
- test Qwen-disagreement CI95 low vs Qwen-substituted: `-0.192`.

## Decision

Promote this as a strict negative/falsification result. The current ARC
Fourier/anchor-syndrome packet is positive and statistically stable in the
same-family Qwen-source setting, but it is not source-family-general with the
available Phi-3 source on the Mac.

This result weakens any ICLR claim that the current packet is already a
cross-model common language. It strengthens reviewer trust because it separates
three surfaces:

- alive positive: Qwen-source `8B` Fourier/anchor-syndrome packet;
- bounded positive diagnostic: stronger same-family Qwen-1.5B frozen-test
  source strength;
- ruled-out branch: Mac-local Phi-3 as the strict cross-family repair.

Next exact method gate: use a stronger true non-Qwen source cache when NVIDIA
is available, or move to a sparse SAE/crosscoder or query-bottleneck connector
that learns a shared feature basis before the same `8B` packet receiver.

