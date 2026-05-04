# Target Self-Resonance Source-Oracle Distill Gate

Date: 2026-05-04

## Paper Status

Current readiness: COLM workshop remains plausible as a scoped
source-private packet/evaluation artifact; ICLR full paper is still blocked.
The estimated ICLR distance remains about 40% because target self-resonance
capacity is real, but a learned source-conditioned positive method has not yet
survived controls.

Current story: LatentWire has a rigorous source-private packet contract,
destructive-control evaluation harness, Mac-local systems accounting, and a
live target-resonance capacity branch. The source-conditioned receiver is still
the missing proof point.

Exact gap: a learned source-conditioned target-prefix receiver must beat
target-only slots, explicit source-index/rank/score packets, same-byte text or
code, and source-destroying controls on frozen rows with paired uncertainty.

## Technical Contributions Under Review

1. Source-private fixed-byte packet contract: strong and reproducible.
2. Destructive-control and paired-CI evaluation harness: strong and central.
3. Mac-local byte/transport systems accounting: useful, but native GPU serving
   rows remain blocked.
4. Selective target-resonance soft-prefix receiver: target-side oracle capacity
   is alive, but learned source-conditioned transfer failed this gate.

## Motivation

The target self-resonance capacity extension showed that per-example optimized
8-token Qwen soft prefixes can reproduce full-prompt Qwen behavior across
HellaSwag validation `0:64` with agreement `0.937500` and mean KL `0.003533`.
That proves target reachability, not communication.

This gate tested the next source-private bridge:

```text
TinyLlama hidden summaries
-> projected source code
-> learned Qwen soft prefix
-> fixed anchor + candidate continuation
```

The target never receives source text, source KV, raw source logits, or the
validation optimized target prefix.

Lay explanation: we first found hidden tokens that make Qwen behave like it saw
the real question. Then we asked whether TinyLlama's hidden state could predict
those hidden tokens for new questions. It could fit train rows, but it did not
produce trustworthy held-out answers.

## Implementation

Added:

- `scripts/build_target_self_resonance_hellaswag_source_oracle_distill_gate.py`
- `tests/test_build_target_self_resonance_hellaswag_source_oracle_distill_gate.py`

The gate:

- optimizes target oracle soft prefixes on official HellaSwag train rows;
- projects TinyLlama source-hidden features with train-fit PCA;
- trains a small source-code-to-Qwen-prefix encoder with candidate-logit KL,
  oracle-prefix distillation, RMS regularization, and wrong-source contrastive
  loss;
- evaluates frozen validation rows against zero-source, wrong-source,
  candidate-roll, target-score-derived, random-prefix, candidate-derangement,
  source-top1, and source-top1/top2 oracle controls.

Novelty and claim boundaries are recorded in:

- `references/714_source_conditioned_soft_prefix_selective_resonance_refs_20260504.md`
- `references/715_source_private_systems_hardware_quantization_review_20260504.md`

## Commands

Smoke with 16 train rows:

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_source_oracle_distill_gate.py \
  --output-dir results/target_self_resonance_hellaswag_source_oracle_distill_gate_20260504_tiny_to_qwen05_train16_validation64_72 \
  --train-rows 16 \
  --eval-start 64 \
  --eval-rows 8 \
  --oracle-steps 24 \
  --epochs 8 \
  --bootstrap-samples 500 \
  --run-date 2026-05-04
```

Rank/capacity ablation with 64 train rows:

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_source_oracle_distill_gate.py \
  --output-dir results/target_self_resonance_hellaswag_source_oracle_distill_gate_20260504_tiny_to_qwen05_train64_validation64_72 \
  --train-rows 64 \
  --eval-start 64 \
  --eval-rows 8 \
  --source-code-dim 64 \
  --oracle-steps 16 \
  --epochs 5 \
  --lr 4e-4 \
  --contrastive-weight 0.2 \
  --bootstrap-samples 500 \
  --run-date 2026-05-04
```

## Results

| Run | Source code | Source-oracle acc | Mean-slot acc | Best destructive acc | Source-oracle KL | Mean-slot KL | Best destructive KL | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train16 -> val64:72 | 16 fp16 values / 32 B | 0.125000 | 0.250000 | 0.375000 | 0.118768 | 0.141946 | 0.107457 | fail |
| train64 -> val64:72 | 64 fp16 values / 128 B | 0.250000 | 0.250000 | 0.250000 | 0.124152 | 0.115766 | 0.107452 | fail |

The 16-row run produced a KL gain versus the mean target prefix, but the answer
surface got worse and candidate derangement beat the method on accuracy. The
64-row run removed the rank objection but still tied zero-source accuracy and
lost on KL to both mean target slots and candidate-roll source-code control.

Source headroom remains visible: source top1/top2 oracle accuracy was
`0.500000` on this 8-row slice, matching full-prompt Qwen accuracy. The learned
source-code prefix did not extract that headroom.

## Decision

Weaken this implementation of source-hidden-to-oracle-prefix distillation. The
target soft-prefix interface remains alive, but a generic projected
source-hidden MLP is not a publishable positive method. Do not widen this
exact branch to the terminal-tail `9216:9344` gate without a sharper
source-specific mechanism.

Promoted next branch: target-error-conditioned repair or ECC/syndrome-style
top1/top2 ambiguity coding, where the source packet is trained to act only on
target/source disagreement cases and is audited with wrong-row, candidate-roll,
target-derived, and same-byte controls.

## What Can Be Cut

- Cut broad claims that source hidden vectors can directly predict target
  resonance prefixes.
- Keep this as a negative diagnostic proving the oracle-capacity bottleneck is
  encoder/source-specificity, not target reachability.
- Keep the source-top1/top2 oracle row because it shows residual headroom, but
  do not claim it is solved.

## Next Exact Gate

Implement a target-error-only receiver gate:

1. train only on rows where target-only prefix disagrees with source top1/top2
   or full-prompt target;
2. send a small source syndrome/ECC code, not a dense hidden projection;
3. freeze the receiver and evaluate a fresh validation slice with wrong-row,
   zero-source, candidate-roll, target-derived, source-index/rank/score, and
   same-byte text/code controls;
4. only if it passes on a small slice, move to the subagent-recommended
   terminal-tail validation `9216:9344`, five seeds, and adversarial
   best-control paired bootstrap.

## Validation

```bash
./venv_arm64/bin/python -m pytest tests/test_build_target_self_resonance_hellaswag_source_oracle_distill_gate.py
./venv_arm64/bin/python -m py_compile scripts/build_target_self_resonance_hellaswag_source_oracle_distill_gate.py
```

Both passed before the experiment runs.
