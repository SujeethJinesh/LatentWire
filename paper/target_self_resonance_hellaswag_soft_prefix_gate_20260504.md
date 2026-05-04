# Target Self-Resonance HellaSwag Soft-Prefix Gate

Date: 2026-05-04

## Readiness Status

- ICLR full paper: not ready, but the resonance branch is now alive.
- COLM workshop: stronger than before because we now have a positive target-side
  capacity result plus explicit novelty boundaries.
- Exact blocker: this is still a per-example oracle soft-prefix fit. It must be
  converted into a held-out encoder that emits the prefix from allowed source or
  public inputs and beats a zero-source target-slot baseline.

## Paper Story Update

The shallow Qwen-to-Phi receiver family is saturated: protected rival packets,
source dictionaries, receiver-calibrated linear rules, and harm-controlled
buckets do not turn the large oracle into a positive cross-family method. The
new live branch is target resonance:

1. Prove the frozen target can be driven into its own full-context decision
   state with a compact continuous prefix when the context text is removed.
2. Train a held-out encoder into the same target-native slots.
3. Add source-conditioned information only after the zero-source target-slot
   baseline is strong and explicit.

This is the prerequisite test for the user's resonance framing: do compact
continuous messages have enough control authority over the target model to
recreate behaviorally useful states?

## What Changed

- Added `scripts/build_target_self_resonance_hellaswag_soft_prefix_gate.py`.
- Added `tests/test_build_target_self_resonance_hellaswag_soft_prefix_gate.py`.
- Reused the repo's existing `inputs_embeds` multiple-choice scoring pattern
  from `scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`.
- Used the new literature memo:
  `references/696_target_self_resonance_interface_refs_20260504.md`.

The gate removes the HellaSwag context and scores each candidate continuation
conditioned on:

- full prompt context;
- a non-learned chunk-mean prompt prefix;
- a per-example optimized soft prefix;
- zero prefix;
- random same-norm prefix;
- shuffled optimized prefix from another row;
- candidate-score derangement.

The optimized prefix is trained only to match the frozen target model's
full-prompt choice distribution, not the gold label.

## Commands

```bash
./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_soft_prefix_gate.py \
  --output-dir results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation0_16 \
  --row-start 0 \
  --row-limit 16 \
  --prefix-len 8 \
  --steps 30 \
  --lr 0.005 \
  --top1-weight 0.0 \
  --norm-weight 0.001 \
  --bootstrap-samples 1000 \
  --device auto \
  --dtype float32 \
  --max-length 256 \
  --max-mean-kl 0.05

./venv_arm64/bin/python scripts/build_target_self_resonance_hellaswag_soft_prefix_gate.py \
  --output-dir results/target_self_resonance_hellaswag_soft_prefix_gate_20260504_qwen05_validation16_32 \
  --row-start 16 \
  --row-limit 16 \
  --prefix-len 8 \
  --steps 30 \
  --lr 0.005 \
  --top1-weight 0.0 \
  --norm-weight 0.001 \
  --bootstrap-samples 1000 \
  --device auto \
  --dtype float32 \
  --max-length 256 \
  --max-mean-kl 0.05
```

## Results

| Slice | Pass | Optimized Full Agreement | Optimized Mean KL | Chunk Agreement | Chunk Mean KL | Best Destructive Agreement | Best Destructive Mean KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation `0:16` | `true` | `0.937500` | `0.000853` | `0.500000` | `0.126353` | `0.625000` | `0.153753` |
| validation `16:32` | `true` | `0.812500` | `0.013037` | `0.375000` | `0.131362` | `0.500000` | `0.135498` |

The optimized 8-token prefix is `14,336` raw fp16 bytes for Qwen2.5-0.5B's
embedding dimension. That is not yet a deployable packet; it is a capacity
probe that shows the target has a compact continuous control surface.

## Interpretation

Promoted: target self-resonance is a real branch. The target can often be
steered back toward its own full-context multiple-choice state with only 8
optimized soft tokens plus a fixed anchor, even when the original context text
is removed.

Still blocked: the result is per-example oracle optimization. It does not show
that a learned encoder generalizes, nor that another model can populate these
slots with source-private information.

Saturated: shallow score/rank receiver switching remains weakened. These new
results do not revive it.

Cut if needed: do not overclaim universal latent language, prefix-tuning
novelty, or systems superiority from this gate. Keep it as the prerequisite
capacity/probing experiment for a target-native communication interface.

## Lay Explanation

We first ask Qwen to answer a HellaSwag question normally. Then we delete the
question text and give Qwen only a few learned soft tokens. Those soft tokens
are adjusted until Qwen is almost as ready to pick the same answer as it was
with the full question. The test passed on two tiny slices, which means the
model can be pushed back into a similar useful state with a compact learned
signal. The next challenge is to make another model produce that signal without
cheating.

## Next Exact Gate

Train a held-out text-to-prefix encoder on official HellaSwag train rows:

- target frozen;
- prefix length fixed at 8 or 16;
- compare source-present against zero-source target-slot, chunk-mean, random,
  shuffled-row, and candidate-deranged controls;
- report held-out full-prompt agreement, KL, gold accuracy, paired CIs, and
  prefix byte/latency accounting;
- then repeat with Phi as target if Mac runtime is acceptable, otherwise write
  a local NVIDIA runbook.

Hooke's Phi cache-only target self-compression gate remains a cheap Phi floor,
but the stronger ICLR path is now a real `inputs_embeds` encoder into these
target-native slots.
