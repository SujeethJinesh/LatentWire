# HellaSwag Qwen-To-Phi Held-Out Uncertainty Router Gate

Date: 2026-05-04

## Status

- ICLR full paper: not ready.
- COLM workshop: still plausible as a conservative source-private packet paper,
  but only if the method story is scoped around fixed packet transfer and
  rigorous controls.
- Current story: Qwen-to-Phi has real source top-1/top-2 oracle headroom, but
  shallow learned receivers do not convert it into reliable cross-family
  overrides.
- Exact blocker: no train-only receiver beats fixed Qwen hybrid on the frozen
  `1024:2048` HellaSwag surface with paired uncertainty and source-destroying
  controls.

## Gate

This gate formalizes the stronger held-out uncertainty-router branch requested
after the top-2/rival codebook failure. The source-side encoder may inspect
Qwen scores, but the receiver-visible packet is only:

- hybrid, Qwen top-1, Qwen top-2, and Qwen mean candidate IDs;
- Qwen margin, entropy, top1-vs-hybrid gap, and selected-margin bins.

The receiver combines those packet fields with Phi-local scores. It is trained
on official HellaSwag train calibration rows, selected on official-train dev,
and frozen before evaluating the cached validation `1024:2048` split. It does
not see source text, source KV, source hidden vectors, or raw source logits.

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_qwen_to_phi_heldout_uncertainty_router_gate.py \
  --output-dir results/source_private_hellaswag_qwen_to_phi_heldout_uncertainty_router_gate_20260504_validation1024_2048 \
  --bootstrap-samples 5000 \
  --run-date 2026-05-04
```

## Result

The gate fails.

| metric | value |
|---|---:|
| official-train calibration rows | `1487` |
| official-train fit/dev rows | `1115 / 372` |
| eval rows | `768` |
| fixed Qwen hybrid | `0.467448` |
| held-out uncertainty router | `0.466146` |
| delta vs fixed hybrid | `-0.001302` |
| CI95 low vs fixed hybrid | `-0.003906` |
| helps / harms / overrides | `0 / 1 / 3` |
| source top-1 | `0.411458` |
| source top-2 | `0.264323` |
| source top-1/top-2 oracle | `0.675781` |
| raw source-score logit fusion control | `0.391927` |
| best destructive control | `target_derived_source_packet_router_control`, `0.467448` |

The official-dev selector found a tiny positive held-out dev rule:
`+0.005376` delta with CI low `0.000000` and `2` helps / `0` harms. On frozen
eval, the same rule made three overrides and produced `0` helps / `1` harm.
The eval-label best-threshold diagnostic is also not positive, so this is not
just a threshold-selection issue.

Slice stability is also negative/tied:

| slice | rows | fixed | router | delta |
|---:|---:|---:|---:|---:|
| `1024` | `384` | `0.486979` | `0.484375` | `-0.002604` |
| `1536` | `384` | `0.447917` | `0.447917` | `0.000000` |

## Decision

Demote the shallow uncertainty-router family on this Qwen-to-Phi surface. The
sequence is now consistent:

1. raw-ish official-train receiver-calibrated linear rule: near tie but harms;
2. harm-controlled quantized buckets: no safe overrides;
3. top-2/rival codebook: harms and source-row shuffle competes;
4. held-out quantized uncertainty router: harms despite positive dev signal.

The source top-1/top-2 oracle remains large, so the problem is not lack of
source information. The problem is that coarse candidate/uncertainty packets do
not tell Phi which candidate to trust robustly enough. For ICLR, the next
positive-method branch should shift away from score-level switchers and toward
a target-native latent receiver:

- train a held-out encoder into target self-resonance soft-prefix slots; or
- learn a decision-supervised sparse/common dictionary with atom-shuffle,
  wrong-row, target-derived, and knockout controls.

## Lay Explanation

We gave Phi a tiny Qwen message that says, roughly, "my best answer is A, my
backup is B, and I am this confident." Phi learned on training examples when to
trust that message. On the real held-out slice, the learned rule changed three
answers and made one of them worse, so this message format is not strong enough
for the paper.

## Cut Or Keep

Keep as a rigorous negative diagnostic and reviewer-facing control. Cut it
from the main method story unless the final paper has a short ablation table
showing why score-level packet routers were abandoned.
