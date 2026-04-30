# Source-Private Contrastive Receiver Pruning

- date: `2026-04-30`
- rung: strict small receiver ablation
- status: pruned as headline contribution
- artifacts:
  - `results/source_private_contrastive_semantic_anchor_small_gate_20260430_n128_noneg/`
  - `results/source_private_contrastive_semantic_anchor_small_gate_20260430_n128_ctrlneg/`
  - `results/source_private_contrastive_semantic_anchor_small_gate_20260430_n128_noneg_threshold070/`
- incomplete oversized diagnostic: an n256 bilinear run was stopped after the
  first core->holdout 4-byte prediction files because the solve was too slow
  for an iterative Mac-local cycle. Those partial scratch files are not
  promoted or copied to `final/`.

## Readiness Snapshot

Current ICLR readiness: strong scoped positive-method paper, not yet a
comfortable broad latent-communication full paper. Estimated distance remains
one less protocol-shaped learned/model-mediated receiver, or a deliberate
scope-down around source-private side-information packets plus systems evidence.

Current story: the source sends a tiny private packet, and the target decodes it
with public side information. The promoted evidence is still the balanced
direct diagnostic packet, frozen Qwen binary verifier, semantic-anchor
held-out receiver, and packet trace-card systems row.

Exact blocker: the less hand-shaped learned contrastive receiver does not pass
bidirectional held-out cross-family controls.

## Layman Summary

This experiment asked whether we could replace the hand-built semantic-anchor
receiver with a learned compatibility scorer. The scorer sees candidate
descriptions and the source packet and learns which candidate should match. A
good result would mean the method is less like a lookup table. The failure is
specific: when the learned scorer is flexible, it sometimes accepts a broken
packet; when trained to reject broken packets, it becomes too conservative and
misses real packets.

## Commands

Unconstrained contrastive receiver:

```bash
./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_contrastive_semantic_anchor_small_gate_20260430_n128_noneg \
  --budgets 4 \
  --train-examples 256 \
  --eval-examples 128 \
  --seed 67 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public \
  --calibration-examples 256 \
  --feature-dim 128 \
  --text-feature-mode semantic_anchor \
  --receiver-mode contrastive_bilinear \
  --contrastive-negative-sources 0 \
  --ridge 0.25 \
  --top-k 8 \
  --min-score 0.05 \
  --min-decision-score 0.30
```

Source-control negative receiver:

```bash
./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_contrastive_semantic_anchor_small_gate_20260430_n128_ctrlneg \
  --budgets 4 \
  --train-examples 256 \
  --eval-examples 128 \
  --seed 67 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public \
  --calibration-examples 256 \
  --feature-dim 128 \
  --text-feature-mode semantic_anchor \
  --receiver-mode contrastive_bilinear \
  --contrastive-negative-sources 2 \
  --ridge 0.25 \
  --top-k 8 \
  --min-score 0.05 \
  --min-decision-score 0.30
```

Target-preservation threshold probe:

```bash
./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_contrastive_semantic_anchor_small_gate_20260430_n128_noneg_threshold070 \
  --budgets 4 \
  --train-examples 256 \
  --eval-examples 128 \
  --seed 67 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public \
  --calibration-examples 256 \
  --feature-dim 128 \
  --text-feature-mode semantic_anchor \
  --receiver-mode contrastive_bilinear \
  --contrastive-negative-sources 0 \
  --ridge 0.25 \
  --top-k 8 \
  --min-score 0.05 \
  --min-decision-score 0.70
```

## Results

| Variant | Pass | Direction pass | Max learned | Failure mode |
|---|---:|---|---:|---|
| no source-control negatives, threshold `0.30` | `false` | core false, holdout true, same true | `1.000` | atom-ID derangement reaches `0.375` in core->holdout |
| source-control negatives, threshold `0.30` | `false` | core false, holdout true, same false | `0.875` | controls fixed, but core->holdout matched drops to `0.375` |
| no negatives, threshold `0.70` | `false` | all false | `0.375` | conservative threshold kills matched source signal |

Detailed rows:

| Variant | Direction | N | Learned | Target | Best control | CI95 low | Oracle | Controls OK |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| no negatives | core_to_holdout | 128 | 0.750 | 0.250 | 0.375 | 0.414 | 0.875 | `false` |
| no negatives | holdout_to_core | 128 | 1.000 | 0.250 | 0.250 | 0.672 | 1.000 | `true` |
| no negatives | same_family_all | 128 | 0.875 | 0.250 | 0.258 | 0.539 | 0.938 | `true` |
| control negatives | core_to_holdout | 128 | 0.375 | 0.250 | 0.250 | 0.070 | 0.875 | `true` |
| control negatives | holdout_to_core | 128 | 0.875 | 0.250 | 0.258 | 0.539 | 1.000 | `true` |
| control negatives | same_family_all | 128 | 0.500 | 0.250 | 0.250 | 0.180 | 0.938 | `true` |
| threshold `0.70` | core_to_holdout | 128 | 0.250 | 0.250 | 0.250 | 0.000 | 0.875 | `true` |
| threshold `0.70` | holdout_to_core | 128 | 0.375 | 0.250 | 0.250 | 0.070 | 1.000 | `true` |
| threshold `0.70` | same_family_all | 128 | 0.312 | 0.250 | 0.250 | 0.023 | 0.938 | `true` |

The aborted n256 diagnostic is still informative but not promoted. The
completed partial core->holdout 4-byte files showed the same pattern:
unconstrained contrastive reached `0.875` with controls clean in that one
direction, while control negatives reached `0.500`. The full n256 run was
stopped because the bilinear solve was too slow for an iterative Mac-local
cycle.

## Interpretation

This does not weaken the promoted semantic-anchor receiver, which already
passed the medium held-out gate. It weakens a specific upgrade hypothesis:
plain bilinear contrastive compatibility is not enough to replace the explicit
semantic-anchor overlap receiver under bidirectional cross-family controls.

The branch should be pruned for now. Future work on this family needs a new
mechanism: orbit-trained invariant dictionaries, real model activations, or a
small target-preserving query bottleneck with explicit derangement/random
packet regularization.

## Next Gate

Do not spend another cycle tuning this bilinear receiver. The highest-value
next gate is a large frozen scale-up of the promoted balanced diagnostic packet
and frozen Qwen binary-verifier receiver, or a genuinely new learned receiver
using model activations/frozen embedding features with source-control negatives
and a held-out paraphrase split.
