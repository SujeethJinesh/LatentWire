# Source-Private Public Adapter Held-Out Receiver

- date: `2026-04-30`
- status: `alive_but_not_passed`
- code:
  `scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`,
  `scripts/summarize_source_private_hf_embedding_heldout_packet_gate.py`
- results:
  `results/source_private_public_adapter_heldout_packet_gate_20260430/summary/`
- references:
  `references/546_public_adapter_heldout_receiver_refs_20260430.md`

## Current Readiness

COLM workshop readiness improves because this is a rigorous probe of whether
the semantic-anchor receiver can be learned from public calibration surfaces.
ICLR full-paper readiness is still blocked: the public adapter produces large
matched-packet lifts, but it does not clear the strict bidirectional held-out
gate and its permuted-teacher negative control passes some individual rows.

## Hypothesis

If the explicit semantic-anchor lexicon is only a convenient public basis, then
a receiver should be able to learn that basis from public calibration text. The
source still sends only byte-scale private atoms; the receiver uses frozen
MiniLM text features to predict public ontology coordinates for each candidate.

Layman version: we teach the receiver a practice dictionary from public
examples, then ask whether the same tiny private clue helps it pick the right
answer on new examples. We also scramble that dictionary as a negative control;
if the scrambled dictionary still works, the method is not clean enough.

## Implementation

Added `--adapter-target-mode` to the learned synonym dictionary gate:

- `native_atoms`: previous behavior.
- `semantic_anchor_teacher`: public calibration surfaces supervise frozen
  features to predict semantic-anchor atom coordinates.
- `permuted_semantic_anchor_teacher`: deterministic negative control that
  permutes the teacher coordinates.

The public adapter path is restricted to `atom_ridge`; contrastive and JEPA
receivers remain separate because they can use answer/source labels during
calibration. The direction audit now records
`calibration_eval_exact_id_overlap_count` and samples.

## Runs

All runs use seed `47`, `n=256`, candidate view `heldout_synonym`, calibration
view `synonym_stress`, MiniLM `hf_last_mean`, ridge `0.05`, and budgets
`2/4/8`.

| Run | Adapter target | Calibration | Top-k / threshold | Pass rows | Direction pass | Max acc | Max lift |
|---|---|---|---|---:|---|---:|---:|
| `source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher` | semantic teacher | all public | 20 / 0.20 | 0 | none | 0.875 | 0.625 |
| `source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_top8_dec040` | semantic teacher | all public | 8 / 0.40 | 1 | holdout_to_core | 0.875 | 0.625 |
| `source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_trainonly_top8_dec040` | semantic teacher | train only | 8 / 0.40 | 0 | none | 0.875 | 0.625 |
| `source_private_public_adapter_heldout_packet_gate_20260430_seed47_n256_minilm_teacher_permuted` | permuted teacher | all public | 20 / 0.20 | 2 | core_to_holdout, same_family | 0.625 | 0.375 |

Summary: `36` rows, `3` pass rows, `3` near misses, no bidirectional
cross-family pass.

## Interpretation

Promoted:

- Public teacher calibration is not dead. It can reach `0.875` matched-packet
  accuracy and `+0.625` over target on held-out synonym surfaces.
- The adapter makes the semantic-anchor dependency more learnable-looking than
  the previous frozen embedding baseline.

Weakened:

- The method is not yet clean enough for ICLR. Shuffled source, atom
  derangement, and private-random controls still rise above target on the best
  rows.
- The permuted-teacher control passing individual rows means current scoring can
  exploit broad source/candidate correlations, not only the intended shared
  ontology.

Ruled out for now:

- "Public adapter alone solves the semantic-anchor criticism."
- Expanding this exact adapter to more seeds before fixing controls.

## Next Gate

Implement a receiver-conditioned residual/codebook branch:

1. Build a local candidate frame from public candidate features.
2. Encode the source-private atoms as residual evidence in that candidate-local
   frame, rather than global atom IDs only.
3. Add candidate-local normalization or source-contrastive margins so shuffled
   source packets collapse.
4. Require the permuted-teacher/control run to fail as part of the pass rule.
5. Keep packet size at `<=8B` preferred, `<=16B` hard cap, with receiver p95
   accounting separated into offline calibration and online packet decode.

This is now the highest-priority method branch because it directly attacks the
observed failure mode: broad public ontology scores are useful, but they need a
candidate-conditioned decoder to distinguish matched source evidence from
plausible shuffled evidence.
