# Train-Only Receiver-Basis Probe

Status: weakened, not promoted.

## Question

Can the remaining ICLR blocker be cleared by replacing the public eval-disjoint
receiver dictionary with a train-only receiver basis while keeping the same
source-private packet contract?

Lay description: the sender sends a tiny hint about what went wrong in a hidden
test. The receiver sees four candidate fixes. This probe asks whether the
receiver can understand that hint using only training-family examples, rather
than a public dictionary built from eval-family-like candidate surfaces.

## Probe 1: Existing Semantic-Anchor Receiver

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir .debug/trainonly_receiver_existing_semantic_anchor_gate_20260501 \
  --budgets 12 --train-examples 256 --eval-examples 128 --seed 47 \
  --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress \
  --candidate-calibration train_only --calibration-examples 256 \
  --feature-dim 384 --ridge 0.05 --top-k 8 --min-score 0.0 \
  --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher \
  --decoder-score-mode candidate_local_residual_norm --min-decision-score 0.48
```

Result: top-level gate failed.

| Direction | Matched | Target | Best control | Pass |
|---|---:|---:|---:|---|
| core_to_holdout | 0.625 | 0.250 | permuted_teacher_receiver=0.375 | false |
| holdout_to_core | 0.500 | 0.250 | private_random_source_atoms=0.258 | true |
| same_family_all | 0.750 | 0.250 | permuted_teacher_receiver=0.312 | false |

Interpretation: the semantic-anchor basis exposes useful source signal, but the
permuted-teacher receiver control rises above the clean-control band. This is
not yet a safe ICLR contribution.

## Probe 2: Train-Only Sender Plus Existing Semantic-Anchor Receiver

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py \
  --output-dir .debug/trainonly_receiver_existing_semantic_anchor_packet_builder_20260501 \
  --budgets 12 --train-examples 256 --eval-examples 128 --seed 47 \
  --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress \
  --candidate-calibration train_only --packet-builder-calibration train_only \
  --calibration-examples 256 --packet-builder-examples 256 \
  --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 \
  --top-k 8 --min-score 0.0 --packet-min-score 0.0 \
  --text-feature-mode semantic_anchor --adapter-target-mode semantic_anchor_teacher \
  --decoder-score-mode candidate_local_residual_norm \
  --packet-builder-composition add_source --source-identity-weight 0.75 \
  --min-decision-score 0.48 --bootstrap-samples 200
```

Result: top-level gate failed.

| Direction | Candidate packet | Base source packet | Target | Best control | Candidate - base | Pass |
|---|---:|---:|---:|---:|---:|---|
| core_to_holdout | 0.625 | 0.625 | 0.250 | 0.250 | 0.000 | false |
| holdout_to_core | 0.500 | 0.500 | 0.250 | 0.266 | 0.000 | false |
| same_family_all | 0.938 | 0.750 | 0.250 | 0.250 | 0.188 | true |

Interpretation: the train-only learned sender packet helps same-family rows but
does not beat the older source-atom packet cross-family under this receiver.

## Probe 3: Candidate-Local Innovation Centering

Change: added `candidate_local_innovation_residual_norm`, which subtracts the
candidate-pool mean from both candidate vectors and the payload before normalized
candidate-local scoring.

Command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_candidate_conditioned_packet_builder_smoke.py \
  --output-dir .debug/trainonly_receiver_candidate_local_innovation_packet_builder_20260501 \
  --budgets 12 --train-examples 256 --eval-examples 128 --seed 47 \
  --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress \
  --candidate-calibration train_only --packet-builder-calibration train_only \
  --calibration-examples 256 --packet-builder-examples 256 \
  --feature-dim 384 --ridge 0.05 --packet-builder-ridge 0.1 \
  --top-k 8 --min-score 0.0 --packet-min-score 0.0 \
  --text-feature-mode hf_last_mean --adapter-target-mode semantic_anchor_teacher \
  --decoder-score-mode candidate_local_innovation_residual_norm \
  --packet-builder-composition add_source --source-identity-weight 0.75 \
  --min-decision-score 0.48 --bootstrap-samples 100
```

Result: top-level gate failed.

| Direction | Candidate packet | Base source packet | Target | Best control | Candidate - base | Pass |
|---|---:|---:|---:|---:|---:|---|
| core_to_holdout | 0.375 | 0.375 | 0.250 | 0.250 | 0.000 | false |
| holdout_to_core | 0.375 | 0.375 | 0.250 | 0.250 | 0.000 | false |
| same_family_all | 0.812 | 0.375 | 0.250 | 0.266 | 0.438 | true |

Interpretation: centering the payload against the receiver's candidate prior
amplifies same-family signal but collapses the cross-family margin. This branch
is weakened unless paired with a new control-blocking or basis-generalization
mechanism.

## Decision

Ruled out for promotion:

- existing semantic-anchor receiver as a standalone train-only receiver repair
- train-only sender plus existing semantic-anchor receiver
- candidate-local payload-innovation centering at the current 12B/0.48 operating point

Still alive:

- train-only sender source-prioritized packet with public eval-disjoint receiver
- candidate-side innovation syndrome with an explicit control-blocking code
- receiver bases learned from non-eval public anchors with stricter candidate-only
  and permuted-teacher controls
- native NVIDIA/vLLM systems rows when hardware is available

Next exact gate: design a train-only receiver basis that passes the
permuted-teacher and candidate-only controls first on n128, then widen to the
n512 seed-repeat surface only if the cross-family rows show source-specific
lift.
