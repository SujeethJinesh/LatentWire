# Source-Codebook Candidate Repair Gates

## Status

Current paper readiness remains below ICLR full-paper standard. These gates
tested whether the current source-conditioned branch can move from a source
top-choice packet toward a learned receiver-side candidate repair method.

Outcome: fail, with useful headroom evidence.

## Gates

Target self-resonance TinyLlama-to-Qwen smoke:

- script:
  `scripts/build_target_self_resonance_hellaswag_source_codebook_candidate_repair_gate.py`;
- tests:
  `tests/test_build_target_self_resonance_hellaswag_source_codebook_candidate_repair_gate.py`;
- artifact:
  `results/target_self_resonance_hellaswag_source_codebook_candidate_repair_gate_20260504_tiny_to_qwen05_train64_validation72_80/`.

Larger cached Qwen-to-Phi top-2/rival gate:

- script:
  `scripts/build_source_private_hellaswag_qwen_to_phi_top2_rival_codebook_gate.py`;
- tests:
  `tests/test_build_source_private_hellaswag_qwen_to_phi_top2_rival_codebook_gate.py`;
- artifact:
  `results/source_private_hellaswag_qwen_to_phi_top2_rival_codebook_gate_20260504_validation1024_2048/`.

References:
`references/704_source_codebook_candidate_repair_refs_20260504.md`.

## Results

Tiny target self-resonance codebook:

| condition | accuracy | note |
|---|---:|---|
| frozen target slots | 0.375000 | target-only compressed baseline |
| source codebook repair | 0.500000 | weak lift over frozen |
| source top-1 label control | 0.750000 | shortcut still stronger |
| source top-1/top-2 oracle | 1.000000 | large unresolved routing headroom |

The method improves over frozen target slots by `+0.125000`, but the paired
CI95 low is `0.000000` and it is worse than direct source-top1. It also has a
candidate-roll source-codebook control at `0.625000`, so the matched source is
not cleanly isolated.

Larger cached Qwen-to-Phi top-2/rival codebook:

| condition | accuracy | delta vs fixed hybrid |
|---|---:|---:|
| fixed Qwen hybrid | 0.467448 | 0.000000 |
| top-2/rival codebook | 0.460938 | -0.006510 |
| source-row shuffle codebook | 0.468750 | +0.001302 |
| Qwen candidate-only | 0.455729 | -0.011719 |
| Phi target-only | 0.263021 | -0.204427 |
| source top-1 label control | 0.411458 | -0.055990 |
| source top-1/top-2 oracle | 0.675781 | +0.208333 |

The larger gate kills the naive protected bucket/codebook implementation: it
overrides only `13` rows, helps `2`, harms `7`, and loses to fixed hybrid.
The source-row shuffle control slightly exceeds both method and fixed hybrid,
so any apparent learned switching is not attributable to matched source
communication.

## Decision

Demote naive score-level codebook repair and protected top-2/rival bucket
switching. The source top-1/top-2 pair still has large oracle headroom, but
simple receiver-local buckets do not extract it reliably.

Promote only the broader question:

```text
Can a learned uncertainty/router or latent codebook choose among
source top-1, source top-2, and target-local evidence without collapsing
to source-copy, logit fusion, or target-only controls?
```

The next method must either use a stronger learned router with held-out
calibration and source-row shuffle controls, or move back into a target-latent
receiver where the source code changes the target model's state rather than
only adding candidate-level score priors.

## Lay Explanation

We tested whether the source model could send a tiny code saying which two
answers looked plausible, and whether the target-side receiver could safely
use that code to fix mistakes. The answer is not yet: an oracle that magically
chooses between the two candidates would help a lot, but the learned codebook
does not know when to trust the first candidate, the second candidate, or the
target model.

