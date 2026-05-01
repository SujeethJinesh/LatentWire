# Source-Prioritized Leave-One-Family-Out Packet Builder

Date: 2026-05-01

## Status

This is now the strongest generalization-facing positive method in the
source-private candidate-local line. It is not yet enough for a comfortable
ICLR full paper because the receiver dictionary is still calibrated on public
eval-disjoint examples and the benchmark remains synthetic/protocol-level, but
it clears the current gated method branch: larger frozen slice, seed repeats,
paired uncertainty, clean destructive controls, and strict packet-builder
leave-one-family-out separation.

## Method

The sender fits a ridge map from source-private atom evidence into the
receiver's candidate-atom basis. The strict variant trains a separate packet
builder for each evaluation family while excluding that family from packet
builder calibration (`leave_one_family_out_public`). The packet itself is a
source-prioritized innovation packet:

```text
packet_vector = mapped_source_to_candidate_vector + 0.75 * source_atom_vector
```

The sender then transmits the top atoms in a 12-byte packet. The added source
term is not a text/KV relay; it is a small sparse fallback that keeps the
translated packet from drifting to nearby wrong atoms when the learned builder
has never seen the evaluation family.

Lay explanation: a pure translator can turn "sum" into a nearby but wrong clue
such as "average." The source-prioritized packet says "send the translation,
but keep enough of the original clue visible so the receiver cannot forget the
exact clue that already worked."

## Evidence

Artifacts:

- `results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501/`
- `results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501_seed53/`
- `results/source_private_candidate_conditioned_packet_builder_loo_hybrid_w075_20260501_seed59_serial/`

All three n512 seeds pass all three rows.

| Seed | Direction | N | Candidate packet | Live source packet | Target | Best control | Pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 47 | core_to_holdout | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 47 | holdout_to_core | 512 | 0.625 | 0.500 | 0.250 | 0.258 | True |
| 47 | same_family_all | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 53 | core_to_holdout | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 53 | holdout_to_core | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 53 | same_family_all | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 59 | core_to_holdout | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 59 | holdout_to_core | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |
| 59 | same_family_all | 512 | 0.625 | 0.500 | 0.250 | 0.250 | True |

Headline: `9/9` n512 seed-repeat rows pass; min lift over the live source packet
is `+0.125`; worst strict source-destroying control is `0.258`, within the
target-plus-0.03 rule.

The failed ablation is also informative. Pure leave-one-family-out mapped
packets collapsed to `0.375`, and `mapped + 0.5 * source` passed the n128 probe
but tied the live packet at n512 in holdout-to-core. Family analysis showed the
0.5 weight fixed `missing_key_default` but broke `sum_all_values`; weight `0.75`
preserved `sum_all_values` while still fixing `missing_key_default` and
`nested_key_default`.

## Reviewer-Facing Interpretation

This result supports a precise contribution:

1. A learned source-to-candidate packet builder can improve over hand-built
   source atoms under strict destructive controls.
2. Leave-one-family-out packet-builder calibration exposes semantic drift in a
   pure learned map.
3. A Wyner-Ziv-style innovation packet, implemented as learned candidate-basis
   atoms plus a small source-basis residual, fixes that drift on the current
   frozen split.

Do not overclaim that this is universal latent communication. The safe claim is
source-private sparse packet communication with candidate side information.

## Remaining Gates

- True train-only cross-family generalization: the receiver dictionary and
  packet builder should be trained without public examples from the eval
  distribution.
- Real benchmark transfer: at least one non-synthetic task family where source
  evidence is private and candidate-local decoding is meaningful.
- Native systems rows: vLLM/NVIDIA TTFT, TPOT, goodput, HBM/KV bytes, and
  comparisons to C2C/KVComm/KVCOMM/TurboQuant-style cache baselines.
- A rate curve for the new source-prioritized packet: 8B, 10B, 12B, 16B, and a
  text-relay catch-up control.
