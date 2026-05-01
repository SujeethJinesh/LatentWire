# Source-Private Anti-Shuffle Innovation Sender, 2026-05-01

## Status

- paper readiness: stronger ICLR evidence, but not yet a comfortable ICLR
  full-paper headline.
- current story: anti-shuffle innovation is the first unified train-only
  sender+receiver branch to clear a larger cross-family gate.
- exact blocker: the passing row uses eval nonoverlap donors as contrast
  examples. A stricter train-mean contrast variant keeps controls clean but
  does not yet beat the base packet in the hard direction.

## Method

The sender starts with the train-only source-to-candidate packet builder and
the train-only permuted-null receiver. Instead of transmitting the top mapped
atoms directly, it scores candidate packet atoms by source-specific receiver
utility:

```text
selection(atom) =
  source_specific_carrier(atom)
  * max(receiver_margin(atom), 0)
  - null_margin_penalty(atom)
  - train_generic_penalty(atom)
```

The target candidate used for selection is the receiver's top candidate under
the proposed packet, not the gold label. The packet therefore does not use
`answer_label` for selection. The diagnostic variant subtracts a nonoverlap
source donor from the current eval slice; the stricter variant uses the
train-set mean source/prediction vector as the contrast.

Lay explanation: the old sender sent clues that often helped even when the
clues came from the wrong source. Anti-shuffle tries to keep only the clue
pieces that help the real source more than a wrong-source or null-reader
version.

## Evidence

Artifacts:

- `.debug/iclr_20260501_trainonly_sender_receiver_antishuffle_innovation_seed47_n128/`
- `.debug/iclr_20260501_trainonly_sender_receiver_antishuffle_innovation_seed53_n128_budget12/`
- `.debug/iclr_20260501_trainonly_sender_receiver_antishuffle_innovation_seed59_n128_budget12/`
- `.debug/iclr_20260501_trainonly_sender_receiver_antishuffle_innovation_seed47_n512_budget12/`
- `.debug/iclr_20260501_trainonly_sender_receiver_trainmean_antishuffle_seed47_n128_budget12/`
- `.debug/iclr_20260501_trainonly_sender_receiver_trainmean_antishuffle_seed47_n128_budget12_w1p0/`

Eval-donor anti-shuffle passes all three n128 seed-repeat cross-family gates at
12B.

| Seed | Direction | N | Candidate | Base | Target | Best control | Pass |
|---:|---|---:|---:|---:|---:|---:|---|
| 47 | core_to_holdout | 128 | 0.750 | 0.625 | 0.250 | 0.266 | yes |
| 47 | holdout_to_core | 128 | 0.625 | 0.500 | 0.250 | 0.250 | yes |
| 53 | core_to_holdout | 128 | 0.750 | 0.625 | 0.250 | 0.266 | yes |
| 53 | holdout_to_core | 128 | 0.625 | 0.500 | 0.250 | 0.258 | yes |
| 59 | core_to_holdout | 128 | 0.750 | 0.625 | 0.250 | 0.266 | yes |
| 59 | holdout_to_core | 128 | 0.625 | 0.500 | 0.250 | 0.266 | yes |

The larger seed-47 n512 gate also passes bidirectional cross-family:

| Direction | N | Candidate | Base | Target | Best control | CI95 low vs base | Pass |
|---|---:|---:|---:|---:|---:|---:|---|
| core_to_holdout | 512 | 0.750 | 0.625 | 0.250 | 0.271 | 0.100 | yes |
| holdout_to_core | 512 | 0.625 | 0.500 | 0.250 | 0.256 | 0.072 | yes |
| same_family_all | 512 | 0.938 | 0.812 | 0.250 | 0.312 | 0.088 | no |

Same-family remains unpromoted because `structured_text_matched` reaches
`0.3125` against a target-only baseline of `0.250`, even though the candidate
packet reaches `0.9375`.

## Stricter Contrast Check

The train-mean contrast variant does not pass:

| Variant | Direction | Candidate | Base | Best control | Interpretation |
|---|---|---:|---:|---:|---|
| train-mean, source weight 0.75 | core_to_holdout | 0.750 | 0.625 | 0.250 | clean positive |
| train-mean, source weight 0.75 | holdout_to_core | 0.500 | 0.500 | 0.250 | too conservative |
| train-mean, source weight 1.0 | core_to_holdout | 0.875 | 0.625 | 0.281 | leaks private-random control |
| train-mean, source weight 1.0 | holdout_to_core | 0.500 | 0.500 | 0.375 | leaks shuffled-source control |

Interpretation: the anti-shuffle idea is alive, but the final ICLR method
should train against sampled train-set donors rather than use the eval
nonoverlap donor or a single train mean.

## Novelty Boundary

This is not new coding theory. Slepian-Wolf and Wyner-Ziv already frame coding
with decoder side information; QJL and TurboQuant already provide strong
residual/sketch/quantization precedents; C2C, KVComm, and KVCOMM are direct
systems competitors for KV/cache communication.

Safe claim: anti-shuffle innovation is a controlled LLM packet-selection
instantiation for source-private, byte-scale candidate disambiguation. It is
different from C2C/KVComm because it does not expose source text or source KV
cache state.

Primary sources:

- C2C: https://arxiv.org/abs/2510.03215
- KVComm: https://arxiv.org/abs/2510.03346
- KVCOMM: https://arxiv.org/abs/2510.12872
- TurboQuant: https://arxiv.org/abs/2504.19874
- QJL: https://arxiv.org/abs/2406.03482
- ARC-Challenge public benchmark bridge: https://arxiv.org/abs/1803.05457

## Next Gate

Implement sampled train-donor anti-shuffle:

```text
score(atom) =
  matched_receiver_gain(atom)
  - base_source_gain(atom)
  - lambda * E_train_nonoverlap_donor[donor_gain(atom)]
  - mu * null_receiver_gain(atom)
```

Run seed-47 n128 first, with 8-16 train nonoverlap donors per example. Promote
only if it preserves the eval-donor row's `holdout_to_core` improvement while
keeping shuffled-source, private-random, and permuted-teacher controls inside
the target band.

## Follow-Up

See `paper/source_private_train_donor_antishuffle_sender_20260501.md`.
Sampled train-donor anti-shuffle now clears the n128 cross-family seed-repeat
frontier at 12-14B and passes the larger seed-47 n512 gate. This removes the
main eval-donor caveat from the anti-shuffle branch. The remaining blocker is
not this eval-donor issue; it is now n512 seed-repeat coverage, public
benchmark transfer, and native GPU systems evidence.
