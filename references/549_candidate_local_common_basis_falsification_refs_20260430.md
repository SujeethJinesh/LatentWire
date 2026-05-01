# Candidate-Local Common-Basis Falsification References

- date: `2026-04-30`
- purpose: primary-source memo for the common-basis falsification table attached
  to the live candidate-local residual receiver.

## Claim Boundary

The common-basis table supports a narrow claim:

> Global public-anchor, orthogonal Procrustes, ridge CCA/SVCCA-style,
> LSTIRP-lite inverse-relative, Sinkhorn OT, and Gromov-Wasserstein transport
> decoders are not enough for source-private communication under strict
> destructive controls; the receiver-side candidate-local residual chart is the
> repeated n512 survivor.

It does not claim to defeat every anchor/translation method. The next ICLR gate
still needs the RR holdout-to-core gap and native cache/KV baselines resolved;
RR remains a live partial competitor. Two simple RR innovation repairs have now
been measured and pruned, so future RR work should require a new hypothesis
rather than another minor scoring tweak.

## Closest Prior Art

- Relative Representations use anchors to express representations in a common
  relative coordinate system and explicitly motivate zero-shot latent-space
  communication. This is the closest common-basis threat.
  Source: https://arxiv.org/abs/2209.15430
- LSTIRP / latent-space translation via relative representations further
  narrows any broad "latent translation" claim. LatentWire must emphasize
  source-private packets and local residual decoding, not anchor coordinates
  alone. The current same-slice LSTIRP-lite inverse-relative packet row is
  measured and unsafe under controls, but this is not a full dense LSTIRP
  reproduction.
  Source: https://arxiv.org/abs/2406.15057
- Orthogonal Procrustes is a direct linear common-basis baseline. On the
  current surface it is measured and unsafe because the permuted-teacher
  control ties the matched row.
  Source: https://web.stanford.edu/class/cs273/refs/procrustes.pdf
- Sinkhorn distances give the cheap entropic OT solver used for the feature-cost
  public transport row. On this surface, both dot and local-stack variants are
  measured and unsafe because the permuted-teacher receiver ties the matched
  transport rows.
  Source:
  https://papers.neurips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport
- Gromov-Wasserstein alignment motivates relational-geometry transport between
  spaces without registered coordinates. On this surface, the GW packet rows
  are measured and unsafe under controls.
  Source: https://aclanthology.org/D18-1214/
- Wasserstein Procrustes is another embedding-alignment reference for jointly
  estimating transport/permutation and orthogonal structure. It supports the
  claim boundary that LatentWire is not simply reviving classical embedding
  alignment.
  Source: https://arxiv.org/abs/1805.11222
- CCA, SVCCA, CKA, and related representation-similarity tools are natural
  linear/subspace common-basis baselines. Ridge CCA/SVCCA-style canonical
  coordinates are now measured on the current surface and fail under the strict
  gate.
  Sources: https://arxiv.org/abs/1706.05806 and
  https://arxiv.org/abs/1905.00414
- C2C and KVComm/KVCOMM move or project model-internal KV/cache state. They are
  not the same as an 8B source-private packet, but they are the closest systems
  competitors for cross-model communication.
  Sources: https://openreview.net/forum?id=LeatkxrBCi,
  https://openreview.net/forum?id=F7rUng23nw, and
  https://arxiv.org/abs/2510.03346
- KV compression methods such as KIVI, KVQuant, QJL, and TurboQuant occupy the
  compressed-cache systems space. They are byte/latency baselines and caveats,
  not evidence against source-private packet communication unless coupled to a
  KV-sharing protocol.
  Sources: https://arxiv.org/abs/2402.02750,
  https://arxiv.org/abs/2401.18079, https://arxiv.org/abs/2406.03482, and
  https://arxiv.org/abs/2504.19874

## Result-Guided Implication

The current n512 common-basis falsification table should be used as a mechanism
ablation:

- global public-anchor dot product: high matched accuracy, but invalidated by
  destructive-control leakage on `9/9` n512 rows;
- public-calibration orthogonal Procrustes dot product: high matched accuracy
  up to `0.875`, but invalidated by destructive-control leakage on `9/9` n512
  rows; the best control also reaches `0.875`;
- ridge CCA/SVCCA-style canonical-coordinate dot product: `0/9` pass rows;
  core-to-holdout reaches `0.750` but the permuted-teacher control ties it, and
  holdout-to-core collapses to target;
- ridge CCA/SVCCA-style local residual stack: `0/9` pass rows; it recovers
  holdout-to-core matched accuracy to `0.500` but leaks the permuted-teacher
  control in every row;
- LSTIRP-lite inverse-relative dot product: `0/9` pass rows; matched accuracy
  reaches `0.500`, but the permuted-teacher control ties the improved
  core-to-holdout and same-family rows;
- LSTIRP-lite inverse-relative local residual stack: `0/9` pass rows; same
  matched ceiling (`0.500`) and same control leak pattern;
- Sinkhorn OT public transport: `0/9` pass rows; dot scoring reaches matched
  accuracy `1.000`, but the permuted-teacher control also reaches `1.000`
  (`9/9` control-leak rows);
- Sinkhorn OT local residual stack: `0/9` pass rows; matched accuracy reaches
  `0.625`, but the permuted-teacher control ties it (`9/9` control-leak rows);
- Gromov-Wasserstein public transport: `0/9` pass rows; dot scoring reaches
  matched accuracy `1.000`, but the permuted-teacher control also reaches
  `1.000` (`9/9` control-leak rows);
- Gromov-Wasserstein local residual stack: `0/9` pass rows; matched accuracy
  reaches `0.625`, but the permuted-teacher control ties it (`9/9`
  control-leak rows);
- Relative Representations-style anchor-coordinate dot product: clean controls
  and strong matched accuracy in core-to-holdout/same-family, but only `6/9`
  rows pass because holdout-to-core collapses to target;
- RR-anchor plus local residual normalization: `0/9` rows pass because the
  stack reintroduces destructive-control leakage;
- RR-anchor innovation residual normalization: `3/9` rows pass, but only in
  core-to-holdout; holdout-to-core remains at target and `6/9` rows leak through
  the permuted-teacher receiver;
- ranked RR-anchor innovation residual normalization: `0/9` rows pass;
  holdout-to-core falls below target (`0.125`) and core-to-holdout still leaks
  controls;
- candidate-local residual without normalization: partial survivor, but fails
  the repeated gate with only `3/9` seed/direction rows passing;
- candidate-local residual with row/payload normalization: `9/9` live n512 rows
  pass.

Safe claim:

> On the held-out n512 gate, the live receiver beats the unsafe global-dot,
> Procrustes, ridge CCA/SVCCA-style, and LSTIRP-lite inverse-relative
> Sinkhorn OT, and Gromov-Wasserstein common-basis/transport ablations and is
> the only all-direction repeated survivor. A true relative-anchor baseline is a
> partial clean competitor and should be reported as alive rather than defeated;
> simple RR innovation repairs have been tried and pruned.

Unsafe claim:

> This is not yet a full comparison against dense latent-transfer RR/LSTIRP,
> C2C, or KVComm.
