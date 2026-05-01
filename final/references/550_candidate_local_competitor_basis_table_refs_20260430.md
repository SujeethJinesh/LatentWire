# Candidate-Local Competitor/Basis Table References

- date: `2026-04-30`
- purpose: citation and claim-boundary memo for the candidate-local
  competitor/basis table attached to the live source-private packet receiver.

## Claim Boundary

The competitor table supports a narrow statement:

> On the current n512 held-out surface, the normalized candidate-local residual
> receiver survives strict source-destroying controls while the nearest
> implemented global common-basis, orthogonal Procrustes, ridge CCA/SVCCA-style,
> LSTIRP-lite inverse-relative, Sinkhorn OT, Gromov-Wasserstein, and
> no-normalization local-chart ablations do not. A Relative
> Representations-style anchor-coordinate decoder is a partial clean competitor
> and remains alive. Two simple RR innovation repairs are measured and pruned.

It does not claim that LatentWire has defeated all latent/cache communication
or quantized-KV systems baselines. Those remain explicit pending rows.

## Required Prior-Work Anchors

- Relative Representations express samples by similarity to anchors and
  explicitly motivate zero-shot latent-space communication and model stitching.
  This is the closest anchor-common-basis threat.
  Source: https://arxiv.org/abs/2209.15430
- LSTIRP uses relative representations for latent-space translation. It
  narrows any broad claim that anchor coordinates are new. The current table
  includes a same-slice LSTIRP-lite inverse-relative packet row, but not a full
  dense latent-transfer reproduction.
  Source: https://arxiv.org/abs/2406.15057
- Orthogonal Procrustes is a measured public-calibration shared-space packet
  row in the current table. It should not be called CCA.
  Source: https://web.stanford.edu/class/cs273/refs/procrustes.pdf
- SVCCA and CKA are standard representation-comparison tools; ridge CCA/SVCCA
  rows are now measured linear alignment baselines on this surface.
  Sources: https://arxiv.org/abs/1706.05806 and
  https://arxiv.org/abs/1905.00414
- Gromov-Wasserstein alignment is a natural nonlinear/metric transport
  baseline for cross-space matching. The current table includes same-slice GW
  packet rows and marks them unsafe because destructive controls tie the matched
  rows.
  Source: https://aclanthology.org/D18-1214/
- Sinkhorn distances provide the entropic OT solver used for the feature-cost
  transport baseline. The current table includes same-slice Sinkhorn OT packet
  rows and marks them unsafe because the permuted-teacher receiver ties the
  matched rows.
  Source:
  https://papers.neurips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport
- Wasserstein Procrustes is relevant prior art for embedding alignment through
  transport/permutation plus orthogonal structure.
  Source: https://arxiv.org/abs/1805.11222
- C2C directly projects and fuses source KV cache into target KV cache, so it
  owns the broad cache-to-cache communication comparison.
  Source: https://arxiv.org/abs/2510.03215
- KVComm and Q-KVComm share or compress KV state for multi-agent
  communication. They are source-KV-exposing systems competitors rather than
  privacy-equivalent packet baselines.
  Sources: https://arxiv.org/abs/2510.03346 and
  https://arxiv.org/abs/2512.17914
- TurboQuant, KIVI, KVQuant, and CacheGen are required systems byte-floor and
  latency caveats for compressed KV state.
  Sources: https://arxiv.org/abs/2504.19874,
  https://arxiv.org/abs/2402.02750, https://arxiv.org/abs/2401.18079, and
  https://arxiv.org/abs/2310.07240

## Result-Guided Implication

Measured same-slice rows now separate:

- live normalized candidate-local residual receiver: `9/9` n512 rows pass;
- global public-anchor dot product: `0/9` rows pass because destructive
  controls leak;
- public-calibration orthogonal Procrustes: `0/9` rows pass because destructive
  controls leak; matched accuracy reaches `0.875`, but the best control also
  reaches `0.875`;
- ridge CCA/SVCCA-style canonical coordinates: `0/9` rows pass; matched
  accuracy reaches `0.750`, but controls leak and holdout-to-core stays at
  target;
- ridge CCA/SVCCA-style local residual stack: `0/9` rows pass; matched
  accuracy reaches `0.750`, but the permuted-teacher control ties every row;
- LSTIRP-lite inverse-relative dot product: `0/9` rows pass; matched accuracy
  reaches `0.500`, but the permuted-teacher control ties the improved rows;
- LSTIRP-lite inverse-relative local residual stack: `0/9` rows pass; matched
  accuracy again reaches `0.500`, with the same control leak pattern;
- Sinkhorn OT public-calibration transport: `0/9` rows pass; matched accuracy
  reaches `1.000`, but the permuted-teacher control also reaches `1.000`;
- Sinkhorn OT local residual stack: `0/9` rows pass; matched accuracy reaches
  `0.625`, but the permuted-teacher control ties every row;
- Gromov-Wasserstein public-calibration transport: `0/9` rows pass; matched
  accuracy reaches `1.000`, but the permuted-teacher control also reaches
  `1.000`;
- Gromov-Wasserstein local residual stack: `0/9` rows pass; matched accuracy
  reaches `0.625`, but the permuted-teacher control ties every row;
- Relative Representations-style anchor-coordinate dot product: `6/9` rows
  pass with clean controls but all holdout-to-core rows collapse to target;
- RR-anchor plus local residual normalization: `0/9` rows pass because controls
  leak again;
- RR-anchor innovation residual normalization: `3/9` rows pass, only in
  core-to-holdout, and controls leak in `6/9` rows;
- ranked RR-anchor innovation residual normalization: `0/9` rows pass, with
  holdout-to-core below target;
- candidate-local residual without row/payload normalization: only `3/9` rows
  pass;
- 8B structured text/log prefix: target floor on the measured rows.

Safe claim:

> The live receiver is not merely a global shared dictionary, an orthogonal
> public rotation, a ridge CCA/SVCCA shared subspace, a LSTIRP-lite
> inverse-relative projection, a Sinkhorn/GW public transport map, or an
> unnormalized local chart. Controls break those ablations on the same
> seeds/directions, but RR anchor coordinates are strong enough to become the
> next stackable baseline only if a future hypothesis goes beyond the simple
> innovation repairs already pruned here.

Unsafe claim:

> The current table is not yet a full defeat of Relative Representations,
> full dense LSTIRP, C2C, KVComm, or KV compression systems.
