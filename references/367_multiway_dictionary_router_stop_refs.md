# Multi-Way Dictionary, Router, and Stop References

Scope: multi-way alignment, shared sparse dictionaries / crosscoders, stable
router objectives, and bounded stop policies for the current cross-model
communication paper loop.

## Why this memo exists

The latest local evidence tightened the problem:

1. Shared hubs remain plausible because oracle routing lifts the hub base above
   raw pairwise communication.
2. The current frontier and stop heuristics are not additive, even under
   oracle routing.
3. We therefore need references that explain how to align more than two spaces,
   stabilize routing under symmetry, and calibrate stop policies separately
   from route selection.

## Source stack

- `Multi-Way Representation Alignment`
  https://arxiv.org/abs/2602.06205
  Use as the cleanest recent reference for initializing a shared multi-model
  basis before learning extra routing or repair machinery.

- `Universal Sparse Autoencoders`
  https://arxiv.org/abs/2502.03714
  Use as the main argument that one sparse dictionary can span multiple related
  representation spaces instead of training only pairwise interfaces.

- `Delta-Crosscoder`
  https://arxiv.org/abs/2603.04426
  Use as the best recent “shared basis plus residual delta” reference when a
  universal dictionary alone is too rigid.

- `SPARC`
  https://arxiv.org/abs/2507.06265
  Use as a modern sparse-crosscoder reference for feature transport across
  model boundaries.

- `Similarity-Preserving Routers`
  https://arxiv.org/abs/2506.14038
  Use as the strongest recent router-stability reference: preserve geometry and
  neighborhood structure, not just route confidence.

- `CoDE-Stop`
  https://arxiv.org/abs/2604.04930
  Use as the main stopping-policy reference for verifier-gated or
  confidence-bounded multi-step reasoning.

- `Step-Level Verifier TTS`
  https://arxiv.org/abs/2507.15512
  Use as the reference for localized verification rather than one global stop
  score over the whole trajectory.

- `Transformers Learn Factored Representations`
  https://arxiv.org/abs/2602.02385
  Use as the conceptual support for why a shared factorized basis may exist at
  all across models, rather than assuming every alignment must stay pairwise.

## Highest-value experiment implied by these papers

Run a **GPA- or multi-way-aligned shared hub dictionary** across the strongest
three model families, then hold out the fourth family at evaluation time.
Compare:

1. Pairwise bridge baseline.
2. Shared hub dictionary only.
3. Shared hub + stable router objective.
4. Shared hub + stable router + calibrated stop policy.

The key question is whether multi-way canonicalization fixes the hub base
before we add frontier or repair complexity.

## Concrete ablations to add

1. Multi-way shared basis initialization vs pairwise initialization.
   - Same byte budget, same held-out family split.
   - Log atom recovery, hub dead-feature rate, and held-out family accuracy.

2. Confidence-only router vs similarity-preserving router.
   - Same hub and same downstream decoder.
   - Log route accuracy, route entropy, perturbation stability, and load
     balance.

3. Global stop score vs step-localized verifier stop.
   - Same trajectory budget and same hub/router.
   - Log stop reason, halt precision, missed help, over-refinement, and route
     conditional stop gain.

## Anti-loop rules

- Do not add more frontier heuristics until the shared hub base is tested under
  a multi-way aligned basis.
- Do not treat router confidence as sufficient; stability and similarity
  preservation should be logged explicitly.
- Do not reuse the current generic stop rule once route-conditioned sweeps show
  it stays negative even under oracle routing.
