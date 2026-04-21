# Query Routing / Geometry Plan After Span-ALM Failure

Current read:
- Span-ALM failure means output-space likelihood alignment did not rescue the bridge.
- So the remaining question is not “can we make the teacher stronger?” but “what symmetry or routing degree of freedom is still real?”
- I would **not** retest likelihood-mass variants here. The remaining lane is geometric / routing, not another output KL variant.

## 1) Query-conditioned transport

What remains mathematically:
- A transport plan can still depend on the live query, not only on static calibration priors.
- This is the strongest remaining shortcut if the transport map is really a query-dependent mixture of a few canonical routes.

Recent references:
- [FLARE: Fast Low-rank Attention Routing Engine](https://arxiv.org/abs/2508.12594), 2025-08-18
- [Route-DETR: Pairwise Query Routing in Transformers for Object Detection](https://arxiv.org/abs/2512.13876), 2025-12-15

What to test:
- Static transport vs query-gated transport vs query-conditioned Sinkhorn over a small route bank.
- Replace the current query-conditioned bridge gate with a plan selector over 2-4 route atoms.

Why it may be saturated already:
- We already tried query-conditioned local bridge families and they tied the floor.
- If query routing only changes an additive correction but not the transport geometry itself, it is probably too weak.

## 2) Head permutation / gauge symmetry

What remains mathematically:
- Head identity can still be non-canonical after alignment.
- Rotational gauge and permutation gauge can both matter, but only if they survive after the best rotation / whitening already used.

Recent references:
- [Learnable Permutation for Structured Sparsity on Transformer Models](https://arxiv.org/abs/2601.22980), 2026-01-30
- [PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](https://arxiv.org/abs/2510.10136), 2025-10-11

What to test:
- Exact permutation vs soft Sinkhorn vs rotation-only transport.
- Check whether permutation gains disappear once rotational transport is already active.

Why it may be saturated already:
- `grouped_permutation` and `grouped_canonical_transport` already plateaued.
- If permutation after rotation does not beat rotation alone, gauge is not the missing factor.

## 3) Low-rank atlas / dictionary routing

What remains mathematically:
- The bridge may be better viewed as routing through a small atlas of atoms than as fitting one global map.
- This is only useful if the atoms are genuinely shared and the coefficients are sparse or simplex-like.

Recent references:
- [Attention Layers Add Into Low-Dimensional Residual Subspaces](https://arxiv.org/abs/2508.16929), 2025-08-23
- [Low-Rank Key Value Attention](https://arxiv.org/abs/2601.11471), 2026-01-16
- [FLARE: Fast Low-rank Attention Routing Engine](https://arxiv.org/abs/2508.12594), 2025-08-18

What to test:
- 4-atom vs 8-atom bridge dictionary.
- Sparse simplex routing vs dense low-rank residual routing.
- Compare against the already-failed small banked bridge family.

Why it may be saturated already:
- Banked residual and generated-adapter variants already tied the controlled floor.
- If a richer dictionary still does not beat the floor, the issue is not atom count but interface class.

## 4) Layer monotonicity / layer interpolation

What remains mathematically:
- If layer order is approximately monotone, `interp` should beat `reverse` and `random`.
- If it does not, layer correspondence is not the real bottleneck.

What to test:
- `interp` vs `cka` vs `shifted` vs `reverse` vs `random`.
- Keep the transport rule fixed while perturbing only the layer map.

Why it may be saturated already:
- We already use interpolation and CKA-style pairing as baselines.
- If these are close, layer monotonicity is not giving extra structure.

## 5) KV cache orientation

What remains mathematically:
- Even after head permutation, the KV geometry may still be misoriented.
- Rotation is the remaining cheap shortcut that can still matter after whitening and basis alignment.

Recent references:
- [Attention Layers Add Into Low-Dimensional Residual Subspaces](https://arxiv.org/abs/2508.16929), 2025-08-23
- [Low-Rank Key Value Attention](https://arxiv.org/abs/2601.11471), 2026-01-16

What to test:
- Orthogonal vs Hadamard vs DCT vs identity rotation.
- Rotation-only vs rotation + whitening vs rotation + shared basis.

Why it may be saturated already:
- `grouped_rotational_transport` is the only symmetry shortcut that moved the floor.
- Everything else around it has looked like conditioning, not the core fix.

## Geometry critique

The current telemetry suggests the following ordering:

1. **Rotation / orientation** is still live.
2. **Permutation / gauge** is a plausible residual symmetry, but probably secondary.
3. **Query-conditioned transport** may help only if it changes route geometry, not if it just gates a fixed bridge.
4. **Low-rank atlas routing** is worth one last test only if the routing coefficients are genuinely sparse and shared.
5. **Layer monotonicity** is a comparator, not a likely fix.

The important negative result:
- saturated likelihood-mass / local bridge variants are not the right thing to retry.
- if the next gains exist, they should come from geometric routing or a module swap, not from another teacher signal.

## Ranked ablation ladder

1. `rotation = identity` vs `orthogonal` vs `hadamard` vs `dct`, holding transport fixed.
2. `interp` vs `cka` vs `reverse` vs `random` layer pairing, holding rotation fixed.
3. exact permutation vs Sinkhorn-soft permutation vs rotation-only transport.
4. query-gated route selection over 2-4 transport atoms vs one static transport map.
5. 4-atom vs 8-atom low-rank atlas routing with sparse simplex coefficients.
6. stop before re-running any likelihood-mass bridge variant.

## Decision rule

- If rotation + permutation + query-gated route selection still tie the floor, the problem is no longer a symmetry problem.
- At that point, move to a module-replacement interface or a tokenizer/shared-interface transplant.
