# Geometry / Symmetry Shortcuts for Cross-Model KV

Working read:
- The current telemetry says the cheap symmetries are mostly spent.
- `grouped_rotational_transport` is the only geometry shortcut that clearly moved the needle: it reached `0.1000` on controlled `gsm10`.
- `grouped_subspace_transport + rank4` reached `0.0571` on `Qwen GSM70`, but covariance, canonical-basis, signature, and local bridge variants either plateaued or tied the weak floor.
- So the useful question is not “can we fit another low-rank map?”, but “which symmetries are still real, and which are now just regularizers?”

The short answer:
- **Still live:** head permutation, rotational gauge.
- **Mostly stabilizers now:** whitening/ZCA, Procrustes, CCA, shared low-rank bases.
- **Conditional / preprocessing-only:** tokenizer span symmetries.
- **Useful as a routing prior, not a fix:** layer interpolation symmetry.

## 1) Head permutation symmetry

Canonical form:
- Treat head identity as a permutation problem over channels or head groups.
- Relevant references in the repo mirror: `134`, `152`, `153`.
- Recent primary sources:
  - [Learnable Permutation for Structured Sparsity on Transformer Models](https://arxiv.org/abs/2601.22980), 2026-01-30
  - [PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](https://arxiv.org/abs/2510.10136), 2025-10-11

Actionable ablation:
- Identity vs exact permutation vs Sinkhorn-soft permutation, all at the same budget.
- Run it both with and without rotational canonicalization.

Why this looks partly saturated:
- `grouped_permutation` and `grouped_canonical_transport` already plateaued.
- That suggests head relabeling matters, but by itself is not enough to explain the remaining error.

Interpretation:
- If permutation only helps before rotation but not after rotation, it is a preprocessing symmetry, not the core bottleneck.

## 2) Rotational gauge symmetry

Canonical form:
- Per-head or per-block orthogonal basis changes.
- Relevant repo mirrors: `137`, `156`, `160`, `161`.
- Recent / adjacent references:
  - [Transport and Merge: Cross-Architecture Merging for Large Language Models](https://arxiv.org/abs/2602.05495), 2026-02-05
  - [Quantized Wasserstein Procrustes Alignment of Word Embedding Spaces](https://arxiv.org/abs/2212.02468), 2022-12-05

Actionable ablation:
- Random orthogonal vs fitted rotation vs Hadamard vs DCT.
- Compare source-only whitening, target-only whitening, and symmetric target whitening.

Why this is still live:
- `grouped_rotational_transport` is the best positive geometry result we have.

Why it is also near saturation:
- Once rotation is fitted, the remaining gap is not explained by orientation alone.
- The current floor suggests the remainder is more likely module/readout mismatch than another basis rotation.

Interpretation:
- Keep rotation as the main canonicalization shortcut, but do not expect it to solve the bridge alone.

## 3) Whitening / ZCA

Canonical form:
- Normalize anisotropic scaling before alignment.
- Relevant repo mirrors: `126`, `141`, `149`.

Actionable ablation:
- None vs source-only ZCA vs symmetric target whitening.
- Turn whitening off after rotation to see if it was only a conditioning aid.

Why this appears saturated:
- Whitening improved numerical stability in earlier branches, but it did not move the final floor once more expressive transport was in place.
- Symmetric target whitening was at best bounded, not decisive.

Interpretation:
- Treat whitening as a conditioning trick, not the method.

## 4) Procrustes / rigid alignment

Canonical form:
- Orthogonal Procrustes after centering / whitening.
- Relevant repo mirrors: `160`, plus the broader similarity notes in `42-44`.

Actionable ablation:
- Orthogonal Procrustes vs ridge vs reduced-rank regression under the same layer pairing and same head grouping.

Why this is likely saturated:
- The recent plateau on canonical and subspace transports means “rigid” alignment is not the missing ingredient.
- If Procrustes and ridge are tied, there is no gain left from a pure rigid transform.

Interpretation:
- Procrustes is a useful diagnostic, but probably not the final bridge.

## 5) Canonical correlation / shared low-rank bases

Canonical form:
- Move both spaces into a shared low-rank basis and compare there.
- Relevant repo mirrors: `42`, `43`, `44`, `151`, `158`, `159`.

Actionable ablation:
- Rank sweep: 4 / 8 / 16 / 32.
- Compare plain low-rank, shared-basis, and shared-basis + residual.

Why this looks saturated:
- `grouped_subspace_transport`, `grouped_shared_basis_transport`, and `grouped_covariance_transport` already failed to beat the strong floor.
- `grouped_subspace_transport + rank4` only reached `0.0571`, which is better than nothing but not enough to justify another low-rank-only symmetry pass.

Interpretation:
- Shared subspaces are real, but the current problem is not solved by subspace geometry alone.

## 6) Tokenizer span symmetries

Canonical form:
- Aggregate mismatched tokenizations into spans, bytes, or shared anchors before alignment.
- Relevant repo mirrors: `245`, `267`, `284`, `292`.
- Recent primary sources:
  - [Enhancing Cross-Tokenizer Knowledge Distillation with Contextual Dynamical Mapping](https://arxiv.org/abs/2502.11104), 2025-02-16
  - [Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083), 2025-03-25
  - [TokAlign: Efficient Vocabulary Adaptation via Token Alignment](https://arxiv.org/abs/2506.03523), 2025-06-04
  - [Training-Free Tokenizer Transplantation via Orthogonal Matching Pursuit](https://arxiv.org/abs/2506.06607), 2025-06-07
  - [CTPD: Cross Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865), 2026-01-17

Actionable ablation:
- Token-level vs span-pooled vs byte/shared-interface alignment.
- If tokenization differs, test whether span pooling improves calibration fit before any KV bridge is trained.

Why this may be saturated or irrelevant:
- If the current Qwen pair is effectively tokenizer-compatible, this symmetry is not the bottleneck.
- If a span-pooled baseline does not lift the floor, the issue is downstream of token alignment.

Interpretation:
- This is a preprocessing symmetry. Use it to remove fake mismatch, not as the main method.

## 7) Layer interpolation symmetry

Canonical form:
- Treat layer index as approximately monotone / affine and test interpolation-based pairing.
- Relevant repo mirrors: `114`, `115`, `118`, `120`, `121`, `129`, `130`.

Actionable ablation:
- `interp` vs `cka` vs `shifted` vs `reverse` vs `random`.
- If `interp` and `cka` are close, and `random` is only slightly worse, layer order is not the bottleneck.

Why this may be saturated:
- The current transport family already uses interpolation / CKA-style pairing.
- If changing layer pairing does not change controlled GSM10 materially, then layer correspondence is not the blocker anymore.

Interpretation:
- Layer interpolation is a routing heuristic, not a symmetry that explains the remaining gap once transport is already decent.

## What the current telemetry implies

The symmetry ladder now looks like this:

1. **Rotation**: still live and worth keeping.
2. **Permutation**: worth testing, but likely secondary to rotation.
3. **Whitening / Procrustes / CCA / shared low-rank bases**: mostly conditioning or diagnostics.
4. **Tokenizer span symmetries**: only important if tokenization mismatch is real in the pair.
5. **Layer interpolation**: useful as a comparator, not a likely final fix.

So the practical conclusion is:
- keep the geometry canonicalization we already know helps,
- stop expecting another purely linear symmetry shortcut to beat the floor,
- and use the remaining gains as the trigger to move into a more global module-replacement interface.

## Minimal next ablation ladder

1. `identity` head pairing + no whitening.
2. `interp` layer pairing + source-only whitening.
3. `grouped_permutation`.
4. `grouped_rotational_transport`.
5. `grouped_rotational_transport + symmetric target whitening`.
6. `grouped_shared_basis_transport`.
7. `grouped_subspace_transport + rank4 residual`.
8. token-span pooling / byte-interface only if tokenizer mismatch is demonstrable.

## Primary sources / mirrors to cite in follow-up notes

- `134`, `152`: Learnable Permutation for Structured Sparsity on Transformer Models
- `153`: PermLLM
- `156`, `157`, `158`: gauge/OT alignment references
- `160`: quantized Wasserstein Procrustes
- `161`: transformers-as-OT framing
- `245`, `267`, `284`, `292`: tokenizer transfer / preference / dynamic mapping / transplantation references
- `298`: this note
