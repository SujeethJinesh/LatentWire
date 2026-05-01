# HellaSwag Dense Validation[0:9216] Stress

## Status

The dense bagged hidden-innovation branch is strengthened again. It now passes
nine contiguous HellaSwag validation slices under the same frozen packet
budget, train samples, split seeds, and control ladder.

Current paper story: the `2B` raw / `5B` framed source-private
hidden-innovation packet is still the strongest HellaSwag positive method.
Common-basis compression remains a negative or weakened branch, so the defensible
claim is fixed-byte source-private per-example communication, not a solved
universal latent language.

## Why This Gate Was Run

The previous gate passed validation rows `0:8192`. This gate tested the next
untouched block, rows `8192:9216`, without changing the method.

In lay terms: we again asked whether a tiny private hint from the source model
helps the receiver on fresh examples. To count as useful, the hint had to beat
copying the source model's top answer, using only the source model's confidence,
and corrupted-hidden controls.

## Validation[8192:9216] Dense Stress

Artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation8192_9216/hellaswag_hidden_innovation_eval_slice_stress.json`

Result: `pass_gate=true`.

- selected accuracy: `0.545898`
- best label-copy: `0.499023`
- delta vs label-copy: `+0.046875`
- CI95 low vs label-copy: `+0.027808`
- score-only bagged: `0.499023`
- delta vs score-only: `+0.046875`
- zero-hidden delta: `+0.046875`
- wrong-example hidden: `0.450195`
- candidate-roll hidden: `0.423828`
- jackknife: `3/3`
- packet: `2B` raw / `5B` framed

## Validation[0:9216] Aggregate

Artifact:
`results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_9216/hellaswag_hidden_innovation_multi_slice_stress.json`

Result: `pass_gate=true`.

- slices passing: `9/9`
- total eval rows: `9216`
- weighted selected accuracy: `0.525499`
- weighted best label-copy: `0.483941`
- weighted score-only: `0.479384`
- min delta vs best label-copy: `+0.034180`
- min CI95 low vs best label-copy: `+0.011719`
- min delta vs score-only: `+0.037109`
- min delta vs zero-hidden: `+0.037109`
- corrupted hidden controls below label-copy: `true`
- source-private packet: `true`

## Interpretation

Promoted:

1. Dense bagged hidden innovation now clears validation `0:9216`, about 91.8%
   of the local HellaSwag validation split.
2. The hardest shortcut controls are still below the method: best label-copy,
   trained label bias, score-only, zero-hidden, wrong-example hidden, and
   candidate-roll hidden.
3. The HellaSwag row is now a stronger Mac-local headline candidate, but still
   should not be called full-validation evidence.

Still blocked:

1. Remaining HellaSwag validation rows `9216:10042`.
2. One strict cross-family falsification pair.
3. Native NVIDIA/vLLM/SGLang systems rows against target-only, text, C2C,
   KVComm, QJL, TurboQuant, KIVI, and KVQuant baselines.

Systems artifact:
`paper/source_private_hellaswag_dense_systems_trace_card_20260501.md`

## Novelty Boundary

Safe claim: LatentWire sends a fixed-byte source-private per-example packet that
improves target decisions without transmitting source text, KV cache, raw hidden
vectors, or raw scores.

Do not claim: new prefix tuning, adapters, sparse autoencoders, sparse
crosscoders, relative representations, KV-cache communication, vector/KV
quantization, diffusion transformers, or latent chain-of-thought. Those are
related work and comparison boundaries.

## Next Gate

Run the remaining dense heldout validation slice `9216:10042` with the same
frozen train samples, split seeds, packet bytes, and controls. If it passes,
rebuild the aggregate as validation `0:10042`. If it fails softly, preserve the
failure and inspect whether the short final slice has a label-copy distribution
shift. If it fails hard, demote HellaSwag from headline candidate to diagnostic
evidence and shift effort to strict cross-family and native systems rows.
