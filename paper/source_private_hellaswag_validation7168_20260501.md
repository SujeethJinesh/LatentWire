# HellaSwag Dense Validation[0:7168] Stress

## Status

The dense bagged hidden-innovation branch is strengthened. It now passes seven
contiguous HellaSwag validation slices under the same frozen packet budget and
control ladder.

Current paper story: the `2B` raw / `5B` framed source-private
hidden-innovation packet is still the only robust HellaSwag positive method.
Cheap common-basis variants remain negative or weakened, so the paper should
claim fixed-byte source-private per-example communication rather than a solved
shared latent language.

## Why This Gate Was Run

The previous gate passed validation rows `0:6144`. This gate tested the next
untouched block, rows `6144:7168`, without changing the train samples, split
seeds, packet bytes, or controls.

In lay terms: we gave the receiver the same tiny hint from the source model and
checked whether that hint still helped on a new chunk of examples. The important
comparison is not target-only; it is whether the hint beats simpler explanations
like copying the source model's top answer or using only the source model's
confidence.

## Validation[6144:7168] Dense Stress

Artifact:
`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260501_qwen05_train512_validation6144_7168/hellaswag_hidden_innovation_eval_slice_stress.json`

Result: `pass_gate=true`.

- selected accuracy: `0.555664`
- best label-copy: `0.515625`
- delta vs label-copy: `+0.040039`
- CI95 low vs label-copy: `+0.013672`
- score-only bagged: `0.511719`
- delta vs score-only: `+0.043945`
- zero-hidden delta: `+0.043945`
- wrong-example hidden: `0.494141`
- candidate-roll hidden: `0.421875`
- jackknife: `3/3`
- packet: `2B` raw / `5B` framed

## Validation[0:7168] Aggregate

Artifact:
`results/source_private_hellaswag_hidden_innovation_multi_slice_stress_20260501_qwen05_validation0_7168/hellaswag_hidden_innovation_multi_slice_stress.json`

Result: `pass_gate=true`.

- slices passing: `7/7`
- total eval rows: `7168`
- weighted selected accuracy: `0.518136`
- weighted best label-copy: `0.476562`
- weighted score-only: `0.470843`
- min delta vs best label-copy: `+0.034180`
- min CI95 low vs best label-copy: `+0.011719`
- min delta vs score-only: `+0.041016`
- min delta vs zero-hidden: `+0.041016`
- corrupted hidden controls below label-copy: `true`
- source-private packet: `true`

## Interpretation

Promoted:

1. Dense bagged hidden innovation is now a stronger Mac-local HellaSwag
   headline candidate: it clears validation `0:7168`, not only `0:6144`.
2. The strongest alternative explanations remain below the packet: label-copy,
   trained label bias, score-only, zero-hidden, wrong-example hidden, and
   candidate-roll hidden all fail the strict gate.

Still blocked:

1. Remaining HellaSwag validation rows `7168:10042`.
2. One strict cross-family falsification pair.
3. Native NVIDIA/vLLM/SGLang systems rows against target-only, text, C2C,
   KVComm, QJL, TurboQuant, and KV-quantization baselines.

Systems artifact:
`paper/source_private_hellaswag_dense_systems_trace_card_20260501.md`

## Novelty Boundary

Safe claim: LatentWire sends a fixed-byte source-private per-example packet that
improves target decisions without transmitting source text, KV cache, raw hidden
vectors, or raw scores.

Do not claim: new prefix tuning, adapters, sparse autoencoders, sparse
crosscoders, relative representations, KV-cache communication, or vector/KV
quantization. Those are related work and comparison boundaries.

## Next Gate

Run dense heldout validation slice `7168:8192` with the same frozen train
samples, split seeds, packet bytes, and controls. If it passes, rebuild the
aggregate as validation `0:8192`. If it fails softly, run `8192:9216` unchanged
to separate isolated slice shift from method collapse. If it fails hard, demote
HellaSwag from headline candidate to diagnostic evidence and shift effort to
systems rows for the already seed-stable ARC/OpenBookQA branches.
