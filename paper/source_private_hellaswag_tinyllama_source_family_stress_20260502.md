# HellaSwag TinyLlama Source-Family Stress

## Status

Current paper readiness: ICLR is stronger than before, but still not
comfortable. COLM workshop claims are now strong if we keep the language
bounded.

Current paper story: the `2B` raw / `5B` framed source-private
hidden-innovation packet is not only a Qwen full-validation positive result.
On a heldout HellaSwag slice with TinyLlama as the source hidden generator, the
same packet contract also beats label-copy, score-only, zero-hidden,
wrong-hidden, and candidate-roll controls.

Exact remaining ICLR blocker: this is a non-Qwen source-family heldout slice,
not a complete cross-family full-validation result and not a native systems
comparison against C2C/KVComm/vLLM/SGLang/KV compression.

## Lay Explanation

The earlier positive result could have been using quirks of Qwen hidden
coordinates. This experiment swaps the sender to TinyLlama and asks whether the
same kind of tiny hidden hint still helps on a fresh HellaSwag validation slice.
It does: the tiny packet improves accuracy beyond TinyLlama's own answer-copy
and score-only shortcuts.

## Artifacts

Primary TinyLlama slice result:

`results/source_private_hellaswag_hidden_innovation_eval_slice_stress_20260502_tinyllama_train512_validation1024_2048/hellaswag_hidden_innovation_eval_slice_stress.json`

Source-family stress card:

`results/source_private_hellaswag_source_family_stress_card_20260502/hellaswag_source_family_stress_card.json`

## Method

- Source family: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Eval slice: HellaSwag validation `1024:2048`
- Eval rows: `1024`
- Train samples: three independent 512-row train samples, seeds
  `2027,2039,2053`
- Device: local Mac MPS
- Source dtype: `float16`
- Prompt mode: continuation
- Packet: `2B` raw / `5B` framed
- Transmitted source fields: no source text, no source KV/cache, no raw hidden
  vector, no raw score vector

The seed choice deliberately avoids seed `1729`, because existing HellaSwag
Qwen caches use that seed. All TinyLlama train caches were regenerated for the
TinyLlama source model.

## Results

TinyLlama heldout slice `1024:2048`:

- pass gate: `true`
- selected accuracy: `0.501953`
- best label-copy: `0.450195`
- source label-copy: `0.446289`
- score-only bagged control: `0.446289`
- zero-hidden control: `0.446289`
- wrong-example hidden control: `0.403320`
- candidate-roll hidden control: `0.358398`
- delta vs best label-copy: `+0.051758`
- paired CI95 low vs best label-copy: `+0.025391`
- delta vs score-only: `+0.055664`
- paired CI95 low vs score-only: `+0.036133`
- train-sample jackknife: `3/3`
- jackknife min delta vs best label-copy: `+0.034180`
- jackknife min CI95 low vs best label-copy: `+0.013672`

The stress card compares this TinyLlama heldout-slice row with the Qwen2.5
full-validation row:

| Source family | Scope | Rows | Accuracy | Best label-copy | Delta | CI95 low | Jackknife | Pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen2.5 | full validation | `10042` | `0.526688` | `0.480880` | `+0.045808` | `+0.039634` | `3/3` | yes |
| TinyLlama | heldout slice | `1024` | `0.501953` | `0.450195` | `+0.051758` | `+0.025391` | `3/3` | yes |

## Decision

Promoted:

1. The hidden-innovation packet is not only a Qwen source-family effect.
2. TinyLlama becomes the first non-Qwen source-family HellaSwag positive row.
3. A full non-Qwen validation run is now worth the Mac/GPU time.

Still blocked:

1. This does not prove a universal cross-family latent basis.
2. This does not test a separate non-Qwen receiver model consuming the packet.
3. This does not establish native systems superiority.

## Reviewer-Framing Boundary

Safe claim:

> A fixed-byte source-private hidden-innovation packet survives a non-Qwen
> source-family heldout-slice stress test, weakening the concern that the
> HellaSwag result is purely Qwen-family hidden-coordinate reuse.

Unsafe claims:

- general cross-family communication;
- solved shared latent basis;
- C2C/KVComm native competitor;
- faster or lower-HBM serving than vLLM/SGLang/KV compression baselines.

The relevant boundaries are unchanged: prefix/prompt tuning and adapters learn
persistent conditioning or parameter updates, while LatentWire sends
per-example source-private packet evidence. C2C and KVComm communicate or fuse
source-side KV/cache state; LatentWire is a stricter rate/privacy point until
native systems rows exist. Relative representations and sparse feature work
remain the right common-basis comparison, but this TinyLlama row does not by
itself solve that problem.

## Next Gate

Run one of the following, in priority order:

1. full TinyLlama HellaSwag validation with the same protocol;
2. Phi-3 source-family heldout slice, because it is a stricter non-Qwen local
   model and likely a better capability source than TinyLlama;
3. sparse residual dictionary/common-basis packet if dense hidden innovation
   fails on the next non-Qwen full-validation or Phi row.

Systems next gate: add a Mac-local systems boundary card that includes the new
HellaSwag full-validation packet row, marks C2C/KVComm as pending native, and
keeps QJL/TurboQuant/KIVI/KVQuant as byte-floor rows rather than defeated
native baselines.
