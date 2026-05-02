# Source-Private ARC Source-Family Cache Falsification, 2026-05-02

## Status

- current paper readiness: COLM is strong; ICLR full paper is still blocked by
  source-family robustness and native NVIDIA systems baselines.
- current story: LatentWire has fixed-byte source-private packets,
  OpenBookQA packet/target receiver-fusion, and an ARC Fourier/anchor-syndrome
  public-basis packet.
- exact blocking gap: the ARC Fourier row is not yet source-family-general. A
  TinyLlama source cache preserves the full-slice ARC test lift, but fails the
  stricter Qwen-disagreement falsification slice.

Artifact:
`results/source_private_arc_challenge_source_family_cache_falsification_20260502_tinyllama_cpu/`

## What This Gate Does

Plain-language version: we ask a different local model, TinyLlama, to pick
answers for ARC without seeing the answer key. We then send those TinyLlama
choices through the same `12B` Fourier/anchor packet and compare it to the old
Qwen-source packet. The strictest check looks only at examples where TinyLlama
and Qwen picked different answers.

This gate tests source-choice/cache dependence. It does not prove that
TinyLlama hidden states are being communicated: the packet still encodes public
candidate residual geometry selected by the source model's top choice.

## Results

ARC-Challenge test:

| Surface | Pass seeds | Matched | Target | Text | Qwen-sub | Min CI target | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| full slice | 5/5 | 0.325 | 0.265 | 0.298 | - | +0.018 | positive |
| Qwen-disagreement | 0/5 | 0.269 | 0.268 | 0.258 | 0.317 | -0.059 | fail |

The TinyLlama and Qwen source caches disagree on `473/1172` ARC test examples
(`40.4%`). On those disagreement rows, Qwen-substituted packets beat
TinyLlama-selected packets by about `+0.048` mean accuracy, and the worst seed
has matched-minus-Qwen-substituted `-0.051`.

Validation is also not promotable: full-slice validation passes `3/5` seeds and
the Qwen-disagreement slice passes `0/5`.

## Systems And Mac-Local Notes

- TinyLlama CPU formal scoring completed locally and produced answer-field-free
  source caches for `299` validation rows and `1172` test rows.
- TinyLlama MPS worked on a `40`-row smoke probe but failed on the full formal
  materialization with an Apple MPS matmul shape error.
- GPT-2 and OPT-350M were feasible on MPS but too weak to promote as alternate
  source families.
- Cached Llama-3.2 and Phi-3 endpoints failed local loader/config compatibility
  before scoring.

## Decision

Do not claim source-family-general ARC communication yet. Keep the ARC
Fourier/anchor-syndrome packet as a positive common-basis row, but state that
it remains source-cache-specific under the strict TinyLlama disagreement test.

Next exact gate: either run a stronger alternate source on NVIDIA, or implement
a learned source endpoint/connector whose signal is not just the selected
candidate index. Native systems comparisons against C2C, KVComm, QJL,
TurboQuant, vLLM, and SGLang remain required before ICLR-ready systems claims.

Related-work boundary is tracked in
`references/632_arc_source_family_cache_falsification_refs_20260502.md`.
