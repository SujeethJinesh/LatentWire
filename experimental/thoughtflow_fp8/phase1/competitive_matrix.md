# ThoughtFlow-FP8 Competitive Matrix

Date: 2026-05-05

Legend: `Y` yes, `N` no, `Partial` limited/indirect, `Unknown` not verified in
quick Phase 1.

| Method | Core idea | Fused kernel / systems integration | Quantization / FP8 | Anchor or fair-span protection | Reasoning / phase awareness | Retrofit existing models | Main weakness for our positioning |
|---|---|---:|---:|---:|---:|---:|---|
| ThoughtFlow-FP8 proposed | FP8 KV plus sink/anchor/fair-span protection plus phase-aware eviction | Target: Y, not built | Target: FP8 | Target: Y | Target: Y | Y | Needs proof; crowded against ThinKV/R-KV/RaaS |
| LongFlow | Current-query attention-derived importance; evict one token each step | Y, fused FlashAttention + importance + eviction | N | N | Partial: long-output reasoning, not phase-aware | Y | Reviews flag quality trade-off, no production E2E, fragile approximations, weak Pareto evidence |
| ThinKV | Thought-adaptive hybrid quantization/eviction by thought importance | Y, extends PagedAttention slot reuse | Y, mixed precision, not specifically FP8 in abstract | Unknown/Partial | Y | Y | Strongest overlap with phase-aware quantized eviction; narrows novelty |
| R-KV | Redundancy-aware reasoning token eviction | Partial/Unknown | N | N | Y: redundant reasoning/self-reflection | Y | Pairwise redundancy estimation cost; quality bar is high |
| R-KVHash | SimHash approximation of R-KV redundancy | Partial/Unknown | N | N | Y: redundant reasoning traces | Y | Workshop-level; mostly redundancy, not anchor/fairness |
| RaaS | Milestone and phoenix token lifecycle; retain prefill, LRU-like decode | N/Python prototype in paper | N | Partial: protects prefill/phoenix tokens | Y | Y | Not a fused quantized kernel; but covers token lifecycle well |
| LazyEviction | Observation-window lagged eviction for token-importance recurrence | Unknown | N | N | Y: recurring token importance | Y | Not FP8/fused; recurrence can undercut simple phase eviction |
| ForesightKV | Learn long-term contribution from future-attention Golden Eviction + GRPO | Unknown | N | N | Y, learned future contribution | Partial: requires training/calibration | Strong if we leave training-free retrofit lane |
| PM-KVQ | Progressive mixed-precision KV quantization for long-CoT | Partial/Unknown | Y, mixed precision | N | Partial: long-CoT quantization | Y with calibration | FP8/quantization baseline; no eviction/fair anchors |
| DeepSeek V4 | Trained hybrid sparse attention: SWA + C4 top-k or C128 dense compressed KV | Y, production SGLang/FlashMLA/Flash Compressor/Lightning TopK | FP4 experts; FP8 rollout/training support; compressed KV architecture | Built into architecture, not retrofit fair spans | Partial/Y through trained indexers and compressed attention | N for arbitrary existing models | Raises systems bar; not directly applicable without retraining/new architecture |

## Differentiation That Survives Quick Review

The only defensible unique cell combination after Phase 1 is:

- retrofit to already-trained reasoning models;
- explicit anchor/fair-span/phase-transition protection;
- FP8 byte budget;
- reviewer-targeted Pareto/e2e/numerical-stability evaluation.

## Differentiation That Does Not Survive

- "Thought adaptive" alone: ThinKV already claims this.
- "Fused compression with attention" alone: LongFlow and DeepSeek V4/SGLang
  already make strong systems claims.
- "KV quantization for long CoT" alone: PM-KVQ and ThinKV already cover it.
- "Reasoning-token lifecycle" alone: RaaS, LazyEviction, R-KV, and ForesightKV
  all cover variants.

## Minimum Baseline Set For Any Proceed Decision

1. Full KV / no compression.
2. Vanilla reasoning-budget truncation or stop-thinking budget baseline.
3. LongFlow.
4. R-KV or R-KVHash.
5. ThinKV.
6. PM-KVQ or another quantization-only baseline.
7. Anchor/fair-span ablations: FP8 only, anchor only, phase only, anchor+phase,
   all.
