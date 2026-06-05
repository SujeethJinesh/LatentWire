# Scoop Report - New 200 Positive-Method Triage

Date: 2026-06-04

Scope: ideation/filter/scoop/queue-emission only. No experiments, no confirm access, no GPU, and no registry cards for the full seed bank.

## Method

- Ingested `ideas/NEW_200_POSITIVE_METHODS_seed.md`, including the 100 Channel-Set and 100 LatentWire/CacheWire seed ideas plus the elevated complementary/private-computation supplement.
- Applied the campaign lineage filters: same-family one-way score/cache packets are killed, source-index/source-confidence disguises are killed, C-F-like methods without identical-row denominators are killed, and rotation re-skins are killed unless they pass a cross-model non-regression gate.
- Performed targeted literature checks against the named moving neighbors and recent adjacent work. Verdicts below are family-level because this pass emits queues and triage artifacts, not per-method registry cards.

## Source Set Checked

- [C2C, arXiv:2510.03215](https://arxiv.org/abs/2510.03215): neural projection/fusion of source KV-cache into receiver cache; direct semantic communication through caches.
- [KVComm, arXiv:2510.03346](https://arxiv.org/abs/2510.03346): selective KV sharing for efficient LLM communication.
- [Latent Space Communication via K-V Cache Alignment, arXiv:2601.06123](https://arxiv.org/abs/2601.06123): learned shared representation space for cross-model KV-cache access.
- [DeltaKV, arXiv:2602.08005](https://arxiv.org/abs/2602.08005): residual-based KV-cache compression for long-context inference; adjacent to cache-delta compression, not model-to-model evidence packets.
- [LCF, arXiv:2605.22863](https://arxiv.org/abs/2605.22863): compressed latent cache flow for text-free model-to-model communication; explicitly targets new information not in the receiver context.
- [UCCI, arXiv:2605.18796](https://arxiv.org/abs/2605.18796): calibrated uncertainty for cost-optimal LLM cascade routing; adjacent to calibration/conformal routing packets.
- [ParoQuant, arXiv:2511.10645](https://huggingface.co/papers/2511.10645): pairwise rotation quantization with Givens rotations and channel scaling.
- [DecDEC, arXiv:2412.20185](https://bytez.com/docs/arxiv/2412.20185/paper): low-bit LLM quantization / dynamic error compensation adjacency.
- [OSC, arXiv:2604.12782](https://arxiv.org/abs/2604.12782): outlier separation in channel dimension for W4A4 quantization.
- [ResQ, arXiv:2412.14363](https://arxiv.org/abs/2412.14363): mixed precision plus low-rank residuals for LLM quantization.
- [OSCAR, arXiv:2605.17757](https://arxiv.org/abs/2605.17757): attention-aware offline rotations and clipping for INT2 KV-cache quantization.
- [OScaR, arXiv:2605.19660](https://arxiv.org/abs/2605.19660): lightweight canalized rotation for extreme KV-cache compression.

## Verdict Summary

| family | verdict | why | differentiation required |
| --- | --- | --- | --- |
| Complementary/private source communication (`L-PC*`) | `OPEN` | The prior campaign mostly tested same-family, free, one-way packets. C2C/LCF cover latent/cache transfer but not a Mac-first conditional headroom screen for private source computation/tool/retrieval signals. | Must report `I(source_signal;Y|receiver_state)` and show the source signal is not source-top1/source-confidence in disguise. |
| Generated-solution verifier rerank (`L-G*`, `L-E*`, `L-X*`) | `OPEN_PARTIAL` | Reranking and verifier methods are broad, but the specific low-byte source verifier over generated candidates remains unscreened because current L-A2 is generator-limited. | Must use high-entropy candidate pools, target/source/verifier score surfaces, and wrong-row/candidate-derangement controls. |
| Receiver-conditioned cache delta / trained fuser (`L-R*`, `L-C*`, `L-F*`, `L-T*`) | `PARTIAL_HIGH_RISK` | C2C, KVComm, LCF, and KV-cache alignment are direct neighbors. | Must be framed as byte-limited, receiver-conditioned, destructive-control-trained evidence extraction, not generic cache transfer or large trained adapters. |
| Calibration / conformal routing packets (`L-D*`, `L-P*`, `L-B*`) | `PARTIAL_HIGH_RISK` | UCCI and conformal/cascade literature cover calibrated routing strongly. | Needs a packet-specific conditional-information gain, not cost-optimal routing alone. |
| Same-family score/source-confidence packets (`L-O*`, many `L-MI*`) | `SCOOPED_OR_KILLED_BY_OWN_EVIDENCE` | Dominated by source-index/source-confidence and campaign kills. | No registry card without a new falsifier and a non-score private signal. |
| C_A1 / GPD-CVaR tail clip (`C-T1`, `C-T2`, `C-T4`, `C-T5`, `C-Y3`) | `OPEN_PARTIAL` | ParoQuant/OSC/OSCAR/OScaR/ResQ are strong adjacent quantization baselines, but the campaign-specific positive is long-reasoning tail/CVaR selection with paired native rows and sentinel non-regression. | Must beat paired ParoQuant on identical rows and avoid DeepSeek/Falcon median/tail regressions. |
| Drift-as-signal routing (`C-U1`, `C-W*`, `C-E*`, `C-K*`) | `OPEN` | Literature focuses on quantization mechanisms; using channel-set drift trajectory as an uncertainty/router signal appears less directly covered in the checked set. | Must show cached drift features predict difficulty/uplift beyond baseline confidence and transfer to at least two models. |
| Survival / hazard core (`C-S*`, `C-B3`, `C-P*`) | `OPEN_PARTIAL` | Survival framing is not directly scooped, but C-F was control-contaminated locally. | Identical-row denominator and random/static controls are mandatory before any queue promotion. |
| Rotation geometry / co-drift / leverage (`C-G*`, `C-D*`, `C-N*`) | `PARTIAL_HIGH_RISK` | Rotation and KV-quant spaces are crowded by ParoQuant, OSCAR/OScaR, RotateKV-style work, ResQ, and OSC. | Must not be a rotation re-skin; needs cross-model non-regression and a drift-specific mechanism. |
| Systems/profiler wrapper protocols (`C-H*`, `C-Y*`, `L-Y*`) | `OPEN_AS_DEFENSE` | Often not claim-bearing positives by themselves. | Keep as systems cards/defense artifacts unless they produce a clear method win. |

## Hardest Scoop Risks

- `L-F3`, `L-T2`, `L-C2`, and `L-R1` are the closest to C2C/LCF/KVComm/KV-cache alignment. They survive only if the claim is byte-limited, receiver-conditioned, and destructive-control-trained.
- `C-G*`, `C-D1`, `C-N*`, and any fixed-rotation/clip idea are high-risk after ParoQuant, OSCAR, OScaR, ResQ, and OSC. They survive only as drift-specific, cross-model-non-regressing methods.
- `L-D4`, `L-P*`, and `L-B*` are high-risk after UCCI unless they transmit source-private evidence rather than merely route/escalate by calibrated uncertainty.

## Immediate Open Shortlist

1. `L-PC1` cross-family specialist source ceiling - `OPEN`.
2. `L-PC2` tool-augmented source ceiling - `OPEN`.
3. `L-PC4` extra-compute source ceiling - `OPEN`.
4. `L-PC5` private verifier over receiver candidates - `OPEN`.
5. `L-G1/L-G2` generated-solution verifier margin/tournament packet - `OPEN_PARTIAL`.
6. `L-C2/L-F3` receiver-conditioned trained fuser proxy - `PARTIAL_HIGH_RISK`, but worth a cheap planted/control-first MPS screen if byte-limited.
7. `C-U1` drift-as-signal trace router - `OPEN`.
8. `C-W1/C-W3` fixed-library warmup/tail-risk selector - `OPEN`.
9. `C-T1` GPD-CVaR tail clip selector - `OPEN_PARTIAL`; keep behind existing C_A1 paired backfill.
10. `C-S1` survival StableCore with clean identical-row denominator - `OPEN_PARTIAL`; cannot reuse contaminated C-F evidence.

## Decision

Proceed to MPS-first queue emission for the open/high-EVI probes. Do not create registry cards yet. Do not alter `queues/gpu_foreground.yaml`. Keep C_A1 as the first GPU backfill.
