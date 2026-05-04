# Reference Memo 722: Sparse Resonance Packet Competitor Refresh

Date: 2026-05-04

## Local Status

Sparse Resonance Packets remain a live ICLR pivot, not a positive result. The
current tiny sparse PCA / soft-prefix gates fail strict controls: target-derived
and same-family substitution controls still dominate, while atom identity is not
yet causally used by the receiver.

The comparison target is therefore a benchmark contract: show that SRP has
better downstream utility per communicated byte, privacy, interpretability,
controllability, or hardware movement than dense cache fusion and cache reuse
systems.

## Primary Competitors and Claims

| Competitor | Primary source | Exact claim/metric to compare |
|---|---|---|
| Cache-to-Cache (C2C) | https://openreview.net/forum?id=LeatkxrBCi | Direct source-to-receiver KV-cache projection/fusion. Reports 6.4-14.2% higher average accuracy than individual models, 3.1-5.4% over text communication, and 2.5x average latency speedup. Strongest direct semantic-communication baseline. |
| C2C-C fuser | https://openreview.net/pdf?id=LeatkxrBCi | Appendix variant with extra 3-layer MLP projection before fusion. For Qwen3-4B -> Qwen3-0.6B, table reports large PGR gains, e.g. ARC-C C2C-C 80.96% versus default C2C 60.17%, receiver 41.04%, sharer 87.48%. If reviewers know this row, default C2C is not the strongest dense baseline. C2C also reports cache-enrichment oracle 58.42% direct, 63.39% few-shot, 62.34% same-length oracle; effective-rank diagnostics K: sharer 539 / receiver 388 / C2C 395 and V: 689 / 532 / 560; task-specific training drops average gate activation to 52.67%, whereas general-purpose training keeps gates >98.21% open and relies on dynamic weights. |
| KVComm selective KV sharing | https://openreview.net/forum?id=F7rUng23nw | ICLR 2026 direct inter-LLM KV-pair sharing baseline. Selects non-contiguous layers by attention-importance score plus Gaussian depth prior. Reports comparable performance to a Skyline upper bound while transmitting as few as 30% of layers' KV pairs, computation reduction of 2.5x-6x versus Skyline, up to 3x communication reduction versus sharing all KV pairs, and 23%-73% lower memory than Skyline on Tipsheets. Limitation: current framework assumes same LLM or fine-tuned variants with compatible KV structures; heterogeneous architectures are future work. |
| KVCOMM online cross-context reuse | https://arxiv.org/abs/2510.12872 | Training-free cross-context KV-cache sharing in multi-agent systems. Reports >70% reuse across RAG/math/coding without quality degradation; five-agent 1K-token fully connected setting reduces TTFT from about 430 ms to 55 ms, up to 7.8x; GSM8K four-agent reuse reaches 95% over 1,319 samples with <2.5% accuracy drop. |
| QCFuse | https://arxiv.org/abs/2604.08585 | Query-centric RAG cache fusion from March 2026. Uses semantic summary anchors to get context-aware query representations, then recomputes query-related tokens at a critical middle layer. Reports up to 2x TTFT speedup over full computation and 40% latency reduction versus existing cache-fusion baselines, with matching or improved quality; reported ROUGE-L is 2.3-3.5 points above CacheBlend and 0.8 points above full computation on HotpotQA at 40% recomputation. |
| DroidSpeak | https://arxiv.org/abs/2411.02820 | Cross-LLM KV reuse for same-architecture models, selectively recomputing a few layers and reusing the rest. Reports up to 4x throughput and about 3.1x faster prefill with negligible quality loss. |
| LatentMAS | https://arxiv.org/abs/2511.20639 | Training-free same-shape latent collaboration through last-layer hidden embeddings/shared latent working memory. Reports up to 14.6% higher accuracy, 70.8%-83.7% output-token reduction, and 4.0-4.3x faster end-to-end inference over text MAS. Heterogeneous agents are deferred to trainable adapters. |
| Communicating Activations Between LM Agents | https://arxiv.org/abs/2501.14082 | Zero-parameter activation grafting at intermediate layers. Reports up to 27.0% improvement over natural-language communication with less than one-quarter compute. Good non-KV latent baseline, but generally not source-private or byte-minimal. |
| Q-KVComm | https://arxiv.org/abs/2512.17914 | Compressed KV protocol with adaptive layer-wise quantization and heterogeneous calibration. Reports 5-6x compression and coherence scores above 0.77 across QA scenarios. Treat as lower-confidence because the public paper is short and less established, but include in related work. |
| CacheGen | https://arxiv.org/abs/2310.07240 | Compresses and streams reusable KV caches. Reports 3.5-4.3x KV size reduction and 3.2-3.7x lower context-fetch/process delay with negligible response-quality impact. Systems baseline for moving cache state. |
| CacheBlend / LMCache | https://arxiv.org/abs/2405.16444 | Non-prefix KV reuse/fusion for RAG by selectively recomputing important tokens. Reports 2.2-3.3x TTFT reduction and 2.8-5x throughput improvement over full KV recompute with negligible quality drop; compared with full KV reuse, improves QA F1 by 0.1-0.2 and summarization ROUGE-L by 0.03-0.25. Compare as a serving/KV reuse baseline rather than source-private cross-model communication. |
| DynamicKV | https://aclanthology.org/2025.findings-emnlp.426/ | Task-aware layer-adaptive KV cache compression. Reports retaining 1.7% of KV cache while preserving 90%, 87%, 78%, and 83% of original accuracy for LLaMA-3-8B-Instruct, Mistral-7B-Instruct-v0.2, Qwen2-7B-Instruct, and InternLM-2.5-7B-Chat-1M; at 6.9% retained cache, performance is nearly indistinguishable from full KV; at 0.9% retained cache, beats SOTA by 11% on Needle-in-a-Haystack with Mistral-7B-Instruct-v0.2. |
| Task-KV | https://arxiv.org/abs/2501.15113 | Task-aware head-level KV allocation via semantic differentiation of attention heads. Reports full-cache-comparable performance on summarization and synthetic tasks while using 40% KV memory. Use as a behavior/task-conditioned cache-allocation baseline, not a communication baseline. |
| vLLM / PagedAttention | https://arxiv.org/abs/2309.06180 | Native serving baseline. Reports near-zero KV memory waste, flexible KV sharing, and 2-4x throughput at same latency versus FasterTransformer/Orca. |
| SGLang / RadixAttention | https://arxiv.org/abs/2312.07104 | Structured-generation serving baseline with automatic KV reuse via radix-tree cache. Required production-style comparator for prefix/repeated-context workloads. |
| LMCache | https://arxiv.org/abs/2510.09665 | KV-cache layer for cross-engine cache offload, reuse, and prefill-decode disaggregation. Compare when claiming serving friendliness or multi-node movement. |
| SCBench | https://arxiv.org/abs/2412.10319 | Benchmark lens for KV lifecycle: generation, compression, retrieval, and loading. Use its lifecycle categories as table columns for SRP. |
| KIVI | https://arxiv.org/abs/2402.02750 | Tuning-free asymmetric 2-bit KV quantization. Reports 2.6x lower peak memory, up to 4x larger batch, and 2.35-3.47x throughput with similar quality. Aggressive dense-KV byte floor. |
| KVQuant | https://arxiv.org/abs/2401.18079 | Sub-4-bit KV quantization with pre-RoPE/per-channel/non-uniform/outlier handling. Reports <0.1 perplexity degradation at 3-bit, up to 1M context on one A100-80GB and 10M on 8 GPUs, with up to about 1.7x CUDA speedup for LLaMA-7B. |
| TurboQuant | https://arxiv.org/abs/2504.19874 | Near-optimal online vector quantization. Reports quality neutrality for KV cache at 3.5 bits/channel and marginal degradation at 2.5 bits/channel. |
| KVTC | https://arxiv.org/abs/2511.01815 | PCA/adaptive-quantization/entropy-coded KV transform coding. Reports up to 20x KV compression while preserving reasoning and long-context accuracy, and 40x+ in specific use cases. |

## Columns to Add to the SRP Comparison Table

Minimum table fields:

- task, slice, source model, target model, same-family flag, cross-family flag;
- method class: text, dense KV fusion, KV reuse, activation graft, latent MAS,
  sparse packet;
- communicated payload bytes;
- framed bytes including headers, atom ids, scales, parity/CRC, alignment, and
  batch padding;
- cache-line bytes per request and amortized batch-line bytes;
- DMA/burst bytes per request and amortized batch-burst bytes;
- dense fp16 KV byte floor;
- dense quantized KV byte floor at 2, 2.5, 3, 3.5, 4, and 8 bits;
- source text exposed, source KV exposed, hidden activation exposed;
- receiver-private target cache used as side information;
- native serving measured flag and backend: Mac proxy, vLLM, SGLang, LMCache,
  custom CUDA;
- TTFT, TPOT, end-to-end latency, HBM bytes moved, host-device bytes moved;
- utility delta over target-only, utility delta over best same-byte text,
  utility delta over best dense baseline;
- paired bootstrap CI and seed stability;
- destructive controls passed: wrong-row, atom-shuffle, coefficient-shuffle,
  candidate-roll, target-derived, source-rank/score/index, same-family
  substitution;
- interpretability fields: atom id, atom family, atom knockout effect,
  coefficient sign/magnitude, per-layer injection/gate pattern;
- cache-conditioning fields: query-conditioned, task-conditioned,
  behavior-conditioned, context-offset-conditioned, static-calibrated, online
  calibrated;
- cache-fusion policy fields: selected layers, selected heads, selected tokens,
  recompute ratio, gate activation ratio, dynamic-weight entropy, effective-rank
  delta.

## Byte / Movement Formulas

Dense KV payload:

```text
B_dense_KV = 2 * L * H_kv * T_source * d_head * bytes_per_value
```

Quantized dense KV:

```text
B_quant_KV =
  2 * L * H_kv * T_source * d_head * bits_per_value / 8
  + B_scales + B_zero_points + B_outliers + B_codebooks + B_layout
```

C2C transmitted/fused cache floor:

```text
B_C2C_floor = B_source_KV_projected_or_raw + B_alignment + B_fuser_metadata
```

For a fair lower bound, use the same dense/quantized KV formula for the source
cache segment actually exposed to the fuser; for a systems upper bound, include
materialized projected K/V tensors and gate/dynamic-weight tensors.

Sparse Resonance Packet:

```text
B_srp_payload =
  C * K * (ceil(log2(A)) + coefficient_bits + sign_bits) / 8
  + B_candidate_ids + B_global_header + B_scales + B_crc
```

where `C` is candidates or packet slots, `K` is top atoms per slot, and `A` is
the atom dictionary size.

Hardware transfer lower bound:

```text
B_line = ceil(B_framed / cache_line_bytes) * cache_line_bytes
B_dma = ceil(B_framed / dma_burst_bytes) * dma_burst_bytes
B_batch_line = ceil(batch * B_framed / cache_line_bytes) * cache_line_bytes / batch
B_batch_dma = ceil(batch * B_framed / dma_burst_bytes) * dma_burst_bytes / batch
```

Utility per byte:

```text
UPB_target = mean_i(correct_method_i - correct_target_i) / B_framed
UPB_best_control =
  mean_i(correct_method_i - correct_best_required_control_i) / B_framed
```

Report paired uncertainty over item IDs and repeat seeds.

## Strongest Reviewer Objections

1. C2C already won the semantic-transfer claim. SRP must claim a different
   regime: lower bytes, source-private packets, interpretability, and stricter
   destructive controls, not latent communication novelty.
2. Dense KV baselines can be heavily quantized. A 5-byte SRP row is only
   meaningful if compared against KIVI/KVQuant/TurboQuant/KVTC byte floors, not
   fp16 KV alone.
3. C2C-C is stronger than default C2C. A reviewer can object if the paper only
   compares to default C2C while ignoring its complex fuser appendix.
4. Current SRP evidence is negative. The latest gates do not beat target-only,
   target-derived, same-byte text, or same-family substitution, so the
   literature refresh cannot be framed as support for a positive result.
5. Same-family leakage is a serious confound. DroidSpeak, KVComm, and LatentMAS
   often rely on identical architecture or same-shape assumptions; SRP must
   show strict same-family versus cross-family separation.
6. Privacy is not automatic. Hiding text while exposing dense hidden/KV state
   may still leak source content. SRP needs explicit source-text, source-KV, and
   hidden-state exposure columns plus reconstruction/probing threat models.
7. Interpretability requires causal controls. Naming sparse atoms is not enough;
   atom shuffle, coefficient shuffle, top-atom knockout, and wrong-row controls
   must change outcomes predictably.
8. Mac-local traffic estimates are not serving throughput. Hardware claims need
   vLLM/SGLang/LMCache or CUDA/H100 measurements with HBM and host-device byte
   accounting.
9. Utility-per-byte can be gamed by tiny denominators. Require positive paired
   utility over target-only and best required control, not just low bytes.
10. Structured text is a stronger baseline than free-form text. Same-byte JSON,
    source rank/score/index, and candidate-local structured text must remain in
    the baseline suite.

## Benchmark Consequence

The next SRP gate should not widen benchmarks. It should first pass a frozen
slice with:

- target-only, target-derived, same-byte structured text, source rank/score,
  wrong-row, atom-shuffle, coefficient-shuffle, candidate-roll, and same-family
  substitution controls;
- dense C2C/C2C-C modeled byte rows even if native execution is deferred;
- quantized KV byte floors at 2-4 bits;
- utility-per-framed-byte and utility-per-cache-line-byte;
- explicit source exposure flags.

Only after that should SRP be widened to native C2C/KVComm/LatentMAS-style
serving benchmarks.
