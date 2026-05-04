# Systems Boundary Figure/Table V3

- pass gate: `True`
- packet framed-byte range: `4-11B`
- minimum source-state floor vs max packet: `69.8x`
- native NVIDIA systems complete: `False`

## Rows

| Method | Object | Raw | Framed | Cacheline | Batch64 | Private | Text | KV | Hidden | Native | Claim |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| LatentWire ARC-Challenge test packet (cached source) | cached-source task-level candidate evidence packet | 8B | 11B | 64B | 704B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire ARC-Challenge test packet (source scoring disclosed) | same packet with source scoring disclosed separately | 8B | 11B | 64B | 704B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire OpenBookQA test packet (cached source) | cached-source task-level candidate evidence packet | 3B | 6B | 64B | 384B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire OpenBookQA test packet (source scoring disclosed) | same packet with source scoring disclosed separately | 3B | 6B | 64B | 384B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire HellaSwag validation_first1024 packet (cached source) | cached-source task-level candidate evidence packet | 2B | 5B | 64B | 320B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire HellaSwag validation_first1024 packet (source scoring disclosed) | same packet with source scoring disclosed separately | 2B | 5B | 64B | 320B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| LatentWire HellaSwag validation_full_compaction packet (cached source) | cached-source task-level candidate evidence packet | 1B | 4B | 64B | 256B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting only; source scoring excluded |
| LatentWire HellaSwag validation_full_compaction packet (source scoring disclosed) | same packet with source scoring disclosed separately | 1B | 4B | 64B | 256B | `true` | `false` | `false` | `false` | `false` | source-private packet byte/exposure accounting with source scoring timing disclosed; not a native GPU serving row |
| Same-byte structured text control (ARC) | text-form control with same packet budget | 8B | 11B | 64B | 704B | `false` | `true` | `false` | `false` | `true` | negative/control row only; cannot support source-private claim |
| Four-choice fp16 score-vector relay floor | source score vector, fp16 | 8B | 8B | 64B | 512B | `false` | `false` | `false` | `true` | `false` | source-state exposure floor only; not a source-private packet |
| Four-choice fp16 logit-vector relay floor | source logit vector, fp16 | 8B | 8B | 64B | 512B | `false` | `false` | `false` | `true` | `false` | source-state exposure floor only; not a source-private packet |
| One hidden-vector fp16 relay floor | one source hidden vector, fp16 | 1792B | 1792B | 1792B | 114688B | `false` | `false` | `false` | `true` | `false` | state-exposure lower bound only; not a baseline win |
| 1-bit/KV-element accounting floor | one-token K+V state at 1 bit/element | 768B | 768B | 768B | 49152B | `false` | `false` | `true` | `false` | `false` | mathematical state-size lower bound only |
| KIVI 2-bit KV floor | one-token K+V state at 2 bits/element | 1536B | 1536B | 1536B | 98304B | `false` | `false` | `true` | `false` | `false` | KV-cache compression comparator only |
| Q-KVComm optimistic 6x floor | compressed source KV cache representation | 2048B | 2048B | 2048B | 131072B | `false` | `false` | `true` | `false` | `false` | compressed-KV communication boundary only |
| KVQuant 3-bit proxy floor | one-token K+V state at 3 bits/element | 2304B | 2304B | 2304B | 147456B | `false` | `false` | `true` | `false` | `false` | sub-4-bit KV comparator only |
| TurboQuant 3.5-bit KV floor | one-token K+V state at 3.5 bits/element | 2688B | 2688B | 2688B | 172032B | `false` | `false` | `true` | `false` | `false` | KV/vector quantization comparator only |
| KVComm 30% fp16 KV floor | selected source KV layers, fp16 | 3686.4B | 3686.4B | 3712B | 235929.6B | `false` | `false` | `true` | `false` | `false` | selective-KV communication boundary only |
| C2C one-token fp16 KV floor | projected/fused source KV cache | 12288B | 12288B | 12288B | 786432B | `false` | `false` | `true` | `false` | `false` | closest cache-transfer baseline; native run still required |
| KVCOMM cross-context fp16 KV floor | aligned/reused source KV cache | 12288B | 12288B | 12288B | 786432B | `false` | `false` | `true` | `false` | `false` | systems neighbor only; native run still required |
| vLLM/PagedAttention one-token KV floor | paged KV-cache serving substrate | 12288B | 12288B | 12288B | 786432B | `true` | `false` | `false` | `false` | `false` | native TTFT/TPOT/goodput/HBM target, not closed on Mac |
| SGLang/RadixAttention one-token KV floor | KV-cache reuse serving substrate | 12288B | 12288B | 12288B | 786432B | `true` | `false` | `false` | `false` | `false` | native TTFT/TPOT/goodput/HBM target, not closed on Mac |

## Claim Boundary

Paper-ready systems boundary artifact: LatentWire cached-source rows count the fixed-byte source-private packet object; paired end-to-end rows disclose source scoring separately. KV/cache rows are byte floors or pending native serving baselines. The artifact supports byte/exposure accounting, not a native C2C/KVComm/TurboQuant/QJL/vLLM/SGLang win.
