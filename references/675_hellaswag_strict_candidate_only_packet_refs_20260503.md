# HellaSwag Strict Candidate-Only Packet References

Date: 2026-05-03

## Why This Memo Exists

The strict HellaSwag audit promotes a narrow systems/privacy claim: the current
selected-packet row can be represented as a `1B` raw / `4B` framed
candidate-only packet over four choices. It does not claim a learned latent
language or native serving speedup.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: benchmark surface for the strict `0:9216` audit.
   - Boundary: multiple-choice candidate-id compaction is a special case; it
     does not prove open-ended reasoning transfer.

2. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: closest soft-prefix conditioning baseline.
   - Boundary: prefix tuning learns continuous task-specific vectors for a
     frozen LM. Our audited row sends a discrete selected candidate id, not a
     learned continuous prefix.

3. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt/context compression comparator.
   - Boundary: gist tokens compress reusable prompt context inside a trained
     LM interface. Our row is a source-private multiple-choice decision packet.

4. Cache-to-Cache: Direct Semantic Communication Between Large Language Models
   - Link: https://arxiv.org/abs/2510.03215
   - Role: direct cache-fusion competitor and systems-positive target.
   - Boundary: C2C transfers/fuses KV-cache state. Our strict candidate-only
     row exposes no KV cache and has byte-accounting only, so native latency
     comparisons require a separate NVIDIA serving run.

5. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://arxiv.org/abs/2510.03346
   - Role: selective KV-sharing communication comparator.
   - Boundary: KVComm transmits selected KV pairs. Our packet is far smaller
     but much less expressive and currently task-format-specific.

6. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
   Overhead
   - Link: https://arxiv.org/abs/2406.03482
   - Role: quantized vector/KV sketch baseline.
   - Boundary: QJL compresses high-dimensional KV vectors for approximate
     attention. Our row does not transmit vectors, so the fair comparison is a
     byte/accuracy/latency table, not a claim of vector-sketch superiority.

7. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
   - Link: https://arxiv.org/abs/2504.19874
   - Role: modern KV/vector quantization baseline.
   - Boundary: TurboQuant is a general vector compression method with KV-cache
     serving relevance. Our `1B` row is a task decision code; it should be
     compared as a different point on the expressivity-vs-systems curve.

8. Efficient Memory Management for Large Language Model Serving with
   PagedAttention
   - Link: https://arxiv.org/abs/2309.06180
   - Role: vLLM/PagedAttention native-serving systems baseline.
   - Boundary: the current Mac audit has no throughput, TTFT, TPOT, HBM, or
     scheduler measurements.

9. SGLang: Efficient Execution of Structured Language Model Programs
   - Link: https://arxiv.org/abs/2312.07104
   - Role: structured serving/runtime baseline for future NVIDIA rows.
   - Boundary: candidate-packet byte accounting is not a substitute for
     runtime-level prefix/cache reuse and scheduling comparisons.

## Reviewer-Facing Comparison Rule

The main paper should separate three rows:

1. `1B` candidate-only packet: source-private decision compaction.
2. Learned receiver/common-basis methods: positive only if they beat
   packet-only plus source-index/source-rank/score controls with paired CIs.
3. Native systems methods: positive only after NVIDIA/vLLM/SGLang-style
   latency, throughput, memory, and exposure measurements against C2C/KVComm
   and QJL/TurboQuant families.

## Next Gate

Do not claim ICLR-level latent communication from this audit. Use it to tighten
the systems/privacy story, then spend the next method pass on a receiver that
beats candidate-only on strict HellaSwag or on native systems measurements that
make the byte-accounting row operational.
