# References: Benchmark Selection and Systems Comparator Update

Web/literature check date: 2026-05-02.

## Benchmark Sources

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   <https://arxiv.org/abs/1905.07830>
   - Role: HellaSwag remains the hard commonsense continuation benchmark.
   - Boundary: current LatentWire HellaSwag hidden-code/receiver branches are
     negative; use it as headroom and negative-ablation evidence.

2. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning
   Challenge.
   <https://arxiv.org/abs/1803.05457>
   - Role: ARC-Challenge is the calibration-positive public benchmark.
   - Boundary: still needs a receiver over packet-only before becoming an ICLR
     learned-communication result.

3. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book
   Question Answering.
   <https://arxiv.org/abs/1809.02789>
   - Role: OpenBookQA is the selected next method surface.
   - Boundary: the current row proves fixed-byte packet transfer, not a learned
     receiver yet.

4. CommonsenseQA: A Question Answering Challenge Targeting Commonsense
   Knowledge.
   <https://aclanthology.org/N19-1421/>
   - Role: CommonsenseQA is the non-science diagnostic.
   - Boundary: same-byte text remains too close for headline promotion.

## Systems and Communication Baselines

1. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   <https://arxiv.org/abs/2510.03215>
   - Overlap: closest cache-fusion communication baseline.
   - Boundary: C2C transmits/fuses source KV state; LatentWire transmits a
     fixed-byte source-private packet.

2. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://arxiv.org/abs/2510.03346>
   - Overlap: selective KV sharing for LLM communication.
   - Boundary: still exposes selected source KV layers.

3. KVCOMM: Online Cross-context KV-cache Communication.
   <https://arxiv.org/abs/2510.12872>
   - Overlap: training-free cross-context KV reuse and alignment.
   - Boundary: KV reuse is not source-private fixed-byte evidence transfer.

4. Q-KVComm: Efficient Multi-Agent Communication via Adaptive KV Cache
   Compression.
   <https://arxiv.org/abs/2512.17914>
   - Overlap: compressed KV communication baseline.
   - Boundary: compressed KV state is still source-state exposure.

5. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   <https://arxiv.org/abs/2402.02750>
   - Overlap: low-bit KV-cache memory baseline.
   - Boundary: same-model KV compression, not inter-model private packet
     communication.

6. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization.
   <https://arxiv.org/abs/2401.18079>
   - Overlap: sub-4-bit KV cache quantization.
   - Boundary: state compression baseline, not a source-private packet method.

7. QJL: 1-bit Quantized Johnson-Lindenstrauss Transform for KV Cache
   Quantization.
   <https://arxiv.org/abs/2406.03482>
   - Overlap: one-bit state-sketch byte floor.
   - Boundary: communicates/keeps a sketch of source state.

8. TurboQuant: Fast and Accurate KV Cache Quantization.
   <https://arxiv.org/abs/2504.19874>
   - Overlap: current strong vector/KV quantization baseline.
   - Boundary: quantized state storage/search, not task-evidence packet
     communication.

9. Efficient Memory Management for Large Language Model Serving with
   PagedAttention.
   <https://arxiv.org/abs/2309.06180>
   - Overlap: vLLM serving baseline for TTFT/TPOT/goodput/memory.
   - Boundary: native serving comparison remains future NVIDIA work.

10. SGLang: Efficient Execution of Structured Language Model Programs.
    <https://arxiv.org/abs/2312.07104>
    - Overlap: serving/runtime baseline with KV reuse and scheduling.
    - Boundary: Mac byte-floor rows cannot claim SGLang speedups.

## Claim Boundary

Safe: fixed-byte source-private packet transfer, benchmark selection, controls,
and byte/exposure accounting.

Unsafe until future gates pass: universal latent language, native systems win,
or superiority over C2C/KVComm/TurboQuant/KIVI/KVQuant on their own systems
axes.
