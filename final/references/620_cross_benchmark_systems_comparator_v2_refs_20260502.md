# References: Cross-Benchmark Systems Comparator V2

Web check: 2026-05-02. Scope: primary-source boundaries for the updated
systems comparator that adds the `1B` raw / `4B` framed HellaSwag compact row.

## Local Result

Artifact:
`results/source_private_cross_benchmark_systems_comparator_20260502/`

- pass gate: `True`
- headline public benchmarks: `ARC-Challenge` and `OpenBookQA`
- diagnostic / systems rows: legacy HellaSwag fixed packet plus compact
  HellaSwag systems-rate row
- framed packet range: `4-15B`
- minimum one-token QJL 1-bit source-state floor versus framed packet: `51.2x`
- compact HellaSwag row: `1B` raw / `4B` framed, exact prediction equivalence
  inherited from
  `results/source_private_hellaswag_minimal_packet_compaction_20260502/`

These are source-state exposure byte floors from the local Qwen2.5-0.5B
configuration, not native quality or throughput comparisons.

## Primary Sources And Boundaries

1. C2C cache-to-cache communication
   - https://arxiv.org/abs/2510.03215
   - Boundary: C2C projects/fuses source KV cache into target KV cache and
     reports accuracy/latency gains. LatentWire does not expose source KV in
     this row and cannot claim native speed superiority without running C2C.

2. KVComm selective KV sharing
   - https://arxiv.org/abs/2510.03346
   - Boundary: KVComm shares selected KV pairs/layers. The comparator's `30%`
     layer row is only a conservative byte-floor assumption.

3. KVCOMM cross-context cache reuse
   - https://arxiv.org/abs/2510.12872
   - Boundary: KVCOMM targets online cross-context KV reuse and offset
     alignment. It is a systems neighbor, not the same source-private packet
     access model.

4. QJL
   - https://arxiv.org/abs/2406.03482
   - Boundary: QJL motivates one-bit source-state sketches. A fair baseline
     needs accuracy and native kernel measurements, not only bytes.

5. TurboQuant
   - https://arxiv.org/abs/2504.19874
   - Boundary: TurboQuant is a low-bit vector/KV quantization method with
     near-optimal distortion-rate goals. LatentWire packets are task-level
     evidence sidebands, not vector-fidelity codecs.

6. vLLM / PagedAttention
   - https://arxiv.org/abs/2309.06180
   - Boundary: vLLM is the serving substrate where TTFT, TPOT, goodput, memory,
     HBM traffic, and cache traffic must be measured on NVIDIA hardware.

## Safe Paper Language

Safe:

- The updated Mac-local comparator shows source-private packet rows at `4-15B`
  framed records while conservative one-source-token KV/source-state floors are
  at least tens of times larger.
- The compact HellaSwag row improves packet accounting from `2B` raw / `5B`
  framed to `1B` raw / `4B` framed with exact prediction equivalence.
- C2C, KVComm/KVCOMM, QJL, TurboQuant, and vLLM remain required native
  baselines.

Unsafe:

- Claiming lower TTFT, TPOT, HBM traffic, GPU memory, or goodput before native
  vLLM/SGLang rows.
- Claiming superiority over C2C/KVComm/TurboQuant/QJL from byte floors alone.
- Calling the compact packet a compressed KV/cache or hidden-state codec.
