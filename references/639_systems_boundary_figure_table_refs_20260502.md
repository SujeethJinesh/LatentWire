# References: Systems Boundary Figure/Table V3

Web check: 2026-05-02. Scope: primary-source boundary checks for
`results/source_private_systems_boundary_figure_table_20260502/`.

## Local Artifact

- Script: `scripts/build_source_private_systems_boundary_figure_table.py`
- Result: `results/source_private_systems_boundary_figure_table_20260502/`
- Pass gate: `True`
- Packet framed-byte range: `4-15B`
- Minimum source-state byte floor: `768B`
- Minimum source-state floor versus largest packet: `51.2x`
- Native NVIDIA systems complete: `False`

## Primary Sources And Boundaries

1. C2C cache-to-cache communication
   - https://arxiv.org/abs/2510.03215
   - https://openreview.net/forum?id=LeatkxrBCi
   - Boundary: C2C projects/fuses source KV cache into target KV cache. It is
     the closest cache-transfer communication baseline, but LatentWire cannot
     claim a native win until C2C is run on the same benchmark/hardware with
     source-KV exposure and byte accounting.

2. KVComm selective KV sharing
   - https://arxiv.org/abs/2510.03346
   - https://openreview.net/forum?id=F7rUng23nw
   - Boundary: KVComm communicates selected KV pairs/layers and reports a
     selective regime with as few as `30%` of layers' KV pairs. The V3 table
     uses this only as a byte-floor assumption.

3. KVCOMM cross-context cache reuse
   - https://arxiv.org/abs/2510.12872
   - Boundary: KVCOMM aligns and reuses cross-context KV caches for prefill.
     It is a serving/cache-reuse neighbor, not the same source-private packet
     threat model.

4. Q-KVComm
   - https://arxiv.org/abs/2512.17914
   - Boundary: Q-KVComm transmits compressed KV-cache representations and
     extracted facts. It remains source-state communication, not a
     source-private task packet.

5. KIVI
   - https://arxiv.org/abs/2402.02750
   - Boundary: KIVI is a tuning-free asymmetric `2-bit` KV-cache quantization
     method. It compresses runtime KV cache state and retains a different
     systems objective from LatentWire packet transfer.

6. KVQuant
   - https://arxiv.org/abs/2401.18079
   - https://openreview.net/forum?id=0LXotew9Du
   - Boundary: KVQuant targets accurate sub-4-bit KV-cache quantization. It
     defines a serious low-bit KV byte floor, not a task-packet method.

7. QJL
   - https://arxiv.org/abs/2406.03482
   - https://openreview.net/forum?id=dRilx3vAIH
   - Boundary: QJL uses JL transforms and sign-bit quantization for KV-cache
     compression. It motivates a one-bit source-state floor but needs native
     accuracy and kernel measurements for a fair systems comparison.

8. TurboQuant
   - https://arxiv.org/abs/2504.19874
   - https://openreview.net/forum?id=tO3ASKZlok
   - Boundary: TurboQuant is online vector/KV quantization with reported
     quality neutrality at `3.5` bits/channel. LatentWire should cite it as a
     vector/KV quantization boundary, not as cross-model task-packet transfer.

9. vLLM / PagedAttention
   - https://arxiv.org/abs/2309.06180
   - Boundary: vLLM manages KV-cache serving memory. It defines the native
     TTFT/TPOT/goodput/GPU-memory/HBM measurement substrate.

10. SGLang / RadixAttention
    - https://arxiv.org/abs/2312.07104
    - Boundary: SGLang uses RadixAttention for KV-cache reuse in structured
      language-model programs. It is a serving baseline, not a source-private
      communication protocol.

## Safe Paper Language

Safe:

- LatentWire currently communicates `4-15B` framed task packets with no source
  text, source KV, or source hidden-vector exposure.
- Conservative one-token KV/source-state floors begin at `768B`, or `51.2x`
  the largest current packet.
- C2C, KVComm/KVCOMM, Q-KVComm, QJL, TurboQuant, KIVI, KVQuant, vLLM, and
  SGLang remain native systems baselines or byte-floor comparators.

Unsafe:

- Claiming lower TTFT, TPOT, HBM traffic, GPU memory, or goodput before native
  vLLM/SGLang/C2C/KVComm runs.
- Claiming to beat C2C/KVComm/TurboQuant/QJL from byte floors alone.
- Describing LatentWire packets as compressed KV cache, hidden-state relay, or
  general latent communication without the current source-private packet
  qualification.
