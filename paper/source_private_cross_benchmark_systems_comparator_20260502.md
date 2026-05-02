# Cross-Benchmark Systems Comparator V2

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM workshop remains plausible; ICLR is stronger on
  systems accounting after this update but still needs native NVIDIA rows.
- Current story: ARC-Challenge and OpenBookQA are the two headline public
  fixed-byte packet benchmarks, the ARC row is now the promoted `8B` payload /
  `11B` framed b2000 artifact, HellaSwag is a hard diagnostic/method-candidate
  benchmark, and the newest HellaSwag compaction gives a `1B` raw / `4B`
  framed systems rate point.
- Exact remaining blocker: byte-floor comparisons are not native C2C/KVComm,
  TurboQuant, QJL, vLLM, or SGLang measurements.

## Lay Explanation

This update asks: if our method sends only a tiny packet, how much larger would
even a very optimistic source-state or KV-cache transfer be? It does not claim
the packet is faster on a GPU. It only records the byte and source-exposure
accounting that reviewers will ask for before we get native NVIDIA runs.

## Artifact

`results/source_private_cross_benchmark_systems_comparator_20260502/`

## Result

| Row | Role | Packet | Accuracy | Comparator floor |
|---|---|---:|---:|---:|
| ARC-Challenge test | headline public benchmark | `8B` raw / `11B` framed | `0.344` | QJL 1-bit one-token KV floor `69.8x` framed |
| OpenBookQA test | headline second public benchmark | `3B` raw / `6B` framed | `0.378` | QJL 1-bit one-token KV floor `128.0x` framed |
| HellaSwag validation1024 | diagnostic label-copy-threat row | `2B` raw / `5B` framed | `0.461` | QJL 1-bit one-token KV floor `153.6x` framed |
| HellaSwag compact full-validation row | systems-rate candidate, not native | `1B` raw / `4B` framed | `0.619` | QJL 1-bit one-token KV floor `192.0x` framed |

Headline summary:

- pass gate: `True`
- headline-eligible public benchmarks: `2`
- diagnostic / systems rows: `2`
- framed packet range: `4-11B`
- minimum QJL 1-bit one-token KV byte-floor ratio: `69.8x`
- minimum QJL 30%-layer byte-floor ratio: `20.95x`
- minimum KVComm30 fp16 byte-floor ratio: `335.13x`
- minimum TurboQuant 3.5-bit byte-floor ratio: `244.36x`
- native systems complete: `False`

## Interpretation

This is a clean systems-side win for the current paper draft because it moves
the ARC headline row from `15B` framed to `11B` framed and the left edge of the
packet frontier from `5B` framed to `4B` framed without changing predictions on
the compacted HellaSwag rows. It also keeps the reviewer boundary explicit:
C2C/KVComm/KVCOMM communicate or reuse KV/cache state, while TurboQuant/QJL/KV
quantizers compress continuous source state. LatentWire's strict row transmits
only a task-level candidate packet.

The claim is still accounting, not serving throughput. Native vLLM/SGLang rows
must report TTFT, TPOT, goodput, GPU memory, HBM/PCIe/NVLink traffic, accuracy,
and source-exposure flags before we claim a hardware systems win.

## Decision

Promote this as the current Mac-local systems comparator. Use it in paper
tables as the byte/exposure boundary, and keep native NVIDIA baselines as the
explicit systems blocker.
