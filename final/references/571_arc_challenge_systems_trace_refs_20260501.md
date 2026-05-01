# ARC-Challenge Systems Trace References, 2026-05-01

## Local Finding

Artifacts:

- `results/source_private_arc_challenge_systems_trace_20260501/`
- `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_validation/`
- `results/source_private_arc_challenge_fixed_packet_gate_20260501_qwen05_hashed_test/`
- `results/source_private_iclr_evidence_bundle_20260501/`

The regenerated ARC shared-basis endpoint now carries Mac-local phase and
systems metadata. On official ARC-Challenge test, the endpoint remains positive:
matched/target/same-byte text is `0.344/0.265/0.311`, with CI95 lower bound
versus target `0.044`.

Systems trace headline:

- source scoring: `251.0 ms/question` on local CPU Qwen2.5-0.5B;
- receiver sparse decode: `31.7/104.3 us` p50/p95 in the Python path;
- payload/framed record: `12B/15B`;
- single-request cacheline/DMA accounting: `64B/128B`;
- batch-64 line/DMA bytes per request: `15.0B/16.0B`;
- test candidate feature cache: `13.73 MiB` float64 (`6.87 MiB` fp32 floor);
- peak process RSS during the regenerated test run: `7261.3 MiB`.

## Primary Sources

- ARC / AI2 Reasoning Challenge. https://arxiv.org/abs/1803.05457
- Cache-to-Cache communication. https://openreview.net/forum?id=LeatkxrBCi
- KVComm selective KV sharing. https://arxiv.org/abs/2510.03346
- Communicating Activations Between Language Model Agents.
  https://arxiv.org/abs/2501.14082
- Relative Representations. https://arxiv.org/abs/2209.15430
- QJL. https://arxiv.org/abs/2406.03482
- TurboQuant. https://arxiv.org/abs/2504.19874
- vLLM / PagedAttention. https://arxiv.org/abs/2309.06180
- vLLM serving metrics. https://docs.vllm.ai/en/stable/design/metrics/

## Safe Claim Boundary

Safe: the ARC endpoint has a source-private 12B payload, a 15B framed boundary
record, explicit cacheline/DMA accounting, Mac-local phase timings, and
process RSS for the full official ARC test split.

Unsafe: claiming production serving speedups, HBM savings, or native wins over
C2C, KVComm/KVCOMM, TurboQuant, QJL, or vLLM before NVIDIA/vLLM measurements
exist.
