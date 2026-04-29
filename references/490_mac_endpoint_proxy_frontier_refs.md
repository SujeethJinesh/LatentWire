# Mac Endpoint-Proxy Frontier References

- date: `2026-04-29`
- blocker: the source-private packet had a strong byte-rate frontier, but the
  paper still needed endpoint-style prompt-token, TTFT, and E2E timing evidence
  against structured text and hidden-log relay.

## Sources And Experiment Implications

1. **vLLM / PagedAttention**
   - source: https://arxiv.org/abs/2309.06180
   - blocker helped: reviewers will expect systems claims to distinguish prompt
     length, prefill/cache behavior, TTFT, and E2E decode cost.
   - mechanism/design idea: report serving-facing timing metrics rather than
     only payload bytes.
   - experiment change: the Mac endpoint-proxy gate records prompt tokens, p50
     TTFT, p95 TTFT, p50 E2E, and p95 E2E for every relay condition.
   - role: systems framing and future serving baseline.

2. **LLMLingua**
   - source: https://aclanthology.org/2023.emnlp-main.825/
   - blocker helped: a reviewer can argue that the packet is just prompt
     compression.
   - mechanism/design idea: compare against compressed text relays at matched
     and larger byte budgets.
   - experiment change: the endpoint proxy includes matched-byte text,
     query-aware diagnostic-span text, structured JSON, and structured free-text
     relays.
   - role: baseline family and framing.

3. **LongLLMLingua**
   - source: https://aclanthology.org/2024.acl-long.91/
   - blocker helped: hidden-log relay creates long-context / position-bias
     concerns, not just raw byte overhead.
   - mechanism/design idea: test compact source payloads against a full private
     log relay with the same receiver model.
   - experiment change: the gate includes a full hidden-log condition and logs
     its prompt-token, TTFT, and E2E deltas versus the packet.
   - role: long-context prompt-compression baseline family.

4. **LLMLingua-2**
   - source: https://aclanthology.org/2024.findings-acl.57/
   - blocker helped: learned task-agnostic prompt compressors are a stronger
     baseline than lexical truncation.
   - mechanism/design idea: separate language-preserving compression from
     source-private diagnostic communication.
   - experiment change: the current gate does not run LLMLingua-2 weights, but
     its query-aware and structured-text relays are a cheap first-stage control;
     a future endpoint run should add an actual LLMLingua-2 relay if cached.
   - role: future baseline.

5. **KIVI**
   - source: https://arxiv.org/abs/2402.02750
   - blocker helped: KV-cache compression papers set the bar for reporting
     memory/byte savings without losing task utility.
   - mechanism/design idea: treat packet transport as an extreme-rate cache or
     state sidecar and compare utility per byte.
   - experiment change: endpoint-proxy rows are added to the CPU systems
     frontier beside KV/TurboQuant-inspired codec comparators.
   - role: systems baseline and paper positioning.

6. **QJL**
   - source: https://arxiv.org/abs/2406.03482
   - blocker helped: low-bit/random-projection state transport is a natural
     competing explanation for compact messages.
   - mechanism/design idea: small packets should be compared to
     geometry-preserving sketches, not only text.
   - experiment change: this cycle does not add a new QJL row, but the aggregate
     frontier keeps QJL/protected-residual comparators adjacent to the endpoint
     rows.
   - role: compression comparator.

7. **TurboQuant**
   - source: https://arxiv.org/abs/2504.19874
   - blocker helped: modern quantization work argues for rotation/preconditioned
     coordinates before low-bit transport.
   - mechanism/design idea: source-private messages should report whether a
     codec or coordinate transform improves the byte frontier.
   - experiment change: the endpoint-proxy gate keeps the promoted packet
     separate from the already-pruned protected residual/TurboQuant-style
     comparator.
   - role: compression inspiration and ablation framing.

8. **CacheGen**
   - source: https://arxiv.org/abs/2310.07240
   - blocker helped: cross-session or distributed cache transfer is a direct
     systems neighbor to source-private state relay.
   - mechanism/design idea: payload size and endpoint latency need to be logged
     for transported state, not just accuracy.
   - experiment change: the current Mac proxy is explicitly scoped as a local
     CPU timing proxy; the next GPU/server run should compare packet relay
     against cache/log relay under real serving.
   - role: systems baseline family.

## Gate Outcome

The Mac endpoint-proxy gate passes on both frozen surfaces at `n=8` and `n=16`
with a Qwen3-0.6B CPU receiver. The stronger `n=16` rows put the 2-byte packet
at `0.688` accuracy versus target-only and matched-byte text at `0.250`, while
query-aware text needs `14` bytes and full hidden-log relay is
`183.2x-186.7x` larger. Full-log TTFT is `+165.4 ms` to `+190.7 ms` relative to
the packet in this local CPU proxy. This is not a server-throughput claim; it
is the first reproducible endpoint-style telemetry row for the systems story.
