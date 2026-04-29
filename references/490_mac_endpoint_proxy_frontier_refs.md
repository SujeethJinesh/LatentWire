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

9. **vLLM Metrics**
   - source: https://docs.vllm.ai/en/stable/design/metrics/
   - blocker helped: the paper must use standard serving terms rather than
     inventing ambiguous latency labels.
   - mechanism/design idea: separate TTFT, per-token decode latency, E2E
     latency, prompt tokens, and generation tokens.
   - experiment change: future server runs should mirror vLLM metric names; the
     current CPU proxy is labeled as a local timing proxy.
   - role: measurement standard.

10. **DistServe**
   - source: https://arxiv.org/abs/2401.09670
   - blocker helped: serving papers evaluate TTFT and TPOT/goodput under
     prefill/decode separation, not just one local runtime number.
   - mechanism/design idea: packet systems claims should eventually report
     SLO-style TTFT/TPOT or goodput when GPUs are available.
   - experiment change: does not change the Mac gate; changes the required GPU
     endpoint gate.
   - role: serving benchmark framing.

11. **SnapKV**
   - source: https://arxiv.org/abs/2404.14469
   - blocker helped: KV compression can reduce long-context memory and latency,
     so full-log relay could be compressed internally rather than sent raw.
   - mechanism/design idea: compare packet relay to cache-pruning/cache-
     compression baselines when the task moves beyond the synthetic source-
     private surface.
   - experiment change: no immediate Mac run; keep as a systems baseline family
     in the discussion.
   - role: KV compression baseline.

12. **CacheBlend**
   - source: https://arxiv.org/abs/2405.16444
   - blocker helped: RAG/cache reuse systems can lower TTFT for reused context,
     threatening any simple "full logs are slow" claim.
   - mechanism/design idea: the defensible claim is currently extreme-rate
     source-private payload, not universal serving superiority over cached
     context systems.
   - experiment change: future server gate should include cached full-log or
     reused-context rows when context repeats.
   - role: cache-reuse comparator.

13. **LMCache**
   - source: https://arxiv.org/abs/2510.09665
   - blocker helped: production-style KV transport/offload systems are the
     real systems neighbors for state relay.
   - mechanism/design idea: if the paper claims systems value, it must define
     whether it competes with prompt bytes, cache bytes, or cross-request KV
     reuse.
   - experiment change: future GPU endpoint run should log whether cache reuse
     is enabled and separate prompt-token savings from cache-transport savings.
   - role: production systems baseline.

14. **Prompt Compression in the Wild**
   - source: https://arxiv.org/abs/2604.02985
   - blocker helped: prompt compression may have its own runtime overhead, so
     byte savings alone are not a systems win.
   - mechanism/design idea: report compressor/source-message construction time
     in addition to target TTFT/E2E.
   - experiment change: next endpoint gate should include packet construction
     time and any text-compressor wall time if learned compression is used.
   - role: latest prompt-compression systems caveat.

## Gate Outcome

The Mac endpoint-proxy gate passes on both frozen surfaces at `n=8` and `n=16`
under the canonical prompt and at `n=16`/`n=32` under an `audit` prompt
paraphrase. The stronger `n=32` audit rows put the 2-byte packet at `0.719`
core and `0.844` holdout versus target-only / matched-byte text at
`0.250-0.312`, while query-aware text needs `14` bytes and full hidden-log
relay is `183.2x-186.7x` larger. Full-log TTFT is `+157.4 ms` to `+163.4 ms`
relative to the packet in this local CPU proxy. A deliberately under-specified
`terse` prompt fails on core `n=16`, so the receiver needs a clear public side-
information contract. This is not a server-throughput claim; it is the first
reproducible endpoint-style telemetry row for the systems story.

Follow-up strict-control outcome: the endpoint harness now reports strict-label
accuracy separately from diagnostic-code-mapped accuracy and includes random
same-byte and deranged public-diagnostic-table controls. The `audit` prompt
passes at `n=16` on both surfaces: core packet `0.750` versus best source-
destroying control `0.250`; holdout packet `0.875` versus best source-
destroying control `0.312`; deranged-table control is `0.000` on both. The
strict-label packet accuracy is much lower (`0.062` core, `0.250` holdout), so
the systems claim should be phrased as protocol-code decoding with public side
information, not natural-language label generation.

Follow-up `n=32` strict-control outcome: the same `audit` endpoint gate passes
at `n=32` on both frozen surfaces. Core packet accuracy is `0.719` versus best
source-destroying control `0.281`; holdout packet accuracy is `0.844` versus
best source-destroying control `0.312`. Random same-byte packets stay at
`0.031`/`0.094`, deranged public tables stay at `0.000`, and full hidden-log
relay adds `+159.2 ms`/`+185.8 ms` p50 TTFT versus the packet. This strengthens
the local byte/TTFT systems case but does not change the literature framing:
the claim remains source-private protocol-code communication with strict
controls, not a replacement for prompt compression or server-side KV/cache
systems baselines.

Parser-risk follow-up: payload-gated rescoring fixed a loophole where the
receiver could hallucinate an untransmitted diagnostic key and still receive
diagnostic-mapped credit. This demotes the audit rows to near-miss/fail under
the valid-output rule, although the source signal remains strong (`n=64` core
packet `0.750` versus target `0.250` and best source-destroying control
`0.203`). A new `label_strict` receiver prompt passes `n=16` on core and
holdout with exact-label outputs and valid rate `1.000` (`0.688`/`0.625`
packet accuracy versus target/control `0.250`). This changes the next
experiment: scale `label_strict` to `n=32` and `n=64` before claiming endpoint
receiver robustness.

Follow-up: `label_strict` now passes `n=32` on core and holdout. Core packet is
`0.688` versus target/control `0.250`; holdout packet is `0.656` versus
target/control `0.250`; packet valid rate is `1.000` on both, and full-log p50
TTFT remains `+164.8 ms`/`+167.1 ms` slower than the packet. This makes
`label_strict` the live endpoint receiver branch; the next experiment is
`n=64` label-strict core+holdout.

Follow-up: `label_strict` now passes `n=64` on core and holdout. Core packet is
`0.703` versus target/control `0.250`; holdout packet is `0.672` versus
target/control `0.250`; packet valid rate is `1.000` on both, and full-log p50
TTFT is `+217.2 ms`/`+192.7 ms` slower than the packet. The next experiment is
paired uncertainty on these n64 rows, then frozen `n=160` label-strict
core+holdout.

Uncertainty follow-up: the paired bootstrap gate passes on the `n=64`
label-strict rows. Minimum packet-vs-target lower CI is `+0.297`, minimum
packet-vs-best-source-destroying-control lower CI is `+0.297`, and minimum
strict-label packet-vs-target lower CI is `+0.281`. Exact sign tests have zero
packet losses versus target-only on both core and holdout. Query-aware text is
kept as a rate/quality comparator because it is accuracy-comparable but uses
`14` bytes rather than the packet's `2` bytes. The next experiment is now
frozen `n=160` label-strict core+holdout.

Scale-up follow-up: core `n=160` label-strict passes the same all-condition CPU
endpoint gate. Packet accuracy is `0.675` and strict-label accuracy is `0.662`
versus target/matched-byte text `0.250` and best source-destroying control
`0.250`; paired lower CIs are `+0.350` versus target and best control. Query-
aware text is slightly higher accuracy (`0.694`) but costs `14` bytes; full-log
relay is `183.2x` larger and adds `+164.3 ms` p50 TTFT. The next experiment is
holdout `n=160`, then paired core+holdout `n=160` uncertainty.

Medium-rung follow-up: holdout `n=160` also passes. Packet accuracy is `0.688`
and strict-label accuracy is `0.675` versus target/matched-byte text `0.250`
and best source-destroying control `0.250`; full-log relay is `186.8x` larger
and adds `+183.5 ms` p50 TTFT. The combined core+holdout `n=160` paired
uncertainty gate passes with minimum lower CIs `+0.350` versus target and best
control, and `+0.338` for strict-label packet versus target. The local endpoint
medium rung is now cleared; the next systems blocker is a real server-side
TTFT/throughput run.
