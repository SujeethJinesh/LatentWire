# Source-Private PQ Systems Comparison Table

Date: 2026-04-30

## Status

Current readiness: stronger scoped positive-method paper, still not comfortably
ICLR-full. COLM workshop readiness is strong.

Current paper story: source-private residual communication with decoder side
information. A source with private evidence sends a tiny packet; the target uses
public candidate state and its own receiver state to select the right answer.
The systems story is boundary traffic and private-state exposure, not generic
latent transfer.

Exact blocker addressed here: after the n500 PQ/OPQ/Hadamard gates, the systems
evidence was scattered across artifacts. Reviewers need one compact table that
puts packets, scalar WZ, text relay, frozen verifier consumption, KV/cache
floors, and external C2C/KVComm-style baselines on the same accounting surface.

Layman explanation: this asks, "What actually gets sent?" A tiny packet sends a
few bytes and keeps private text/KV inside the source. Text baselines send
private words. KV baselines send model internals. The table makes those
tradeoffs explicit instead of pretending they are the same kind of method.

## Gate

- script:
  `scripts/build_source_private_pq_systems_comparison_table.py`
- test:
  `tests/test_build_source_private_pq_systems_comparison_table.py`
- artifact:
  `results/source_private_pq_systems_comparison_table_20260430/`
- references:
  `references/538_pq_systems_comparison_refs_20260430.md`

Inputs:

- n500 PQ packet gate:
  `results/source_private_product_codebook_packet_gate_n500_20260430/`
- n500 geometry knockout stress:
  `results/source_private_product_codebook_geometry_knockout_stress_n500_20260430/`
- n500 cached decode frontier:
  `results/source_private_product_codebook_decode_frontier_n500_20260430/`
- frozen Qwen3 verifier consumption trace:
  `results/source_private_verifier_consumption_trace_20260430/qwen3_seed31_core_holdout_n160_binary_logprob_combined_cpu/`
- packet trace card v2:
  `results/source_private_packet_trace_card_v2_20260430/`
- systems rate/assumption frontier:
  `results/source_private_systems_rate_assumption_frontier_20260430/`

## Headline Results

Artifact headline:

- pass gate: `true`
- PQ geometry rows: `4`
- PQ mitigation rows: `3`
- PQ min delta vs best source-destroying control: `+0.212`
- canonical PQ max cached decode p50: `0.0212 ms`
- protected Hadamard min unique payloads: `386/500`
- frozen verifier min accuracy: `1.000`
- frozen verifier max Mac CPU p50: `1674.1 ms`
- same-byte text max accuracy: `0.250`
- query-aware text raw-byte ratio: `7.0x`
- full-log raw-byte ratio min: `183.25x`
- KV raw-byte ratio min: `10752.0x`

Representative rows:

| Method | Bytes | Accuracy | Best control | Exposure | Use |
|---|---:|---:|---:|---|---|
| canonical PQ | 4 | 0.482-0.520 | 0.268 | no source text/KV | baseline PQ packet |
| utility-OPQ PQ | 4 | 0.480-0.514 | 0.268 | no source text/KV | public-mean-sensitive geometry |
| protected Hadamard PQ | 4 | 0.498-0.514 | 0.268 | no source text/KV | hardware-friendly geometry |
| utility-protected Hadamard PQ | 4 | 0.504-0.516 | 0.268 | no source text/KV | strongest lookup-risk mitigation |
| scalar WZ residual packet | 4 | 0.424-0.504 | 0.268 | no source text/KV | scalar syndrome comparator |
| frozen Qwen3 verifier packet | 2 | 1.000 | 0.250 | no source text/KV | model-mediated receiver |
| same-byte structured text | 2 | 0.250 | target | exposes text | text control |
| query-aware text oracle | 14 | 1.000 | n/a | exposes text | text catch-up row |
| full hidden-log relay | 366.5+ | n/a | n/a | exposes private text | high-rate oracle relay |
| QJL 1-bit KV byte floor | 21504 | n/a | n/a | exposes source KV | accounting contrast |
| KIVI/KVQuant 2-bit KV floor | 43008 | n/a | n/a | exposes source KV | accounting contrast |

External rows for C2C, KVComm/Q-KVComm, TurboQuant, QJL, LLMLingua, and Gist
tokens are marked as reference/accounting rows, not LatentWire pass rows.

## Interpretation

Promote:

- The systems claim is now reviewer-readable: source-private packets keep
  private text and source KV/cache state behind the boundary while preserving
  controlled transfer.
- Geometry-mitigated PQ is a stronger systems/code contribution than canonical
  PQ alone: utility-OPQ improves public-mean top-byte sensitivity, and protected
  Hadamard reduces singleton-payload behavior with a structured rotation.
- The frozen Qwen3 verifier remains the best model-mediated consumption row,
  but its Mac CPU latency is a caveat, not a production speedup.
- KV/cache baselines are included honestly as byte-floor and exposure
  contrasts, not dismissed.

Do not overclaim:

- No production GPU/vLLM TTFT, TPOT, goodput, HBM, or interconnect measurement
  is added here.
- C2C/KVComm are real and relevant, but they move source KV/cache state; the
  comparison is about privacy boundary and byte object, not universal
  superiority.
- Query-aware text catches up at higher raw bytes while exposing private text.
- This table supports source-private residual communication, not broad
  protocol-free latent reasoning.

## Readiness Impact

This strengthens the systems contribution and makes the current three main
technical contributions cleaner:

1. strict source-private benchmark/control protocol,
2. frozen target-verifier packet consumption,
3. geometry-mitigated PQ residual packets with cached decode, byte-causality
   diagnostics, lookup-risk mitigation, and explicit systems/exposure table.

Comfortable ICLR still needs one of:

- product-codebook model-mediated receiver evidence,
- n500 frozen-verifier or batched/fused receiver evidence,
- native GPU/vLLM/KV TTFT/TPOT/goodput telemetry,
- broader cross-family or less synthetic benchmark evidence.

Next exact gate: product-codebook model-mediated receiver at n256 first, then
n500 only if strict controls pass. If that fails, run seed-repeat/held-out-remap
stress for utility-protected Hadamard PQ.

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_build_source_private_pq_systems_comparison_table.py -q
```

Outcome: `1 passed in 0.03s`.
