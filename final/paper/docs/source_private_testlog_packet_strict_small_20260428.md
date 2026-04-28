# Source-Private Test-Log Packet Strict-Small Gate

Date: `2026-04-28`

## Cycle Start

1. Current ICLR readiness: not ready; this gate tests a less protocol-shaped
   source-private packet, but still with deterministic packet generation and
   decoding.
2. Current paper story: source-private residual communication under decoder
   side information. The target sees a public issue and candidate fixes; the
   source sees a private tool/test log; a rate-capped packet lets the target
   select the candidate whose public handled signature matches the private log.
3. Exact blocker to submission: show this source-private effect with
   model-extracted packets and eventually real hidden-test/code-repair evidence,
   not only synthetic deterministic traces.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate: deterministic strict-small test-log packet benchmark.
6. Scale-up rung: strict small gate.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

## Subagent Guidance Integrated

- Reviewer: the gate is only useful if public target prompts do not contain the
  private log, source-destroying controls stay flat, answer/patch-label packets
  fail, and matched-byte text relay cannot explain the gain.
- Planner: use private execution/test-log evidence as the next branch because
  it is naturally producible by agents and tools. The fully realistic
  hidden-code-repair gate can follow after a deterministic contract pass.
- Method scout: prioritize a tool-trace verdict/signature packet before learned
  query bottlenecks. It provides a compact source-derived signal with obvious
  zero, shuffled, random, and target-derived destruction controls.
- Harness audit: preserve exact ID parity, byte accounting, artifact hashes,
  nonself shuffled packets, and leakage telemetry.

## Implemented Gate

Script:

```bash
./venv_arm64/bin/python scripts/run_source_private_testlog_packet_strict_small.py \
  --examples 160 \
  --candidates 4 \
  --seed 28 \
  --budgets 2,4,8,16,32 \
  --output-dir results/source_private_testlog_packet_strict_small_20260428
```

Task:

- `160` frozen examples.
- Target sees a public production issue and `4` candidate patches.
- Each candidate advertises a public `handles_trace_signature` field.
- Source sees a private pytest-style tool log containing `TRACE_SIG=<2-byte>`.
- Matched source packet is the two-byte trace signature.
- Target decodes by matching the packet to the candidate-side handled
  signature.

Controls:

- target-only
- target-wrapper/no-source
- zero-source
- shuffled-source nonself packet
- random same-byte packet
- answer-only packet
- answer-masked packet
- target-derived sidecar
- matched-byte raw test-log truncation
- full structured log oracle
- full signature text oracle

## Results

Strict-small pass: `true`.

| Budget bytes | Matched | Best no-source | Best source-destroying control | Matched-byte text | Full log | Full signature | Pass |
|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 | true |
| 4 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 | true |
| 8 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 | true |
| 16 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 | true |
| 32 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 | 1.000 | true |

Key telemetry for `2` bytes:

- exact ID parity: `true`
- exact ID count: `160`
- exact ID SHA256:
  `fcfd2cfcecfa51f4caae6e5de39cf0632dafb634e4f19db0dfdc12c2ef8dbd2e`
- candidate-pool recall: `1.000`
- target/no-source accuracy: `40/160`, `0.250`
- matched test-log packet accuracy: `160/160`, `1.000`
- best source-destroying control accuracy: `40/160`, `0.250`
- matched-byte raw-log truncation accuracy: `40/160`, `0.250`
- full structured log oracle: `160/160`, `1.000`
- full signature text oracle: `160/160`, `1.000`
- matched minus best no-source: `+0.750`
- matched minus best control: `+0.750`
- matched minus matched-byte text: `+0.750`

Leakage audit:

- public target private-log hits: `0`
- public target `TRACE_SIG` hits: `0`
- matched packet answer-label copies: `0`
- matched packet candidate-label copies: `0`
- matched packet over-budget count: `0`

## Interpretation

This is a strict-small pass for the source-private tool/test-log handoff
contract. It is stronger than the cryptographic digest surface in one important
way: the source packet is a natural artifact a tool or agent can emit from a
private log (`TRACE_SIG=GE` -> `GE`), rather than a hash computation.

It is still not an ICLR-ready positive method. The benchmark is synthetic, the
packetizer is deterministic, and the decoder is protocol-aware. This should be
treated as a clean next surface for model-mediated extraction and then a more
realistic hidden-test/code-repair benchmark.

## Decision

Promote source-private test-log packets to the live branch. Prune further
prompting on cryptographic digest packets. The next decisive gate should keep
the same frozen benchmark but ask a source model to emit the two-byte
`TRACE_SIG` packet from the private test log.

## Artifacts

- `results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl`
- `results/source_private_testlog_packet_strict_small_20260428/sweep_summary.json`
- `results/source_private_testlog_packet_strict_small_20260428/sweep_summary.md`
- `results/source_private_testlog_packet_strict_small_20260428/leakage_audit.json`
- `results/source_private_testlog_packet_strict_small_20260428/leakage_audit.md`
- `results/source_private_testlog_packet_strict_small_20260428/manifest.json`
- `results/source_private_testlog_packet_strict_small_20260428/manifest.md`
- `results/source_private_testlog_packet_strict_small_20260428/predictions_budget{2,4,8,16,32}.jsonl`
- `results/source_private_testlog_packet_strict_small_20260428/summary_budget{2,4,8,16,32}.json`

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_testlog_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `16 passed`.

## Next Exact Gate

Run `source_private_testlog_packet_llm_packet_20260428`:

- freeze the same `160` examples
- source model sees private test log and must output only the two-byte
  `TRACE_SIG` packet
- deterministic target decoder consumes the model packet and public candidate
  signatures
- preserve target-only, zero, shuffled, random, answer-only, answer-masked, and
  target-derived controls
- pass if matched model packets beat best no-source by `>=15` points and all
  source-destroying controls stay within `2` points of no-source
