# Source-Private Test-Log Model-Packet Gate

Date: `2026-04-28`

## Cycle Start

1. Current ICLR readiness: not ready, but this gate is the first positive
   model-mediated source-private packet result on the new test-log surface.
2. Current paper story: source-private residual communication under decoder
   side information. The source has a private tool log, emits a compact packet,
   and the target uses that packet with public candidate-side signatures.
3. Exact blocker to submission: seed repeat, prompt/helper-line ablation,
   stronger source/target model pair, and a less synthetic hidden-test/code
   repair benchmark.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate: model-extracted two-byte `TRACE_SIG` packets on the
   frozen strict-small benchmark.
6. Scale-up rung: strict small model-mediated gate.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

## Method

The source prompt includes the private test log and a source-side copied trace
line:

```text
Private TRACE_SIG line copied from the log: private_tool_trace: TRACE_SIG=GE
Packet:
```

The source model must emit only the two-character packet. The target decoder
then matches that packet to the public candidate field
`handles_trace_signature=<packet>`.

This is not a free-form reasoning result. It is a compact source-private
handoff: source-side private log isolation plus model extraction of the
rate-capped packet.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py \
  --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl \
  --output-dir results/source_private_testlog_packet_llm_packet_20260428 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --dtype float32 \
  --limit 160 \
  --seed 28 \
  --max-new-tokens 8 \
  --no-enable-thinking || true
```

## Results

Model-packet gate pass: `true`.

| Condition | Correct | Accuracy | Mean bytes | Mean tokens | p50 latency ms |
|---|---:|---:|---:|---:|---:|
| target_only | 40/160 | 0.250 | 0.00 | 0.00 | 0.00 |
| matched_model_packet | 150/160 | 0.938 | 1.84 | 2.43 | 162.51 |
| zero_source | 40/160 | 0.250 | 0.00 | 0.00 | 0.00 |
| shuffled_model_packet | 40/160 | 0.250 | 1.84 | 0.92 | 0.00 |
| random_same_byte | 40/160 | 0.250 | 2.00 | 1.00 | 0.00 |
| answer_only | 40/160 | 0.250 | 2.00 | 1.00 | 0.00 |
| answer_masked | 40/160 | 0.250 | 0.00 | 0.00 | 0.00 |
| target_derived_sidecar | 40/160 | 0.250 | 2.00 | 1.00 | 0.00 |
| full_signature_oracle | 160/160 | 1.000 | 2.00 | 1.00 | 0.00 |

Key telemetry:

- exact ID parity: `true`
- exact ID SHA256:
  `fcfd2cfcecfa51f4caae6e5de39cf0632dafb634e4f19db0dfdc12c2ef8dbd2e`
- packet valid rate: `0.91875`
- matched minus best no-source: `+0.6875`
- matched minus best source-destroying control: `+0.6875`
- full signature oracle headroom over matched: `+0.0625`

## Interpretation

This is the first live positive model-mediated result for the source-private
packet story. The gain is not explained by target priors or source-destroying
controls:

- matched model packet: `150/160`
- best no-source/control: `40/160`
- shuffled model packet: `40/160`
- random same-byte packet: `40/160`
- answer-only and answer-masked: `40/160`

The main caveat is that the source prompt includes a source-side copied
`TRACE_SIG` line from the private log. This is acceptable as a tool-log handoff
primitive, but it is not yet a full unstructured-log extraction or code-repair
claim.

## Decision

Promote source-private test-log packet handoff as the current live positive
branch. The branch now has:

- deterministic strict-small pass
- model-extracted strict-small pass
- flat source-destroying controls
- byte/latency/token telemetry
- exact ID parity

Do not yet call it ICLR-ready. The next gate must test robustness:

1. rerun with seeds `29` and `30`
2. add a no-helper-line extraction ablation
3. add one cross-family/source model
4. move from synthetic signatures to private hidden-test/code-repair logs

## Artifacts

- `results/source_private_testlog_packet_llm_packet_20260428/model_packets.jsonl`
- `results/source_private_testlog_packet_llm_packet_20260428/predictions.jsonl`
- `results/source_private_testlog_packet_llm_packet_20260428/summary.json`
- `results/source_private_testlog_packet_llm_packet_20260428/summary.md`
- `results/source_private_testlog_packet_llm_packet_20260428/manifest.json`
- `results/source_private_testlog_packet_llm_packet_20260428/manifest.md`

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_testlog_packet_llm_packet.py \
  tests/test_run_source_private_testlog_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `20 passed`.

## Next Exact Gate

Run `source_private_testlog_packet_llm_packet_seed_repeat_20260428`:

- same frozen benchmark
- seeds `29` and `30`
- same model and controls
- add no-helper-line prompt ablation
- pass if matched model packet remains at least `+15` points over no-source on
  all seeds and controls stay within `2` points of no-source
