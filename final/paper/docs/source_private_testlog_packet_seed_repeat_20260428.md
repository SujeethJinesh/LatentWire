# Source-Private Test-Log Packet Seed Repeat

Date: `2026-04-28`

## Cycle Start

1. Current ICLR readiness: not ready, but the source-private test-log packet
   branch now has a positive model-mediated result.
2. Current paper story: a source model extracts a compact private tool-log
   packet and the target decodes it using public candidate-side signatures.
3. Exact blocker to submission: separate protocol-assisted packet emission from
   general unstructured-log extraction, then confirm cross-model and on a less
   synthetic hidden-test/code-repair surface.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate: seed repeat plus no-helper-line ablation.
6. Scale-up rung: strict-small model-mediated confirmation.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

## Subagent Guidance Integrated

- Reviewer: if the helper line is required, frame the method as
  protocol-assisted private tool-log packet emission, not generic natural-log
  extraction. No-helper-line collapse weakens breadth but does not kill the
  narrower handoff claim.
- Planner: aggregate exact same frozen IDs over prompt mode and seed, report
  pass/fail per run, and require source-destroying controls to stay flat.
- Harness audit: greedy decoding means seeds are reproducibility checks rather
  than stochastic diversity; record prompt mode, script/benchmark hashes,
  deterministic decode settings, and nonself shuffled-source provenance.

## Implemented Gate

Updated `scripts/run_source_private_testlog_packet_llm_packet.py` with:

- `--prompt-mode helper_line|full_log`
- prompt-mode metadata in packets and manifest
- benchmark/script hashes in manifest
- explicit `do_sample: false` metadata
- nonself `source_example_id` logging for shuffled model-packet controls

Runs:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py \
  --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl \
  --output-dir results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/helper_seed29 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device mps \
  --dtype float32 \
  --limit 160 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode helper_line \
  --no-enable-thinking || true
```

The same command was run for:

- `helper_seed30`
- `full_log_seed29`
- `full_log_seed30`

## Results

Aggregate pass: `true`, with a narrow interpretation.

| Prompt mode | Seeds | All pass | Mean matched | Min matched | Mean valid packets | Mean lift vs no-source | Mean lift vs controls |
|---|---|---|---:|---:|---:|---:|---:|
| helper_line | `[29, 30]` | `true` | 0.938 | 0.938 | 0.919 | 0.688 | 0.688 |
| full_log | `[29, 30]` | `false` | 0.344 | 0.344 | 0.163 | 0.094 | 0.094 |

Per-run details:

| Run | Mode | Seed | Pass | Matched | Target-only | Best control | Valid packets | p50 latency ms |
|---|---|---:|---|---:|---:|---:|---:|---:|
| helper_seed29 | helper_line | 29 | `true` | 0.938 | 0.250 | 0.250 | 0.919 | 167.09 |
| helper_seed30 | helper_line | 30 | `true` | 0.938 | 0.250 | 0.250 | 0.919 | 164.70 |
| full_log_seed29 | full_log | 29 | `false` | 0.344 | 0.250 | 0.250 | 0.163 | 132.97 |
| full_log_seed30 | full_log | 30 | `false` | 0.344 | 0.250 | 0.250 | 0.163 | 129.01 |

## Interpretation

The positive result is stable for the helper-line protocol: the source-side
private tool log is reduced to the relevant `TRACE_SIG` line, and the model
emits a two-byte packet that transfers private source information to the
target. Controls remain at the target/no-source floor.

The no-helper full-log ablation fails. The source model often emits invalid
strings such as `TRACE`, `FAIL`, or partial markers, and valid packet rate drops
to `0.163`. This means the current claim must be framed as protocol-assisted
private tool-log handoff, not as robust unstructured-log extraction.

Because generation is greedy, seeds `29` and `30` are reproducibility checks,
not stochastic robustness. The identical accuracies are expected and should be
reported as such.

## Decision

Promote the branch only under the narrow claim:

> A source agent can communicate private tool-log state through a compact
> protocol packet that the target decodes with public side information, yielding
> large gains over no-source and source-destroying controls.

Do not claim general log understanding or latent transfer yet.

## Artifacts

- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/aggregate_summary.json`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/aggregate_summary.md`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/manifest.json`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/manifest.md`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/helper_seed29/*`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/helper_seed30/*`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/full_log_seed29/*`
- `results/source_private_testlog_packet_llm_packet_seed_repeat_20260428/full_log_seed30/*`

## Tests

```bash
./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_testlog_packet_llm_packet.py \
  tests/test_run_source_private_testlog_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_llm_packet.py \
  tests/test_run_source_private_evidence_packet_strict_small.py \
  tests/test_run_source_private_evidence_packet_gate.py -q
```

Result: `22 passed`.

## Next Exact Gate

Run `source_private_testlog_packet_cross_model_20260428`:

- same frozen `160` examples
- helper-line protocol only
- source model: `Qwen/Qwen3-0.6B` if cached, or another cached family/model
- same controls and exact IDs
- pass if matched model packets beat no-source by `>=15` points and controls
  stay within `2` points of no-source

Then move to a hidden-test/code-repair variant where the source-side tool
packet is produced from actual private execution evidence rather than synthetic
signature fields.
