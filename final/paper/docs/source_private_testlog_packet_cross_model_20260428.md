# Source-Private Test-Log Packet Cross-Model Gate

Date: `2026-04-28`

## Cycle Start

1. Current ICLR readiness: not ready, but the live branch has stable
   model-mediated helper-line results.
2. Current paper story: protocol-assisted source-private tool-log packet
   handoff. A source model extracts a compact private `TRACE_SIG` packet; the
   target decodes it using public candidate-side signatures.
3. Exact blocker to submission: show the result is not one-source-model
   specific, then move from synthetic signatures to real hidden-test/code-repair
   logs.
4. Current live branch: source-private test-log packet handoff.
5. Highest-priority gate: cross-model helper-line extraction on the same frozen
   `160` examples.
6. Scale-up rung: cross-model strict-small falsification.

MPS guard:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Result: no live blocker was present.

## Subagent Guidance Integrated

- Reviewer: helper-line protocol remains publishable only under a narrow
  protocol-assisted private-log packet claim. Cross-model success is required,
  and TinyLlama-style failure should be treated as a model capability boundary.
- Planner: run same frozen IDs, same helper-line protocol, same deterministic
  decoder, same source-destroying controls. Promote only if at least one
  non-Qwen model passes.

## Commands

Example command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_testlog_packet_llm_packet.py \
  --benchmark-jsonl results/source_private_testlog_packet_strict_small_20260428/benchmark.jsonl \
  --output-dir results/source_private_testlog_packet_cross_model_20260428/phi3_mini_helper \
  --model microsoft/Phi-3-mini-4k-instruct \
  --device mps \
  --dtype float32 \
  --limit 160 \
  --seed 28 \
  --max-new-tokens 8 \
  --prompt-mode helper_line \
  --no-enable-thinking || true
```

Runs:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen3-0.6B`
- `microsoft/Phi-3-mini-4k-instruct`
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

Attempted but not counted:

- `meta-llama/Llama-3.2-1B-Instruct`: local cache entry was incomplete for
  offline `transformers` loading.

## Results

Cross-model gate pass: `true`.

| Run | Model | Family | Pass | Matched | Target-only | Best control | Valid packets | p50 latency ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen25_0_5b_helper | Qwen/Qwen2.5-0.5B-Instruct | qwen2.5 | `true` | 0.938 | 0.250 | 0.250 | 0.919 | 164.86 |
| qwen3_0_6b_helper | Qwen/Qwen3-0.6B | qwen3 | `true` | 1.000 | 0.250 | 0.250 | 1.000 | 334.17 |
| phi3_mini_helper | microsoft/Phi-3-mini-4k-instruct | phi3 | `true` | 0.912 | 0.250 | 0.250 | 0.950 | 595.25 |
| tinyllama_1_1b_helper | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | tinyllama | `false` | 0.250 | 0.250 | 0.250 | 0.000 | 496.49 |

Aggregate:

- passing models: `3/4`
- non-Qwen passing models: `phi3_mini_helper`
- mean matched accuracy across all models: `0.775`
- mean matched accuracy among passing models: `0.950`
- controls stayed at target/no-source floor for every model

## Interpretation

The helper-line packet protocol generalizes beyond the original source model:

- same-vendor/generation: `Qwen3-0.6B` reaches `160/160`
- cross-family: `Phi-3-mini` reaches `146/160`
- negative capability row: `TinyLlama` emits the instruction phrase rather than
  the packet and stays at target-only

This supports a narrow cross-model claim for capable instruction-tuned source
models. It does not support universal model-agnostic extraction, and it does
not repair the failed no-helper full-log ablation.

## Decision

Promote the branch as cross-model protocol-assisted private tool-log packet
handoff. Keep the following caveats explicit:

- helper-line protocol is required
- deterministic target decoder
- synthetic signature benchmark
- TinyLlama failure shows a capability boundary

## Artifacts

- `results/source_private_testlog_packet_cross_model_20260428/cross_model_summary.json`
- `results/source_private_testlog_packet_cross_model_20260428/cross_model_summary.md`
- `results/source_private_testlog_packet_cross_model_20260428/manifest.json`
- `results/source_private_testlog_packet_cross_model_20260428/manifest.md`
- per-model subdirectories under
  `results/source_private_testlog_packet_cross_model_20260428/`

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

Build `source_private_testlog_packet_hidden_repair_smoke_20260428`:

- public side: small Python function, public issue, and candidate patches
- source-private side: actual hidden pytest failure log containing a compact
  diagnostic signature or assertion code
- source model emits a helper-line packet derived from private execution
  evidence
- target decoder selects candidate patch using public candidate metadata
- controls: zero, shuffled, random same-byte, answer-only, answer-masked,
  target-derived, matched-byte text, full-log oracle
- pass if matched model packet beats best no-source by `>=15` points and all
  source-destroying controls stay within `2` points
