# SVAMP32 C2C Pre-Answer State Gate

- date: `2026-05-05`
- status: `implementation_ready_full_mps_run_stalled`
- code:
  `scripts/analyze_svamp32_c2c_preanswer_state_gate.py`
- tests:
  `tests/test_analyze_svamp32_c2c_preanswer_state_gate.py`
- smoke artifact:
  `.debug/svamp32_c2c_preanswer_state_gate_smoke/preanswer_state_gate.json`

## Question

Can we distill C2C's dense teacher behavior from C2C state before the final
numeric answer appears, rather than from answer-value packets, teacher-prefix
logit deltas, or short-answer candidate scores?

## Implementation

The new gate:

- replays repaired local C2C generation with `output_scores=True`;
- locates the final numeric answer span in generated text;
- builds a `pre_answer_exclusive` feature window that excludes token scores at
  and after the detected answer onset;
- builds a `post_answer_inclusive` leakage-control window that includes the
  answer span;
- combines C2C projector trace summaries with fixed-schema generation-logit
  window summaries;
- reuses the existing leave-one-out SVAMP32 syndrome evaluator and controls:
  matched, zero-source, row-shuffle, label-shuffle, target-only, and slots-only;
- supports `--start-index` and `--limit` so the expensive C2C replay can be run
  in small, resumable slices if full n32 MPS replay stalls.

## Local Smoke

A one-row MPS smoke passed with:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/analyze_svamp32_c2c_preanswer_state_gate.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target_alone=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c_replay=path=results/svamp32_c2c_mps_compat_replay_20260505/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --candidate source_alone=path=results/svamp_exactid_baselines32_20260423/source_alone.jsonl,method=source_alone \
  --candidate text_to_text=path=results/svamp_exactid_baselines32_20260423/text_to_text.jsonl,method=text_to_text \
  --target-set-json results/svamp32_c2c_mps_compat_replay_20260505/c2c_replay_target_set.json \
  --fallback-label target_self_repair \
  --residual-projection-dim 1 \
  --moduli 2,3 \
  --device mps \
  --max-new-tokens 8 \
  --limit 1 \
  --min-numeric-coverage 1 \
  --min-correct 1 \
  --min-clean-source-necessary 0 \
  --output-json .debug/svamp32_c2c_preanswer_state_gate_smoke/preanswer_state_gate.json \
  --output-md .debug/svamp32_c2c_preanswer_state_gate_smoke/preanswer_state_gate.md
```

Smoke output status: `pre_answer_c2c_state_fails_controls`. This only verifies
the implementation path; it is not a scientific result because it uses one row
and a shortened generation budget.

## Full n32 Attempt

The full n32 MPS command loaded both C2C models and then stalled after MPS
command-buffer warnings:

```text
Impacting Interactivity (0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
```

After roughly seven and a half minutes the process was still alive with low CPU
and had produced no output artifact, so it was stopped to avoid leaving a stuck
local MPS job running.

## Decision

The pre-answer gate is now implemented, but the full local n32 MPS run is not
validated. The next exact gate is to run this script in small slices, for
example `--start-index 0 --limit 4`, then aggregate if the slice behavior is
stable, or run the full command on NVIDIA hardware.

## Claim Boundary

No ICLR claim improves from this turn yet. The implementation sharpens the next
experiment; it does not establish a positive method.
