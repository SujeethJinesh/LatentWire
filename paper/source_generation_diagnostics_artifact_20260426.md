# Source Generation Diagnostics Artifact

- date: `2026-04-26`
- status: `tooling_smoke_passed`
- scale-up rung: micro tooling smoke
- live branch entering run: source-internal confidence/logit artifact for
  future source routers

## Question

Shallow source-text routers failed live/holdout gates. This tooling gate asks
whether we can collect source-internal greedy-generation confidence features
without mutating existing `source_alone.jsonl` baselines.

## Implementation

Added `scripts/collect_source_generation_diagnostics.py`, which reruns source
generation only and writes a sidecar JSONL keyed by exact `example_id`.

Recorded fields include:

- generated token IDs
- per-token chosen logprob
- per-token entropy
- top-1 probability
- top-1/top-2 logit margin
- aggregate mean/min/final logprob, entropy, probability, and margin
- prediction, normalized numeric prediction, correctness, TTFT, and latency

## Smoke Command

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
./venv_arm64/bin/python scripts/collect_source_generation_diagnostics.py \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --eval-file results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/_artifacts/svamp_eval_70_32_32.jsonl \
  --device mps \
  --dtype float32 \
  --max-new-tokens 8 \
  --limit 2 \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --source-enable-thinking false \
  --output-jsonl .debug/source_generation_diagnostics_smoke/source_diagnostics.jsonl \
  --output-md .debug/source_generation_diagnostics_smoke/source_diagnostics.md
```

The sandboxed MPS run reported the MPS backend as unavailable; the same command
passed with approved escalation outside the sandbox.

## Result

- examples: `2`
- correct: `1/2`
- output JSONL sha256:
  `016e669f76de07666e9d13212e1c2fcc50565daa01e120916665c08f8f2f456f`
- readout sha256:
  `31714855c70d2a84c016a549ba9c9264dc7929fbf004d9003247d130e86c7eec`

The artifact contains the intended source-internal confidence features and
does not overwrite any existing baseline prediction file.

## Tests

- `./venv_arm64/bin/python -m pytest tests/test_collect_source_generation_diagnostics.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/collect_source_generation_diagnostics.py`

## Next Exact Gate

Run the diagnostics collector on `svamp70_live`, then test whether
source-internal logprob/entropy/margin features separate clean source-only wins
from source/text failures. If they do not, switch to the `chal311-380`
source-surface scout before spending C2C or connector compute.
