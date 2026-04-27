# MPS Micro Stronger-Source Surface Gate

- date: `2026-04-27`
- readiness: not ICLR-ready
- rung: micro smoke
- branch: stronger-source answer-masked surface discovery

## Cycle Status

1. Current ICLR readiness: not ready; no positive communication method has
   survived source-destroying controls.
2. Current paper story: CPU-only and old 1.5B Math surfaces are saturated; MPS
   is usable again, so the live branch is fresh stronger-source discovery with
   answer-masking before any learned sidecar or latent connector.
3. Exact blocker: find a source-only target-pool row that is not explained by
   the source final/verified numeric answer.
4. Candidate branches: cached `Qwen/Qwen2.5-7B-Instruct -> Qwen/Qwen3-0.6B`
   surface discovery, with Math-7B as a later higher-cost source if needed.
5. Highest-priority gate: micro same-ID MPS scout with text relay and
   answer-masking audit.
6. Scale-up rung: micro smoke.

## Commands

```bash
./venv_arm64/bin/python scripts/check_mps_blocker.py --json
```

Result: `blocked=false`; PID `31103` absent.

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/mps_micro_qwen25_7b_qwen3_svamp8_surface_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-7B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 8 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

Fallback compatibility canary:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file data/svamp_eval_70.jsonl \
  --results-dir results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427 \
  --translator checkpoints/qwen25_to_qwen3_headhalf_lowrank_ridgecorr_20260419.pt \
  --source-model Qwen/Qwen2.5-Math-1.5B \
  --target-model Qwen/Qwen3-0.6B \
  --methods source target t2t \
  --limit 8 \
  --device mps \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

For each result directory:

```bash
./venv_arm64/bin/python scripts/build_source_contrastive_target_set.py \
  --target target=path=<DIR>/target_alone.jsonl,method=target_alone \
  --source source=path=<DIR>/source_alone.jsonl,method=source_alone \
  --control t2t=path=<DIR>/text_to_text.jsonl,method=text_to_text \
  --min-source-only 1 \
  --date 2026-04-27 \
  --output-json <DIR>/source_contrastive_target_set.json \
  --output-md <DIR>/source_contrastive_target_set.md

./venv_arm64/bin/python scripts/audit_source_surface_answer_masking.py \
  --results-root <DIR> \
  --date 2026-04-27 \
  --output-json <DIR>/answer_masking_audit.json \
  --output-md <DIR>/answer_masking_audit.md
```

## Results

### `Qwen2.5-7B-Instruct -> Qwen3-0.6B`

- artifact: `results/mps_micro_qwen25_7b_qwen3_svamp8_surface_20260427/`
- source: `2/8`, numeric coverage `8/8`
- target: `2/8`, numeric coverage `8/8`
- text relay: `1/8`, numeric coverage `8/8`
- source-only over target: `1` ID, `d64f6e35083ffe8c`
- clean source-only after text relay: `1` ID, `d64f6e35083ffe8c`
- answer-unexplained clean in pool: `0`
- target-or-source oracle: `3/8`
- exact ID parity: true for source, target, and text relay

The one clean source-only ID is fully explained by source final answer `2`.
It is useful as a systems/operational MPS check and a candidate-headroom
diagnostic, not as positive latent/source-side communication evidence.

### `Qwen2.5-Math-1.5B -> Qwen3-0.6B`

- artifact: `results/mps_micro_qwen25math15_qwen3_svamp8_answermasked_20260427/`
- source: `2/8`, numeric coverage `7/8`
- target: `2/8`, numeric coverage `8/8`
- text relay: `2/8`, numeric coverage `8/8`
- source-only over target: `1` ID, `aee922049c757331`
- clean source-only after text relay: `0`
- answer-unexplained clean in pool: `0`
- target-or-source oracle: `3/8`
- exact ID parity: true for source, target, and text relay

The source-only ID is also recovered by text relay and is explained by the
source final answer `1`, so it does not produce a usable side-information
target.

## Decision

Operational pass: MPS generation is usable again and both scouts preserve exact
ID parity, numeric coverage, and sidecar config validation.

Scientific fail: neither micro surface has answer-unexplained clean target-pool
headroom. Do not promote a learned sidecar, zero-init connector, KV transport,
or candidate-syndrome claim from these rows.

Next exact gate: run a different answer-masked discovery surface, preferably a
bounded 7B or Math-7B slice chosen for source/target disagreement rather than
the first eight SVAMP IDs. Promote only if `answer_unexplained_clean_in_pool >
0` and the clean IDs survive text relay, zero-source, shuffled-source,
target-only/slots-only, and random sidecar controls.
