# SVAMP32 C2C Mechanism Projection Probe

- date: `2026-04-26`
- scale-up rung: strict small diagnostic gate
- status: `fails_gate`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- target set:
  `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`

## Readiness Snapshot

Current ICLR readiness: not ready. The same-family SVAMP32 surface has real
C2C headroom, but the project still lacks a deployable source-derived method
that recovers clean C2C residual IDs under source-destroying controls.

Current story: the C2C teacher reaches `16/32` versus target-alone `8/32` and
target self-repair `14/32`, so it remains the best same-pair decision surface.
Prior source-hidden and C2C scalar/residual summary probes failed to recover
clean source-necessary IDs. This cycle tested whether richer deterministic
signed projections of C2C projector residuals make the C2C mechanism readable.

Exact blocker: C2C-mechanism features must recover at least `2/6` clean C2C
residual IDs while zero-source, shuffled-source, label-shuffle, target-only,
and slots-only controls recover none.

## Code Change

Added deterministic signed residual projections to `latent_bridge/c2c_eval.py`.
The default behavior is unchanged; callers can pass
`--residual-projection-dim` to append bucketed signed projections of full
residual deltas and tail residual deltas for each key/value C2C projector
trace.

Added tests covering:

- deterministic residual projection schema
- decoder-free C2C trace feature names
- existing C2C mechanism probe status relabeling

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target target=path=results/svamp_exactid_baselines32_20260423/target_alone.jsonl,method=target_alone \
  --teacher c2c=path=results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl,method=c2c_generate \
  --candidate target_self_repair=path=results/svamp32_query_innovation_query_pool_transport_20260423/target_self_repair_exact32.jsonl,method=target_self_repair \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --fallback-label target_self_repair \
  --moduli 2,3,5,7 \
  --ridge-lambda 1.0 \
  --residual-projection-dim 16 \
  --device mps \
  --max-new-tokens 1 \
  --min-correct 14 \
  --min-clean-source-necessary 2 \
  --min-numeric-coverage 31 \
  --output-json results/svamp32_c2c_mechanism_projection_probe_20260426/prefill_residual_projection16_targetpool_probe.json \
  --output-md results/svamp32_c2c_mechanism_projection_probe_20260426/prefill_residual_projection16_targetpool_probe.md
```

## Evidence

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 13/32 | 0/6 | 3/3 |
| zero-source | 14/32 | 0/6 | 3/3 |
| shuffled-source | 11/32 | 0/6 | 2/3 |
| label-shuffled | 14/32 | 0/6 | 3/3 |
| target-only | 14/32 | 0/6 | 3/3 |
| slots-only | 8/32 | 0/6 | 0/3 |

Compared with the prior residual summary probe, signed projections improve the
matched row from `12/32` to `13/32`, but still remain below zero-source,
label-shuffle, and target-only controls. Clean source-necessary recovery is
`0/6`.

## Decision

Kill C2C scalar/residual summary and signed-projection mechanism features as a
live source-derived method on this SVAMP32 surface. The richer features still
do not recover clean C2C residual IDs beyond target/control artifacts.

Weakened branch: any C2C-mechanism distillation that only summarizes projector
statistics or random projections without a stronger anti-cache/source-control
objective.

Promoted next branch: do not tune more summary/projection features. The next
highest-value move is either:

1. implement a genuinely token/layer-local C2C residual distillation objective
   with held-out source controls and target-cache debiasing, or
2. return to same-family Qwen2.5 -> Qwen3 sidecar/source-surface discovery using
   the sequence-aligned sidecar as an explicit component rather than a toy-only
   proxy.

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_c2c_mechanism_trace.py tests/test_c2c_eval.py -q
./venv_arm64/bin/python -m py_compile latent_bridge/c2c_eval.py scripts/analyze_svamp32_c2c_mechanism_syndrome_probe.py
```

Results: `5 passed`; compile checks passed.

