# Overnight V2 Forced Probe Report

- locality: CPU-only local run; no SSH, CUDA, 30B, or confirm access.
- completion rule: powered verdict only when minimum n and MDE target are met; otherwise `INCONCLUSIVE_UNDERPOWERED`.

## Wall-Clock / n / MDE

| probe | status | wall_clock_seconds | achieved_n_dev | achieved_n_gate | achieved_total_n | achieved_MDE | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EXP1 | `INCONCLUSIVE_UNDERPOWERED` | 380.6 | 346 | 154 | 500 | 0.064935 | INCONCLUSIVE_UNDERPOWERED |
| EXP4 | `INCONCLUSIVE_UNDERPOWERED` | 12379.6 | 70 | 30 | 100 | 0.066667 | INCONCLUSIVE_UNDERPOWERED |

## State Updates

- KVComm/C2C packet smokes: `KILLED_AS_METHOD_EVIDENCE` where deterministic controls explain the signal.
- C_F: `KILLED_CONTROL_CONTAMINATED`.
- CacheWire deployable trace from v1: `INCONCLUSIVE_UNDERPOWERED`, not killed.
- CacheWire oracle: `ORACLE_ALIVE_UNMEASURED_AT_POWER` until EXP1 v2 is powered.
- L_A2: `NEEDS_SCORE_CACHE` unless EXP4 v2 reaches generated/scored candidate power.
- C_A1: `NEXT_GPU_BACKFILL`; foreground remains empty.

## Artifacts

- `results/overnight_v2/20260604_forced_powered_probes/exp1_cachewire_powered_ceiling/summary.json`
- `results/overnight_v2/20260604_forced_powered_probes/exp4_l_a2_generated_solution_ceiling/summary.json`

## EXP1 CacheWire

- receiver feature accuracy: `0.103896`
- receiver+source feature accuracy: `0.129870`
- gain: `{'ci95_high': 0.08441558441558442, 'ci95_low': -0.03896103896103896, 'delta': 0.025974025974025976, 'mde_half_width': 0.06493506493506493, 'n': 154}`
- dense-fusion oracle gain: `{'ci95_high': 0.0, 'ci95_low': 0.0, 'delta': 0.0, 'mde_half_width': 0.0, 'n': 154}`
- GPU command if underpowered: `venv_arm64/bin/python scripts/overnight_v2_forced_probes.py --device cpu --run-exp exp1 --exp1-rows 2500 --exp1-min-rows 500 --batch-size 2 --no-confirm`

## EXP4 L-A2

- generated prompts: `100`
- generated candidates: `1600`
- source-evidence CMI: `0.000000` bits
- combo gain: `{'n': 30, 'delta': 0.03333333333333333, 'ci95_low': 0.0, 'ci95_high': 0.1, 'mde_half_width': 0.06666666666666668}`
- GPU command if underpowered: `venv_arm64/bin/python scripts/overnight_v2_forced_probes.py --device cpu --run-exp exp4 --exp4-prompts 300 --exp4-candidates 16 --exp4-max-new-tokens 96 --no-confirm`
