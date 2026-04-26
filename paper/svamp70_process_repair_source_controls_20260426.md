# SVAMP70 Process-Repair Source Controls

- date: `2026-04-26`
- status: process-repair route fails source-control gate
- readiness: not ICLR-ready
- commit at run time: `ff9b95ab2f72e12163c1efed8a4cac6012fc9281`

## Start Status

- current paper story: old held-out process-repair rows were the strongest
  recovered result after the latest C2C-residual connector failures.
- exact blocker: decide whether process-repair gains are source-derived
  communication or target-side repair / route-selection artifacts.
- live branch: process-repair selected routes over three stochastic
  `rotalign_kv_gate_0.10` route pools.
- scale-up rung: medium source-control falsification.

## Matched Baseline

On `data/svamp_eval_70.jsonl`, the existing matched-source process-repair row:

| Method | Accuracy | Delta vs Target | Delta vs Target Self-Repair |
|---|---:|---:|---:|
| target-alone | `21/70` | - | - |
| selected-route no repair | `25/70` | `+4/70` | `-10/70` |
| target self-repair | `35/70` | `+14/70` | - |
| process-repair selected route | `38/70` | `+17/70` | `+3/70` |

The matched process-repair row looked positive versus target self-repair, but
only by `3` examples.

## Controls Run

Two source-destroying controls were run with the same three-salt route-pool and
repair selector contract:

1. `zero_source_kv`: source K/V is zeroed before translation.
2. `shuffled_source_prompt`: source prompt is replaced by a deterministic
   shuffled example before source K/V construction.

Commands are captured in:

- `.debug/run_svamp70_process_repair_zero_kv_control_20260426.sh`
- `.debug/run_svamp70_process_repair_shuffled_source_control_20260426.sh`

## Results

Raw selected route before repair:

| Condition | Salt 0 | Salt 1 | Salt 2 |
|---|---:|---:|---:|
| matched-source original | `not rerun` | `not rerun` | `not rerun` |
| zero-source K/V | `23/70` | `17/70` | `16/70` |
| shuffled-source prompt | `20/70` | `20/70` | `17/70` |

Repaired rows:

| Condition | Selected No Repair | Target Self-Repair | Process Repair |
|---|---:|---:|---:|
| matched-source original | `25/70` | `35/70` | `38/70` |
| zero-source K/V | `22/70` | `35/70` | `35/70` |
| shuffled-source prompt | `26/70` | `35/70` | `37/70` |

Source-control gate:

- matched correct: `38/70`
- target correct: `21/70`
- target-self repair correct: `35/70`
- matched-only vs target-self IDs: `3`
- zero-source control overlaps matched-only IDs: `1/3`
- shuffled-source control overlaps matched-only IDs: `3/3`
- source-specific matched-only IDs after both controls: `0`

## Decision

Kill process-repair selected routes as a source-communication method on this
SVAMP70 surface. The aggregate matched row remains numerically strong, but the
route-specific improvement over target self-repair is fully recovered by a
source-destroying shuffled-source control.

This branch can still be cited as:

- a target-side repair baseline;
- evidence that route/candidate diversity has headroom;
- a warning that repair gains require strict source controls.

It must not be promoted as latent/source communication without a new
source-derived candidate or route signal.

## Next Gate

Stop verifier/repair-only tuning until a source-derived signal exists. The next
highest-value branch is the true source-conditioned soft-prefix or gated
cross-attention logprob objective:

- train a source-conditioned prefix/cross-attention connector directly on
  gold-vs-distractor logprob;
- include matched-source, zero-source, shuffled-source, target-only learned
  prefix, slots-only learned prefix, projected soft prompt, and label-shuffled
  controls;
- run only the teacher-forced pre-generation gate first;
- promote only if matched-source recovers at least `2/6` clean IDs with positive
  matched-minus-best-control margin and target-self preservation.

## Artifacts

- result manifest:
  - `results/process_repair_source_controls_20260426/manifest.md`
- zero-source attribution:
  - `results/process_repair_source_controls_20260426/svamp70_zero_source_kv_attribution.md`
  - sha256: `6d83b6b1ede1e166bd1e18629239188ffc396942f90ae1542b5268a078ebb92c`
- zero-source gate:
  - `results/process_repair_source_controls_20260426/svamp70_zero_source_kv_source_control_gate.md`
  - sha256: `802e521354b9b0c5673f8262ea1b617b9dbf57427f031c33d78fd6d92288e837`
- combined attribution:
  - `results/process_repair_source_controls_20260426/svamp70_zero_and_shuffled_source_attribution.md`
  - sha256: `a7b9d3594392721b29d1db3c0036c6750aabb98209c43a7b78dc9b377e946875`
- combined source-control gate:
  - `results/process_repair_source_controls_20260426/svamp70_zero_and_shuffled_source_control_gate.md`
  - sha256: `05e7c38e73f012e47345f7430fac2e93d9177a51e6505cae62cffaefd919ca72`
- source-control analyzer:
  - `scripts/analyze_process_repair_source_controls.py`
- tests:
  - `tests/test_analyze_process_repair_source_controls.py`
- reference memo:
  - `references/463_process_repair_source_control_followup_refs.md`

Large JSONL telemetry is intentionally untracked; `sha256.txt` records paths and
hashes for regeneration/provenance.

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_process_repair_source_controls.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_process_repair_source_controls.py
./venv_arm64/bin/python -m json.tool references/research_memo_manifest.json >/dev/null
```
