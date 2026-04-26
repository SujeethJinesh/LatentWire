# Process Repair Source Controls Manifest

- date: `2026-04-26`
- status: `process_repair_source_controls_do_not_clear_gate`
- matched artifact:
  - `results/process_repair_holdout_20260421/qwen_svamp70_process_repair_controls_strict_selector_telemetry.jsonl`
- control artifacts:
  - zero-source K/V route pools and repair:
    - `results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_salt0_telemetry.jsonl`
    - `results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_salt1_telemetry.jsonl`
    - `results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_salt2_telemetry.jsonl`
    - `results/process_repair_source_controls_20260426/qwen_svamp70_zero_source_kv_process_repair_controls_telemetry.jsonl`
  - shuffled-source prompt route pools and repair:
    - `results/process_repair_source_controls_20260426/qwen_svamp70_shuffled_source_prompt_salt0_telemetry.jsonl`
    - `results/process_repair_source_controls_20260426/qwen_svamp70_shuffled_source_prompt_salt1_telemetry.jsonl`
    - `results/process_repair_source_controls_20260426/qwen_svamp70_shuffled_source_prompt_salt2_telemetry.jsonl`
    - `results/process_repair_source_controls_20260426/qwen_svamp70_shuffled_source_prompt_process_repair_controls_telemetry.jsonl`

## Decision

The matched process-repair row reaches `38/70`, but its `3` matched-only wins
over target self-repair are fully recovered by source-destroying controls:

- zero-source K/V overlaps `1/3`;
- shuffled-source prompt overlaps `3/3`;
- retained source-specific matched-only IDs after both controls: `0`.

Process repair is therefore killed as a source-communication method on this
surface. It remains a target-side repair baseline and a diagnostic for route
headroom.

## Key Readouts

| Artifact | SHA256 |
|---|---|
| `qwen_svamp70_zero_source_kv_process_repair_controls_summary.md` | `53371e3de25b5781fe8782427c63ecf912cf02c9e74aa7474283c4b08656e44a` |
| `qwen_svamp70_shuffled_source_prompt_process_repair_controls_summary.md` | `ed6e513cff6d47af46c24c6e5fb32c9d6c647a5332c59942ac4d0e7061c09c2f` |
| `svamp70_zero_source_kv_attribution.md` | `6d83b6b1ede1e166bd1e18629239188ffc396942f90ae1542b5268a078ebb92c` |
| `svamp70_zero_source_kv_source_control_gate.md` | `802e521354b9b0c5673f8262ea1b617b9dbf57427f031c33d78fd6d92288e837` |
| `svamp70_zero_and_shuffled_source_attribution.md` | `a7b9d3594392721b29d1db3c0036c6750aabb98209c43a7b78dc9b377e946875` |
| `svamp70_zero_and_shuffled_source_control_gate.md` | `05e7c38e73f012e47345f7430fac2e93d9177a51e6505cae62cffaefd919ca72` |
| `sha256.txt` | `d9ab6be581a1e863ac5007385d85dda47ddf0c97c24811269aa11c5e194d9b96` |

Full JSONL and metadata hashes are in:

- `results/process_repair_source_controls_20260426/sha256.txt`
