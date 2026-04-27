# SVAMP32 Source Sampling New Clean2 S16 Replay

- date: `2026-04-27`
- status: `prompt_wrapper_control_explains_surface`
- scale rung: `smoke`
- eval subset: `newclean2_eval.jsonl`
- clean IDs:
  - `6e9745b37ab6fc45`
  - `de1bf4d142544e5b`

## Results

| Condition | Model | Prompt Mode | Oracle |
|---|---|---|---:|
| `source_sample` | `Qwen/Qwen2.5-Math-1.5B` | `source_reasoning` | `2/2` |
| `target_direct_sample` | `Qwen/Qwen3-0.6B` | `direct` | `0/2` |
| `target_brief_sample` | `Qwen/Qwen3-0.6B` | `source_reasoning` | `2/2` |
| `source_direct_sample` | `Qwen/Qwen2.5-Math-1.5B` | `direct` | `1/2` |

## Decision

Fail as source-specific surface. The matched source replay is stable at `2/2`,
but the target model with the same brief-analysis/source-reasoning prompt also
reaches `2/2`. The original source-sampling finding is therefore explained by
prompt-wrapper candidate discovery, not a source-model signal. Do not train a
connector on this two-ID surface.

## Artifacts

- `newclean2_eval.jsonl`
- `newclean2_eval.meta.json`
- `source_samples.jsonl`
- `source_samples.md`
- `target_direct_samples.jsonl`
- `target_direct_samples.md`
- `target_brief_samples.jsonl`
- `target_brief_samples.md`
- `source_direct_samples.jsonl`
- `source_direct_samples.md`
