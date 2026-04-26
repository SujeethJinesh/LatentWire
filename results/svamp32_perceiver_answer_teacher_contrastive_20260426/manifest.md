# SVAMP32 Perceiver Answer-Teacher Contrastive Manifest

- date: `2026-04-26`
- scale-up rung: strict small teacher-forced pre-gate
- status: `fails_pre_generation_gate`
- source model: `Qwen/Qwen2.5-0.5B-Instruct`
- target model: `Qwen/Qwen3-0.6B`
- eval file: `results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl`
- target set: `results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json`

## Checkpoint

- path: `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/checkpoints/qwen25_to_qwen3_svamp32_perceiver_answer_teacher_w080_ctrl050_r16_b16_seed1.pt`
- size: `1.8G`
- sha256: `65aea1fc6db7e96d5a0df5e3d98380fe44549a3a2eb35dff4bc7c09a1d89a485`
- tracked: no, checkpoint is too large for git
- calibration log: `.debug/svamp32_perceiver_answer_teacher_contrastive_20260426/logs/calibrate_w080_ctrl050_seed1.log`
- calibration log sha256:
  `dcee16b600a8918fc7fadcd433baf27dabe8ad9ef89586146daaeb6bce737101`

## Calibration

Key settings:

- correction: `bridge_ridge_qk_dynalign_query_innovation_resampler_replace`
- connector mode: `perceiver_queries`
- rank: `16`
- bridge bank size: `16`
- answer-teacher weight: `0.8`
- target-self preserve weight: `16`
- source-control weight: `0.5`
- source-control mode: `zero_and_shuffle`
- conditional delta memory: enabled
- value loss weight: `0.0`

Calibration readout:

- prompts: `32`
- dynamic mixture samples: `1411`
- answer-teacher injected prompts: `6`
- answer-teacher injected samples: `277`
- average K alignment cosine: `0.951`
- average V alignment cosine: `0.734`

## Teacher-Forced Gates

| Gate | Status | Matched Positive Clean | Matched-Only Clean | Control Leak Clean | Mean Matched-Control Delta | JSON SHA256 |
|---:|---|---:|---:|---:|---:|---|
| `0.125` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -1.1011 | `db0641fb41a2e49106fd7a63c72b2c09f97d3946969c03df3598c201cb49435f` |
| `0.150` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -1.1543 | `7ce7a255e8f847c43caccfd98ed5f37131a515e023c6660d93b14b1b485f82c5` |
| `0.200` | `no_teacher_forced_source_signal` | 2/6 | 0/6 | 2/6 | -1.2968 | `9aed1c804d854e4193281b0e24df71407f465dd1fc647c044e79a8f0db6a8802` |

Markdown hashes:

- `teacher_forced_gate0125.md`:
  `92e5ef9d38fb11d3f96e785f12d24febbe6bceec57834cb12a49f95603be290c`
- `teacher_forced_gate015.md`:
  `1344fca2f5c1a649203541c82d497338acfd439048cbfee67319c0583baae9df`
- `teacher_forced_gate020.md`:
  `558fa598de3c392747dcad19c610bcc78ce720ae7a6b25b022224eaeeba0ecf3`

## Decision

Do not run generation for this checkpoint. It fails the teacher-forced pre-gate:
no fixed gate tested produces matched-only positive margins on clean residual
IDs. The two positive clean IDs are explained by shuffled-source, target-only,
or slots-only controls.

This weakens the current Perceiver answer-teacher plus contrastive
delta-memory variant. The next branch should either move to a fresh surface
with more measured clean source-only headroom, or change the objective so
matched-source information is explicitly separated from target-only memory
before answer-teacher supervision is applied.
