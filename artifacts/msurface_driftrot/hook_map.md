# M-SURFACE DriftRot Hook Map

## Granite-4.0-H-Small

Model snapshot: `b8c0982bab7fde4eb48110f5a069527c008fab39`

| Surface | Hook point | Layers | Cached? | Complexity | Decision use |
|---|---|---:|---|---:|---|
| `ssm_input_x` | local tensor after Mamba split in `GraniteMoeHybridMambaLayer.torch_forward` | 36 Mamba | no | 2-3 h | Wrapper-only follow-up. |
| `ssm_BC_parameter_generation` | local `B` and `C` after split in Mamba forward | 36 Mamba | no | 2-4 h | Noisy small-width mechanism readout. |
| `mamba_out_projection_input` | `model.layers.{i}.mamba.out_proj` forward pre-hook input | 36 Mamba | no | 0.5-1 h | Primary cheap diagnostic. |
| `attention_out_projection_input` | `model.layers.{i}.self_attn.o_proj` forward pre-hook input | 4 attention | no | 0.5-1 h | Branch contrast/control. |
| `post_block_output` | `model.layers.{i}` forward hook output | 40 | yes | 0-0.5 h | Same-run control; cached reference `0.566`. |

## Falcon-H1-0.5B-Instruct

Model snapshot: `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368`

| Surface | Hook point | Layers | Cached? | Complexity | Decision use |
|---|---|---:|---|---:|---|
| `ssm_input_x` | local tensor after Falcon Mamba split | 36 | no | 2-3 h | Defer until Granite positive. |
| `ssm_BC_parameter_generation` | local `B` and `C` after split | 36 | no | 2-4 h | Defer; noisy. |
| `mamba_out_projection_input` | `model.layers.{i}.mamba.out_proj` forward pre-hook input | 36 | no | 0.5-1 h | Clean module hook if Falcon diagnostic is gated. |
| `attention_out_projection_input` | `model.layers.{i}.self_attn.o_proj` forward pre-hook input | 36 | no | 0.5-1 h | Clean branch contrast. |
| `post_block_output` | `model.layers.{i}` forward hook output | 36 | yes | 0-0.5 h | Same-run control; cached reference `0.674`. |

Recommended first run: Granite, prompt indices `[0, 1]`, positions `[100, 20000]`, surfaces `mamba_out_projection_input`, `attention_out_projection_input`, `post_block_output`.

