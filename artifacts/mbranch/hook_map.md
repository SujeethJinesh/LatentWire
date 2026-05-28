# Falcon-H1 M-BRANCH Hook Map

## Status

Paper readiness remains not ICLR-ready. The current paper story still needs a positive method that survives larger frozen slices, seed repeats, and strict cross-family separation. This artifact only decides whether Falcon-H1 branch-local hooks are worth promoting to a tiny sanity packet.

Decision surface read this turn:

- `RUN_LEDGER.md`: M-BRANCH is a queued CPU artifact gate and should only decide whether to request a tiny Falcon hook run.
- `DECISIONS.md`: M-BRANCH is Falcon-only and needs a hook map before any GPU sanity.
- `paper/reviewer_feedback.md`: reviewer risk is dominated by small noisy effects, so this diagnostic should not claim method progress.

Cached Falcon branch activations were not found. Existing Falcon activation packets are post-layer/post-MLP output magnitudes only.

## Implementation Facts

Primary implementation inspected:

- HuggingFace Transformers: `/workspace/LatentWire/.venv_gpu/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py`
- SGLang serving path: `/workspace/.sglang/lib/python3.12/site-packages/sglang/srt/models/falcon_h1.py`
- SGLang Mamba mixer: `/workspace/.sglang/lib/python3.12/site-packages/sglang/srt/layers/attention/mamba/mamba.py`

Falcon-H1 does not concatenate attention and Mamba outputs at the decoder layer. It computes branch-projected attention and Mamba outputs, applies branch multipliers, sums them, adds the residual, then runs the MLP.

## Branch Surfaces

| Surface | HuggingFace hook point | SGLang hook point | 0.5B native dimension | Use |
| --- | --- | --- | ---: | --- |
| `attention_pre_o_proj_native` | `FalconH1Attention.forward`: capture `attn_output` after reshape and before `self.o_proj` at lines 386-387 | `FalconH1HybridAttentionDecoderLayer.self_attention`: capture `attn_output` before `self.o_proj` at lines 315-317 | 512 | Attention branch before projection into residual width. |
| `attention_projected_presum` | Capture `attention_hidden_states` after `self.self_attn(...)` and `attn_out_multiplier`, before line 1125 sum | Capture `attention_hidden_states` after multiplier at lines 334-339 | 1024 | Best comparable attention branch surface. |
| `mamba_pre_out_proj_native` | Register a `forward_pre_hook` on `model.layers.{i}.mamba.out_proj`; this captures `scan_output` before `out_proj` at lines 978-986 on the torch path and equivalent pre-proj tensors at lines 659-666 or 768-773 on the fast path | Register a `forward_pre_hook` on `model.layers.{i}.mamba.out_proj`; captures normalized/gated SSM output before projection at SGLang Mamba lines 690-693 | 1536 | Mamba branch before projection into residual width. |
| `mamba_projected_presum` | Capture `mamba_hidden_states` after `self.mamba(...)` and `ssm_out_multiplier`, before line 1125 sum | Capture `mamba_hidden_states` after SGLang Mamba backend and multiplier at lines 345-353 | 1024 | Best comparable Mamba branch surface. |
| `post_mixer_pre_residual` | Capture `hidden_states = mamba_hidden_states + attention_hidden_states` at line 1125 before residual add at line 1128 | Capture `hidden_states = attention_hidden_states + mamba_hidden_states` at line 355 before MLP preparation | 1024 | True post-mixer branch-sum surface. |
| `post_layer_output_cached` | Existing layer forward hook captures returned output after residual plus MLP at lines 1131-1136 | Existing serving traces do not expose this exact module-return hook | 1024 | Current cached drift baseline only. |
| `mamba_gate_optional` | Capture `gate` from Mamba projection split at lines 793-795, or precomputed fast path line 617 | Capture `gate` from Mamba split at lines 416-424 | 1536 | Optional gate diagnostic, not a primary branch drift surface. |
| `mlp_gate_optional` | Capture `gate_proj(x) * gate_multiplier`, `up_proj(x)`, and the product before `down_proj` at lines 1019-1021 | Capture `gate_up` after `gate_up_proj` and before `SiluAndMul` at lines 91-95 | 2048 | Optional FFN/gate surface after the mixer. |

## Minimal Hook Plan

No hook run was executed in this task. If Falcon M-BRANCH is reopened, use a hook-only packet before any quantized scoring:

1. Capture only positions `[100, 10000]`, traces `0-1`, and layers `[0, 18, 35]`.
2. Capture last generated token absolute magnitudes for `attention_projected_presum`, `mamba_projected_presum`, `post_mixer_pre_residual`, and `post_layer_output_cached`.
3. Add native-dimension diagnostics for `attention_pre_o_proj_native` and `mamba_pre_out_proj_native` only if the comparable projected surfaces look lower.
4. Promote to a full 12-trace hook packet only if either projected branch has strict set-leaving at least `0.15` absolute below the cached post-layer mean and the layer-ordering is stable across the sampled layers.
5. Keep this diagnostic separate from quantization scoring; it should answer only whether branch-local drift is meaningfully cleaner than the current post-layer drift.
