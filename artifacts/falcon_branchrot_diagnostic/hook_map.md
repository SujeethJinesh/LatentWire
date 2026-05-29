# Falcon-H1 M-BRANCH Hook Map

## Status

Paper readiness remains not ICLR-ready. The current story is still a gated positive-method funnel: V1 is running, HYST/LAMBDA are the only DeepSeek/Falcon smoke candidates, and M-BRANCH is only a CPU hook-feasibility diagnostic. The blocking gap for submission is unchanged: no positive method has survived larger frozen slices, seed repeats, paired uncertainty, and strict same-family vs cross-family separation.

Latest decision surface read this turn:

- `RUN_LEDGER.md`: M-BRANCH is complete/deferred; do not request Falcon GPU scoring now.
- `DECISIONS.md`: M-BRANCH is deferred because Falcon branch-local cached activations are absent and static inspection alone does not justify a GPU branch run before smoke.
- `artifacts/funnel_prefilters/`: Falcon remains alive for HYST/LAMBDA smoke only, using fixed traces `[7, 1, 11]`.
- `paper/reviewer_feedback.md`: small noisy effects remain the dominant reviewer risk; this artifact must not be used as method progress.

Cached Falcon branch activations were not found. Existing Falcon activation packets are `transformer_layer_forward_output` post-block/post-MLP magnitudes only.

## Implementation Facts

Primary implementation inspected:

- HuggingFace Transformers: `/workspace/LatentWire/.venv_gpu/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py`
- SGLang serving path: `/workspace/.sglang/lib/python3.12/site-packages/sglang/srt/models/falcon_h1.py`
- SGLang Mamba mixer: `/workspace/.sglang/lib/python3.12/site-packages/sglang/srt/layers/attention/mamba/mamba.py`

Falcon-H1 does not concatenate attention and Mamba outputs at the decoder layer. It computes projected attention and Mamba branch outputs, applies branch multipliers, sums them, adds the residual, then runs the MLP. That makes branch-local hooks feasible, but the current cached drift signal is post-block only.

## Branch Surfaces

| Surface | Hook name / location | Dimension on 0.5B | Complexity | Risk | Decision use |
| --- | --- | ---: | --- | --- | --- |
| `attention_pre_o_proj_native` | HF: `model.layers.{i}.self_attn.o_proj` `forward_pre_hook` captures `attn_output` after reshape and before `o_proj` (`modeling_falcon_h1.py:386-387`). SGLang: capture `attn_output` before `o_proj` in `FalconH1HybridAttentionDecoderLayer.self_attention` (`falcon_h1.py:315-317`). | 512 | Medium | Medium: native head-width surface differs from residual width and SGLang tensor-parallel layouts need care. | Optional native attention branch check; not directly comparable to post-block unless normalized separately. |
| `attention_projected_presum` | HF: capture `attention_hidden_states` after `self.self_attn(...)` and `attn_out_multiplier`, before branch sum (`modeling_falcon_h1.py:1112-1125`). SGLang: capture after `self_attention` and multiplier (`falcon_h1.py:334-339`). | 1024 | Medium | Low/medium: best comparable attention branch surface, but needs layer-forward instrumentation or a wrapper rather than a plain module hook. | Primary attention branch drift estimate if a hook packet is ever run. |
| `mamba_pre_out_proj_native` | HF: `model.layers.{i}.mamba.out_proj` `forward_pre_hook` captures normalized/gated scan output before projection (`modeling_falcon_h1.py:659-666`, `768-773`, `978-986`). SGLang: capture `hidden_states` before `out_proj` in `MambaMixer2.forward` (`mamba.py:690-693`). | 1536 | Medium/high | Medium/high: fused/serving paths and tensor-parallel dimensions can differ; native surface is not residual-width comparable. | Optional native Mamba branch check after projected surfaces. |
| `mamba_projected_presum` | HF: capture `mamba_hidden_states` after `self.mamba(...)` and `ssm_out_multiplier`, before branch sum (`modeling_falcon_h1.py:1104-1125`). SGLang: capture after Mamba backend output and multiplier (`falcon_h1.py:345-353`). | 1024 | Medium | Low/medium: comparable residual-width branch surface, but layer-forward instrumentation is required. | Primary Mamba branch drift estimate if a hook packet is ever run. |
| `post_mixer_pre_residual` | HF: capture `hidden_states = mamba_hidden_states + attention_hidden_states` before residual add (`modeling_falcon_h1.py:1125-1128`). SGLang: capture `attention_hidden_states + mamba_hidden_states` before MLP preparation (`falcon_h1.py:355-358`). | 1024 | Medium | Low: true branch-sum surface, but uncached today. | Separates branch-sum drift from residual/MLP drift. |
| `post_block_output_cached` | Existing runner hooks `model.layers.{i}` output after residual plus MLP (`modeling_falcon_h1.py:1131-1136`). | 1024 | Low | Low: already cached, but not branch-local. | Current broad baseline only. |
| `mamba_gate_optional` | HF: capture `gate` from projection split (`modeling_falcon_h1.py:617-618`, `793-795`). SGLang: capture `gate` from split (`mamba.py:416-424`). | 1536 | High | High: gate semantics vary across fast/torch paths and are not an output branch by themselves. | Optional mechanism readout, not a promotion surface. |
| `mlp_gate_optional` | HF: capture `gate_proj(x) * gate_multiplier`, `up_proj(x)`, and product before `down_proj` (`modeling_falcon_h1.py:1019-1021`). SGLang: capture `gate_up` before `SiluAndMul` (`falcon_h1.py:91-95`). | 2048 | Medium/high | Medium: post-mixer FFN surface, not branch-local to attention/Mamba. | Optional post-mixer confound check only. |

## Minimal Hook Plan If Reopened

No hook run was executed in this task. A Falcon M-BRANCH GPU diagnostic is not recommended under the current orchestrated funnel. If a future orchestrator explicitly reopens this branch, use a hook-only packet before any quantized scoring:

1. Capture only traces `[7, 1]` or `[7, 1, 11]`, positions `[100, 10000]`, and layers `[0, 18, 35]`.
2. Capture last generated token absolute magnitudes for `attention_projected_presum`, `mamba_projected_presum`, `post_mixer_pre_residual`, and `post_block_output_cached`.
3. Add `attention_pre_o_proj_native` and `mamba_pre_out_proj_native` only if projected branch surfaces look cleaner.
4. Promote to any full Falcon M-BRANCH run only if projected attention or projected Mamba strict set-leaving is at least `0.15` absolute below the matched post-block baseline and the direction is stable across sampled layers/traces.
5. Keep this diagnostic separate from HYST/LAMBDA smoke; it answers only whether Falcon post-block drift is branch-local or broadband.

## Current Readout

Static hook feasibility is real, but cached evidence is insufficient for branch-local drift. The latest funnel post-block evidence shows Falcon is still drifting broadly enough to support HYST/LAMBDA smoke, not M-BRANCH promotion. M-BRANCH remains likely broadband/inconclusive until branch-local hooks show a material drop below the post-block baseline.
