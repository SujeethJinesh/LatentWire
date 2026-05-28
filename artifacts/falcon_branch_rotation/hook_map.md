# Falcon-H1 Branch Hook Map

Model: `tiiuae/Falcon-H1-0.5B-Instruct`  
Snapshot: `8f2587ca06bff78d8fa1adfccbe8c24d5f86b368`

Implementation references inspected from prior artifacts:

- `.venv_gpu/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py`
- `/workspace/.sglang/lib/python3.12/site-packages/sglang/srt/models/falcon_h1.py`
- `/workspace/.sglang/lib/python3.12/site-packages/sglang/srt/layers/attention/mamba/mamba.py`

Falcon-H1 computes projected attention and Mamba branch outputs, applies branch multipliers, sums the branches, adds the residual, and then runs the MLP. It does not expose branch outputs as separate named modules, so projected branch captures need layer-forward instrumentation rather than a plain module hook.

| Surface | Capture point | Hook complexity | Cached? | Notes |
|---|---|---:|---|---|
| `attention_pre_o_proj_native` | `model.layers.{i}.self_attn.o_proj` forward pre-hook input | medium | no | Native attention-width surface; less comparable to residual width. |
| `attention_projected_presum` | local `attention_hidden_states` after attention projection and multiplier, before branch sum | medium | no | Primary attention branch surface. |
| `mamba_pre_out_proj_native` | `model.layers.{i}.mamba.out_proj` forward pre-hook input | medium/high | no | Native Mamba-width surface; useful only after projected surfaces. |
| `mamba_projected_presum` | local `mamba_hidden_states` after Mamba projection and multiplier, before branch sum | medium | no | Primary Mamba branch surface. |
| `post_mixer_pre_residual` | local branch sum before residual add | medium | no | Same-run control separating branch sum from residual/MLP. |
| `post_block_output` | `model.layers.{i}` forward hook output | low | yes | Existing high-drift reference only. |
| `mlp_gate_optional` | MLP gate/up product before down projection | medium/high | no | Optional confound check; not branch-local. |

Recommended tiny diagnostic:

1. Use Falcon smoke traces `[7, 1, 11]`; minimum diagnostic can start with `[7, 1]`.
2. Capture positions `[100, 10000]`.
3. Capture layers `[0, 18, 35]` first; full smoke can capture all 36 layers.
4. Record strict set-leaving and rotated covariance range for the four primary surfaces.
5. Promote only if one branch is materially cleaner than post-block.

