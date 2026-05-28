# M-SURFACE Hook Map

Refresh timestamp: `2026-05-28T06:26:48Z`.

Status: CPU/static inspection only. No GPU jobs were run. Cached block-output
activation summaries exist for Granite and Falcon, but no cached internal
activation packet was found for SSM input, SSM B/C, Mamba `out_proj` input, or
attention `o_proj` input.

## Evidence Read

- `RUN_LEDGER.md`: M-SURFACE remains deferred; tiny Granite sanity is allowed
  only if a finalist needs placement evidence.
- `DECISIONS.md`: M-SURFACE is hookable but must not be promoted without
  lower-drift hook evidence.
- `paper/reviewer_feedback.md`: structural claims need larger slices, controls,
  and telemetry; this artifact is diagnostic only.
- `release/results/set_leaving/set_leaving.json`: cached block-output strict
  set-leaving is `0.566234756098` for Granite-Small and `0.673611111111` for
  Falcon.

## Cached Activations

| Model | Cached activation exists | What it contains | M-SURFACE usability |
|---|---|---|---|
| Granite-4.0-H-Small | yes | `activation_magnitudes.jsonl.gz` manifests record `transformer_layer_forward_output`; release strict leaving is cached | Usable only as post-block/block-output reference; internal surfaces missing |
| Falcon-H1-0.5B-Instruct | yes | `activation_magnitudes.jsonl.gz` and `activation_means.npz` manifests record `transformer_layer_forward_output`; release strict leaving is cached | Usable only as post-block/block-output reference; internal surfaces missing |

## Granite-4.0-H-Small

Source config and code:

- `/workspace/hf_cache/hub/models--ibm-granite--granite-4.0-h-small/snapshots/b8c0982bab7fde4eb48110f5a069527c008fab39/config.json`
- `experimental/hybridkernel/phase0/configs/ibm-granite-4.0-h-small.config.json`
- `.venv_gpu/lib/python3.12/site-packages/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py`

Architecture facts:

- Top-level class: `GraniteMoeHybridForCausalLM`; named module prefix under the
  causal LM is `model.layers.{i}`.
- Layers: 40 total. Attention layers: `5, 15, 25, 35`. Mamba layers:
  `0-4, 6-14, 16-24, 26-34, 36-39`.
- Hidden size `4096`; Mamba intermediate size `8192`; B/C concat dimension
  `2 * mamba_n_groups * mamba_d_state = 256`.

| Surface | Exact hook point | Layer scope | Cached? | Hook complexity | Risks |
|---|---|---:|---|---:|---|
| SSM input x | Local `hidden_states` after `hidden_states_B_C = apply_mask_to_padding_states(...)` and `hidden_states, B, C = torch.split(...)` inside `GraniteMoeHybridMambaLayer.torch_forward`; not a named submodule | 36 Mamba layers | no | 2-3 h | Requires temporary wrapper/patch of Mamba forward and fast-path disablement; local tensor is invisible to normal hooks |
| SSM B/C generation | Local `B` and `C` from the same split inside `GraniteMoeHybridMambaLayer.torch_forward`, before `repeat_interleave` / head repetition | 36 Mamba layers | no | 2-4 h | Same wrapper risk; 256-dim surface makes top-1% only about 2-3 channels, so estimates are noisy |
| Mamba out_proj input | Named module `model.layers.{i}.mamba.out_proj`; register `forward_pre_hook` and capture input tensor 0 | 36 Mamba layers | no | 0.5-1 h | Must force unfused PyTorch path so the module call is exercised; best first internal surface |
| Attention o_proj input | Named module `model.layers.{i}.self_attn.o_proj`; register `forward_pre_hook` and capture input tensor 0 | layers `5,15,25,35` | no | 0.5-1 h | Sparse attention-layer coverage; useful as branch contrast, not enough alone |
| Post-block residual/block output | Named module `model.layers.{i}`; forward hook output tuple item 0, or `output_hidden_states=True` | all 40 layers | yes | 0-0.5 h | Cached reference exists; same-run control still needed if internal hook run changes decode path |

## Falcon-H1-0.5B-Instruct

Source config and code:

- `/workspace/hf_cache/hub/models--tiiuae--Falcon-H1-0.5B-Instruct/snapshots/8f2587ca06bff78d8fa1adfccbe8c24d5f86b368/config.json`
- `experimental/outlier_migrate/phase9/results/om_stage1_e1_deepseek_falcon_20260526T2335Z/falcon_h1_0_5b/model_provenance.json`
- `.venv_gpu/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py`

Architecture facts:

- Top-level class: `FalconH1ForCausalLM`; named module prefix under the causal
  LM is `model.layers.{i}`.
- The current local HF snapshot is the authority for hook mapping:
  `num_hidden_layers=36`, `hidden_size=1024`, `mamba_d_ssm=1536`,
  `mamba_n_heads=24`, `mamba_d_state=128`, `mamba_n_groups=1`.
- `release/configs/falcon_h1.yaml` says 32 layers / hidden 2048, but this is
  stale for the cached `tiiuae/Falcon-H1-0.5B-Instruct` artifacts. Use the
  snapshot config and manifests for diagnostics.
- Falcon-H1 runs Mamba and attention in parallel in every decoder block.

| Surface | Exact hook point | Layer scope | Cached? | Hook complexity | Risks |
|---|---|---:|---|---:|---|
| SSM input x | Local `hidden_states` after `hidden_states_B_C = apply_mask_to_padding_states(...)` and `hidden_states, B, C = torch.split(...)` inside `FalconH1Mixer.torch_forward`; not a named submodule | all 36 layers | no | 2-3 h | Requires temporary wrapper/patch and fast-path disablement; local tensor is invisible to normal hooks |
| SSM B/C generation | Local `B` and `C` from the same split inside `FalconH1Mixer.torch_forward`, before `repeat_interleave` / head repetition | all 36 layers | no | 2-4 h | Same wrapper risk; 256-dim surface makes top-1% unstable |
| Mamba out_proj input | Named module `model.layers.{i}.mamba.out_proj`; register `forward_pre_hook` and capture input tensor 0 | all 36 layers | no | 0.5-1 h | Must force unfused PyTorch path; cleanest Falcon internal Mamba surface |
| Attention o_proj input | Named module `model.layers.{i}.self_attn.o_proj`; register `forward_pre_hook` and capture input tensor 0 | all 36 layers | no | 0.5-1 h | Clean module hook; parallel branch means attention/Mamba separation is interpretable |
| Post-block residual/block output | Named module `model.layers.{i}`; forward hook output tuple item 0, or `output_hidden_states=True` | all 36 layers | yes | 0-0.5 h | Cached reference exists; same-run control required if Falcon follow-up is run |

## Diagnostic Recommendation

Recommend a tiny GPU diagnostic, but only for Granite first and only as a
surface-placement sanity check. The clean module hooks can answer the gated
question cheaply: whether `mamba_out_projection_input` has strict leaving
`<0.30` while same-run post-block output remains in the high-drift regime.

Run `mamba_out_projection_input`, `attention_o_proj_input`, and post-block
control first. Add SSM input and B/C only if the GPU operator can afford the
temporary wrapper; do not let the wrapper block the cheap module-hook result.

Promotion rule: promote M-SURFACE only if an internal Granite surface has strict
set-leaving `<0.30` or at least `0.15` absolute below same-run post-block
leaving. If Granite does not show that, skip Falcon.
