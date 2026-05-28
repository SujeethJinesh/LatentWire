# M-SURFACE Hook Map

Status: CPU/static inspection only. No GPU jobs were run. No cached internal
activation packets were found for the requested SSM or attention hook surfaces.

## Context Read

- `RUN_LEDGER.md`: M-SURFACE is queued and gates only a tiny Granite hook run.
- `DECISIONS.md`: M-SURFACE is diagnostic only until a hook map exists.
- `paper/reviewer_feedback.md`: current evidence is too small for claims, and
  structural claims need larger slices, controls, and telemetry.
- `release/results/set_leaving/set_leaving.json`: cached block-output strict
  set-leaving claims are available for comparison.

## Granite 4.0-H

Inspected local config:

- `experimental/hybridkernel/phase0/configs/ibm-granite-4.0-h-small.config.json`
- `experimental/hybridkernel/phase0/configs/ibm-granite-4.0-h-tiny.config.json`

Inspected implementation:

- `.venv_gpu/lib/python3.12/site-packages/transformers/models/granitemoehybrid/modeling_granitemoehybrid.py`
- Config uses 40 decoder layers with attention at layers `5, 15, 25, 35`
  and Mamba at `0-4, 6-14, 16-24, 26-34, 36-39`.
- Small: hidden size `4096`, Mamba intermediate size `8192`, B/C concat
  dimension `256`.
- Tiny: hidden size `1536`, Mamba intermediate size `3072`, B/C concat
  dimension `256`.

### Requested Surfaces

| Surface | Granite hook | Layer scope | Shape basis | Hook status |
|---|---|---:|---|---|
| SSM input x | Local variable after `hidden_states_B_C` convolution and split inside `GraniteMoeHybridMambaLayer.torch_forward` or `cuda_kernels_forward`; specifically the `hidden_states` tensor passed to scan | 36 Mamba layers | `mamba_expand * hidden_size` | Requires temporary instrumentation, not a plain module hook |
| SSM B/C generation | Same local split as SSM input x; capture `B`, `C`, and preferably `cat([B, C], dim=-1)` before head repetition | 36 Mamba layers | `2 * mamba_n_groups * mamba_d_state = 256` | Requires temporary instrumentation; low dimension makes top-1% noisy |
| Mamba output-projection input | `model.model.layers[i].mamba.out_proj` forward pre-hook | 36 Mamba layers | `mamba_expand * hidden_size` | Clean module hook in unfused PyTorch path |
| Attention output-projection input | `model.model.layers[i].self_attn.o_proj` forward pre-hook | layers `5, 15, 25, 35` | `hidden_size` | Clean module hook |
| Post-block residual/block output | decoder layer forward hook on `model.model.layers[i]`, output tuple item `0`; `output_hidden_states=True` also recovers adjacent block outputs | all 40 layers | `hidden_size` | Clean module hook; cached release drift exists |

### Granite Hook Cautions

- The installed code selects `cuda_kernels_forward` on CUDA when Mamba fast path
  is available. For a diagnostic hook run, force the Python path before model
  execution, for example by setting the module-level
  `is_fast_path_available = False` in the inspected Transformers module.
- Capturing SSM input x and B/C requires a small temporary wrapper around the
  Mamba forward body because those tensors are local variables after the
  convolution and split.
- Do not use fused vLLM kernels for this diagnostic. The hook test is about
  surface drift, not serving throughput.

## Falcon-H1

Inspected local config:

- `release/configs/falcon_h1.yaml`

Inspected implementation:

- `.venv_gpu/lib/python3.12/site-packages/transformers/models/falcon_h1/modeling_falcon_h1.py`
- Release config identifies a 32-layer parallel hybrid model with hidden size
  `2048`.
- The exact Hugging Face checkpoint config was not cached locally, but the
  installed model code is straightforward.
- Falcon-H1 runs Mamba and attention in parallel in every decoder block.

| Surface | Falcon hook | Layer scope | Hook status |
|---|---|---:|---|
| SSM input x | Local variable after `hidden_states_B_C` convolution and split inside `FalconH1Mixer.torch_forward` or `cuda_kernels_forward` | all 32 layers | Requires temporary instrumentation |
| SSM B/C generation | Same local split as SSM input x; capture B/C before head repetition | all 32 layers | Requires temporary instrumentation |
| Mamba output-projection input | `model.model.layers[i].mamba.out_proj` forward pre-hook | all 32 layers | Clean module hook in unfused PyTorch path |
| Attention output-projection input | `model.model.layers[i].self_attn.o_proj` forward pre-hook | all 32 layers | Clean module hook |
| Post-block residual/block output | decoder layer forward hook on `model.model.layers[i]`, output tuple item `0` | all 32 layers | Clean module hook; cached release drift exists |

## Minimal Hook Plan

1. Start with Granite-4.0-H-Small, not Falcon, because the queued gate is
   explicitly Granite-first and the cached block-output comparison is
   `0.566234756098`.
2. Use the fixed long-reasoning traces from the release drift packet if
   available to the GPU operator. Capture only per-channel absolute magnitudes
   at decode positions `100` and `20000`; use the full release grid
   `[100, 500, 1000, 5000, 10000, 20000]` only if it is already cheap.
3. For each trace, layer, and surface, form the top-1% channel set at position
   100. At the final position, compute strict set-leaving as the fraction of
   position-100 top channels no longer in the final top-1% set.
4. Save only summarized per-channel magnitude vectors and a table of strict
   set-leaving by surface. Do not save full token-by-layer activations.
5. Promote M-SURFACE only if at least one internal Granite surface has strict
   set-leaving below `0.30` while the same run's block-output surface remains
   near the cached high-drift regime.

If Granite does not show a materially lower internal surface, skip Falcon. If
Granite does show one, Falcon is straightforward to repeat because its clean
module hooks match Granite and only the local SSM split wrapper differs.
