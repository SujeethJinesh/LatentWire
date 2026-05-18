# Phase 9 Per-Component Dissection of the Quamba2 Contradiction

Generated: 2026-05-18

This is a no-GPU analysis over existing activation packets. Its purpose is to
check whether the Quamba2 contradiction is visible specifically in SSM/Mamba
components, or only after averaging over complete hybrid blocks.

## Scope and Identifiability

The existing OutlierMigrate activation packets record one activation-magnitude
row per model block output:

- `layer_index`
- `layer_name`
- `decode_position`
- `prompt_index`
- `channel_magnitudes`

They do **not** record separate Q, K, V, attention-output, MLP-intermediate,
MLP-output, or pre-sum SSM pathway tensors. Therefore this analysis cannot
honestly produce a true QKV / attention-output / MLP-intermediate /
MLP-output decomposition from existing packets.

The strongest available dissection is layer-type stratification:

- Granite-4.0-H-Small: block outputs classified by `config.layer_types`
  into `attention` and `mamba`.
- Nemotron-3-Nano: block outputs classified by `config.hybrid_override_pattern`
  into `attention`, `mamba`, and `moe_expert`.
- Falcon-H1: only post-sum residual block outputs are available. The Phase 7
  packet explicitly records that separate pre-sum attention and SSM pathway
  hooks were not available without source modification.
- DeepSeek-R1-Distill-Qwen-1.5B: pure-Transformer post-block residual output;
  no SSM component exists.

This means the result below can support or weaken the Quamba2 headline, but it
does not fully settle the component-level question. A true component-level
test requires future packets that hook SSM output, attention QKV/output, and
MLP intermediate/output tensors separately.

## Method

This report reuses the deterministic layer-level readout from
`experimental/outlier_migrate/phase9/post_m18_analysis/per_layer_dissection.md`.
For each layer/block, it computes strict set-leaving from prompt-averaged
activation magnitudes:

1. Average per-channel magnitude at decode position 100 over the packet's
   prompts.
2. Select the top-1% channels at position 100.
3. Average per-channel magnitude at the packet's final measured position
   (20,000 for Phase 1/2/5'/7).
4. Count the fraction of position-100 top-1% channels whose final rank is
   outside the final top-1% set.

This deterministic layer-level readout is not the same statistic as the
trace-bootstrap packet gate. The gate remains the authoritative result; this
analysis is a mechanistic stratification.

## Results

| Model / packet | Available stratum | Layer count | Mean strict set-leaving | Median strict set-leaving | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Granite-4.0-H-Small Phase 1 | `attention` block output | 4 | 0.341463414634 | 0.353658536585 | Attention-class blocks drift. |
| Granite-4.0-H-Small Phase 1 | `mamba` block output | 36 | 0.335365853659 | 0.329268292683 | SSM/Mamba-class blocks drift at nearly the same rate as attention-class blocks. |
| Nemotron-3-Nano Phase 2 | `attention` block output | 6 | 0.358024691358 | 0.370370370370 | Attention-class blocks drift. |
| Nemotron-3-Nano Phase 2 | `mamba` block output | 23 | 0.342995169082 | 0.370370370370 | SSM/Mamba-class blocks drift. |
| Nemotron-3-Nano Phase 2 | `moe_expert` block output | 23 | 0.338164251208 | 0.333333333333 | MoE-class blocks drift at a comparable rate. |
| Falcon-H1 Phase 7 | `post_sum_residual` block output | 36 | 0.320707070707 | 0.318181818182 | Drift appears in the post-sum residual, but pre-sum pathway attribution is unavailable. |
| DeepSeek-R1-Distill-Qwen-1.5B Phase 5' | Transformer post-block output | 28 | 0.361607142857 | 0.375000000000 | Pure-Transformer control drifts without any SSM component. |

## Quamba2 Readout

The component-level conclusion is conditional:

- **What the existing packets support:** SSM/Mamba-class block outputs drift
  in both measured hybrid Mamba-family systems where layer-type classification
  is available. Granite-Small `mamba` blocks have mean strict set-leaving
  `0.335365853659`, and Nemotron `mamba` blocks have mean strict set-leaving
  `0.342995169082`. These values are close to their attention/MoE neighbors,
  so the measured drift is not isolated to attention-only blocks.
- **What they do not support:** The packets do not prove that the internal SSM
  state tensor, SSM selective-scan output, attention QKV tensors, or MLP
  intermediate tensors drift at those rates. The data are post-block activation
  outputs.
- **Best current wording for the paper:** "At the block-output level, the
  persistence assumption is not borne out in the measured long-decode W4A16
  packets; SSM/Mamba-class blocks drift at rates comparable to attention-class
  blocks. A future component-hook packet is needed to isolate SSM pathway
  state from attention and MLP contributions."

## Implication

The headline Quamba2 contrast is strengthened but should remain scoped.
Existing evidence supports a practical contradiction for static
protected-channel deployment in hybrid models: the block-output channel set
does not remain persistent at reasoning-scale horizons. It does not yet
support a stronger biological claim that every internal SSM tensor violates
Quamba2's persistence assumption.

## Artifact References

- Granite-Small Phase 1 packet:
  `experimental/outlier_migrate/phase1/results/om_phase1_20260508T014959Z`
- Nemotron-3-Nano Phase 2 packet:
  `experimental/outlier_migrate/phase2/results/om_phase2_nemotron3_20260508T231723Z`
- Falcon-H1 Phase 7 packet:
  `experimental/outlier_migrate/phase7/results/om_phase7_falcon_h1_20260512T223600Z`
- DeepSeek Phase 5' packet:
  `experimental/outlier_migrate/phase5_prime/results/om_phase5p_20260512T053800Z`
- Prior per-layer report:
  `experimental/outlier_migrate/phase9/post_m18_analysis/per_layer_dissection.md`
