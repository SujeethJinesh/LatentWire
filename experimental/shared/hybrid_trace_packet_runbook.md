# Hybrid Trace Packet Runbook

This runbook defines the first real Mac/GPU-admissible trace packet for
SSQ-LR, HORN, and HBSM. Synthetic packets validate script mechanics only.

## Required Packet Shape

Every real packet must contain:

- `config.json`: model id, model revision/hash, tokenizer revision, prompt
  source, prompt ids/hash, seed list, context lengths, dtype, device, and exact
  command. It must also include `architecture_map_hash` from
  `shared/results/hybrid_architecture_maps_20260506/architecture_maps.json`.
- `raw_rows.jsonl`: one JSON row per layer/boundary/state measurement.
- `summary.json`: aggregate readouts, decision, claim boundary, and row count.
- `summary.md`: human-readable table and interpretation.
- `decision.md`: pass/fail/continue decision tied to the preregistered gate.
- optional `tensors/`: state or activation tensor packet written with
  `activation_dumper.py`.

Validate any packet with:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/<project>/phase2/results/<packet_dir>
```

For real packets, use the stricter project contract:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/<project>/phase2/results/<packet_dir> \
  --mode real --project <ssq_lr|horn|hbsm>
```

The checker rejects real packets without provenance fields, `summary.md`,
matching `row_count`, project-specific row schemas, required controls, and
project-specific admissible coverage.

## SSQ-LR Real S1 Packet

Minimum admissible row fields:

- `model_id`, `model_revision`, `prompt_id`, `layer`, `position_bucket`
- `state_shape`, `max_abs`, `rms`, `std`, `kurtosis`, `outlier_mass`
- `control_type`, including `bf16_no_quant`

Required controls:

- BF16/no-quant state recording.
- At least the preregistered `prefill_end`, `2k_or_end`, `8k_or_end`, and `final_minus_128` position buckets.
- At least 12 fixed prompts or an explicit resource-limit note.

## HORN Real H1 Packet

Minimum admissible row fields:

- `model_id`, `layer_left`, `layer_right`, `direction`, `boundary_index`
- `prompt_id`, `pre_norm_position`, `post_norm_position`, `max_abs`, `rms`,
  `kurtosis`
- `control_type` in `boundary`, `non_boundary`, or `permuted_direction`

Required controls:

- non-boundary adjacent pairs;
- direction-label permutation matched to an observed boundary tuple;
- at least one `attention->ssm` boundary and one `ssm->attention` boundary;
- matched normalization placement.
- at least 12 fixed prompts or an explicit resource-limit note.

## HBSM Real B1 Packet

Minimum admissible row fields:

- `model_id`, `layer`, `boundary_flag`, `precision_perturbation`
- `kl_or_nll_drift`, `cheap_predictor`, `parameter_count`, `weight_norm`
- `top_decile_flag`, `random_top_decile`, `train_test_split`
- `control_type` in `perturbation_off`, `random_flags`, `layer_index`,
  `parameter_count_norm`, or `boundary_only`

Required controls:

- perturbation-off/no-op row with near-zero drift;
- random top-decile flags;
- layer-index baseline;
- parameter-count/norm baseline;
- boundary-only baseline;
- both `boundary_flag=true` and `boundary_flag=false`;
- matched counts for `top_decile_flag=true` and `random_top_decile=true`;
- both `train` and `test` split rows, unless a resource-limit note is present;
- train/test layer split if layer count permits.

## Promotion Boundary

A real packet can promote only the next gate, never the full paper. Native GPU
speed, HBM, latency, and production-packing claims require separate NVIDIA
validation.
