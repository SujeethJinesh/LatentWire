# Experimental Shared Utilities

Shared Mac-local utilities for the relevant hybrid-quantization branches:
SSQ-LR, HORN, and HBSM.

These helpers are intentionally small and deterministic. They are not GPU
kernels and they do not support throughput, latency, HBM, or energy claims.
Use them for preregistered Mac gates only.

## Utilities

- `fp4_simulator.py`: deterministic INT/FP-style cast-and-cast-back
  quantization simulators, including FP8 E4M3/E5M2 helpers and MXFP4-style
  block scaling, plus quality-gap recovery helpers.
- `activation_dumper.py`: lightweight tensor packet save/load helpers for
  cached traces.
- `boundary_inspector.py`: layer-kind and attention/SSM boundary helpers.
- `hybrid_manifest_local_capture_runner.py`: resource-limited Granite Tiny
  SSQ-LR/HORN tensor runner. The current dedicated SSQ-LR artifact is
  `results/ssq_lr_local_bucket_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`, using
  bucket-specific 2/4/6/8-token recurrent-state replays. The current
  multilayer SSQ-LR artifact is
  `results/ssq_lr_local_multilayer_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`, with one
  prompt and four layers. Only the compact multilayer readout is tracked; the
  full tensor packet is regenerated on demand before checker replay. These
  validate SSQ-LR bucket/layer capture plumbing only; neither packet can
  promote S1.
- `ssq_lr_all_layer_scout.py`: metrics-only SSQ-LR scout over recurrent cache
  layers. Current artifact: `results/ssq_lr_all_layer_scout_20260507/`,
  decision
  `RESOURCE_LIMITED_ALL_LAYER_SCOUT_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`.
  It scanned 36 recurrent layers from one short Granite Tiny prompt and found
  only 4 local passing layers (`0`, `12`, `18`, `30`) against the 9-layer S1
  requirement. It writes compact JSON/Markdown rows, not tensor packets, and
  cannot promote S1.
- `hybrid_manifest_local_capture_runner.py --ssq-layers`: selected-layer
  tensor-provenance repeat for SSQ-LR. Current artifact:
  `results/ssq_lr_prompt_repeat_tensor_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`; it repeats
  layers `0`, `12`, `18`, and `30` across all 12 frozen prompts, writes saved
  tensors, and passes the real SSQ-LR checker. It remains non-promoting because
  those layers were selected after the all-layer scout.
- `prompts/hybrid_reasoning_s1b_holdout_12_20260507.jsonl`: held-out
  SSQ-LR S1b prompt split. Current held-out trace plan:
  `results/hybrid_trace_plan_s1b_holdout_20260507/`; current held-out tensor
  packet: `results/ssq_lr_s1b_holdout_tensor_capture_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`. It freezes
  layers `0`, `12`, and `30` as primary with layer `18` as near-miss/control,
  passes checker provenance through an explicit held-out
  `trace_plan_config_path`, and remains non-promoting.
- `ssq_lr_s2_state_replay_scout.py`: resource-limited SSQ-LR S2 continuation
  replay scout from cached recurrent states. Current artifacts:
  `results/ssq_lr_s2_state_replay_scout_20260507/` and
  `results/ssq_lr_s2_state_replay_scout_block256_20260507/`, both official S2
  failures because honest scale-byte accounting stays below the preregistered
  `4x` memory-reduction threshold. The scouts are useful information-content
  checks only; they are not quality, GPU, or throughput evidence.
- `ssq_lr_s2_state_replay_scout.py --block-size {64,256}` now includes an
  INT3 candidate with scale bytes counted. Current INT3 artifacts:
  `results/ssq_lr_s2_state_replay_scout_int3_block256_20260507/`,
  `results/ssq_lr_s2_state_replay_scout_int3_block256_12p_20260507/`, and
  `results/ssq_lr_s2_state_replay_scout_int3_block64_12p_20260507/`. The
  4-prompt block-256 scout passes the S2 contract but is explicitly
  resource-limited; both 12-prompt held-out INT3 scouts fail S2 because the
  recipe clears bytes but loses quality on at least one prompt.
- `ssq_lr_s3_transfer_prefilter.py`: freezes the layer-selective S2b recipe
  into the strict S3 schema and inventories local hybrid model-cache
  completeness before any GPU promotion. Current artifact:
  `results/ssq_lr_s3_transfer_prefilter_mixed25_layers0_30_20260507/`,
  decision `FAIL_REAL_SSQ_LR_S3_CROSS_MODEL_TRANSFER`; the packet freezes
  `mixed_int3_mxfp4_low_error_25pct` on layers `0,30`, validates cleanly, emits
  no retuned rows, and was a cache-only blocker with one complete hybrid
  transfer model (`ibm-granite/granite-4.0-h-tiny`); it is superseded by the
  local transfer rows below.
- `ssq_lr_s3_local_transfer_prefilter.py`: combines actual source and transfer
  S2 replay rows into the strict S3 schema. Current artifacts
  `results/ssq_lr_s3_local_transfer_prefilter_mixed25_granite_tiny_350m_layer0_12p_20260507/`
  and
  `results/ssq_lr_s3_local_transfer_prefilter_int3_granite_tiny_350m_layer0_12p_20260507/`
  are validator-clean failures: two models, 12 prompts per model, no retuning,
  one frozen recipe hash, and one source S2 hash, but only one model passes in
  each packet.
- `horn_h2_noise_replay_scout.py`: resource-limited HORN H2
  noisy-continuation replay from the failed Granite Tiny H1a packet. Current
  artifact: `results/horn_h2_noise_replay_scout_20260507/`, decision
  `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`; it passes the H2 follow-up
  contract with complete paired units and hook-off max delta `0.0`, but the
  directional drift ratio is only `1.037`, so this demotes HORN as a
  standalone branch instead of promoting H2.
- `hybrid_architecture_maps.py`: explicit config-derived boundary maps used to
  validate real trace packet provenance.
- `hybrid_model_eligibility.py`: metadata-only Hugging Face size/cache
  preflight for the live hybrid targets.
- `hybrid_trace_plan.py`: deterministic SSQ-LR/HORN/HBSM trace-collection
  plan from the frozen prompt manifest and config-derived architecture maps.
- `hybrid_trace_packet_builder.py`: converts future saved trace tensor packets
  into strict SSQ-LR/HORN real gate packets and converts HBSM sensitivity rows
  into strict real B1 packets. Built SSQ-LR/HORN packets copy the tensor
  manifest and `.pt` files into `tensors/`; built HBSM packets copy the source
  sensitivity row packet into `evidence/` with a SHA-256 manifest.
- `hbsm_local_sensitivity_runner.py`: resource-limited Granite Tiny HBSM B1
  runner that fills the row-packet template from local forward perturbation
  replays. Current artifact:
  `results/hbsm_local_sensitivity_20260507/`, decision
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_B1_SENSITIVITY_HETEROGENEITY`.
  This validates the HBSM row/provenance path only; it is one short prompt and
  cannot promote B1.
  The prompt-repeat scout
  `results/hbsm_prompt2_sensitivity_20260507/` uses the same runner with
  `--prompt-limit 2`; it is checker-valid and fails B1 with Fisher p-value
  `1.0`, boundary top-decile count `0`, and cheap-predictor Spearman `-0.667`.
  This weakens HBSM and is still not promotable evidence.
- `hybrid_gate_evaluators.py`: recomputes S1/H1/B1 pass/fail summaries from
  packet rows so real packets cannot promote from hand-written aggregate labels.
  The active evaluators use prompt-level S1 lower bounds, H1 non-boundary and
  permuted-direction controls through matched boundary labels, and HBSM
  measured top-decile scoring derived from primary-row drift rather than
  caller-supplied labels.
- `sensitivity_metrics.py`: quality, drift, and rank-correlation metrics.
- `check_gate_packet.py`: packet validator for synthetic and real Mac-local
  gate results, with stricter `--mode real --project ...` contracts and an
  explicit non-promoting schema-rehearsal path for checker-path tests. For
  non-rehearsal SSQ-LR/HORN packets it reloads saved tensors and recomputes
  row metrics from the bytes; for HBSM it verifies the copied sensitivity row
  packet hash before interpreting B1 rows.
- `hybrid_trace_packet_runbook.md`: required real-packet schema for SSQ-LR,
  HORN, and HBSM.
- `prompts/hybrid_reasoning_smoke_12_20260506.jsonl`: frozen 12-prompt
  smoke surface for the first real Mac packets. SHA-256:
  `48e68434371a648c3984e85a7207d71d2ac68617c640b37da04bd1aaeea45fe0`.

## Local Test

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

## Architecture Map Artifact

The current config-only map is:

```text
experimental/shared/results/hybrid_architecture_maps_20260506/
```

Regenerate it with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_architecture_maps
```

## Model Eligibility Artifact

The current metadata-only model preflight is:

```text
experimental/shared/results/hybrid_model_eligibility_20260506/
```

Regenerate it without downloading weights:

```bash
HF_HOME="$PWD/.debug/hf_home" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_model_eligibility
```

## Real Trace Packet Builder

Before dumping tensors, generate the exact row plan:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_plan
```

Current output:

```text
experimental/shared/results/hybrid_trace_plan_20260507/
```

This writes `ssq_lr_trace_plan.jsonl`, `horn_trace_plan.jsonl`, and
`hbsm_trace_plan.jsonl`. These files are trace-capture checklists only; they do
not contain activations, SSM state, quality metrics, or GPU evidence.

Then generate fill-in capture manifests:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_capture_manifest
```

Current output:

```text
experimental/shared/results/hybrid_capture_manifests_20260507/
```

Fill every `TO_FILL_BEFORE_CAPTURE` field before packet building. Builders
canonicalize registered served/HF model IDs to the shared architecture-map
`model_id` while preserving `served_model_id` for audit. HORN
`permuted_direction` rows use `tensor_alias_of` to reuse the observed boundary
tensor rather than requiring a duplicate tensor capture.

After a model run writes tensors with `activation_dumper.py`, build strict
project packets with:

```bash
./venv_arm64/bin/python -m experimental.shared.hybrid_trace_packet_builder \
  --project ssq_lr \
  --tensor-packet experimental/shared/results/<tensor_packet> \
  --output-dir experimental/ssq_lr/phase2/results/ssq_lr_gate_s1_<date>_<model>

./venv_arm64/bin/python -m experimental.shared.hybrid_trace_packet_builder \
  --project horn \
  --tensor-packet experimental/shared/results/<tensor_packet> \
  --output-dir experimental/horn/phase2/results/horn_gate_h1_<date>_<model>

./venv_arm64/bin/python -m experimental.shared.hybrid_trace_packet_builder \
  --project hbsm \
  --row-packet experimental/shared/results/<hbsm_rows>.json \
  --output-dir experimental/hbsm/phase2/results/hbsm_gate_b1_<date>_<model>
```

Then validate with `check_gate_packet.py --mode real --project ...`. The real
checker enforces admissible coverage, not just schema shape: SSQ-LR needs all
preregistered S1 buckets for every prompt/layer pair, HORN needs both boundary
directions plus both-direction non-boundary and prompt-paired flipped controls,
and HBSM needs `boundary_only` primary rows with prompt-level boundary/non-
boundary coverage, aggregated scoring-layer top-decile cardinality, a
non-enriched random baseline, and perturbation-off rows with near-zero drift.
For HORN, `matched_boundary_direction` lets non-boundary controls keep true
architecture directions while still pairing against both boundary directions on
every prompt; permuted controls must flip the actual `direction` label.
For HBSM, supplied `top_decile_flag` values must match the measured
`kl_or_nll_drift` top-decile ranking after aggregation on every primary prompt
row.
Real packets also need 64-hex `prompt_ids_hash` values, a `trace_plan_hash`
for the project trace-plan JSONL used during capture, and an
`architecture_map_hash` that matches the claimed `model_id` in
`shared/results/hybrid_architecture_maps_20260506/architecture_maps.json`; a
random hash-shaped value is rejected. The `model_revision` and
`tokenizer_revision` fields must match the registered Hugging Face snapshot SHA
in `shared/results/hybrid_model_eligibility_20260506/raw_rows.jsonl`; arbitrary
revision strings are rejected. Non-rehearsal real packets must also cite
`trace_plan_path`; the checker rejects rows outside that frozen trace-plan row
set instead of allowing uncited row coverage. For a promotable packet, the
file cited by `trace_plan_path` must hash to the registered project
`trace_plan_hash`. Alternate held-out trace-plan registries must be cited
explicitly through `trace_plan_config_path`; caller-created subset plans are
accepted only when `resource_limit_note` makes the packet explicitly
non-promotable. Real packets
need project-specific aggregate `summary.json` fields and a decision equal to
the recomputed S1/H1/B1 gate status, or a non-promotable decision whenever
`resource_limit_note` is present. The checker recomputes the active S1/H1/B1
gate summaries from rows and rejects stale or fabricated summary/decision
fields. The builder now also prefixes resource-limited packet decisions with
`RESOURCE_LIMITED_NOT_PROMOTABLE_`, so small Mac smoke packets can validate
hooks without accidentally promoting a gate.

## Claim Boundary

Passing these tests only means the local utilities are deterministic and
internally consistent. Any promoted paper claim still requires the relevant
project gate to pass.

## Tensor Provenance

`activation_dumper.py` writes `tensor_manifest.json` beside every tensor packet.
The manifest records each tensor's original hook name, packet-safe storage name,
SHA-256 digest, dtype, shape, and element count, and it rejects hook names that
would collide after path/space normalization. SSQ-LR and HORN packet rows carry
that manifest provenance through `tensor_name`, `tensor_source_name`,
`tensor_storage_name`, `tensor_sha256`, `tensor_dtype`, and `tensor_shape`.
HORN `permuted_direction` rows must additionally record `tensor_alias_of` and
reuse the observed boundary tensor hash.
For non-rehearsal SSQ-LR/HORN packets, `hybrid_trace_packet_builder.py` copies
the tensor manifest and `.pt` tensor files into the packet's `tensors/`
directory. The real checker reloads those tensors and rejects rows whose
reported max-abs, RMS, standard deviation, kurtosis, or outlier-mass metrics do
not match the saved bytes.

## HBSM Sensitivity Provenance

HBSM B1 consumes forward-sensitivity rows rather than raw state/boundary
tensors. Non-rehearsal HBSM packets therefore include
`evidence/hbsm_row_packet.json`, `evidence/source_manifest.json`, and
`config.json` field `source_row_packet_sha256`. The checker verifies the copied
source packet's SHA-256 before interpreting B1 rows, so sensitivity tables
remain tied to a reviewable source artifact.

The current non-promoting smoke packet can be regenerated with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hbsm_local_sensitivity_runner \
  --max-input-tokens 8 --layer-limit 8 --block-size 32
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/hbsm_local_sensitivity_20260507/hbsm_gate_packet \
  --mode real --project hbsm
```
