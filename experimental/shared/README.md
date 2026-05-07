# Experimental Shared Utilities

Shared Mac-local utilities for the relevant hybrid-quantization branches:
SSQ-LR, HORN, and HBSM.

These helpers are intentionally small and deterministic. They are not GPU
kernels and they do not support throughput, latency, HBM, or energy claims.
Use them for preregistered Mac gates only.

## Utilities

- `fp4_simulator.py`: deterministic INT/FP-style cast-and-cast-back
  quantization simulators and quality-gap recovery helpers.
- `activation_dumper.py`: lightweight tensor packet save/load helpers for
  cached traces.
- `boundary_inspector.py`: layer-kind and attention/SSM boundary helpers.
- `hybrid_architecture_maps.py`: explicit config-derived boundary maps used to
  validate real trace packet provenance.
- `hybrid_model_eligibility.py`: metadata-only Hugging Face size/cache
  preflight for the live hybrid targets.
- `hybrid_trace_plan.py`: deterministic SSQ-LR/HORN/HBSM trace-collection
  plan from the frozen prompt manifest and config-derived architecture maps.
- `hybrid_trace_packet_builder.py`: converts future saved trace tensor packets
  into strict SSQ-LR/HORN real gate packets and converts HBSM sensitivity rows
  into strict real B1 packets.
- `hybrid_gate_evaluators.py`: recomputes S1/H1/B1 pass/fail summaries from
  packet rows so real packets cannot promote from hand-written aggregate labels.
  The active evaluators use prompt-level S1 lower bounds, H1 non-boundary and
  permuted-direction controls through matched boundary labels, and HBSM
  measured top-decile scoring derived from primary-row drift rather than
  caller-supplied labels.
- `sensitivity_metrics.py`: quality, drift, and rank-correlation metrics.
- `check_gate_packet.py`: packet validator for synthetic and real Mac-local
  gate results, with stricter `--mode real --project ...` contracts and an
  explicit non-promoting schema-rehearsal path for checker-path tests.
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
set instead of allowing uncited row coverage. Real packets
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
