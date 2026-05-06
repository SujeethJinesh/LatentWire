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
- `hybrid_trace_packet_builder.py`: converts future saved trace tensor packets
  into strict SSQ-LR/HORN real gate packets and converts HBSM sensitivity rows
  into strict real B1 packets.
- `sensitivity_metrics.py`: quality, drift, and rank-correlation metrics.
- `check_gate_packet.py`: packet validator for synthetic and real Mac-local
  gate results, with stricter `--mode real --project ...` contracts.
- `hybrid_trace_packet_runbook.md`: required real-packet schema for SSQ-LR,
  HORN, and HBSM.

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
directions with prompt-paired flipped controls, and HBSM needs both boundary
flags plus a perturbation-off row with near-zero drift. Real packets also need
`prompt_ids_hash`, `architecture_map_hash`, project-specific aggregate
`summary.json` fields, and a non-promotable decision whenever
`resource_limit_note` is present.

## Claim Boundary

Passing these tests only means the local utilities are deterministic and
internally consistent. Any promoted paper claim still requires the relevant
project gate to pass.
