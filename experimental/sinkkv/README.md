# SinkKV

SinkKV tests whether protecting the first sink K/V positions at high precision
while quantizing the rest of the KV cache can recover quality at the same
or near-same memory budget as uniform low-precision KV.

## Current Readiness

Status: **NEW / deterministic synthetic probe passed; real Mac gate pending**.

Estimated completion:

- **20%** as a positive-method paper: branch selected, gate defined, shared
  utilities scaffolded, and the deterministic no-download policy probe passed.
- **0%** as a systems-result paper: no native GPU implementation or benchmark
  exists.

## Paper Story

SinkAware showed that sink positions are stable and important, but approximating
only sink logits has too small a systems wedge. SinkKV reverses the use of that
evidence: keep sink K/V exact or high precision, quantize non-sink K/V
aggressively, and test whether protected outlier positions recover the quality
gap introduced by uniform FP4-style KV quantization.

## Mac Gate

Primary preregistration:

- `phase2/preregister_sink_protected_kv_20260506.md`

The first real gate is quality at fixed or near-fixed memory, not native speed.
Use simulated quantization from `experimental/shared/fp4_simulator.py`.

Synthetic policy sanity check:

- script: `phase2/sinkkv_deterministic_probe.py`
- packet: `phase2/results/sinkkv_deterministic_probe/`
- decision: `SYNTHETIC_PASS_REAL_DUMPS_NEXT`
- boundary: synthetic-only; not GPU speed; not benchmark accuracy; does not
  skip query-dependent `QK_sink`.

Gate condition:

```text
(sink_FP4 - BF16) / (uniform_FP4 - BF16) <= 0.5
```

for lower-is-better quality metrics, with the minimum-row confidence interval
excluding 1.0 on at least two model/length surfaces.

## First Experiments To Run

1. Create or reuse cached K/V tensors for a small model and synthetic/real
   continuation traces.
2. Compare BF16 K/V, uniform simulated MXFP4 K/V, and sink-protected simulated
   MXFP4 K/V.
3. Record sink mass, continuation NLL if available, and relative attention
   output drift.
4. Kill quickly if sink protection does not recover at least half of the
   uniform-FP4 quality gap.

## Output Paths

Use one directory per gate run:

```text
experimental/sinkkv/results/sinkkv_gate_<YYYYMMDD>_<model_slug>_<surface_slug>/
```

Each result packet should contain:

- `config.json`
- `raw_rows.jsonl`
- `summary.md`
- `summary.json`
- `decision.md`

## Local Setup

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python -m pytest experimental/shared/tests -q
```

## GPU Rule

Do not spend GPU time on SinkKV until the Mac gate passes and the exact
protected-position recipe is frozen.
