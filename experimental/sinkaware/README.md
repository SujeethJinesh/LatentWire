# SinkAware

SinkAware is a bounded experiment for testing whether attention sink tokens can
be handled as a static prior in long-context attention kernels.

This directory is self-contained inside the main repository. Use the repo-local
virtual environment at `./venv_arm64` from the repository root for current Mac
work. The older `experimental/sinkaware/.venv` is kept only as local scratch
state.

## Completion Estimate And Roadmap

Estimated workshop-paper completion: **80%**.

What is already complete:

- exact static-prior counterexample and kill decision;
- rank-2 approximate branch on GPT2/OPT Mac controls;
- split, length, sink-count, cross-model-family, downstream patch, and rank
  frontier diagnostics;
- Triton interpreter correctness scaffolds;
- native packet validator, COLM-style draft, and reviewer pack.

What remains:

- **15%**: native NVIDIA packet with matched quality drift, per-head drift,
  latency, and NCU memory/occupancy counters.
- **5%**: paper update with the native promote/kill decision.

The next high-value action is the native packet. More Mac-local sweeps on this
same branch are not expected to change the claim.

## Local Setup

From the repository root:

```bash
./venv_arm64/bin/python -m pip install -r experimental/sinkaware/requirements.txt
```

Keep external reference repositories and downloaded artifacts outside git, for
example under `experimental/sinkaware/external/` and
`experimental/sinkaware/artifacts/`.

## Current Gate

The exact static-prior branch is killed: fixed sink-token logits remain
query-dependent. The only live branch is approximate per-head low-rank
sink-logit prediction while keeping all non-sink scores exact.

The current 48-trace distilgpt2 probe is weakly positive at the aggregate layer
level (`rank2 output rel-L2=0.141` versus `position=0.170`) but mixed at
layer-head granularity (`+0.0297 +/- 0.0378` output rel-L2 improvement, 20/72
head wins). Simple validation head selection failed. A randomized split/seed
repeat kept all-head rank-2 positive across three token splits
(`+0.0368 +/- 0.0006` output rel-L2 improvement), but the layer-head win rate
remained low (`0.282 +/- 0.024`). A bounded length/sink sweep over
`max_length={64,96}` and `sink_tokens={2,4}` kept all-head rank-2 positive
(`+0.0366 +/- 0.0024`, minimum config `+0.0342`), while preserving the same
per-head caveat. A larger trace-level frozen split repeat on 48 traces also kept
all-head rank-2 positive (`+0.0379 +/- 0.0014`, minimum split `+0.0367`), but
the head win rate stayed low (`0.278 +/- 0.016`). Treat this as a
correctness/repeatability gate, not as a quality or speed result.

A repeated held-out/model-family falsification gate on 48 traces with split
seeds `0,1,2` also stayed positive for separately fit per-model predictors:
`distilgpt2` improved by a measured `+0.0306 +/- 0.0023` output rel-L2 and
`facebook/opt-125m` improved by a measured `+0.0788 +/- 0.0069`. This is not cross-model
predictor transfer and makes no GPU speed claim.

The downstream causal-LM patch control is also Mac-saturated for the current
branch. On 48 traces, split seeds `0,1,2`, lengths `64,96`, sink counts `2,4`,
and separately fit `distilgpt2`/`facebook/opt-125m` predictors, exact
replacement remains a no-op and rank-2 is closer than position-only in loss
drift and KL for every model/config row. Minimum model loss improvement is
`+0.0263`. This is still a quality-control diagnostic only: top-1 disagreement
remains non-negligible and no benchmark or GPU speed result is claimed.

A 48-trace downstream rank frontier at length 96/sink4 confirms the
quality/cost shape: rank1, rank2, rank4, and rank8 reduce absolute loss drift
to `0.137`, `0.096`, `0.062`, and `0.044`, respectively, but the cost model
estimates rank4/rank8 above exact four-sink QK multiply-add cost. Rank2 remains
the live systems compromise.

Phase 4 Macbook kernel work must run through the Triton interpreter with
`TRITON_INTERPRET=1`, `TRITON_CPU_BACKEND=1`, and a repo-local `TRITON_HOME`
against a CPU reference. Interpreter-mode correctness is not GPU performance
evidence, and native speed claims require NVIDIA hardware.

Future native GPU packets must pass
`phase2/check_native_gpu_packet.py` before the paper cites latency, HBM, or
quality numbers. The checker is an artifact-completeness guard only; it is not
performance evidence.

## GPU-Node Quickstart

Use this only on a local NVIDIA/CUDA node. Do not SSH from this repo. Copy or
checkout the repository on the GPU node, then run from the repository root.

### 1. Environment

```bash
cd /path/to/LatentWire
python3 -m venv .venv_gpu
source .venv_gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r experimental/sinkaware/requirements.txt

nvidia-smi
python - <<'PY'
import torch, triton
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("triton", triton.__version__)
assert torch.cuda.is_available()
PY
```

If the GPU node already has a working CUDA/Torch/Triton environment, use it and
record the exact package versions in `metadata.json`.

### 2. Result Directory

Create one packet directory per native attempt:

```bash
export SINKAWARE_ROOT="$PWD/experimental/sinkaware"
export SINKAWARE_GPU_PACKET="$SINKAWARE_ROOT/artifacts/native_gpu/sinkaware_native_$(date -u +%Y%m%dT%H%M%SZ)_rank2"
mkdir -p "$SINKAWARE_GPU_PACKET"
```

Return the whole `$SINKAWARE_GPU_PACKET` directory after validation.

### 3. Required Files

The packet must contain exactly these review artifacts:

- `metadata.json`
- `quality_drift.csv`
- `quality_drift_by_head.csv`
- `latency.csv`
- `ncu_summary.csv`
- `decision.md`
- `artifact_check.json` after validation

Use these canonical row ids in every CSV:

- `exact_attention`
- `exact_fixed_sink_decomposition`
- `rank2_sink_logit_predictor`
- `position_only_predictor`

For limited GPU time, prioritize:

- model: `distilgpt2`;
- shape: `sequence_length=96`, `batch_size=1`;
- optional breadth only if time remains: `facebook/opt-125m`,
  `sequence_length=64`, and sink counts 2/4.

Every measured `model`/`sequence_length`/`batch_size` group must include all
four row ids. Each latency row/model/shape group must have at least three
distinct `run_id` values.

### 4. Required Schemas

`metadata.json` must identify a native NVIDIA CUDA environment:

```json
{
  "gpu": "NVIDIA ...",
  "driver": "...",
  "cuda": "...",
  "pytorch": "...",
  "triton": "...",
  "model": "distilgpt2",
  "dtype": "bfloat16",
  "sequence_shapes": [{"sequence_length": 96, "batch_size": 1}]
}
```

`quality_drift.csv` columns:

```text
row,model,sequence_length,batch_size,layer,output_rel_l2,sink_mass_mae,attention_l1
```

`quality_drift_by_head.csv` columns:

```text
row,model,sequence_length,batch_size,layer,head,output_rel_l2,sink_mass_mae,attention_l1
```

`latency.csv` columns:

```text
row,model,sequence_length,batch_size,run_id,latency_ms
```

`ncu_summary.csv` columns:

```text
row,model,sequence_length,batch_size,dram_bytes,l2_bytes,achieved_occupancy,registers_per_thread
```

`decision.md` must contain an explicit `PROMOTE` or `KILL`, the rank-2 quality
threshold, and the native speed/memory/HBM evidence.

### 5. Validate Before Leaving The GPU Node

```bash
python experimental/sinkaware/phase2/check_native_gpu_packet.py \
  --run-dir "$SINKAWARE_GPU_PACKET" \
  | tee "$SINKAWARE_GPU_PACKET/artifact_check.json"
```

Expected validation output:

- JSON `status` is `PASS`;
- `errors` is empty;
- `artifact_check.json` is saved in the packet directory.

A validator pass only means the packet is complete enough for review. It is not
a speed, HBM, quality, or promotion claim.

### 6. Decision Rule

Promote only if all are true:

- rank-2 mean output relative-L2 is no worse than 0.15;
- rank-2 beats position-only in every matched model/shape group on output
  relative-L2, downstream loss drift, and KL-to-exact;
- aggregate top-1 disagreement is at most 0.15, with model/shape subgroup
  values reported;
- rank-2 improves speed or memory traffic over exact attention by at least 3%;
- exact fixed-sink decomposition does not already capture the systems win.

Kill if rank-2 is slower than exact attention, quality drift is unbounded, or
position-only is indistinguishable from rank-2.
