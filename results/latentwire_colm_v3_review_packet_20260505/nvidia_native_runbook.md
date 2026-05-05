# COLM v3 Native NVIDIA Systems Runbook

This runbook is for future native GPU evidence. It does not authorize any COLM_v3
latency, HBM, energy, throughput, or C2C superiority claim until the measurements
are actually run on NVIDIA hardware.

## Setup

1. Work on a native NVIDIA host. Do not use SSH from this agent session.
2. Create a fresh repo-local virtual environment on that host.
3. Install the pinned CUDA/PyTorch/vLLM/SGLang stack recorded by the run.
4. Record GPU model, driver, CUDA version, PyTorch version, and clock/power settings.

## Measurements

- LatentWire packet encode/decode microbenchmarks with cached-source and end-to-end source-scoring rows separated.
- Dense C2C or KV/cache transfer byte movement for matched source/target/task rows.
- vLLM and SGLang serving baselines: TTFT, TPOT, goodput, peak memory, and cache movement where instrumentable.
- Nsight Systems or PyTorch profiler traces for packet decode, source scoring, and any KV/cache baseline.
- Cacheline/DMA-rounded bytes and batch-level framed bytes for every communicated object.

## Required Outputs

- `results/native_nvidia_colm_v3_<date>/metadata.json`
- `results/native_nvidia_colm_v3_<date>/packet_microbench.csv`
- `results/native_nvidia_colm_v3_<date>/dense_cache_baselines.csv`
- `results/native_nvidia_colm_v3_<date>/serving_baselines.csv`
- `results/native_nvidia_colm_v3_<date>/profiler_manifest.md`

## Claim Bar

A systems win can be claimed only if the native rows use matched tasks/models, include
packet-source scoring separately, include a dense cache baseline, and pass the same
source-private claim audit used by COLM_v3.
