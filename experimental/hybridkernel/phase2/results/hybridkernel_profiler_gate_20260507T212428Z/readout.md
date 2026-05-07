# HybridKernel Native Profiler Readout

| Question | Evidence | Decision |
|---|---|---|
| Distinct boundary conversion/materialization kernel? | Full-matrix Nsight Systems traces contain ordinary vLLM CUDA kernels but no distinct boundary-local conversion/materialization kernel and no layer-boundary NVTX range in the fixed request windows. Evidence rows are listed below and reduction inputs are in `metadata/reduction_worksheet.tsv`. | No; row reduction uses `no_boundary_signal_kill`. |
| Boundary idle or launch gap? | No boundary-local interval could be isolated from the server-side CUDA trace, so no launch-gap window was selected for Nsight Compute. | No measurable boundary-specific launch gap. |
| Extra DRAM/L2 traffic near boundary? | Nsight Compute was not run because there was no boundary-local kernel/window to profile after Nsight Systems reduction. The preregistered no-boundary path requires a clean kill rather than inventing an NCU selection. | Not measured; skipped per no-boundary-signal rule. |
| End-to-end impact estimate clears 3%? | `profiler_analysis_gate.json` recomputes a 0.000000 recoverable-gain upper bound for all nine rows; primary bootstrap CI is [0.000000, 0.000000]. | No; the 3% promotion shelf is not reached. |
| Same-family controls available? | Three same-family Granite non-boundary control traces are present: `granite_same_family_r1`, `granite_same_family_r2`, and `granite_same_family_r3`. | Available and below the 3% gate. |
| Cross-family falsification attempted? | Three pre-committed replacement Nemotron Nano 9B v2 traces are present: `cross_family_r1`, `cross_family_r2`, and `cross_family_r3`. | Attempted and below the 3% gate. |

## Decision

`KILL or shelve: native profiler summaries show less than 1% recoverable gain.`

Do not spend kernel implementation time without a new profiler anomaly.

## Row Evidence

- `granite_primary_r1` `primary_hybrid` `ibm-granite/granite-4.0-h-tiny`: gain upper bound 0.000000, window 174308.878000-182107.938000 ms, artifact `nsys/granite_primary_r1.sanitized.sqlite`.
- `granite_primary_r2` `primary_hybrid` `ibm-granite/granite-4.0-h-tiny`: gain upper bound 0.000000, window 145263.865000-152748.181000 ms, artifact `nsys/granite_primary_r2.sanitized.sqlite`.
- `granite_primary_r3` `primary_hybrid` `ibm-granite/granite-4.0-h-tiny`: gain upper bound 0.000000, window 174479.942000-182274.068000 ms, artifact `nsys/granite_primary_r3.sanitized.sqlite`.
- `granite_same_family_r1` `same_family_control` `ibm-granite/granite-4.0-h-tiny`: gain upper bound 0.000000, window 119679.109000-127353.605000 ms, artifact `nsys/granite_same_family_r1.sanitized.sqlite`.
- `granite_same_family_r2` `same_family_control` `ibm-granite/granite-4.0-h-tiny`: gain upper bound 0.000000, window 93203.896000-100924.410000 ms, artifact `nsys/granite_same_family_r2.sanitized.sqlite`.
- `granite_same_family_r3` `same_family_control` `ibm-granite/granite-4.0-h-tiny`: gain upper bound 0.000000, window 96329.843000-103797.805000 ms, artifact `nsys/granite_same_family_r3.sanitized.sqlite`.
- `cross_family_r1` `cross_family_falsification` `nvidia/NVIDIA-Nemotron-Nano-9B-v2`: gain upper bound 0.000000, window 196767.318000-211585.459000 ms, artifact `nsys/cross_family_r1.sanitized.sqlite`.
- `cross_family_r2` `cross_family_falsification` `nvidia/NVIDIA-Nemotron-Nano-9B-v2`: gain upper bound 0.000000, window 169955.791000-185004.831000 ms, artifact `nsys/cross_family_r2.sanitized.sqlite`.
- `cross_family_r3` `cross_family_falsification` `nvidia/NVIDIA-Nemotron-Nano-9B-v2`: gain upper bound 0.000000, window 100055.000000-114891.435000 ms, artifact `nsys/cross_family_r3.sanitized.sqlite`.

## Reduction Notes

This packet does not claim a speedup. It records a full-matrix native profiler
kill because the server-side Nsight Systems traces did not expose a distinct
boundary-local conversion/materialization or launch/locality signal that could
support a boundary-fusion prototype. The reduction therefore sets
`attention_ssm_boundary_ms = 0.0`, `matched_non_boundary_ms = 0.0`, and
`recoverable_fraction = 0.0` for every row, with exact request windows and
artifact hashes preserved in `metadata/reduction_input_manifest.json`.
