# KILLED: Mac-Only HybridKernel Iteration

## What Was Tried

HybridKernel completed Mac-local architecture mapping, source/runtime audit,
threshold modeling, packet generation, artifact checking, profiler-analysis
parsing, and toy Triton interpreter correctness.

## Why It Died

Additional Mac-only kernel work cannot answer the paper's core systems question:
whether a real NVIDIA/vLLM serving path has separable attention/SSM boundary
overhead. The next admissible evidence is server-side Nsight Systems and Nsight
Compute data reduced by the preregistered parser.

## Salvage Value

The Mac artifacts remain the GPU handoff packet and reviewer audit trail. They
should be used to run or reject the native profiler gate, not to justify more
local kernels or speed claims.
