# HybridKernel Pre-Profile Command Notes

- No Nsight profiler metric rows had been collected before choosing the cross-family replacement.
- Qwen3-Next BF16 is cached but its weights exceed this single 96GB node.
- nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 was tested first and failed under vLLM 0.10.2 with `NemotronHConfig` missing `rms_norm_eps`; vLLM recipes recommend newer vLLM for that model.
- nvidia/NVIDIA-Nemotron-Nano-9B-v2 loaded and served a non-evidence vLLM preflight at BF16 with CUDA graph capture on this node.
- The replacement metadata is committed before profiler execution and copied to `metadata/cross_family_control_replacement_template.json`.

- 2026-05-07T22:03Z: vLLM 0.10.2 rejected --profiler-config.profiler cuda, so the packet uses static server-side Nsight Systems capture instead of CUDA-profiler API bracketing. Failed pre-profile log is kept in .debug/hybridkernel_preflight/ and not used as evidence.
