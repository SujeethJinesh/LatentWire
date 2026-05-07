# HybridKernel Pre-Profile Command Notes

- No Nsight profiler metric rows had been collected before choosing the cross-family replacement.
- Qwen3-Next BF16 is cached but its weights exceed this single 96GB node.
- nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 was tested first and failed under vLLM 0.10.2 with `NemotronHConfig` missing `rms_norm_eps`; vLLM recipes recommend newer vLLM for that model.
- nvidia/NVIDIA-Nemotron-Nano-9B-v2 loaded and served a non-evidence vLLM preflight at BF16 with CUDA graph capture on this node.
- The replacement metadata is committed before profiler execution and copied to `metadata/cross_family_control_replacement_template.json`.
