# Phase 2 Compatibility Block: cross_model_validation_outlier_migrate

Blocked at: 2026-05-08T21:28:19Z

Resolved for the 10-hour authorized window at: 2026-05-08T21:44:47Z

Resolution: `swarm/goal.md` now authorizes a non-vLLM-upgrade path. The
Qwen3.6/Kimi Linear compatibility block remains for full Phase 2 validation,
but the swarm may resume by running Nemotron-3-only partial cross-validation
and documenting Qwen3.6/Kimi Linear as deferred. vLLM upgrade and Qwen3.6/Kimi
weight downloads remain forbidden during the window.

Current entry: `cross_model_validation_outlier_migrate`

## Summary

OutlierMigrate Phase 2 cross-model validation cannot proceed safely on the
current GPU environment without a human decision about the serving backend.

The queue entry requires:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- `Qwen/Qwen3.6-35B-A3B`

No Phase 2 weights were downloaded for these models during this check.

## Local environment

- torch: `2.8.0+cu128`
- CUDA reported by torch: `12.8`
- triton: `3.4.0`
- transformers: `4.57.6`
- vLLM: `0.10.2`

This is the pre-validated pod stack. Replacing torch/triton has high blast
radius because previous profiler, vLLM, and Mamba fast-path work depends on
this stack.

## Compatibility checks

`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`:

- `AutoConfig.from_pretrained(..., trust_remote_code=True)` succeeds.
- Config class: `NemotronHConfig`
- Architecture: `NemotronHForCausalLM`
- Local vLLM 0.10.2 registry includes `NemotronHForCausalLM`.
- This model appears compatible with the current stack at the config/registry
  level.

`Qwen/Qwen3.6-35B-A3B`:

- `AutoConfig.from_pretrained(..., trust_remote_code=True)` fails under
  transformers 4.57.6:
  `ValueError: The checkpoint you are trying to load has model type
  qwen3_5_moe but Transformers does not recognize this architecture.`
- The HuggingFace config declares architecture
  `Qwen3_5MoeForConditionalGeneration` and model type `qwen3_5_moe`.
- Local vLLM 0.10.2 registry does not include
  `Qwen3_5MoeForConditionalGeneration` or
  `Qwen3_5ForConditionalGeneration`.
- Local vLLM 0.10.2 does include `Qwen3MoeForCausalLM` and
  `Qwen3NextForCausalLM`, but those are not the Qwen3.6/Qwen3.5 architecture
  declared by this checkpoint.

## Upstream evidence checked

- Latest vLLM supported-models documentation lists
  `Qwen3_5MoeForConditionalGeneration` for Qwen3.5-MOE, indicating support
  exists in newer vLLM releases:
  `https://docs.vllm.ai/en/latest/models/supported_models/`
- The vLLM Qwen3.6 recipe lists the prerequisite as `vLLM version:
  >= 0.17.0`, not the current local `0.10.2`:
  `https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B`
- The Qwen3.6 HuggingFace config declares
  `Qwen3_5MoeForConditionalGeneration` and `qwen3_5_moe`:
  `https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json`
- A vLLM issue records the same unsupported-architecture failure mode for
  `Qwen3_5MoeForConditionalGeneration` in older stacks:
  `https://github.com/vllm-project/vllm/issues/35344`

## Blast-radius evaluation

`pip install --dry-run transformers==5.8.0` would install only:

- `transformers-5.8.0`
- `huggingface_hub-1.14.0`

This may resolve Transformers config loading, but it does not add the missing
model architecture to local vLLM 0.10.2.

`pip install --dry-run vllm==0.20.1 transformers==5.8.0` would install or
replace, among many packages:

- `torch-2.11.0`
- `triton-3.6.0`
- `cuda-toolkit-13.0.2`
- CUDA 13 `nvidia-*` packages
- `vllm-0.20.1`
- `transformers-5.8.0`

That would replace the validated torch 2.8 / CUDA 12.8 / triton 3.4 stack.
Per `swarm/goal.md`, this is too large a dependency change to apply without
human approval.

## Human decision needed

Choose one of:

1. Approve a new vLLM/torch/CUDA environment for Phase 2 only, isolated from
   the current `.venv_gpu` if possible.
2. Approve a non-vLLM backend for Qwen3.6 Phase 2 validation, with explicit
   acceptance that runner/checker artifacts will differ from the vLLM path.
3. Amend the Phase 2 queue/model choice before any Phase 2 data is collected.

Do not silently substitute Qwen3.6 or download Phase 2 weights until this is
resolved.

## Independent check

A fresh compatibility subagent independently reached the same conclusion:
Qwen3.6 is unsupported on local vLLM 0.10.2, no low-blast workaround exists
without replacing torch/triton, and a human decision is justified.
