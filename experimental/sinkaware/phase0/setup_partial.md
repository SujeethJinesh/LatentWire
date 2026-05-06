# SinkAware Phase 0 Partial Setup

Date: 2026-05-05

Superseded: see `phase0/setup_complete.md` for the current 2026-05-06
repo-root `./venv_arm64` setup, import verification, and `pip check` result.

Status: partial Mac-only setup. No SSH, GPU, global installs, large model
downloads, or dependency installs were used.

## Local State

- Existing project venv: `experimental/sinkaware/.venv` (`Python 3.9.13` recorded in progress).
- Created ignored local directories:
  - `experimental/sinkaware/external/`
  - `experimental/sinkaware/artifacts/`
  - `experimental/sinkaware/phase0/`
  - `experimental/sinkaware/phase1/`

## Primary-Source Checkouts

The following repos were cloned under ignored `external/` for source inspection:

- FlashInfer: `https://github.com/flashinfer-ai/flashinfer`
- FlashAttention: `https://github.com/Dao-AILab/flash-attention`
- StreamingLLM: `https://github.com/mit-han-lab/streaming-llm`
- FlashMLA: `https://github.com/deepseek-ai/FlashMLA`
- DeepSeek-V3.2-Exp: `https://github.com/deepseek-ai/DeepSeek-V3.2-Exp`
- Native Sparse Attention: `https://github.com/fla-org/native-sparse-attention`
- GPT-OSS: `https://github.com/openai/gpt-oss`

## Not Completed

- `requirements.txt` was not installed.
- No tests or kernels were run.
- Phase 0 remains partial because setup was limited to source-audit readiness.
