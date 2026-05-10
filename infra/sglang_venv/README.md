# SGLang venv setup for Phase 5'' (Qwen3.6 cross-lineage validation)

This directory contains the human-operated setup for the Phase 5'' Qwen3.6
measurement. Codex does NOT execute these scripts; only humans do.

## What this is

A separate Python venv at `/workspace/.sglang` containing SGLang 0.5.9 with
PyTorch 2.9.1+cu128, used exclusively for Phase 5'' (Qwen3.6-35B-A3B
cross-lineage measurement). The original vLLM 0.10.2 / PyTorch 2.8 / triton 3.4
environment used by Phase 4, Phase 5', and Phase 6 is NOT modified.

## Why this exists

Qwen3.6-35B-A3B uses the `Qwen3_5MoeForConditionalGeneration` architecture
which is not supported by vLLM 0.10.2 (the pinned version of our validated
stack). Upgrading vLLM in place would break the validated environment for
Phase 4/5'/6. Instead, we install a parallel SGLang venv that has native
Qwen3.6 support including the Gated DeltaNet kernels.

## How to run

In a fresh tmux pane (no other venv active):

```bash
cd /workspace/LatentWire/infra/sglang_venv
./setup_sglang_venv.sh         # ~10-15 min, downloads ~5GB of packages
./smoke_test_qwen35.sh          # ~5-10 min, downloads ~18GB Qwen3.5-9B
```

Setup logs are written to this directory with timestamps. If either script
fails, read the log and decide whether to:
- Retry (transient pip/network errors)
- Fall back to second-pod approach (sgl-kernel cu128 wheel missing)
- Skip Phase 5'' entirely (paper falls back to Mamba-2-specific framing)

## After setup succeeds

Notify Codex that the venv is ready by:
1. Committing the setup logs in this directory
2. Updating swarm/state.json to add `"sglang_venv_ready": true` (manually)
3. In the Codex pane, sending: "The /workspace/.sglang venv smoke test passed.
   Phase 5'' is now unblocked once Phase 5'/6 sprint completes."

## Hygiene rule (CRITICAL)

NEVER source both venvs in the same shell. The triton ABI versions differ
(3.4 in vLLM stack, 3.5.1 in SGLang stack); cross-contamination causes
runtime errors. Use separate tmux panes.

## Cost / time budget

- Setup: ~15 min one-time
- Smoke test: ~10 min one-time
- Phase 5'' measurement (after pre-flight): ~3-4 GPU hours
- Incremental cost: ~$15-25
