#!/usr/bin/env bash
# Smoke test: load a small Qwen3.5 model in the SGLang venv to verify the install.
# Uses Qwen3.5-9B because it's small enough to download quickly (~18GB) and tests
# the same Gated DeltaNet architecture family as Qwen3.6.
#
# Run AFTER setup_sglang_venv.sh succeeds.

set -euo pipefail

VENV_PATH="/workspace/.sglang"
SMOKE_MODEL="Qwen/Qwen3.5-9B"
SMOKE_DIR="/workspace/hf_cache/qwen35_smoke"
LOG_FILE="/workspace/LatentWire/infra/sglang_venv/smoke_test_$(date +%Y%m%dT%H%M%SZ).log"

echo "=== SGLang smoke test ===" | tee "$LOG_FILE"
echo "Model: $SMOKE_MODEL" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"

# Check venv exists
if [[ ! -d "$VENV_PATH" ]]; then
  echo "ERROR: venv not found at $VENV_PATH. Run setup_sglang_venv.sh first." | tee -a "$LOG_FILE"
  exit 1
fi

# Check we are NOT in another venv
if [[ -n "${VIRTUAL_ENV:-}" && "$VIRTUAL_ENV" != "$VENV_PATH" ]]; then
  echo "ERROR: A different venv is active: $VIRTUAL_ENV" | tee -a "$LOG_FILE"
  echo "Run 'deactivate' first." | tee -a "$LOG_FILE"
  exit 1
fi

# Activate
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

# Verify HF_TOKEN is set if model is gated
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "INFO: HF_TOKEN not set. Qwen3.5-9B should be open-access, but may need login." | tee -a "$LOG_FILE"
fi

echo "Step 1: Download Qwen3.5-9B to $SMOKE_DIR" | tee -a "$LOG_FILE"
if [[ ! -d "$SMOKE_DIR" || -z "$(ls -A "$SMOKE_DIR" 2>/dev/null)" ]]; then
  huggingface-cli download "$SMOKE_MODEL" --local-dir "$SMOKE_DIR" 2>&1 | tee -a "$LOG_FILE"
else
  echo "Model directory exists and non-empty. Skipping download." | tee -a "$LOG_FILE"
fi

echo "Step 2: Quick forward-pass smoke test" | tee -a "$LOG_FILE"
python - << 'PYTEST' 2>&1 | tee -a "$LOG_FILE"
import sys
import torch
print(f"Pre-load GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Use SGLang's offline engine API for the smoke test, not the server
# This avoids needing to set up HTTP endpoints just to verify model load
try:
    import sglang as sgl
    from sglang import Engine
    print("SGLang Engine imported OK")
except Exception as e:
    print(f"ERROR: failed to import SGLang Engine: {e}")
    sys.exit(1)

print("Attempting to load Qwen3.5-9B via SGLang Engine...")
try:
    engine = Engine(
        model_path="/workspace/hf_cache/qwen35_smoke",
        mem_fraction_static=0.6,  # conservative for smoke test
        trust_remote_code=True,
        log_level="warning",
        # Skip multimodal components for smoke test
        # (Qwen3.5-9B is text-only, but flag is safe to include for forward-compat)
    )
    print("Engine initialized OK")
except Exception as e:
    print(f"ERROR: Engine init failed: {e}")
    sys.exit(1)

print("Running single forward pass...")
try:
    output = engine.generate(
        prompt="Hello, the answer to 2+2 is",
        sampling_params={"temperature": 0.0, "max_new_tokens": 10},
    )
    print(f"Output: {output}")
except Exception as e:
    print(f"ERROR: generate failed: {e}")
    sys.exit(1)

print(f"Post-load GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print("=== Smoke test PASSED ===")
PYTEST

SMOKE_EXIT=${PIPESTATUS[0]}
if (( SMOKE_EXIT != 0 )); then
  echo "ERROR: smoke test failed with exit code $SMOKE_EXIT" | tee -a "$LOG_FILE"
  exit $SMOKE_EXIT
fi

echo "" | tee -a "$LOG_FILE"
echo "=== Smoke test complete ===" | tee -a "$LOG_FILE"
echo "Venv at $VENV_PATH is ready for Phase 5'' (Qwen3.6-35B-A3B)." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Next steps for human:" | tee -a "$LOG_FILE"
echo "  1. Commit and push setup logs to /workspace/LatentWire" | tee -a "$LOG_FILE"
echo "  2. Notify Codex that Phase 5'' venv is ready (after Phase 5'/6 sprint completes)" | tee -a "$LOG_FILE"
