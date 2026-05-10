#!/usr/bin/env bash
# Sets up a separate SGLang venv at /workspace/.sglang for Phase 5'' (Qwen3.6 cross-lineage).
# Does NOT touch the original vLLM environment used by Phase 4/5'/6.
#
# Run this ONLY in a fresh tmux pane that has NOT activated any other venv.
# Run from any directory (the venv lives at /workspace/.sglang, not in this repo).

set -euo pipefail

VENV_PATH="/workspace/.sglang"
LOG_FILE="/workspace/LatentWire/infra/sglang_venv/setup_$(date +%Y%m%dT%H%M%SZ).log"

echo "=== SGLang venv setup ===" | tee "$LOG_FILE"
echo "Venv path: $VENV_PATH" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Check we are NOT inside another venv
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: A venv is already active: $VIRTUAL_ENV" | tee -a "$LOG_FILE"
  echo "Run 'deactivate' first and ensure no other venv is sourced before retrying." | tee -a "$LOG_FILE"
  exit 1
fi

# Check Python 3.12 is available
if ! command -v python3.12 &>/dev/null; then
  echo "ERROR: python3.12 not found. SGLang requires Python 3.10-3.14." | tee -a "$LOG_FILE"
  echo "Available pythons:" | tee -a "$LOG_FILE"
  ls /usr/bin/python3* 2>&1 | tee -a "$LOG_FILE" || true
  exit 1
fi

# Check existing venv
if [[ -d "$VENV_PATH" ]]; then
  echo "Existing venv found at $VENV_PATH. Aborting to avoid clobbering." | tee -a "$LOG_FILE"
  echo "If you intend to recreate, run: rm -rf $VENV_PATH" | tee -a "$LOG_FILE"
  exit 1
fi

# Check disk space (need at least 30GB for torch + cuda libs + sglang)
AVAIL_GB=$(df -BG /workspace | awk 'NR==2 {gsub("G",""); print $4}')
if (( AVAIL_GB < 50 )); then
  echo "ERROR: only ${AVAIL_GB}GB free on /workspace. Need at least 50GB for venv + model weights." | tee -a "$LOG_FILE"
  exit 1
fi

echo "Step 1: Create venv at $VENV_PATH" | tee -a "$LOG_FILE"
python3.12 -m venv "$VENV_PATH" 2>&1 | tee -a "$LOG_FILE"

echo "Step 2: Activate venv" | tee -a "$LOG_FILE"
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

echo "Step 3: Upgrade pip and install uv" | tee -a "$LOG_FILE"
pip install --upgrade pip 2>&1 | tee -a "$LOG_FILE"
pip install uv 2>&1 | tee -a "$LOG_FILE"

echo "Step 4: Install PyTorch 2.9.1 with CUDA 12.8 support" | tee -a "$LOG_FILE"
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128 \
  2>&1 | tee -a "$LOG_FILE"

echo "Step 5: Install SGLang 0.5.9 and dependencies" | tee -a "$LOG_FILE"
uv pip install sglang==0.5.9 2>&1 | tee -a "$LOG_FILE"

echo "Step 6: Install flashinfer for CUDA 12.8 + torch 2.9" | tee -a "$LOG_FILE"
uv pip install flashinfer-python==0.5.0 \
  --index-url https://flashinfer.ai/whl/cu128/torch2.9 \
  2>&1 | tee -a "$LOG_FILE" || {
    echo "WARNING: flashinfer install failed. Trying without index-url..." | tee -a "$LOG_FILE"
    uv pip install flashinfer-python==0.5.0 2>&1 | tee -a "$LOG_FILE" || {
      echo "WARNING: flashinfer install still failed. SGLang may fall back to slower backend." | tee -a "$LOG_FILE"
    }
  }

echo "Step 7: Install sgl-kernel for CUDA 12.8" | tee -a "$LOG_FILE"
uv pip install sgl-kernel --index-url https://docs.sglang.ai/whl/cu128 2>&1 | tee -a "$LOG_FILE" || {
  echo "WARNING: sgl-kernel cu128 install failed. Trying default index..." | tee -a "$LOG_FILE"
  uv pip install sgl-kernel 2>&1 | tee -a "$LOG_FILE" || {
    echo "ERROR: sgl-kernel install failed. Phase 5'' cannot proceed on venv path." | tee -a "$LOG_FILE"
    echo "Fallback options: 1) downgrade SGLang to 0.5.7 which had cu128 wheels, 2) use second-pod approach" | tee -a "$LOG_FILE"
    exit 1
  }
}

echo "Step 8: Install transformers >= 4.57.1 (required for Qwen3.6)" | tee -a "$LOG_FILE"
uv pip install "transformers>=4.57.1" 2>&1 | tee -a "$LOG_FILE"

echo "Step 9: Verify install" | tee -a "$LOG_FILE"
python -c "
import sys
import torch
print(f'Python: {sys.version}')
print(f'Torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')

import sglang
print(f'SGLang: {sglang.__version__ if hasattr(sglang, \"__version__\") else \"installed (no __version__)\"}')

try:
    import sgl_kernel
    print(f'sgl_kernel: imported successfully')
except Exception as e:
    print(f'sgl_kernel: FAILED to import: {e}')
    sys.exit(1)

try:
    import flashinfer
    print(f'flashinfer: imported successfully')
except Exception as e:
    print(f'flashinfer: WARNING - failed to import: {e}')

try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except Exception as e:
    print(f'transformers: FAILED: {e}')
    sys.exit(1)

print()
print('=== All required packages imported successfully ===')
" 2>&1 | tee -a "$LOG_FILE"

VERIFY_EXIT=${PIPESTATUS[0]}
if (( VERIFY_EXIT != 0 )); then
  echo "ERROR: verification failed with exit code $VERIFY_EXIT" | tee -a "$LOG_FILE"
  exit $VERIFY_EXIT
fi

echo "" | tee -a "$LOG_FILE"
echo "=== SGLang venv setup complete ===" | tee -a "$LOG_FILE"
echo "Activate with: source $VENV_PATH/bin/activate" | tee -a "$LOG_FILE"
echo "Next step: run smoke_test_qwen35.sh to verify model loading" | tee -a "$LOG_FILE"
