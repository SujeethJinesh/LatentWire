#!/bin/bash
#SBATCH --job-name=quant_milestones
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=marlowe-m000066
#SBATCH --partition=preempt
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --output=runs/quant_milestones_%j.log
#SBATCH --error=runs/quant_milestones_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

mkdir -p runs

# Ensure conda is visible on the cluster (adjust if your module path differs).
if [ -x /cm/shared/apps/Mambaforge/24.3.0-0/bin/conda ]; then
  export PATH="/cm/shared/apps/Mambaforge/24.3.0-0/bin:${PATH}"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/projects/m000066/sujinesh/LatentWire}}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}"

# Caches on scratch to avoid quota issues on /projects.
export HF_HOME="${HF_HOME:-/scratch/m000066/${USER}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export C2C_CKPT_ROOT="${C2C_CKPT_ROOT:-/scratch/m000066/${USER}/c2c_checkpoints}"

# Milestone toggles (set to 1 when you want to run them).
RUN_MILESTONE_0="${RUN_MILESTONE_0:-0}"
RUN_MILESTONE_2="${RUN_MILESTONE_2:-0}"

# Chunked eval settings (resume at chunk boundaries).
CHUNK_SIZE="${CHUNK_SIZE:-200}"
EVAL_SMOKE="${EVAL_SMOKE:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-50}"

# Run tag prefix for deterministic folder names.
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-$(date +%Y%m%d_%H%M%S)}"

handle_signal() {
  echo "Received preemption signal. Job will exit and can be requeued."
}
trap handle_signal USR1 TERM

dataset_len() {
  python - <<'PY' "$1"
import sys
from datasets import load_dataset

dataset = sys.argv[1]
if dataset == "openbookqa":
    ds = load_dataset("openbookqa")
    split = ds["test"]
elif dataset == "ai2-arc":
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    split = ds["test"]
else:
    raise SystemExit(f"Unknown dataset: {dataset}")
print(len(split))
PY
}

patch_limit() {
  python - <<'PY' "$1" "$2" "$3"
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
cfg = yaml.safe_load(cfg_path.read_text())
cfg.setdefault("eval", {})["limit"] = [start, end]
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
}

write_progress() {
  python - <<'PY' "$1" "$2" "$3" "$4"
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1])
start = int(sys.argv[2])
end = int(sys.argv[3])
total = int(sys.argv[4])
payload = {
    "next_start": end,
    "total": total,
}
progress_path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

read_next_start() {
  python - <<'PY' "$1"
import json
import sys
from pathlib import Path

progress_path = Path(sys.argv[1])
if not progress_path.exists():
    print(0)
    raise SystemExit(0)
data = json.loads(progress_path.read_text())
print(int(data.get("next_start", 0)))
PY
}

run_eval_chunks() {
  local run_root="$1"
  local config_name="$2"
  local dataset="$3"
  local log_path="$4"
  local progress_dir="${run_root}/progress"
  local progress_file="${progress_dir}/${dataset}.json"
  local chunks_log="${progress_dir}/${dataset}_chunks.jsonl"

  mkdir -p "${progress_dir}"

  local total
  total="$(dataset_len "${dataset}")"

  if [ "${EVAL_SMOKE}" = "1" ]; then
    local end="${SMOKE_LIMIT}"
    if [ "${end}" -gt "${total}" ]; then
      end="${total}"
    fi
    echo "Smoke eval ${dataset}: [0, ${end}) of ${total}"
    patch_limit "${run_root}/configs/${config_name}" 0 "${end}"
    python quantization/C2C/script/evaluation/unified_evaluator.py \
      --config "${run_root}/configs/${config_name}" 2>&1 | tee -a "${log_path}"
    echo "{\"start\":0,\"end\":${end},\"total\":${total},\"timestamp\":\"$(date -Iseconds)\"}" >> "${chunks_log}"
    return
  fi

  local start
  start="$(read_next_start "${progress_file}")"
  if [ "${start}" -ge "${total}" ]; then
    echo "Resume: ${dataset} already complete (start=${start}, total=${total})."
    return
  fi

  while [ "${start}" -lt "${total}" ]; do
    local end=$((start + CHUNK_SIZE))
    if [ "${end}" -gt "${total}" ]; then
      end="${total}"
    fi
    echo "Eval ${dataset}: [${start}, ${end}) of ${total}"
    patch_limit "${run_root}/configs/${config_name}" "${start}" "${end}"
    python quantization/C2C/script/evaluation/unified_evaluator.py \
      --config "${run_root}/configs/${config_name}" 2>&1 | tee -a "${log_path}"
    echo "{\"start\":${start},\"end\":${end},\"total\":${total},\"timestamp\":\"$(date -Iseconds)\"}" >> "${chunks_log}"
    write_progress "${progress_file}" "${start}" "${end}" "${total}"
    start="${end}"
  done
}

# =============================================================================
# Milestone 0: Baseline evals (receiver-only, no quant).
# =============================================================================
if [ "${RUN_MILESTONE_0}" = "1" ]; then
  RUN_TAG="step0_${RUN_TAG_PREFIX}"
  python quantization/scripts/run_step0_baselines.py --mode gpu --prep-only --run-tag "${RUN_TAG}"
  RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_0_baselines/${RUN_TAG}"
  run_eval_chunks "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step0.log"
  run_eval_chunks "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step0.log"
fi

# =============================================================================
# Milestone 2: KV PTQ evals (INT8 + INT4).
# =============================================================================
if [ "${RUN_MILESTONE_2}" = "1" ]; then
  for SCHEME in int8 int4; do
    RUN_TAG="step1_${SCHEME}_${RUN_TAG_PREFIX}"
    python quantization/scripts/run_step1_kv_ptq.py --mode gpu --kv-quant-scheme "${SCHEME}" --prep-only --run-tag "${RUN_TAG}"
    RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${RUN_TAG}"
    run_eval_chunks "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step1.log"
    run_eval_chunks "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step1.log"
  done
fi

echo "Done."
