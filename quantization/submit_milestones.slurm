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

CONDA_ENV="${CONDA_ENV:-rosetta}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/projects/m000066/${USER}/conda/envs/${CONDA_ENV}}"
if [ ! -x "${CONDA_ENV_PATH}/bin/python" ] && [ -x "/workspace/conda/envs/${CONDA_ENV}/bin/python" ]; then
  CONDA_ENV_PATH="/workspace/conda/envs/${CONDA_ENV}"
fi
if [ -x "${CONDA_ENV_PATH}/bin/python" ]; then
  export CONDA_DEFAULT_ENV="${CONDA_ENV}"
  export CONDA_PREFIX="${CONDA_ENV_PATH}"
  PYTHON=( "${CONDA_ENV_PATH}/bin/python" )
  echo "Using conda env python: ${CONDA_ENV_PATH}/bin/python"
else
  CONDA_EXE="${CONDA_EXE:-}"
  if [ -z "${CONDA_EXE}" ]; then
    CONDA_EXE="$(command -v conda || true)"
  fi
  if [ -z "${CONDA_EXE}" ]; then
    echo "ERROR: conda not found in PATH. Load your conda module and retry." >&2
    exit 1
  fi
  PYTHON=( "${CONDA_EXE}" run --no-capture-output -n "${CONDA_ENV}" python )
  echo "Using conda run: ${CONDA_EXE} run --no-capture-output -n ${CONDA_ENV} python"
fi

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-${PWD}}}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}"

# Cache roots (prefer scratch on HPC, fallback to /workspace for RunPod/containers).
DEFAULT_CACHE_ROOT="/scratch/m000066/${USER}"
if [ ! -d "/scratch" ]; then
  DEFAULT_CACHE_ROOT="/workspace"
fi

# Caches on scratch/volume to avoid filling container root.
export HF_HOME="${HF_HOME:-${DEFAULT_CACHE_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export C2C_CKPT_ROOT="${C2C_CKPT_ROOT:-${DEFAULT_CACHE_ROOT}/c2c_checkpoints}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

# RunPod bootstrap checks (fail fast on missing pod setup).
RUNPOD_BOOTSTRAP="${RUNPOD_BOOTSTRAP:-1}"
if [ "${RUNPOD_BOOTSTRAP}" = "1" ] && [ -x "quantization/scripts/runpod_bootstrap.sh" ]; then
  RUNPOD_BOOTSTRAP_ARGS="${RUNPOD_BOOTSTRAP_ARGS:-}"
  bash quantization/scripts/runpod_bootstrap.sh ${RUNPOD_BOOTSTRAP_ARGS}
fi

# Guard: ensure HF auth is present when gated models are used (Llama/Gemma).
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set. Export HF_TOKEN or run 'huggingface-cli login' before launching." >&2
  exit 1
fi

# Milestone toggles (set to 1 when you want to run them).
RUN_MILESTONE_0="${RUN_MILESTONE_0:-0}"
RUN_MILESTONE_2="${RUN_MILESTONE_2:-0}"
RUN_MILESTONE_3="${RUN_MILESTONE_3:-0}"
RUN_MILESTONE_5="${RUN_MILESTONE_5:-0}"
RUN_MILESTONE_6="${RUN_MILESTONE_6:-0}"
RUN_MILESTONE_7="${RUN_MILESTONE_7:-0}"
RUN_MILESTONE_8="${RUN_MILESTONE_8:-0}"
RUN_MILESTONE_9="${RUN_MILESTONE_9:-0}"
RUN_MILESTONE_10="${RUN_MILESTONE_10:-0}"
RUN_FULL="${RUN_FULL:-0}"
PREFLIGHT="${PREFLIGHT:-1}"
DRY_RUN="${DRY_RUN:-0}"

# Chunked eval settings (resume at chunk boundaries).
CHUNK_SIZE="${CHUNK_SIZE:-200}"
EVAL_SMOKE="${EVAL_SMOKE:-0}"
SMOKE_LIMIT="${SMOKE_LIMIT:-50}"

# Skip/registry controls (avoid re-running completed configs).
SKIP_DONE="${SKIP_DONE:-1}"
FORCE_RERUN="${FORCE_RERUN:-0}"
REGISTRY_PATH="${REGISTRY_PATH:-${PROJECT_ROOT}/quantization/registry/run_registry.json}"
mkdir -p "$(dirname "${REGISTRY_PATH}")"

if [ "${RUN_FULL}" = "1" ]; then
  EVAL_SMOKE=0
fi

# Auto-push results after each milestone (full runs only by default).
AUTO_PUSH="${AUTO_PUSH:-0}"
AUTO_PUSH_FULL_ONLY="${AUTO_PUSH_FULL_ONLY:-1}"
AUTO_PUSH_REBASE="${AUTO_PUSH_REBASE:-1}"
AUTO_PUSH_REMOTE="${AUTO_PUSH_REMOTE:-origin}"
AUTO_PUSH_BRANCH="${AUTO_PUSH_BRANCH:-main}"

# Run tag prefix for deterministic folder names (stable across requeues).
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

# Milestone 3 grid (cache-length reduction).
M3_KV_QUANT_SCHEME="${M3_KV_QUANT_SCHEME:-int8}"
M3_PROPORTIONS="${M3_PROPORTIONS:-1.0 0.75 0.5 0.25 0.1}"
M3_ORDER_MODES="${M3_ORDER_MODES:-front back}"
PREFLIGHT_M3_PROPORTION="${PREFLIGHT_M3_PROPORTION:-0.5}"
PREFLIGHT_M3_ORDER_MODE="${PREFLIGHT_M3_ORDER_MODE:-front}"

# Milestone 5 (QAT smoke/full).
M5_RECIPE="${M5_RECIPE:-quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8_smoke.json}"

# Milestone 6 (mixed precision).
M6_KV_QUANT_SCHEME="${M6_KV_QUANT_SCHEME:-int8}"
M6_LAYER_SCHEDULE="${M6_LAYER_SCHEDULE:-quantization/configs/kv_layer_schedule.yaml}"
M6_LAYER_SCHEDULES="${M6_LAYER_SCHEDULES:-${M6_LAYER_SCHEDULE}}"

# Milestone 7 (heterogeneity).
M7_BASE_MODEL="${M7_BASE_MODEL:-Qwen/Qwen3-0.6B}"
M7_TEACHER_MODEL="${M7_TEACHER_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
M7_DO_ALIGNMENT="${M7_DO_ALIGNMENT:-1}"
M7_ALIGNMENT_STRATEGY="${M7_ALIGNMENT_STRATEGY:-}"
M7_ALIGNMENT_MODES="${M7_ALIGNMENT_MODES:-${M7_DO_ALIGNMENT}}"
M7_CHECKPOINT_DIR="${M7_CHECKPOINT_DIR:-}"
M7_REQUIRE_CHECKPOINT="${M7_REQUIRE_CHECKPOINT:-1}"

# Milestone 8 grid (selective transfer).
M8_KV_QUANT_SCHEME="${M8_KV_QUANT_SCHEME:-int8}"
M8_SELECT_MODES="${M8_SELECT_MODES:-proj_vnorm_topk}"
M8_SELECT_PROPORTIONS="${M8_SELECT_PROPORTIONS:-1.0 0.5 0.25 0.1}"
M8_SELECT_MIN_TOKENS="${M8_SELECT_MIN_TOKENS:-64}"
M8_SELECT_SCOPES="${M8_SELECT_SCOPES:-prompt}"
M8_SCATTER_FILL="${M8_SCATTER_FILL:-receiver_only}"
M8_SPARSE_FUSE="${M8_SPARSE_FUSE:-1}"
M8_SPARSE_FUSE_MODES="${M8_SPARSE_FUSE_MODES:-${M8_SPARSE_FUSE}}"
M8_EVAL_DATASETS="${M8_EVAL_DATASETS:-}"
M8_EVAL_LIMIT="${M8_EVAL_LIMIT:-}"
M8_EVAL_RANGE="${M8_EVAL_RANGE:-}"
PREFLIGHT_M8_PROPORTION="${PREFLIGHT_M8_PROPORTION:-0.25}"
PREFLIGHT_M8_MODE="${PREFLIGHT_M8_MODE:-proj_vnorm_topk}"

# Milestone 9 grid (delta selection).
M9_KV_QUANT_SCHEME="${M9_KV_QUANT_SCHEME:-int8}"
M9_BASE_MODEL="${M9_BASE_MODEL:-}"
M9_TEACHER_MODEL="${M9_TEACHER_MODEL:-}"
M9_DO_ALIGNMENT="${M9_DO_ALIGNMENT:-0}"
M9_ALIGNMENT_STRATEGY="${M9_ALIGNMENT_STRATEGY:-}"
M9_CHECKPOINT_DIR="${M9_CHECKPOINT_DIR:-}"
M9_SELECT_MODES="${M9_SELECT_MODES:-vnorm_topk proj_vnorm_topk delta_proj_vnorm_topk}"
M9_SELECT_PROPORTIONS="${M9_SELECT_PROPORTIONS:-1.0 0.5 0.25 0.10}"
M9_SELECT_MIN_TOKENS="${M9_SELECT_MIN_TOKENS:-1}"
M9_SELECT_SCOPES="${M9_SELECT_SCOPES:-prompt}"
M9_SCATTER_FILL="${M9_SCATTER_FILL:-receiver_only}"
M9_SPARSE_FUSE="${M9_SPARSE_FUSE:-1}"
M9_SPARSE_FUSE_MODES="${M9_SPARSE_FUSE_MODES:-${M9_SPARSE_FUSE}}"
M9_EXTRA_LOW_PROP="${M9_EXTRA_LOW_PROP:-}"
M9_EXTRA_LOW_MODE="${M9_EXTRA_LOW_MODE:-delta_proj_vnorm_topk}"
M9_TIMING_SYNC="${M9_TIMING_SYNC:-0}"
M9_EVAL_DATASETS="${M9_EVAL_DATASETS:-}"
M9_EVAL_LIMIT="${M9_EVAL_LIMIT:-}"
M9_EVAL_RANGE="${M9_EVAL_RANGE:-}"
M9_KEY_SUFFIX="${M9_KEY_SUFFIX:-}"

# Milestone 10 grid (RD token x precision).
M10_KV_QUANT_SCHEME="${M10_KV_QUANT_SCHEME:-int8}"
M10_KV_QUANT_AXIS="${M10_KV_QUANT_AXIS:-head}"
M10_BASE_MODEL="${M10_BASE_MODEL:-Qwen/Qwen3-0.6B}"
M10_TEACHER_MODEL="${M10_TEACHER_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
M10_DO_ALIGNMENT="${M10_DO_ALIGNMENT:-0}"
M10_ALIGNMENT_STRATEGY="${M10_ALIGNMENT_STRATEGY:-}"
M10_CHECKPOINT_DIR="${M10_CHECKPOINT_DIR:-}"
M10_SELECT_MODE="${M10_SELECT_MODE:-delta_proj_vnorm_topk}"
M10_SELECT_SCOPE="${M10_SELECT_SCOPE:-prompt}"
M10_SELECT_MIN_TOKENS="${M10_SELECT_MIN_TOKENS:-1}"
M10_SCATTER_FILL="${M10_SCATTER_FILL:-receiver_only}"
M10_TOKEN_PRECISION_CANDIDATES="${M10_TOKEN_PRECISION_CANDIDATES:-drop,int4,int8}"
M10_CANDIDATE_SETS="${M10_CANDIDATE_SETS:-}"
M10_BUDGET_FRACS="${M10_BUDGET_FRACS:-0.03125 0.0625 0.125 0.25}"
M10_BUDGET_BYTES="${M10_BUDGET_BYTES:-}"
M10_AVG_INPUT_LEN="${M10_AVG_INPUT_LEN:-256}"
M10_INDEX_BYTES="${M10_INDEX_BYTES:-2}"
M10_INCLUDE_SCALE_OVERHEAD="${M10_INCLUDE_SCALE_OVERHEAD:-1}"
M10_TIMING_SYNC="${M10_TIMING_SYNC:-0}"
M10_EVAL_DATASETS="${M10_EVAL_DATASETS:-}"
M10_EVAL_LIMIT="${M10_EVAL_LIMIT:-}"
M10_EVAL_RANGE="${M10_EVAL_RANGE:-}"
M10_KEY_SUFFIX="${M10_KEY_SUFFIX:-}"

STEP8_EVAL_DATASETS="${STEP8_EVAL_DATASETS:-openbookqa,ai2-arc}"

handle_signal() {
  echo "Received preemption signal. Job will exit and can be requeued."
}
trap handle_signal USR1 TERM

normalize_dataset_list() {
  echo "$1" | tr ',' ' '
}

dataset_slug() {
  if [ "$1" = "ai2-arc" ]; then
    echo "arc_c"
  else
    echo "$1" | tr '/-' '__' | tr -cd '[:alnum:]_'
  fi
}

sanitize_candidates() {
  echo "$1" | tr ',' '-' | tr -cd '[:alnum:]-'
}

dataset_len() {
  "${PYTHON[@]}" - <<'PY' "$1"
import sys
from datasets import load_dataset
from datasets.utils import logging as ds_logging
from script.evaluation.unified_evaluator import DATASET_CONFIGS

ds_logging.disable_progress_bar()
ds_logging.set_verbosity_error()

dataset = sys.argv[1]
revision = "main"
if dataset == "openbookqa":
    ds = load_dataset("allenai/openbookqa", revision=revision)
    split = ds["test"]
    sys.stdout.write(str(len(split)))
    raise SystemExit(0)
if dataset == "ai2-arc":
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", revision=revision)
    split = ds["test"]
    sys.stdout.write(str(len(split)))
    raise SystemExit(0)

cfg = DATASET_CONFIGS.get(dataset)
if not cfg:
    raise SystemExit(f"Unknown dataset: {dataset}")
name = cfg.get("dataset_name")
split_name = cfg.get("test_split", "test")
subjects = cfg.get("subjects") or []
if subjects:
    total = 0
    for subject in subjects:
        try:
            ds = load_dataset(name, subject)
        except Exception:
            ds = load_dataset(name)
        total += len(ds[split_name])
    sys.stdout.write(str(total))
else:
    ds = load_dataset(name)
    sys.stdout.write(str(len(ds[split_name])))
PY
}

patch_limit() {
  "${PYTHON[@]}" - <<'PY' "$1" "$2" "$3"
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
  "${PYTHON[@]}" - <<'PY' "$1" "$2" "$3" "$4"
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
  "${PYTHON[@]}" - <<'PY' "$1"
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

registry_should_skip() {
  local key="$1"
  local dataset="$2"
  local smoke="$3"
  "${PYTHON[@]}" - <<'PY' "$REGISTRY_PATH" "$key" "$dataset" "$smoke" "$SKIP_DONE" "$FORCE_RERUN"
import json
import sys
from pathlib import Path

registry_path = Path(sys.argv[1])
key = sys.argv[2]
dataset = sys.argv[3]
smoke = sys.argv[4] == "1"
skip_done = sys.argv[5] == "1"
force = sys.argv[6] == "1"

if force or not skip_done:
    print("0")
    raise SystemExit(0)

if not registry_path.exists():
    print("0")
    raise SystemExit(0)

data = json.loads(registry_path.read_text())
entry = data.get(key, {}).get(dataset)
if not entry:
    print("0")
    raise SystemExit(0)

status = entry.get("status")
entry_smoke = bool(entry.get("smoke", False))
if status != "complete":
    print("0")
    raise SystemExit(0)

if smoke:
    # Skip smokes if any complete run already exists.
    print("1")
    raise SystemExit(0)

# Full runs: skip only if a full (non-smoke) run exists.
print("1" if not entry_smoke else "0")
PY
}

registry_record() {
  local key="$1"
  local dataset="$2"
  local status="$3"
  local run_root="$4"
  local smoke="$5"
  local config_hash="$6"
  local reason="${7:-}"
  "${PYTHON[@]}" - <<'PY' "$REGISTRY_PATH" "$key" "$dataset" "$status" "$run_root" "$smoke" "$config_hash" "$reason"
import json
import sys
from datetime import datetime
from pathlib import Path

registry_path = Path(sys.argv[1])
key = sys.argv[2]
dataset = sys.argv[3]
status = sys.argv[4]
run_root = sys.argv[5]
smoke = sys.argv[6] == "1"
config_hash = sys.argv[7]
reason = sys.argv[8]

data = {}
if registry_path.exists():
    try:
        data = json.loads(registry_path.read_text())
    except Exception:
        data = {}

data.setdefault(key, {})
data[key][dataset] = {
    "status": status,
    "smoke": smoke,
    "run_root": run_root,
    "config_hash": config_hash,
    "reason": reason,
    "timestamp": datetime.utcnow().isoformat() + "Z",
}

registry_path.write_text(json.dumps(data, indent=2) + "\n")
PY
}

config_hash() {
  local cfg_path="$1"
  "${PYTHON[@]}" - <<'PY' "$cfg_path"
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
print(hashlib.sha1(data).hexdigest())
PY
}

git_autopush() {
  local label="$1"
  if [ "${AUTO_PUSH}" != "1" ]; then
    return
  fi
  if [ "${AUTO_PUSH_FULL_ONLY}" = "1" ] && [ "${RUN_FULL}" != "1" ]; then
    return
  fi
  if [ ! -d "${PROJECT_ROOT}/.git" ]; then
    echo "Auto-push skipped: no git repo at ${PROJECT_ROOT}"
    return
  fi
  (
    set +e
    cd "${PROJECT_ROOT}" || exit 0
    git add quantization/data/step_* quantization/registry/run_registry.json quantization/golden 2>/dev/null
    if git diff --cached --quiet; then
      echo "Auto-push: no new data to commit for ${label}"
      exit 0
    fi
    git commit -m "Add results ${label} ${RUN_TAG_PREFIX}" >/dev/null 2>&1
    if [ "${AUTO_PUSH_REBASE}" = "1" ]; then
      git pull --rebase "${AUTO_PUSH_REMOTE}" "${AUTO_PUSH_BRANCH}"
      if [ $? -ne 0 ]; then
        git rebase --abort >/dev/null 2>&1 || true
        echo "Auto-push: rebase failed for ${label}"
        exit 0
      fi
    fi
    git push "${AUTO_PUSH_REMOTE}" "${AUTO_PUSH_BRANCH}"
    if [ $? -ne 0 ]; then
      echo "Auto-push: push failed for ${label}"
    fi
  )
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
  if ! [[ "${total}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: dataset_len returned non-numeric value for ${dataset}: '${total}'" >&2
    exit 1
  fi

  if [ "${EVAL_SMOKE}" = "1" ]; then
    local end="${SMOKE_LIMIT}"
    if [ "${end}" -gt "${total}" ]; then
      end="${total}"
    fi
    echo "Smoke eval ${dataset}: [0, ${end}) of ${total}"
    patch_limit "${run_root}/configs/${config_name}" 0 "${end}"
    "${PYTHON[@]}" quantization/C2C/script/evaluation/unified_evaluator.py \
      --config "${run_root}/configs/${config_name}" 2>&1 | tee -a "${log_path}"
    echo "{\"start\":0,\"end\":${end},\"total\":${total},\"timestamp\":\"$(date -Iseconds)\"}" >> "${chunks_log}"
    return
  fi

  local start
  start="$(read_next_start "${progress_file}")"
  if [ "${start}" -ge "${total}" ]; then
    echo "Resume: ${dataset} already complete (start=${start}, total=${total})."
    patch_limit "${run_root}/configs/${config_name}" 0 "${total}"
    return
  fi

  while [ "${start}" -lt "${total}" ]; do
    local end=$((start + CHUNK_SIZE))
    if [ "${end}" -gt "${total}" ]; then
      end="${total}"
    fi
    echo "Eval ${dataset}: [${start}, ${end}) of ${total}"
    patch_limit "${run_root}/configs/${config_name}" "${start}" "${end}"
    "${PYTHON[@]}" quantization/C2C/script/evaluation/unified_evaluator.py \
      --config "${run_root}/configs/${config_name}" 2>&1 | tee -a "${log_path}"
    echo "{\"start\":${start},\"end\":${end},\"total\":${total},\"timestamp\":\"$(date -Iseconds)\"}" >> "${chunks_log}"
    write_progress "${progress_file}" "${start}" "${end}" "${total}"
    start="${end}"
  done

  # Reset limit after completion so manual re-runs don't stay pinned to the last chunk.
  patch_limit "${run_root}/configs/${config_name}" 0 "${total}"
}

run_eval_chunks_with_skip() {
  local run_root="$1"
  local config_name="$2"
  local dataset="$3"
  local log_path="$4"
  local registry_key="$5"

  local progress_dir="${run_root}/progress"
  local progress_file="${progress_dir}/${dataset}.json"
  mkdir -p "${progress_dir}"

  local total
  total="$(dataset_len "${dataset}")"
  if ! [[ "${total}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: dataset_len returned non-numeric value for ${dataset}: '${total}'" >&2
    exit 1
  fi

  if [ -f "${progress_file}" ]; then
    local start
    start="$(read_next_start "${progress_file}")"
    if [ "${start}" -lt "${total}" ]; then
      run_eval_chunks "${run_root}" "${config_name}" "${dataset}" "${log_path}"
      local cfg_hash
      cfg_hash="$(config_hash "${run_root}/configs/${config_name}")"
      registry_record "${registry_key}" "${dataset}" "complete" "${run_root}" "${EVAL_SMOKE}" "${cfg_hash}"
      return
    fi
  fi

  local skip
  skip="$(registry_should_skip "${registry_key}" "${dataset}" "${EVAL_SMOKE}")"
  if [ "${skip}" = "1" ]; then
    local cfg_hash
    cfg_hash="$(config_hash "${run_root}/configs/${config_name}")"
    registry_record "${registry_key}" "${dataset}" "skipped" "${run_root}" "${EVAL_SMOKE}" "${cfg_hash}" "skip_done"
    echo "Skipping ${dataset} for ${registry_key} (registry says complete)."
    return
  fi

  run_eval_chunks "${run_root}" "${config_name}" "${dataset}" "${log_path}"
  local cfg_hash
  cfg_hash="$(config_hash "${run_root}/configs/${config_name}")"
  registry_record "${registry_key}" "${dataset}" "complete" "${run_root}" "${EVAL_SMOKE}" "${cfg_hash}"
}

sanitize_prop() {
  echo "$1" | sed 's/\./p/g'
}

sanitize_frac() {
  echo "$1" | sed 's/\//over/g' | sed 's/\./p/g'
}

compute_rd_budget_bytes() {
  local frac="$1"
  local avg_len="$2"
  local base_model="$3"
  local index_bytes="$4"
  local axis="$5"
  local include_scale="$6"
  local kv_scheme="$7"
  "${PYTHON[@]}" - <<'PY' "${frac}" "${avg_len}" "${base_model}" "${index_bytes}" "${axis}" "${include_scale}" "${kv_scheme}"
import sys
from transformers import AutoConfig

frac = float(sys.argv[1])
avg_len = float(sys.argv[2])
base_model = sys.argv[3]
index_bytes = float(sys.argv[4])
axis = sys.argv[5]
include_scale = sys.argv[6] == "1"
scheme = sys.argv[7].lower()

cfg = AutoConfig.from_pretrained(base_model)
num_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
head_dim = cfg.hidden_size // cfg.num_attention_heads

if scheme in ("int4", "nf4"):
    bytes_per_elem = 0.5
elif scheme in ("int8", "fp8"):
    bytes_per_elem = 1.0
else:
    bytes_per_elem = 2.0

payload = 2 * num_heads * head_dim * bytes_per_elem
full_layer = avg_len * (payload + index_bytes)
if include_scale:
    if axis == "head":
        full_layer += 2 * num_heads * 4
    else:
        full_layer += 2 * 4

budget = full_layer * frac
print(f"{budget:.6f}")
PY
}

preflight_cleanup() {
  local run_root="$1"
  if [ -d "${run_root}" ]; then
    rm -rf "${run_root}"
  fi
}

run_preflight() {
  if [ "${PREFLIGHT}" != "1" ]; then
    return
  fi

  echo "Preflight: running prep-only to validate plumbing."

  if [ "${RUN_MILESTONE_0}" = "1" ]; then
    local pre_tag="preflight_step0_${RUN_TAG_PREFIX}"
    "${PYTHON[@]}" quantization/scripts/run_step0_baselines.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --prep-only \
      --run-tag "${pre_tag}"
    preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_0_baselines/${pre_tag}"
  fi

  if [ "${RUN_MILESTONE_2}" = "1" ]; then
    for SCHEME in int8 int4; do
      local pre_tag="preflight_step1_${SCHEME}_${RUN_TAG_PREFIX}"
      "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
        --env "${CONDA_ENV}" \
        --mode gpu \
        --kv-quant-scheme "${SCHEME}" \
        --prep-only \
        --run-tag "${pre_tag}"
      preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${pre_tag}"
    done
  fi

  if [ "${RUN_MILESTONE_3}" = "1" ]; then
    local pre_prop="${PREFLIGHT_M3_PROPORTION}"
    local pre_order="${PREFLIGHT_M3_ORDER_MODE}"
    local pre_prop_tag
    pre_prop_tag="$(sanitize_prop "${pre_prop}")"
    local pre_tag="preflight_step3_p${pre_prop_tag}_${pre_order}_${RUN_TAG_PREFIX}"
    "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${M3_KV_QUANT_SCHEME}" \
      --kv-cache-proportion "${pre_prop}" \
      --kv-cache-order-mode "${pre_order}" \
      --prep-only \
      --run-tag "${pre_tag}"
    preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${pre_tag}"
  fi

  if [ "${RUN_MILESTONE_6}" = "1" ]; then
    local pre_tag="preflight_step6_${RUN_TAG_PREFIX}"
    "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${M6_KV_QUANT_SCHEME}" \
      --kv-quant-layer-schedule "${M6_LAYER_SCHEDULE}" \
      --prep-only \
      --run-tag "${pre_tag}"
    preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${pre_tag}"
  fi

  if [ "${RUN_MILESTONE_7}" = "1" ]; then
    local m7_pre_args=()
    if [ -n "${M7_CHECKPOINT_DIR}" ]; then
      m7_pre_args+=( "--checkpoint-dir-override" "${M7_CHECKPOINT_DIR}" )
    elif [ "${M7_REQUIRE_CHECKPOINT}" = "1" ]; then
      echo "ERROR: M7_CHECKPOINT_DIR not set for heterogeneity preflight. Provide a hetero fuser checkpoint or set M7_REQUIRE_CHECKPOINT=0 to force default (not recommended)." >&2
      exit 1
    fi
    local pre_tag="preflight_step7_${RUN_TAG_PREFIX}"
    "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --base-model-override "${M7_BASE_MODEL}" \
      --teacher-model-override "${M7_TEACHER_MODEL}" \
      --kv-quant-scheme int8 \
      --prep-only \
      --run-tag "${pre_tag}" \
      "${m7_pre_args[@]}"
    preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${pre_tag}"
  fi

  if [ "${RUN_MILESTONE_8}" = "1" ]; then
    local pre_prop="${PREFLIGHT_M8_PROPORTION}"
    local pre_mode="${PREFLIGHT_M8_MODE}"
    local pre_scope="${M8_SELECT_SCOPES}"
    if [ -n "${pre_scope}" ]; then
      pre_scope="${pre_scope%% *}"
    else
      pre_scope="prompt"
    fi
    local pre_sparse="${M8_SPARSE_FUSE_MODES}"
    if [ -n "${pre_sparse}" ]; then
      pre_sparse="${pre_sparse%% *}"
    else
      pre_sparse="${M8_SPARSE_FUSE}"
    fi
    local pre_prop_tag
    pre_prop_tag="$(sanitize_prop "${pre_prop}")"
    local pre_tag="preflight_step8_${pre_mode}_p${pre_prop_tag}_${pre_scope}_sf${pre_sparse}_${RUN_TAG_PREFIX}"
    local pre_extra_args=()
    if [ "${pre_sparse}" = "0" ]; then
      pre_extra_args+=( "--no-kv-sparse-fuse" )
    fi
    if [ -n "${M8_EVAL_LIMIT}" ]; then
      pre_extra_args+=( "--eval-limit" "${M8_EVAL_LIMIT}" )
    fi
    if [ -n "${M8_EVAL_RANGE}" ]; then
      pre_extra_args+=( "--eval-range" "${M8_EVAL_RANGE}" )
    fi
    "${PYTHON[@]}" quantization/scripts/run_step8_selective_transfer.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${M8_KV_QUANT_SCHEME}" \
      --kv-select-mode "${pre_mode}" \
      --kv-select-proportion "${pre_prop}" \
      --kv-select-scope "${pre_scope}" \
      --kv-select-min-tokens "${M8_SELECT_MIN_TOKENS}" \
      --kv-scatter-fill "${M8_SCATTER_FILL}" \
      --eval-datasets "${M8_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}" \
      --prep-only \
      --run-tag "${pre_tag}" \
      "${pre_extra_args[@]}"
    preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_8_selective_transfer/${pre_tag}"
  fi

  if [ "${RUN_MILESTONE_9}" = "1" ]; then
    local pre_prop="${M9_SELECT_PROPORTIONS}"
    if [ -n "${pre_prop}" ]; then
      pre_prop="${pre_prop%% *}"
    else
      pre_prop="0.25"
    fi
    local pre_mode="${M9_SELECT_MODES}"
    if [ -n "${pre_mode}" ]; then
      pre_mode="${pre_mode%% *}"
    else
      pre_mode="delta_proj_vnorm_topk"
    fi
    local pre_scope="${M9_SELECT_SCOPES}"
    if [ -n "${pre_scope}" ]; then
      pre_scope="${pre_scope%% *}"
    else
      pre_scope="prompt"
    fi
    local pre_sparse="${M9_SPARSE_FUSE_MODES}"
    if [ -n "${pre_sparse}" ]; then
      pre_sparse="${pre_sparse%% *}"
    else
      pre_sparse="${M9_SPARSE_FUSE}"
    fi
    local pre_prop_tag
    pre_prop_tag="$(sanitize_prop "${pre_prop}")"
    local pre_tag="preflight_step9_${pre_mode}_p${pre_prop_tag}_${pre_scope}_sf${pre_sparse}_${RUN_TAG_PREFIX}"
    local pre_extra_args=()
    if [ "${pre_sparse}" = "0" ]; then
      pre_extra_args+=( "--no-kv-sparse-fuse" )
    fi
    if [ -n "${M9_BASE_MODEL}" ]; then
      pre_extra_args+=( "--base-model-override" "${M9_BASE_MODEL}" )
    fi
    if [ -n "${M9_TEACHER_MODEL}" ]; then
      pre_extra_args+=( "--teacher-model-override" "${M9_TEACHER_MODEL}" )
    fi
    if [ -n "${M9_CHECKPOINT_DIR}" ]; then
      pre_extra_args+=( "--checkpoint-dir-override" "${M9_CHECKPOINT_DIR}" )
    fi
    if [ "${M9_DO_ALIGNMENT}" = "1" ]; then
      pre_extra_args+=( "--do-alignment" )
    fi
    if [ -n "${M9_ALIGNMENT_STRATEGY}" ]; then
      pre_extra_args+=( "--alignment-strategy" "${M9_ALIGNMENT_STRATEGY}" )
    fi
    if [ "${M9_TIMING_SYNC}" = "1" ]; then
      pre_extra_args+=( "--kv-timing-sync" )
    fi
    if [ -n "${M9_EVAL_LIMIT}" ]; then
      pre_extra_args+=( "--eval-limit" "${M9_EVAL_LIMIT}" )
    fi
    if [ -n "${M9_EVAL_RANGE}" ]; then
      pre_extra_args+=( "--eval-range" "${M9_EVAL_RANGE}" )
    fi
    "${PYTHON[@]}" quantization/scripts/run_step8_selective_transfer.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${M9_KV_QUANT_SCHEME}" \
      --kv-select-mode "${pre_mode}" \
      --kv-select-proportion "${pre_prop}" \
      --kv-select-scope "${pre_scope}" \
      --kv-select-min-tokens "${M9_SELECT_MIN_TOKENS}" \
      --kv-scatter-fill "${M9_SCATTER_FILL}" \
      --eval-datasets "${M9_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}" \
      --prep-only \
      --run-tag "${pre_tag}" \
      "${pre_extra_args[@]}"
    preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_8_selective_transfer/${pre_tag}"
  fi

  if [ "${RUN_MILESTONE_10}" = "1" ]; then
    local pre_budget=""
    if [ -n "${M10_BUDGET_BYTES}" ]; then
      pre_budget="${M10_BUDGET_BYTES%% *}"
    else
      local pre_frac="${M10_BUDGET_FRACS}"
      if [ -n "${pre_frac}" ]; then
        pre_frac="${pre_frac%% *}"
      else
        pre_frac="0.125"
      fi
      pre_budget="$(compute_rd_budget_bytes "${pre_frac}" "${M10_AVG_INPUT_LEN}" "${M10_BASE_MODEL}" "${M10_INDEX_BYTES}" "${M10_KV_QUANT_AXIS}" "${M10_INCLUDE_SCALE_OVERHEAD}" "${M10_KV_QUANT_SCHEME}")"
    fi
    local pre_tag="preflight_step10_rd_${RUN_TAG_PREFIX}"
    local pre_extra_args=()
    if [ -n "${M10_BASE_MODEL}" ]; then
      pre_extra_args+=( "--base-model-override" "${M10_BASE_MODEL}" )
    fi
    if [ -n "${M10_TEACHER_MODEL}" ]; then
      pre_extra_args+=( "--teacher-model-override" "${M10_TEACHER_MODEL}" )
    fi
    if [ -n "${M10_CHECKPOINT_DIR}" ]; then
      pre_extra_args+=( "--checkpoint-dir-override" "${M10_CHECKPOINT_DIR}" )
    fi
    if [ "${M10_DO_ALIGNMENT}" = "1" ]; then
      pre_extra_args+=( "--do-alignment" )
    fi
    if [ -n "${M10_ALIGNMENT_STRATEGY}" ]; then
      pre_extra_args+=( "--alignment-strategy" "${M10_ALIGNMENT_STRATEGY}" )
    fi
    if [ "${M10_INCLUDE_SCALE_OVERHEAD}" = "1" ]; then
      pre_extra_args+=( "--kv-include-scale-overhead" )
    fi
    if [ "${M10_TIMING_SYNC}" = "1" ]; then
      pre_extra_args+=( "--kv-timing-sync" )
    fi
    if [ -n "${M10_EVAL_LIMIT}" ]; then
      pre_extra_args+=( "--eval-limit" "${M10_EVAL_LIMIT}" )
    fi
    if [ -n "${M10_EVAL_RANGE}" ]; then
      pre_extra_args+=( "--eval-range" "${M10_EVAL_RANGE}" )
    fi
    "${PYTHON[@]}" quantization/scripts/run_step8_selective_transfer.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${M10_KV_QUANT_SCHEME}" \
      --kv-quant-axis "${M10_KV_QUANT_AXIS}" \
      --kv-select-mode "${M10_SELECT_MODE}" \
      --kv-select-proportion "1.0" \
      --kv-select-scope "${M10_SELECT_SCOPE}" \
      --kv-select-min-tokens "${M10_SELECT_MIN_TOKENS}" \
      --kv-scatter-fill "${M10_SCATTER_FILL}" \
      --token-precision-mode rd_greedy \
      --token-precision-candidates "${M10_TOKEN_PRECISION_CANDIDATES}" \
      --token-precision-budget-bytes "${pre_budget}" \
      --token-precision-scope "${M10_SELECT_SCOPE}" \
      --kv-index-dtype-bytes "${M10_INDEX_BYTES}" \
      --eval-datasets "${M10_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}" \
      --prep-only \
      --run-tag "${pre_tag}" \
      "${pre_extra_args[@]}"
    preflight_cleanup "${PROJECT_ROOT}/quantization/data/step_8_selective_transfer/${pre_tag}"
  fi

  if [ "${DRY_RUN}" = "1" ]; then
    echo "DRY_RUN=1: preflight complete; exiting before evaluation."
    exit 0
  fi
}

run_preflight

# =============================================================================
# Milestone 0: Baseline evals (receiver-only, no quant).
# =============================================================================
if [ "${RUN_MILESTONE_0}" = "1" ]; then
  RUN_TAG="step0_${RUN_TAG_PREFIX}"
  "${PYTHON[@]}" quantization/scripts/run_step0_baselines.py \
    --env "${CONDA_ENV}" \
    --mode gpu \
    --prep-only \
    --run-tag "${RUN_TAG}"
  RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_0_baselines/${RUN_TAG}"
  run_eval_chunks_with_skip "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step0.log" "m0"
  run_eval_chunks_with_skip "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step0.log" "m0"
  git_autopush "m0"
fi

# =============================================================================
# Milestone 2: KV PTQ evals (INT8 + INT4).
# =============================================================================
if [ "${RUN_MILESTONE_2}" = "1" ]; then
  for SCHEME in int8 int4; do
    RUN_TAG="step1_${SCHEME}_${RUN_TAG_PREFIX}"
    "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${SCHEME}" \
      --prep-only \
      --run-tag "${RUN_TAG}"
    RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${RUN_TAG}"
    run_eval_chunks_with_skip "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step1.log" "m2|${SCHEME}"
    run_eval_chunks_with_skip "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step1.log" "m2|${SCHEME}"
  done
  git_autopush "m2"
fi

# =============================================================================
# Milestone 3: Cache-length reduction (INT8 + proportion/order grid).
# =============================================================================
if [ "${RUN_MILESTONE_3}" = "1" ]; then
  for PROP in ${M3_PROPORTIONS}; do
    for ORDER in ${M3_ORDER_MODES}; do
      PROP_TAG="$(sanitize_prop "${PROP}")"
      RUN_TAG="step3_${M3_KV_QUANT_SCHEME}_p${PROP_TAG}_${ORDER}_${RUN_TAG_PREFIX}"
      "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
        --env "${CONDA_ENV}" \
        --mode gpu \
        --kv-quant-scheme "${M3_KV_QUANT_SCHEME}" \
        --kv-cache-proportion "${PROP}" \
        --kv-cache-order-mode "${ORDER}" \
        --prep-only \
        --run-tag "${RUN_TAG}"
      RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${RUN_TAG}"
      run_eval_chunks_with_skip "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step1.log" "m3|${M3_KV_QUANT_SCHEME}|${PROP}|${ORDER}"
      run_eval_chunks_with_skip "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step1.log" "m3|${M3_KV_QUANT_SCHEME}|${PROP}|${ORDER}"
    done
  done
  git_autopush "m3"
fi

# =============================================================================
# Milestone 5: QAT smoke/full (training).
# =============================================================================
if [ "${RUN_MILESTONE_5}" = "1" ]; then
  M5_RUN_TAG="step5_qat_${RUN_TAG_PREFIX}"
  M5_RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_5_qat/${M5_RUN_TAG}"
  M5_OUTPUT_DIR="${M5_OUTPUT_DIR:-${C2C_CKPT_ROOT}/qat_${RUN_TAG_PREFIX}}"
  mkdir -p "${M5_RUN_ROOT}/configs" "${M5_RUN_ROOT}/logs" "${M5_RUN_ROOT}/manifests"

  M5_RECIPE_PATH="${M5_RECIPE}"
  if [ "${RUN_FULL}" = "1" ] && [ "${M5_RECIPE_PATH}" = "quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8_smoke.json" ]; then
    M5_RECIPE_PATH="quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8.json"
  fi

  if [ ! -f "${M5_RECIPE_PATH}" ]; then
    if [ -f "quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8.json" ]; then
      echo "WARN: M5_RECIPE missing (${M5_RECIPE_PATH}); falling back to C2C_0.6+0.5_qat_int8.json"
      M5_RECIPE_PATH="quantization/C2C/recipe/train_recipe/C2C_0.6+0.5_qat_int8.json"
    else
      echo "ERROR: M5_RECIPE not found: ${M5_RECIPE_PATH}" >&2
      exit 1
    fi
  fi

  cp "${M5_RECIPE_PATH}" "${M5_RUN_ROOT}/configs/m5_train.json"
  "${PYTHON[@]}" - <<'PY' "${M5_RUN_ROOT}/configs/m5_train.json" "${M5_OUTPUT_DIR}"
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
out_dir = sys.argv[2]

data = json.loads(cfg_path.read_text())
data.setdefault("output", {})
data["output"]["output_dir"] = out_dir
cfg_path.write_text(json.dumps(data, indent=2) + "\n")
print(f"Wrote M5 recipe: {cfg_path}")
PY

  "${PYTHON[@]}" quantization/C2C/script/train/SFT_train.py \
    --config "${M5_RUN_ROOT}/configs/m5_train.json" 2>&1 | tee "${M5_RUN_ROOT}/logs/m5_train.log"

  "${PYTHON[@]}" - <<'PY' "${M5_RUN_ROOT}/manifests/train_manifest.json" "${M5_OUTPUT_DIR}" "${M5_RUN_ROOT}/configs/m5_train.json"
import json
import sys
from datetime import datetime

manifest_path = sys.argv[1]
output_dir = sys.argv[2]
config_path = sys.argv[3]
manifest = {
    "output_dir": output_dir,
    "config_path": config_path,
    "timestamp": datetime.utcnow().isoformat() + "Z",
}
open(manifest_path, "w").write(json.dumps(manifest, indent=2) + "\\n")
PY

  M5_EVAL_TAG="step5_qat_eval_${RUN_TAG_PREFIX}"
  "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
    --env "${CONDA_ENV}" \
    --mode gpu \
    --kv-quant-scheme int8 \
    --checkpoint-dir-override "${M5_OUTPUT_DIR}" \
    --run-tag "${M5_EVAL_TAG}"
  RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${M5_EVAL_TAG}"
  run_eval_chunks_with_skip "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step1.log" "m5|qat|int8"
  run_eval_chunks_with_skip "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step1.log" "m5|qat|int8"
  git_autopush "m5"
fi

# =============================================================================
# Milestone 6: Mixed precision (layer schedule).
# =============================================================================
if [ "${RUN_MILESTONE_6}" = "1" ]; then
  for SCHEDULE_PATH in ${M6_LAYER_SCHEDULES}; do
    schedule_tag="$(basename "${SCHEDULE_PATH}")"
    schedule_tag="${schedule_tag%.*}"
    RUN_TAG="step6_${M6_KV_QUANT_SCHEME}_${schedule_tag}_${RUN_TAG_PREFIX}"
    "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${M6_KV_QUANT_SCHEME}" \
      --kv-quant-layer-schedule "${SCHEDULE_PATH}" \
      --prep-only \
      --run-tag "${RUN_TAG}"
    RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${RUN_TAG}"
    schedule_hash="$(config_hash "${SCHEDULE_PATH}")"
    run_eval_chunks_with_skip "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step1.log" "m6|${M6_KV_QUANT_SCHEME}|${schedule_hash}"
    run_eval_chunks_with_skip "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step1.log" "m6|${M6_KV_QUANT_SCHEME}|${schedule_hash}"
  done
  git_autopush "m6"
fi

# =============================================================================
# Milestone 7: Heterogeneity (model pair swap).
# =============================================================================
if [ "${RUN_MILESTONE_7}" = "1" ]; then
  for ALIGN_MODE in ${M7_ALIGNMENT_MODES}; do
    RUN_TAG="step7_align${ALIGN_MODE}_${RUN_TAG_PREFIX}"
    EXTRA_ARGS=()
    if [ -n "${M7_CHECKPOINT_DIR}" ]; then
      EXTRA_ARGS+=( "--checkpoint-dir-override" "${M7_CHECKPOINT_DIR}" )
    elif [ "${M7_REQUIRE_CHECKPOINT}" = "1" ]; then
      echo "ERROR: M7_CHECKPOINT_DIR not set for heterogeneity. Provide a hetero fuser checkpoint or set M7_REQUIRE_CHECKPOINT=0 to force default (not recommended)." >&2
      exit 1
    fi
    if [ "${ALIGN_MODE}" = "1" ]; then
      EXTRA_ARGS+=( "--do-alignment" )
    fi
    if [ -n "${M7_ALIGNMENT_STRATEGY}" ]; then
      EXTRA_ARGS+=( "--alignment-strategy" "${M7_ALIGNMENT_STRATEGY}" )
    fi
    "${PYTHON[@]}" quantization/scripts/run_step1_kv_ptq.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --base-model-override "${M7_BASE_MODEL}" \
      --teacher-model-override "${M7_TEACHER_MODEL}" \
      --kv-quant-scheme int8 \
      --prep-only \
      --run-tag "${RUN_TAG}" \
      "${EXTRA_ARGS[@]}"
    RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_1_kv_ptq/${RUN_TAG}"
    run_eval_chunks_with_skip "${RUN_ROOT}" "openbookqa.yaml" "openbookqa" "${RUN_ROOT}/logs/step1.log" "m7|${M7_BASE_MODEL}|${M7_TEACHER_MODEL}|align${ALIGN_MODE}"
    run_eval_chunks_with_skip "${RUN_ROOT}" "arc_c.yaml" "ai2-arc" "${RUN_ROOT}/logs/step1.log" "m7|${M7_BASE_MODEL}|${M7_TEACHER_MODEL}|align${ALIGN_MODE}"
  done
  git_autopush "m7"
fi

# =============================================================================
# Milestone 8: Selective transfer (token-level sparsity + quantization).
# =============================================================================
if [ "${RUN_MILESTONE_8}" = "1" ]; then
  DATASETS="$(normalize_dataset_list "${M8_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}")"
  for MODE in ${M8_SELECT_MODES}; do
    for PROP in ${M8_SELECT_PROPORTIONS}; do
      for SCOPE in ${M8_SELECT_SCOPES}; do
        for SPARSE_FUSE_MODE in ${M8_SPARSE_FUSE_MODES}; do
          PROP_TAG="$(sanitize_prop "${PROP}")"
          scope_tag="${SCOPE}"
          sf_tag="sf${SPARSE_FUSE_MODE}"
          RUN_TAG="step8_${M8_KV_QUANT_SCHEME}_${MODE}_p${PROP_TAG}_${scope_tag}_${sf_tag}_${RUN_TAG_PREFIX}"
          EXTRA_ARGS=()
          if [ "${SPARSE_FUSE_MODE}" = "0" ]; then
            EXTRA_ARGS+=( "--no-kv-sparse-fuse" )
          fi
          if [ -n "${M8_EVAL_LIMIT}" ]; then
            EXTRA_ARGS+=( "--eval-limit" "${M8_EVAL_LIMIT}" )
          fi
          if [ -n "${M8_EVAL_RANGE}" ]; then
            EXTRA_ARGS+=( "--eval-range" "${M8_EVAL_RANGE}" )
          fi
          "${PYTHON[@]}" quantization/scripts/run_step8_selective_transfer.py \
            --env "${CONDA_ENV}" \
            --mode gpu \
            --kv-quant-scheme "${M8_KV_QUANT_SCHEME}" \
            --kv-select-mode "${MODE}" \
            --kv-select-proportion "${PROP}" \
            --kv-select-scope "${SCOPE}" \
            --kv-select-min-tokens "${M8_SELECT_MIN_TOKENS}" \
            --kv-scatter-fill "${M8_SCATTER_FILL}" \
            --eval-datasets "${M8_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}" \
            --prep-only \
            --run-tag "${RUN_TAG}" \
            "${EXTRA_ARGS[@]}"
          RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_8_selective_transfer/${RUN_TAG}"
          for DATASET in ${DATASETS}; do
            SLUG="$(dataset_slug "${DATASET}")"
            run_eval_chunks_with_skip "${RUN_ROOT}" "${SLUG}.yaml" "${DATASET}" "${RUN_ROOT}/logs/step8.log" "m8|${M8_KV_QUANT_SCHEME}|${MODE}|${PROP}|${SCOPE}|${SPARSE_FUSE_MODE}"
          done
        done
      done
    done
  done
  git_autopush "m8"
fi

# =============================================================================
# Milestone 9: Delta selection (receiver-aware token scoring).
# =============================================================================
if [ "${RUN_MILESTONE_9}" = "1" ]; then
  DATASETS="$(normalize_dataset_list "${M9_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}")"
  for MODE in ${M9_SELECT_MODES}; do
    for PROP in ${M9_SELECT_PROPORTIONS}; do
      for SCOPE in ${M9_SELECT_SCOPES}; do
        for SPARSE_FUSE_MODE in ${M9_SPARSE_FUSE_MODES}; do
          PROP_TAG="$(sanitize_prop "${PROP}")"
          scope_tag="${SCOPE}"
          sf_tag="sf${SPARSE_FUSE_MODE}"
          RUN_TAG="step9_${M9_KV_QUANT_SCHEME}_${MODE}_p${PROP_TAG}_${scope_tag}_${sf_tag}_${RUN_TAG_PREFIX}"
          EXTRA_ARGS=()
          if [ "${SPARSE_FUSE_MODE}" = "0" ]; then
            EXTRA_ARGS+=( "--no-kv-sparse-fuse" )
          fi
          if [ -n "${M9_BASE_MODEL}" ]; then
            EXTRA_ARGS+=( "--base-model-override" "${M9_BASE_MODEL}" )
          fi
          if [ -n "${M9_TEACHER_MODEL}" ]; then
            EXTRA_ARGS+=( "--teacher-model-override" "${M9_TEACHER_MODEL}" )
          fi
          if [ -n "${M9_CHECKPOINT_DIR}" ]; then
            EXTRA_ARGS+=( "--checkpoint-dir-override" "${M9_CHECKPOINT_DIR}" )
          fi
          if [ "${M9_DO_ALIGNMENT}" = "1" ]; then
            EXTRA_ARGS+=( "--do-alignment" )
          fi
          if [ -n "${M9_ALIGNMENT_STRATEGY}" ]; then
            EXTRA_ARGS+=( "--alignment-strategy" "${M9_ALIGNMENT_STRATEGY}" )
          fi
          if [ "${M9_TIMING_SYNC}" = "1" ]; then
            EXTRA_ARGS+=( "--kv-timing-sync" )
          fi
          if [ -n "${M9_EVAL_LIMIT}" ]; then
            EXTRA_ARGS+=( "--eval-limit" "${M9_EVAL_LIMIT}" )
          fi
          if [ -n "${M9_EVAL_RANGE}" ]; then
            EXTRA_ARGS+=( "--eval-range" "${M9_EVAL_RANGE}" )
          fi
          "${PYTHON[@]}" quantization/scripts/run_step8_selective_transfer.py \
            --env "${CONDA_ENV}" \
            --mode gpu \
            --kv-quant-scheme "${M9_KV_QUANT_SCHEME}" \
            --kv-select-mode "${MODE}" \
            --kv-select-proportion "${PROP}" \
            --kv-select-scope "${SCOPE}" \
            --kv-select-min-tokens "${M9_SELECT_MIN_TOKENS}" \
            --kv-scatter-fill "${M9_SCATTER_FILL}" \
            --eval-datasets "${M9_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}" \
            --prep-only \
            --run-tag "${RUN_TAG}" \
            "${EXTRA_ARGS[@]}"
          RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_8_selective_transfer/${RUN_TAG}"
          for DATASET in ${DATASETS}; do
            SLUG="$(dataset_slug "${DATASET}")"
            KEY="m9|${M9_KV_QUANT_SCHEME}|${MODE}|${PROP}|${SCOPE}|${SPARSE_FUSE_MODE}"
            if [ -n "${M9_KEY_SUFFIX}" ]; then
              KEY="${KEY}|${M9_KEY_SUFFIX}"
            fi
            run_eval_chunks_with_skip "${RUN_ROOT}" "${SLUG}.yaml" "${DATASET}" "${RUN_ROOT}/logs/step8.log" "${KEY}"
          done
        done
      done
    done
  done
  if [ -n "${M9_EXTRA_LOW_PROP}" ]; then
    PROP_TAG="$(sanitize_prop "${M9_EXTRA_LOW_PROP}")"
    RUN_TAG="step9_${M9_KV_QUANT_SCHEME}_${M9_EXTRA_LOW_MODE}_p${PROP_TAG}_prompt_sf${M9_SPARSE_FUSE}_${RUN_TAG_PREFIX}"
    EXTRA_ARGS=()
    if [ "${M9_SPARSE_FUSE}" = "0" ]; then
      EXTRA_ARGS+=( "--no-kv-sparse-fuse" )
    fi
    if [ -n "${M9_BASE_MODEL}" ]; then
      EXTRA_ARGS+=( "--base-model-override" "${M9_BASE_MODEL}" )
    fi
    if [ -n "${M9_TEACHER_MODEL}" ]; then
      EXTRA_ARGS+=( "--teacher-model-override" "${M9_TEACHER_MODEL}" )
    fi
    if [ -n "${M9_CHECKPOINT_DIR}" ]; then
      EXTRA_ARGS+=( "--checkpoint-dir-override" "${M9_CHECKPOINT_DIR}" )
    fi
    if [ "${M9_DO_ALIGNMENT}" = "1" ]; then
      EXTRA_ARGS+=( "--do-alignment" )
    fi
    if [ -n "${M9_ALIGNMENT_STRATEGY}" ]; then
      EXTRA_ARGS+=( "--alignment-strategy" "${M9_ALIGNMENT_STRATEGY}" )
    fi
    if [ "${M9_TIMING_SYNC}" = "1" ]; then
      EXTRA_ARGS+=( "--kv-timing-sync" )
    fi
    if [ -n "${M9_EVAL_LIMIT}" ]; then
      EXTRA_ARGS+=( "--eval-limit" "${M9_EVAL_LIMIT}" )
    fi
    if [ -n "${M9_EVAL_RANGE}" ]; then
      EXTRA_ARGS+=( "--eval-range" "${M9_EVAL_RANGE}" )
    fi
    "${PYTHON[@]}" quantization/scripts/run_step8_selective_transfer.py \
      --env "${CONDA_ENV}" \
      --mode gpu \
      --kv-quant-scheme "${M9_KV_QUANT_SCHEME}" \
      --kv-select-mode "${M9_EXTRA_LOW_MODE}" \
      --kv-select-proportion "${M9_EXTRA_LOW_PROP}" \
      --kv-select-scope "prompt" \
      --kv-select-min-tokens "${M9_SELECT_MIN_TOKENS}" \
      --kv-scatter-fill "${M9_SCATTER_FILL}" \
      --eval-datasets "${M9_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}" \
      --prep-only \
      --run-tag "${RUN_TAG}" \
      "${EXTRA_ARGS[@]}"
    RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_8_selective_transfer/${RUN_TAG}"
    for DATASET in ${DATASETS}; do
      SLUG="$(dataset_slug "${DATASET}")"
      KEY="m9|${M9_KV_QUANT_SCHEME}|${M9_EXTRA_LOW_MODE}|${M9_EXTRA_LOW_PROP}|prompt|${M9_SPARSE_FUSE}"
      if [ -n "${M9_KEY_SUFFIX}" ]; then
        KEY="${KEY}|${M9_KEY_SUFFIX}"
      fi
      run_eval_chunks_with_skip "${RUN_ROOT}" "${SLUG}.yaml" "${DATASET}" "${RUN_ROOT}/logs/step8.log" "${KEY}"
    done
  fi
  git_autopush "m9"
fi

# =============================================================================
# Milestone 10: RD-C2C (token x precision scheduling).
# =============================================================================
if [ "${RUN_MILESTONE_10}" = "1" ]; then
  DATASETS="$(normalize_dataset_list "${M10_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}")"
  BUDGET_LIST=()
  TAG_LIST=()
  if [ -n "${M10_BUDGET_BYTES}" ]; then
    idx=0
    for BUDGET in ${M10_BUDGET_BYTES}; do
      idx=$((idx+1))
      BUDGET_LIST+=( "${BUDGET}" )
      TAG_LIST+=( "bytes${idx}" )
    done
  else
    for FRAC in ${M10_BUDGET_FRACS}; do
      BUDGET="$(compute_rd_budget_bytes "${FRAC}" "${M10_AVG_INPUT_LEN}" "${M10_BASE_MODEL}" "${M10_INDEX_BYTES}" "${M10_KV_QUANT_AXIS}" "${M10_INCLUDE_SCALE_OVERHEAD}" "${M10_KV_QUANT_SCHEME}")"
      BUDGET_LIST+=( "${BUDGET}" )
      TAG_LIST+=( "$(sanitize_frac "${FRAC}")" )
    done
  fi

  if [ -n "${M10_CANDIDATE_SETS}" ]; then
    CANDIDATE_SETS="$(echo "${M10_CANDIDATE_SETS}" | tr ';' ' ')"
    CAND_TAG_ALWAYS="1"
  else
    CANDIDATE_SETS="${M10_TOKEN_PRECISION_CANDIDATES}"
    CAND_TAG_ALWAYS="0"
  fi

  for CAND_SET in ${CANDIDATE_SETS}; do
    CAND_TAG=""
    if [ "${CAND_TAG_ALWAYS}" = "1" ] || [ "${CAND_SET}" != "${M10_TOKEN_PRECISION_CANDIDATES}" ]; then
      CAND_TAG="_cand$(sanitize_candidates "${CAND_SET}")"
    fi

    EXTRA_ARGS=()
    if [ -n "${M10_BASE_MODEL}" ]; then
      EXTRA_ARGS+=( "--base-model-override" "${M10_BASE_MODEL}" )
    fi
    if [ -n "${M10_TEACHER_MODEL}" ]; then
      EXTRA_ARGS+=( "--teacher-model-override" "${M10_TEACHER_MODEL}" )
    fi
    if [ -n "${M10_CHECKPOINT_DIR}" ]; then
      EXTRA_ARGS+=( "--checkpoint-dir-override" "${M10_CHECKPOINT_DIR}" )
    fi
    if [ "${M10_DO_ALIGNMENT}" = "1" ]; then
      EXTRA_ARGS+=( "--do-alignment" )
    fi
    if [ -n "${M10_ALIGNMENT_STRATEGY}" ]; then
      EXTRA_ARGS+=( "--alignment-strategy" "${M10_ALIGNMENT_STRATEGY}" )
    fi
    if [ "${M10_INCLUDE_SCALE_OVERHEAD}" = "1" ]; then
      EXTRA_ARGS+=( "--kv-include-scale-overhead" )
    fi
    if [ "${M10_TIMING_SYNC}" = "1" ]; then
      EXTRA_ARGS+=( "--kv-timing-sync" )
    fi
    if [ -n "${M10_EVAL_LIMIT}" ]; then
      EXTRA_ARGS+=( "--eval-limit" "${M10_EVAL_LIMIT}" )
    fi
    if [ -n "${M10_EVAL_RANGE}" ]; then
      EXTRA_ARGS+=( "--eval-range" "${M10_EVAL_RANGE}" )
    fi

    for i in "${!BUDGET_LIST[@]}"; do
      BUDGET="${BUDGET_LIST[$i]}"
      TAG="${TAG_LIST[$i]}${CAND_TAG}"
      RUN_TAG="step10_${M10_KV_QUANT_SCHEME}_rd_${TAG}_${M10_SELECT_SCOPE}_${RUN_TAG_PREFIX}"
      "${PYTHON[@]}" quantization/scripts/run_step8_selective_transfer.py \
        --env "${CONDA_ENV}" \
        --mode gpu \
        --kv-quant-scheme "${M10_KV_QUANT_SCHEME}" \
        --kv-quant-axis "${M10_KV_QUANT_AXIS}" \
        --kv-select-mode "${M10_SELECT_MODE}" \
        --kv-select-proportion "1.0" \
        --kv-select-scope "${M10_SELECT_SCOPE}" \
        --kv-select-min-tokens "${M10_SELECT_MIN_TOKENS}" \
        --kv-scatter-fill "${M10_SCATTER_FILL}" \
        --token-precision-mode rd_greedy \
        --token-precision-candidates "${CAND_SET}" \
        --token-precision-budget-bytes "${BUDGET}" \
        --token-precision-scope "${M10_SELECT_SCOPE}" \
        --kv-index-dtype-bytes "${M10_INDEX_BYTES}" \
        --eval-datasets "${M10_EVAL_DATASETS:-${STEP8_EVAL_DATASETS}}" \
        --prep-only \
        --run-tag "${RUN_TAG}" \
        "${EXTRA_ARGS[@]}"
      RUN_ROOT="${PROJECT_ROOT}/quantization/data/step_8_selective_transfer/${RUN_TAG}"
      for DATASET in ${DATASETS}; do
        SLUG="$(dataset_slug "${DATASET}")"
        KEY="m10|${M10_KV_QUANT_SCHEME}|${TAG}|${M10_SELECT_SCOPE}"
        if [ -n "${M10_KEY_SUFFIX}" ]; then
          KEY="${KEY}|${M10_KEY_SUFFIX}"
        fi
        run_eval_chunks_with_skip "${RUN_ROOT}" "${SLUG}.yaml" "${DATASET}" "${RUN_ROOT}/logs/step8.log" "${KEY}"
      done
    done
  done
  git_autopush "m10"
fi

echo "Done."
