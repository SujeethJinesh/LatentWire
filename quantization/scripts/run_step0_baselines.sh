#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/projects/m000066/sujinesh/LatentWire}"
export PROJECT_ROOT

if [ ! -d "$PROJECT_ROOT" ]; then
  echo "ERROR: PROJECT_ROOT not found: $PROJECT_ROOT" >&2
  exit 1
fi

# Require a visible GPU (avoid running on login nodes).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. This script requires a GPU node." >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: No GPUs detected by nvidia-smi. Run this on a GPU node." >&2
  exit 1
fi

cd "$PROJECT_ROOT"

git submodule update --init --recursive quantization/C2C

# Optional caches on scratch
export HF_HOME="${HF_HOME:-/scratch/m000066/${USER}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME"

# Disable W&B by default for evaluation
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

RUN_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ROOT="$PROJECT_ROOT/data/step_0_baselines/$RUN_TAG"
export RUN_ROOT
mkdir -p "$RUN_ROOT"/configs "$RUN_ROOT"/logs "$RUN_ROOT"/results "$RUN_ROOT"/manifests

# Log everything into the run folder
exec > >(tee -a "$RUN_ROOT/logs/step0.log") 2>&1

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH" >&2
  exit 1
fi

# Shell hook for conda
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "rosetta"; then
  conda create -n rosetta python=3.10 -y
fi
conda activate rosetta

NEED_INSTALL=1
python - <<'PY' && NEED_INSTALL=0 || NEED_INSTALL=1
import importlib
mods = [
    "rosetta",
    "torch",
    "transformers",
    "datasets",
    "huggingface_hub",
    "yaml",
    "numpy",
    "tqdm",
]
for m in mods:
    importlib.import_module(m)
print("Environment OK")
PY

pushd quantization/C2C
if [ "$NEED_INSTALL" -eq 1 ]; then
  pip install -e .
  pip install -e ".[training,evaluation]"
else
  echo "Dependencies already installed; skipping pip install."
fi

python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json, os, subprocess, time

run_root = Path(os.environ["RUN_ROOT"])
ckpt_root = Path(os.environ.get("C2C_CKPT_ROOT", f"/scratch/m000066/{os.environ.get('USER','user')}/c2c_checkpoints"))
ckpt_root.mkdir(parents=True, exist_ok=True)

repo_id = "nics-efc/C2C_Fuser"
pattern = "qwen3_0.6b+qwen2.5_0.5b_Fuser/*"
local_dir = ckpt_root / "C2C_Fuser"

snapshot_path = snapshot_download(
    repo_id=repo_id,
    allow_patterns=[pattern],
    local_dir=str(local_dir),
    local_dir_use_symlinks=False,
)

# Capture git provenance
try:
    repo_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    lw_commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).decode().strip()
    c2c_commit = subprocess.check_output(["git", "-C", str(repo_root / "quantization" / "C2C"), "rev-parse", "HEAD"]).decode().strip()
except Exception:
    lw_commit = None
    c2c_commit = None

manifest = {
    "repo_id": repo_id,
    "allow_patterns": [pattern],
    "snapshot_path": snapshot_path,
    "checkpoint_dir": str(local_dir / "qwen3_0.6b+qwen2.5_0.5b_Fuser" / "final"),
    "latentwire_commit": lw_commit,
    "c2c_commit": c2c_commit,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

out = run_root / "manifests" / "step_0_checkpoint_manifest.json"
out.write_text(json.dumps(manifest, indent=2))
print("Wrote manifest:", out)
PY

# Copy and patch eval configs
cp recipe/eval_recipe/unified_eval.yaml "$RUN_ROOT/configs/openbookqa.yaml"
cp recipe/eval_recipe/unified_eval.yaml "$RUN_ROOT/configs/arc_c.yaml"

python - <<'PY'
import yaml, json, os
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT"])
manifest = json.loads((run_root / "manifests/step_0_checkpoint_manifest.json").read_text())
ckpt_dir = manifest["checkpoint_dir"]

def patch(cfg_path, dataset, out_dir):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    cfg["model"]["rosetta_config"]["checkpoints_dir"] = ckpt_dir
    cfg["output"]["output_dir"] = str(out_dir)
    cfg["eval"]["dataset"] = dataset
    Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

patch(run_root / "configs/openbookqa.yaml", "openbookqa", run_root / "results/openbookqa")
patch(run_root / "configs/arc_c.yaml", "ai2-arc", run_root / "results/arc_c")
PY

# Run baseline evals
python script/evaluation/unified_evaluator.py --config "$RUN_ROOT/configs/openbookqa.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/openbookqa.log"

python script/evaluation/unified_evaluator.py --config "$RUN_ROOT/configs/arc_c.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/arc_c.log"

popd

echo "Step 0 complete. Results in: $RUN_ROOT"
