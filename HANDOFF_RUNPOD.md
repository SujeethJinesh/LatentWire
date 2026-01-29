# RunPod Handoff (LatentWire / Quantized C2C)

This file is the **single source of truth** for resuming on the RunPod H100. It captures *all current context*, decisions, fixes, and the exact commands to run. If this file is up to date, you can pick up without any other chat context.

---

## 0) Current state (latest local repo)
- **Repo:** `LatentWire` on `main`
- **Latest commit (local):** `340b73d6` – *Update C2C submodule (disable W&B in QAT recipe)*
- **Submodule:** `quantization/C2C` → fork `SujeethJinesh/C2C`, branch `quant-kv`, latest commit `ac5adba` (W&B disabled in QAT recipe)
- **Key fixes in place:**
  - `quantization/submit_milestones.slurm`:
    - Uses direct conda python if `CONDA_ENV_PATH` exists; otherwise `conda run --no-capture-output`
    - Requires `M7_CHECKPOINT_DIR` for heterogeneity (preflight and full) to prevent CUDA assert
    - Auto‑selects full QAT recipe if `RUN_FULL=1`
    - W&B disabled by default (`WANDB_MODE=disabled`, `WANDB_DISABLED=true`)
  - `run_registry.json` updated: marked **invalid** runs for M5 (smoke-only) and M7 (CUDA assert) so they are **not skipped**
  - Added hetero fuser training recipe: `quantization/C2C/recipe/train_recipe/C2C_0.6+llama3.2_1b.json` (W&B disabled)
  - Updated QAT recipe `C2C_0.6+0.5_qat_int8.json` to disable W&B

---

## 1) RunPod environment (persistent)
Recommended setup:
- `/workspace/env.sh` and `/workspace/bootstrap.sh` are used to persist env vars and setup.
- `.bashrc` sources `/workspace/env.sh` on login.
- Cache & checkpoints on `/workspace` to avoid container root.

**Required env vars:**
```
export HF_HOME=/workspace/.cache/huggingface
export C2C_CKPT_ROOT=/workspace/c2c_checkpoints
export CONDA_EXE=/workspace/conda/bin/conda
export HF_TOKEN=***
```
W&B is disabled by default in the scripts; no key required.

---

## 2) Submodule sanity
**Always use the fork** (the upstream doesn’t have our commits).
```
cd /workspace/LatentWire

git pull
git submodule sync --recursive
git submodule update --init --recursive quantization/C2C
```
If you see “not our ref …” then force URL:
```
git config submodule."quantization/C2C".url git@github.com:SujeethJinesh/C2C.git
# or https
# git config submodule."quantization/C2C".url https://github.com/SujeethJinesh/C2C.git

git submodule sync --recursive
git submodule update --init --recursive quantization/C2C
```

---

## 3) Sanity checks (RunPod)
```
source /workspace/env.sh

/workspace/conda/envs/rosetta/bin/python - <<'PY'
from datasets import load_dataset
print("openbookqa", len(load_dataset("allenai/openbookqa", revision="main")["test"]))
print("ai2_arc", len(load_dataset("allenai/ai2_arc", "ARC-Challenge", revision="main")["test"]))
PY
```

---

## 4) Critical failures encountered (and fixes)
### M7 heterogeneity (Llama)
- **Failure:** CUDA device‑side assert for every sample → all accuracies 0.
- **Cause:** using Qwen↔Qwen fuser for Llama run.
- **Fix:** train hetero fuser first, then pass `M7_CHECKPOINT_DIR` to the eval.

### M5 QAT training
- **Failure:** W&B login error.
- **Fix:** W&B disabled in QAT recipe and in `submit_milestones.slurm`.

### Background jobs dying
- **Cause:** SIGHUP when shell exits.
- **Fix:** always run training/eval via `tmux`.

---

## 5) Hetero fuser training (required for M7)
**Must train before M7 eval.** W&B is disabled in recipe.

```
# Run in tmux
FUSER_TAG=$(date +%Y%m%d_%H%M%S)_m7_fuser
FUSER_RUN_ROOT="quantization/data/step_7_fuser/${FUSER_TAG}"
FUSER_OUT="${C2C_CKPT_ROOT}/fuser_m7_${FUSER_TAG}"
FUSER_LOG="${FUSER_RUN_ROOT}/logs/m7_fuser.log"
FUSER_ERR="${FUSER_RUN_ROOT}/logs/m7_fuser.err"

mkdir -p "${FUSER_RUN_ROOT}/configs" "${FUSER_RUN_ROOT}/logs" "${FUSER_RUN_ROOT}/manifests"
cp quantization/C2C/recipe/train_recipe/C2C_0.6+llama3.2_1b.json "${FUSER_RUN_ROOT}/configs/m7_fuser.json"
/workspace/conda/envs/rosetta/bin/python - <<'PY' "${FUSER_RUN_ROOT}/configs/m7_fuser.json" "${FUSER_OUT}"
import json, sys
cfg = json.loads(open(sys.argv[1]).read())
cfg.setdefault("output", {})["output_dir"] = sys.argv[2]
open(sys.argv[1], "w").write(json.dumps(cfg, indent=2)+"\n")
print("Wrote fuser config:", sys.argv[1])
PY

tmux new -d -s m7_fuser "cd /workspace/LatentWire && source /workspace/env.sh && /workspace/conda/envs/rosetta/bin/python quantization/C2C/script/train/SFT_train.py --config '${FUSER_RUN_ROOT}/configs/m7_fuser.json' > '${FUSER_LOG}' 2> '${FUSER_ERR}'"
```

**Progress monitoring:**
- `m7_fuser.err` shows progress lines.
- If logs are buffered, check checkpoint directory timestamps:
  `ls -lt ${FUSER_OUT} | head`

---

## 6) M7 eval (heterogeneity)
**Only after fuser training completes.**
```
M7_CHECKPOINT_DIR="${FUSER_OUT}"

TAG1=$(date +%Y%m%d_%H%M%S)_m7
LOG1="runs/quant_m7_${TAG1}.log"
ERR1="runs/quant_m7_${TAG1}.err"

tmux new -d -s m7_eval "cd /workspace/LatentWire && source /workspace/env.sh && export M7_CHECKPOINT_DIR='${M7_CHECKPOINT_DIR}' && RUN_MILESTONE_7=1 RUN_FULL=1 M7_ALIGNMENT_MODES='0 1' RUN_TAG_PREFIX='${TAG1}' bash quantization/submit_milestones.slurm > '${LOG1}' 2> '${ERR1}'"
```

---

## 7) M5 QAT full (training + eval)
```
TAG2=$(date +%Y%m%d_%H%M%S)_m5
LOG2="runs/quant_m5_${TAG2}.log"
ERR2="runs/quant_m5_${TAG2}.err"

tmux new -d -s m5_run "cd /workspace/LatentWire && source /workspace/env.sh && RUN_MILESTONE_5=1 RUN_FULL=1 RUN_TAG_PREFIX='${TAG2}' bash quantization/submit_milestones.slurm > '${LOG2}' 2> '${ERR2}'"
```

---

## 8) Registry behavior
- **run_registry.json** used to skip completed runs.
- Invalid runs are explicitly marked `status=invalid` so they are re‑run.
- Key invalidations already applied for:
  - `m5|qat|int8` (smoke only)
  - `m7|Qwen3|Llama3.2|align0/align1` (CUDA assert)

---

## 9) Data output paths
- Baselines: `quantization/data/step_0_baselines/<run_tag>/`
- PTQ / cache length / M6 / M7 / M5 eval: `quantization/data/step_1_kv_ptq/<run_tag>/`
- QAT training: `quantization/data/step_5_qat/<run_tag>/`
- Hetero fuser training: `quantization/data/step_7_fuser/<run_tag>/`

---

## 10) Known valid results (summary)
- M0/M2/M3/M6/M8 results are already in `golden_summary.md` and `golden_runs.json`.
- M7 Llama heterogeneity and M5 QAT full are **NOT done** yet (must run after fuser training).

---

## 11) Time estimate for fuser training
- At ~13% progress, ETA was ~7–8 hours remaining. If `checkpoint-500` exists, remaining ~3.5–5.5 hours.
- Update ETA by checking `m7_fuser.err` or checkpoint timestamps.

---

## 12) Minimal status check (no log spam)
```
# tmux status
 tmux has-session -t m7_fuser
 tmux has-session -t m7_eval
 tmux has-session -t m5_run

# GPU activity
 nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used --format=csv -l 5

# latest checkpoint
 ls -lt ${FUSER_OUT} | head
```

---

## 13) Files to keep in mind
- `quantization/submit_milestones.slurm` – main orchestrator
- `quantization/scripts/run_step1_kv_ptq.py` – eval runner
- `quantization/registry/run_registry.json` – skip/registry
- `quantization/golden/golden_summary.md` – consolidated results
- `quantization/IMPLEMENTATION_PLAN.md` – master plan

---

## 14) Next actions
1) Let fuser finish (m7_fuser)
2) Run M7 eval with `M7_CHECKPOINT_DIR` set
3) Run M5 full QAT (W&B disabled)
4) Pull logs back to local, update goldens + registry, update paper

