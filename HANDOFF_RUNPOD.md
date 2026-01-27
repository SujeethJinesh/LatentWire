# RunPod Handoff (LatentWire / Quantized C2C)

This file is the detailed handoff so we can resume immediately on the RunPod H100 node.

## 0) Current status (as of last local session)
- **Repo:** `LatentWire` on `main`.
- **Latest commit:** `75b7244` — *Pin slurm to user conda python*.
- **Submodule:** `quantization/C2C` pinned to **fork** `SujeethJinesh/C2C` on branch `quant-kv`, commit `cad82ef`.
- **Key fix:** `quantization/submit_milestones.slurm` now prefers the user conda env Python (`/projects/.../conda/envs/rosetta/bin/python`) to avoid env mismatch. This fixed `ModuleNotFoundError: datasets` in Slurm runs.
- **Implementation plan updated:** `quantization/IMPLEMENTATION_PLAN.md` contains full milestone plan, GPU commands, and a **References (ArXiv)** section with 20 papers (quantization, cross‑LLM, KV‑cache, inference).

## 1) RunPod objective
Run M0–M3 on a single H100 (baseline + PTQ + cache‑length grid) and then M4 analysis (budget curves).

## 2) IMPORTANT: use forked submodule
The C2C submodule must point at the fork (`SujeethJinesh/C2C`) because the commit `cad82ef` does **not** exist on the upstream `thu-nics` repo.

If you see `fatal: remote error: upload-pack: not our ref cad82ef...`, do:
```
cd /workspace/LatentWire
git submodule sync --recursive
git submodule update --init --recursive quantization/C2C
```
If it still pulls upstream, force the URL:
```
git config submodule."quantization/C2C".url git@github.com:SujeethJinesh/C2C.git
# or HTTPS if SSH key issues
# git config submodule."quantization/C2C".url https://github.com/SujeethJinesh/C2C.git

git submodule sync --recursive
git submodule update --init --recursive quantization/C2C
```

## 3) RunPod setup (Option A = Conda/Miniforge)
We decided to use **conda** even on RunPod, to match HPC behavior and avoid script edits.

### 3.1 Install Miniforge and create env
```
# Install Miniforge
wget -qO /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p /opt/miniforge
export PATH=/opt/miniforge/bin:$PATH

# Create env
conda create -n rosetta python=3.10 -y
conda activate rosetta
```

### 3.2 Clone + submodule
```
cd /workspace
git clone git@github.com:SujeethJinesh/LatentWire.git
cd LatentWire

git submodule sync --recursive
git submodule update --init --recursive quantization/C2C
```

### 3.3 Install deps
```
python -m pip install -U pip
python -m pip install torch transformers datasets huggingface_hub pyyaml tqdm numpy
python -m pip install -e quantization/C2C
python -m pip install -e "quantization/C2C[training,evaluation]"
```

### 3.4 Cache locations (to avoid filling HOME)
```
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export C2C_CKPT_ROOT=/workspace/c2c_checkpoints
```

### 3.5 Sanity check
```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import datasets, transformers; print(datasets.__version__, transformers.__version__)"
```

## 4) Running the milestones
We run the Slurm script as a normal bash script on RunPod. It already handles:
- preflight (prep‑only) runs
- chunked eval with resume
- caching to scratch (adjusted via env vars above)

### 4.1 Default run (M0 + M2 + M3)
```
CONDA_ENV_PATH=/opt/miniforge/envs/rosetta \
RUN_MILESTONE_0=1 RUN_MILESTONE_2=1 RUN_MILESTONE_3=1 \
bash quantization/submit_milestones.slurm
```

### 4.2 Dry‑run only (preflight + exit)
```
CONDA_ENV_PATH=/opt/miniforge/envs/rosetta \
RUN_MILESTONE_0=1 RUN_MILESTONE_2=1 RUN_MILESTONE_3=1 DRY_RUN=1 \
bash quantization/submit_milestones.slurm
```

### 4.3 Smoke limits
```
CONDA_ENV_PATH=/opt/miniforge/envs/rosetta \
RUN_MILESTONE_0=1 RUN_MILESTONE_2=1 RUN_MILESTONE_3=1 \
EVAL_SMOKE=1 SMOKE_LIMIT=50 \
bash quantization/submit_milestones.slurm
```

### 4.4 Logs
- Logs go to `runs/quant_milestones_<JOBID>.log` and `.err`.
- If something fails, `git add . && git commit -m 'Upload logs' && git push` so we can inspect.

## 5) Data output paths
- M0: `quantization/data/step_0_baselines/<run_tag>/`
- M2/M3: `quantization/data/step_1_kv_ptq/<run_tag>/`
- Contains configs/, logs/, results/, manifests/.

## 6) Known issues + fixes
- **Earlier failure:** `ModuleNotFoundError: datasets` during Slurm chunking.
  - **Fix:** Slurm now pins **user conda python** via `CONDA_ENV_PATH`. If this is set, it bypasses `conda run` and uses the correct env.
- **Submodule ref error:** `not our ref cad82ef` if submodule URL still points to upstream. Fix via `git submodule sync --recursive` and set URL to fork.
- **Login node buffering:** Use `python -u` or `conda run --no-capture-output` if logs appear stuck.

## 7) M4 analysis (budget curves)
After M2+M3 GPU runs finish:
```
python quantization/scripts/analyze_budget_curve.py \
  --runs-root quantization/data \
  --output-dir quantization/analysis/m4_budget_curve
```
Outputs:
- `quantization/analysis/m4_budget_curve/budget_curve.csv`
- `quantization/analysis/m4_budget_curve/budget_curve_<dataset>.png`

## 8) References section
`quantization/IMPLEMENTATION_PLAN.md` includes a **References (ArXiv)** list of 20 papers across:
- Quantization
- Cross‑LLM / Multi‑Agent
- KV‑cache / long‑context
- Inference / Decoding / Serving

## 9) Files to know
- `quantization/submit_milestones.slurm` — main orchestrator (now uses CONDA_ENV_PATH)
- `quantization/scripts/run_step0_baselines.py` — M0 prep + eval
- `quantization/scripts/run_step1_kv_ptq.py` — M1/M2/M3 prep + eval
- `quantization/scripts/analyze_budget_curve.py` — M4 analysis
- `quantization/IMPLEMENTATION_PLAN.md` — master plan + commands

## 10) Next steps after RunPod run
1) Confirm logs show eval chunking lines ("Eval <dataset>: [start, end)").
2) Verify results JSONs exist under `quantization/data/.../results`.
3) Run M4 analysis.
4) Commit results + logs for review.

