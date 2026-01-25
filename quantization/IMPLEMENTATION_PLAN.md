# Implementation Plan (1x H100 per step)

Goal: deliver a workshop‑ready paper on **Quantized Cache‑to‑Cache** with communication‑budget curves, and a clear extension path to main‑conference quality (QAT + mixed precision + heterogeneity scaling). Each step below is runnable on a single H100 node.

Data layout (committed to git):
```
data/
  step_0_baselines/<run_tag>/
    configs/
    logs/
    results/
    manifests/
```
Every step follows the same pattern: create a new `data/step_X_*` run folder, copy the config, log all commands, and store results JSONs there. Checkpoints can live on scratch; the run folder must include a manifest (repo + commit + path).

---

## Step 0: Environment + Baseline Sanity (1–2 hours)

**What**
- Create the C2C environment, confirm code runs, and reproduce 1–2 baseline scores.

**Why**
- Establish a trusted baseline before introducing quantization effects.

**How (1x H100)**
```bash
# After salloc:
# salloc -N 1 -G 1 -A marlowe-m000066 -p preempt --time=3:00:00 --mem=32GB

# One-time setup (can be done on login node)
conda create -n rosetta python=3.10 -y

# Login node prep (downloads + configs, no eval)
cd /projects/m000066/sujinesh/LatentWire
python quantization/scripts/run_step0_baselines.py --prep-only

# One-command runner on GPU node (recommended, from repo root)
cd /projects/m000066/sujinesh/LatentWire
python quantization/scripts/run_step0_baselines.py

# Repo + submodule
cd /path/to/LatentWire
git submodule update --init --recursive quantization/C2C

# (Optional) use scratch for HF caches
export HF_HOME=/scratch/m000066/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_HOME"

# Create run folder for step 0
RUN_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ROOT="$PWD/data/step_0_baselines/$RUN_TAG"
export RUN_ROOT
mkdir -p "$RUN_ROOT"/{configs,logs,results,manifests}

cd quantization/C2C
conda create -n rosetta python=3.10 -y
conda activate rosetta
pip install -e .
pip install -e ".[training,evaluation]"

# Download C2C fuser checkpoint to scratch (do not commit weights)
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json, os, time

run_root = Path(os.environ["RUN_ROOT"]) if "RUN_ROOT" in os.environ else Path("data/step_0_baselines/unknown")
ckpt_root = Path(os.environ.get("C2C_CKPT_ROOT", "/scratch/m000066/%s/c2c_checkpoints" % os.environ.get("USER", "user")))
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

manifest = {
    "repo_id": repo_id,
    "allow_patterns": [pattern],
    "snapshot_path": snapshot_path,
    "checkpoint_dir": str(local_dir / "qwen3_0.6b+qwen2.5_0.5b_Fuser" / "final"),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

out = run_root / "manifests" / "step_0_checkpoint_manifest.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
print("Wrote manifest:", out)
PY

# Copy eval config and point to checkpoint/output dir
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

# Run baseline evals (OpenBookQA + ARC-C)
python script/evaluation/unified_evaluator.py --config "$RUN_ROOT/configs/openbookqa.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/openbookqa.log"

python script/evaluation/unified_evaluator.py --config "$RUN_ROOT/configs/arc_c.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/arc_c.log"
```

**Output**
- Results JSON under `data/step_0_baselines/<run_tag>/results/*`.
- Logs under `data/step_0_baselines/<run_tag>/logs/*`.
- Manifest with checkpoint provenance under `data/step_0_baselines/<run_tag>/manifests/`.

**GPU guard**
- The script exits early if no GPU is detected (prevents accidental login-node runs).
 - Use `--prep-only` on the login node to run setup and downloads without evaluation.

**Environment auto-detect**
- The script checks for required Python modules; if they are already installed, it skips `pip install` and proceeds to evaluation.
- If not running inside the requested conda env, the script re-execs itself via `conda run -n rosetta ...`.
- Environment/module paths are always printed and written to `data/step_0_baselines/<run_tag>/manifests/env_info.json`.

**Workshop/Main‑conf connection**
- Valid baseline needed to attribute any gains to quantization or cache‑budgeting.

**Git capture (after step 0)**
```bash
cd /path/to/LatentWire
git add "data/step_0_baselines/$RUN_TAG"
git status -sb
```

---

## Step 1: Implement KV PTQ Utilities (2–4 hours)

**What**
- Add post‑training quantization (PTQ) for KV tensors (INT8 + INT4/NF4).

**Why**
- This is the core missing piece in C2C and a strong workshop‑level contribution.

**How (1x H100)**
- Create `rosetta/utils/quant.py` with:
  - `quantize_kv(t, scheme, axis)`
  - `dequantize_kv(q, scale, zero)`
  - optional per‑head / per‑token scales
- Add quantization hooks in `rosetta/model/wrapper.py` near projector calls:
  - Quantize source KV before projection (and optionally target KV).
- Add config flags in `recipe/eval_recipe/unified_eval.yaml`:
  - `kv_quant_scheme: int8 | int4 | nf4 | fp8`
  - `kv_quant_axis: per_head | per_layer`

**Output**
- Code compiles and PTQ toggles via config.

**Workshop/Main‑conf connection**
- Enables controlled accuracy‑vs‑bytes comparisons.

---

## Step 2: PTQ Evaluation (3–6 hours)

**What**
- Run PTQ evaluation on OpenBookQA + ARC‑C with INT8 and INT4/NF4.

**Why**
- Establish accuracy‑drop vs bandwidth reduction.

**How (1x H100)**
```bash
# Create a new run folder
RUN_TAG=$(date +"%Y%m%d_%H%M%S")
RUN_ROOT="$PWD/data/step_2_ptq/$RUN_TAG"
export RUN_ROOT
mkdir -p "$RUN_ROOT"/{configs,logs,results,manifests}

# Point to the Step 0 run folder
STEP0_ROOT="/path/to/LatentWire/data/step_0_baselines/<RUN_TAG_FROM_STEP0>"

# Copy baseline configs from Step 0
cp "$STEP0_ROOT/configs/openbookqa.yaml" "$RUN_ROOT/configs/openbookqa_int8.yaml"
cp "$STEP0_ROOT/configs/openbookqa.yaml" "$RUN_ROOT/configs/openbookqa_nf4.yaml"

# Edit configs (set quant scheme + output dir)
python - <<'PY'
import yaml
from pathlib import Path

def patch(cfg_path, scheme, out_dir):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    cfg.setdefault("quant", {})
    cfg["quant"]["kv_quant_scheme"] = scheme
    cfg["quant"]["kv_quant_axis"] = "per_head"
    cfg["output"]["output_dir"] = str(out_dir)
    Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))

root = Path("data/step_2_ptq")
for cfg_path in root.rglob("openbookqa_int8.yaml"):
    patch(cfg_path, "int8", cfg_path.parent.parent / "results/openbookqa_int8")
for cfg_path in root.rglob("openbookqa_nf4.yaml"):
    patch(cfg_path, "nf4", cfg_path.parent.parent / "results/openbookqa_nf4")
PY

python script/evaluation/unified_evaluator.py --config "$RUN_ROOT/configs/openbookqa_int8.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/openbookqa_int8.log"

python script/evaluation/unified_evaluator.py --config "$RUN_ROOT/configs/openbookqa_nf4.yaml" \
  2>&1 | tee "$RUN_ROOT/logs/openbookqa_nf4.log"
```

**Output**
- Eval logs + metrics per dataset.

**Workshop/Main‑conf connection**
- Core results table for workshop submission.

**Git capture (after step 2)**
```bash
cd /path/to/LatentWire
git add "data/step_2_ptq/$RUN_TAG"
git status -sb
```

---

## Step 3: Cache‑Length Reduction (2–6 hours)

**What**
- Add KV token pruning (top‑k or fixed ratio) and evaluate accuracy vs cache length.

**Why**
- Shows direct communication‑budget control (bytes vs accuracy) beyond quantization.

**How (1x H100)**
- Add a selection function (e.g., top‑k by attention norm or key magnitude).
- Implement in `rosetta/model/wrapper.py` before projection (prune source KV).
- Evaluate 50%, 25%, 10% cache ratios with INT8.

**Output**
- Accuracy vs cache ratio curves.

**Workshop/Main‑conf connection**
- Strong “budgeted communication” framing for workshop paper.

---

## Step 4: Communication‑Budget Curves (1–3 hours)

**What**
- Log total bytes transmitted for each setting and plot accuracy vs bytes.

**Why**
- Provides a new evaluation dimension (accuracy per byte) not in C2C.

**How (1x H100)**
- Compute bytes = num_tokens * heads * head_dim * 2 * bytes_per_value.
- Extend `script/evaluation/unified_evaluator.py` to emit `bytes_tx`.
- Plot (matplotlib) in `script/analysis/`.

**Output**
- Main plot for paper: Accuracy vs Bytes (C2C FP16 vs PTQ vs PTQ+pruning).

**Workshop/Main‑conf connection**
- Strong single‑figure contribution; helps justify novelty.

---

## Step 5: QAT Recovery (Optional for Main‑Conf, 4–12 hours)

**What**
- Quantization‑aware training (QAT) of projector weights under INT8 noise.

**Why**
- Demonstrates that accuracy can be recovered at low precision.

**How (1x H100)**
- Inject fake‑quant in projector forward.
- Train on 10–50k samples (OpenHermes subset).
- Evaluate on ARC‑C + OpenBookQA.

**Output**
- Accuracy improvements vs PTQ; report training cost.

**Workshop/Main‑conf connection**
- Moves the story from “engineering tweak” to “learning under constraints”.

---

## Step 6: Mixed Precision by Layer (Optional, 4–10 hours)

**What**
- Use higher precision for late layers, lower precision for early layers.

**Why**
- Exploit layer sensitivity for better accuracy/bytes trade‑off.

**How (1x H100)**
- Add per‑layer precision config (e.g., last 4 layers FP16, middle INT8, early INT4).
- Evaluate with same datasets.

**Output**
- Mixed‑precision curve beating uniform INT4/INT8.

**Workshop/Main‑conf connection**
- New technical idea with clear empirical win.

---

## Step 7: Heterogeneity Scaling (Optional, 3–6 hours)

**What**
- Test one cross‑family pair (Qwen3 ← Llama3.2 or Gemma3).

**Why**
- Demonstrates generality beyond a single model family.

**How (1x H100)**
- Load different base/sharer pairs in `recipe/train_recipe/` or eval recipes.
- Run PTQ + pruning on ARC‑C.

**Output**
- A single robustness table.

**Workshop/Main‑conf connection**
- Adds external validity; helps main‑conf story.

---

# Minimal Workshop Path (Shortest)

- Step 0 → Step 1 → Step 2 → Step 3 → Step 4

Expected GPU time: **~1 day** total on 1 H100.

# Main‑Conf Path (Extended)

- Workshop path + Step 5 + Step 6 + Step 7

Expected GPU time: **~3–5 days** total on 1 H100.
