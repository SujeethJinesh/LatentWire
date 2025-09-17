# LatentWire – Quickstart (Patched Package Layout)

This archive turns your files into an importable Python package (`latentwire/`) and adds:
- a missing `metrics.py` (EM/F1) used by `eval.py`,
- a tiny `diagnostics.py` that saves `env_snapshot.json` to each run directory,
- minimal patches to `latentwire/train.py` and `latentwire/eval.py` to call the diagnostics snapshot,
- a `requirements.txt` for reproducible installs.

## 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Run the pipeline
The provided `run_pipeline.sh` expects the `latentwire/` folder to exist (which this patch provides).
```bash
bash run_pipeline.sh
```

## 3) Outputs
- Checkpoints and per-epoch eval go under `runs/<RUN>/...` (as defined in the script).
- Each train/eval directory will include `env_snapshot.json` with basic system info to help debugging.
- Eval will also dump `metrics.json` and `predictions.jsonl`.

## Notes
- The code supports NF4 quantization via bitsandbytes when running on Linux with NVIDIA GPUs.
- If running on CPU or Apple Silicon, set `CUDA_VISIBLE_DEVICES=""` or adjust the script toggles.