# SinkAware Phase 0 Setup Complete

- date: 2026-05-06
- environment: repo-root `./venv_arm64`
- python: `Python 3.11.6`
- platform: `macOS-26.4.1-arm64-arm-64bit`
- pip: `pip 26.0.1`
- scope: Mac-local reproducibility surface only; no SSH, CUDA, GPU, or global
  installs were used.

## Commands

```bash
PIP_CACHE_DIR=.debug/pip_cache \
  ./venv_arm64/bin/python -m pip install \
  -r experimental/sinkaware/requirements.txt

./venv_arm64/bin/python -m pip check

./venv_arm64/bin/python - <<'PY'
mods = [
    "IPython",
    "einops",
    "tiktoken",
    "torch",
    "transformers",
    "numpy",
    "matplotlib",
    "jupyter",
    "datasets",
    "huggingface_hub",
    "safetensors",
    "triton",
    "sentencepiece",
]
for mod in mods:
    __import__(mod)
    print(mod, "ok")
PY
```

## Result

`experimental/sinkaware/requirements.txt` installs successfully into
`./venv_arm64`. The previously missing imports `IPython`, `einops`, and
`tiktoken` now import successfully. `pip check` reports:

```text
No broken requirements found.
```

Verified package versions:

```text
datasets==4.8.4
einops==0.8.2
huggingface_hub==1.12.0
ipython==9.13.0
matplotlib==3.10.8
numpy==1.26.4
pytest==9.0.3
safetensors==0.7.0
sentencepiece==0.2.1
tiktoken==0.12.0
torch==2.6.0
transformers==5.7.0
triton==3.7.0+git270e696d
```

## Decision

Phase 0 setup is complete for the current Mac-local SinkAware reproducibility
surface. This does not add scientific evidence and does not change the native
GPU gate: promotion still requires a native NVIDIA packet with matched quality
drift, downstream loss/KL/top-1 checks, repeated latency, and NCU memory/HBM
counters.
