# LatentWire: A minimal interlingua between two LLMs (Llama & Qwen)

This proof‑of‑concept trains a tiny encoder + adapters so that the same
compressed latent sequence (“soft tokens”) can drive two different LLMs—
one from the Llama family and one from the Qwen family—on a small HotpotQA slice.

- Input: bytes of `Question + Context`
- Interlingua: `M` soft tokens (default `M=8`, dim `d_z=256`)
- Adapters: map soft tokens → each model’s hidden size
- Objective: next‑token loss on answers while both LLMs stay frozen

## Quickstart

### 0) Environment
- Python 3.10+ recommended.
- macOS (Apple Silicon or Intel) or Linux.
- GPU optional (CUDA recommended for full runs); CPU works for smoke tests.

### 1) Install
```bash
bash scripts/setup_mac.sh      # on macOS
# or
bash scripts/setup_linux.sh    # on Linux
```

### 2) Smoke test (fast, CPU-friendly)
```bash
bash scripts/run_smoke_cpu.sh
```

You should see shapes/logs and short generations from tiny HF models.
This validates the inputs_embeds prefix → token decoding path.

### 3) Train small demo on HotpotQA
```bash
bash scripts/run_train_small.sh
```

This trains the shared encoder + adapters w/ frozen TinyLlama + Qwen2‑0.5B
on a small subset.

### 4) Evaluate
```bash
bash scripts/run_eval_small.sh
```

This reports:
- EM/F1: Text vs Latent vs Token‑budget (same M) + Joint pick
- NLL/token on gold answers (conditioning quality)
- Compression ratio, payload bytes, latency, agreement, oracle bound

### Figures of Merit
- Task quality: EM/F1 on HotpotQA.
- Conditioning: NLL/token (gold) under text vs latent.
- Efficiency: compression ratio; estimated interlingua payload; wall‑clock.
- Two‑model gains: joint‑pick vs best single model; oracle bound.

### Notes
- Default models are small for accessibility:
  - Llama‑like: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - Qwen: `Qwen/Qwen2-0.5B-Instruct`
- You can substitute larger models if you have GPU (e.g., Llama‑3.1‑8B, Qwen2‑7B).
- On macOS (M‑series), MPS is supported by PyTorch; we default to float32 on non‑CUDA.

### License
Uses public checkpoints under their own licenses. Follow model providers’ terms.
