# LatentWire Quickstart

LatentWire turns latent soft-prompts into a shared wire that both Llama and Qwen can decode. The repo now contains smoke-scale runners, hero-scale automation, and structured diagnostics so you can reproduce the staged pipeline end-to-end.

---

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you are planning to use NF4/4-bit quantisation, make sure `bitsandbytes`, CUDA, and compatible GPU drivers are installed. The helper scripts will check for `peft`, `accelerate`, and `bitsandbytes` automatically and install when missing.

---

## 2. Smoke vs Hero runs

| Runner | Scope | GPUs | Output |
| --- | --- | --- | --- |
| `scripts/run_llama_single.sh` | Llama-only encoder/adapters. Supports latent/refiner sweeps (`LATENT_LEN_LIST`, `D_Z_LIST`, `REFINER_*`). | default `CUDA_VISIBLE_DEVICES=0,1,2,3` | `runs/<run_tag>/` with checkpoints, `pipeline.log`, `diagnostics.jsonl` |
| `scripts/run_scoped_softprompt_multi.sh` | Full Llama + Qwen shared latent pipeline. | defaults to a 2+2 GPU split (override with env vars). | Same structure as above. |

Both runners accept `--hero` to enable the larger Stage B schedules described in the research proposal.

Key environment variables you can override:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3     # GPU selection
export RUN_TAG=my_latent_sweep          # human-friendly run folder name
export LATENT_LEN_LIST=48,64,80         # comma separated sweep values
export D_Z_LIST=192,256
export USE_GIST_HEAD=1                  # enable/disable gist reconstruction loss
```

Each run writes a structured diagnostics stream to `runs/<tag>/diagnostics.jsonl` capturing:

- Per-model losses (`tf`, `first`, `kCE`, `KD`, `gist`, etc.)
- First-token accuracy (`first_acc`)
- Gradient norms (`grad_*` components) for every supervised objective
- Latent dropout keep probability, current `K`, and wall-clock per step

This JSONL feed is what we use in Milestone 4 to check acceptance metrics on SQuAD smoke subsets before launching hero sweeps.

---

## 3. Stage layout (A/B/C)

1. **Stage A – Latent fit**: trains the shared encoder + adapters with text/latent alternating warm-up. Deep prefixes and the gist reconstruction head are enabled by default.
2. **Stage B – Prefix training**: continues with latent-only batches, longer schedules (`--hero`) and per-loss diagnostics to ensure the wire stays healthy.
3. **Stage C – Evaluation**: compares latent vs text prompting, including first-token acceptance, EM/F1, NLL/token, and compression metrics.

Checkpoints land under `runs/<tag>/ckpt/{stageA,stageB}`. We keep `encoder.pt`, `adapter_{model}.pt`, `deep_prefix_{model}.pt`, and `gist_{model}.pt` so eval/resume runs mirror the training artefacts exactly.

---

## 4. Useful toggles

| Flag | Description |
| --- | --- |
| `--use_deep_prefix --deep_prefix_len` | Enable P-Tuning-style per-layer KV prompts derived from the latent wire. |
| `--use_gist_head --gist_weight` | Turn on the gist reconstruction loss (Step 3 milestone). |
| `--grad_diag_interval` / `--grad_diag_components` | Controls how often we log gradient norms for each supervised objective. |
| `--diagnostic_log` | Path to the JSONL stream (both runners set this automatically). |

You can pass additional arguments through the runners by appending them after `--`.

---

## 5. Paper / proposal alignment

- Milestone 1: deep prefix injection is enabled through `--use_deep_prefix`.
- Milestone 2: latent/refiner sweeps and per-loss gradient diagnostics are exposed via the runner list variables and `Diagnosis.jsonl`.
- Milestone 3: gist reconstruction is on by default (`USE_GIST_HEAD=1`).
- Milestone 4: controlled experiments rely on `diagnostics.jsonl` for acceptance checkpoints.
- Milestone 5: hero prep simply means running with `--hero`; the longer Stage B schedule and updated documentation you’re reading now were part of this step.

---

## 6. Troubleshooting

- **Torch/PEFT mismatches**: rerun the script inside the project virtualenv so the helper installers can patch versions.
- **bitsandbytes import errors**: set `USE_GIST_HEAD=0` and `CUDA_VISIBLE_DEVICES=""` to force CPU smoke runs if you only need quick functional tests.
- **Structured logs**: open `runs/<tag>/diagnostics.jsonl` in your notebook or a simple Python REPL to trace how acceptance improves across steps.

Happy wiring! 🛠️
