# LatentMAS CPU Probe Blocker

Command:

```bash
./venv_arm64/bin/python scripts/run_latentmas_competitor_eval.py \
  --latentmas-root references/repos/LatentMAS \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --method latent_mas \
  --task gsm8k \
  --prompt sequential \
  --eval-file data/gsm8k_eval_70.jsonl \
  --limit 1 \
  --prediction-output results/latentmas_competitor_20260421/qwen25_05b_gsm1_latent_mas_cpu_probe.jsonl \
  --device cpu \
  --device2 cpu \
  --max-new-tokens 32 \
  --latent-steps 1 \
  --generate-bs 1
```

Outcome:

- Wrapper exits cleanly.
- `qwen25_05b_gsm1_latent_mas_cpu_probe.blocker.json` records the failure.
- Failure type: `IndexError`
- Failure message: `index -1 is out of bounds for dimension 0 with size 0`
- No JSONL predictions were written.

Interpretation:

- The wrapper/runtime boundary is hardened.
- The remaining blocker is inside the vendor latent generation path, in HF cache-position handling during `generate()`, not in import wiring or probe orchestration.
