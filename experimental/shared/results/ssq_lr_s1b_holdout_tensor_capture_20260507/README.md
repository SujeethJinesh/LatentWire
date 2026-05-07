# SSQ-LR Held-Out S1b Tensor Capture

Date: 2026-05-07

This packet is resource-limited local evidence for SSQ-LR Gate S1b on held-out
reasoning prompts. It is not GPU evidence, not quality evidence, and cannot
promote SSQ-LR to the 5090 validation phase by itself.

Command:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_manifest_local_capture_runner \
  --project ssq_lr \
  --prompt-path experimental/shared/prompts/hybrid_reasoning_s1b_holdout_12_20260507.jsonl \
  --manifest-dir experimental/shared/results/hybrid_capture_manifests_s1b_holdout_20260507 \
  --ssq-prompt-limit 12 \
  --ssq-layers 0,12,18,30 \
  --max-input-tokens 8 \
  --output-dir experimental/shared/results/ssq_lr_s1b_holdout_tensor_capture_20260507
```

Checker:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/ssq_lr_s1b_holdout_tensor_capture_20260507/ssq_lr_gate_packet \
  --mode real --project ssq_lr
```

Result:

- Packet decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`
- Checker: pass after citing the held-out trace-plan registry via
  `trace_plan_config_path`.
- Rows: 192 saved-tensor rows from 12 prompts, 4 buckets, and layers
  `0,12,18,30`.
- Gate interpretation: layers `0,12,30` pass the S1 heterogeneity gate; layer
  `18` remains the near-miss/control.

The tensor `.pt` files under `ssq_lr_gate_packet/tensors/` and
`ssq_lr_tensor_packet/` are intentionally untracked. Re-run the command above
to regenerate them from the local Hugging Face cache.
