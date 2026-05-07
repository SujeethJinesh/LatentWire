# SSQ-LR Prompt-Repeat Tensor Capture

Decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_PASS_REAL_S1_HETEROGENEITY`

This packet repeats the four layers selected by the all-layer metrics scout
(`0`, `12`, `18`, `30`) across the frozen 12-prompt Granite Tiny smoke surface.
It is a tensor-provenance packet and passes the real SSQ-LR checker, but it is
not promotable because the layer subset was chosen post-hoc from the all-layer
scout.

Key readout:

- Rows: `192` (`12` prompts x `4` layers x `4` buckets)
- Checker: passes `check_gate_packet --mode real --project ssq_lr`
- Passing layers: `3 / 4`
- Required passing layers: `3`
- Selected S1 ratio: `2.561113`
- Selected S1 CI low: `2.014131`
- Holm minimum p-value: `2.775512e-05`
- Passing layer indices: `0`, `12`, `30`

The full local tensor packet is intentionally not tracked because it is a
resource-limited smoke artifact and stores duplicated `.pt` tensors. Regenerate
it with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_manifest_local_capture_runner \
  --project ssq_lr --ssq-prompt-limit 12 --ssq-layers 0,12,18,30 \
  --max-input-tokens 8 \
  --output-dir experimental/shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/ssq_lr_prompt_repeat_tensor_capture_20260507/ssq_lr_gate_packet \
  --mode real --project ssq_lr
```

Claim boundary: this keeps SSQ-LR alive as a layer-selective hypothesis only.
It does not justify GPU validation until the layer subset is frozen into a
fresh gate or the branch clears S2 quantization sensitivity.
