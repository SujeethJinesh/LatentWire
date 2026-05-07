# SSQ-LR Local Multilayer Capture Readout

This directory records the compact readout for a resource-limited local
SSQ-LR S1 smoke packet.

Decision: `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_S1_HETEROGENEITY`.

The full local run produced tensor artifacts and passed:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/shared/results/ssq_lr_local_multilayer_capture_20260507/ssq_lr_gate_packet \
  --mode real --project ssq_lr
```

The `.pt` tensor artifacts are intentionally not tracked here because the
packet is a non-promoting smoke run and duplicates tensor bytes in both the
source tensor packet and checker packet. Regenerate the full local packet with:

```bash
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.hybrid_manifest_local_capture_runner \
  --project ssq_lr --max-input-tokens 8 --ssq-layer-limit 4 \
  --output-dir experimental/shared/results/ssq_lr_local_multilayer_capture_20260507
```

Readout:

- rows: `16`
- prompts: `1`
- layers: `4`
- passing layers: `1`
- required passing layers: `3`
- selected S1 ratio: `0.9868217218`
- layer 0 max-abs ratio: `3.2938470385`
- layer 1 max-abs ratio: `1.3735632184`
- layer 2 max-abs ratio: `0.8810572687`
- layer 3 max-abs ratio: `0.8679756263`
