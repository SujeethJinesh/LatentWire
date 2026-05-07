# HORN H2 Noise Replay Scout

Decision: `FAIL_REAL_HORN_H2_DIRECTIONAL_NOISE_PROPAGATION`

This is a resource-limited Mac-local demotion scout. It is not H2 promotion,
not GPU evidence, and not a precision-allocation claim.

## Reproduce

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
HF_HOME="$PWD/.debug/hf_home" HF_HUB_CACHE="$PWD/.debug/hf_home/hub" \
  ./venv_arm64/bin/python -m experimental.shared.horn_h2_noise_replay_scout \
  --prompt-limit 2 --max-input-tokens 2 --seeds 1,2,3 --noise-scale 0.05
./venv_arm64/bin/python -m experimental.shared.followup_gate_contracts \
  experimental/shared/results/horn_h2_noise_replay_scout_20260507 \
  --gate horn_h2
```

## Readout

- Source H1a decision:
  `RESOURCE_LIMITED_NOT_PROMOTABLE_FAIL_REAL_H1A_DIRECTIONAL_ASYMMETRY_SCREEN`
- H1-selected direction: `ssm->attention`
- Directional drift ratio: `1.037371`
- Paired lower bound: `1.072249`
- Paired units: `6/6`
- Hook-off max delta: `0.0`
- Demotion recommendation: `DEMOTE_HORN_STANDALONE_WEAK_H2`

The result demotes HORN as a standalone branch. Use it as negative/control
evidence unless a future full H2/H3 reopening is preregistered.
