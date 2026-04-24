# SVAMP32 Syndrome Sidecar Probe Results - 2026-04-24

## Summary

- status: `syndrome_sidecar_bound_clears_gate_not_method`
- gate: compact C2C-derived residue must recover at least `2/6` clean SVAMP32
  residual IDs from target-side candidate pools while zero/shuffle/target-only/
  slots-only controls recover none
- strict target-pool outcome: `[2,3,5,7]` residue set, `1` byte, matched
  `14/32`, target-only fallback `14/32`, clean source-necessary `2/6`
- augmented sensitivity outcome: `[2,3,5,7]` residue set, `1` byte, matched
  `15/32`, target-only fallback `14/32`, clean source-necessary `3/6`

This is a feasibility bound, not a deployable method. It uses C2C numeric
answers as a proxy syndrome and does not prove source latent states can predict
the syndrome.

## Tracked Artifacts

- `targetpool_syndrome_probe.json`
- sha256: `48f94eb7f14081b7c2b662a207dfdc96a2b81e3df4cfd54e919a2e55e3891ffb`
- `targetpool_syndrome_probe.md`
- sha256: `007f405df8536417c73e84546de8c1ad8e30c91d6e67f0bfa5dab6cb64563bd1`
- `augmentedpool_syndrome_probe.json`
- sha256: `aa108b5b3ffd2e78acc8010e2f39c45876c79497e7e4944a15f719eecd46dfd5`
- `augmentedpool_syndrome_probe.md`
- sha256: `1c91b084fdf5b392e22feef57086b33735f9619a4962e423019074bc9d9f7c0e`

## Scratch Logs

- `.debug/svamp32_syndrome_sidecar_probe_20260424/logs/targetpool_syndrome_probe.log`
- sha256: `5416a7dbc8a8a45e802f9413a6ce69a03f5c6848476898db8fc142519a2f1acb`
- `.debug/svamp32_syndrome_sidecar_probe_20260424/logs/augmentedpool_syndrome_probe.log`
- sha256: `c3e35b73aa7530c3914a11e8a14c95f750c8d645cafc037dff19d640aa5760b2`

## Interpretation

The strict target-side pool clears the bound with gold present in only `2/6`
clean candidate pools. That is exactly enough to justify training a source-
latent syndrome predictor, but not enough to claim a method row. The next gate
must replace C2C numeric answers with matched source latents and keep the same
controls.
