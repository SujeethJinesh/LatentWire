# Qwen2.5-Math -> Qwen3 SVAMP70 Source Confidence Diagnostics Manifest

- date: `2026-04-26`
- status: `source_confidence_router_fails_gate`
- git commit before diagnostics tooling: `992b93ac`

## Artifacts

- `source_diagnostics.jsonl`
  - sha256: `b17755be3db764f6130830cc516b18b6e4fadce7a78de36d20f10dd8c84c69b2`
- `source_diagnostics.md`
  - sha256: `28c36bebb71c3f698311916b3b5071984972063cb06fbb6a085a2a2350065977`
- `holdout_source_diagnostics.jsonl`
  - sha256: `2fc5226940ea4fc743324534bb51c938829910810619040f78afea2c905ecb0e`
- `holdout_source_diagnostics.md`
  - sha256: `3b7fb620579184aacaa7ddfa99cb63f7e00918a2f978929a67bc6a128efa87d0`
- `confidence_router.json`
  - sha256: `291ee7015a7b28f41f7c5e1b397e18b29da1b0781ae0f30c7c528ac3e860b4a8`
- `confidence_router.md`
  - sha256: `6f574b2864036520da9d769c3c4aac875f13ba3a9df65a7ca21dc5b4994baaa7`

## Decision

The source-internal confidence feature family is useful instrumentation but not
a positive method on this surface. Live CV is clean but weak; the frozen rule
fails on holdout with zero clean source-necessary IDs.
