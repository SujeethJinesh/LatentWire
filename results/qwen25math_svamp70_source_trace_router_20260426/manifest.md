# Qwen2.5-Math -> Qwen3 SVAMP70 Source-Trace Router Manifest

- date: `2026-04-26`
- status: `source_trace_router_fails_gate`
- git commit at run start: `2a5cb36368f709eb2de49a2c3cf36f06fccedc8c`

## Artifacts

- `trace_router.json`
  - sha256: `e4e5600e139efbf7bc068ff2117e172cba9f87055e9477f51839a90175c54c03`
- `trace_router.md`
  - sha256: `e439be6bf00ec99ccf9f63aa72ca0b3a47f423d4a950155846d7417322e99a6c`

## Decision

The frozen source-trace self-consistency router fails the live/holdout gate.
It has no standard clean-control leakage, but it is too weak on live CV, harms
two target-correct examples, and the single holdout clean source-necessary ID
survives equation-result permutation. Do not promote this text-trace router as
real communication.
