# Qwen2.5-Math SVAMP32 Target-Query Source Bottleneck Manifest

- date: `2026-04-26`
- status: `failed_strict_small_gate`
- analyzer hash: `7fcaa9901ea5a78e23e7d4af8a64f513c0cf9da40ab1697fc0ea88c719143203`
- test hash: `3149ad0dbdedf7d03edf9004a0220626ede2d4c1dcd4ab075c10147dd8e2930a`
- result JSON hash: `06141d71be5fc57230aa7346525731618f554b023d7230c794ab681c34b05280`
- readout hash: `482a661d22065e93a83a0d9b2fb5cd5fb5c343d4d051a4eba70fc305bd7be9aa`
- log hash: `2c338d2c18ede9c25e2c4fc106bfa8d5a09524bb99fc2d5b5a495c6e1bb89636`

## Result

- matched: `7/32`, clean `0/6`
- target-only: `8/32`, clean `0/6`
- target-only-prefix: `8/32`, clean `0/6`
- projected-soft-prompt: `8/32`, clean `0/6`
- control clean union: `0/6`

## Decision

Do not scale this residue-classifier branch. It has no clean source-necessary
recovery and does not preserve the target floor.
