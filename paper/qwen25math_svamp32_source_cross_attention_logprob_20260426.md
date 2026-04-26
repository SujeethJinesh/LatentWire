# Qwen2.5-Math -> Qwen3 SVAMP32 Source Cross-Attention Logprob Gate

- date: `2026-04-26`
- status: `first_rung_fails_gate`
- rung: strict-small teacher-forced pre-generation smoke
- base commit: `a960d68256b8fd51e7108ce62cc6309a2367e89f`

## Question

After global summary soft-prefix connectors failed, can a token-local
target-query cross-attention connector recover clean C2C-headroom IDs by
attending directly over source token states?

This is still a pre-generation gate. The method is promotable only if matched
source beats source-destroying and target-only controls on clean IDs.

## Result

- clean IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `4/6`
- control-leak clean IDs: `4/6`
- mean matched-minus-best-control clean margin: `-0.383649`
- target-preservation IDs scored: `8`
- target-preservation matched-positive count: `5/8`

Clean rows:

| Example ID | Gold | Distractor | Matched Margin | Best Control | Best Control Margin | Delta |
|---|---:|---:|---:|---|---:|---:|
| `3e8a5691f5443495` | 1 | 3 | 0.642 | label_shuffled | 0.907 | -0.265 |
| `1d50b408c8f5cd2c` | 949 | 1 | 2.517 | shuffled_source | 2.941 | -0.424 |
| `de1bf4d142544e5b` | 57 | 2 | 2.194 | target_only_prefix | 2.991 | -0.797 |
| `47464cc0b064f172` | 24 | 2 | 2.798 | target_only_prefix | 3.214 | -0.416 |
| `6e9745b37ab6fc45` | 61 | 600 | -4.122 | target_only_prefix | -3.900 | -0.222 |
| `575d7e83d84c1e67` | 2 | 24 | -4.371 | label_shuffled | -4.193 | -0.178 |

## Decision

Fail this first-rung token-local cross-attention implementation. It improves
over the summary-prefix branch only in mean margin, but it recovers `0/6` clean
IDs and remains dominated by label-shuffled, shuffled-source, and target-only
controls.

Do not scale this exact prefix-emitting connector by epochs, folds, or hidden
dimension without a new hypothesis that directly addresses control dominance.

## Next Gate

Switch away from tiny learned prefix emitters on this surface. Highest-value
next moves are either:

- source-surface discovery for a slice where source/text/C2C headroom is less
  target-control dominated
- a discrete source-derived candidate/routing stack with strict attribution
  controls, using the process-repair result only as a target-side baseline and
  confound

Artifacts:

- `results/qwen25math_svamp32_source_cross_attention_logprob_20260426/manifest.md`
- `results/qwen25math_svamp32_source_cross_attention_logprob_20260426/smoke.md`
- `results/qwen25math_svamp32_source_cross_attention_logprob_20260426/smoke.json`
- `results/qwen25math_svamp32_source_cross_attention_logprob_20260426/sha256.txt`
