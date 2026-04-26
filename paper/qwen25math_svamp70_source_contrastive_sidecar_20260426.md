# Qwen2.5-Math -> Qwen3 SVAMP70 Source-Contrastive Sidecar

- date: `2026-04-26`
- status: medium confirmation positive versus target/text, not C2C
- readiness: not ICLR-ready
- commit at run time: `d7b72008d015b880f79e2c76ef0ca97faf534374`

## Start Status

- current paper story: the strict-small Qwen2.5-Math -> Qwen3
  source-contrastive sidecar stack cleared SVAMP32 by using a target/text
  agreement guard plus a 1-byte source residue sidecar.
- exact blocker: verify whether the result survives a larger same-family
  surface and whether it approaches the C2C baseline.
- live branch: source-contrastive sidecar stack.
- scale-up rung: medium confirmation.

## Surface

SVAMP70 chat-template generation baselines:

| Method | Correct | Accuracy | Numeric Coverage |
|---|---:|---:|---:|
| source-alone | 13/70 | 0.186 | 61/70 |
| target-alone | 21/70 | 0.300 | 70/70 |
| text relay | 22/70 | 0.314 | 70/70 |
| C2C | 31/70 | 0.443 | 70/70 |

Source-contrastive target set:

- source-only over target: `9`
- clean source-only after text exclusion: `6`
- target-or-source oracle: `30/70`

## Guarded Sidecar Result

Best row uses moduli `2,3,5,7` or `97`, both one byte.

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 25 | 4 | 16 |
| zero-source | 21 | 0 | 13 |
| shuffled-source | 20 | 0 | 13 |
| label-shuffle | 19 | 0 | 13 |
| same-norm noise | 21 | 0 | 13 |
| target-only | 21 | 0 | 13 |
| slots-only | 14 | 0 | 10 |

- clean source-necessary IDs: `4/6`
- control clean union: `0/6`
- source numeric coverage: `61/70`
- sidecar bytes: `1`

Paired uncertainty:

| Comparison | Delta | Method-only | Baseline-only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|
| sidecar vs target | +0.0571 | 7 | 3 | [-0.0286, +0.1429] | 0.3428 |
| sidecar vs text | +0.0429 | 9 | 6 | [-0.0571, +0.1429] | 0.6056 |
| sidecar vs C2C | -0.0857 | 9 | 15 | [-0.2143, +0.0571] | 0.3074 |

## C2C-Composition Check

Using C2C as fallback with the same source sidecar and text agreement guard
fails:

- C2C fallback alone: `31/70`
- matched with source sidecar: `23/70`
- clean source-necessary: `1/6`
- control clean union: `4/6`

Decision: do not stack this sidecar naively on C2C.

## Textless Preservation Guard

Follow-up: replace the text-relay agreement guard with a cheaper source/target
guard. Apply the source residue sidecar only when the source produces a numeric
prediction and the source decoded output is shorter than the target decoded
output; otherwise keep target.

This uses only target and source rows, no text relay and no C2C.

Best row: moduli `2,3,5,7`, one byte.

| Condition | Correct | Clean Correct | Target-Self Correct |
|---|---:|---:|---:|
| matched | 26 | 4 | 14 |
| zero-source | 21 | 0 | 13 |
| shuffled-source | 20 | 0 | 13 |
| label-shuffle | 19 | 0 | 13 |
| same-norm noise | 20 | 0 | 12 |
| target-only | 21 | 0 | 13 |
| slots-only | 10 | 0 | 4 |

- clean source-necessary IDs: `4/6`
- control clean union: `0/6`
- sidecar bytes: `1`

Paired uncertainty:

| Comparison | Delta | Method-only | Baseline-only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|
| textless sidecar vs target | +0.0714 | 6 | 1 | [+0.0000, +0.1429] | 0.1306 |
| textless sidecar vs text | +0.0571 | 12 | 8 | [-0.0714, +0.1857] | 0.5023 |
| textless sidecar vs C2C | -0.0714 | 11 | 16 | [-0.2143, +0.0714] | 0.4414 |

Decision: promote the textless guard as the better systems branch. It removes
the text-relay preservation cost and improves the medium row from `25/70` to
`26/70`, but still is not ICLR-ready because uncertainty versus text crosses
zero and C2C remains stronger.

## Interpretation

This is a real medium-scale source-derived signal: the matched rows beat target
and text relay, recover clean source-only IDs, and source-destroying controls
do not recover clean IDs. It is not yet a paper headline because:

- paired bootstrap intervals versus target/text cross zero;
- it remains below C2C;
- the best systems row uses a brittle decoded-output-length guard that needs
  seed/surface replication;
- no seed/surface replication exists yet.

## Next Gate

Do not widen to 500 examples yet. The next branch should improve the method
surface before scale-up:

1. replace the text-relay preservation guard with a cheaper source/target
   confidence or source-quality guard, or
2. add a source-derived router that reduces target-correct losses without using
   C2C, then rerun SVAMP70 with paired uncertainty.

Promotion to large frozen slice requires:

- matched clearly above target and text with paired CI excluding zero, or a
  strong systems tradeoff;
- no clean control leakage;
- competitive behavior against C2C or a credible bytes/latency advantage.

## Artifacts

- generation manifest:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/manifest.md`
  - sha256: `1155f7f1eee547d0d36e6d62fb6305d9c01cf7042758e19e3cd293383033b0fa`
- guarded sidecar JSON:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_only_sidecar_router_t2t_guard.json`
  - sha256: `0d5971c8152650b31e2fda9ccf0b1263061f6adc045e488ed5feb92841e8389d`
- textless shorter-than-target guard JSON:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_shorter_than_target_guard_sidecar.json`
  - sha256: `19e5ec627968ea943c1483b2d6b19fffc8f642d51242c389ed1b341c0034cb81`
- textless shorter-than-target predictions:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_shorter_than_target_guard_predictions.jsonl`
  - sha256: `6b56da11c6846d4a86f8d12d5eb18ad3653ed1bd82fbe14212014b096bd85778`
- sidecar prediction JSONL:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_sidecar_t2t_guard_predictions.jsonl`
  - sha256: `4f8662b051d6df452f97c3c9791d48dcad361a15459bc31f50fdbe7c0e9ac83d`
- paired comparisons:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_target.md`
  - sha256: `e10c069362919592a414b550ec0e7322080fc15b144d4fdd97bdc7bde14d3e7d`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_text.md`
  - sha256: `eb4c8eb66b6882fcb97eb7fc094555cd47ed0a83d87cf6a829f51bad559de38d`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/paired_vs_c2c.md`
  - sha256: `2de7635ecec14de137db4e178d62940a9e6485cd8975f235f9a8c170166d2037`
- textless paired comparisons:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/shorter_guard_paired_vs_target.md`
  - sha256: `23e647976d95d23c20151d537f33425002c780f7f5c802a350d265a9485ba558`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/shorter_guard_paired_vs_text.md`
  - sha256: `a4a5be9476cf017712c906afdb04d1c1eabd5da3195a644d7da04ef0740dddef`
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/shorter_guard_paired_vs_c2c.md`
  - sha256: `124583b41ded0af0eda272bad4891442d379d5a6bff09ad700593adfc523505f`
- C2C fallback composition:
  - `results/qwen25math_qwen3_svamp70_source_surface_20260426/source_sidecar_c2c_fallback_t2t_guard.json`
  - sha256: `17e4a88171e737f52d5b8a106de9f7156b83641fd41533e33b30543244a91e07`

## Tests

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_only_sidecar_router_gate.py tests/test_build_source_contrastive_target_set.py -q
./venv_arm64/bin/python -m py_compile scripts/analyze_svamp32_source_only_sidecar_router_gate.py scripts/build_source_contrastive_target_set.py
```
