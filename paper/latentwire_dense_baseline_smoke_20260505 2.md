# LatentWire Dense Baseline Smoke, 2026-05-05

This artifact records the local work done to turn the C2C/KVComm caveat into
an attempted matched-baseline gate rather than a purely textual limitation.

## Converted MCQA Generation Rows

`scripts/materialize_mcqa_generation_eval.py` converts ARC-Challenge and
OpenBookQA rows into letter-only generation prompts so generation-oriented C2C
and KVComm evaluators can run against the same public multiple-choice surfaces.

Generated smoke files:

- `results/dense_baseline_mcqa_smoke_20260505/openbookqa_train_generation_smoke8.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/openbookqa_test_generation_smoke4.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/arc_train_generation_smoke8.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/arc_test_generation_smoke4.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/openbookqa_train_generation_n16.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/openbookqa_test_generation_n16.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/arc_train_generation_n16.jsonl`
- `results/dense_baseline_mcqa_smoke_20260505/arc_test_generation_n16.jsonl`

## C2C Smoke

Command family:

```bash
./venv_arm64/bin/python scripts/run_c2c_eval.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/dense_baseline_mcqa_smoke_20260505/openbookqa_test_generation_smoke4.jsonl \
  --device mps \
  --max-new-tokens 4 \
  --limit 4 \
  --prediction-output results/dense_baseline_mcqa_smoke_20260505/c2c_openbookqa_smoke4.jsonl
```

Local result:

| Task | rows | exact generation accuracy | parsed letter accuracy | mean latency |
|---|---:|---:|---:|---:|
| OpenBookQA smoke | 4 | 0.000 | 0.250 | 1.300s |
| ARC-Challenge smoke | 4 | 0.000 | 0.750 | 2.756s |

The exact generation matcher is too strict for MCQA because C2C can emit a
letter plus candidate text, so `scripts/summarize_mcqa_generation_smoke.py`
adds an answer-letter parser for this smoke gate. The current evaluator also
supports constrained first-token answer-letter decoding via
`--constrain-answer-letters`, which forces the same letter-only output shape
used by the target-only KVComm control.

Scaled constrained rows:

| Task | rows | parsed letter accuracy | unparsed | mean latency |
|---|---:|---:|---:|---:|
| OpenBookQA n16 | 16 | 0.438 | 0 | 0.789s |
| ARC-Challenge n16 | 16 | 0.688 | 0 | 0.841s |

These rows prove that the local C2C runtime path can execute on converted
matched-task prompts beyond n=4. They are still not a full C2C head-to-head
baseline because they use the available published artifact path and local MPS
generation surface rather than the original paper's native benchmark harness.

## KVComm Smoke

KVComm now runs on the same converted OpenBookQA and ARC smoke rows. Local fixes
added across the dense-baseline pass:

- eager attention selection via `--attn-implementation eager`;
- current Transformers `BatchEncoding` handling for Qwen chat templates;
- `DynamicCache.key_cache` / `value_cache` compatibility shim reuse.
- current Qwen3 `past_key_values` tracer signature with q/k norm;
- target-layer-complete, same-length source-cache construction so Qwen3's
  single causal mask is compatible with selective source layers;
- explicit target-side attention masks through `CVCommunicator.forward`, which
  removes the Qwen3 pad/eos attention-mask warning on constrained reruns;
- constrained first-token answer-letter decoding for matched, zero-source,
  shuffled-source, and target-only rows.

The earlier blocker was:

```text
RuntimeError: The size of tensor a (73) must match the size of tensor b (146)
at non-singleton dimension 3
```

That is fixed in the local compatibility shim. The prompt/scoring calibration
gate is also partly fixed: constrained decoding eliminates unparsed KVComm rows.
The remaining issue is substantive, not parsing: matched KVComm does not beat
target-only on the scaled local slices.

| Task | method/control | rows | parsed letter accuracy | unparsed |
|---|---|---:|---:|---:|
| OpenBookQA | C2C constrained | 16 | 0.438 | 0 |
| OpenBookQA | KVComm matched constrained | 16 | 0.188 | 0 |
| OpenBookQA | KVComm zero-source constrained | 16 | 0.188 | 0 |
| OpenBookQA | KVComm shuffled-source constrained | 16 | 0.375 | 0 |
| OpenBookQA | target-only constrained | 16 | 0.250 | 0 |
| ARC-Challenge | C2C constrained | 16 | 0.688 | 0 |
| ARC-Challenge | KVComm matched constrained | 16 | 0.062 | 0 |
| ARC-Challenge | KVComm zero-source constrained | 16 | 0.062 | 0 |
| ARC-Challenge | KVComm shuffled-source constrained | 16 | 0.250 | 0 |
| ARC-Challenge | target-only constrained | 16 | 0.688 | 0 |

Artifacts:

- `results/dense_baseline_mcqa_smoke_20260505/c2c_openbookqa_smoke4_rerun_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/c2c_arc_smoke4_rerun_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/kvcomm_openbookqa_smoke4_controls_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/kvcomm_arc_smoke4_controls_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/c2c_openbookqa_n16_constrained_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/c2c_arc_n16_constrained_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/kvcomm_openbookqa_n16_controls_constrained_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/kvcomm_arc_n16_controls_constrained_letter_summary.md`

The next dense-baseline gate is no longer cache/mask construction or malformed
answer decoding. It is a stronger matched baseline: either run native C2C/KVComm
with their paper harnesses on GPU, or diagnose why the local KVComm selected
source layers damage Qwen3 target predictions relative to target-only and
shuffled-source controls.

## KVComm Damage Diagnostic

The Mac-feasible diagnosis points away from useful source content and toward the
cache-injection path. `scripts/summarize_kvcomm_damage_diagnostic.py` compares
matched KVComm, zero-source KVComm, shuffled-source KVComm, and target-only on
the same examples.

First n16 run:

| Task | matched | zero-source | target-only | matched = zero predictions | matched damages target | matched repairs target |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | 0.062 | 0.062 | 0.688 | 1.000 | 0.625 | 0.000 |
| OpenBookQA | 0.188 | 0.188 | 0.250 | 1.000 | 0.188 | 0.125 |

Layer-fraction sweep (`0.04,0.07,0.1,0.25,0.5,1.0`, calibration limit 16):

| Task | best selected layers | matched | zero-source | target-only | matched = zero predictions | matched damages target | matched repairs target |
|---|---|---:|---:|---:|---:|---:|---:|
| ARC-Challenge | `[5]` | 0.250 | 0.250 | 0.688 | 1.000 | 0.562 | 0.125 |
| OpenBookQA | `[5, 7]` | 0.188 | 0.188 | 0.250 | 1.000 | 0.188 | 0.125 |

Artifacts:

- `paper/latentwire_kvcomm_damage_diagnostic_20260505.md`
- `paper/latentwire_kvcomm_damage_diagnostic_layer_sweep_20260505.md`
- `results/dense_baseline_mcqa_smoke_20260505/kvcomm_arc_n16_controls_layer_sweep_constrained_letter_summary.md`
- `results/dense_baseline_mcqa_smoke_20260505/kvcomm_openbookqa_n16_controls_layer_sweep_constrained_letter_summary.md`

Conclusion: prompt/scoring calibration and layer selection do not rescue the
local KVComm row. Matched and zero-source predictions are identical on every
paired n16 example in both tasks, including after the broader layer sweep. That
rules out "the selected source values are useful but noisy" as the local
failure explanation. The most likely culprit is the cache-prefix/position path:
feeding Qwen3 through a source-cache-shaped prefix changes the receiver's
answer distribution before any source content can help. The next useful gate is
therefore either a native/harness-faithful KVComm replication or a deeper
position/cache-ablation patch, not more prompt parsing.
