# Source-Private Latest-Model Generalization Scout

- date: `2026-04-28`
- gate: `latest_model_generalization_scout_20260428`
- status: model matrix added; Qwen3.5-0.8B CPU n64 passed; latest/MoE broad claim remains unproven

## Current Readiness

The existing ICLR package is ready for the scoped claim. This scout is a
post-package strengthening gate for model-family breadth.

## Answer

The method should plausibly generalize to MoE models because the source-side
task is not architectural: the source reads a private diagnostic line and emits
a compact packet. A dense model, sparse MoE model, or quantized MoE deployment
can all succeed if they preserve exact instruction following and short-code
copying.

We now have small-slice Qwen3.5 evidence: `Qwen/Qwen3.5-0.8B` passes the
hidden-repair packet gate on CPU at `n=16` and `n=64` after upgrading the repo-local
Transformers stack. This is useful compatibility evidence for the latest small
Qwen family, but it is not yet enough to claim latest-model or MoE
generalization. The paper should not claim MoE/latest-model generalization until
larger Qwen3.5 rows and off-machine Qwen3.6 MoE/FP8 rows pass the same
source-destroying controls.

## What I Checked

Official Hugging Face model cards/API identify these candidate rows:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.5-35B-A3B-FP8`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B-FP8`
- `Qwen/Qwen3.6-27B`

The Qwen3.6 `35B-A3B` row is the key MoE test: `35B` total parameters with `3B`
activated parameters. The FP8 row tests whether quantized deployment preserves
packet emission.

## Local Smoke Attempts

I attempted a local `n=16` smoke for `Qwen/Qwen3.5-0.8B`:

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16 \
  --model Qwen/Qwen3.5-0.8B \
  --device mps \
  --dtype float32 \
  --limit 16 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

It failed before generation because the repo-local venv had
`transformers==4.51.0`, while the cached Qwen3.5 config uses
`model_type: qwen3_5` and declares `transformers_version: 4.57.0.dev0`.

I upgraded the repo-local environment to:

- `transformers==5.7.0`
- `tokenizers==0.22.2`
- `huggingface_hub==1.12.0`

With that stack, the model loads locally as `Qwen3_5ForCausalLM`. The same MPS
run still fails before generation with an Apple MPS incompatible-dimensions
matmul in the hybrid-attention path, so MPS remains a backend compatibility
blocker rather than a source-packet failure.

The CPU fallback completed the `n=16` smoke and `n=64` confirmation:

```bash
HF_HOME=.hf_home ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_medium_20260429/benchmark.jsonl \
  --output-dir results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16_cpu \
  --model Qwen/Qwen3.5-0.8B \
  --device cpu \
  --dtype float32 \
  --limit 16 \
  --seed 29 \
  --max-new-tokens 8 \
  --prompt-mode trace_no_hint \
  --no-enable-thinking
```

Outcome:

- matched model packet: `16/16 = 1.000` and `64/64 = 1.000`
- target-only / best no-source: `4/16 = 0.250` and `16/64 = 0.250`
- zero-source, shuffled, random same-byte, answer-only, answer-masked,
  target-derived controls: all `4/16 = 0.250` at n16 and `16/64 = 0.250` at n64
- matched-minus-best-control: `+0.750`
- packet valid rate: `1.000`
- exact-ID parity: `true`
- median packet generation latency on CPU: `11525 ms` at n16 and `13471 ms` at n64

Artifacts:

- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16_cpu/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16_cpu/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16_cpu/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n64_cpu/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n64_cpu/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n64_cpu/predictions.jsonl`

## Added Matrix

Artifacts:

- `results/source_private_latest_model_matrix_20260428/latest_model_matrix.md`
- `results/source_private_latest_model_matrix_20260428/latest_model_matrix.json`
- `results/source_private_latest_model_matrix_20260428/manifest.md`

Code:

- `scripts/build_source_private_latest_model_matrix.py`
- `tests/test_build_source_private_latest_model_matrix.py`

The matrix now separates two contribution paths:

- Qwen latest/MoE path: Qwen3.5 small rows, then Qwen3.6 35B-A3B and FP8
  off-machine.
- Cross-family falsification path: `OLMo-2-0425-1B-Instruct`,
  `gemma-3-1b-it`, `granite-3.3-2b-instruct`, `SmolLM3-3B`,
  `Phi-4-mini-instruct`, `Ministral-3-3B`, `Nemotron-Nano-9B-v2`, and
  `Kimi-K2-Thinking`.

Reference memo:

- `references/480_latest_cross_family_model_scout_refs.md`

## Recommended Next Gate

1. Run `Qwen/Qwen3.5-0.8B` CPU `n=160` with at least one seed repeat.
2. Run `allenai/OLMo-2-0425-1B-Instruct` local `n=16` as the first non-Qwen
   falsification row.
3. Run `Qwen/Qwen3.5-2B` `n=16` if it fits local memory, otherwise CPU/off-machine.
4. If both pass, run `Qwen/Qwen3.5-4B` and widen the best small row to `n=160`.
5. Run `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.6-35B-A3B-FP8` off-machine at
   `n=32`, then `n=500` only if controls hold.

Pass rule remains unchanged: matched packets must beat target/no-source by at
least `15` points, and source-destroying controls must stay within `2` points
of target-only.

## Paper Impact

The Qwen3.5-0.8B `n=64` pass lets the paper add a modest post-package
compatibility contribution: the packet protocol is executable on the latest
small Qwen3.5 stack once dependencies are updated. If Qwen3.5 small and Qwen3.6
MoE rows pass at larger scale, the paper can strengthen its external-validity
claim from "works across Qwen3/Phi-3/Qwen2.5-era source emitters" to "also
transfers to latest small hybrid and sparse MoE source emitters." Until then,
keep the current scoped wording.
