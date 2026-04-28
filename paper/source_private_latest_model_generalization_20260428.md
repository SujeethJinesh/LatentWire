# Source-Private Latest-Model Generalization Scout

- date: `2026-04-28`
- gate: `latest_model_generalization_scout_20260428`
- status: Qwen3.5-0.8B CPU n160 seed-repeat passed; Qwen3.5-2B n64 passed; Granite non-Qwen CPU n160 passed; MoE broad claim remains unproven

## Current Readiness

The existing ICLR package is ready for the scoped claim. This scout is a
post-package strengthening gate for model-family breadth.

## Answer

The method should plausibly generalize to MoE models because the source-side
task is not architectural: the source reads a private diagnostic line and emits
a compact packet. A dense model, sparse MoE model, or quantized MoE deployment
can all succeed if they preserve exact instruction following and short-code
copying.

We now have seed-stable medium-slice Qwen3.5 evidence: `Qwen/Qwen3.5-0.8B`
passes the hidden-repair packet gate on CPU at `n=16`, `n=64`, and `n=160`
after upgrading the repo-local Transformers stack, with the `n=160` row passing
on seeds `29` and `31`. This is useful latest-small evidence for the Qwen family,
but it is not yet enough to claim MoE generalization.

The next latest-small size also passes at confirmation scale: `Qwen/Qwen3.5-2B`
reaches `16/16 = 1.000` on CPU n16 and `64/64 = 1.000` on CPU n64, with
target/control at `0.250` and packet valid rate `1.000`.

We also have the first non-Qwen positive row: `ibm-granite/granite-3.3-2b-instruct`
passes at `n=160` on CPU under the copied-helper prompt. The cross-family claim
must be scoped: OLMo fails behaviorally with zero valid packets, and Granite's
stricter trace-no-hint row is positive but weaker than Qwen3.5.

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

The CPU fallback completed the `n=16` smoke, `n=64` confirmation, and `n=160`
medium row:

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

- matched model packet: `16/16 = 1.000`, `64/64 = 1.000`, and `160/160 = 1.000`
  on both n160 seeds (`29` and `31`)
- target-only / best no-source: `4/16 = 0.250`, `16/64 = 0.250`, and `40/160 = 0.250`
- zero-source, shuffled, random same-byte, answer-only, answer-masked,
  target-derived controls: all `4/16 = 0.250` at n16, `16/64 = 0.250` at n64,
  and no control above `41/160 = 0.256` at n160
- matched-minus-best-control: `+0.750` at n16/n64 and `+0.744` at n160
- packet valid rate: `1.000`
- exact-ID parity: `true`
- median packet generation latency on CPU: `11525 ms` at n16, `13471 ms` at n64,
  and `12059 ms` at n160

Artifacts:

- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16_cpu/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16_cpu/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n16_cpu/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n64_cpu/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n64_cpu/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n64_cpu/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed29/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed29/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed29/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed31/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed31/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_0_8b_trace_no_hint_n160_cpu_seed31/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n16_cpu_seed29/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n16_cpu_seed29/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n16_cpu_seed29/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n64_cpu_seed29/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n64_cpu_seed29/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n64_cpu_seed29/predictions.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n160_cpu_seed29/summary.json`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n160_cpu_seed29/model_packets.jsonl`
- `results/source_private_latest_model_matrix_20260428/qwen35_2b_trace_no_hint_n160_cpu_seed29/predictions.jsonl`

## Qwen3.5-2B n160 Confirmation

`Qwen/Qwen3.5-2B` now clears the same frozen `n=160` source-private packet gate
as the 0.8B row:

- matched model packet: `160/160 = 1.000`
- target-only: `40/160 = 0.250`
- best source-destroying control: `41/160 = 0.256`
- matched-minus-best-control: `+0.744`
- packet valid rate: `1.000`
- exact-ID parity: `true`
- median packet generation latency on CPU: `13878 ms`

This upgrades Qwen3.5-2B from medium confirmation (`n=64`) to a larger
latest-small cross-size confirmation. It does not by itself prove MoE or
cross-family generalization; it strengthens the same-generation Qwen3.5
emitter breadth claim.

## Cross-Family Rows

I ran the first non-Qwen falsification rows:

- `allenai/OLMo-2-0425-1B-Instruct`, MPS n16, trace-no-hint:
  `4/16 = 0.250`, packet valid rate `0.000`, controls `0.250`. This is a
  behavioral failure: mostly empty/code-fence outputs.
- `allenai/OLMo-2-0425-1B-Instruct`, MPS n16, copied-helper:
  `4/16 = 0.250`, packet valid rate `0.000`, controls `0.250`. This confirms
  the OLMo failure is not just the stricter prompt.
- `ibm-granite/granite-3.3-2b-instruct`, MPS n16, trace-no-hint:
  failed before generation with an Apple MPS matmul shape error. This is a
  backend compatibility failure, not method evidence.
- `ibm-granite/granite-3.3-2b-instruct`, CPU n16, trace-no-hint:
  `10/16 = 0.625`, packet valid rate `0.500`, controls `0.250`, pass.
- `ibm-granite/granite-3.3-2b-instruct`, CPU n16, copied-helper:
  `12/16 = 0.750`, packet valid rate `0.625`, controls `0.250`, pass.
- `ibm-granite/granite-3.3-2b-instruct`, CPU n64, copied-helper:
  `51/64 = 0.797`, packet valid rate `0.734`, controls `0.250`, pass.
- `ibm-granite/granite-3.3-2b-instruct`, CPU n160, copied-helper:
  `128/160 = 0.800`, packet valid rate `0.738`, best control `41/160 = 0.256`, pass.
- `ibm-granite/granite-3.3-2b-instruct`, CPU n64, trace-no-hint:
  `37/64 = 0.578`, packet valid rate `0.500`, controls `0.250`, pass but weaker.

Artifacts:

- `results/source_private_non_qwen_packet_20260428/allenai__olmo_2_0425_1b_instruct_n16_seed29_mps/summary.json`
- `results/source_private_non_qwen_packet_20260428/allenai__olmo_2_0425_1b_instruct_n16_seed29_copied_helper_mps/summary.json`
- `results/source_private_non_qwen_packet_20260428/ibm_granite__granite_3_3_2b_instruct_n16_seed29_cpu/summary.json`
- `results/source_private_non_qwen_packet_20260428/ibm_granite__granite_3_3_2b_instruct_n16_seed29_copied_helper_cpu/summary.json`
- `results/source_private_non_qwen_packet_20260428/ibm_granite__granite_3_3_2b_instruct_n64_seed29_copied_helper_cpu/summary.json`
- `results/source_private_non_qwen_packet_20260428/ibm_granite__granite_3_3_2b_instruct_n160_seed29_copied_helper_cpu/summary.json`
- `results/source_private_non_qwen_packet_20260428/ibm_granite__granite_3_3_2b_instruct_n64_seed29_trace_no_hint_cpu/summary.json`

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

MoE runbook:

- `paper/source_private_qwen36_moe_falsification_runbook_20260428.md`

## Recommended Next Gate

1. Run Qwen3.5-4B n16/n64 if local CPU time and disk permit, or run the
   off-machine Qwen3.6 MoE n32 row if CUDA serving is available.
2. Try one stricter Granite prompt-contract variant to reduce missing letter
   prefixes without using copied-helper.
3. Run Granite copied-helper n160 seed repeat if cross-family stability is more
   valuable than another Qwen size.
4. Run `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.6-35B-A3B-FP8` off-machine at
   `n=32`, then `n=500` only if controls hold.

Pass rule remains unchanged: matched packets must beat target/no-source by at
least `15` points, and source-destroying controls must stay within `2` points
of target-only.

## Paper Impact

The Qwen3.5-0.8B seed-stable `n=160` result plus Qwen3.5-2B n160 confirmation lets the
paper add a stronger post-package latest-small contribution: the packet protocol
is executable across two latest small Qwen3.5 sizes once dependencies are updated.
The Granite n160 pass adds a non-Qwen positive but under an easier copied-helper
prompt, so it supports cross-family feasibility and prompt-contract sensitivity
rather than a fully prompt-invariant claim. If Qwen3.5 small and Qwen3.6 MoE
rows pass at larger scale, the paper can strengthen its external-validity claim
from "works across Qwen3/Phi-3/Qwen2.5-era source emitters" to "also transfers to
latest small hybrid and sparse MoE source emitters." Until then, keep the
current scoped wording.
