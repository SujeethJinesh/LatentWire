# Binary-Verifier n128 Scale-Up and Consumption Trace

- date: `2026-04-30`
- artifacts:
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n128_binary_logprob_combined_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_holdout_n128_binary_logprob_combined_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_seed29_core_holdout_n128_binary_logprob_combined_cpu/`
  - `results/source_private_verifier_consumption_trace_20260430/qwen3_seed29_core_holdout_n128_binary_logprob_combined_cpu/`
- status: pass as a larger frozen cross-family receiver gate; strengthens the
  systems accounting but does not make a production GPU serving claim

## Cycle Start

1. Current ICLR readiness and distance: stronger scoped positive-method paper,
   still not comfortable broad latent-transfer ICLR. The distance is now one
   seed-stable larger frozen slice, one less table-shaped learned receiver, or
   native serving telemetry.
2. Current paper story: a source with private evidence sends a 2-byte packet;
   a target with public candidate side information and no source text/KV can
   use a frozen Qwen verifier to select the correct candidate.
3. Exact blocker: the receiver still verifies an explicit packet-to-public
   handle contract, so the headline must be source-private communication with
   decoder side information, not protocol-free latent semantic transfer.
4. Current live branch: balanced diagnostic packet plus frozen Qwen3-0.6B
   binary verifier.
5. Highest-priority gate: scale the cross-family receiver from n64 to n128
   with all source-destroying controls in one combined run.
6. Scale-up rung: strict-small to early medium receiver confirmation.

## Layman Version

The source privately sees a clue and can only send a two-character code. The
target sees four candidate fixes and the public code attached to each fix. I
asked a small Qwen model four yes/no questions: "does the source code match
this candidate's public code?" If the real source code is present, Qwen picks
the right fix every time. If the source code is shuffled, random, non-matching,
or the public code table is rotated, the gain disappears.

## Commands

CPU was used because the Mac MPS smoke with Qwen3-0.6B failed with an MPS
matmul shape-inference error. The MPS guard was checked before the attempt:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Core:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_balanced_diag_cross_family_20260430/direct_core_n500_seed29/benchmark.jsonl \
  --output-dir results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n128_binary_logprob_combined_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
  --limit 128 --seed 29 --max-new-tokens 1 --no-enable-thinking \
  --conditions target_only matched_packet deranged_candidate_diag_table shuffled_packet random_same_byte random_noncandidate_same_byte structured_json_2byte structured_free_text_2byte \
  --prompt-mode label --decode-mode candidate_binary_logprob \
  --progress-jsonl results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n128_binary_logprob_combined_cpu/progress.jsonl \
  --partial-predictions-jsonl results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n128_binary_logprob_combined_cpu/target_predictions.partial.jsonl \
  --progress-every 16
```

Holdout used the same command with
`direct_holdout_n500_seed29/benchmark.jsonl` and output directory
`qwen3_seed29_holdout_n128_binary_logprob_combined_cpu`.

Paired uncertainty:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_target_decoder_uncertainty.py \
  --result-dirs \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n128_binary_logprob_combined_cpu \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_holdout_n128_binary_logprob_combined_cpu \
  --output-dir results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_seed29_core_holdout_n128_binary_logprob_combined_cpu \
  --bootstrap-samples 5000 \
  --seed 20260430
```

Verifier consumption trace:

```bash
./venv_arm64/bin/python scripts/build_source_private_verifier_consumption_trace.py \
  --result-dirs \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n128_binary_logprob_combined_cpu \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_holdout_n128_binary_logprob_combined_cpu \
  --output-dir results/source_private_verifier_consumption_trace_20260430/qwen3_seed29_core_holdout_n128_binary_logprob_combined_cpu
```

## Results

| Surface | N | Matched | Target | Best control | Deranged table | Valid | p50 matched ms | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core | 128 | 1.000 | 0.250 | 0.250 | 0.000 | 1.000 | 1665.5 | `True` |
| holdout | 128 | 1.000 | 0.250 | 0.250 | 0.000 | 1.000 | 1630.8 | `True` |

Paired uncertainty:

- pass rows: `2/2`
- min matched minus target: `+0.750`
- min matched minus best control: `+0.750`
- min CI95 low vs target: `+0.672`
- min CI95 low vs best control: `+0.672`
- min valid prediction rate: `1.000`
- exact ID parity: `True` on both surfaces

## Systems Consumption Trace

`scripts/build_source_private_verifier_consumption_trace.py` converts the
target-decoder JSONL into a reviewer-facing systems trace:

- matched source-boundary payload: `2` bytes;
- packet record with header/check overhead: `5` bytes;
- single request transfer accounting: `64` cache-line bytes and `128` DMA
  bytes;
- batch-64 packed accounting: `5.0` line bytes/request and `6.0` DMA
  bytes/request;
- target-side verifier cost: `4.0` binary forward passes/example;
- Mac CPU matched p50: `1630.8-1665.5` ms.

This strengthens the systems story by separating boundary traffic from
receiver compute. It also sharpens the non-claim: the current frozen verifier
is accurate and source-private but not a production latency win until the
receiver is batched, fused, distilled, or served in a native GPU stack.

## Literature and Systems Framing

The new systems memo is
`references/531_systems_novelty_private_packet_verifier_refs_20260430.md`.
It positions this row against TurboQuant/QJL/KIVI/KVQuant-style compact state
transport, C2C/KVCOMM cache transfer, vLLM/PagedAttention and DistServe serving
metrics, FlashAttention-style IO accounting, and diffusion/consistency/flow
inspiration for future learned receivers.

Decision from the literature pass:

- do not claim LatentWire replaces KV compression or cache reuse;
- claim a different operating point: source-private, low-rate evidence packets
  consumed with public decoder side information;
- next systems work should measure verifier consumption under batching and
  eventually native TTFT/TPOT/goodput.

## Interpretation

This is the strongest current frozen receiver row:

- real source packet transfers private evidence on both cross-family surfaces;
- every source-destroying and matched-byte control stays at target prior;
- deranging the public handle table collapses accuracy to zero, showing the
  receiver follows the public side-information contract rather than an answer
  prior;
- exact-ID parity and valid prediction rate are clean.

The limitation is equally important: this remains source-private communication
with decoder side information, not model-internal latent reasoning. It is a
defensible scoped contribution, but not yet the broad latent-transfer paper.

## Next Exact Gate

`source_private_binary_verifier_seed31_n128_or_n160`

Run the same combined-control cross-family receiver on seed31, preferably n160
if runtime is acceptable, then add a learned/diffusion-style candidate-logit
receiver only if the frozen row remains stable. Pass rule: matched stays
`>= +0.15` over target and best control, CI95 low remains `> +0.10`, all
destructive controls stay within target `+0.05`, valid rate remains `>=0.95`,
and consumption trace reports the same source-boundary accounting.

