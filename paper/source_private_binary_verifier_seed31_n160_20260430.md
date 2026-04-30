# Binary-Verifier Seed31 n160 Stability Gate

- date: `2026-04-30`
- artifacts:
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_core_n160_binary_logprob_combined_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_holdout_n160_binary_logprob_combined_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_seed31_core_holdout_n160_binary_logprob_combined_cpu/`
  - `results/source_private_verifier_consumption_trace_20260430/qwen3_seed31_core_holdout_n160_binary_logprob_combined_cpu/`
- status: pass for the current frozen-verifier seed-stability gate; still not
  a comfortable broad ICLR latent-transfer claim

## Cycle Start

1. Current ICLR readiness and distance: strong scoped positive-method /
   workshop-ready evidence, but not comfortably ICLR-full yet. Distance:
   one n500 or multi-seed receiver scale-up, one less table-shaped receiver, and
   native GPU/vLLM serving telemetry.
2. Current paper story: source-private residual communication with decoder side
   information. The source sees private evidence, sends a tiny packet, and a
   frozen target model with only public candidate metadata uses that packet to
   select the right candidate.
3. Exact blocker: the live positive row still depends on a public diagnostic
   table and a binary verifier prompt. Reviewers can accept this as
   source-private communication, but not as protocol-free latent reasoning.
4. Current live branch: balanced diagnostic 2-byte packet plus frozen
   Qwen3-0.6B binary-logprob verifier.
5. Highest-priority gate: repeat the combined-control receiver gate on seed31
   at n160 core and holdout after the seed29 n128 pass.
6. Scale-up rung: seed-stability strict-small to medium receiver confirmation.

## Layman Version

Imagine one model privately sees a clue and can send only a two-character code.
The other model sees four possible answers and a public tag on each answer. We
ask the receiving model four yes/no questions, one per candidate: "does this
candidate's public tag match the code?" When the real private code is sent, the
receiver picks the right answer every time. When the code is shuffled, random,
or converted into same-size visible text that carries no useful source signal,
the receiver falls back to chance.

The deranged-table control is slightly different: it rotates the public tag
table. That makes the receiver confidently follow the packet to the wrong
candidate, so accuracy drops to zero. This is useful causality evidence, but it
is not a target-like control.

## Commands

The machine was limited to the local Mac. The MPS guard was checked first:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

CPU was used for this gate.

Core:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_balanced_diag_cross_family_20260430/direct_core_n500_seed31/benchmark.jsonl \
  --output-dir results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_core_n160_binary_logprob_combined_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
  --limit 160 --seed 31 --max-new-tokens 1 --no-enable-thinking \
  --conditions target_only matched_packet deranged_candidate_diag_table shuffled_packet random_same_byte random_noncandidate_same_byte structured_json_2byte structured_free_text_2byte \
  --prompt-mode label --decode-mode candidate_binary_logprob \
  --progress-jsonl results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_core_n160_binary_logprob_combined_cpu/progress.jsonl \
  --partial-predictions-jsonl results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_core_n160_binary_logprob_combined_cpu/target_predictions.partial.jsonl \
  --progress-every 16
```

Holdout used the same command with
`direct_holdout_n500_seed31/benchmark.jsonl` and output directory
`qwen3_seed31_holdout_n160_binary_logprob_combined_cpu`.

Paired uncertainty:

```bash
./venv_arm64/bin/python scripts/summarize_source_private_target_decoder_uncertainty.py \
  --result-dirs \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_core_n160_binary_logprob_combined_cpu \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_holdout_n160_binary_logprob_combined_cpu \
  --output-dir results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_seed31_core_holdout_n160_binary_logprob_combined_cpu \
  --bootstrap-samples 5000 \
  --seed 20260430
```

Verifier consumption trace:

```bash
./venv_arm64/bin/python scripts/build_source_private_verifier_consumption_trace.py \
  --result-dirs \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_core_n160_binary_logprob_combined_cpu \
  results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed31_holdout_n160_binary_logprob_combined_cpu \
  --output-dir results/source_private_verifier_consumption_trace_20260430/qwen3_seed31_core_holdout_n160_binary_logprob_combined_cpu
```

## Results

| Surface | N | Matched | Target | Best control | Deranged table | Valid | p50 matched ms | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core | 160 | 1.000 | 0.250 | 0.250 | 0.000 | 1.000 | 1651.6 | `True` |
| holdout | 160 | 1.000 | 0.250 | 0.250 | 0.000 | 1.000 | 1674.1 | `True` |

Paired uncertainty:

- pass rows: `2/2`
- min matched minus target: `+0.750`
- min matched minus best control: `+0.750`
- min CI95 low vs target: `+0.681`
- min CI95 low vs best control: `+0.681`
- min valid prediction rate: `1.000`
- exact-ID parity: `True` on both surfaces

The matched packet wins against target-only with `120` paired wins and `40`
ties on each surface. Against random same-byte, it has `121` wins and `39`
ties on both surfaces. There are no paired losses in the artifact.

## Systems Consumption Trace

The seed31 consumption trace passes and is consistent with the seed29 n128
trace:

- matched source-boundary payload: `2` bytes;
- packet record with record overhead: `5` bytes;
- single request transfer accounting: `64` cache-line bytes and `128` DMA
  bytes;
- batch-64 packed accounting: `5.0` line bytes/request and `6.0` DMA
  bytes/request;
- target-side verifier cost: `4.0` binary forward passes/example;
- Mac CPU matched p50: `1651.6-1674.1` ms;
- Mac CPU matched p95: `1757.5-1773.5` ms.

This is a systems-accounting contribution, not a production serving result.
The source-boundary traffic is tiny, but the current receiver is compute-heavy
because it evaluates four candidate yes/no prompts per example.

## Literature and Reviewer Signals

The focused literature/reviewer memo is
`references/532_seed_stability_uniqueness_and_next_receivers_refs_20260430.md`.
It treats C2C and KVCOMM as high-rate cache-state competitors, LLMLingua and
RAG/tool traces as visible-text baselines, TurboQuant/QJL/KIVI/KVQuant as
compressed-KV systems comparators, and Wyner-Ziv/DISCUS as the right theory
language for decoder-side information.

Decision from the subagent pass:

- promote the frozen verifier only as source-private packet consumption with
  public decoder side information;
- do not market it as broad cross-model latent reasoning;
- use the packet/consumption trace as a separate systems contribution;
- next non-table method branch should be an anchor-relative sparse crosscoder
  or candidate-logit flow receiver, because those attack the public-table
  objection directly.

## Harness Hardening

`scripts/build_source_private_verifier_consumption_trace.py` now rejects
`target_predictions.partial.jsonl` by default. Scratch traces can opt in with
`--allow-partial-predictions`, but final artifacts must use finalized
`target_predictions.jsonl`. This prevents an unfinished target-decoder run from
being accidentally summarized as a final systems trace.

## Interpretation

This is a real positive row under the current threat model:

- matched private packets beat target-only and every chance-level same-byte or
  source-destroyed control by `+75` points;
- matched-byte text relays do not explain the gain;
- exact IDs match across conditions;
- random same-byte and shuffled packets remain at target prior;
- deranging public side information collapses the receiver, confirming that it
  is using the packet/table relation rather than answer priors.

It is not enough for a comfortable full ICLR claim by itself. The result is too
protocol-shaped, and the Mac CPU receiver is too slow for a systems headline.

## Next Exact Gate

`source_private_anchor_relative_sparse_crosscoder_receiver_n256`

Build a less table-shaped receiver that sends sparse source innovations
relative to public anchors. Required controls: public-only sparse classifier,
source shuffle, random same-byte, feature-ID permutation, top-feature knockout,
and matched-byte text. Pass rule: matched source packet beats target and best
control by at least `+15` points, CI95 low stays above `+10` points, controls do
not lift above target by more than `+2` points, and the packet has a clear byte
or compute tradeoff versus text/KV relays.

Parallel systems gate when hardware is available:
`source_private_batched_verifier_consumption_trace`, requiring model calls per
example to drop from `4.0` to `<=0.25` at batch `>=16` while preserving accuracy
and control parity.
