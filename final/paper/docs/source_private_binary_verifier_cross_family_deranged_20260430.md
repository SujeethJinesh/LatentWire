# Binary-Verifier Cross-Family Deranged-Control Gate

- date: `2026-04-30`
- artifacts:
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n64_binary_logprob_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_holdout_n64_binary_logprob_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_seed29_core_holdout_n64_binary_logprob_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n64_binary_logprob_deranged_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_holdout_n64_binary_logprob_deranged_cpu/`
  - `results/source_private_balanced_diag_target_decoder_20260430/paired_uncertainty_qwen3_seed29_core_holdout_n64_binary_logprob_deranged_cpu/`
- status: cross-family n64 exact-table receiver passes; deranged table control
  exposes the intended side-information boundary

## Cycle Start

1. Current ICLR readiness and distance: stronger scoped positive-method paper,
   still not comfortable as broad latent transfer. The distance is now a less
   protocol-shaped learned receiver or native serving telemetry, not another
   exact-table packet row.
2. Current story: source-private packets can transmit hidden diagnostic
   evidence to a target that has public candidate side information; the target
   can be a frozen Qwen receiver rather than only a hand-written parser.
3. Exact blocker: the receiver still uses explicit packet-to-public-handle
   equality, so reviewers can call it a side-information protocol rather than
   latent semantic reasoning.
4. Current live branch: calibrated binary-verifier receiver plus controls that
   define the claim boundary.
5. Highest-priority gate: cross-family core/holdout n64 receiver replication
   and deranged public-handle control.
6. Scale-up rung: strict-small cross-family receiver gate.

## Layman Version

The target sees four possible fixes. Each fix has a public short handle. The
source privately sees a hidden test and sends a two-character clue. I asked a
small Qwen model whether the clue matches each public handle. If the public
handles are correct, Qwen picks the right fix every time. If I rotate the
public handles onto the wrong fixes, Qwen confidently follows the wrong public
handle and accuracy drops to zero. That means the packet is real, but the
receiver depends on the public side-information table.

## Harness Additions

`scripts/run_source_private_tool_trace_target_decoder_smoke.py` now includes
two additional reviewer controls:

- `deranged_candidate_diag_table`: sends the real source packet but rotates the
  candidate `handles_repair_diag` table. This tests whether the receiver follows
  public side information rather than some hidden answer artifact.
- `random_noncandidate_same_byte`: sends a valid two-character packet that is
  guaranteed not to match any candidate handle. This avoids accidental random
  collisions.

`scripts/summarize_source_private_target_decoder_uncertainty.py` now treats both
conditions as source-destroying controls when present.

## Cross-Family Exact-Table Receiver

Commands followed this form for `core` and `holdout`:

```bash
env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_balanced_diag_cross_family_20260430/direct_core_n500_seed29/benchmark.jsonl \
  --output-dir results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n64_binary_logprob_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
  --limit 64 --seed 29 --max-new-tokens 1 --no-enable-thinking \
  --conditions target_only matched_packet shuffled_packet random_same_byte structured_json_2byte structured_free_text_2byte \
  --prompt-mode label --decode-mode candidate_binary_logprob
```

Results:

| Surface | N | Matched | Target | Best control | Valid | p50 matched ms | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| core | 64 | 1.000 | 0.250 | 0.250 | 1.000 | 1373.0 | `True` |
| holdout | 64 | 1.000 | 0.250 | 0.250 | 1.000 | 1377.0 | `True` |

Paired uncertainty:

- rows: `2`
- pass rows: `2`
- min matched-target: `+0.750`
- min matched-best-control: `+0.750`
- min CI95 low vs target: `+0.641`
- min CI95 low vs best control: `+0.641`
- min valid prediction rate: `1.000`

## Deranged Public-Handle Control

Commands followed this form:

```bash
env PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  --benchmark-jsonl results/source_private_balanced_diag_cross_family_20260430/direct_core_n500_seed29/benchmark.jsonl \
  --output-dir results/source_private_balanced_diag_target_decoder_20260430/qwen3_seed29_core_n64_binary_logprob_deranged_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
  --limit 64 --seed 29 --max-new-tokens 1 --no-enable-thinking \
  --conditions target_only matched_packet deranged_candidate_diag_table random_noncandidate_same_byte \
  --prompt-mode label --decode-mode candidate_binary_logprob
```

Results:

| Surface | N | Matched | Target | Deranged table | Random noncandidate | Valid | Pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| core | 64 | 1.000 | 0.250 | 0.000 | 0.250 | 1.000 | `True` |
| holdout | 64 | 1.000 | 0.250 | 0.000 | 0.250 | 1.000 | `True` |

Paired uncertainty:

- rows: `2`
- pass rows: `2`
- min matched-target: `+0.750`
- min matched-best-control: `+0.750`
- min CI95 low vs target: `+0.641`
- min CI95 low vs best control: `+0.641`
- min valid prediction rate: `1.000`

## Interpretation

This strengthens source causality and weakens leakage explanations:

- the real packet works on both core and holdout family sets;
- shuffled/random/truncated/collision-free controls collapse to target;
- the deranged public table collapses to zero accuracy, so the receiver is
  following the public side-information contract rather than answer priors.

It also sharpens the limitation:

- this is not protocol-free semantic transfer;
- the source packet only helps because the target has a correct public mapping
  from packet values to candidate metadata;
- the binary receiver is still an exact equality verifier.

Reviewer-facing claim:

> LatentWire provides source-private, rate-capped side-information
> communication: a frozen target model can consume a 2-byte packet under strict
> source-destroying controls, and deranged-table controls show the dependency on
> public decoder side information.

Do not claim:

- broad cross-family latent reasoning;
- C2C/KVComm dominance;
- production latency wins from the Mac CPU verifier.

## Subagent Integration

Systems/quantization scout: the systems contribution should be an auditable
packet-interface trace card. TurboQuant, QJL, KIVI, KVQuant, vLLM,
FlashAttention, DistServe, and GainSight are baselines/framing sources. The next
Mac systems gate should report raw bytes, cache-line bytes, DMA bytes,
batch-packed bytes/request, p50/p95 encode/decode, and source text/KV exposure.

Diffusion/JEPA scout: do not keep tuning the existing JEPA-Q threshold family.
The most credible less-protocol-shaped branch is n500 masked-consistency with
public-only separation, then a tiny posterior-consistency or flow-matching
receiver over candidate logits/features.

Reviewer/data-audit scout: the main rejection risk is public-table lookup. This
gate deliberately confirms that boundary; it should be presented as
side-information communication, not as semantic latent transfer.

## Next Exact Gate

`source_private_masked_consistency_receiver_disjoint_n500_20260501`

Run full, semantic, and remapped-slot views at train 512 / eval 500 over two
seed pairs, with a public-only learned receiver ablation. Promotion rule:
packet-matched full/semantic accuracy must beat target and best destructive
control by `>= +0.15`, paired CI95 low must exceed `+0.10`, controls must stay
within target `+0.05`, slot-remap opaque lift must stay `<= +0.05`, and
public-only lift must explain less than `25%` of packet lift.

Systems follow-up when not using NVIDIA:
`source_private_packet_trace_card_v2_20260501`, using existing artifacts to
report hierarchy-aware traffic and exposure.

## Tests

```bash
./venv_arm64/bin/python -m py_compile \
  scripts/run_source_private_tool_trace_target_decoder_smoke.py \
  scripts/summarize_source_private_target_decoder_uncertainty.py

./venv_arm64/bin/python -m pytest \
  tests/test_run_source_private_tool_trace_target_decoder_smoke.py \
  tests/test_summarize_source_private_target_decoder_uncertainty.py
```

Outcome during development: `18 passed in 0.09s`.
