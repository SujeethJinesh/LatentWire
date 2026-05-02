# ARC Source-Control Contrastive Innovation Receiver

Date: 2026-05-02

## Status

- Current paper readiness: COLM workshop remains plausible; ICLR full paper is
  still blocked.
- Current story: LatentWire has a source-private fixed-byte packet scaffold,
  destructive controls, public-basis/innovation diagnostics, and systems
  byte/exposure accounting.
- Exact gap: the learned receiver still does not prove matched-source
  necessity against target-cache, source-destroying, and packet-permutation
  controls on a strict ARC/OpenBookQA surface.

## What Changed

Extended `scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`
with a source-control contrastive training option for the matched soft-prefix
connector:

- new CLI knobs: `--contrastive-weight`, `--contrastive-margin`,
  `--contrastive-loss-cap`, and `--contrastive-controls`;
- supported training controls: `zero_source`, `shuffled_source`,
  `same_norm_noise`, and `candidate_roll_source`;
- new report/pass control: `candidate_roll_source`, which rolls the candidate
  packet slots inside a row while preserving packet norm and row identity;
- capped pairwise margin penalty that asks the matched source packet to have a
  higher gold-vs-distractor margin than each corrupted source packet.

Lay explanation: the receiver is no longer trained only to answer correctly.
It is also punished when a broken version of the same source clue looks just as
good as the real clue. If the method is truly using source information, the
real packet should beat zeroed, shuffled, random, and candidate-rolled packets.

## Evidence

All runs use the same ARC n8 CPU `label_and_choice` surface with 4 fit rows and
4 eval rows, Qwen2.5-0.5B-Instruct as source, Qwen3-0.6B as target,
`hf_choice_hidden_score_public_innovation_candidate_pool_residual`, ridge 10,
4 source candidate tokens, 128 source feature dims, 64 target feature dims, and
contrastive controls
`zero_source,shuffled_source,same_norm_noise,candidate_roll_source`.

| Run | Epochs | Matched | Best Control | Margin Delta | Pass |
|---|---:|---:|---|---:|---|
| contrastive w0.2 | 1 | `3/4` | candidate-roll source `3/4` | `-0.043` | `False` |
| contrastive w0.2 | 3 | `1/4` | target-only `1/4` | `-0.234` | `False` |

Artifacts:

- `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_score_public_innovation_candidate_pool_residual_n8_cpu_label_choice_contrastive_w0p2/`
- `results/source_private_arc_openbookqa_soft_prefix_preflight_20260502_arc_hf_hidden_score_public_innovation_candidate_pool_residual_n8_cpu_label_choice_contrastive_w0p2_e3/`

Tests:

- `./venv_arm64/bin/python -m pytest tests/test_run_source_private_arc_openbookqa_soft_prefix_preflight.py -q`
- `./venv_arm64/bin/python -m py_compile scripts/run_source_private_arc_openbookqa_soft_prefix_preflight.py`

## Interpretation

This is not a positive-method result. The one-epoch smoke run is the most
promising row because matched accuracy rises to `3/4`, above target-only,
target-cache-only, zero-source, shuffled-source, same-norm noise, and visible
same-byte text. However, candidate-roll source also reaches `3/4`, and matched
does not win the margin gate. The three-epoch optimization control collapses
matched accuracy to `1/4`.

The candidate-roll tie is the important falsification. It suggests that this
soft-prefix receiver can use candidate-pool geometry or score factors without
being sensitive enough to the correct source-candidate alignment. That is not
the evidence reviewers need for cross-model latent reasoning.

## Decision

Weaken the current source-control contrastive soft-prefix branch. Do not widen
this exact method to n32/n64 or new benchmarks until a receiver passes the
candidate-roll source control.

The next method gate, if we continue the learned-receiver branch, should remove
the soft-prefix-only bottleneck and test a candidate-alignment-sensitive
receiver on a frozen slice: matched packets must beat zero-source,
shuffled-source, same-norm noise, train-mean source, candidate-roll source, and
candidate derangement with paired uncertainty.

The next systems-side gate that is useful on Mac is a byte-amplification
ablation: compare the same cached predictions under 4-5B framed packets,
64B cache-line padded packets, synthetic one-token QJL/TurboQuant/KVQuant byte
floors, and full hidden/KV relays. That strengthens the systems contribution
without claiming native NVIDIA throughput before we measure it.
