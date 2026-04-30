# Source-Private Product-Codebook Model Receiver Diagnostics

Date: 2026-04-30

## Cycle Header

1. Current ICLR readiness and distance: stronger scoped positive-method
   candidate, not comfortable full ICLR. Distance is one reviewer-satisfying
   non-hand-coded/product-codebook receiver or a narrower final claim boundary,
   plus native GPU systems telemetry.
2. Current paper story: source-private packets can transmit private task
   evidence under strict destructive controls; semantic-anchor/scalar packets
   have seed-stable evidence, product-codebook packets give a learned discrete
   4-byte codec with low receiver-side decode, and direct Qwen target decoding
   supports model-mediated consumption for the diagnostic packet protocol.
3. Exact blocker to submission: product-codebook packets are still best decoded
   by a deterministic PQ L2 receiver, while model-mediated and learned
   replacement receivers do not yet improve or generalize.
4. Current live branch: scoped source-private packet communication with
   product-codebook systems frontier as supporting contribution.
5. Highest-priority gate this cycle: product-codebook-specific target receiver
   smoke, then masked-PQ consistency receiver if the prompt receiver failed.
6. Scale-up rung: strict-small smoke on Mac CPU for model receiver; n256 frozen
   product-codebook slice for learned receiver.

## Layman Summary

The source sends four tiny numbers that encode private evidence. The target has
public candidate information. We tried to make a small target model read those
numbers and pick the right candidate. It did not: it ignored the numeric packet
and kept picking its fallback choice. We then trained a tiny receiver module to
use the same packet while treating corrupted packets as "fall back to target
prior." That module can reproduce the hand-coded PQ distance decoder when
weighted toward matched packets, but it does not beat it.

## Implemented

- `scripts/run_source_private_product_codebook_target_decoder_smoke.py`
  - builds the same train/eval/product-codebook state as the product-codebook
    packet gate;
  - uses blinded `A/B/C/D` choices so prompt labels cannot leak example or
    patch indices;
  - supports signature and distance-table prompt modes;
  - includes target-only, zero-source, label-shuffled ridge, constrained
    shuffled source, answer-masked source, permuted-code, wrong-codebook,
    random same-byte, same-byte structured text, and target-derived controls;
  - reports exact-ID parity and paired bootstrap readouts.

- `scripts/run_source_private_masked_pq_consistency_receiver.py`
  - trains a tiny target-side linear score adapter over public PQ distance/rank
    features;
  - labels matched packets with the gold candidate and corrupted packets with
    the target-prior fallback;
  - supports masked matched-packet consistency rounds and matched/control loss
    weighting;
  - compares learned receiver accuracy against deterministic PQ L2 at the same
    byte budget.

## Results

### Product-Codebook Target-Decoder Prompt Receiver

Current reproducible prompt artifact:
`results/source_private_product_codebook_target_decoder_smoke_20260430/remap101_budget4_n16_distance_no_explicit_prior_cpu/`

Configuration:

```bash
./venv_arm64/bin/python scripts/run_source_private_product_codebook_target_decoder_smoke.py \
  --output-dir results/source_private_product_codebook_target_decoder_smoke_20260430/remap101_budget4_n16_distance_no_explicit_prior_cpu \
  --model Qwen/Qwen3-0.6B --device cpu --dtype float32 \
  --train-examples 512 --eval-examples 16 \
  --train-seed 29 --eval-seed 30 \
  --train-family-set all --eval-family-set all \
  --candidates 4 --feature-dim 512 --budget-bytes 4 \
  --ridge 0.01 --candidate-view slot --no-fit-intercept \
  --remap-slot-seed 101 --candidate-metadata-mode distance \
  --seed 29 --max-new-tokens 8 --no-enable-thinking --no-require-pass
```

Outcome:

| Row | Accuracy |
|---|---:|
| target-only | 0.312 |
| matched product-codebook packet | 0.312 |
| best control | 0.312 |

Pass gate: `False`. The model returned `A` for every condition after removing
the explicit target-prior line. Earlier prompt-anchored n32 diagnostics also
returned the target-prior distribution for every condition. Analytical probe on
the n32 surface showed why the signature-only prompt was not meaningful:
exact byte-overlap Hamming accuracy was only `0.281`, while deterministic PQ
L2 on the same slice was `0.562`.

Interpretation: Qwen3-0.6B on CPU is not a useful prompt-only product-codebook
receiver for numeric PQ tables. This prunes the current product-codebook
prompt-decoder branch.

### Masked-PQ Consistency Receiver

Unweighted artifact:
`results/source_private_masked_pq_consistency_receiver_20260430/remap101_budget4_n256/`

Weighted artifact:
`results/source_private_masked_pq_consistency_receiver_20260430/remap101_budget4_n256_weighted/`

Weighted command:

```bash
./venv_arm64/bin/python scripts/run_source_private_masked_pq_consistency_receiver.py \
  --output-dir results/source_private_masked_pq_consistency_receiver_20260430/remap101_budget4_n256_weighted \
  --train-examples 512 --eval-examples 256 \
  --train-seed 29 --eval-seed 30 \
  --train-family-set all --eval-family-set all \
  --candidates 4 --feature-dim 512 --budget-bytes 4 \
  --ridge 0.01 --receiver-ridge 0.01 \
  --candidate-view slot --no-fit-intercept \
  --remap-slot-seed 101 --seed 29 \
  --mask-rounds 2 --random-rounds 2 \
  --matched-weight 8 --mask-weight 4 \
  --control-weight 1 --target-only-weight 1 \
  --no-require-pass
```

Outcome:

| Receiver | Matched | Target | Best control | Source pass | Beats deterministic L2 |
|---|---:|---:|---:|---:|---:|
| unweighted masked-PQ | 0.250 | 0.250 | 0.250 | false | false |
| weighted masked-PQ | 0.582 | 0.250 | 0.273 | true | false |
| deterministic PQ L2 | 0.582 | 0.250 | 0.273 | n/a | n/a |

The weighted learned receiver exactly reproduces deterministic PQ L2 on every
reported condition; it does not improve accuracy, robustness, or latency. Its
current Python feature path is slower than the canonical L2 lookup, so it is
not a systems contribution.

## Decision

Prune the current product-codebook prompt receiver and masked-PQ consistency
adapter as headline contribution candidates.

Keep product-codebook packets as a supporting contribution because they already
pass the n256 source-control and paired-uncertainty gates, have a clean systems
frontier, and offer an interpretable learned discrete codec. Do not claim that
the target model can currently reason over PQ bytes in prompt text.

## Next Exact Gate

Prioritize one of:

1. A true feature-surface change for product-codebook packets, such as OPQ-like
   rotation/protected-basis PQ trained against source controls, with the same
   deterministic receiver and byte/SLO accounting.
2. Native GPU/server telemetry for the existing strongest packet rows.
3. A final claim-boundary decision: make the paper about controlled
   source-private packet communication plus systems accounting, with PQ as a
   compression-native supporting result, rather than broad cross-model latent
   reasoning.

The next Mac-local method gate should not be another prompt-only numeric PQ
receiver unless a stronger model endpoint is available.
