# Source-Private Contrastive Receiver Smoke

- date: `2026-04-30`
- gate: `source_private_contrastive_receiver_smoke`
- status: mixed; semantic-anchor receiver passes, frozen BGE receiver remains
  negative but now has a sharper control-aware boundary

## Question

Can a small contrastive receiver turn a source-private packet into target-side
candidate scores without relying only on the explicit semantic-anchor decoder?

## Implementation

I added `--receiver-mode contrastive_bilinear` to
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`.
The receiver fits a bilinear score

```text
score(candidate, packet) = feature(candidate)^T W atom_vector(packet) + b
```

using public calibration examples. The implementation uses a dual ridge solve
when the bilinear feature map is wider than the number of calibration rows, so
frozen BGE features are feasible on the Mac CPU.

I also added `--contrastive-negative-sources K`. With `K > 0`, each calibration
example gets `K` shuffled source packets as explicit zero-label negatives. This
turns the receiver into a source-control contrastive objective rather than a
plain matched-source classifier.

## Commands

Semantic-anchor functional smoke:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_contrastive_receiver_semantic_anchor_smoke_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 32 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 384 --text-feature-mode semantic_anchor \
  --receiver-mode contrastive_bilinear \
  --ridge 0.25 --top-k 8 --min-score 0.05 --min-decision-score 0.30
```

Frozen BGE bilinear smoke:

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_contrastive_frozen_receiver_bge_smoke_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 32 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 768 --text-feature-mode hf_mid_last_mean \
  --receiver-mode contrastive_bilinear \
  --feature-model BAAI/bge-small-en --feature-device cpu \
  --feature-dtype float32 --feature-max-length 96 --local-files-only \
  --ridge 0.25 --top-k 8 --min-score 0.05 --min-decision-score 0.30
```

Frozen BGE with source-control negatives:

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_control_contrastive_frozen_receiver_bge_smoke_20260430 \
  --budgets 2 4 --train-examples 64 --eval-examples 32 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 768 --text-feature-mode hf_mid_last_mean \
  --receiver-mode contrastive_bilinear \
  --contrastive-negative-sources 2 \
  --feature-model BAAI/bge-small-en --feature-device cpu \
  --feature-dtype float32 --feature-max-length 96 --local-files-only \
  --ridge 0.25 --top-k 8 --min-score 0.05 --min-decision-score 0.30
```

## Results

| Run | Pass | Pass rows | Best learned | Best lift | Control status |
|---|---:|---:|---:|---:|---|
| semantic-anchor contrastive, n32 | `True` | 4/6 | 1.000 | +0.750 | source-destroying controls clean on passing rows |
| frozen BGE contrastive, n32 | `False` | 1/6 | 1.000 | +0.750 | shuffled-source controls leak in holdout/same-family |
| frozen BGE + 2 shuffled-source negatives, n32 | `False` | 2/6 | 0.750 | +0.500 | strict controls clean, but lift drops |

Detailed BGE boundary:

- Plain BGE contrastive has real matched signal: core -> holdout reaches `0.75`
  at 4 bytes with target/control `0.25`; holdout -> core reaches `1.00`.
- The same plain BGE run fails promotion because shuffled-source controls rise
  to `0.50` in holdout -> core and `0.4375` in same-family.
- Adding shuffled-source negatives fixes that leakage: every strict
  source-destroying control is at target (`0.25`) in all passing rows.
- The cost is lower matched accuracy. The control-aware BGE run only passes
  holdout -> core and same-family at 4 bytes (`0.625` vs target/control
  `0.25`), while core -> holdout at 4 bytes stays at `0.75` but misses the
  current private-random-knockout robustness rule.

Exact ID parity holds in all runs. Exact transformed held-out surface overlap is
`0`; native surface overlap remains nonzero because `all_public` calibration can
contain untransformed public candidate strings.

## Interpretation

This is useful but not yet a paper headline.

The positive part is that a bilinear source-packet receiver over frozen BGE
features can recover strong matched signal. The reviewer-critical part is that
the signal is partly explained by source-control leakage unless shuffled-source
negatives are included during fitting. The control-aware objective fixes the
leakage, which supports the source-control contrastive framing, but the reduced
lift means the frozen receiver should remain an ablation.

Decision: do not promote frozen BGE contrastive as a new claim. Keep it as a
methodological boundary and use it to justify the next architecture: a
query-resampler / information-bottleneck receiver trained with source-destroying
negatives and a rate-distortion sweep.

## Next Gate

Implement a tiny query-resampler receiver, still Mac-local:

- source input: private packet atoms or frozen source/candidate features
- bottleneck: 4/8/16 learned query vectors or a low-rank bilinear factorization
- loss: candidate ranking plus shuffled-source and random-same-byte negatives
- controls: zero-source, shuffled-source, random same-byte, answer-only,
  answer-masked, target-derived sidecar, atom derangement
- first pass threshold: `n=64`, at least one bidirectional row with matched
  accuracy >= target + `0.25`, all destructive controls <= target + `0.03`,
  and no exact transformed surface overlap
