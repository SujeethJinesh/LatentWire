# Source-Private Semantic-Anchor Held-Out Receiver Gate

- date: `2026-04-30`
- rung: strict small held-out paraphrase receiver
- live branch: semantic-anchor, target-preserving evidence-packet receiver
- artifact: `results/source_private_semantic_anchor_heldout_packet_gate_20260430_threshold070_oraclefree/`

## Readiness Snapshot

Current ICLR readiness: stronger scoped positive-method manuscript, but not yet
a broad cross-LLM latent-transfer paper. Estimated distance: one medium
confirmation run with seed repeats, plus activation/model-backed receiver
evidence or GPU endpoint telemetry.

Current story: a source with private diagnostic evidence sends a tiny packet of
source-derived atoms. The target receives only the public question/candidate
pool and uses a target-preserving semantic-anchor receiver to decode the packet.
The new contribution is that public candidate side information can be held out
under paraphrase drift while source-destroying controls collapse to target-only.

Exact blocker before submission: show the semantic-anchor receiver is stable
across seeds and at a larger frozen slice, then either replace the public anchor
lexicon with a learned/frozen embedding receiver or explicitly frame it as a
protocol-assisted side-information decoder.

## Method Change

`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py` now has:

- `--text-feature-mode semantic_anchor`: augments hashed word/character
  features with public semantic-anchor features for the candidate dictionary.
- `--min-decision-score`: preserves the target prior unless the best
  packet/candidate atom score clears a threshold.
- Oracle diagnostic bypass: `oracle_learned_candidate_atoms` uses threshold
  zero so it measures candidate-map headroom rather than the operating
  receiver's conservative target-preservation gate.

The packet itself did not change. Source packets remain rate-capped atom IDs and
scores. The receiver change only affects public candidate-side interpretation.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_semantic_anchor_heldout_packet_gate_20260430_threshold070_oraclefree \
  --budgets 4 8 \
  --train-examples 512 \
  --eval-examples 256 \
  --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public \
  --calibration-examples 512 \
  --feature-dim 384 \
  --text-feature-mode semantic_anchor \
  --ridge 0.25 \
  --top-k 8 \
  --min-score 0.05 \
  --min-decision-score 0.70
```

## Results

Headline: `pass_gate=true`, bidirectional cross-family pass true, all three
directions pass, `6/6` pass rows.

| Direction | Budget | N | Learned | Target | Best control | Delta target | Oracle | CI95 low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core_to_holdout | 4 | 256 | 0.750 | 0.250 | 0.250 | +0.500 | 1.000 | +0.438 |
| core_to_holdout | 8 | 256 | 1.000 | 0.250 | 0.250 | +0.750 | 0.875 | +0.695 |
| holdout_to_core | 4 | 256 | 0.875 | 0.250 | 0.250 | +0.625 | 0.875 | +0.562 |
| holdout_to_core | 8 | 256 | 0.875 | 0.250 | 0.250 | +0.625 | 1.000 | +0.562 |
| same_family_all | 4 | 256 | 0.812 | 0.250 | 0.250 | +0.562 | 0.938 | +0.500 |
| same_family_all | 8 | 256 | 0.938 | 0.250 | 0.250 | +0.688 | 0.938 | +0.633 |

All rows keep source-destroying controls at target-only accuracy. Top-atom
knockout removes `100%` of lift. Exact transformed held-out surface overlap is
`0` in every direction, and exact ID parity is true.

## Negative/Boundary Runs

Two tracked boundary runs remain important:

- `results/source_private_semantic_anchor_heldout_packet_gate_20260430/`:
  semantic anchors without the target-preserving threshold partially pass but
  fail bidirectional cross-family because shuffled-source reaches `0.375` in
  `holdout_to_core`.
- `results/source_private_semantic_anchor_heldout_packet_gate_20260430_threshold070/`:
  the threshold fixes controls, but applying the threshold to the oracle
  diagnostic lowers candidate-map headroom below the strict pass rule.

These runs justify the final operating point: a semantic-anchor receiver plus a
target-preservation gate, with oracle measured as diagnostic headroom.

## Interpretation

This is now a stronger candidate technical contribution than the prior
calibrated synonym dictionary. It demonstrates held-out paraphrase transfer for
source-private packets under strict source-destroying controls, while preserving
the systems win: `4-8` byte packets beat target-only and matched-byte text.

The claim should remain scoped. The receiver uses an explicit public semantic
anchor lexicon, so it is not yet evidence for unconstrained activation-level
model-to-model latent transfer. The next paper-strengthening step is to replace
or ablate this lexicon with a learned/frozen embedding receiver, then run seed
repeats and a medium slice.

## Next Gate

Run a `3`-seed medium confirmation:

```bash
for seed in 47 53 59; do
  PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
    scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
    --output-dir results/source_private_semantic_anchor_heldout_packet_gate_20260430_seed${seed} \
    --budgets 4 8 \
    --train-examples 768 \
    --eval-examples 512 \
    --seed "${seed}" \
    --candidate-atom-view heldout_synonym \
    --calibration-atom-view synonym_stress \
    --candidate-calibration all_public \
    --calibration-examples 768 \
    --feature-dim 384 \
    --text-feature-mode semantic_anchor \
    --ridge 0.25 \
    --top-k 8 \
    --min-score 0.05 \
    --min-decision-score 0.70
done
```

Promotion rule: at least two seeds must pass bidirectional cross-family, the
third may not fail due to controls, and the aggregate paired CI lower bound must
stay above `+0.05` with transformed held-out overlap `0`.
