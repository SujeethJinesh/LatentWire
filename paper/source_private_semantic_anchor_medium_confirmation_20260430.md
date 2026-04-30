# Source-Private Semantic-Anchor Medium Confirmation

- date: `2026-04-30`
- rung: medium confirmation
- live branch: semantic-anchor, target-preserving source-private packet receiver
- aggregate artifact: `results/source_private_semantic_anchor_heldout_medium_confirmation_20260430/`
- run artifacts:
  - `results/source_private_semantic_anchor_heldout_packet_gate_20260430_seed47_n512/`
  - `results/source_private_semantic_anchor_heldout_packet_gate_20260430_seed53_n512/`
  - `results/source_private_semantic_anchor_heldout_packet_gate_20260430_seed59_n512/`

## Readiness Snapshot

Current ICLR readiness: materially stronger scoped positive-method paper. This
is now medium, seed-stable evidence for source-private packet communication
under held-out paraphrase drift. It is still not a broad activation-level
cross-LLM latent-transfer claim because the receiver uses a public semantic
anchor lexicon.

Current story: the source observes private diagnostic evidence and emits a
`4-8` byte packet. The target receives only the public candidate set and a
public semantic-anchor receiver. The target-preserving gate prevents
source-destroyed packets from overriding the target prior.

Exact blocker to full-paper submission: add a less hand-coded receiver
baseline, preferably frozen SBERT/SimCSE-style public candidate embeddings or a
learned contrastive receiver, and add a reviewer-facing systems
rate/assumption table.

## Command

```bash
for seed in 47 53 59; do
  PYTHONUNBUFFERED=1 ./venv_arm64/bin/python \
    scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
    --output-dir results/source_private_semantic_anchor_heldout_packet_gate_20260430_seed${seed}_n512 \
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

## Result

All three seeds pass. Every row passes: `18/18`.

| Aggregate | Value |
|---|---:|
| seeds | `47, 53, 59` |
| eval examples per direction/seed | `512` |
| pass rows | `18/18` |
| minimum passing CI95 low vs target | `+0.457` |
| minimum learned-target lift | `+0.500` |
| minimum learned-best-control lift | `+0.496` |
| maximum best source-destroying control | `0.254` |
| minimum oracle candidate-map accuracy | `0.875` |
| exact transformed held-out overlap | `0` in every direction |

| Direction | Passing rows |
|---|---:|
| core_to_holdout | `6/6` |
| holdout_to_core | `6/6` |
| same_family_all | `6/6` |

| Budget | Passing rows |
|---:|---:|
| 4 bytes | `9/9` |
| 8 bytes | `9/9` |

## Interpretation

This upgrades the semantic-anchor receiver from a strict-small positive to a
medium seed-stable positive. It directly addresses the previous held-out
paraphrase failure: calibration sees `synonym_stress`, evaluation uses
`heldout_synonym`, and exact transformed surface overlap remains zero.

The result also strengthens the method contribution stack:

1. Source-private benchmark and controls.
2. Extreme-rate packet interface.
3. Learned/calibrated public dictionary.
4. Semantic-anchor target-preserving receiver that survives held-out
   paraphrase drift.

The remaining weakness is method generality. The semantic-anchor lexicon is
explicit. The next decisive ablation is whether a frozen or learned semantic
embedding receiver can reproduce the pass without handwritten anchor
expansions.

## Next Gate

Implement an embedding-receiver ablation on the same artifacts:

- candidate view: `heldout_synonym`
- calibration view: `synonym_stress`
- seeds: `47, 53, 59`
- budgets: `4, 8`
- baselines: semantic-anchor receiver, hashed receiver, frozen embedding
  receiver, target-only, shuffled, random same-byte, answer-only, structured
  text, and top-atom knockout
- pass rule: frozen/learned embedding receiver must preserve at least `80%` of
  semantic-anchor lift while controls stay within target + `0.03`.

In parallel, add the systems rate/assumption frontier recommended by the
systems scout: packet, same-byte text, structured text ladder, full hidden-log
relay, and KV/TurboQuant/QJL byte-floor rows with assumptions stated.
