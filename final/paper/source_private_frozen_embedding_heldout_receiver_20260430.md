# Source-Private Frozen-Embedding Held-Out Receiver Ablation

- date: `2026-04-30`
- rung: receiver-generalization ablation
- live reference branch: semantic-anchor held-out receiver
- aggregate artifact:
  `results/source_private_hf_embedding_heldout_packet_gate_20260430/summary/`
- reference memo:
  `references/545_frozen_embedding_heldout_receiver_refs_20260430.md`

## Readiness Snapshot

Current paper readiness: COLM workshop strong for a scoped source-private
packet paper; ICLR full remains gated.

Current story: a source model observes private decision evidence and sends a
tiny packet. A target receiver combines that packet with public candidate-side
information. Semantic-anchor receivers pass a held-out paraphrase gate, but the
question is whether generic frozen embedding geometry is enough to replace the
explicit semantic-anchor lexicon.

Exact blocker: no generic frozen BGE/MiniLM receiver cleared the strict
bidirectional held-out gate. The next live branch is a learned public ontology
adapter or public-conditioned residual codebook, not another fixed embedding
variant.

## Layman Explanation

The passing semantic-anchor method gives the receiver a small public dictionary
of what different task phrases mean. This ablation asks: can we remove that
dictionary and instead use a normal off-the-shelf sentence embedding model, the
kind that puts similar sentences near each other? The answer is partly yes but
not enough. The frozen embeddings often understand one direction and same-family
cases, but they miss enough of the held-out synonym direction that the full
gate fails.

## Commands

All runs use the same held-out synonym surface as the semantic-anchor medium
confirmation, with source-destroying controls and top/private packet knockout
diagnostics.

```bash
./venv_arm64/bin/python scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_hf_embedding_heldout_packet_gate_20260430_seed47_n256_minilm_last_top12 \
  --budgets 4 8 --train-examples 512 --eval-examples 256 --seed 47 \
  --candidate-atom-view heldout_synonym --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 512 \
  --feature-dim 384 --text-feature-mode hf_last_mean \
  --feature-model sentence-transformers/all-MiniLM-L6-v2 \
  --ridge 0.25 --top-k 12 --min-score 0.05 --min-decision-score 0.70
```

The aggregate summary combines ten BGE/MiniLM variants:

- `BAAI/bge-small-en`: last and mid+last features, plus hashed+HF.
- `sentence-transformers/all-MiniLM-L6-v2`: last, mid+last, top-k/ridge/decision
  variants, plus hashed+HF.

## Result

| Aggregate | Value |
|---|---:|
| frozen embedding rows | `60` |
| strict pass rows | `20` |
| near-miss rows | `32` |
| bidirectional pass gate | `false` |
| semantic-anchor reference pass rows | `18/18` |
| semantic-anchor min passing CI95 low | `+0.457` |

Best frozen-embedding reads:

| Receiver | Read |
|---|---|
| MiniLM last/top12 | `4` pass rows; holdout-to-core and same-family pass, core-to-holdout fails with oracle ceiling `0.750` |
| MiniLM mid+last/top20 | `4` pass rows; stronger asymmetric signal, but core-to-holdout oracle falls to `0.625` |
| MiniLM top20/ridge0.25/decision0.50 | core-to-holdout 4B reaches `0.625` vs target/best-control `0.250`, CI95 low `+0.316`, but oracle is only `0.750`; holdout-to-core 4B reaches `0.750`, but private-random-knockout lift reduction is `0.906`, above the strict `<0.75` gate |
| BGE last | partial signal only: max accuracy `0.500`, min oracle `0.625` |
| hashed+HF | worsens the gate: `0` pass rows across tested BGE/MiniLM variants |

The refreshed summary table now exposes both top-atom and private-random
knockout lift reductions, so rows that look strong but fail the
private-random-knockout condition are auditable.

## Interpretation

Frozen sentence embeddings recover some public semantic structure, but they do
not reproduce the explicit semantic-anchor receiver. This is useful negative
evidence: the current positive held-out result is not simply "any embedding
model can do it." The successful branch needs an explicit public semantic
adapter, or the packet basis must be conditioned on the receiver's actual
candidate geometry.

Hypotheses updated:

- Weakened: generic frozen semantic embeddings are enough for held-out packet
  reception.
- Weakened: adding hashed lexical features to HF embeddings helps. It increases
  control sensitivity and produces no strict pass rows.
- Still alive: learned public ontology calibration on top of frozen embeddings.
- Still alive: public-conditioned residual codebooks / receiver-conditioned
  slots, where the receiver defines the local basis from candidate geometry.

## Negative Scratch Probes

I also tried local public-conditioned packet sketches before committing to the
frozen embedding ablation:

- Pairwise margin bits can look cross-family-positive, but deranged/opaque
  controls recover the same signal. That is answer-index-like and not
  acceptable.
- Local-frame residual quantizers built from public candidate geometry keep
  destructive controls clean, but the source signal collapses.
- Direct local-coordinate encoders recover some rows but leak into shuffled or
  deranged controls.

These are not promoted artifacts; they are logged as pruned branches.

## Next Gate

Build a learned public ontology adapter with the semantic-anchor run as the
ceiling and this frozen-embedding sweep as the floor. Required pass rule:

1. `core_to_holdout`, `holdout_to_core`, and same-family pass on frozen held-out
   synonym rows.
2. Controls remain within target + `0.03`.
3. Top-atom knockout removes at least `50%` of lift.
4. Private-random knockout lift reduction stays below `0.75`.
5. Exact transformed held-out overlap remains `0`.
