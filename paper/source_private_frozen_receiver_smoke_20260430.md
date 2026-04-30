# Frozen Receiver Smoke

- date: `2026-04-30`
- gate: `source_private_frozen_receiver_smoke`
- status: negative / useful boundary, with partial signal

## Question

Can an off-the-shelf frozen text/LLM embedding receiver replace the explicit
semantic-anchor lexicon in the live held-out synonym packet gate?

## Implementation

I added frozen Transformer feature modes to
`scripts/run_source_private_learned_synonym_dictionary_packet_gate.py`:

- `hf_last_mean`: mean-pooled final hidden states
- `hf_mid_last_mean`: concatenated mean-pooled middle and final hidden states

The implementation uses local Hugging Face models with `local_files_only` by
default, caches text features, and keeps the existing source-private packet
protocol, paired bootstrap, top-atom knockout, exact-overlap audit, and
source-destroying controls.

I also added a frozen-feature backend to
`scripts/run_source_private_candidate_embedding_receiver.py` for later
activation-receiver ablations.

## Commands

```bash
PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_frozen_receiver_bge_smoke_20260430 \
  --budgets 4 --train-examples 64 --eval-examples 32 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 384 --text-feature-mode hf_last_mean \
  --feature-model BAAI/bge-small-en --feature-device cpu \
  --feature-dtype float32 --feature-max-length 96 --local-files-only \
  --ridge 0.25 --top-k 8 --min-score 0.05 --min-decision-score 0.70

PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_frozen_receiver_bge_midlast_smoke_20260430 \
  --budgets 4 8 --train-examples 64 --eval-examples 32 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 64 \
  --feature-dim 768 --text-feature-mode hf_mid_last_mean \
  --feature-model BAAI/bge-small-en --feature-device cpu \
  --feature-dtype float32 --feature-max-length 96 --local-files-only \
  --ridge 0.25 --top-k 8 --min-score 0.05 --min-decision-score 0.70

PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false ./venv_arm64/bin/python \
  scripts/run_source_private_learned_synonym_dictionary_packet_gate.py \
  --output-dir results/source_private_frozen_receiver_qwen3_0_6b_cpu_tiny_20260430 \
  --budgets 4 --train-examples 16 --eval-examples 8 --seed 47 \
  --candidate-atom-view heldout_synonym \
  --calibration-atom-view synonym_stress \
  --candidate-calibration all_public --calibration-examples 16 \
  --feature-dim 1024 --text-feature-mode hf_last_mean \
  --feature-model Qwen/Qwen3-0.6B --feature-device cpu \
  --feature-dtype float32 --feature-max-length 80 --local-files-only \
  --ridge 0.25 --top-k 8 --min-score 0.05 --min-decision-score 0.70
```

The attempted `Qwen/Qwen3-0.6B` MPS run failed in an MPS matmul shape inference
path, so it is recorded as backend instability rather than method evidence.

## Results

| Run | Pass | Best learned packet | Best lift | Controls | Best oracle |
|---|---:|---:|---:|---|---:|
| BGE final mean, n32 | `False` | 0.625 | +0.375 | clean | 0.625 |
| BGE mid+last mean, n32 | `False` | 0.625 | +0.375 | clean | 0.875 in one direction, inconsistent |
| Qwen3-0.6B final mean, n8 CPU | `False` | 0.500 | +0.250 | clean | 0.750 |

Exact transformed held-out overlap remains `0` in the BGE smoke directions.

## Interpretation

Frozen embeddings produce real but insufficient signal. They improve over target
and keep destructive controls at target, but they do not reliably map held-out
candidate paraphrases to the source-private atom packet with enough oracle
headroom to pass the live gate.

Decision: do not promote frozen embedding substitution as a contribution yet.
The next receiver branch should be a trained contrastive source-control or
JEPA-style candidate-latent objective over frozen embeddings/activations, not
another off-the-shelf embedding swap.

## Next Gate

Train a tiny contrastive receiver on public calibration examples using frozen
features:

- positive: source-private packet atoms paired with the correct candidate
- negatives: shuffled-source, random same-byte, answer-only, target-derived,
  and same candidate set with wrong atom packet
- objective: candidate ranking / atom-consistency loss
- pass: preserve at least `80%` of semantic-anchor lift on held-out synonym
  `n=256`, controls within target + `0.03`, oracle >= `0.80`
