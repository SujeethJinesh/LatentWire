# Source-Private ARC Soft-Prefix Resonance Gate

Date: 2026-05-04

## Readiness Status

- COLM workshop: useful negative/diagnostic evidence for a controlled
  source-conditioned soft-prefix interface.
- ICLR full paper: not ready. This gate adds the strict evaluation surface the
  reviewers need, but the tested receivers do not pass it.

## Gate

The gate trains on ARC-Challenge validation rows where TinyLlama and Qwen
packet predictions disagree, then evaluates once on held-out ARC-Challenge test
disagreement rows. The receiver is Qwen3-0.6B. The compressed path emits target
input-embedding soft prefixes and never gives the target the source text.

Required controls:

- target-only and target-derived prefixes;
- zero-source, row-shuffled source, candidate-roll, candidate derangement;
- source-index, source-rank, and source-score controls;
- same-byte visible text;
- Qwen-substituted source-family packet.

## Runs

| Run | Train/Test | Source feature | Matched | Best required control | Pass |
|---|---:|---|---:|---:|---|
| `n8_cached_score` | 8 / 8 | cached score-pool residual | 0.250 | 0.625 (`qwen_substituted_packet`) | False |
| `n8_hidden_public_innovation` | 8 / 8 | TinyLlama hidden+score public-innovation residual | 0.500 | 0.625 (`qwen_substituted_packet`) | False |
| `n8_hidden_public_innovation_contrastive` | 8 / 8 | same + contrastive controls | 0.250 | 0.750 (`zero_source`) | False |

## Interpretation

The strict gate implementation is now useful: it separates source-conditioned
soft prefixes from target-only, same-byte text, raw source rank/score shortcuts,
candidate-order artifacts, and source-family substitution.

The result is not positive. Cached score-pool soft prefixes are worse than
target-only and Qwen-substitution controls. TinyLlama hidden public-innovation
features improve the matched receiver to 0.500 and beat zero-source,
source-row-shuffle, raw rank/score, and candidate derangement controls, but
they still tie target-only/same-byte and lose to the Qwen-substituted packet.
The contrastive rescue overfits the control objective and creates a zero-source
artifact.

## Decision

Demote shallow ARC soft-prefix receivers based on cached source score pools and
single-step TinyLlama hidden public-innovation pooling. Keep target-native
soft-prefix resonance alive only with a stronger source representation:
train-fit common-basis/SAE/crosscoder features, atom-shuffle controls, and a
larger frozen disagreement slice.

## Next Exact Gate

Implement a compact common-basis source encoder for this same strict wrapper:
train-fit PCA/SAE or sparse crosscoder atoms over answer-key-forbidden
TinyLlama candidate hidden states, then emit Qwen-native soft prefixes. Required
controls should include atom shuffle, wrong-row, candidate roll, target-derived
prefix, raw source rank/score, same-byte text, and Qwen-substitution.

Lay explanation: we tried to train a small translator that converts TinyLlama's
hidden clue into invisible soft tokens for Qwen. The better hidden version got
some real signal, but it was no better than Qwen reading no source clue or a
visible source hint, so it is not a publishable positive method yet.
