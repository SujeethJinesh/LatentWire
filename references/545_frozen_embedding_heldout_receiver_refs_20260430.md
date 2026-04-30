# Frozen Embedding Held-Out Receiver References

- date: `2026-04-30`
- purpose: primary-source memo for the frozen BGE/MiniLM held-out semantic
  receiver ablation and the public-conditioned residual-codebook next branch.

## What Was Tested

The semantic-anchor receiver passes the held-out synonym gate, but it uses an
explicit public semantic-anchor lexicon. The new ablation asks whether a generic
frozen sentence embedding model can replace that lexicon while preserving the
same source-private packet controls.

The tested receivers use frozen transformer text features for public candidate
surface forms, with source packets still limited to private residual evidence.
This is a stricter novelty boundary than claiming the semantic lexicon alone.

## Closest Prior Work And Why It Does Not Subsume The Result

- Sentence-BERT introduces Siamese/triplet BERT sentence embeddings for
  semantic textual similarity and retrieval-style comparison, but it does not
  study byte-limited source-private packets decoded against target-side
  candidate side information.
  Source: https://arxiv.org/abs/1908.10084
- MiniLM distills transformer self-attention for compact language models; the
  `all-MiniLM-L6-v2` model card positions the checkpoint as a sentence
  transformer embedding model. This is a frozen embedding baseline, not a
  communication protocol.
  Sources: https://arxiv.org/abs/2002.10957 and
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- BGE provides general text embeddings and is used here as a public semantic
  feature baseline. It can score candidate meaning, but it does not by itself
  define a source-private residual packet protocol with destructive controls.
  Source: https://huggingface.co/BAAI/bge-small-en
- C2C and related cache/activation communication work are stronger direct
  communication competitors. They motivate strict cache/text exposure
  accounting, but our frozen-embedding ablation does not transmit KV/cache
  state and should not be framed as a C2C replacement until matched benchmarks
  are run.
  Source: https://openreview.net/forum?id=LeatkxrBCi
- QINCo-style implicit neural codebooks motivate the next public-conditioned
  residual-codebook branch: learn small residual stages rather than rely on a
  fixed sentence embedding geometry. The current ablation does not implement
  QINCo; it only identifies why fixed public embeddings are insufficient.
  Source: https://arxiv.org/abs/2401.14732

## Novelty Boundary

Do claim:

- The held-out semantic-anchor result is not explained away by a generic frozen
  embedding receiver, because frozen BGE/MiniLM recover partial signal but fail
  the strict bidirectional gate.
- The ablation is useful evidence for the paper's control discipline: source
  packets can show real signal while still failing when receiver side
  information is too weak or too generic.

Do not claim:

- First use of sentence embeddings for candidate matching.
- Broad semantic latent communication from generic frozen embeddings.
- A production serving win over cache-transfer systems such as C2C/KVComm.

## Result-Guided Next Branches

1. Learned public ontology calibration: train a small public adapter from frozen
   embeddings and public candidate paraphrase stress, then require the same
   source-destroying controls and held-out no-overlap audit.
2. Public-conditioned residual codebooks: build the code basis from the
   receiver's public candidate geometry and encode source-private residual
   evidence in those local coordinates.
3. Model-mediated receiver: ask the target model to consume the packet plus
   public candidate descriptors, but require answer-masked, shuffled-source,
   deranged-table, and private-random-knockout controls.
