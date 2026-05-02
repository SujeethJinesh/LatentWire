# References: ARC Source-Family/Source-Cache Falsification

Web/literature check date: 2026-05-02.

## Closest Communication Baselines

1. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   <https://openreview.net/forum?id=LeatkxrBCi>
   - Role: closest ICLR 2026 direct latent/KV communication baseline.
   - Boundary: C2C trains a neural projector/fuser over source and target
     KV-cache states. This gate tests whether a fixed-byte public-basis packet
     still carries source-private decisions when the source-choice cache is
     rebuilt with a different model family.

2. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://openreview.net/forum?id=F7rUng23nw>
   - Role: closest selective KV-sharing systems baseline.
   - Boundary: KVComm communicates selected KV entries. LatentWire's current
     claim is lower exposure and fixed small packet size; native quality and
     latency comparisons remain pending.

3. Relative Representations Enable Zero-Shot Latent Space Communication.
   <https://openreview.net/forum?id=SrC-nwieGJ>
   - Role: motivates anchor-relative coordinate charts for comparing model
     representations.
   - Boundary: this gate is stricter and narrower: the alternate source must
     produce a separate answer-key-forbidden source-choice cache, and the
     packet must survive both full-slice and Qwen-disagreement slices.

## Quantization and Systems Context

1. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://openreview.net/forum?id=tO3ASKZlok>
   - Role: current strong vector/KV quantization baseline and random-rotation
     inspiration for packet compression.
   - Boundary: TurboQuant compresses vector representations/KV caches; this
     gate evaluates task-level communication under a fixed packet budget.

2. QJL: 1-Bit Quantized Johnson-Lindenstrauss for KV Cache Quantization.
   <https://arxiv.org/abs/2406.03482>
   - Role: relevant one-bit projection baseline and systems byte-floor
     comparator.
   - Boundary: QJL preserves attention inner products, while this gate asks
     whether a source-selected candidate residual can be communicated through
     a small public-basis packet.

3. FNet: Mixing Tokens with Fourier Transforms.
   <https://arxiv.org/abs/2105.03824>
   - Role: precedent for unparameterized Fourier-style mixing in neural
     sequence models.
   - Boundary: FNet is an internal model architecture change, not a
     source-private cross-model communication protocol.

## Learned Latent Interfaces

1. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
   <https://aclanthology.org/2021.acl-long.353/>
   - Role: compact continuous-conditioning baseline.
   - Boundary: prefix vectors are learned task parameters; this gate sends
     per-example fixed-byte packets from a separate source model.

2. Sparse Autoencoders Find Highly Interpretable Features in Language Models.
   <https://arxiv.org/abs/2309.08600>
   - Role: future shared-atom direction for making packet dimensions more
     interpretable.
   - Boundary: no SAE interpretability claim is made by this gate.

3. Sparse Crosscoders for Cross-Layer Features and Model Diffing.
   <https://transformer-circuits.pub/2024/crosscoders/index.html>
   - Role: relevant learned sparse common-language direction.
   - Boundary: a crosscoder packet would be a separate learned connector; the
     current gate isolates source-family/cache dependence first.

4. Training Large Language Models to Reason in a Continuous Latent Space.
   <https://arxiv.org/abs/2412.06769>
   - Role: nearby latent reasoning motivation.
   - Boundary: Coconut feeds hidden states back into the same model; this gate
     transfers a compact source-private packet between independently evaluated
     source and receiver surfaces.

## Claim Boundary

Safe claim if the gate passes: the ARC Fourier/anchor-syndrome packet is not
merely replaying the original Qwen source-choice cache; it survives a stricter
source-family/source-cache falsification and examples where Qwen and the
alternate source disagree.

Safe claim if the gate fails: the ARC positive row remains useful but
source-cache-specific, so ICLR framing must stay below universal latent
communication until a stronger alternate source endpoint or learned connector
lands.

Unsafe claims either way: first latent communication method, superiority over
C2C/KVComm without native matched baselines, semantic SAE-like interpretability,
or TurboQuant-level vector-compression optimality.
