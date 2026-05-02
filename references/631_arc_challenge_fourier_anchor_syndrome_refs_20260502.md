# References: ARC-Challenge Fourier/Anchor-Syndrome Gate

Web/literature check date: 2026-05-02.

## Common-Basis Motivation

1. Relative Representations Enable Zero-Shot Latent Space Communication.
   <https://openreview.net/forum?id=SrC-nwieGJ>
   - Role: motivates public anchor-relative coordinate charts as a way to make
     representations comparable across models.
   - Boundary: LatentWire does not claim universal latent communication; the
     result is an operational fixed-byte packet on ARC.

2. Product of Invariances to Enhance Latent Space Communication.
   <https://openreview.net/forum?id=vngVydDWft>
   - Role: related invariance-based latent communication framing.
   - Boundary: this gate does not prove broad invariance; it checks a single
     frozen ARC packet surface with anchor and spectral mismatch controls.

3. FNet: Mixing Tokens with Fourier Transforms.
   <https://arxiv.org/abs/2105.03824>
   - Role: primary precedent for fixed, unparameterized Fourier-style bases in
     neural sequence models.
   - Boundary: FNet is token mixing inside a model, not source-private
     model-to-model communication.

4. Fast Johnson-Lindenstrauss Transform.
   <https://www.cs.princeton.edu/~chazelle/pubs/stoc06.pdf>
   - Role: classic basis for fast structured random projections.
   - Boundary: the current gate uses a deterministic DCT-II chart plus the
     existing sparse packet projection; it is not a new JL theorem.

## Communication and Systems Comparisons

1. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   <https://openreview.net/forum?id=LeatkxrBCi>
   - Role: closest direct latent/KV communication baseline.
   - Boundary: C2C projects and fuses source KV-cache state; this gate sends a
     fixed-byte packet over a public basis.

2. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://arxiv.org/abs/2510.03346>
   - Role: close selective KV-sharing communication baseline.
   - Boundary: KVComm still transports selected KV layers; LatentWire's claim
     is source-state exposure reduction unless native latency/quality baselines
     are run.

3. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://openreview.net/forum?id=tO3ASKZlok>
   - Role: current strong vector/KV quantization systems baseline and
     random-rotation/QJL inspiration.
   - Boundary: TurboQuant compresses vectors/KV caches; it is not a task-level
     source-private packet protocol.

4. QJL: 1-Bit Quantized Johnson-Lindenstrauss for KV Cache Quantization.
   <https://arxiv.org/abs/2406.03482>
   - Role: relevant one-bit projection and cache-compression baseline.
   - Boundary: useful byte-floor comparator, not evidence transfer across a
     public MCQ receiver.

## Learned Interfaces and Sparse Codes

1. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
   <https://aclanthology.org/2021.acl-long.353/>
   - Role: compact continuous conditioning baseline family.
   - Boundary: prefix vectors are learned task parameters, not per-example
     fixed-byte source-private packets.

2. Sparse Autoencoders Find Highly Interpretable Features in Language Models.
   <https://arxiv.org/abs/2309.08600>
   - Role: future path for sparse shared atoms and interpretability.
   - Boundary: this gate uses public hashed anchor/DCT coordinates, not learned
     SAE features.

3. Sparse Crosscoders for Cross-Layer Features and Model Diffing.
   <https://transformer-circuits.pub/2024/crosscoders/index.html>
   - Role: related approach for shared sparse features across layers/models.
   - Boundary: a crosscoder atom packet would be a separate learned-connector
     contribution.

## Claim Boundary

Safe claim: a fixed-byte source-private packet over a public low-frequency
anchor/DCT basis preserves the frozen ARC packet signal and fails under
anchor-ID, anchor-value, and spectral-bin mismatch controls.

Unsafe claim: first Fourier communication method, universal latent language,
semantic anchor isomorphism, or superiority over C2C/KVComm/TurboQuant. The
random shared-anchor diagnostic shows that shared coordinate agreement matters
more than semantic anchor names in this hashed ARC gate.
