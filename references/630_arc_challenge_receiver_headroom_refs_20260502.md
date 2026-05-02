# References: ARC-Challenge Receiver/Headroom Replication

Web/literature check date: 2026-05-02.

## Benchmark and Selector Framing

1. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning
   Challenge.
   <https://arxiv.org/abs/1803.05457>
   - Role: ARC-Challenge is the primary public science QA surface used for the
     receiver-fusion replication.
   - Boundary: the failed receiver gate means the OpenBookQA receiver result
     cannot be claimed as cross-science-QA evidence yet.

2. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book
   Question Answering.
   <https://arxiv.org/abs/1809.02789>
   - Role: source of the positive OpenBookQA packet/target receiver row being
     replicated.
   - Boundary: OpenBookQA remains positive, but ARC shows the selector is not
     robust enough for an ICLR full-paper claim by itself.

3. SelectiveNet: A Deep Neural Network with an Integrated Reject Option.
   <https://arxiv.org/abs/1901.09192>
   - Role: prior art for learned accept/reject or abstention decisions.
   - Boundary: the selector alone is not the novelty. The defensible novelty
     must come from fixed-byte source-private packet evidence and controls.

4. FrugalGPT: How to Use Large Language Models While Reducing Cost and
   Improving Performance.
   <https://arxiv.org/abs/2305.05176>
   - Role: related model routing/cascade framing.
   - Boundary: LatentWire is not choosing which full model to call; it fuses a
     tiny source-private packet with target-side public evidence.

5. LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and
   Generative Fusion.
   <https://arxiv.org/abs/2306.02561>
   - Role: related multi-model fusion baseline.
   - Boundary: LLM-Blender fuses generated text/model outputs; LatentWire's
     current receiver should be compared against such routing/fusion baselines
     before claiming superior multi-model collaboration.

## Common-Basis and Systems Directions

1. Relative Representations Enable Zero-Shot Latent Space Communication.
   <https://openreview.net/forum?id=SrC-nwieGJ>
   - Role: primary motivation for using a shared public coordinate chart or
     anchors to communicate across latent spaces.
   - Boundary: the current receiver does not yet establish universal latent
     communication; ARC replication failure argues for a stronger common-basis
     method.

2. FNet: Mixing Tokens with Fourier Transforms.
   <https://arxiv.org/abs/2105.03824>
   - Role: motivation for fixed, unparameterized spectral bases as efficient
     shared transforms.
   - Boundary: FNet is not a communication protocol, but it motivates a
     Fourier/anchor-syndrome packet as a cleaner next experiment.

3. Cache-to-Cache: Direct Semantic Communication Between Large Language
   Models.
   <https://arxiv.org/abs/2510.03215>
   - Role: close model-to-model communication baseline through KV-cache
     projection/fusion.
   - Boundary: C2C exposes and fuses source KV state; LatentWire exposes a
     fixed-byte packet. We cannot claim accuracy superiority without direct
     baseline runs.

4. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://arxiv.org/abs/2510.03346>
   - Role: close communication systems baseline using selected KV sharing.
   - Boundary: KV sharing is still source-state transport; LatentWire's systems
     claim must be byte/exposure reduction unless native latency rows are run.

5. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://arxiv.org/abs/2504.19874>
   - Role: current strong vector/KV quantization baseline and inspiration for
     random-rotation or syndrome-style hidden-innovation codecs.
   - Boundary: TurboQuant is cache/vector compression, not task-level
     source-private evidence transfer.

6. Sparse Crosscoders for Cross-Layer Features and Model Diffing.
   <https://transformer-circuits.pub/2024/crosscoders/index.html>
   - Role: interpretability-adjacent path toward shared sparse atoms across
     models/layers.
   - Boundary: a sparse-atom packet would be a different technical
     contribution; the current ARC receiver gate does not use learned sparse
     features.

## Claim Boundary

Safe claim: the OpenBookQA receiver is a positive source-private
evidence-fusion row, but ARC falsifies the same validation-selected selector as
a cross-benchmark receiver.

Unsafe claim: universal latent language, selector novelty, or systems
superiority over C2C/KVComm/TurboQuant. Those require a new common-basis method,
native systems rows, and direct competitor baselines.
