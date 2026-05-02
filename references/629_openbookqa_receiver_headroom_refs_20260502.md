# References: OpenBookQA Receiver/Headroom Gate

Web/literature check date: 2026-05-02.

## Benchmark and Receiver Framing

1. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book
   Question Answering.
   <https://arxiv.org/abs/1809.02789>
   - Role: OpenBookQA is the held-out public science QA benchmark used for the
     receiver-fusion gate.
   - Boundary: the result should be framed as benchmark-specific evidence
     fusion until replicated on ARC and other surfaces.

2. SelectiveNet: A Deep Neural Network with an Integrated Reject Option.
   <https://arxiv.org/abs/1901.09192>
   - Role: prior art for learned accept/reject decisions.
   - Boundary: SelectiveNet is not cross-model communication and does not send
     a fixed-byte source-private packet.

3. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
   <https://aclanthology.org/2021.acl-long.353/>
   - Role: compact conditioning baseline family.
   - Boundary: prefix vectors are learned task parameters, not per-example
     source-private packets emitted by another model.

4. Relative Representations Enable Zero-Shot Latent Space Communication.
   <https://openreview.net/forum?id=SrC-nwieGJ>
   - Role: common-coordinate/anchor-basis motivation.
   - Boundary: the OpenBookQA receiver does not yet prove universal latent
     communication; it fuses a compact source candidate sketch with a public
     target scorer.

## Communication and Systems Baselines

1. Cache-to-Cache: Direct Semantic Communication Between Large Language Models.
   <https://arxiv.org/abs/2510.03215>
   - Overlap: direct model-to-model semantic communication through KV-cache
     fusion.
   - Boundary: C2C transmits/fuses source KV state; LatentWire transmits a
     small source-private packet and keeps source text/KV/raw hidden state out
     of the receiver boundary.

2. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing.
   <https://arxiv.org/abs/2510.03346>
   - Overlap: inter-LLM communication by sharing selected KV layers.
   - Boundary: selected KV state is still source-state exposure, not a
     fixed-byte packet.

3. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
   <https://arxiv.org/abs/2402.02750>
   - Overlap: low-bit KV-cache systems baseline.
   - Boundary: same-model KV compression, not source-private evidence transfer.

4. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
   Quantization.
   <https://arxiv.org/abs/2401.18079>
   - Overlap: sub-4-bit KV-cache compression for long-context inference.
   - Boundary: useful byte-floor comparator, not a communication protocol.

5. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.
   <https://arxiv.org/abs/2504.19874>
   - Overlap: current strong vector/KV quantization baseline and product
     quantization motivation.
   - Boundary: quantized vector/KV storage is not a task-level source-private
     packet receiver.

## Claim Boundary

Safe claim: validation-selected source-private packet/target scorer fusion can
improve OpenBookQA held-out test accuracy over packet-only and target-public
baselines under several source-destroy controls.

Unsafe claim: universal latent language, native systems superiority over
C2C/KVComm/KV quantization, or source-label-copy separation. Those require the
next ARC replication, stricter seed/control stability, and native NVIDIA runs.
