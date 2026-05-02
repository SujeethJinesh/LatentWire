# References: HellaSwag Receiver Headroom Decomposition

## Claim Boundary

This memo supports the receiver-headroom diagnostic, not a solved
cross-family latent-language claim. The safe result is that TinyLlama and Qwen
make complementary HellaSwag decisions under a fixed `2B` raw / `5B` framed
source-private packet boundary, while simple train-only confidence selectors
do not recover the oracle headroom.

## Primary Sources And Why They Matter

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - https://arxiv.org/abs/1905.07830
   - Why it matters: benchmark surface for the full-validation receiver
     headroom decomposition.

2. TinyLlama: An Open-Source Small Language Model
   - https://arxiv.org/abs/2401.02385
   - Why it matters: non-Qwen source family for the packet-only row.

3. Relative Representations Enable Zero-Shot Latent Space Communication
   - https://openreview.net/forum?id=SrC-nwieGJ
   - Why it matters: anchor-relative coordinates are the cleanest mathematical
     common-basis next branch after simple selectors fail.

4. Sparse Autoencoders Find Highly Interpretable Features in Language Models
   - https://arxiv.org/abs/2309.08600
   - Why it matters: sparse dictionaries motivate an interpretable shared
     feature basis, but this artifact does not train or claim an SAE.

5. Crosscoders
   - https://transformer-circuits.pub/2025/crosscoder-diffing/index.html
   - Why it matters: crosscoders separate shared and model-specific features,
     which directly matches the observed TinyLlama/Qwen complementary-error
     problem.

6. Selective Classification for Deep Neural Networks and SelectiveNet
   - https://arxiv.org/abs/1705.08500
   - https://arxiv.org/abs/1901.09192
   - Why it matters: the receiver problem can be framed as selective
     prediction or override: decide when to trust the source packet versus the
     target model. Our simple selective rules fail under train-only selection.

7. Prefix-Tuning and Prompt Tuning
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - Boundary: soft prompts are learned conditioning/adaptation vectors for a
     model, while this artifact evaluates per-example source-private packets
     sent between model families.

8. Cache-to-Cache and KVComm
   - https://arxiv.org/abs/2510.03215
   - https://arxiv.org/abs/2510.03346
   - Boundary: these communicate through projected/fused or selectively shared
     KV/cache state. This artifact transmits no source KV, raw hidden vector,
     raw score vector, or source text.

9. Communicating Activations Between Language Model Agents and CIPHER
   - https://arxiv.org/abs/2501.14082
   - https://arxiv.org/abs/2310.06272
   - Boundary: LatentWire is not first in non-natural-language model
     communication. The distinct claim here is the fixed-byte source-private
     packet plus destructive controls and receiver-headroom accounting.

10. BLIP-2 Q-Former and Perceiver IO
    - https://arxiv.org/abs/2301.12597
    - https://arxiv.org/abs/2107.14795
    - Why it matters: learned query bottlenecks between frozen heterogeneous
      systems motivate the next train-only receiver selector, especially after
      scalar threshold rules fail.

11. QJL, KIVI, KVQuant, and TurboQuant
    - https://arxiv.org/abs/2406.03482
    - https://arxiv.org/abs/2402.02750
    - https://arxiv.org/abs/2401.18079
    - https://arxiv.org/abs/2504.19874
    - Boundary: these are KV/cache sketching or quantization systems
      comparators. They motivate byte-floor and hardware baselines but do not
      duplicate a `2B` source-private task packet.

12. vLLM and SGLang
    - https://arxiv.org/abs/2309.06180
    - https://arxiv.org/abs/2312.07104
    - Boundary: native serving comparisons remain pending until NVIDIA rows
      are available.

## Reviewer-Facing Framing

Safe:

- There is real receiver headroom: Qwen's hybrid packet is correct on heldout
  rows where TinyLlama packet-only is wrong.
- Simple train-prefix confidence thresholding fails, so the current evidence
  does not support a solved common-language receiver claim.
- The next branch should be a train-only relative/dictionary/learned-query
  receiver, not another eval-selected threshold.

Unsafe:

- Claiming the current receiver beats packet-only.
- Claiming universal latent-language transfer.
- Claiming native systems superiority over C2C, KVComm, KIVI, KVQuant, QJL,
  TurboQuant, vLLM, or SGLang.
