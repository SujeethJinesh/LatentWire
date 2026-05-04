# HellaSwag Decision-Sparse Common-Basis Hidden Packet References

Date: 2026-05-04

## Why This Memo Exists

This memo records the related-work and novelty boundary for the failed
decision-supervised sparse common-basis HellaSwag packet gate on validation
`1024:2048`. The result is a controlled negative ablation for a learned
one-byte hidden-atom packet, not a positive universal-latent-language result.

## Primary Sources And Boundaries

1. HellaSwag: Can a Machine Really Finish Your Sentence?
   - Link: https://arxiv.org/abs/1905.07830
   - Role: adversarially filtered commonsense continuation benchmark.
   - Boundary: HellaSwag is multiple-choice, so candidate packet baselines and
     option-order hardening remain mandatory.

2. Towards Monosemanticity: Decomposing Language Models With Dictionary
   Learning
   - Link: https://transformer-circuits.pub/2023/monosemantic-features/index.html
   - Role: sparse autoencoder/dictionary-learning motivation for feature bases.
   - Boundary: this gate did not recover interpretable monosemantic features;
     it used a decision-supervised sparse encoder only as a byte-limited packet
     generator.

3. Sparse Crosscoders for Cross-Layer Features and Model Diffing
   - Link: https://transformer-circuits.pub/2024/crosscoders/index.html
   - Role: shared/private sparse-feature inspiration.
   - Boundary: sparse crosscoders already cover shared-feature discovery; this
     gate tests whether such features can become a source-private one-byte
     communication object under destructive controls.

4. Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language
   Models
   - Link: https://openreview.net/forum?id=rbHOLX8OWh
   - Role: feature-universality motivation for common bases.
   - Boundary: do not claim universal SAE features. The local evidence is
     negative: atom shuffle and label-permuted SAE controls match or exceed the
     selected packet.

5. Prefix-Tuning: Optimizing Continuous Prompts for Generation
   - Link: https://aclanthology.org/2021.acl-long.353/
   - Role: continuous virtual-token baseline.
   - Boundary: prefix tuning learns task/prompt conditioning for a frozen LM;
     this gate transmits a per-example discrete source-private atom packet.

6. Learning to Compress Prompts with Gist Tokens
   - Link: https://arxiv.org/abs/2304.08467
   - Role: prompt/context compression baseline.
   - Boundary: gist tokens compress visible prompts for reuse; this gate
     transmits no source text or reusable prompt memory.

7. Training Large Language Models to Reason in a Continuous Latent Space
   - Link: https://arxiv.org/abs/2412.06769
   - Role: latent-reasoning comparator.
   - Boundary: Coconut feeds hidden states back into the same model as
     continuous thought; this gate tests downstream candidate-evidence transfer
     and failed the positive-method bar.

8. DroidSpeak: KV Cache Sharing for Cross-LLM Communication and Multi-LLM
   Serving
   - Link: https://arxiv.org/abs/2411.02820
   - Role: KV-cache communication and serving comparator.
   - Boundary: DroidSpeak reuses KV-cache layers; this gate transmits no dense
     source KV/cache state.

9. KVComm: Enabling Efficient LLM Communication through Selective KV Sharing
   - Link: https://arxiv.org/abs/2510.03346
   - Role: selective KV-sharing comparator.
   - Boundary: KVComm transmits selected KV pairs; this gate sends a one-byte
     task packet and does not compete on matched KV-sharing benchmarks.

10. QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
    Overhead
    - Link: https://arxiv.org/abs/2406.03482
    - Role: quantized KV-cache systems floor.
    - Boundary: QJL is a vector/KV quantizer; LatentWire should compare byte
      floors but should not claim quantizer novelty.

11. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
    - Link: https://arxiv.org/abs/2504.19874
    - Role: modern vector/KV quantization baseline.
    - Boundary: TurboQuant compresses dense vectors; this gate transmits no
      continuous vector and does not make near-optimal distortion claims.

## Decision Boundary

Cite this artifact as:

```text
a controlled negative ablation for a train-only decision-supervised sparse
common-basis hidden-atom packet under a one-byte source-private contract.
```

Do not cite it as:

```text
universal latent language
interpretable sparse atoms
positive SAE/common-basis transfer
latent reasoning
native systems speedup
C2C/KVComm/DroidSpeak/TurboQuant comparison win
```

