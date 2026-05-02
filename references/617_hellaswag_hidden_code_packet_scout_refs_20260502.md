# References: HellaSwag Hidden-Code Packet Scout

## Claim Boundary

This memo supports the failed hidden-code packet scout. The safe claim is that
one-byte source-hidden PCA/k-means and train-only hidden-reliability codes were
tested on a frozen HellaSwag slice and did not beat compact packet-only. It does
not claim a universal latent language, SAE/crosscoder feature alignment, prefix
conditioning, KV/cache communication, or native serving speedup.

## Primary Sources And Why They Matter

1. Relative representations for latent communication
   - https://arxiv.org/abs/2209.15430
   - Boundary: anchor-relative coordinates are a prior method for handling
     latent-space gauge mismatch. This scout does not implement anchor-relative
     communication; it tests ordinary source-hidden codebooks.

2. Sparse autoencoder feature-space universality
   - https://arxiv.org/abs/2410.06981
   - Boundary: SAE feature spaces may have rotation-invariant similarities
     across LLMs. This artifact does not train an SAE and cannot claim universal
     features.

3. Universal Sparse Autoencoders
   - https://arxiv.org/abs/2502.03714
   - Boundary: USAEs jointly learn a shared concept space across models. The
     current hidden-code scout uses shallow PCA/k-means and reliability bins,
     so USAE/crosscoder-style shared bases remain future comparators.

4. Sparse Crosscoders
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Boundary: crosscoders motivate shared dictionaries across activation
     spaces. This gate does not learn a multi-model sparse dictionary.

5. VQ-VAE / Neural Discrete Representation Learning
   - https://arxiv.org/abs/1711.00937
   - Boundary: discrete latent codebooks are prior work. The tested question is
     only whether a byte-scale source-hidden task code helps Qwen beyond a
     candidate packet.

6. Neural Distributed Source Coding
   - https://arxiv.org/abs/2106.02797
   - Boundary: learned distributed source coding with decoder side information
     already exists. Our artifact is an LLM task-packet scout, not a general
     DSC method.

7. Prefix-Tuning and Prompt Tuning
   - https://aclanthology.org/2021.acl-long.353/
   - https://aclanthology.org/2021.emnlp-main.243/
   - Boundary: prefix/prompt tuning learns persistent continuous conditioning
     vectors. LatentWire packets are per-example discrete records and are not
     inserted as learned virtual tokens.

8. C2C and KVComm
   - https://arxiv.org/abs/2510.03215
   - https://arxiv.org/abs/2510.03346
   - Boundary: these systems communicate or fuse KV/cache states. This scout
     transmits no source KV, no raw hidden vector, and no raw score vector.

9. QJL, TurboQuant, KIVI, and KVQuant
   - https://arxiv.org/abs/2406.03482
   - https://arxiv.org/abs/2504.19874
   - https://arxiv.org/abs/2402.02750
   - https://arxiv.org/abs/2401.18079
   - Boundary: these are vector/KV fidelity-compression methods. The hidden
     packet is a task-level decision sideband; it cannot claim KV compression
     quality, HBM savings, or native serving speedups.

10. Latent diffusion and Diffusion Transformers
    - https://arxiv.org/abs/2112.10752
    - https://arxiv.org/abs/2212.09748
    - Boundary: iterative latent denoising/refinement remains a possible
      receiver idea. No diffusion-style refinement model is implemented here.

## Reviewer-Facing Framing

Safe:

- We tested simple source-hidden byte codes before spending full-validation
  hidden materialization compute.
- The official-train-dev-selected hidden code is slightly worse than
  packet-only on the frozen 1024-row validation slice.
- The best diagnostic row is small (`+0.005859`) and has a CI crossing zero.
- Supervised hidden reliability bins also fail to beat packet-only.
- This kills shallow hidden-code summaries and sharpens the next branch toward
  true common-basis/crosscoder objectives or less packet-saturated benchmarks.

Unsafe:

- Claiming hidden-code packet communication works.
- Claiming a shared latent language.
- Claiming equivalence to SAE/crosscoder methods.
- Claiming superiority over prefix tokens, C2C/KVComm, or KV quantization.
- Claiming native serving/HBM wins before NVIDIA/vLLM/SGLang measurements.
