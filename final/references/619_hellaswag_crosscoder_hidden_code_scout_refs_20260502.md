# References: HellaSwag Linear Crosscoder Hidden-Code Scout

## Claim Boundary

This memo supports the failed linear crosscoder hidden-code scout. The safe
claim is that train-only PCA+CCA/SVD shared projections were evaluated as an
internal basis for one-byte source-private packets and did not robustly beat
compact packet-only. It does not claim a new representation-similarity method,
SAE/crosscoder alignment, query-token connector, prefix-token transfer,
KV/cache communication, or native serving speedup.

## Primary Sources And Why They Matter

1. SVCCA
   - https://arxiv.org/abs/1706.05806
   - Boundary: SVD plus canonical correlation is established representation
     analysis. A CCA-style shared projection is not novel by itself.

2. PWCCA
   - https://arxiv.org/abs/1806.05759
   - Boundary: projection-weighted CCA is prior work for comparing neural
     representations, not a fixed-byte communication protocol.

3. Centered Kernel Alignment
   - https://arxiv.org/abs/1905.00414
   - Boundary: CKA and related similarity tools are representation diagnostics.
     They motivate controls but are not the contribution.

4. Relative Representations
   - https://arxiv.org/abs/2209.15430
   - Boundary: latent communication through shared relative coordinates is
     direct prior art. LatentWire must differ through the packet contract and
     source-destroying evaluation.

5. Sparse Autoencoders
   - https://arxiv.org/abs/2309.08600
   - Boundary: sparse feature dictionaries for LLM activations are prior work.
     This scout does not train an SAE.

6. Universal Sparse Autoencoders
   - https://arxiv.org/abs/2502.03714
   - Boundary: USAE-style shared concept spaces are stronger common-basis
     baselines than this linear CCA scout.

7. Sparse Crosscoders
   - https://transformer-circuits.pub/2024/crosscoders/index.html
   - Boundary: cross-model sparse dictionaries are prior work. The safe
     LatentWire framing is fixed-byte source-private packets using shared
     projections internally, not raw shared-dictionary transfer.

8. Perceiver IO, Flamingo, and BLIP-2 Q-Former
   - https://arxiv.org/abs/2107.14795
   - https://arxiv.org/abs/2204.14198
   - https://arxiv.org/abs/2301.12597
   - Boundary: learned query/resampler connectors are established continuous
     bottlenecks. They remain future work unless implemented under the
     fixed-byte or explicit-rate contract.

9. Prefix-Tuning, Prompt Tuning, and P-Tuning v2
   - https://aclanthology.org/2021.acl-long.353/
   - https://arxiv.org/abs/2104.08691
   - https://arxiv.org/abs/2110.07602
   - Boundary: prompt/prefix methods inject continuous learned vectors. This
     artifact transmits only a per-example discrete byte.

10. C2C, KVComm, and KVCOMM
    - https://arxiv.org/abs/2510.03215
    - https://openreview.net/forum?id=LeatkxrBCi
    - https://arxiv.org/abs/2510.03346
    - https://arxiv.org/abs/2510.12872
    - Boundary: these communicate or fuse KV/cache states. This scout sends no
      source KV, no raw hidden vector, and no raw score vector.

11. QJL and TurboQuant
    - https://arxiv.org/abs/2406.03482
    - https://arxiv.org/abs/2504.19874
    - Boundary: these are vector/KV compression methods. LatentWire packets are
      task-level evidence sidebands, not vector-fidelity codecs.

## Reviewer-Facing Framing

Safe:

- We tested a train-only linear shared projection after raw-hidden and
  anchor-relative codebooks failed.
- The matched source code does not beat compact packet-only by a meaningful or
  statistically stable margin.
- The result argues that the next method branch should be nonlinear and
  output-aware, such as a cross-attention resampler or true sparse crosscoder
  objective.

Unsafe:

- Claiming CCA/common-basis alignment is novel.
- Claiming a shared latent language.
- Claiming equivalence to SAE/crosscoder learned dictionaries.
- Claiming superiority over prefix tokens, C2C/KVComm, or KV quantization.
- Claiming GPU serving or HBM wins before native NVIDIA measurements.
