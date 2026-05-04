# References: Qwen-To-Phi Harm-Controlled Bucket Gate

## Why These References Matter

This gate tests a Wyner-Ziv-style receiver with side information: Qwen sends a
tiny quantized innovation packet, and Phi accepts it only when official-train
evidence indicates low harm. The result is negative, so these references are
used to locate the boundary of the failed branch and to motivate the next
interface branch.

## Selective Prediction, Deferral, And Risk Control

- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks,"
  NeurIPS 2017.
  https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks

- Madras et al., "Predict Responsibly: Improving Fairness and Accuracy by
  Learning to Defer," NeurIPS 2018.
  https://arxiv.org/abs/1711.06664

- Angelopoulos et al., "Conformal Risk Control," ICLR 2024.
  https://arxiv.org/abs/2208.02814

Boundary: these works calibrate answer/defer decisions or risk. LatentWire's
candidate packet adds an explicit source-private innovation channel. Since the
harm-controlled bucket gate selected no safe override buckets, we cannot claim a
new positive selective receiver yet.

## Side-Information Coding

- Slepian and Wolf, "Noiseless Coding of Correlated Information Sources," IEEE
  Transactions on Information Theory, 1973.
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources

- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder," IEEE Transactions on Information Theory, 1976.
  https://doi.org/10.1109/TIT.1976.1055508

Boundary: the right theoretical analogy is conditional innovation under decoder
side information. The failed gate shows that the current quantized
candidate-switch packet does not provide enough calibrated conditional signal
for Phi to exploit safely.

## Direct Model-To-Model And KV Communication Baselines

- "Cache-to-Cache: Direct Semantic Communication Between Large Language
  Models."
  https://openreview.net/forum?id=LeatkxrBCi
  https://arxiv.org/abs/2510.03215

- "KVComm: Enabling Efficient LLM Communication through Selective KV Sharing."
  https://openreview.net/forum?id=F7rUng23nw

- "KVCOMM: Online Cross-context KV-cache Communication for Efficient LLM-based
  Multi-agent Systems."
  https://arxiv.org/abs/2510.12872

Boundary: these methods share or fuse high-dimensional KV/cache state. The
current LatentWire packet is much stricter: no source KV, hidden vectors,
scores, logits, or text are transmitted. However, the present learned receiver
does not beat fixed hybrid, so this is not yet a stronger communication method.

## Prefix, Gist, And Compact Context Interfaces

- Li and Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation,"
  ACL-IJCNLP 2021.
  https://aclanthology.org/2021.acl-long.353/

- Mu et al., "Learning to Compress Prompts with Gist Tokens," NeurIPS 2023.
  https://arxiv.org/abs/2304.08467

Boundary: prefix/gist methods are stronger precedents for the next live branch:
learned compact target-side interfaces. The bucket gate's no-op selection
suggests shallow packet switching is saturated and that a target self-resonance
or soft-token intervention should be tested next.

## Quantization And Hardware/Systems Comparators

- "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate."
  https://arxiv.org/abs/2504.19874

- "QJL: 1-Bit Quantized Johnson-Lindenstrauss Transform for KV Cache
  Quantization."
  https://arxiv.org/abs/2406.03482

- "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache."
  https://arxiv.org/abs/2402.02750

- "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache
  Quantization."
  https://arxiv.org/abs/2401.18079

Boundary: these works compress high-dimensional KV/vector state. They are
mandatory systems baselines once NVIDIA measurements exist. The current
harm-controlled bucket packet remains a byte/exposure accounting artifact, not
a native serving win.

## Next-Branch Motivation

- Moschella et al., "Relative Representations Enable Zero-Shot Latent Space
  Communication," ICLR 2023.
  https://arxiv.org/abs/2209.15430

- Anthropic, "Crosscoders: A Cross-Layer Sparse Autoencoder for Model
  Diffing."
  https://transformer-circuits.pub/2024/crosscoders/index.html

These are not claims for the present result. They motivate the next branch:
find a richer shared basis or compact learned intervention, then subject it to
the same wrong-row, atom-shuffle, source-destruction, and target-cache controls.
