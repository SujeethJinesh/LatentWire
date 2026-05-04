# HellaSwag Qwen-To-Phi Official-Train Source Dictionary References

Date: 2026-05-04

## Gate Boundary

This memo supports the failed official-train Qwen source-side utility
dictionary gate. The gate is not a new learning-to-rank, calibration,
selective-classification, or source-coding theory result. Its possible
LatentWire contribution would have been narrower: learn a frozen source-side
dictionary on official-train Qwen rows, then transmit only a byte-scale
candidate-frontier code to a cross-family receiver.

The result is negative, so these references should be used to explain the
boundary and motivate the next receiver-calibrated branch.

## Benchmark

- HellaSwag: Zellers et al., "HellaSwag: Can a Machine Really Finish Your
  Sentence?" ACL 2019. https://arxiv.org/abs/1905.07830

## Pairwise Utility And Ranking

- Bradley and Terry, "Rank Analysis of Incomplete Block Designs: I. The Method
  of Paired Comparisons", Biometrika 1952.
  https://academic.oup.com/biomet/article/39/3-4/324/326091
- Burges et al., "Learning to Rank using Gradient Descent", ICML 2005.
  https://icml.cc/Conferences/2005/proceedings/papers/012_LearningToRank_BurgesEtAl.pdf
- Christiano et al., "Deep Reinforcement Learning from Human Preferences",
  NeurIPS 2017.
  https://papers.nips.cc/paper/7017-deep-reinforcement-learning-from-human-preferences
- Rafailov et al., "Direct Preference Optimization: Your Language Model is
  Secretly a Reward Model", NeurIPS 2023.
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html

Boundary: the source dictionary borrows the pairwise utility lens, but it is
not trying to rank documents or learn a general reward model. It predicts a
single protected hybrid-vs-rival swap and sends only the selected candidate ID.

## Calibration, Selective Prediction, And Deferral

- Niculescu-Mizil and Caruana, "Predicting Good Probabilities with Supervised
  Learning", ICML 2005.
  https://icml.cc/Conferences/2005/proceedings/papers/079_GoodProbabilities_NiculescuMizilCaruana.pdf
- Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
  https://proceedings.mlr.press/v70/guo17a
- Chow, "On Optimum Recognition Error and Reject Tradeoff", IEEE Transactions
  on Information Theory 1970.
  https://research.ibm.com/publications/on-optimum-recognition-error-and-reject-tradeoff
- El-Yaniv and Wiener, "On the Foundations of Noise-free Selective
  Classification", JMLR 2010. https://jmlr.org/papers/v11/el-yaniv10a.html
- Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks",
  NeurIPS 2017. https://papers.neurips.cc/paper/7073-selective-classification-for-deep-neural-networks
- Romano, Sesia, and Candes, "Classification with Valid and Adaptive Coverage",
  NeurIPS 2020.
  https://proceedings.neurips.cc/paper_files/paper/2020/hash/244edd7e85dc81602b7615cd705545f5-Abstract.html
- Angelopoulos et al., "Conformal Risk Control", ICLR 2024.
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
- Madras et al., "Predict Responsibly: Improving Fairness and Accuracy by
  Learning to Defer", NeurIPS 2018.
  https://papers.nips.cc/paper/7853-predict-responsibly-improving-fairness-and-accuracy-by-learning-to-defer
- Mozannar and Sontag, "Consistent Estimators for Learning to Defer to an
  Expert", ICML 2020. https://proceedings.mlr.press/v119/mozannar20b.html
- Verma and Nalisnick, "Calibrated Learning to Defer with One-vs-All
  Classifiers", ICML 2022. https://proceedings.mlr.press/v162/verma22c.html

Boundary: LatentWire's receiver problem is a selective deferral problem over a
source-provided candidate frontier, but the failed gate used source-side
calibration only. The next live branch needs receiver-side calibration rather
than more source-only thresholds.

## Side-Information Source Coding

- Slepian and Wolf, "Noiseless Coding of Correlated Information Sources", IEEE
  Transactions on Information Theory 1973.
  https://www.itsoc.org/publications/papers/noiseless-coding-of-correlated-information-sources
- Wyner and Ziv, "The Rate-Distortion Function for Source Coding with Side
  Information at the Decoder", IEEE Transactions on Information Theory 1976.
  https://cir.nii.ac.jp/crid/1360564063947537280
- Pradhan and Ramchandran, "Distributed Source Coding Using Syndromes (DISCUS):
  Design and Construction", IEEE Transactions on Information Theory 2003.
  https://doi.org/10.1109/TIT.2002.808103

Boundary: the shared HellaSwag candidate IDs act like a tiny common basis and
the receiver has side information from its own model scores. The negative
result says the current source-only dictionary is not the right code for that
side information.

## Iterative Refinement And Diffusion Inspiration

- Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020.
  https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html
- Song et al., "Score-Based Generative Modeling through Stochastic
  Differential Equations", ICLR 2021.
  https://research.google/pubs/score-based-generative-modeling-through-stochastic-differential-equations/

Boundary: diffusion/score methods motivate iterative latent repair, but this
gate is a one-shot dictionary. Diffusion-style refinement should be treated as
a future branch only if it can be formulated as a bounded receiver-side repair
with strict source-private controls.

## Systems And Quantization Context

- QJL: "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero
  Overhead", arXiv 2024. https://arxiv.org/abs/2406.03482
- KIVI: "A Tuning-Free Asymmetric 2bit Quantization for KV Cache", ICML 2024.
  https://arxiv.org/abs/2402.02750
- KVQuant: "KVQuant: Towards 10 Million Context Length LLM Inference with KV
  Cache Quantization", NeurIPS 2024. https://arxiv.org/abs/2401.18079
- TurboQuant: "TurboQuant: Taming LLMs with Ternary Quantization", arXiv 2025.
  https://arxiv.org/abs/2504.19874
- FlashAttention: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness", NeurIPS 2022. https://arxiv.org/abs/2205.14135
- vLLM: Kwon et al., "Efficient Memory Management for Large Language Model
  Serving with PagedAttention", SOSP 2023. https://arxiv.org/abs/2309.06180

Boundary: QJL/KIVI/KVQuant/TurboQuant compress continuous state or KV cache.
LatentWire sends a task-level candidate decision packet. Because this gate is
negative on quality, we cannot claim a systems win from smaller bytes; these
methods are threat-model and byte-floor context only.

## Paper Implication

The official-train source dictionary should be logged as weakened. For ICLR,
the next positive branch should add receiver-side information or a new
interface. For COLM, this result is useful as a reviewer-clean falsification:
larger source-only calibration is not enough, even though oracle headroom
remains large.
