# Quantization And Mixed-Precision Routing References

Primary-source memo for using quantization ideas as inspiration for LatentWire
communication: calibration-aware saliency, mixed precision, outlier channels,
rotation/equalization, and error-compensated residuals.

| Source | Primary links | Transferable idea | Concrete LatentWire ablation |
|---|---|---|---|
| **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration** | [paper](https://arxiv.org/abs/2306.00978), [code](https://github.com/mit-han-lab/llm-awq) | Protect a small set of activation-salient channels instead of treating all coordinates equally. | Learn or calibrate a protected KV/channel mask using activation magnitude, QK fidelity, and answer-flip saliency. Compare fixed channels, PCA channels, supervised signal channels, and AWQ-style activation channels. |
| **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** | [paper](https://arxiv.org/abs/2210.17323), [code](https://github.com/IST-DASLab/gptq) | Use second-order error compensation so local compression decisions minimize downstream reconstruction loss. | Add Hessian/Gram-weighted residual bridge fitting for translated K/V, then compare against plain ridge and low-rank correction at matched bytes. |
| **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models** | [paper](https://arxiv.org/abs/2211.10438), [code](https://github.com/mit-han-lab/smoothquant) | Move activation outliers into weights with a calibrated scaling transform. | Add a source-target per-channel scaling equalization before Procrustes/CCA transport; log outlier mass, post-scale cosine, and reasoning accuracy. |
| **QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs** | [paper](https://arxiv.org/abs/2404.00456), [code](https://github.com/spcl/QuaRot) | Orthogonal rotations can spread outliers and make quantized spaces easier to compress. | Test random/Hadamard/DCT rotations as gauge preconditioners before KV transport and protected-channel selection. |
| **SpinQuant: LLM Quantization with Learned Rotations** | [paper](https://arxiv.org/abs/2405.16406) | Learn rotations to improve quantization geometry rather than relying on fixed transforms. | Train a small orthogonal adapter that optimizes K/V reconstruction plus downstream QK-fidelity, then freeze it for held-out routing. |
| **EXL2 / ExLlamaV2 mixed-bit quantization** | [code](https://github.com/turboderp-org/exllamav2) | Allocate different bit widths to different tensors/layers under a target average bit rate. | Add mixed-budget KV transport: spend more route/value slots or bits on high-flip-saliency layers and fewer on stable layers. Report accuracy per transmitted byte. |
| **SqueezeLLM: Dense-and-Sparse Quantization** | [paper](https://arxiv.org/abs/2306.07629), [code](https://github.com/SqueezeAILab/SqueezeLLM) | Separate dense bulk structure from sparse outliers. | Split translated KV into dense low-rank transport plus sparse residual/outlier packets. Ablate sparse residual budget and outlier selection metric. |
| **OmniQuant: Omnidirectionally Calibrated Quantization for LLMs** | [paper](https://arxiv.org/abs/2308.13137), [code](https://github.com/OpenGVLab/OmniQuant) | Calibrate learnable clipping/scaling using reconstruction loss after quantization. | Add post-transport calibration knobs for per-layer clipping, scale, and residual gate; optimize on calibration prompts and report held-out generalization. |

## Highest-Priority LatentWire Tests

- AWQ-style protected channels: compare activation-salient, QK-salient,
  supervised-signal, PCA, and fixed-coordinate masks.
- Mixed-budget transport: allocate byte budget by flip saliency, QK error, or
  layer sensitivity instead of uniform layer budgets.
- Rotation/equalization preconditioner: SmoothQuant-style scaling plus
  QuaRot/SpinQuant-style orthogonal transforms before Procrustes/CCA.
- Dense-plus-sparse residual packet: low-rank K/V transport plus sparse
  outlier residuals, measured by reconstruction error and paired GSM flips.
- Seed-ensemble compression: use stochastic KV routes as candidates, then
  compress/verify the final answer rather than trusting one random mask.

## Guardrails

- Any mixed-budget method must report bytes, selected layers, and per-layer
  allocation; otherwise it can hide extra compute.
- Any protected-channel win must include rotated and slot-permuted controls.
- Any stochastic route win must report seed variance and paired flips, not only
  aggregate accuracy.
