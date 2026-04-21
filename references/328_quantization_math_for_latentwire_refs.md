# Quantization Math for LatentWire

This memo is a targeted bridge from classic quantization/compression math to
LatentWire design. The transferable lesson is not "use fewer bits" in the
abstract. It is:

1. choose a basis where the signal is flatter,
2. protect the few channels that dominate error,
3. allocate bits asymmetrically across keys, values, layers, and routes, and
4. compensate residual error locally instead of hoping one global compressor is
   enough.

That maps directly onto bridge design, cache transport, tokenizer transfer, and
test-time route selection.

## Core math patterns worth stealing

- **Equivalent transforms.** Many PTQ methods apply an identity-preserving
  reparameterization before compression:
  `W x = (W D^{-1})(D x)` or `Q(x) = R^T q(Rx)` with `R^T R = I`.
  The point is to move mass away from outlier coordinates before quantizing.
- **Saliency protection.** Not all channels contribute equally. AWQ-style
  methods protect a small set of salient channels and compress the rest more
  aggressively.
- **Second-order residual fitting.** GPTQ approximates the local objective by a
  quadratic form:
  `L(W + Δ) ≈ L(W) + g^T Δ + 1/2 Δ^T H Δ`.
  Quantization is then a residual-correction problem, not just a clipping
  problem.
- **Asymmetric budgets.** KV-cache methods almost always split precision
  unevenly across keys vs values, or across channels vs tokens.
  For LatentWire, this suggests `b_K != b_V`, and possibly
  `b_route != b_content`.
- **Adaptive bit allocation.** EXL2/TurboQuant-style systems spend more budget
  where geometry is hardest. The useful abstraction is budget allocation under
  a fixed byte cap, not one global precision.

## Classic references to port into LatentWire

| Method | Primary link | What the math says | LatentWire transfer |
|---|---|---|---|
| **EXL2 / mixed-bit packing** | [ExLlamaV2 EXL2 docs](https://github.com/turboderp-org/exllamav2) | Heterogeneous bit allocation under an average budget. The model is compressed where it is easy, preserved where it is hard. | Sweep `uniform 4-bit` vs `mixed-bit per layer/head/route` at fixed bytes. Log which bridge components absorb the budget. |
| **GPTQ** | [arXiv](https://arxiv.org/abs/2210.17323) | Greedy second-order residual compensation. Each block is quantized while the remaining error is pushed forward. | Quantize bridge modules sequentially and measure whether residual correction after each block reduces end-to-end transport loss. |
| **AWQ** | [arXiv](https://arxiv.org/abs/2306.00978) | Protect salient channels identified by activation statistics; scale them so the quantizer sees a flatter distribution. | Add protected-channel routing for bridge K/V channels and compare it to uniform compression at the same byte budget. |
| **SmoothQuant** | [arXiv](https://arxiv.org/abs/2211.10438) | Diagonal rescaling migrates difficulty from hard activations to easier weights with an equivalent transform. | Try a bridge preconditioner that smooths outlier latent channels before any projection or codebook step. |
| **QuaRot** | [arXiv](https://arxiv.org/abs/2404.00456) | Orthogonal rotation preserves outputs but changes coordinate geometry so outliers are easier to quantize. | Rotate bridge latents before compression and compare `identity`, `random orthogonal`, and `learned orthogonal` bases. |
| **SpinQuant** | [arXiv](https://arxiv.org/abs/2405.16406) | Learned rotations matter; random rotations can vary a lot, so geometry selection is itself part of the optimization. | Treat basis choice as a tunable variable and report variance across random orthogonal seeds, not just the best run. |
| **KIVI** | [arXiv](https://arxiv.org/abs/2402.02750) | Keys and values want different quantizers: keys are more channel-sensitive, values more token-sensitive. | Split LatentWire into routing-state vs content-state transport and test asymmetric budgets for each. |
| **KVQuant** | [arXiv](https://arxiv.org/abs/2401.18079) | Pre-RoPE key handling, per-channel treatment, and outlier-aware packing help preserve long-context fidelity. | Compare pre-transport vs post-transport rotation/quantization and measure attention-score drift on held-out prompts. |

## 2025-2026 update sweep

| Method | Primary link | New idea to watch | LatentWire diagnostic |
|---|---|---|---|
| **TurboQuant** | [arXiv](https://arxiv.org/abs/2504.19874) | Random rotation + coordinatewise quantization + residual inner-product correction gives near-optimal geometry preservation. | Test whether a random/learned rotation plus residual bridge correction beats direct low-rank transport at equal bytes. |
| **AQUA-KV** | [arXiv](https://arxiv.org/abs/2501.19392) | The compressible part of KV is what can be predicted; compact adapters recover the predictable remainder. | Add a "predictable vs residual" split to the bridge and report how much of the message is recoverable without high precision. |
| **PolarQuant** | [arXiv](https://arxiv.org/abs/2502.02617) | Random preconditioning makes the quantized coordinates more uniform; polar coordinates can remove explicit normalization overhead. | Compare Cartesian vs rotated vs polar-style latent packing and track reconstruction error per channel. |
| **LogQuant** | [arXiv](https://arxiv.org/abs/2503.19950) | Do not assume later tokens are the only important ones; use a distribution-aware filter across the full context. | Replace tail-only retention with distribution-aware token selection and verify whether the bridge is missing early but informative tokens. |
| **XQuant** | [arXiv](https://arxiv.org/abs/2510.11236) | Cross-layer compression can push below 2 bits when similar structure is shared across layers. | Try cross-layer latent sharing or codebook reuse instead of per-layer independent transport. |
| **KVLinC** | [arXiv](https://arxiv.org/abs/2510.05373) | Hadamard-style rotation plus linear correction repairs low-bit KV errors. | Add a lightweight correction head after latent transport and measure whether it repairs quantization-like bridge error. |
| **InnerQ** | [arXiv](https://arxiv.org/abs/2602.23200) | Inner-dimension grouping, recent-token windows, and attention-sink protection improve hardware-aware KV compression. | Test whether recent-token and sink-token windows should get explicit high-precision treatment in the bridge. |
| **MASQuant** | [arXiv](https://arxiv.org/abs/2603.04800) | Modalities need different smoothing factors; one shared preconditioner can misalign cross-modal statistics. | If LatentWire becomes multimodal, do not share one smoothing rule across modalities; use modality-specific bridge scaling. |

## Concrete LatentWire ablations and diagnostics

1. **Protected-channel vs uniform budget.** Keep the same byte budget and compare
   uniform compression against AWQ-style protected channels.
2. **Rotation before compression.** Compare identity, random orthogonal, learned
   orthogonal, Hadamard, and TurboQuant-style rotations.
3. **Key/value asymmetry.** Give keys and values different bit budgets and check
   whether the optimal split is stable across tasks.
4. **Residual correction.** Add a small GPTQ/KVLinC-style correction head after
   transport and log whether the error drops mainly in outlier-heavy examples.
5. **Cross-layer reuse.** Reuse codebooks or latent bases across layers instead
   of training each bridge independently.
6. **Geometry diagnostics.** Track cosine drift, attention-score drift,
   Procrustes error, outlier mass, and reconstruction loss per byte, not just
   end-task accuracy.

## Practical readout for the paper

- If a bridge only works after rotation, the issue is geometry, not capacity.
- If it only works with protected channels, the issue is outliers, not rank.
- If it needs asymmetric `K/V` budgets, then the bottleneck is transport
  structure, not just compression rate.
- If a light residual head recovers most of the loss, the bridge is close and
  the remaining problem is correction, not representation.
- If cross-layer sharing works, the model has reusable latent structure and we
  should stop paying per-layer transport costs.
