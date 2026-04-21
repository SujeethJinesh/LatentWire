# Quantization and KV Communication References for LatentWire

Primary-source memo for compression methods that are most relevant to cross-model latent or KV communication. The shared design lesson across these papers is not simply "use fewer bits"; it is to preserve the right channels, in the right basis, with the right asymmetry.

Inference for LatentWire: the bridge should probably not be a single uniform compressor. The more defensible hypothesis is that transport quality depends on outlier-channel protection, K/V asymmetry, and basis-aware preprocessing before any low-bit packing.

| Source | Primary links | Mechanism to steal | Concrete LatentWire ablation |
|---|---|---|---|
| **AWQ** | [paper](https://arxiv.org/abs/2306.00978), [repo](https://github.com/mit-han-lab/llm-awq) | Activation-aware weight quantization identifies a small set of salient channels to preserve while compressing the rest aggressively. The key idea is saliency-based protection, not uniform bit reduction. | Add a protected-channel path for the top `p%` latent/KV channels and compare it to uniform 4-bit packing at the same byte budget. Log task accuracy, protected-channel retention, and failure cases on outlier-heavy prompts. |
| **OWQ** | [paper](https://arxiv.org/abs/2306.02272), [repo](https://github.com/xvyaward/owq) | Outlier-aware weight quantization keeps a few weak columns in FP16 and quantizes the rest. This is the cleanest explicit outlier-channel baseline. | Keep a tiny FP16 escape path for the most outlier-prone bridge channels and quantify everything else. Compare `no protection`, `hard protection`, and `learned protection` under a fixed byte budget. |
| **SmoothQuant** | [paper](https://arxiv.org/abs/2211.10438), [repo](https://github.com/mit-han-lab/smoothquant) | Offline smoothing migrates activation outliers into weights via a mathematically equivalent transform, enabling W8A8 inference with little accuracy loss. | Add a smoothing transform before latent transport and compare `identity`, `SmoothQuant-style smoothing`, and `no smoothing`. Measure cosine drift, reconstruction loss, and accuracy per byte. |
| **GPTQ** | [paper](https://arxiv.org/abs/2210.17323), [repo](https://github.com/IST-DASLab/gptq) | One-shot PTQ with approximate second-order information and greedy error compensation. The important part is local residual correction after each quantized block. | Quantize bridge blocks sequentially and test whether compensating later blocks reduces end-to-end transport error. Compare quantization orderings and report per-stage residual norm, not just final accuracy. |
| **EXL2 / mixed precision** | [repo/docs](https://github.com/turboderp-org/exllamav2#exl2-quantization) | EXL2 mixes 2-8 bit levels within a model to hit a target average bitrate. The central idea is adaptive bit allocation, not a single global precision. | Sweep a fixed average-byte budget across `uniform 4-bit`, `mixed-bit per layer`, and `mixed-bit per head / route`. Log bytes, accuracy, and whether the bits concentrate on bridge layers that carry most task signal. |
| **QuaRot / SpinQuant** | [QuaRot paper](https://arxiv.org/abs/2404.00456), [QuaRot repo](https://github.com/spcl/QuaRot), [SpinQuant paper](https://arxiv.org/abs/2405.16406), [SpinQuant repo](https://github.com/facebookresearch/SpinQuant) | Rotations can remove outliers and make all of weights, activations, and KV cache easier to quantize. SpinQuant shows that learned rotations can beat random ones. | Insert a transport-basis transform before quantizing the bridge. Compare `identity`, `random orthogonal`, `Hadamard`, and `learned orthogonal` preprocessing, then quantify the same payload. |
| **KV-cache quantization** | [KIVI paper](https://arxiv.org/abs/2402.02750), [KVQuant paper](https://arxiv.org/abs/2401.18079), [KVQuant repo](https://github.com/SqueezeAILab/KVQuant), [XQuant paper](https://arxiv.org/abs/2510.11236) | KIVI makes keys per-channel and values per-token; KVQuant adds per-channel pre-RoPE key quantization, non-uniform datatypes, and dense-sparse outlier handling; XQuant pushes toward ultra-low-bit cross-layer cache compression. | Split LatentWire transport into `K-like routing state` and `V-like content state`. Compare `K-only high precision`, `V-only high precision`, `pre-RoPE vs post-RoPE keys`, and `cross-layer cache sharing` under the same byte cap. |
| **APTQ / mixed-precision sensitivity** | [paper](https://arxiv.org/abs/2402.14866) | Attention-aware mixed-precision quantization uses Hessian trace plus attention-output sensitivity to choose where higher precision is worth spending. | Use sensitivity-weighted bit allocation for the bridge: keep a few high-sensitivity layers/channels at higher precision and compress the rest harder. Compare against a flat bitrate schedule. |

## What these papers collectively suggest

- Outliers are the enemy, but the cure is not always the same. AWQ/OWQ protect channels, SmoothQuant moves outlier burden into a better place, and QuaRot/SpinQuant change basis so the outliers are less harmful.
- Keys and values should not be treated symmetrically. KIVI and KVQuant both point to asymmetric treatment, with keys usually demanding more careful structure-aware handling than values.
- Bit budgets should be assigned by sensitivity, not by tensor shape. EXL2 and APTQ both argue for nonuniform allocation when accuracy matters.
- For LatentWire, this is an inference-driven design claim: the bridge should likely separate `routing state`, `content state`, and `outlier state` before applying uniform compression.

## Highest-priority LatentWire ablations

1. **Asymmetric K/V transport**: sweep `K_bits != V_bits` with matched total bytes. This is the clearest direct test of whether routing information needs more fidelity than content.
2. **Protected outlier channels**: keep the top saliency channels in FP16 and quantize the rest. This tests whether a tiny escape path is enough to recover most of the lost signal.
3. **Rotation before transport**: compare `identity`, `random orthogonal`, `Hadamard`, and `learned orthogonal` preprocessing before quantizing the bridge. This isolates basis choice from bit budget.
4. **Mixed-bit layer routing**: allocate bits per layer or per head under one fixed average budget. This tests whether bridge layers are more sensitive than non-bridge layers.
5. **Pre-RoPE vs post-RoPE key quantization**: quantize keys before vs after positional rotation and compare long-context retrieval, not just perplexity.

## What to log

- Exact byte budget, not just nominal bit width.
- Per-channel or per-head outlier mass.
- Reconstruction loss before and after any rotation or smoothing step.
- Accuracy on both short-context and long-context slices.
- Whether the method wins because of geometry, outlier protection, or bit reallocation.

## Guardrails

- Always compare at matched bytes and matched wall-clock.
- Always include a uniform baseline and a no-transform baseline.
- Do not claim a win from mixed precision unless the allocation rule is explicit and reproducible.
- Keep the bridge frozen while testing compression unless the ablation is explicitly about adaptation.
