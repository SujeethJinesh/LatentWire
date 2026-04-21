# Quantization and Compression Inspirations for LatentWire

## Bottom line

The strongest LatentWire lesson from the quantization/compression literature is not just "make things smaller." It is:

1. identify the dominant geometric failure mode,
2. move the distortion onto a better-conditioned subspace,
3. preserve the task-critical directions with explicit allocation,
4. and log enough telemetry to tell whether the bridge learned communication or only learned a compression trick.

For LatentWire, that means the next useful ablations are not generic compression sweeps. They should test whether the bridge is improved by:

- per-channel or per-head allocation,
- orthogonal preconditioning,
- query-aware KV compression,
- product/codebook transport,
- and low-rank or sketch-based transport bottlenecks.

## Primary sources

### Weight / activation quantization

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456)
- [SpinQuant: LLM Quantization with Learned Rotations](https://arxiv.org/abs/2405.16406)
- [EXL2 implementation anchor in ExLlamaV2](https://github.com/turboderp-org/exllamav2)

### KV cache compression

- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)
- [KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization](https://arxiv.org/abs/2401.18079)
- [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- [QAQ: Quality Adaptive Quantization for LLM KV Cache](https://arxiv.org/abs/2403.04643)
- [KVPress](https://arxiv.org/abs/2510.00636)
- [KVLinC: KV Cache Quantization with Hadamard Rotation and Linear Correction](https://arxiv.org/abs/2510.05373)
- [KVzip: Query-Agnostic KV Cache Eviction with Context Reconstruction](https://arxiv.org/abs/2505.23416)
- [XQuant: Ultra-Low Bit KV Cache Quantization with Cross-Layer Compression](https://arxiv.org/abs/2510.11236)

### Product quantization and low-rank / sketching

- [Product Quantization for Nearest Neighbor Search](https://ieeexplore.ieee.org/document/5432202/)
- [Matrix Compression via Randomized Low Rank and Low Precision Factorization](https://arxiv.org/abs/2310.11028)
- [LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation](https://arxiv.org/abs/2306.11222)
- [Deterministic Matrix Sketches for Low-Rank Compression of High-Dimensional Simulation Data](https://arxiv.org/abs/2105.01271)

### Interface / tokenizer mismatch

- [Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083)
- [Token Alignment for Vocabulary Adaptation](https://arxiv.org/abs/2506.03523)
- [Cross-Tokenizer LLM Distillation Through a Byte-Level Interface](https://arxiv.org/abs/2604.07466)
- [Cross-Tokenizer Preference Distillation](https://arxiv.org/abs/2601.11865)

## Concrete math properties to reuse

### 1) SmoothQuant-style diagonal rescaling

SmoothQuant’s core move is a mathematically equivalent diagonal transform that shifts difficulty from activation outliers to weights.

For a linear map `y = xW`, define a positive diagonal matrix `S`.

```math
y = xW = (xS)(S^{-1}W)
```

The product is unchanged, but the quantization error landscape changes because the heavy-tailed coordinates can be redistributed. For LatentWire, this suggests a bridge preconditioner:

- rescale source hidden states before fitting the transport,
- rescale target-side bridge coefficients inversely,
- and test whether the bridge failure is mostly a conditioning problem.

### 2) GPTQ-style second-order residual fitting

GPTQ can be read as greedy residual minimization under a local quadratic approximation.

If `W` is the original matrix and `Q` is the quantized approximation, the layerwise objective is approximately:

```math
\min_Q \|W - Q\|_H^2 = \operatorname{tr}((W-Q)^\top H (W-Q))
```

where `H` is a local Hessian or curvature proxy. For LatentWire, the analog is to fit the bridge with a curvature-aware loss rather than only MSE. That matters if a small subset of bridge directions dominate task loss.

### 3) Rotation-based whitening

QuaRot and SpinQuant use orthogonal rotations to reduce outliers before low-bit quantization.

```math
x \mapsto xR, \qquad W \mapsto R^\top W
```

for orthogonal `R`. The bridge is unchanged in exact arithmetic, but the coordinate system becomes easier to quantize. For LatentWire, this suggests:

- orthogonal preconditioning of source/target residual streams,
- Hadamard or learned rotations before bridge fitting,
- and a controlled check that the gain is from conditioning, not extra capacity.

### 4) Asymmetric KV quantization

KV cache methods repeatedly separate `K` and `V` precision because the two tensors play different roles in attention.

That implies LatentWire should not treat all bridge tensors symmetrically. We should test:

- lower precision for keys than values,
- per-head asymmetric precision,
- and query-aware retention vs uniform compression.

### 5) Product quantization

PQ decomposes a vector into sub-vectors:

```math
x = [x^{(1)}, \dots, x^{(m)}]
```

and assigns each subvector to a codebook entry:

```math
q(x) = [c^{(1)}_{i_1}, \dots, c^{(m)}_{i_m}]
```

This is a direct template for route atoms or bridge atoms:

- assign one codebook per head or per layer block,
- use independent subspaces for content vs routing,
- and measure whether the bridge failure is caused by cross-subspace interference.

### 6) Low-rank sketching

Randomized sketching compresses a matrix `A` with a projection `S`:

```math
B = SA
```

with `S` chosen so that the sketch preserves geometry or subspace structure. In LatentWire terms, that is a low-rank transport bottleneck:

- project query features into a smaller sketch space,
- fit bridge transport in the sketch,
- and reconstruct only the task-critical directions.

### 7) Cache eviction vs cache quantization

The KV literature makes a clean distinction between:

- evicting unimportant entries,
- compressing retained entries,
- and using query-aware scoring to decide which entries survive.

For LatentWire, this matters because some failures are not due to insufficient capacity. They are due to the wrong entries surviving the bridge.

## Toy experiments to run first

### Toy 1: quantized bridge under gauge transforms

Construct a synthetic transport problem with latent state `z`, source K/V slots, and target queries. Apply:

- identity coordinates,
- random permutation of heads/slots,
- orthogonal rotation,
- and diagonal outlier scaling.

Compare:

- direct top-k routing,
- AWQ-style channel scaling,
- rotation-preconditioned transport,
- and low-rank sketch transport.

Measure reconstruction error, route entropy, and permutation robustness.

### Toy 2: outlier injection and bit-budget sweep

Inject heavy-tailed coordinates into a few channels and sweep:

- 2-bit, 4-bit, 8-bit, mixed precision,
- per-channel vs per-tensor scaling,
- symmetric vs asymmetric quantization.

This will tell us whether the bridge is mostly losing due to a few outlier channels or due to global mismatch.

### Toy 3: cache compression with query reuse

Build a synthetic needle task with repeated prompts and variable query reuse. Compare:

- no compression,
- eviction-only,
- quantization-only,
- eviction + quantization,
- query-aware retention.

This isolates whether the bridge needs stable memory or just a smaller memory footprint.

### Toy 4: codebook transport vs low-rank transport

Compare a PQ-style bridge codebook against a low-rank sketch bridge on the same synthetic latent task. Track:

- accuracy,
- reconstruction,
- dead-code rate,
- and robustness to head permutation.

If PQ wins, the bridge needs discrete route atoms. If low-rank wins, the bridge needs smooth geometry.

## LatentWire ablations to run next

1. **AWQ-style bridge scaling**
   Add per-channel bridge scaling on source hidden states before fitting the translator. Keep the same bit budget and compare against the current dense bridge.

2. **GPTQ-style curvature-aware bridge fit**
   Replace plain MSE with a Hessian-weighted or Fisher-weighted bridge objective. Use the same calibration set and check whether accuracy improves without increasing route collapse.

3. **SmoothQuant-style gauge migration**
   Apply an exact rescaling transform before bridge fitting, then invert it at the target interface. This tests whether the current failures are mostly conditioning failures.

4. **QuaRot / SpinQuant rotation preconditioning**
   Insert an orthogonal or Hadamard rotation on the residual stream before transport. Sweep whether the gain comes from whitening, not from extra capacity.

5. **KV-style asymmetric compression at the bridge boundary**
   Quantize keys and values differently, or keep keys high precision and compress values more aggressively. This is the LatentWire analog of asymmetric KV cache compression.

6. **Product-quantized route atoms**
   Split the bridge latent into subspaces and quantize each subspace with its own codebook. Track codebook usage and dead atoms.

7. **Low-rank sketch bridge**
   Force the bridge through a sketch matrix `S` and vary sketch rank. Compare reconstruction and task accuracy against the full bridge.

8. **Mixed-precision layer schedule**
   Allocate more bits or larger codebooks to the most sensitive layers / heads. Compare a uniform budget against an EXL2-style heterogeneous budget.

## What to log for interpretability

- per-layer bridge error,
- per-head route entropy,
- outlier mass before and after transforms,
- dead-code / dead-atom rate,
- codebook occupancy,
- bytes per sample and bytes per generated token,
- accuracy at fixed budget,
- and paired deltas against the target-alone baseline.

## Recommendation

The highest-value next branch is:

1. **SmoothQuant-style gauge migration**
2. **rotation preconditioning**
3. **query-aware KV-style asymmetric compression**

Those three together tell us whether LatentWire is failing because of coordinate conditioning, route selection, or budget allocation. If that stack still fails, the next move should be **product-quantized route atoms** or a **low-rank sketch bridge**, not more dense transport.
