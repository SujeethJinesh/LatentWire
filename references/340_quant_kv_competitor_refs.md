# Quantization, KV-Cache Compression, and Competitor Benchmark References for LatentWire

Primary-source memo for LatentWire’s bridge design. The main lesson from this cluster is not “compress harder”; it is that the bridge must preserve the right subspace, with the right asymmetry, under a fixed byte and latency budget.

The most defensible LatentWire hypothesis after these sources is:

- routing state and content state should not be compressed identically
- basis choice matters before any low-bit transport
- a tiny protected path may outperform uniform compression
- benchmark claims must be made at matched bytes, matched wall-clock, and matched repair budget

| Source | Primary links | Mechanism worth borrowing | Concrete LatentWire ablation | Telemetry fields to log | Claim risk |
|---|---|---|---|---|---|
| **AWQ** | [paper](https://arxiv.org/abs/2306.00978), [repo](https://github.com/mit-han-lab/llm-awq) | Activation-aware weight quantization protects a small set of salient channels and quantizes the rest aggressively. The design lesson is saliency-based protection, not uniform bit reduction. | Add a protected-channel path for the top `p%` bridge channels and compare it to uniform 4-bit transport at the same byte budget. | Protected-channel mass, per-channel activation outlier mass, byte budget, accuracy by slice, failure cases | Do not claim a bridge win unless the protected path is compared at equal bytes and equal repair compute. |
| **GPTQ** | [paper](https://arxiv.org/abs/2210.17323), [repo](https://github.com/IST-DASLab/gptq) | One-shot PTQ with approximate second-order information and greedy residual compensation. The important part is local error correction after each block. | Quantize bridge blocks sequentially and test whether compensating later blocks reduces end-to-end transport error. | Per-stage residual norm, order of quantization, final reconstruction loss, final task accuracy | A final accuracy gain alone is not enough; if the residuals are unstable, the method is fragile. |
| **SmoothQuant** | [paper](https://arxiv.org/abs/2211.10438), [repo](https://github.com/mit-han-lab/smoothquant) | Offline smoothing moves activation outliers into weights via a mathematically equivalent transform. This is a basis/scale management trick, not just a compression trick. | Add a smoothing transform before latent transport and compare `identity`, `SmoothQuant-style smoothing`, and `no smoothing`. | Pre/post smoothing activation range, cosine drift, reconstruction loss, accuracy per byte | Any win must be attributed to the transform, not to hidden prompt changes or different decode settings. |
| **EXL2 / mixed precision** | [repo](https://github.com/turboderp-org/exllamav2) | EXL2 mixes bit-widths to hit a target average bitrate. The key idea is adaptive bit allocation, not a single global precision. | Sweep a fixed average-byte budget across `uniform 4-bit`, `mixed-bit per layer`, and `mixed-bit per head / route`. | Average bits, per-layer bit allocation, accuracy, wall-clock, bytes transmitted | Mixed precision is easy to overclaim; the allocation rule must be explicit and reproducible. |
| **QuIP** | [paper](https://arxiv.org/abs/2307.13304), [repo](https://github.com/Cornell-RelaxML/QuIP) | Incoherence preprocessing with random orthogonal matrices reduces quantization difficulty in low-bit regimes. | Insert a transport-basis transform before quantizing the bridge and compare `identity`, `random orthogonal`, and `Hadamard` preprocessing. | Basis choice, reconstruction loss, angle drift, byte budget, outlier mass before/after transform | Do not claim “rotation helped” unless the gain survives a uniform-byte baseline and the same candidate pool. |
| **QuIP#** | [paper](https://arxiv.org/abs/2402.04396), [repo](https://github.com/Cornell-RelaxML/quip-sharp) | Lattice codebooks plus incoherence processing improve ultra-low-bit quantization. This is the strongest source for “geometry-aware” compression. | Compare a lattice-like bridge codebook against plain scalar quantization on the same latent payload. | Codebook occupancy, quantization error, entropy of code usage, task accuracy | Codebook wins can be brittle if they only help one model family or one prompt format. |
| **KIVI / KVQuant** | [KIVI paper](https://arxiv.org/abs/2402.02750), [KIVI repo](https://github.com/jy-yuan/KIVI), [KVQuant paper](https://arxiv.org/abs/2401.18079), [KVQuant repo](https://github.com/SqueezeAILab/KVQuant) | KIVI is the clean asymmetric KV-cache baseline; KVQuant adds per-channel keys, pre-RoPE quantization, and non-uniform datatypes. This is the most direct inspiration for K/V asymmetry and RoPE-sensitive compression. | Compare `K-only high precision`, `V-only high precision`, `pre-RoPE vs post-RoPE key quantization`, and `per-channel vs per-token` K/V quantization. | Key/value bit split, pre/post-RoPE error, per-channel outlier mass, retrieval accuracy, long-context accuracy | A win only means something if the K/V split and the RoPE placement are both explicit and budget-matched. |
| **KVPress** | [repo](https://github.com/NVIDIA/kvpress) | A modular toolkit for KV cache compression. The value here is not a single method; it is a clean test harness with pluggable presses. | Reuse the same bridge interface to evaluate multiple compression policies without changing the downstream evaluator. | Policy name, cache ratio, retained-token histogram, latency, memory peak | A framework win is not a method win. Separate interface convenience from actual algorithmic gains. |
| **H2O** | [paper](https://arxiv.org/abs/2306.14048), [repo](https://github.com/FMInference/H2O) | Heavy-hitter plus recency retention shows that a small token subset often carries most of the useful state. | Test whether LatentWire bridge memory should preserve heavy hitters, recency, or a hybrid of both. | Heavy-hitter score, recency score, retained-token overlap, accuracy by sequence position | Heavy-hitter heuristics can look strong on one benchmark and fail on reasoning-heavy prompts. |
| **SnapKV** | [paper](https://arxiv.org/abs/2404.14469) | Prompt-side attention observation can identify which KV positions matter before generation. | Compare `observation-window` selection against uniform retention and against a route-specific importance score. | Observation-window size, selected positions, prompt-vs-decode retention overlap, accuracy | SnapKV-style selection is prompt sensitive; it may not transfer to cross-model communication without retuning. |
| **Quest** | [repo](https://github.com/mit-han-lab/quest) | Query-aware top-k KV loading is a practical sparsity recipe for long-context inference. | Use query-aware loading as a baseline for “which latent state gets transmitted” in LatentWire. | Query score, loaded-page count, load latency, end accuracy | Query-aware sparsity is not the same as semantic transfer; do not over-interpret it as communication. |
| **KVzip** | [paper](https://arxiv.org/abs/2505.23416), [project](https://janghyun1230.github.io/kvzip/), [repo](https://github.com/snu-mllab/kvzip) | Query-agnostic KV eviction with context reconstruction is a direct fit for reusable bridge payloads. | Compare one-shot compressed payloads versus query-agnostic payloads reconstructed before decode. | Reconstruction loss, reuse rate, latency, memory reduction, downstream accuracy | Query-agnostic reuse can hide failure on out-of-distribution prompts. |
| **C2C** | [paper](https://arxiv.org/abs/2510.03215), [repo](https://github.com/thu-nics/C2C) | Direct KV-cache fusion is the main competitor benchmark for semantic cross-model communication. It is the clearest “no text in the middle” baseline. | Benchmark LatentWire against C2C-style direct cache fusion under matched byte budget, matched model pair, and matched repair budget. | Byte budget, route length, projection layers used, target-side repair cost, final accuracy | C2C is a moving target; the comparison must specify exact model pair, prompt set, and compute envelope. |
| **KVComm** | [paper](https://arxiv.org/abs/2510.03346), [repo](https://github.com/HankYe/KVCOMM) | Selective KV sharing is the closest benchmark for selective semantic transmission across models. | Compare LatentWire against selective-layer sharing and selective-token sharing as direct competitor baselines. | Shared-layer fraction, shared-token fraction, accuracy, latency, memory | Any win must survive identical prompt formatting and identical layer-selection policy definitions. |

## What These Sources Suggest for LatentWire

- **Asymmetric transport is probably necessary.** Keys, values, routing state, and content state should not share one compression policy.
- **A better basis may matter as much as a better bit-width.** Rotation, Hadamard-style transforms, and incoherence preprocessing are the most defensible first tests.
- **A small protected path is worth testing early.** If a few channels or a few layers dominate the error, uniform compression is likely leaving performance on the table.
- **Direct competitor baselines should be budget-matched.** C2C and KVComm are the right semantic-communication comparators, but only if the compute envelope is explicitly controlled.

## Concrete LatentWire Ablations

1. **Matched-byte K/V asymmetry**
   - Sweep `K_bits != V_bits` while holding total bytes fixed.
   - Add a control where `K_bits == V_bits`.
   - Log whether routing quality or content quality dominates the error.

2. **Protected outlier channels**
   - Keep the top `p%` bridge channels in FP16 and compress the rest.
   - Compare against uniform low-bit transport at the same byte budget.
   - Check whether the gain comes from a few extreme channels or from broad redistribution.

3. **Rotation before transport**
   - Compare `identity`, `random orthogonal`, `Hadamard`, and any learned orthogonal transform.
   - Measure whether the transform reduces outlier mass or just moves the error around.

4. **Mixed-bit bridge allocation**
   - Allocate bits per layer or per head under a fixed average budget.
   - Compare to uniform 4-bit and to an explicit saliency-based allocation.
   - This is the EXL2-style bridge analogue.

5. **Pre-RoPE vs post-RoPE keys**
   - Quantize or compress keys before vs after positional rotation.
   - Evaluate long-context retrieval and answer stability separately.

6. **Cache policy baselines**
   - Add H2O-style heavy-hitter retention, SnapKV-style observation-window selection, and Quest-style query-aware loading as low-cost baselines.
   - If LatentWire cannot beat these on its own terms, the bridge is not yet robust enough for a paper claim.

7. **Direct semantic-communication baselines**
   - Compare against C2C and KVComm with exact same prompt set, same model pair, and same token/byte budget.
   - Report both raw accuracy and budget-normalized accuracy.

## Benchmark Bootstraps

- **C2C / KVComm-style semantic communication**
  - Use as direct competitor anchors for cross-model communication.
  - Report matched-model and matched-budget results only.

- **KV compression baselines**
  - `H2O`, `SnapKV`, `Quest`, and `KVPress` should be treated as cache-policy baselines, not as communication methods.
  - Use them to test whether LatentWire’s gains come from semantics or from better cache curation.

- **Quantization baselines**
  - `AWQ`, `GPTQ`, `SmoothQuant`, `QuIP`, and `QuIP#` should be used as bridge-compression baselines.
  - Keep byte budgets and decode settings identical across methods.

## Telemetry Fields Worth Standardizing

- exact byte budget
- token budget and repair budget
- pre-repair accuracy
- post-repair accuracy
- repair help / repair harm
- target-selection rate
- outlier mass before and after any transform
- per-layer or per-head bit allocation
- retained-token histogram
- reconstruction loss and cosine drift
- wall-clock latency
- model pair, prompt template, and decode temperature

## Claim Risks

- A raw accuracy gain is not enough if the method adds unreported repair compute.
- A mixed-precision gain is not enough if the bit allocator is not explicit and reproducible.
- A basis-transform gain is not enough if the same effect can be obtained by simply increasing bytes.
- A cache-policy gain is not enough if it does not beat direct semantic-communication baselines like `C2C` and `KVComm` at the same budget.
- A benchmark win on one prompt family is not enough to claim general cross-model communication.

## Practical Readout

If LatentWire is going to be paper-safe, the next experiments should answer three questions:

1. Does the bridge still win when the byte budget is matched?
2. Does the bridge still win when the repair budget is matched?
3. Does the bridge still win against direct semantic-communication competitors, not just against cache heuristics?

If the answer to any of those is “no”, the result is still useful, but it is a diagnostic result, not a method claim.
