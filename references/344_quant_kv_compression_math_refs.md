# Quantization and KV-Compression Math References for LatentWire

This memo keeps only primary sources or official repos. The useful lesson is not "compress more" in the abstract. It is:

1. flatten the geometry before you quantize,
2. protect the few coordinates that dominate error,
3. allocate bits asymmetrically across roles,
4. and separate eviction, compression, and correction so we know which part actually helps.

## Reference Map

| Theme | Primary source links | Math idea worth stealing | LatentWire ablation | Telemetry fields to log | Claim risk |
|---|---|---|---|---|---|
| **AWQ / outlier protection** | [AWQ paper](https://arxiv.org/abs/2306.00978), [AWQ repo](https://github.com/mit-han-lab/llm-awq), [OWQ paper](https://arxiv.org/abs/2306.02272), [OWQ repo](https://github.com/xvyaward/owq), [OS+ paper](https://arxiv.org/abs/2304.09145), [OS+ repo](https://github.com/ModelTC/Outlier_Suppression_Plus) | Small sets of activation-sensitive channels dominate error; a diagonal rescale or explicit outlier reservation can make the rest easier to quantize. | Keep the top `p%` bridge channels in FP16 or BF16 and compress the rest under the same byte budget; compare with uniform low-bit transport. | Outlier mass, protected-channel fraction, per-channel error, byte budget, accuracy by slice | A win only counts if the same budget is used and the protected path is not secretly buying extra compute. |
| **SmoothQuant / equivalent scaling** | [SmoothQuant paper](https://arxiv.org/abs/2211.10438), [SmoothQuant repo](https://github.com/mit-han-lab/smoothquant) | A mathematically equivalent diagonal transform moves difficulty from activations into weights. | Apply a bridge preconditioner before fitting the translator, then invert it at the interface; compare identity vs smoothing. | Pre/post range, cosine drift, reconstruction loss, calibration-set sensitivity | If the result disappears when bytes are matched, the gain was conditioning, not a better bridge. |
| **QuIP / incoherence preprocessing** | [QuIP paper](https://arxiv.org/abs/2307.13304), [QuIP repo](https://github.com/Cornell-RelaxML/QuIP) | Orthogonal or randomized incoherence preprocessing reduces coordinate spikes before low-bit quantization. | Insert a random orthogonal or Hadamard-like rotation before bridge quantization and compare against identity. | Basis choice, outlier mass after rotation, reconstruction loss, task accuracy | Rotation gains are easy to overclaim unless the comparison is strictly budget-matched. |
| **QuIP# / rotation + lattice codebooks** | [QuIP# paper](https://arxiv.org/abs/2402.04396), [QuIP# repo](https://github.com/Cornell-RelaxML/quip-sharp) | Randomized Hadamard transforms and lattice codebooks replace scalar quantization with geometry-aware codebooks. | Compare scalar bridge quantization vs a small lattice/codebook bridge at equal bytes. | Codebook occupancy, dead-code rate, entropy of code usage, reconstruction loss | Codebook success can be model-family specific; do not generalize from one architecture or prompt mix. |
| **HIGGS / Hadamard + MSE-optimal grids** | [HIGGS paper](https://aclanthology.org/2025.naacl-long.543/), [arXiv](https://arxiv.org/abs/2411.17525) | A Hadamard-style basis plus an MSE-optimal grid gives a clean data-free quantization objective; the paper also frames non-uniform per-layer bit allocation as a dynamic-programming problem. | Try a HIGGS-style bridge quantizer for the latent payload and a DP-based per-layer bit schedule under a fixed budget. | Layerwise reconstruction error, per-layer bits, perplexity / task score, variance over calibration sets | If the per-layer schedule is not explicit, mixed-precision claims are not reproducible. |
| **QuaRot / SpinQuant / learned rotations** | [QuaRot paper](https://arxiv.org/abs/2404.00456), [QuaRot repo](https://github.com/spcl/QuaRot), [SpinQuant paper](https://arxiv.org/abs/2405.16406), [SpinQuant repo](https://github.com/facebookresearch/SpinQuant) | Orthogonal transforms preserve outputs while changing quantization geometry; learned rotations can beat random ones when outliers are structured. | Compare identity, random Hadamard, learned orthogonal, and per-head rotation before transport. | Seed variance, rotation norm, outlier mass, reconstruction loss, final accuracy | A learned rotation is extra optimization, so it must beat a strong random-rotation control. |
| **AQLM / additive and residual codebooks** | [AQLM paper](https://arxiv.org/abs/2401.06118), [AQLM repo](https://github.com/vahe1994/AQLM), [RVQ-for-KV paper](https://arxiv.org/abs/2410.15704), [RVQ repo](https://github.com/iankur/vqllm) | Additive or residual codebooks approximate a vector by a sum of codewords, making extreme compression viable when scalar quantization fails. | Replace the dense bridge with a small additive codebook or RVQ stack; compare to low-rank and scalar baselines at the same bytes. | Codebook depth, residual norm after each stage, dead-code rate, bytes/sample, accuracy | Codebook methods are brittle if the calibration set is too small or the codebook only fits one domain. |
| **EXL2 / heterogeneous bit allocation** | [ExLlamaV2 repo](https://github.com/turboderp-org/exllamav2) | EXL2’s core idea is average-bit budgeting with heterogeneous precision, not one global bit width. | Allocate bridge bits per layer, per head, or per route under a fixed average budget; compare to uniform 4-bit. | Bits per layer/head/route, wall-clock, peak memory, accuracy, error by depth | Mixed precision is meaningless unless the allocation rule is spelled out and held fixed across runs. |
| **KIVI / KVQuant / adaptive KV quantization** | [KIVI paper](https://arxiv.org/abs/2402.02750), [KIVI repo](https://github.com/jy-yuan/KIVI), [KVQuant paper](https://arxiv.org/abs/2401.18079), [KVQuant repo](https://github.com/SqueezeAILab/KVQuant), [QAQ paper](https://arxiv.org/abs/2403.04643), [QAQ repo](https://github.com/ClubieDong/QAQ-KVCacheQuantization), [More for Keys, Less for Values](https://arxiv.org/abs/2502.15075), [KV Cache is 1 Bit Per Channel](https://arxiv.org/abs/2405.03917) | Keys and values behave differently, and keys often want more precision, per-channel treatment, or pre-RoPE handling. | Test `K_bits != V_bits`, per-channel vs per-token quantization, and pre-RoPE vs post-RoPE key handling. | K/V bit split, pre/post-RoPE error, long-context accuracy, attention-score drift, outlier mass | A K/V split win only means something if the exact split and the RoPE placement are explicit. |
| **Residual correction / QJL / linear repair** | [KVLinC paper](https://arxiv.org/abs/2510.05373), [QJL paper](https://arxiv.org/abs/2406.03482), [QJL repo](https://github.com/amirzandieh/QJL), [TurboQuant paper](https://arxiv.org/abs/2504.19874) | A small correction stage can repair quantization bias better than pushing all error into the coarse quantizer. | Add a lightweight correction head after the bridge and compare it with a no-correction control at equal bytes. | Residual norm, correction rank, attention-logit drift, latency overhead, accuracy delta | If correction compute is omitted from the budget, the result is not paper-safe. |
| **Query-aware / query-agnostic KV selection** | [H2O paper](https://arxiv.org/abs/2306.14048), [H2O repo](https://github.com/FMInference/H2O), [SnapKV paper](https://arxiv.org/abs/2404.14469), [SnapKV repo](https://github.com/FasterDecoding/SnapKV), [Quest paper](https://arxiv.org/abs/2406.10774), [Quest repo](https://github.com/mit-han-lab/Quest), [KVzip paper](https://arxiv.org/abs/2505.23416) | Eviction and selection are separate from quantization; query-aware and query-agnostic policies answer different questions. | Compare LatentWire against H2O/SnapKV/Quest-style selection and KVzip-style query-agnostic reuse under matched bytes. | Retained-token histogram, reuse rate, query sensitivity, memory peak, latency, accuracy by sequence position | A cache-policy win is not a semantic-communication win unless the downstream benchmark is identical. |

## What LatentWire Should Test First

1. **Rotation before transport.**
   Compare identity, random Hadamard, learned orthogonal, and per-head rotations before any bridge quantization.

2. **Protected outlier channels.**
   Hold a small protected FP16/BF16 channel subset and compare it with uniform 4-bit transport at the same byte budget.

3. **Asymmetric K/V budgets.**
   Sweep `K_bits != V_bits`, plus pre-RoPE vs post-RoPE keys, and check whether routing quality or content quality dominates the error.

4. **Codebook or RVQ bridge.**
   Replace the dense latent with an additive or residual codebook, and log dead-code rate and residual decay.

5. **Correction head.**
   Add a tiny linear repair stage after transport and verify that any gain survives a strict compute-budget accounting.

6. **Selection baseline stack.**
   Benchmark against H2O, SnapKV, Quest, and KVzip so we can tell whether the bridge is better than cache curation.

## Telemetry Standard

- exact byte budget
- repair budget and repair latency
- K/V bit split
- per-layer or per-head bit allocation
- pre/post rotation outlier mass
- reconstruction loss and cosine drift
- attention-score drift
- codebook occupancy and dead-code rate
- retained-token histogram
- wall-clock latency and peak memory
- model pair, prompt template, and decode temperature

## Claim Risks

- Raw accuracy wins are weak if they depend on hidden repair compute or a looser byte budget.
- Mixed-precision wins are weak if the allocation rule is not reproducible.
- Rotation wins are weak if a uniform-byte baseline was not run with the same candidate pool.
- Codebook wins are weak if the codebook overfits one model family or one prompt format.
- Cache-selection wins are weak if they are compared against the wrong downstream task.

## Practical Readout

If LatentWire wants a paper-safe claim, the next three questions are:

1. Does the bridge still win when bytes are matched?
2. Does the bridge still win when repair compute is matched?
3. Does the bridge still win against direct cache-selection baselines, not just against dense transport?

If any answer is no, the result is still useful, but it is a diagnostic result rather than a method claim.
