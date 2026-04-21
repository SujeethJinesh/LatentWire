# Quantization / Preconditioning Memo for LatentWire

Scope: primary-source quantization and preconditioning ideas that map cleanly onto cross-model KV communication. EXL2 itself is an implementation format rather than a canonical paper, so I use GPTQ-style mixed precision and exllama-v2 as the engineering analogue.

## Sources to steal from

- [GPTQ](https://arxiv.org/abs/2210.17323): layerwise reconstruction with second-order sensitivity. Core idea: preserve output by minimizing local quantization error under a curvature-aware objective.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) / [PMLR version](https://proceedings.mlr.press/v202/xiao23c.html): diagonal smoothing shifts outlier mass from activations into weights before quantization.
- [AWQ](https://arxiv.org/abs/2306.00978): activation-aware protection of salient channels. Useful for salience-driven bit allocation.
- [QuaRot](https://arxiv.org/abs/2404.00456): orthogonal rotation to spread outliers so quantization sees a flatter distribution.
- [SpinQuant](https://arxiv.org/abs/2405.16406) and [ICLR 2025 page](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html): learned rotations instead of fixed Hadamard-style transforms.
- [QuIP#](https://arxiv.org/abs/2402.04396): Hadamard incoherence plus lattice codebooks. Good template for preconditioning + discrete codebook residuals.
- [AQLM](https://arxiv.org/abs/2401.06118): additive quantization with learned codebooks across blocks. Good mental model for multi-codebook bridge states.
- [APTQ](https://arxiv.org/abs/2402.14866): mixed-precision selection that accounts for attention output nonlinearity, not just local weight error.
- [HIGGS](https://aclanthology.org/2025.naacl-long.543/): Hadamard rotation plus MSE-optimal grids plus dynamic-programming bit allocation.

Recent KV-cache and compression sources:

- [KIVI](https://arxiv.org/abs/2402.02750): asymmetric 2-bit KV cache quantization. Strong reminder that K and V are not equally sensitive.
- [AsymKV](https://aclanthology.org/2025.coling-main.158/): explicit layerwise asymmetric KV quantization; useful for per-layer K/V budget splits.
- [CommVQ](https://arxiv.org/abs/2506.18879): commutative vector quantization for KV cache compression. Good analogue for order-insensitive bridge states.
- [Palu](https://arxiv.org/abs/2407.21118): low-rank latent KV projection. Strong low-rank bottleneck reference.
- [xKV](https://arxiv.org/abs/2503.18893): cross-layer SVD for KV cache compression. Suggests shared subspaces across adjacent layers.
- [ReCalKV](https://arxiv.org/abs/2505.24357): head reordering plus offline calibration before low-rank compression.
- [KVzip](https://janghyun1230.github.io/kvzip/) / arXiv 2505.23416: query-agnostic eviction with context reconstruction. Good bridge for LatentWire because it separates retention from reconstruction.
- [KVLinC](https://arxiv.org/abs/2510.05373): Hadamard rotation plus linear correction in extreme low-bit KV cache regimes.
- [CommonKV](https://arxiv.org/abs/2508.16134): cross-layer parameter sharing for KV cache compression.
- [DynaKV](https://arxiv.org/abs/2603.04411): token-adaptive low-rank KV compression. Good for adaptive budgets over reasoning-critical tokens.
- [TurboQuant](https://arxiv.org/abs/2504.19874): near-optimal online vector quantization for KV caches with rotation and sketch-like residual correction. The most direct source for a quantization-plus-correction bridge.

## Math ideas to steal

- Diagonal smoothing: choose per-channel scale `s` so `x / s` is easier to quantize and `W * s` absorbs the range shift.
- Orthogonal preconditioning: apply `R` with `R^T R = I` so outliers are dispersed before low-bit transport.
- Mixed-bit allocation: assign more bits to high-sensitivity heads, layers, or tokens, with the budget chosen by salience/curvature.
- Asymmetric K/V treatment: do not assume one precision fits both; K often carries sharper attention sensitivity than V.
- Low-rank shared subspaces: represent bridge states as `U z` with small `rank(z)`, then quantize or sketch `z` instead of the full cache.
- Additive / codebook residuals: keep a coarse quantized base plus small residual codes, rather than one flat quantizer.
- Reconstruction-aware eviction: remove a slot only if the decoder can recover it from retained context.
- One-bit sketch correction: use a tiny residual projection to correct systematic quantization bias after the main compression step.

## Concrete LatentWire ablations

- `bridge_precondition = {none, diag_smooth, hadamard, learned_rotation}` before source-to-target KV translation.
- `kv_precision = {fp16, int8, int4, mixed_bits}` with separate K/V budgets; compare symmetric vs asymmetric K/V allocation.
- `mixed_bits_policy = {uniform, attention_salience, head_curvature, token_entropy}` for bridge slots or heads.
- `bridge_rank = {4, 8, 16, 32}` and `bridge_rank_shared_across_layers = {on, off}` for low-rank latent transport.
- `bridge_residual = {none, additive_codebook, linear_correction, 1bit_sketch}` to test whether residual correction rescues accuracy.
- `token_eviction = {none, query_agnostic, query_aware}` with reconstruction telemetry to separate retention from decoding quality.
- `head_budgeting = {uniform, head_importance, headwise_asymkv, route_atom}` to see if routing and quantization should be coupled.
- `tokenizer_bridge = {subword, byte, byte_plus_lowrank}` for mismatch-heavy source/target pairs.

## Telemetry to log

- Accuracy, bytes/token, latency, and throughput at each budget point.
- Per-layer and per-head reconstruction MSE, cosine similarity, and KL on attention logits.
- K-vs-V distortion separately; do not collapse them into one number.
- Effective rank of bridge states before and after compression.
- Head/token entropy, dead-head rate, dead-token rate, and top-gap for selection policies.
- Budget allocation curves by layer, head, and token position.
- Robustness across context length, tokenizer mismatch, and source-target family.
- Correlation between compression error and downstream accuracy delta.

## Practical read on LatentWire

- If we want a positive-method paper, the most promising path is not uniform compression. It is adaptive transport with preconditioning plus residual correction.
- The second best axis is asymmetric K/V handling, because most methods above converge on K being the fragile part.
- The third axis is low-rank/shared-subspace transport with reconstruction telemetry, because that gives an interpretable bridge that can be debugged.
