# Diffusion, Iterative Refinement, and Latent Update References for LatentWire

## Bottom line

The useful lesson from the diffusion / flow-matching / iterative-refinement literature is not simply "use a generative model." It is:

1. treat the bridge as a path in latent space rather than a one-shot projection,
2. separate coarse global organization from local refinement,
3. preserve symmetry or equivariance where the task should not care about token order,
4. and log enough telemetry to tell whether the bridge learned structure or only learned a different compression trick.

For LatentWire, this points to bridge variants that are:

- multi-pass instead of single-shot,
- blockwise or proxy-token based instead of fully dense,
- flow/velocity based instead of only regression-based,
- and explicit about what gets refreshed versus what gets frozen.

## Recent primary sources

### Diffusion transformers and flow-matching hybrids

- [DiT-Flow: Speech Enhancement Robust to Multiple Distortions based on Flow Matching in Latent Space and Diffusion Transformers](https://arxiv.org/abs/2603.21608)
- [From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs](https://arxiv.org/abs/2512.06776)
- [Generative Pre-trained Autoregressive Diffusion Transformer](https://arxiv.org/abs/2505.07344)

### Iterative refinement in latent / belief space

- [LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)
- [Latent Refinement Decoding: Enhancing Diffusion-Based Language Models by Refining Belief States](https://arxiv.org/abs/2510.11052)
- [Inner Loop Inference for Pretrained Transformers: Unlocking Latent Capabilities Without Training](https://arxiv.org/abs/2602.14759)
- [AIR: Zero-shot Generative Model Adaptation with Iterative Refinement](https://arxiv.org/abs/2506.10895)

### Latent token updating / vocabulary coupling

- [V2Flow: Unifying Visual Tokenization and Large Language Model Vocabularies for Autoregressive Image Generation](https://arxiv.org/abs/2503.07493)
- [Flexible Language Modeling in Continuous Space with Transformer-based Autoregressive Flows](https://arxiv.org/abs/2507.00425)
- [Next-Latent Prediction Transformers Learn Compact World Models](https://arxiv.org/abs/2511.05963)
- [Flow to the Mode: Mode-Seeking Diffusion Autoencoders for State-of-the-Art Image Tokenization](https://arxiv.org/abs/2503.11056)

### Attention-block variants and local/global factorization

- [Generative Pre-trained Autoregressive Diffusion Transformer](https://arxiv.org/abs/2505.07344) uses a lightweight causal attention variant and rotation-based conditioning.
- [From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs](https://arxiv.org/abs/2512.06776) explicitly uses context-causal attention with bidirectional attention only inside the active block.
- [Qihoo-T2X: An Efficient Proxy-Tokenized Diffusion Transformer for Text-to-Any-Task](https://arxiv.org/abs/2409.04005) is a good nearby anchor for proxy-token attention.

## Relevant symmetry / geometry idea

The common geometry across these papers is a split between:

- a **coarse, symmetry-respecting latent path** that handles global coordination, and
- a **local refinement operator** that only updates the uncertain or active part of the state.

For LatentWire, that suggests a bridge should probably be tested as a composition of:

- a slot- or head-equivariant preconditioner,
- a blockwise refinement map with shared weights across passes,
- and a query-conditioned refresh mask that is invariant to irrelevant permutations of the unused slots.

The strongest symmetry analogy here is:

- **permutation symmetry** over latent slots and heads should be preserved as much as possible,
- while **blockwise causality** can be relaxed only inside a selected refinement region.

That is a cleaner match to cross-model communication than a fully dense one-shot translator.

## Six concrete LatentWire ablations

1. **Multi-pass bridge refinement**
   Run the translator for 1, 2, and 4 refinement passes with shared weights. Stop on a fixed budget and compare against the current single-pass bridge. Telemetry: paired delta vs target-alone, per-pass entropy change, and whether later passes only reshuffle noise.

2. **Blockwise bidirectional bridge**
   Replace full causal update with context-causal attention plus bidirectional mixing only inside the active bridge block. This mirrors the AR-to-block-diffusion path and tests whether the blocker is global causality rather than representation quality.

3. **Query-conditioned token refresh**
   Only refresh the latent slots whose query score is above a threshold, and keep confident slots frozen across passes. Telemetry: refresh fraction, confidence retention, and route overlap with source attention.

4. **Proxy-slot transport**
   Compress each prefix window into a small number of proxy slots before translation, then broadcast those proxies back into the original cache. Compare `m = 2, 4, 8` proxies per layer/head group. Telemetry: proxy entropy, dead-proxy rate, and reconstruction loss.

5. **Flow-matched latent update**
   Replace direct regression on target hidden states with a velocity-field objective over interpolated latent states. Compare standard MSE, rectified-flow style interpolation, and a hybrid loss. Telemetry: path smoothness, step-wise residual norm, and calibration of the update norm.

6. **Rotation-preconditioned refinement**
   Add a cheap orthogonal or Hadamard preconditioning step before each bridge pass, then invert it at the interface. This tests whether the bridge is failing because of conditioning / outliers rather than missing semantic capacity. Telemetry: channel kurtosis, singular value spread, and route entropy.

## Telemetry to preserve on every run

- `paired_delta_vs_target_alone`
- `prefix_mid_suffix_coverage`
- `route_entropy` and `pool_entropy`
- `dead_slot_rate` or `dead_proxy_rate`
- `refresh_fraction`
- `per_pass_residual_norm`
- `support_overlap_with_source_attention`
- `head_specialization_index`
- `bridge_update_norm`
- `token_confidence_retention`

## Current LatentWire telemetry to anchor the next paper section

- `dynalign_prefdist` is still the strongest current route on `gsm8k_5` at `0.4000`.
- `attention_stratified` and `query_pool_transport` both landed at `0.2000` on `gsm8k_5`.
- On controlled `gsm8k_eval_10`, `query_pool_transport` matched target-alone at `0.1000`.
- The paired delta versus target-alone was `+0.0000`, so the pooled bridge is interpretable but not yet a positive method.

## Interpretation rule for the paper

Do not present pooling, iteration, or flow matching as wins unless they improve accuracy **and** preserve or improve the telemetry above. Right now these methods are best treated as blockers and geometry probes: they show which part of the bridge is stable, but they do not yet show a positive-method gain.
