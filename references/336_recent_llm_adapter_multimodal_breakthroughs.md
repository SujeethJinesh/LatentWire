# Recent LLM / Multimodal Bridge Ideas for LatentWire

Web check: 2026-04-21. Primary sources only where possible. This is a paper-safe idea map, not a claims list.

## Claim policy

- `Dev-smoke` = toy bridges, GSM30 slices, calibration-only tuning, prompt-shape checks, and telemetry validation.
- `Held-out` = frozen eval split, fixed total budget, and unchanged serialization / selection contract.
- Do not promote any idea to a paper claim until it wins on held-out data under the same cost budget.

## 1) BLIP-2 / Q-Former

Link: [BLIP-2](https://arxiv.org/abs/2301.12597)

Core mechanism: a lightweight Querying Transformer compresses frozen upstream features into a small set of learned query slots before handing them to the frozen LLM.

Why it could help LatentWire: it is the cleanest precedent for a small, trainable bridge that preserves a frozen source / target backbone.

Concrete ablation: fixed query bank vs learned query bank; 4 vs 8 vs 16 slots; source-conditioned vs target-conditioned queries; bridge-only training.

Telemetry fields: `query_slot_occupancy`, `dead_slot_rate`, `slot_entropy`, `slot_to_token_alignment`, `source_token_coverage`, `bridge_bytes`.

Claim risk: high until the same slot budget beats the flat bridge on held-out GSM70 / SVAMP.

## 2) Flamingo-style gated cross-attention

Link: [Flamingo](https://arxiv.org/abs/2204.14198)

Core mechanism: gated cross-attention layers inject external context into the language model while preserving a strong autoregressive backbone.

Why it could help LatentWire: a gate gives a direct way to test when latent transport should be suppressed, rather than always forcing the bridge to write.

Concrete ablation: gate on/off, scalar gate vs vector gate, gated cross-attn every layer vs every N layers, source-only vs source+target gating.

Telemetry fields: `gate_value_mean`, `gate_sparsity`, `cross_attention_mass`, `token_change_rate`, `route_entropy`, `help_rate`, `harm_rate`.

Claim risk: medium-high; gains can come from extra capacity unless the budget is matched exactly.

## 3) Perceiver IO / latent bottlenecks

Link: [Perceiver IO](https://arxiv.org/abs/2107.14795)

Core mechanism: compress arbitrary-sized inputs into a small latent array and query outputs from that fixed bottleneck.

Why it could help LatentWire: this is the most direct template for a budgeted communication bottleneck with interpretable slots.

Concrete ablation: fixed latent size vs adaptive latent size; one latent pass vs iterative latent refinement; source-centric vs target-centric latents.

Telemetry fields: `latent_utilization`, `slot_coverage`, `latent_reconstruction_error`, `route_overlap`, `bytes_moved`, `accuracy_per_byte`.

Claim risk: high unless the bottleneck stays fixed and outperforms the flat projection under matched compute.

## 4) Libra / routed visual expert + bridge

Link: [Libra](https://arxiv.org/abs/2405.10140)

Core mechanism: decouples inner-modal modeling from cross-modal interaction with a routed expert and a bridge module.

Why it could help LatentWire: it gives a precedent for separating local modeling from cross-model transport, which matches our source/target asymmetry.

Concrete ablation: routed expert vs shared bridge; decoupled source/target experts; discrete vs continuous bridge tokens.

Telemetry fields: `inner_modal_attention_mass`, `cross_modal_attention_mass`, `expert_route_histogram`, `bridge_token_count`, `source_target_asymmetry`.

Claim risk: medium-high; the architecture can look good in-sample while hiding route collapse.

## 5) Mixture-of-Depths

Link: [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258)

Core mechanism: routes compute to a subset of tokens/layers under a fixed budget using top-k routing.

Why it could help LatentWire: if bridge quality depends on a few high-value tokens, compute should be allocated there instead of evenly.

Concrete ablation: top-k token routing vs threshold routing; bridge tokens computed at shallow vs deep layers; fixed k sweep.

Telemetry fields: `selected_token_fraction`, `depth_route_histogram`, `topk_stability`, `accuracy_per_flop`, `route_entropy`.

Claim risk: medium; routing can improve efficiency without improving communication unless we report fixed-budget comparisons.

## 6) Multi-Head Latent Attention / TransMLA

Link: [TransMLA](https://arxiv.org/abs/2502.07864)

Core mechanism: projects KV states into a compact latent space, reducing cache size while preserving expressiveness.

Why it could help LatentWire: KV compression is directly analogous to shrinking the communication surface between source and target models.

Concrete ablation: full KV vs latent KV; rank sweep; source-only latent cache vs shared latent cache; decode-only compression vs full bridge compression.

Telemetry fields: `kv_cache_bytes`, `latent_rank`, `compression_ratio`, `decode_latency_ms`, `answer_accuracy`, `token_per_byte`.

Claim risk: high if the gain comes mostly from memory savings; the paper needs quality under equal budget, not only speed.

## 7) LV-XAttn / distributed cross-attention

Link: [LV-XAttn](https://arxiv.org/abs/2502.02406)

Core mechanism: makes cross-attention cheaper by moving the smaller query side rather than the larger key-value side.

Why it could help LatentWire: it is a strong budget heuristic for asymmetric communication, which is exactly what source-to-target latent transport is.

Concrete ablation: query-move vs KV-move analogs; asymmetric bridge placement; source-heavy vs target-heavy exchange.

Telemetry fields: `query_bytes`, `kv_bytes`, `communication_cost`, `cross_attention_latency_ms`, `accuracy_per_byte`.

Claim risk: medium; it is most useful as a systems prior unless we show a quality gain from asymmetric routing itself.

## 8) Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers

Link: [TACA](https://arxiv.org/abs/2506.07986)

Core mechanism: adjusts cross-modal attention temperature and weighting to fix modality imbalance.

Why it could help LatentWire: if source/target imbalance is hurting us, temperature or gate scaling may be the smallest useful fix.

Concrete ablation: attention temperature sweep; source/target reweighting; timestep-style schedule replaced by decode-step schedule; fixed vs adaptive balance.

Telemetry fields: `attention_temperature`, `cross_modal_mass`, `token_imbalance`, `answer_change_rate`, `help_rate`, `harm_rate`.

Claim risk: high; this can easily become a tuning trick unless evaluated under a frozen budget and held-out split.

## 9) Mixture-of-Depths Attention

Link: [MoDA](https://arxiv.org/abs/2603.15619)

Core mechanism: lets heads attend to both current-layer and preceding-layer KV pairs, so shallow information can survive depth scaling.

Why it could help LatentWire: a bridge may need access to earlier latent state, not only the current projection, if repair depends on preserving shallow evidence.

Concrete ablation: current-layer only vs current+previous-layer latent memory; depth memory size sweep; post-norm vs pre-norm bridge blocks.

Telemetry fields: `depth_memory_usage`, `shallow_signal_retention`, `bridge_recall_rate`, `accuracy_per_flop`, `latency_ms`.

Claim risk: high; this is promising for iterative repair, but it needs a strict held-out evaluation and a matched token budget.

## Highest-priority LatentWire moves

1. Promote process repair from GSM30 dev-smoke to held-out GSM70 / SVAMP with a strict final-answer contract and matched total budget.
2. Replace flat bridge projections with learned query-slot adapters before adding more routing complexity.
3. Add KV / latent-compression ablations so we can report `accuracy_per_byte` and not just raw accuracy.
4. Test one-step vs multi-step repair under the same budget, because iterative refinement is the most plausible way to convert a weak route into a correct final answer.
